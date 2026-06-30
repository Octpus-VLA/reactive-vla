#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scripted IK expert that records SO-101 pick-and-place demos in the MuJoCo sim.

Why this exists: a SmolVLA checkpoint trained on *real* SO-101 data is out of
distribution on MuJoCo-rendered observations (different textures/lighting), so it
barely moves in `sim-eval`. To close that real->sim visual gap we need training
data whose *observations* are sim-rendered. This module produces it: a privileged
pick-and-place controller (reads the cube's true pose from sim state, solves IK)
drives the `SimSO101` robot, and every control step's (observation, action) pair
is written to a `LeRobotDataset` — the exact schema `lerobot-record` produces, so
the result feeds straight into `pixi run train`.

The expert is allowed to use privileged sim state (cube pose/velocity); the
dataset only ever stores what a real rig could observe (wrist camera + joint
state + the commanded joint targets). That separation is the whole point: the
learned policy sees images, the expert that generated the demo did not have to.

Entry point: `pixi run sim-collect` (see cli/so101.py). Design notes in
docs/sim-scripted-collect.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Body joints solved by IK (gripper is commanded directly, not via IK).
BODY_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER = "gripper"
# Site used as the tool-centre point (between the jaws) for IK targeting. Defined
# in so101.xml on the gripper body.
TCP_SITE = "gripperframe"
CUBE_JOINT = "cube_free"


@dataclass
class GraspConfig:
    """Tunable geometry for the scripted pick-and-place, all in metres / m·s."""

    # Height above the cube/box centre the TCP approaches and retreats to.
    approach_height: float = 0.10
    # TCP z offset relative to the cube centre at the moment of grasp. Slightly
    # below centre so the jaws straddle the cube rather than skim its top.
    grasp_z_offset: float = -0.005
    # TCP z offset above the box floor when releasing.
    place_height: float = 0.06
    # Gripper command (robot 0..100 scale) for open / closed. 0 = fully closed;
    # commanding fully closed lets the 3 cm cube physically stop the jaws so they
    # clamp it (see probe: ctrl maps 0->closed gap 0.4cm, 100->open gap 13cm).
    gripper_open: float = 70.0
    gripper_closed: float = 0.0
    # Per-phase budget in control steps (at --fps) for the held move phases
    # (lift/carry/place); they also advance early once within `reach_tol`.
    phase_steps: int = 30
    # Settle budget (steps) for the grasp and release phases — long enough for the
    # jaws to close/open, short enough that a moving cube doesn't slide out before
    # the jaws clamp it.
    grip_steps: int = 12
    # Max steps the approach phase will hover waiting for a moving cube to enter
    # reach before giving up (a passed/unreachable cube → the episode is a miss).
    wait_steps: int = 240
    # TCP-to-target distance (m) that counts as "arrived" for a held move phase.
    reach_tol: float = 0.012
    # Horizontal TCP-to-cube distance (m) at which the hovering approach commits to
    # descending (the gripper is over the cube).
    align_tol: float = 0.02
    # 3-D TCP-to-cube distance (m) at which the descent commits to grasping.
    grasp_tol: float = 0.02
    # Descend/grasp clamp the tracked cube y to ±this (m) so the arm never chases a
    # cube past its workspace; it grasps within reach even at high belt speed.
    reach_window_y: float = 0.12
    # Small lead (s) added to the tracked cube xy so the servo, which lags a moving
    # target, aims slightly ahead and closes the gap. ×belt_speed → metres of lead.
    track_lead_s: float = 0.12
    # Where on the belt (world y, m) the arm waits to grasp a moving cube — the
    # home-pose "sweet spot" in front of the robot. Grasping here (rather than at
    # the edge of reach the instant the cube enters) gives a consistent, strong
    # top-down grip at every belt speed; a grip at full -y extension slips on lift,
    # which is why low belt speeds used to fail. Ignored for a static cube.
    grasp_y: float = 0.0
    # Lead (s) before the cube reaches grasp_y at which the arm starts descending,
    # so the jaws close around it near the sweet spot rather than behind it.
    # ×belt_speed → metres. Closure tracks the live cube, so this only needs to be
    # roughly the descend+close duration.
    descend_lead_s: float = 0.45


@dataclass
class IKConfig:
    """Damped least-squares IK / servo settings."""

    iters: int = 60
    damping: float = 0.08
    pos_tol: float = 1e-3
    step: float = 0.5
    # Closed-loop servo gain. The MuJoCo position actuators droop under gravity
    # load (the elbow settles ~8° short of an open-loop command), so the expert
    # doesn't command absolute IK angles — it integrates a Jacobian step driven by
    # the live TCP error into a running joint command, which grows past the droop
    # point until the actual TCP reaches the target. This is that integral gain.
    servo_gain: float = 1.0
    # Max TCP displacement (m) the servo commands per control step. Caps the
    # Jacobian step so a far target (e.g. the lateral jump from carry to place)
    # can't produce a violent one-step swing that flings the held cube out of the
    # jaws — the arm instead glides toward it at a bounded ~max_tcp_step·fps speed.
    max_tcp_step: float = 0.015


class _Sim:
    """Thin privileged accessor over a connected SimSO101's MuJoCo model/data.

    Reaches into the robot's MuJoCo handles for IK and ground-truth object poses.
    This is the *expert's* privileged channel — none of it is written to the
    dataset; only `robot.get_observation()` output is.
    """

    def __init__(self, robot) -> None:
        import mujoco

        self._mj = mujoco
        self.model = robot._model
        self.data = robot._data
        self.tcp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        if self.tcp_site < 0:
            raise ValueError(f"scene has no site '{TCP_SITE}' to use as the IK tool point.")
        cube_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
        if cube_jid < 0:
            raise ValueError(f"scene has no freejoint '{CUBE_JOINT}' to read the cube pose from.")
        self.cube_qadr = int(self.model.jnt_qposadr[cube_jid])
        self.cube_dofadr = int(self.model.jnt_dofadr[cube_jid])
        # qpos addresses + joint-range limits for the body joints IK controls.
        self.body_qadr = np.array(
            [
                self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, m)]
                for m in BODY_MOTORS
            ]
        )
        self.body_dofadr = np.array(
            [
                self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, m)]
                for m in BODY_MOTORS
            ]
        )
        self.body_ranges = np.array(
            [
                self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, m)]
                for m in BODY_MOTORS
            ]
        )

    def cube_pos(self) -> np.ndarray:
        return np.array(self.data.qpos[self.cube_qadr : self.cube_qadr + 3], dtype=float)

    def cube_vel(self) -> np.ndarray:
        return np.array(self.data.qvel[self.cube_dofadr : self.cube_dofadr + 3], dtype=float)

    def tcp_pos(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.tcp_site], dtype=float)

    def body_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self.body_qadr], dtype=float)


def solve_ik(sim: _Sim, target_pos: np.ndarray, q_seed: np.ndarray, cfg: IKConfig) -> np.ndarray:
    """Position-only damped least-squares IK for the body joints.

    Iterates on a *scratch* MjData copy (so the live sim is untouched), seeding
    from `q_seed` (radians) and stepping the body joints to bring the TCP site to
    `target_pos`. Redundancy (5 joints for a 3-DoF position) is resolved near the
    seed, which keeps the wrist near its current top-down-ish orientation rather
    than flipping. Returns body-joint angles in radians, clamped to joint ranges.
    """
    mj = sim._mj
    scratch = mj.MjData(sim.model)
    scratch.qpos[:] = sim.data.qpos
    q = q_seed.astype(float).copy()
    jacp = np.zeros((3, sim.model.nv))
    for _ in range(cfg.iters):
        for i, adr in enumerate(sim.body_qadr):
            scratch.qpos[adr] = q[i]
        mj.mj_kinematics(sim.model, scratch)
        mj.mj_comPos(sim.model, scratch)
        err = target_pos - scratch.site_xpos[sim.tcp_site]
        if np.linalg.norm(err) < cfg.pos_tol:
            break
        mj.mj_jacSite(sim.model, scratch, jacp, None, sim.tcp_site)
        j = jacp[:, sim.body_dofadr]  # 3 x 5
        # Damped least squares: dq = Jᵀ (J Jᵀ + λ²I)⁻¹ err
        jjt = j @ j.T + (cfg.damping**2) * np.eye(3)
        dq = j.T @ np.linalg.solve(jjt, err)
        q = q + cfg.step * dq
        q = np.clip(q, sim.body_ranges[:, 0], sim.body_ranges[:, 1])
    return q


class PickPlaceExpert:
    """Privileged scripted state machine: approach → grasp → lift → place → release.

    Produces one action dict per control step from the current sim state. Pre-grasp
    phases reactively track the cube's *live* xy (clamped to the reachable window
    and led slightly forward), so the gripper hovers over a cube on the belt, waits
    for it to enter reach, follows it down, and closes around it while it is still
    moving — handling any belt speed, including one that varies between episodes,
    without per-speed tuning. Once grasped, the cube is held, so carry/place use a
    fixed carry height and the box's static pose.
    """

    PHASES = ("approach", "descend", "grasp", "lift", "carry", "place", "release", "done")

    def __init__(
        self,
        sim: _Sim,
        box_xy: np.ndarray,
        grasp: GraspConfig,
        ik: IKConfig,
        belt_speed: float,
        control_fps: float,
    ):
        self.sim = sim
        self.box_xy = np.asarray(box_xy, dtype=float)
        self.g = grasp
        self.ik = ik
        self.belt_speed = float(belt_speed)
        self.control_fps = float(control_fps)
        self.phase = "approach"
        self._phase_step = 0
        self._grasp_xy: np.ndarray | None = None
        # Cube's resting height at episode start. Held-phase target heights are
        # computed from this *fixed* value, never from the live (rising) cube z —
        # referencing the held cube's own z creates a positive-feedback runaway
        # that flings the arm to full extension.
        self._rest_z = float(sim.cube_pos()[2])
        # Running joint command (radians), integrated by the closed-loop servo.
        # Seeded from the current pose so the first step is a no-op nudge.
        self.q_cmd = sim.body_qpos()

    @property
    def done(self) -> bool:
        return self.phase == "done"

    def _jac_step(self, err: np.ndarray) -> np.ndarray:
        """Damped least-squares joint delta that moves the TCP by `err`, evaluated
        at the *live* (actual) arm configuration — the integral servo's increment."""
        mj = self.sim._mj
        jacp = np.zeros((3, self.sim.model.nv))
        mj.mj_jacSite(self.sim.model, self.sim.data, jacp, None, self.sim.tcp_site)
        j = jacp[:, self.sim.body_dofadr]  # 3 x 5
        jjt = j @ j.T + (self.ik.damping**2) * np.eye(3)
        return j.T @ np.linalg.solve(jjt, err)

    def _track_xy(self) -> np.ndarray:
        """Pre-grasp aim point: the cube's *live* xy, led slightly forward to
        offset the servo's lag on a moving target, with y clamped to the reachable
        window so a cube still out at the feed end is hovered-for at the near edge
        of reach rather than chased past the workspace. Reactive (uses the cube's
        actual position every step) rather than a one-shot intercept, so it handles
        any belt speed — including a speed that varies between episodes. Static belt
        (speed 0) collapses to the cube's current xy."""
        cube = self.sim.cube_pos()
        lead_y = self.belt_speed * self.g.track_lead_s
        y = np.clip(cube[1] + lead_y, -self.g.reach_window_y, self.g.reach_window_y)
        return np.array([cube[0], y])

    def _target_for_phase(self) -> tuple[np.ndarray, float]:
        """Return (tcp_target_xyz, gripper_cmd) for the current phase. Approach,
        descend and grasp track the live cube (so the gripper moves with a cube on
        the belt while the jaws close); held phases use a fixed carry height."""
        cube = self.sim.cube_pos()
        if self.phase == "approach":
            # Static: hover directly over the (fixed) cube. Moving: hover at the
            # fixed sweet spot and let the belt bring the cube to it.
            xy = self._track_xy() if self.belt_speed == 0.0 else np.array([cube[0], self.g.grasp_y])
            return np.array([xy[0], xy[1], self._rest_z + self.g.approach_height]), self.g.gripper_open
        if self.phase == "descend":
            xy = self._track_xy()
            return np.array([xy[0], xy[1], self._rest_z + self.g.grasp_z_offset]), self.g.gripper_open
        if self.phase == "grasp":
            # Keep tracking the (possibly still-moving) cube while the jaws close so
            # the gripper closes *around* it instead of behind it.
            xy = self._track_xy()
            self._grasp_xy = xy
            return np.array([xy[0], xy[1], self._rest_z + self.g.grasp_z_offset]), self.g.gripper_closed
        carry_z = self._rest_z + self.g.approach_height
        if self.phase == "lift":
            xy = self._grasp_xy if self._grasp_xy is not None else cube[:2]
            return np.array([xy[0], xy[1], carry_z]), self.g.gripper_closed
        if self.phase == "carry":
            return np.array([self.box_xy[0], self.box_xy[1], carry_z]), self.g.gripper_closed
        if self.phase == "place":
            return np.array([self.box_xy[0], self.box_xy[1], self.g.place_height]), self.g.gripper_closed
        if self.phase == "release":
            return np.array([self.box_xy[0], self.box_xy[1], self.g.place_height]), self.g.gripper_open
        return self.sim.tcp_pos(), self.g.gripper_open

    def _should_advance(self, target_xyz: np.ndarray) -> bool:
        """Per-phase transition test. Pre-grasp phases are event-triggered off the
        live cube (so the arm waits for a moving cube and commits only when it is
        actually over / within reach of it); held phases advance on arrival or a
        time budget; grasp/release settle on a short budget."""
        tcp = self.sim.tcp_pos()
        cube = self.sim.cube_pos()
        if self.phase == "approach":
            if self.belt_speed == 0.0:
                # Static: descend once the gripper is hovering over the cube.
                horiz = float(np.linalg.norm(tcp[:2] - cube[:2]))
                return horiz < self.g.align_tol or self._phase_step >= self.g.wait_steps
            # Moving: start descending when the cube reaches the descend-start line
            # `descend_lead_s` (in time) before the sweet spot, so closure — which
            # tracks the live cube — lands near grasp_y. Also require the arm to be
            # in place over the sweet spot first.
            over_spot = (
                float(np.linalg.norm(tcp[:2] - np.array([cube[0], self.g.grasp_y]))) < self.g.align_tol
            )
            cube_ready = cube[1] >= self.g.grasp_y - self.belt_speed * self.g.descend_lead_s
            return (over_spot and cube_ready) or self._phase_step >= self.g.wait_steps
        if self.phase == "descend":
            return (
                float(np.linalg.norm(tcp - cube)) < self.g.grasp_tol or self._phase_step >= self.g.phase_steps
            )
        if self.phase in ("grasp", "release"):
            return self._phase_step >= self.g.grip_steps
        # lift / carry / place: arrived at the (fixed) target, or budget spent.
        return (
            float(np.linalg.norm(tcp - target_xyz)) < self.g.reach_tol
            or self._phase_step >= self.g.phase_steps
        )

    def _advance(self) -> None:
        idx = self.PHASES.index(self.phase)
        self.phase = self.PHASES[min(idx + 1, len(self.PHASES) - 1)]
        self._phase_step = 0

    def step(self) -> dict:
        """Compute and return the action dict for this control step (and advance
        the phase machine). Action keys/units match SimSO101.action_features:
        body motor `<name>.pos` in degrees, `gripper.pos` on the 0..100 scale.

        Closed-loop integral servo: nudge the running joint command by a
        Jacobian step driven by the current TCP error, so the command grows past
        the actuators' gravity droop until the actual TCP reaches the target."""
        target_xyz, grip = self._target_for_phase()
        full_err = target_xyz - self.sim.tcp_pos()
        # Cap the commanded TCP step so far targets glide rather than whip.
        dist = float(np.linalg.norm(full_err))
        err = full_err * (self.ik.max_tcp_step / dist) if dist > self.ik.max_tcp_step else full_err
        dq = self._jac_step(err)
        self.q_cmd = np.clip(
            self.q_cmd + self.ik.servo_gain * dq,
            self.sim.body_ranges[:, 0],
            self.sim.body_ranges[:, 1],
        )
        action = {f"{m}.pos": float(np.rad2deg(self.q_cmd[i])) for i, m in enumerate(BODY_MOTORS)}
        action[f"{GRIPPER}.pos"] = float(grip)

        # Phase transition (event-triggered off the live cube for pre-grasp).
        self._phase_step += 1
        if self._should_advance(target_xyz):
            self._advance()
        return action


# --- Episode reset + dataset recording -------------------------------------


def _box_xy(sim: _Sim) -> np.ndarray:
    """World xy of the drop-off box centre (scene_cube.xml's static 'box' body)."""
    bid = sim._mj.mj_name2id(sim.model, sim._mj.mjtObj.mjOBJ_BODY, "box")
    if bid < 0:
        raise ValueError("scene has no 'box' body to place the cube into.")
    return np.array(sim.model.body_pos[bid][:2], dtype=float)


def _reset_episode(robot, sim: _Sim, rng: np.random.Generator, jitter_xy: float) -> None:
    """Re-apply the home keyframe and (re)place the cube for a fresh episode.

    Mirrors SimSO101.connect()'s placement: static belt parks the cube in front of
    the robot (y=0), a running belt feeds it from the -y end. On top of that we add
    a small uniform xy jitter so demos cover a spread of grasp positions rather than
    one fixed pose (a single-pose dataset teaches nothing reactive)."""
    mj = sim._mj
    key = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mj.mj_resetDataKeyframe(sim.model, sim.data, key)
    qa = sim.cube_qadr
    base_x = float(sim.data.qpos[qa])  # belt-centre x set by the keyframe/connect
    base_y = -0.20 if robot.config.belt_speed != 0 else 0.0
    sim.data.qpos[qa] = base_x + rng.uniform(-jitter_xy, jitter_xy)
    sim.data.qpos[qa + 1] = base_y + (
        0.0 if robot.config.belt_speed != 0 else rng.uniform(-jitter_xy, jitter_xy)
    )
    # Zero the cube's free-joint velocity so it starts at rest.
    sim.data.qvel[sim.cube_dofadr : sim.cube_dofadr + 6] = 0.0
    if robot.config.belt_speed != 0:
        belt_act = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_ACTUATOR, "belt_motor")
        if belt_act >= 0:
            sim.data.ctrl[belt_act] = robot.config.belt_speed
    mj.mj_forward(sim.model, sim.data)


def collect(
    repo_id: str,
    task: str,
    *,
    mjcf_path: str,
    root: str | None = None,
    episodes: int = 20,
    max_steps: int = 320,
    fps: int = 30,
    belt_speed: float = 0.0,
    belt_speed_max: float | None = None,
    belt_distance: float = 0.14,
    jitter_xy: float = 0.03,
    cameras: dict | None = None,
    seed: int = 0,
    push: bool = False,
    grasp: GraspConfig | None = None,
    ik: IKConfig | None = None,
) -> dict:
    """Record `episodes` scripted pick-and-place demos to a LeRobotDataset.

    Returns a summary dict (per-episode success + overall rate). Episodes whose
    grasp fails (cube never lifted) are still recorded by default — set the caller
    to drop them later, or filter on the returned success flags. The dataset
    schema is identical to `lerobot-record`, so `pixi run train` consumes it
    directly.

    Belt speed: `belt_speed` is the fixed speed (0 = static cube). If
    `belt_speed_max` is given (and exceeds `belt_speed`), each episode samples a
    speed uniformly from `[belt_speed, belt_speed_max]`, so one dataset spans a
    range of conveyor speeds — the reactive expert tracks the live cube and so
    handles any speed without per-speed tuning.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.sim_so101 import SimCameraConfig, SimSO101, SimSO101Config
    from lerobot.robots.sim_so101.config_sim_so101 import SimLiftSuccessConfig
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    cam_specs = cameras or {
        "camera1": SimCameraConfig(mujoco_name="wrist_cam", width=320, height=240),
        "overview": SimCameraConfig(mujoco_name="overview", width=320, height=240),
    }
    config = SimSO101Config(
        mjcf_path=str(mjcf_path),
        cameras=cam_specs,
        control_fps=fps,
        belt_speed=belt_speed,
        belt_distance=belt_distance,
        use_degrees=True,
        # Score the demo the way the task is defined: cube ends up settled inside
        # the box. (The expert reaches lift partway through every successful
        # episode too, but place_in_box is the criterion that matches the goal.)
        success=SimLiftSuccessConfig(body_name="cube", criterion="place_in_box"),
    )
    robot = SimSO101(config)
    robot.connect()
    sim = _Sim(robot)
    box_xy = _box_xy(sim)
    grasp = grasp or GraspConfig()
    ik = ik or IKConfig()
    rng = np.random.default_rng(seed)

    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    action_features = hw_to_dataset_features(robot.action_features, "action")
    features = {**obs_features, **action_features}
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=robot.name,
        use_videos=True,
    )

    vary_belt = belt_speed_max is not None and belt_speed_max > belt_speed
    results = []
    speeds = []
    for ep in range(episodes):
        # Per-episode belt speed. Updating robot.config.belt_speed is enough:
        # _reset_episode re-applies it to the belt actuator's ctrl on reset.
        ep_speed = float(rng.uniform(belt_speed, belt_speed_max)) if vary_belt else belt_speed
        robot.config.belt_speed = ep_speed
        speeds.append(ep_speed)
        _reset_episode(robot, sim, rng, jitter_xy)
        expert = PickPlaceExpert(sim, box_xy, grasp, ik, ep_speed, control_fps=fps)
        success = False
        frames = 0
        for _step in range(max_steps):
            obs = robot.get_observation()
            action = expert.step()
            frame = {
                **build_dataset_frame(features, obs, "observation"),
                **build_dataset_frame(features, action, "action"),
                "task": task,
            }
            dataset.add_frame(frame)
            robot.send_action(action)
            frames += 1
            # Latch success: the cube may pass the criterion mid-episode (e.g.
            # while settling) even if jostled later.
            if robot.check_success():
                success = True
            if expert.done:
                break
        dataset.save_episode()
        results.append(success)
        tag = "placed" if success else "miss"
        belt_note = f", belt {speeds[ep]:.3f} m/s" if vary_belt else ""
        logger.info("episode %d/%d: %s (%d frames%s)", ep + 1, episodes, tag, frames, belt_note)
        print(f"  episode {ep + 1}/{episodes}: {tag} ({frames} frames{belt_note})")

    robot.disconnect()
    rate = float(np.mean(results)) if results else 0.0
    summary = {"episodes": episodes, "success": results, "success_rate": rate, "belt_speeds": speeds}
    print(f"done: {sum(results)}/{episodes} placed in box (success rate {rate:.0%})")
    if push:
        dataset.push_to_hub()
    return summary
