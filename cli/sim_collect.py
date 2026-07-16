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
# scene_cube.xml distractor mocap bodies (see the file for why mocap: settable pose,
# no physics/collision). Order doesn't matter; count must match the XML.
DISTRACTOR_GEOMS = ["distractor0_geom", "distractor1_geom", "distractor2_geom", "distractor3_geom"]
FLOOR_MATERIAL = "groundplane"


@dataclass
class DomainRandomizationConfig:
    """Per-episode visual randomization for sim2real robustness (floor tint, light,
    background distractors). Baked into the rendered video at collection time — unlike
    online RL domain randomization, there is no re-rendering at train time, so the
    diversity a trained policy ever sees is capped by how many *episodes* used a
    distinct draw, not by any parameter range alone. Off by default (`enabled=False`)
    so existing collection recipes are unaffected unless explicitly turned on.
    """

    enabled: bool = False
    # Floor tint: material rgba is resampled per episode within [lo, hi] per channel
    # (alpha left at 1). Multiplies the groundplane checker texture rather than
    # replacing it, so the checker pattern itself still reads, just recoloured.
    floor_rgb_lo: tuple[float, float, float] = (0.05, 0.05, 0.05)
    floor_rgb_hi: tuple[float, float, float] = (0.9, 0.9, 0.9)
    # Light: scene.xml defines exactly one directional light (index 0, unnamed —
    # upstream Menagerie file, kept unedited). Position/diffuse resampled per episode;
    # direction stays pointing generally down (randomized within a cone) so the scene
    # doesn't go unlit.
    light_pos_xy_range: float = 1.5  # metres, uniform in [-range, range] for x and y
    light_height_lo: float = 2.5
    light_height_hi: float = 4.5
    light_diffuse_lo: float = 0.35
    light_diffuse_hi: float = 0.85
    light_tilt_max_deg: float = 25.0  # max deviation from straight-down
    # Distractors: up to len(DISTRACTOR_GEOMS) simple boxes scattered around the
    # periphery (outside the robot/belt/box working area) as generic background
    # clutter — not modelling any specific real object, just "stuff a vision model
    # must learn to ignore". `spawn_prob` per distractor per episode; the rest stay
    # parked out of view (see scene_cube.xml), giving a spread of 0..N visible.
    spawn_prob: float = 0.6
    distractor_xy_range: float = 0.45  # metres from robot base, excluding the keepout
    distractor_keepout_radius: float = 0.22  # metres from base; nothing spawns closer
    # Extra clearance (metres) added around the belt's and box's actual footprint
    # (see _aabb_for_bodies) so distractors can't render on top of/overlapping either —
    # the base-centred keepout circle alone doesn't cover them, since both sit well
    # outside a small radius from the robot base (belt runs out to y=+-0.33, box is at
    # x=0.30+belt_distance).
    distractor_belt_box_margin: float = 0.05
    distractor_size_lo: float = 0.01
    distractor_size_hi: float = 0.035
    distractor_height_lo: float = 0.02
    distractor_height_hi: float = 0.12


def _aabb_for_bodies(
    sim: "_Sim", body_names: list[str], margin: float
) -> tuple[float, float, float, float] | None:
    """World-frame xy axis-aligned bounding box (min_x, max_x, min_y, max_y) unioning
    every geom attached to the given bodies, expanded by `margin`. Assumes those
    bodies/geoms carry no rotation (true for scene_cube.xml's belt/box, which are
    positioned by `pos` only) so body_pos + geom_pos +/- geom_size is exact. Returns
    None if none of the bodies (or their geoms) exist."""
    mj = sim._mj
    model = sim.model
    xs: list[float] = []
    ys: list[float] = []
    for body_name in body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            continue
        bx, by = float(model.body_pos[bid][0]), float(model.body_pos[bid][1])
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != bid:
                continue
            gx = bx + float(model.geom_pos[gid][0])
            gy = by + float(model.geom_pos[gid][1])
            hx = float(model.geom_size[gid][0])
            hy = float(model.geom_size[gid][1])
            xs += [gx - hx, gx + hx]
            ys += [gy - hy, gy + hy]
    if not xs:
        return None
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def randomize_background(sim: "_Sim", rng: np.random.Generator, cfg: DomainRandomizationConfig) -> None:
    """Resample floor tint, light pose/intensity, and distractor placement for one
    episode. Pure visual state (materials, mocap poses, light fields) — never touches
    the robot, cube, or belt, so it has no effect on the expert's grasp logic. All the
    written arrays are read live by the renderer; no recompilation needed."""
    if not cfg.enabled:
        return
    mj = sim._mj
    model = sim.model

    floor_mat = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MATERIAL, FLOOR_MATERIAL)
    if floor_mat >= 0:
        rgb = rng.uniform(cfg.floor_rgb_lo, cfg.floor_rgb_hi)
        model.mat_rgba[floor_mat] = [*rgb, 1.0]

    if model.nlight > 0:
        x, y = rng.uniform(-cfg.light_pos_xy_range, cfg.light_pos_xy_range, size=2)
        z = rng.uniform(cfg.light_height_lo, cfg.light_height_hi)
        model.light_pos[0] = [x, y, z]
        tilt = np.deg2rad(rng.uniform(0, cfg.light_tilt_max_deg))
        az = rng.uniform(0, 2 * np.pi)
        dir_xy = np.sin(tilt) * np.array([np.cos(az), np.sin(az)])
        model.light_dir[0] = [dir_xy[0], dir_xy[1], -np.cos(tilt)]
        d = rng.uniform(cfg.light_diffuse_lo, cfg.light_diffuse_hi)
        model.light_diffuse[0] = [d, d, d]

    # Belt + box footprints (world xy, with clearance) so distractors never render on
    # top of/overlapping either — a base-centred keepout circle alone doesn't cover
    # them, since both sit well outside a small radius from the robot base.
    exclude_aabbs = [
        aabb
        for aabb in (
            _aabb_for_bodies(sim, ["conveyor_frame", "belt"], cfg.distractor_belt_box_margin),
            _aabb_for_bodies(sim, ["box"], cfg.distractor_belt_box_margin),
        )
        if aabb is not None
    ]

    for geom_name in DISTRACTOR_GEOMS:
        gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
        body_id = model.geom_bodyid[gid]
        mocap_id = model.body_mocapid[body_id]
        if mocap_id < 0:
            continue
        if rng.uniform() >= cfg.spawn_prob:
            sim.data.mocap_pos[mocap_id] = [0.0, 0.0, -1.0]  # parked out of view
            continue
        # Reject-sample xy outside the robot keepout disc and the belt/box AABBs so
        # distractors never overlap the robot/belt/box working area. Falls back to
        # parking out of view if 20 draws can't find a clear spot (should be rare —
        # the valid area is still >50% of the sampling square).
        xy = None
        for _ in range(20):
            candidate = rng.uniform(-cfg.distractor_xy_range, cfg.distractor_xy_range, size=2)
            if np.linalg.norm(candidate) < cfg.distractor_keepout_radius:
                continue
            if any(
                min_x <= candidate[0] <= max_x and min_y <= candidate[1] <= max_y
                for min_x, max_x, min_y, max_y in exclude_aabbs
            ):
                continue
            xy = candidate
            break
        if xy is None:
            sim.data.mocap_pos[mocap_id] = [0.0, 0.0, -1.0]
            continue
        z = rng.uniform(cfg.distractor_height_lo, cfg.distractor_height_hi)
        sim.data.mocap_pos[mocap_id] = [xy[0], xy[1], z]
        size = rng.uniform(cfg.distractor_size_lo, cfg.distractor_size_hi, size=3)
        model.geom_size[gid] = size
        model.geom_rgba[gid] = [*rng.uniform(0.05, 0.95, size=3), 1.0]


@dataclass
class GraspConfig:
    """Tunable geometry for the scripted pick-and-place, all in metres / m·s."""

    # Height above the cube/box centre the TCP approaches and retreats to. Also
    # sets how close wrist_cam hovers over the belt while waiting for the cube —
    # 0.10 pointed the downward-canted eye-in-hand view mostly past the belt at
    # the far-off checkered floor (only a thin sliver of belt at the frame top);
    # 0.06 fills the frame with the belt while it waits. Re-verified 56/56 across
    # belt_speed 0.03-0.12 (jitter=0.01) after lowering it, so no functional cost.
    approach_height: float = 0.06
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
    # Max steps the wait phase will sit at the gate waiting for a moving cube to
    # arrive before giving up (a passed/unreachable cube → the episode is a miss).
    wait_steps: int = 240
    # TCP-to-target distance (m) that counts as "arrived" for a held move phase.
    reach_tol: float = 0.012
    # Horizontal TCP-to-target distance (m) at which the hovering approach commits
    # to descending (the gripper is over the cube / the gate spot).
    align_tol: float = 0.02
    # Where on the belt (world y, m) the arm waits to grasp a moving cube — the
    # home-pose "sweet spot" in front of the robot. Grasping here (rather than at
    # the edge of reach the instant the cube enters) gives a consistent, strong
    # top-down grip at every belt speed; a grip at full -y extension slips on lift,
    # which is why low belt speeds used to fail. Ignored for a static cube.
    grasp_y: float = 0.0
    # Extra TCP height (m) above the grasp height at which the open jaws hover
    # while waiting for the cube: high enough that both jaw tips clear the cube's
    # top face (so the fingers never block the belt path), low enough that the
    # final drop into the grasp finishes before the closing jaws reach cube width.
    hover_clearance: float = 0.035
    # The jaws are asymmetric around the TCP (measured at the grasp pose,
    # wrist_roll=+90°): the *fixed* finger tip sits ~2.0 cm upstream of the TCP
    # and low (its shaft rides below cube-top height at grasp z), while the
    # *moving* finger sits ~7.5 cm downstream and ~3.7 cm higher — high enough to
    # clear the cube's top until the closure swings it down. The only thing that
    # can collide with the approaching cube is therefore the fixed finger, so the
    # drop goes *straight down* at the sweet spot (never tracking backwards into
    # the cube) and is timed by land_at_y/drop_time_s so the fixed finger reaches
    # cube height just after the cube's trailing face has cleared its plane. The
    # jaws then close immediately: the moving jaw sweeps in over the cube's top,
    # catches its leading face and presses it back against the fixed-finger
    # anvil, exactly like the (always-clean) static grasp.
    #
    # Drop trigger: start the straight-down drop when the cube's centre passes
    # grasp_y + land_at_y − belt_speed × drop_time_s. Both constants were fitted
    # empirically (grid search over the trigger offset at belt speeds 0.03-0.14,
    # 3 episodes each, scoring success + cube rotation): the optimum sits at
    # +0.000 m for 0.03 m/s drifting to −0.010 m at 0.14 m/s — i.e. nearly
    # speed-independent, because the descending fixed jaw clears the cube's top
    # early in the drop while the closure (which starts on landing) needs the
    # cube almost at the sweet spot already. Triggering ≥1 cm earlier lands the
    # fixed jaw's shaft on the cube's top face and rolls it (~110-180°);
    # triggering ≥1 cm later lets the cube slip past before the pinch closes
    # (misses at ≥0.10 m/s). The fitted line keeps 5-15° of cube rotation and a
    # full grasp across the whole 0.03-0.14 m/s range.
    land_at_y: float = 0.003
    drop_time_s: float = 0.09


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
    # 0.015 was tuned for the original jaw orientation (open/close axis ~world X);
    # after rotating the grasp to align with the belt's Y axis (see scene_cube.xml's
    # home keyframe wrist_roll), the grip holds the cube less securely against a
    # sideways swing, and 0.015 let carry->place fling it clear of the box at some
    # belt speeds (cube ending up 5-9cm past the box) and turned x-position
    # reachability at the grasp into a chaotic, non-monotonic pass/fail pattern.
    # 0.008 fixed both (verified: monotonic x-reach boundary restored, no more
    # mid-transition drops) at the cost of slightly slower phase transitions.
    max_tcp_step: float = 0.008
    # Separate (higher) TCP step cap used only for the "drop" phase (see
    # PickPlaceExpert._TRACKING_PHASES): its vertical descent must complete before
    # the belt carries the cube past the trigger point. At the default 0.008
    # (≈0.24 m/s max TCP speed) the descent itself takes long enough that
    # belt_speed ≳ 0.18 m/s carries the cube past the gate before the fixed finger
    # lands, regardless of drop-trigger timing — confirmed by sweeping the trigger
    # offset at 0.18-0.40 m/s and finding no offset that recovers a clean grasp.
    # 0.02 (≈0.6 m/s) fixes that. Previously also applied to approach/descend/wait/
    # grasp (reasoning: nothing held yet, so no flinging risk) — but that let the
    # servo close the ~3cm approach->descend target jump fast enough to visibly
    # overshoot/undershoot before settling (~3cm z oscillation over ~0.5s, visible
    # in recorded video as vertical shaking). Restricting to "drop" only cuts that
    # to ~0.25cm without touching the fast-belt fix this field exists for.
    max_tcp_step_pregrasp: float = 0.02


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
    """Privileged scripted state machine: approach → descend → wait → drop →
    grasp → lift → carry → place → release.

    Produces one action dict per control step from the current sim state. The arm
    hovers over the grasp sweet spot with the open jaws high enough that the
    finger tips clear the cube's top face (so nothing blocks the belt path),
    drops straight down at exactly the moment the cube's position calls for it —
    timed so the low fixed finger reaches cube height just after the cube's
    trailing face has cleared its plane (dropping any earlier lands that finger
    on the cube and flips it) — and closes immediately: the moving jaw sweeps in
    over the cube's top, catches its leading face and presses it back against
    the fixed-finger anvil, the same pinch the (always-clean) static grasp ends
    in. The drop timing scales with the live cube position and belt speed, so
    any speed — including one that varies between episodes — works without
    per-speed tuning. Once grasped, the cube is held, so carry/place use a fixed
    carry height and the box's static pose.
    """

    PHASES = ("approach", "descend", "wait", "drop", "grasp", "lift", "carry", "place", "release", "done")
    # Only "drop" needs the higher ik.max_tcp_step_pregrasp cap (its vertical descent
    # must complete before a fast belt carries the cube past the trigger point — see
    # IKConfig.max_tcp_step_pregrasp). Applying that same faster cap to
    # approach/descend/wait/grasp too (as previously done, reasoning "no held cube to
    # fling") had an unintended side effect: the approach->descend target jumps ~3cm
    # discontinuously, and the faster cap let the servo close that gap quickly enough
    # to visibly overshoot/undershoot before settling (measured ~3cm z oscillation
    # over ~0.5s). Restricting the fast cap to just "drop" cuts that oscillation to
    # ~0.25cm while leaving drop's fast-belt timing fix untouched.
    _TRACKING_PHASES = ("drop",)

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

    def _gate_xy(self) -> np.ndarray:
        """Hover aim point: the cube's live x (jitter tracking), and — for a moving
        belt — the fixed sweet-spot y where the arm waits for the belt to deliver
        the cube. Static belt aims at the cube itself."""
        cube = self.sim.cube_pos()
        y = cube[1] if self.belt_speed == 0.0 else self.g.grasp_y
        return np.array([cube[0], y])


    def _target_for_phase(self) -> tuple[np.ndarray, float]:
        """Return (tcp_target_xyz, gripper_cmd) for the current phase. Pre-grasp
        phases aim at the gate spot (live cube x, fixed sweet-spot y); descend/wait
        hold the jaw tips just above cube-top height, and grasp drops the rest of
        the way while closing. Held phases use a fixed carry height."""
        if self.phase == "approach":
            xy = self._gate_xy()
            return np.array([xy[0], xy[1], self._rest_z + self.g.approach_height]), self.g.gripper_open
        if self.phase in ("descend", "wait"):
            # Hover with the finger tips clear of the cube's top so the open jaws
            # never block the belt path while waiting for the cube.
            xy = self._gate_xy()
            z = self._rest_z + self.g.grasp_z_offset + (
                self.g.hover_clearance if self.belt_speed != 0.0 else 0.0
            )
            return np.array([xy[0], xy[1], z]), self.g.gripper_open
        if self.phase == "drop":
            # Straight down at the sweet spot (tracking only cube x): the fixed
            # finger lands just behind the cube's trailing face (see land_at_y).
            xy = self._gate_xy()
            return np.array([xy[0], xy[1], self._rest_z + self.g.grasp_z_offset]), self.g.gripper_open
        if self.phase == "grasp":
            # Close in place: the moving jaw sweeps in over the cube's top and
            # presses it back against the fixed-finger anvil.
            xy = self._gate_xy()
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
        """Per-phase transition test. Approach/descend advance on arrival (the
        hovering jaw tips stay above cube-top height, so the arm can settle over
        the spot early); wait is event-triggered off the live cube so the
        drop+close meets it centred under the gripper; held phases advance on
        arrival or a time budget; grasp/release settle on a short budget."""
        tcp = self.sim.tcp_pos()
        cube = self.sim.cube_pos()
        if self.phase == "approach":
            horiz = float(np.linalg.norm(tcp[:2] - target_xyz[:2]))
            return horiz < self.g.align_tol or self._phase_step >= self.g.wait_steps
        if self.phase == "descend":
            return (
                float(np.linalg.norm(tcp - target_xyz)) < self.g.reach_tol
                or self._phase_step >= self.g.phase_steps
            )
        if self.phase == "wait":
            if self.belt_speed == 0.0:
                return True  # static cube is already under the gripper — drop now
            # Trigger the drop so the fixed finger reaches cube height just as
            # the cube's trailing face clears its plane (see land_at_y).
            drop_at = self.g.grasp_y + self.g.land_at_y - self.belt_speed * self.g.drop_time_s
            return cube[1] >= drop_at or self._phase_step >= self.g.wait_steps
        if self.phase == "drop":
            # Land, then close immediately — every extra step lets the belt carry
            # the cube further from the fixed-finger anvil before the pinch.
            return abs(tcp[2] - target_xyz[2]) < self.g.reach_tol or (
                self._phase_step >= self.g.phase_steps
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
        # Cap the commanded TCP step so far targets glide rather than whip. A
        # higher cap applies pre-grasp (no held cube to fling — see
        # ik.max_tcp_step_pregrasp), so the drop lands before a fast belt
        # carries the cube past the trigger point.
        step_cap = (
            self.ik.max_tcp_step_pregrasp if self.phase in self._TRACKING_PHASES else self.ik.max_tcp_step
        )
        dist = float(np.linalg.norm(full_err))
        err = full_err * (step_cap / dist) if dist > step_cap else full_err
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


def _reset_episode(
    robot,
    sim: _Sim,
    rng: np.random.Generator,
    jitter_xy: float,
    domain_rand: DomainRandomizationConfig | None = None,
    yaw_jitter_deg: float = 0.0,
) -> None:
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
    if yaw_jitter_deg:
        # Rotate the cube about its vertical (Z) axis so the policy doesn't just
        # memorize one canonical orientation. Kept small by default (see the CLI
        # help) — the cube's square cross-section means a ~45° yaw presents a
        # diamond profile to the jaws instead of a flat face, which the anvil-grasp
        # timing (see PickPlaceExpert) wasn't tuned against.
        yaw = np.deg2rad(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))
        sim.data.qpos[qa + 3 : qa + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    # Zero the cube's free-joint velocity so it starts at rest.
    sim.data.qvel[sim.cube_dofadr : sim.cube_dofadr + 6] = 0.0
    if robot.config.belt_speed != 0:
        belt_act = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_ACTUATOR, "belt_motor")
        if belt_act >= 0:
            sim.data.ctrl[belt_act] = robot.config.belt_speed
    # After the keyframe reset (which would otherwise clobber mocap poses back to
    # their XML default) so distractor placement actually sticks for this episode.
    if domain_rand is not None:
        randomize_background(sim, rng, domain_rand)
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
    yaw_jitter_deg: float = 0.0,
    cameras: dict | None = None,
    seed: int = 0,
    push: bool = False,
    grasp: GraspConfig | None = None,
    ik: IKConfig | None = None,
    domain_rand: DomainRandomizationConfig | None = None,
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

    `domain_rand`: optional per-episode visual randomization (floor tint, light,
    background distractors) for sim2real robustness — see `DomainRandomizationConfig`.
    Off by default (`domain_rand=None` or `DomainRandomizationConfig(enabled=False)`).

    `yaw_jitter_deg`: optional per-episode rotation of the cube about its vertical
    axis, uniform in [-yaw_jitter_deg, +yaw_jitter_deg]. 0 (off) by default — the
    cube otherwise always starts at the same fixed orientation, which a policy can
    memorize instead of learning to recognize the cube at any heading. Verified
    47/50 (94%) success at +-20 deg across belt_speed 0.0-0.14 (jitter_xy=0.03);
    the 3 misses didn't correlate with the sampled angle (small angles missed,
    near-max angles succeeded), so they're pre-existing baseline noise, not a new
    failure mode from the rotation — +-20 deg is a safe ceiling. Untested beyond
    it — the cube's square cross-section means a
    ~45 deg yaw presents a diamond profile to the jaws instead of a flat face,
    which the anvil-grasp timing (see PickPlaceExpert) wasn't tuned against.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.sim_so101 import SimCameraConfig, SimSO101, SimSO101Config
    from lerobot.robots.sim_so101.config_sim_so101 import SimLiftSuccessConfig
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    cam_specs = cameras or {
        # Policy input: the real SO-101's only camera (wrist-mounted eye-in-hand).
        # Named "front" (not "camera1") to match the raw key real data natively
        # records under (see jobs/train/smolvla.pbs's rename_map, and
        # so101.py's `'{"observation.images.front": "observation.images.camera1"}'`
        # example) — training/eval still rename whichever camera is chosen to
        # observation.images.camera1, this is just the on-disk key.
        "front": SimCameraConfig(mujoco_name="wrist_cam", width=320, height=240),
        # Recording-only privileged external view (defined in scene_cube.xml, was
        # box_top). Not for a wrist-cam-only transfer policy — kept in the dataset
        # for a future cube-position/velocity predictor and place verification.
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
        _reset_episode(robot, sim, rng, jitter_xy, domain_rand, yaw_jitter_deg)
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
