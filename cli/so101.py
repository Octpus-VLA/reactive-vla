"""Convenience CLI for the SO-101 leader/follower arms.

Register each arm once by role (leader / follower) and the port + id are saved to
`.so101_arms.json`, so later commands never need `--port` / `--id` again:

    pixi run set-port leader      # unplug-detect & save the leader's serial port
    pixi run set-port follower    # same for the follower
    pixi run check follower       # per-motor diagnostic on the saved port
    pixi run calibrate leader     # lerobot-calibrate with the saved settings
    pixi run teleop               # lerobot-teleoperate with both saved arms

Extra flags after `calibrate` / `teleop` are forwarded to the underlying lerobot
command, e.g. `pixi run teleop --robot.cameras='{...}' --display-data=true`.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import suppress
from enum import StrEnum
from pathlib import Path

import serial.tools.list_ports
import typer
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

CONFIG_PATH = Path(__file__).resolve().parent.parent / ".so101_arms.json"

# SO-101 leader and follower share the same bus layout (IDs 1-6, all sts3215).
MOTOR_IDS: dict[str, int] = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
HALF_TURN = 2047  # int((4096 - 1) / 2) for a 12-bit sts3215 encoder
MAX_OFFSET = 2047  # 11-bit sign-magnitude limit of the Homing_Offset register


class Role(StrEnum):
    leader = "leader"
    follower = "follower"


# Per-role defaults: lerobot CLI flag prefix, device type, and a default id whose
# calibration is reused across runs.
ROLE_META: dict[str, dict[str, str]] = {
    "leader": {"prefix": "teleop", "type": "so101_leader", "id": "my_awesome_leader_arm"},
    "follower": {"prefix": "robot", "type": "so101_follower", "id": "my_awesome_follower_arm"},
}

app = typer.Typer(
    add_completion=False, help="Register SO-101 arms by role and drive them without typing ports."
)


def _load() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def _save(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")


def _ports() -> set[str]:
    return {p.device for p in serial.tools.list_ports.comports()}


def _require(cfg: dict, role: str) -> dict:
    if role not in cfg:
        raise typer.BadParameter(f"'{role}' is not registered yet. Run: pixi run set-port {role}")
    return cfg[role]


def _cameras_arg(cams: dict) -> str:
    """Render a saved cameras dict into lerobot's draccus CLI form."""
    parts = [
        f"{name}: {{type: {c['type']}, index_or_path: {c['index_or_path']}, "
        f"width: {c['width']}, height: {c['height']}, fps: {c['fps']}}}"
        for name, c in cams.items()
    ]
    return "{ " + ", ".join(parts) + "}"


PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


def _add_cameras_display(cmd: list[str], foll: dict, extra: list[str], cameras: bool, display: bool) -> None:
    """Append the follower's `--robot.cameras` and `--display_data` unless the user passed their own."""
    user_cams = any(a.startswith("--robot.cameras") for a in extra)
    user_disp = any(a.startswith(("--display_data", "--display-data")) for a in extra)
    if cameras and foll.get("cameras") and not user_cams:
        cmd.append(f"--robot.cameras={_cameras_arg(foll['cameras'])}")
    if display and not user_disp:
        cmd.append("--display_data=true")


def _add_max_rel(cmd: list[str], extra: list[str], max_rel: float | None) -> None:
    """Cap how far the follower may move per control step (degrees), for a gentler, safer motion."""
    if max_rel is not None and not any(a.startswith("--robot.max_relative_target") for a in extra):
        cmd.append(f"--robot.max_relative_target={max_rel}")


def _hf_user() -> str | None:
    """Hugging Face username via the saved token (huggingface_hub API); None if not logged in."""
    try:
        from huggingface_hub import whoami

        return whoami().get("name")
    except Exception:
        return None


def _slugify(text: str, max_len: int = 40) -> str:
    """Turn a free-text task prompt into a safe dataset-name fragment (lowercase, alnum + underscores)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return slug[:max_len].rstrip("_") or "task"


def _auto_repo_id(task: str) -> str:
    """Default dataset name when --repo-id is omitted: <task-slug>/<timestamp>.

    Mirrors the policy/dataset/timestamp nesting _job_and_output_dir uses for outputs/train.
    Note this already contains '/', so _resolve_repo passes it through as an explicit
    namespace/name rather than prefixing it with the HF user — fine for local recording,
    but pushing to the Hub later would require a real 'task-slug' namespace.
    """
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    return f"{_slugify(task)}/{ts}"


def _resolve_repo(repo_id: str, for_creation: bool = False) -> str:
    """Turn a bare `name` into `user/name`; pass `user/name` through unchanged.

    Namespace lookup order: explicit (has '/') > HF login > existing local dataset
    under $HF_LEROBOT_HOME (any namespace). When creating a new dataset without an
    HF login, fall back to the 'local' namespace; when *consuming* one, error out
    instead — a guessed namespace would send lerobot to the Hub and 401/404 there.
    """
    if "/" in repo_id:
        return repo_id
    user = _hf_user()
    if user:
        return f"{user}/{repo_id}"
    if for_creation:
        typer.secho(f"(not logged in to HF — creating dataset under local/{repo_id})", fg="yellow")
        return f"local/{repo_id}"
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        candidates = sorted(
            p
            for p in Path(HF_LEROBOT_HOME).glob(f"*/{repo_id}")
            # Skip junk dirs: a HF namespace is alphanumeric with -_. and no spaces.
            if p.is_dir() and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", p.parent.name)
        )
    except Exception:
        candidates = []
    if len(candidates) == 1:
        ns = candidates[0].parent.name
        typer.secho(f"(not logged in to HF — using local dataset {ns}/{repo_id})", fg="yellow")
        return f"{ns}/{repo_id}"
    if len(candidates) > 1:
        names = ", ".join(f"{c.parent.name}/{repo_id}" for c in candidates)
        raise typer.BadParameter(
            f"Multiple local datasets named '{repo_id}' ({names}). Pass the full 'user/name'."
        )
    raise typer.BadParameter(
        f"Can't resolve '{repo_id}': not logged in to Hugging Face and no local dataset named "
        f"'{repo_id}' under ~/.cache/huggingface/lerobot. Either pass the full id (e.g. "
        f"<user>/{repo_id}), run `pixi run hf-login`, or copy the dataset to "
        f"~/.cache/huggingface/lerobot/<user>/{repo_id} first (rsync)."
    )


def _resolve_policy(policy: str) -> str:
    """Resolve a policy reference to what lerobot loads: a checkpoint's `pretrained_model` dir or a Hub id."""
    p = Path(policy)
    if p.exists():
        # A checkpoint dir holds the loadable policy under `pretrained_model/` (alongside `training_state/`).
        if p.is_dir() and (p / "pretrained_model").is_dir() and not (p / "config.json").exists():
            return str(p / "pretrained_model")
        return policy
    # Not a local path → lerobot would treat it as a Hub repo id, which must be exactly 'namespace/name'.
    if policy.count("/") != 1:
        raise typer.BadParameter(
            f"Policy '{policy}' is not a local path and is not a valid Hub repo id ('user/name'). "
            f"Check for typos; a local checkpoint looks like "
            f"outputs/train/<job>/checkpoints/last/pretrained_model"
        )
    return policy


def _dataset_root(repo_id: str) -> Path:
    """Local directory where lerobot stores a dataset: $HF_LEROBOT_HOME/<repo_id>."""
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME) / repo_id


def _maybe_overwrite(repo: str, overwrite: bool) -> None:
    """Delete an existing local dataset dir so lerobot-record can recreate it."""
    if not overwrite:
        return
    root = _dataset_root(repo)
    if root.exists():
        typer.secho(f"--overwrite: removing existing dataset at {root}", fg="yellow")
        shutil.rmtree(root)


def _safe_video_backend() -> str | None:
    """Return 'pyav' when torchcodec is installed but cannot actually load its native libs
    (e.g. old system libstdc++ on the host clashing with the env's ffmpeg); None = use lerobot default.

    lerobot's own default only checks that torchcodec is *installed*, so a broken load
    crashes mid-training in a dataloader worker. pyav ships its own ffmpeg and always works.
    """
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        return None
    except Exception:
        typer.secho("(torchcodec can't load on this machine — using video_backend=pyav)", fg="yellow")
        return "pyav"


def _auto_device() -> str:
    """Pick the best available torch device: cuda > mps > cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# output先のフォルダ名
def _job_and_output_dir(
    policy_path_or_type: str, repo: str, job_name: str | None, output_dir: str | None
) -> tuple[str, str]:
    """Derive a job name and output dir from policy/dataset/timestamp.

    job_name is flat (used by W&B) and may repeat across runs. output_dir is always
    policy/dataset/timestamp — job_name is never folded into it, since policy+dataset already
    identify the run and a fixed extra segment would just collide with lerobot-train's
    FileExistsError on rerun instead of giving each run its own directory.
    """
    policy_slug = policy_path_or_type.rstrip("/").split("/")[-1].replace(".", "_")
    dataset_slug = repo.split("/")[-1]
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    job = job_name or f"{policy_slug}_{dataset_slug}_{ts}"
    final_output_dir = output_dir or f"outputs/train/{policy_slug}/{dataset_slug}/{ts}"
    return job, final_output_dir


@app.command("find-port")
def find_port(
    role: Role,
    id: str = typer.Option(None, help="Calibration id to store (defaults to a per-role name)."),
    type: str = typer.Option(None, help="lerobot device type (defaults to so101_leader/so101_follower)."),
) -> None:
    """Detect the serial port of the ROLE board by unplugging it, then save it."""
    meta = ROLE_META[role.value]
    before = _ports()
    if not before:
        typer.secho("No serial ports found at all — is anything connected?", fg="red")
        raise typer.Exit(1)

    typer.echo(f"Currently connected ports: {sorted(before)}")
    typer.echo(f"Unplug the USB cable of the {role.value} board, then press Enter...")
    input()

    removed: set[str] = set()
    for _ in range(20):  # poll up to ~5 s for the OS to drop the device
        removed = before - _ports()
        if removed:
            break
        time.sleep(0.25)

    if len(removed) == 0:
        typer.secho("No port disappeared. Did you unplug the right board?", fg="red")
        raise typer.Exit(1)
    if len(removed) > 1:
        typer.secho(
            f"Multiple ports disappeared ({sorted(removed)}). Unplug only the {role.value}.", fg="red"
        )
        raise typer.Exit(1)

    port = removed.pop()
    typer.echo(f"Reconnect the {role.value} cable now, then press Enter...")
    input()

    cfg = _load()
    cfg[role.value] = {
        "port": port,
        "id": id or cfg.get(role.value, {}).get("id") or meta["id"],
        "type": type or cfg.get(role.value, {}).get("type") or meta["type"],
    }
    _save(cfg)
    typer.secho(f"Saved {role.value}: {cfg[role.value]}  ->  {CONFIG_PATH.name}", fg="green")


@app.command()
def show() -> None:
    """Print the registered arms and cameras."""
    cfg = _load()
    if not cfg:
        typer.echo("No arms registered yet. Run: pixi run set-port leader / follower")
        return
    for role, info in cfg.items():
        typer.echo(f"{role:9s} port={info['port']}  id={info['id']}  type={info['type']}")
        for name, c in info.get("cameras", {}).items():
            typer.echo(
                f"          camera '{name}': index={c['index_or_path']} {c['width']}x{c['height']}@{c['fps']} ({c['type']})"
            )


@app.command("set-camera")
def set_camera(
    name: str = typer.Argument(..., help="Camera name shown in the viewer, e.g. 'front' or 'wrist'."),
    index: int = typer.Option(0, help="OpenCV camera index from `pixi run find-cameras`."),
    width: int = typer.Option(640, help="Requested frame width."),
    height: int = typer.Option(480, help="Requested frame height."),
    fps: int = typer.Option(30, help="Requested frames per second."),
    remove: bool = typer.Option(False, "--remove", help="Remove this camera instead of adding it."),
) -> None:
    """Attach (or remove) an OpenCV camera on the follower; `teleop` shows it automatically."""
    cfg = _load()
    foll = _require(cfg, "follower")  # cameras live on the follower robot
    cams = foll.get("cameras", {})
    if remove:
        cams.pop(name, None)
    else:
        cams[name] = {"type": "opencv", "index_or_path": index, "width": width, "height": height, "fps": fps}
    foll["cameras"] = cams
    _save(cfg)
    typer.secho(f"follower cameras: {foll['cameras'] or '(none)'}", fg="green")


@app.command()
def check(role: Role) -> None:
    """Per-motor diagnostic (raw position, homing offset, reachability) on the saved port."""
    info = _require(_load(), role.value)
    motors = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for n, i in MOTOR_IDS.items()}
    bus = FeetechMotorsBus(port=info["port"], motors=motors)
    bus.connect(handshake=False)  # don't abort if a motor is missing; report per-motor below
    try:
        typer.echo(f"{role.value} @ {info['port']}")
        typer.echo(
            f"{'motor':14s} {'id':>2s} {'raw_pos':>8s} {'homing_off':>11s} {'true_pos':>9s} {'calib_off':>10s}"
        )
        bad: list[str] = []
        missing: list[str] = []
        stale: list[str] = []
        for name in motors:
            try:
                pos = bus.read("Present_Position", name, normalize=False, num_retry=2)
                off = bus.read("Homing_Offset", name, normalize=False, num_retry=2)
            except Exception:
                typer.secho(f"{name:14s} {motors[name].id:>2d} {'NO RESPONSE':>45s}", fg="red")
                missing.append(name)
                continue
            actual = pos + off
            needed = actual - HALF_TURN
            out = abs(needed) > MAX_OFFSET
            flag = "  <-- OUT OF RANGE" if out else ""
            typer.echo(
                f"{name:14s} {motors[name].id:>2d} {pos:>8d} {off:>11d} {actual:>9d} {needed:>10d}{flag}"
            )
            if out:
                bad.append(name)
            if off != 0:
                stale.append(name)
    finally:
        bus.disconnect()

    typer.echo("")
    if missing:
        typer.secho(
            f"Not responding: {', '.join(missing)} — check the daisy-chain cable/power to those", fg="red"
        )
        typer.secho("motors, and that their IDs were assigned (pixi run lerobot-setup-motors ...).", fg="red")
    if stale:
        typer.echo(f"Stale Homing_Offset (≠0) on: {', '.join(stale)} — leftover from a previous calibration;")
        typer.echo("lerobot-calibrate resets these before reading, so it is normally fine.")
    if bad:
        typer.secho(
            f"Out-of-range joints: {', '.join(bad)} — move them toward centre (true_pos≈2047),", fg="yellow"
        )
        typer.secho(
            "then re-run calibration. If a joint can't reach centre, its horn is mounted off-centre.",
            fg="yellow",
        )
    if not missing and not bad:
        typer.secho("All motors responded and are within range; calibration should succeed.", fg="green")


@app.command("setup-motors", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def setup_motors(
    ctx: typer.Context,
    role: Role,
    motor: str = typer.Option(
        None,
        "--motor",
        help="Assign only these motor(s) by name or id, e.g. 'shoulder_lift', '2', or '2,4'. "
        "Omit to set all six (the standard lerobot flow).",
    ),
) -> None:
    """Assign Feetech motor IDs for ROLE using the saved port.

    Without --motor: runs the standard lerobot-setup-motors (all six, one at a time).
    With --motor: re-assigns just the given motor(s) — connect ONLY that motor to the bus
    when prompted (same per-motor primitive the full flow uses, exposed for fixing one joint).
    """
    info = _require(_load(), role.value)
    if not motor:
        p = ROLE_META[role.value]["prefix"]
        cmd = [
            "lerobot-setup-motors",
            f"--{p}.type={info['type']}",
            f"--{p}.port={info['port']}",
            *ctx.args,
        ]
        _run(cmd)

    # Single/selected-motor path via the bus primitive.
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    id_to_name = {i: n for n, i in MOTOR_IDS.items()}
    names = []
    for tok in motor.split(","):
        tok = tok.strip()
        if tok.isdigit():
            mid = int(tok)
            if mid not in id_to_name:
                raise typer.BadParameter(f"id {mid} is not 1-6 (motors: {MOTOR_IDS})")
            names.append(id_to_name[mid])
        elif tok in MOTOR_IDS:
            names.append(tok)
        else:
            raise typer.BadParameter(f"unknown motor '{tok}'. Use a name {list(MOTOR_IDS)} or id 1-6.")

    motors = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for n, i in MOTOR_IDS.items()}
    bus = FeetechMotorsBus(port=info["port"], motors=motors)
    try:
        for name in names:
            typer.secho(
                f"Connect the controller board to the '{name}' (id {MOTOR_IDS[name]}) motor ONLY, then press Enter...",
                fg="yellow",
            )
            input()
            bus.setup_motor(name)
            typer.secho(f"'{name}' id set to {MOTOR_IDS[name]}", fg="green")
    finally:
        # Only one motor is physically on the bus, so the default disconnect (which
        # disables torque on ALL six mapped motors) would fail on the absent ids.
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
    raise typer.Exit(0)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def calibrate(ctx: typer.Context, role: Role) -> None:
    """Run lerobot-calibrate for ROLE using the saved port/id. Extra flags are forwarded."""
    info = _require(_load(), role.value)
    p = ROLE_META[role.value]["prefix"]
    cmd = [
        "lerobot-calibrate",
        f"--{p}.type={info['type']}",
        f"--{p}.port={info['port']}",
        f"--{p}.id={info['id']}",
        *ctx.args,
    ]
    typer.secho("$ " + " ".join(cmd), fg="blue")
    raise typer.Exit(subprocess.run(cmd).returncode)


def _rerun_pids() -> set[int]:
    """PIDs of running Rerun viewer processes."""
    try:
        import psutil
    except Exception:
        return set()
    pids = set()
    for p in psutil.process_iter(["name"]):
        try:
            if "rerun" in (p.info["name"] or "").lower():
                pids.add(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return pids


def _kill_new_rerun(before: set[int]) -> None:
    """Close Rerun viewers that appeared since `before` (lerobot's `rr.spawn` leaves them running)."""
    try:
        import psutil
    except Exception:
        return
    victims = []
    for p in psutil.process_iter(["name"]):
        try:
            if p.pid not in before and "rerun" in (p.info["name"] or "").lower():
                victims.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if not victims:
        return
    for p in victims:
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            p.terminate()
    _, alive = psutil.wait_procs(victims, timeout=3)
    for p in alive:
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            p.kill()
    typer.secho(f"(closed {len(victims)} leftover Rerun viewer process(es))", fg="yellow")


def _run(cmd: list[str], cleanup_rerun: bool = False) -> None:
    typer.secho("$ " + " ".join(cmd), fg="blue")
    before = _rerun_pids() if cleanup_rerun else set()
    rc = subprocess.run(cmd).returncode
    if cleanup_rerun:
        _kill_new_rerun(before)
    raise typer.Exit(rc)


def _arm_flags(prefix: str, info: dict) -> list[str]:
    return [
        f"--{prefix}.type={info['type']}",
        f"--{prefix}.port={info['port']}",
        f"--{prefix}.id={info['id']}",
    ]


@app.command(context_settings=PASSTHROUGH)
def teleop(
    ctx: typer.Context,
    max_rel: float = typer.Option(
        None,
        "--max-rel",
        help="Safety cap: max degrees a follower joint may move per control step (e.g. 5). Makes the initial sync ramp up gently instead of snapping.",
    ),
    display: bool = typer.Option(
        True, "--display/--no-display", help="Show camera & joint data in the Rerun viewer."
    ),
    keep_viewer: bool = typer.Option(
        False,
        "--keep-viewer",
        help="Leave the Rerun viewer open after exit (default: close the viewer this run spawned).",
    ),
    cameras: bool = typer.Option(
        True, "--cameras/--no-cameras", help="Attach the follower's registered cameras."
    ),
) -> None:
    """Run lerobot-teleoperate using both saved arms (+ registered cameras). Extra flags are forwarded."""
    cfg = _load()
    lead = _require(cfg, "leader")
    foll = _require(cfg, "follower")
    cmd = ["lerobot-teleoperate", *_arm_flags("robot", foll), *_arm_flags("teleop", lead)]
    extra = list(ctx.args)
    _add_cameras_display(cmd, foll, extra, cameras, display)
    _add_max_rel(cmd, extra, max_rel)
    _run(cmd + extra, cleanup_rerun=display and not keep_viewer)


@app.command(context_settings=PASSTHROUGH)
def record(
    ctx: typer.Context,
    task: str = typer.Option(
        ..., "--task", help="Natural-language task description stored with the dataset."
    ),
    repo_id: str = typer.Option(
        None,
        "--repo-id",
        help="Dataset id: a bare 'name' (prefixed with your HF user) or 'user/name'. "
        "Auto-generated as '<task-slug>/<timestamp>' if omitted (required with --resume) — "
        "note this skips HF user prefixing, so pushing it to the Hub later needs a real "
        "'<task-slug>' namespace.",
    ),
    episodes: int = typer.Option(5, "--episodes", help="Number of episodes to record."),
    episode_time: float = typer.Option(
        20,
        "--episode-time",
        help="Seconds per episode before it auto-stops (lerobot default 60). Right-arrow ends one early.",
    ),
    reset_time: float = typer.Option(
        5, "--reset-time", help="Seconds to reset the scene between episodes (lerobot default 60)."
    ),
    fps: int = typer.Option(30, "--fps"),
    push: bool = typer.Option(
        False, "--push/--no-push", help="Upload the dataset to the Hugging Face Hub (needs `hf auth login`)."
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Delete an existing local dataset with this id before recording (lerobot won't overwrite on its own).",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Append to an existing dataset; --episodes then means N *additional* episodes (e.g. after `drop`).",
    ),
    max_rel: float = typer.Option(
        None, "--max-rel", help="Safety cap: max degrees a follower joint may move per control step (e.g. 5)."
    ),
    display: bool = typer.Option(True, "--display/--no-display"),
    keep_viewer: bool = typer.Option(
        False, "--keep-viewer", help="Leave the Rerun viewer open after exit (default: close it)."
    ),
    cameras: bool = typer.Option(True, "--cameras/--no-cameras"),
) -> None:
    """Record a teleoperated dataset (lerobot-record) using both saved arms + cameras. Extra flags forwarded.

    Recording starts automatically; control it from the (focused) terminal with the arrow keys:
    Right=stop the current episode and continue, Left=re-record it, Esc=stop the whole session.
    """
    if overwrite and resume:
        raise typer.BadParameter("--overwrite and --resume are mutually exclusive.")
    if resume and not repo_id:
        raise typer.BadParameter("--repo-id is required with --resume (it must name an existing dataset).")
    if not repo_id:
        repo_id = _auto_repo_id(task)
        typer.secho(f"(--repo-id omitted — using auto-generated '{repo_id}')", fg="yellow")
    cfg = _load()
    lead = _require(cfg, "leader")
    foll = _require(cfg, "follower")
    repo = _resolve_repo(repo_id, for_creation=not resume)
    _maybe_overwrite(repo, overwrite)
    cmd = [
        "lerobot-record",
        *_arm_flags("robot", foll),
        *_arm_flags("teleop", lead),
        f"--dataset.repo_id={repo}",
        f"--dataset.num_episodes={episodes}",
        f"--dataset.single_task={task}",
        f"--dataset.fps={fps}",
        f"--dataset.push_to_hub={'true' if push else 'false'}",
    ]
    if resume:
        cmd.append("--resume=true")
    if episode_time is not None:
        cmd.append(f"--dataset.episode_time_s={episode_time}")
    if reset_time is not None:
        cmd.append(f"--dataset.reset_time_s={reset_time}")
    extra = list(ctx.args)
    _add_cameras_display(cmd, foll, extra, cameras, display)
    _add_max_rel(cmd, extra, max_rel)
    _run(cmd + extra, cleanup_rerun=display and not keep_viewer)


@app.command(context_settings=PASSTHROUGH)
def train(
    ctx: typer.Context,
    repo_id: str = typer.Option(
        None, "--repo-id", help="Dataset id ('name' → prefixed with your HF user). Required unless --resume."
    ),
    policy: str = typer.Option(
        None,
        "--policy",
        help="Policy architecture for training from scratch: act, diffusion, smolvla, pi0, ... "
        "Mutually exclusive with --policy-path.",
    ),
    policy_path: str = typer.Option(
        None,
        "--policy-path",
        help="Pretrained model to fine-tune: a local checkpoint dir or a Hub model id "
        "(e.g. 'lerobot/smolvla_base'). Mutually exclusive with --policy.",
    ),
    device: str = typer.Option(None, "--device", help="cuda / mps / cpu (auto-detected if omitted)."),
    job_name: str = typer.Option(None, "--job-name", help="Defaults to <policy>_<dataset>."),
    output_dir: str = typer.Option(
        None, "--output-dir", help="Defaults to outputs/train/<policy>/<dataset>/<timestamp>."
    ),
    steps: int = typer.Option(
        None, "--steps", help="Total number of training steps (lerobot's default depends on the policy)."
    ),
    batch_size: int = typer.Option(None, "--batch-size", help="Training batch size."),
    save_freq: int = typer.Option(None, "--save-freq", help="Save a checkpoint every N steps."),
    use_wandb: bool = typer.Option(
        False, "--wandb/--no-wandb", help="Log to Weights & Biases (needs `pixi run wandb-login`)."
    ),
    wandb_project: str = typer.Option(None, "--wandb-project", help="W&B project name (default: 'lerobot')."),
    wandb_entity: str = typer.Option(None, "--wandb-entity", help="W&B entity (team or user) to log under."),
    push_repo_id: str = typer.Option(
        None,
        "--push-repo-id",
        help="Hub model repo to push the trained policy after training ('name' → prefixed with HF user). "
        "Omit to keep it local only.",
    ),
    resume: str = typer.Option(
        None, "--resume", help="Resume from a checkpoint dir or its train_config.json."
    ),
) -> None:
    """Train a policy (lerobot-train). Extra flags are forwarded.

    Two modes:
      --policy act            Train from scratch with that architecture.
      --policy-path user/repo Fine-tune from a pretrained Hub model or local checkpoint.
    """
    if policy and policy_path:
        raise typer.BadParameter("--policy and --policy-path are mutually exclusive.")
    if resume:
        cfg_path = (
            resume
            if resume.endswith(".json")
            else str(Path(resume) / "pretrained_model" / "train_config.json")
        )
        _run(["lerobot-train", f"--config_path={cfg_path}", "--resume=true", *ctx.args])
    if not repo_id:
        raise typer.BadParameter("--repo-id is required (unless you pass --resume).")
    if not policy and not policy_path:
        raise typer.BadParameter("One of --policy or --policy-path is required.")
    repo = _resolve_repo(repo_id)
    job, final_output_dir = _job_and_output_dir(policy or policy_path, repo, job_name, output_dir)
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={repo}",
        f"--output_dir={final_output_dir}",
        f"--job_name={job}",
        f"--policy.device={device or _auto_device()}",
        f"--wandb.enable={'true' if use_wandb else 'false'}",
    ]
    if policy_path:
        cmd.append(f"--policy.path={policy_path}")
    else:
        cmd.append(f"--policy.type={policy}")
    if use_wandb and wandb_project:
        cmd.append(f"--wandb.project={wandb_project}")
    if use_wandb and wandb_entity:
        cmd.append(f"--wandb.entity={wandb_entity}")
    if steps is not None:
        cmd.append(f"--steps={steps}")
    if batch_size is not None:
        cmd.append(f"--batch_size={batch_size}")
    if save_freq is not None:
        cmd.append(f"--save_freq={save_freq}")
    if not any(a.startswith("--dataset.video_backend") for a in ctx.args):
        backend = _safe_video_backend()
        if backend:
            cmd.append(f"--dataset.video_backend={backend}")
    if push_repo_id:
        hub_repo = _resolve_repo(push_repo_id, for_creation=True)
        cmd += [f"--policy.repo_id={hub_repo}", "--policy.push_to_hub=true"]
    else:
        cmd.append("--policy.push_to_hub=false")
    _run(cmd + list(ctx.args))


@app.command("eval", context_settings=PASSTHROUGH)
def evaluate(
    ctx: typer.Context,
    policy: str = typer.Option(
        ..., "--policy", help="Trained policy: a local checkpoint dir or a Hub repo id."
    ),
    task: str = typer.Option(..., "--task", help="Natural-language task description."),
    repo_id: str = typer.Option(
        ...,
        "--repo-id",
        help="Eval dataset id; should start with 'rollout_'. 'name' → prefixed with HF user.",
    ),
    episodes: int = typer.Option(10, "--episodes"),
    episode_time: float = typer.Option(
        None,
        "--episode-time",
        help="Seconds per episode before it auto-stops (lerobot default 60). Right-arrow ends one early.",
    ),
    reset_time: float = typer.Option(
        None,
        "--reset-time",
        help="Seconds between episodes for the follower to return to its initial position (lerobot default 60).",
    ),
    fps: int = typer.Option(30, "--fps"),
    push: bool = typer.Option(False, "--push/--no-push"),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Delete an existing local eval dataset with this id first."
    ),
    max_rel: float = typer.Option(
        None,
        "--max-rel",
        help="Safety cap: max degrees the follower may move per step (e.g. 5). Recommended for autonomous eval to limit sudden motion.",
    ),
    display: bool = typer.Option(True, "--display/--no-display"),
    keep_viewer: bool = typer.Option(
        False, "--keep-viewer", help="Leave the Rerun viewer open after exit (default: close it)."
    ),
    cameras: bool = typer.Option(True, "--cameras/--no-cameras"),
    rtc: bool = typer.Option(
        False,
        "--rtc",
        help="Use async Real-Time Chunking inference (background-thread replan) instead of the default sync engine.",
    ),
    execution_horizon: int = typer.Option(
        10,
        "--execution-horizon",
        help="RTC only: leftover-blend / guidance horizon in frames. Policy and engine side are kept equal.",
    ),
    queue_threshold: int = typer.Option(
        30,
        "--queue-threshold",
        help="RTC only: replan when the action queue drops to this many steps (replan interval ~= chunk_size - this).",
    ),
    detector: str = typer.Option(
        "none",
        "--detector",
        help="RTC only: add event-triggered / speed-adaptive replanning. Options: none, motion, red_cube_speed.",
    ),
    detector_camera: str = typer.Option(
        None,
        "--detector-camera",
        help="Camera key the detector watches (defaults to a registered camera named 'overall', else the first). Must be a registered camera.",
    ),
    require_target_visible: bool = typer.Option(
        False,
        "--require-target-visible/--no-require-target-visible",
        help="RTC detector only: suppress queue-based planning until the detector sees the target.",
    ),
) -> None:
    """Run a trained policy on the follower and record eval episodes (lerobot-rollout, episodic strategy).

    Between episodes the follower automatically returns to its startup position (no leader needed).
    Pass --rtc to drive the follower with async Real-Time Chunking inference instead of the sync engine.
    Add --detector to let a camera detector control the RTC replan timing (red_cube_speed adapts it to
    cube speed). Fine-tune the detector with passthrough flags, e.g.
    --inference.supervisor.detector.urgent_speed_px_s=300.
    """
    cfg = _load()
    foll = _require(cfg, "follower")  # the policy drives the follower; no leader needed
    repo = _resolve_repo(repo_id, for_creation=True)
    name = repo.split("/")[-1]
    if not name.startswith("rollout_"):
        raise typer.BadParameter(
            f"lerobot requires rollout dataset names to start with 'rollout_' (you gave '{name}'). "
            f"Use e.g. --repo-id rollout_test."
        )
    _maybe_overwrite(repo, overwrite)
    cmd = [
        "lerobot-rollout",
        *_arm_flags("robot", foll),
        f"--policy.path={_resolve_policy(policy)}",
        "--strategy.type=episodic",
        f"--task={task}",
        f"--fps={fps}",
        f"--dataset.repo_id={repo}",
        f"--dataset.num_episodes={episodes}",
        f"--dataset.fps={fps}",
        f"--dataset.push_to_hub={'true' if push else 'false'}",
    ]
    if rtc:
        # Async RTC: replan in a background thread, blending the leftover chunk over execution_horizon.
        # execution_horizon lives in two places (policy guidance + engine queue) and must match — see
        # docs/rtc-sim-rollout.md. Guidance reads the policy-side config, so enable it there too.
        cmd += [
            "--inference.type=rtc",
            f"--inference.rtc.execution_horizon={execution_horizon}",
            f"--inference.queue_threshold={queue_threshold}",
            "--policy.rtc_config.enabled=true",
            f"--policy.rtc_config.execution_horizon={execution_horizon}",
        ]
    else:
        cmd.append("--inference.type=sync")
    if detector != "none":
        if not rtc:
            raise typer.BadParameter("--detector requires --rtc (the detector drives the RTC replan gate).")
        if detector not in {"motion", "red_cube_speed"}:
            raise typer.BadParameter(
                f"--detector must be one of none, motion, red_cube_speed (got '{detector}')."
            )
        cam_key = detector_camera
        if cam_key is None:
            cam_keys = list((foll.get("cameras") or {}).keys())
            if not cam_keys:
                raise typer.BadParameter(
                    "--detector needs a camera but none are registered. "
                    "Register one with `pixi run set-camera`, or pass --detector-camera."
                )
            cam_key = "overall" if "overall" in cam_keys else cam_keys[0]
        # Detector drives the RTC replan gate (see lerobot.detectors / docs/supervisor-trigger.md).
        cmd += [
            "--inference.supervisor.enabled=true",
            f"--inference.supervisor.detector.type={detector}",
            f"--inference.supervisor.camera={cam_key}",
        ]
        if require_target_visible:
            cmd.append("--inference.supervisor.require_target_visible=true")
    if episode_time is not None:
        cmd.append(f"--dataset.episode_time_s={episode_time}")
    if reset_time is not None:
        cmd.append(f"--dataset.reset_time_s={reset_time}")
    extra = list(ctx.args)
    _add_cameras_display(cmd, foll, extra, cameras, display)
    _add_max_rel(cmd, extra, max_rel)
    _run(cmd + extra, cleanup_rerun=display and not keep_viewer)


@app.command("sim-eval", context_settings=PASSTHROUGH)
def sim_eval(
    ctx: typer.Context,
    policy: str = typer.Option(
        ..., "--policy", help="Trained policy: a local checkpoint dir or a Hub repo id."
    ),
    task: str = typer.Option("Grab the cube", "--task", help="Natural-language task description."),
    mjcf_path: str = typer.Option(
        "assets/so101/scene_cube.xml",
        "--mjcf-path",
        help="MuJoCo scene XML. Must include the success-check body (default scene has 'cube').",
    ),
    episodes: int = typer.Option(10, "--episodes", help="Number of episodes to run."),
    episode_time: float = typer.Option(30, "--episode-time", help="Max wall-clock seconds per episode."),
    episode_steps: int = typer.Option(
        None,
        "--episode-steps",
        help="Max control steps per episode, in addition to --episode-time (whichever hits first "
        "ends the episode). --episode-time alone is wall-clock, not sim time, so the actual step "
        "count it produces varies with render/inference speed — set this for a reproducible step "
        "count (e.g. 600 for ~20s of sim time at --fps 30).",
    ),
    reset_time: float = typer.Option(3, "--reset-time", help="Seconds held between episodes."),
    belt_speed: float = typer.Option(
        0.0,
        "--belt-speed",
        help="Conveyor belt speed in m/s (scene_cube.xml's belt; 0 = stationary, the default). "
        "Constant for the whole rollout, not a per-step control.",
    ),
    belt_distance: float = typer.Option(
        0.14,
        "--belt-distance",
        help="Distance in meters from the robot base to the belt's near edge (default 0.14 = "
        "14cm). Slides the whole belt + box + cube layout forward/back. Keep the cube within "
        "the arm's ~0.40m reach; the start pose is tuned for the default, so large changes may "
        "not frame the cube in the wrist camera.",
    ),
    success_body: str = typer.Option(
        "cube", "--success-body", help="MJCF body to track for success (must have a freejoint)."
    ),
    success_criterion: str = typer.Option(
        "lift",
        "--success-criterion",
        help="How an episode counts as a success: 'lift' (body raised --success-height above "
        "rest) or 'place_in_box' (body settled inside the box). Default 'lift'.",
    ),
    success_height: float = typer.Option(
        0.05,
        "--success-height",
        help="'lift' only: meters the body must rise above its resting height to count as lifted.",
    ),
    fps: int = typer.Option(30, "--fps"),
    output: str = typer.Option(
        None,
        "--output",
        help="Where to write the per-episode + aggregate JSON summary. "
        "Defaults to outputs/eval/<policy-slug>/<MMDD_HHMM>/summary.json.",
    ),
    rtc: bool = typer.Option(
        False, "--rtc", help="Use async Real-Time Chunking inference instead of the default sync engine."
    ),
    execution_horizon: int = typer.Option(10, "--execution-horizon", help="RTC only: see `eval --help`."),
    queue_threshold: int = typer.Option(30, "--queue-threshold", help="RTC only: see `eval --help`."),
    repo_id: str = typer.Option(
        None,
        "--repo-id",
        help="Eval dataset id to also record video/frames for every episode (must start with "
        "'rollout_', e.g. rollout_sim_test). Omit to skip recording (success metrics only).",
    ),
    push: bool = typer.Option(False, "--push/--no-push", help="Upload the recorded dataset to the Hub."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Delete an existing local eval dataset with this id first."
    ),
) -> None:
    """Run a trained policy in the MuJoCo sim and score task success rate / success step.

    Success is read directly from privileged sim state (never shown to the
    policy), via `--success-criterion`:
    - `lift` (default): the tracked body rose `--success-height` above where it
      rested (a robosuite/LIBERO-style Lift criterion).
    - `place_in_box`: the tracked body came to rest inside the scene's `box`
      body (pick-and-place). Pair with `--belt-speed` for the dynamic task.
    Requires a freejoint body named `--success-body` (the bundled
    `scene_cube.xml` has one named "cube"). See docs/rtc-sim-rollout.md.

    No recording happens unless `--repo-id` is given — by default this only
    measures success rate / success step, exactly like the rest of `eval`'s
    docs above describe.
    """
    # Headless offscreen rendering by default — without it MuJoCo falls back to a
    # windowed GLFW context and crashes on HPC nodes with no DISPLAY. osmesa (CPU
    # rendering) rather than egl (GPU rendering) because sim-eval interleaves
    # MuJoCo renders with CUDA policy inference every control tick, and egl+CUDA
    # contending for the same GPU was measured to stall individual renders by
    # ~19s (see docs/rtc-sim-rollout.md); osmesa has no such GPU contention,
    # even though a single render is nominally slower in isolation (~80ms vs
    # ~6ms). Respect an explicit override (e.g. MUJOCO_GL=egl on a multi-GPU
    # node where rendering and inference can live on separate devices).
    os.environ.setdefault("MUJOCO_GL", "osmesa")

    policy_slug = policy.rstrip("/").split("/")[-1].replace(".", "_")
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    out_path = output or f"outputs/eval/{policy_slug}/{ts}/summary.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    dataset_args = []
    if repo_id:
        repo = _resolve_repo(repo_id, for_creation=True)
        name = repo.split("/")[-1]
        if not name.startswith("rollout_"):
            raise typer.BadParameter(
                f"lerobot requires rollout dataset names to start with 'rollout_' (you gave '{name}'). "
                f"Use e.g. --repo-id rollout_sim_test."
            )
        _maybe_overwrite(repo, overwrite)
        dataset_args = [
            f"--dataset.repo_id={repo}",
            f"--dataset.fps={fps}",
            f"--dataset.push_to_hub={'true' if push else 'false'}",
        ]
        typer.secho(f"(recording episodes to {repo})", fg="yellow")

    cmd = [
        "lerobot-rollout",
        f"--policy.path={_resolve_policy(policy)}",
        "--robot.type=sim_so101",
        f"--robot.mjcf_path={Path(mjcf_path).resolve()}",
        # camera1=wrist_cam: the real SO-101 rig's only camera is wrist-mounted
        # (eye-in-hand); wrist_cam is the Menagerie so101.xml's built-in camera
        # at that same CAD-derived mount, so that's the policy's only input
        # camera too. overview: fixed external view (added in scene_cameras.xml,
        # not part of upstream so101.xml), not consumed by this policy but
        # recorded into the dataset (with --repo-id) for a future cube-position/
        # velocity predictor. The rollout context rejects any robot camera the
        # policy doesn't expect unless --rename_map is set (it skips that check
        # entirely) — the no-op entry below exists only to take that branch.
        "--robot.cameras={camera1: {mujoco_name: wrist_cam, width: 320, height: 240}, "
        "overview: {mujoco_name: overview, width: 320, height: 240}}",
        '--rename_map={"observation.images.overview": "observation.images.overview"}',
        f"--robot.control_fps={fps}",
        f"--robot.belt_speed={belt_speed}",
        f"--robot.belt_distance={belt_distance}",
        f"--robot.success.body_name={success_body}",
        f"--robot.success.criterion={success_criterion}",
        f"--robot.success.height_m={success_height}",
        "--strategy.type=eval",
        f"--strategy.num_episodes={episodes}",
        f"--strategy.episode_time_s={episode_time}",
        f"--strategy.reset_time_s={reset_time}",
        f"--strategy.output_path={out_path}",
        f"--task={task}",
        f"--fps={fps}",
        "--play_sounds=false",
        *dataset_args,
    ]
    if episode_steps is not None:
        cmd.append(f"--strategy.episode_steps={episode_steps}")
    if rtc:
        cmd += [
            "--inference.type=rtc",
            f"--inference.rtc.execution_horizon={execution_horizon}",
            f"--inference.queue_threshold={queue_threshold}",
            "--policy.rtc_config.enabled=true",
            f"--policy.rtc_config.execution_horizon={execution_horizon}",
        ]
    else:
        cmd.append("--inference.type=sync")
    typer.secho(f"(summary will be written to {out_path})", fg="yellow")
    _run(cmd + list(ctx.args))


@app.command("sim-collect")
def sim_collect(
    repo_id: str = typer.Option(
        None,
        "--repo-id",
        help="Dataset id to create ('name' → prefixed with your HF user, or 'user/name'). "
        "Auto-generated as '<task-slug>/<timestamp>' if omitted.",
    ),
    task: str = typer.Option("Grab the cube", "--task", help="Natural-language task stored with the dataset."),
    mjcf_path: str = typer.Option(
        "assets/so101/scene_cube.xml", "--mjcf-path", help="MuJoCo scene XML (needs the 'cube' and 'box' bodies)."
    ),
    episodes: int = typer.Option(20, "--episodes", help="Number of demo episodes to record."),
    max_steps: int = typer.Option(240, "--max-steps", help="Hard cap on control steps per episode."),
    fps: int = typer.Option(30, "--fps", help="Control rate; also the dataset fps. Keep matched to training."),
    belt_speed: float = typer.Option(
        0.0,
        "--belt-speed",
        help="Conveyor speed (m/s). 0 = static cube parked in front of the robot; non-zero feeds the "
        "cube from the -y end and the expert leads a constant-velocity intercept.",
    ),
    belt_distance: float = typer.Option(0.14, "--belt-distance", help="Metres from the robot base to the belt's near edge."),
    jitter: float = typer.Option(
        0.03, "--jitter", help="Uniform ±metres of xy randomisation on the cube start, so demos span grasp positions."
    ),
    seed: int = typer.Option(0, "--seed", help="RNG seed for cube-position randomisation (reproducible datasets)."),
    push: bool = typer.Option(False, "--push/--no-push", help="Upload the recorded dataset to the Hugging Face Hub."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Delete an existing local dataset with this id first."),
) -> None:
    """Record scripted-expert pick-and-place demos in the MuJoCo sim (no hardware).

    A privileged IK controller (reads the cube's true pose, solves IK, servos the
    arm) drives the SimSO101 robot through approach → grasp → carry → place, and
    every (observation, action) pair is written to a LeRobotDataset with the same
    schema `record` produces — so `pixi run train` consumes it directly. The point
    is sim-*rendered* observations: fine-tuning a real-data SmolVLA on these closes
    the real→sim visual gap that leaves it motionless in `sim-eval`. The expert's
    privileged cube reads never enter the dataset (only wrist cam + joint state +
    commanded targets). Design notes: docs/sim-scripted-collect.md.
    """
    # Headless GPU rendering: unlike sim-eval (which interleaves CUDA policy
    # inference and contends for the GPU, so it forces osmesa), collection runs no
    # policy, so egl is both safe and much faster per render. Respect an override.
    os.environ.setdefault("MUJOCO_GL", "egl")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sim_collect

    if not repo_id:
        repo_id = _auto_repo_id(task)
        typer.secho(f"(--repo-id omitted — using auto-generated '{repo_id}')", fg="yellow")
    repo = _resolve_repo(repo_id, for_creation=True)
    _maybe_overwrite(repo, overwrite)
    typer.secho(f"(recording {episodes} scripted episodes to {repo})", fg="yellow")
    summary = sim_collect.collect(
        repo_id=repo,
        task=task,
        mjcf_path=str(Path(mjcf_path).resolve()),
        episodes=episodes,
        max_steps=max_steps,
        fps=fps,
        belt_speed=belt_speed,
        belt_distance=belt_distance,
        jitter_xy=jitter,
        seed=seed,
        push=push,
    )
    typer.secho(
        f"recorded {summary['episodes']} episodes, "
        f"{sum(summary['success'])} placed in box ({summary['success_rate']:.0%})",
        fg="green",
    )


@app.command(context_settings=PASSTHROUGH)
def replay(
    ctx: typer.Context,
    repo_id: str = typer.Option(..., "--repo-id", help="Dataset id ('name' → prefixed with your HF user)."),
    episode: int = typer.Option(0, "--episode", help="Episode index to replay on the follower."),
) -> None:
    """Replay one recorded episode on the follower (lerobot-replay). Extra flags are forwarded."""
    cfg = _load()
    foll = _require(cfg, "follower")  # replay drives the follower; no leader/cameras needed
    cmd = [
        "lerobot-replay",
        *_arm_flags("robot", foll),
        f"--dataset.repo_id={_resolve_repo(repo_id)}",
        f"--dataset.episode={episode}",
    ]
    _run(cmd + list(ctx.args))


@app.command(context_settings=PASSTHROUGH)
def viz(
    ctx: typer.Context,
    repo_id: str = typer.Option(..., "--repo-id", help="Dataset id ('name' → prefixed with your HF user)."),
    episode: int = typer.Option(0, "--episode", help="Episode index to visualize."),
) -> None:
    """Visualize a recorded episode (frames, states, actions) in a Rerun viewer (lerobot-dataset-viz)."""
    cmd = ["lerobot-dataset-viz", "--repo-id", _resolve_repo(repo_id), "--episode-index", str(episode)]
    _run(cmd + list(ctx.args))


@app.command(context_settings=PASSTHROUGH)
def drop(
    ctx: typer.Context,
    repo_id: str = typer.Option(..., "--repo-id", help="Dataset id ('name' → prefixed with your HF user)."),
    episodes: str = typer.Option(
        ..., "--episodes", help="Comma-separated episode indices to delete, e.g. 0,2,5"
    ),
) -> None:
    """Delete bad episodes from a local dataset in place (lerobot-edit-dataset; a backup is created).

    Remaining episodes are re-indexed from 0, so re-check indices with `viz` before dropping again.
    Re-record the dropped count afterwards with: record --resume --episodes N.
    """
    repo = _resolve_repo(repo_id)
    try:
        idx = sorted({int(e) for e in episodes.split(",")})
    except ValueError:
        raise typer.BadParameter(f"--episodes must be comma-separated integers, got '{episodes}'") from None
    cmd = [
        "lerobot-edit-dataset",
        f"--repo_id={repo}",
        "--operation.type=delete_episodes",
        f"--operation.episode_indices=[{', '.join(map(str, idx))}]",
    ]
    _run(cmd + list(ctx.args))


@app.command("policy-test")
def policy_test(
    policy: str = typer.Option(..., "--policy", help="Trained policy: a checkpoint dir or a Hub repo id."),
    repo_id: str = typer.Option(
        ..., "--repo-id", help="Dataset whose recorded frames are used as observations."
    ),
    device: str = typer.Option(None, "--device", help="cuda / mps / cpu (auto-detected if omitted)."),
    steps: int = typer.Option(20, "--steps", help="Number of inference steps to run."),
    episode: int = typer.Option(0, "--episode", help="Episode to take frames from."),
    rename_map: str = typer.Option(
        None,
        "--rename_map",
        help="JSON dict mapping dataset observation keys to the policy's expected names "
        "(same option as `train`/`eval`), e.g. "
        '\'{"observation.images.front": "observation.images.camera1"}\'. Required if the '
        "dataset's camera keys don't match what the checkpoint was fine-tuned with.",
    ),
) -> None:
    """Offline inference smoke test: run the trained policy on recorded dataset frames — no robot needed.

    Exercises the same pipeline as `eval` (policy load, pre/post-processors, video decode,
    predict_action) and reports latency plus deviation from the recorded actions.
    """
    import numpy as np
    import torch
    from lerobot.common.control_utils import predict_action
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.device_utils import get_safe_torch_device

    dev = device or _auto_device()
    pol_path = _resolve_policy(policy)
    repo = _resolve_repo(repo_id)
    renames = json.loads(rename_map) if rename_map else {}

    typer.secho(f"Loading dataset {repo} (downloads from the Hub if not cached locally)...", fg="blue")
    dataset = LeRobotDataset(repo, video_backend=_safe_video_backend())
    ep_from = dataset.meta.episodes[episode]["dataset_from_index"]
    ep_to = dataset.meta.episodes[episode]["dataset_to_index"]

    typer.secho(f"Loading policy from {pol_path} on {dev}...", fg="blue")
    cfg = PreTrainedConfig.from_pretrained(pol_path)
    cfg.pretrained_path = pol_path
    cfg.device = dev
    pol = make_policy(cfg, ds_meta=dataset.meta, rename_map=renames)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=pol_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": dev},
            "rename_observations_processor": {"rename_map": renames},
        },
    )
    for p in (pol, pre, post):
        if hasattr(p, "reset"):
            p.reset()

    torch_device = get_safe_torch_device(dev)
    diffs, times = [], []
    for i in range(steps):
        frame = dataset[ep_from + i % max(ep_to - ep_from, 1)]
        # Rebuild the dataset-format observation that record_loop feeds to predict_action.
        obs = {}
        for key in dataset.meta.features:
            if not key.startswith("observation."):
                continue
            t = frame[key]
            if "image" in key:  # (C,H,W) float [0,1] -> (H,W,C) uint8, as a camera would produce
                obs[key] = (t.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
            else:
                obs[key] = t.numpy().astype(np.float32)
        t0 = time.perf_counter()
        action = predict_action(
            observation=obs,
            policy=pol,
            device=torch_device,
            preprocessor=pre,
            postprocessor=post,
            use_amp=pol.config.use_amp,
            task=dataset.meta.tasks.index[0] if len(dataset.meta.tasks) else "",
            robot_type=dataset.meta.robot_type,
        )
        times.append(time.perf_counter() - t0)
        action = action.cpu().numpy() if hasattr(action, "cpu") else np.asarray(action)
        diffs.append(np.abs(action - frame["action"].numpy()).mean())

    typer.secho(
        f"OK: {steps} inference steps on {dev} | "
        f"first {times[0] * 1e3:.0f} ms, avg {np.mean(times[1:]) * 1e3 if len(times) > 1 else times[0] * 1e3:.1f} ms "
        f"(~{1.0 / np.mean(times[1:]) if len(times) > 1 else 1.0 / times[0]:.0f} Hz) | "
        f"mean |action - recorded| = {np.mean(diffs):.2f} deg",
        fg="green",
    )


@app.command()
def upload(
    repo_id: str = typer.Option(
        ..., "--repo-id", help="Local dataset id to upload ('name' → prefixed with your HF user)."
    ),
    private: bool = typer.Option(False, "--private", help="Create the Hub dataset as private."),
    tags: str = typer.Option(None, "--tags", help="Comma-separated tags for the dataset card."),
) -> None:
    """Upload an already-recorded local dataset to the Hugging Face Hub (needs `pixi run hf-login`)."""
    repo = _resolve_repo(repo_id)
    root = _dataset_root(repo)
    if not root.exists():
        raise typer.BadParameter(
            f"No local dataset at {root}. Record it first (without --push), or check --repo-id."
        )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    typer.secho(f"Uploading {repo} from {root} ...", fg="blue")
    try:
        ds = LeRobotDataset(repo, root=root)
        ds.push_to_hub(private=private, tags=[t.strip() for t in tags.split(",")] if tags else None)
    except Exception as exc:  # surface a clean hint instead of a raw traceback
        raise typer.BadParameter(
            f"Upload failed ({type(exc).__name__}: {exc}). Are you logged in? Run: pixi run hf-login"
        ) from None
    typer.secho(f"Uploaded → https://huggingface.co/datasets/{repo}", fg="green")


@app.command("push-policy")
def push_policy(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        help="Checkpoint dir to upload (e.g. outputs/train/<job>/checkpoints/last). "
        "The 'pretrained_model/' sub-dir is used when present.",
    ),
    repo_id: str = typer.Option(
        ..., "--repo-id", help="Hub model repo id ('name' → prefixed with HF user, e.g. 'act_my_task')."
    ),
    private: bool = typer.Option(False, "--private", help="Create the Hub repo as private."),
) -> None:
    """Upload a trained policy checkpoint to the Hugging Face Hub (needs `pixi run hf-login`)."""
    from huggingface_hub import HfApi

    ckpt = Path(checkpoint)
    pretrained = ckpt / "pretrained_model" if (ckpt / "pretrained_model").is_dir() else ckpt
    if not pretrained.is_dir():
        raise typer.BadParameter(
            f"'{checkpoint}' is not a valid checkpoint directory. "
            f"Expected a path like outputs/train/<job>/checkpoints/last "
            f"or outputs/train/<job>/checkpoints/last/pretrained_model."
        )

    repo = _resolve_repo(repo_id, for_creation=True)
    typer.secho(f"Uploading {pretrained} → huggingface.co/{repo} ...", fg="blue")
    try:
        api = HfApi()
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=private)
        commit = api.upload_folder(
            repo_id=repo,
            repo_type="model",
            folder_path=str(pretrained),
            commit_message="Upload policy weights and config",
            allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
        )
        typer.secho(f"Uploaded → {commit.commit_url}", fg="green")
    except Exception as exc:
        raise typer.BadParameter(
            f"Upload failed ({type(exc).__name__}: {exc}). Are you logged in? Run: pixi run hf-login"
        ) from None


if __name__ == "__main__":
    app()
