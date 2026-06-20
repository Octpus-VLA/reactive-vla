# reactive-vla

English | [日本語](README-ja.md)

first octpus vla project repository

📖 **Documentation:** <https://octpus-vla.github.io/reactive-vla/> — step-by-step guides (setup, SmolVLA fine-tuning, editable lerobot, RTC sim rollout). This README is the command/feature reference; the docs site is the narrative walkthrough.

## Features

- **SO-101 hardware CLI** (`cli/so101.py`, exposed as `pixi run <command>`) — register the leader/follower arms once, then calibrate, teleoperate, record/replay/visualize/edit datasets, and push them to the Hub. See the [command table](#so-101-commands-pixi-run-command) below. The arm registration/teleop flow (`set-port` → `setup-motors` → `calibrate` → `teleop`) follows the pattern in [Adwaver4157/lecture_lerobot_teleop](https://github.com/Adwaver4157/lecture_lerobot_teleop).
- **Imitation-learning fine-tuning** — fine-tune `smolvla_base` or `pi0_base` on a SO-101 dataset (or train a policy from scratch), with optional W&B logging and Hugging Face Hub push. See [Trial run: Fine-tuning SmolVLA](#trial-run-fine-tuning-smolvla-on-a-so-101-dataset) and [Fine-tuning with pi0](#fine-tuning-with-pi0) below.
- **HPC batch training** — submit fine-tuning as a PBS job ([`jobs/train_smolvla.pbs`](jobs/train_smolvla.pbs)) instead of running `pixi run train` interactively.
- **MuJoCo simulation** — a bundled SO-ARM100 model + `sim_so101` robot adapter let you exercise the async RTC rollout path without physical hardware. See [MuJoCo simulation](#mujoco-simulation-async-rtc-rollout) below.

### SO-101 commands (`pixi run <command>`)

| Command | Purpose |
|---|---|
| `set-port leader\|follower` | Detect & save the arm's serial port |
| `arms` | Show registered arms/cameras |
| `check leader\|follower` | Per-motor diagnostic on the saved port |
| `set-camera <name> --index N` | Attach (or remove) a camera on the follower |
| `setup-motors leader\|follower` | Assign Feetech motor IDs |
| `calibrate leader\|follower` | Run `lerobot-calibrate` with the saved port/id |
| `teleop` | Drive both saved arms (`lerobot-teleoperate`) |
| `record --task "..." --repo-id name` | Record a teleoperated dataset |
| `replay --repo-id name --episode N` | Replay a recorded episode on the follower |
| `viz --repo-id name --episode N` | Visualize an episode (frames/states/actions) in Rerun |
| `drop --repo-id name --episodes 0,2` | Delete bad episodes from a local dataset |
| `upload --repo-id name` | Push a local dataset to the Hugging Face Hub |
| `train --repo-id name [--policy act \| --policy-path ...]` | Fine-tune or train a policy (see below) |
| `push-policy --checkpoint ... --repo-id name` | Push a trained checkpoint to the Hub |
| `policy-test --policy ... --repo-id ...` | Offline inference smoke test (no robot needed) |
| `eval --policy ... --task "..." --repo-id rollout_name` | Run a trained policy on the follower and record eval episodes |
| `hf-login` / `wandb-login` | One-time login helpers (needed before pushing/logging) |

Run `pixi run <command> --help` for the full flag list. Flags placed after a forwarding command (`teleop`, `record`, `train`, `eval`, `replay`) are passed straight through to the underlying `lerobot-*` CLI.

## Setup

This repository pulls in `lerobot` as a git submodule at `third_party/lerobot` and uses pixi's editable install.

### 1. Fetch the submodule

```bash
git submodule update --init --recursive
```

The submodule is referenced over HTTPS (`https://github.com/Octpus-VLA/lerobot.git`), so no SSH key setup is required.

### 2. Set up the environment

```bash
pixi install
```

- [pixi.toml](pixi.toml) registers `osx-arm64` / `linux-64` / `linux-aarch64` under `platforms`. If your machine uses a different architecture, add it with `pixi workspace platform add <platform>`.
- `ffmpeg` is included as a conda dependency, which is required for video decoding (`lerobot[dataset]` / torchcodec).

### 3. Lint / Format

```bash
pixi run lint   # ruff check
pixi run fmt    # ruff format
pixi run fix    # check --fix + format
```

For detailed configuration and how to add custom policies, see [docs/lerobot-editable-setup.md](docs/lerobot-editable-setup.md).

## Trial run: Fine-tuning SmolVLA on a SO-101 dataset

This is an example of fine-tuning the pretrained [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) model (450M) on the SO-101 pick & place demo dataset [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace). It's meant as a smoke test, so no physical robot or camera is required.

### 1. (On HPC) Move to a GPU node

On HPC systems, move from a CPU node (login/interactive node) to a GPU node before running this. As long as `/work` is on a shared filesystem such as Lustre, the `.pixi` environment and the submodule can be used as-is.

```bash
qsub -I -q interact-g -W group_list=gw13 -l select=1 -l walltime=00:15:00
```

Once on the GPU node, `cd` back into the project directory.

```bash
cd /work/gw13/$USER/$PROJECT
```

### 2. Run fine-tuning

`pixi run train` accepts either `--policy-path` (fine-tune from a pretrained model) or `--policy` (train from scratch); the two are mutually exclusive.

```bash
pixi run train \
  --policy-path lerobot/smolvla_base \
  --repo-id lerobot/svla_so101_pickplace \
  --batch-size 64 \
  --steps 20000 \
  --job-name smolvla_so101_pickplace \
  --device cuda \
  -- --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}'
```

- `lerobot/svla_so101_pickplace`'s camera names (`observation.images.up` / `observation.images.side`) differ from what `smolvla_base` expects (3 cameras: `camera1`–`camera3`), so `--rename_map` remaps them (`camera3` is left unused). Arguments after `--` are forwarded verbatim to `lerobot-train`.
- A GPU is recommended (about 4 hours for 20k steps on an A100). For a quick smoke test, use `--steps 2000`.
- `--device` accepts `cuda` / `mps` / `cpu`. Omit it for auto-detection.
- Training output is written to `outputs/` (gitignored).
- On a PBS-scheduled HPC, submit [`jobs/train_smolvla.pbs`](jobs/train_smolvla.pbs) instead of running the command above interactively: `qsub jobs/train_smolvla.pbs` (it wraps the same `pixi run train` invocation; override knobs like `STEPS`/`BATCH_SIZE`/`RESUME` with `qsub -v`, see comments in the script).

#### Logging to W&B

```bash
pixi run wandb-login   # first time only (skipped if already logged in)

pixi run train \
  --policy-path lerobot/smolvla_base \
  --repo-id lerobot/svla_so101_pickplace \
  --batch-size 64 \
  --steps 20000 \
  --job-name smolvla_so101_pickplace \
  --device cuda \
  --wandb \
  --wandb-project <project> \
  --wandb-entity <team> \
  -- --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}'
```

- `--wandb-project` defaults to `lerobot` if omitted.
- `--wandb-entity` defaults to your personal account. Set it to your W&B team name (the `<team>` part of `wandb.ai/<team>`) to log under a team.
- Standard metrics logged: `train/loss`, `train/lr`, `train/grad_norm`.
- The default `log_freq` is 200 (one log entry per 200 steps). Override it with `-- --log_freq=50` if you need finer resolution.

#### Pushing a trained policy to Hugging Face Hub

**Push during training** (just add `--push-repo-id`):

```bash
pixi run hf-login   # first time only (skipped if already logged in)

pixi run train \
  --policy-path lerobot/smolvla_base \
  --repo-id lerobot/svla_so101_pickplace \
  --batch-size 64 \
  --steps 20000 \
  --job-name smolvla_so101_pickplace \
  --device cuda \
  --push-repo-id smolvla_so101_pickplace \
  -- --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}'
```

A bare name (no slash) is automatically prefixed with your HF username (e.g. `smolvla_so101_pickplace` → `<HF-user>/smolvla_so101_pickplace`).

**Push after training**:

```bash
pixi run push-policy \
  --checkpoint outputs/train/smolvla_base/svla_so101_pickplace/<timestamp>/checkpoints/last \
  --repo-id smolvla_so101_pickplace
```

Pass the checkpoint directory; if a `pretrained_model/` subdirectory exists it is detected automatically. Add `--private` to create a private Hub repo.

> `output_dir` is always `outputs/train/<policy>/<dataset>/<timestamp>` (`MMDD_HHMM`). `--job-name` only sets the W&B display name and is never folded into the directory, so rerunning with the same `--job-name` won't collide with an existing directory. Check the training log (`--output_dir=...`) for the actual path.

### 3. Verify with offline inference (no robot needed)

```bash
pixi run policy-test \
  --policy outputs/train/smolvla_base/svla_so101_pickplace/<timestamp>/checkpoints/last/pretrained_model \
  --repo-id lerobot/svla_so101_pickplace
```

This feeds recorded dataset frames into the fine-tuned policy and reports inference latency and the deviation from the recorded actions.

Reference: [SmolVLA fine-tuning guide](https://huggingface.co/docs/lerobot/en/smolvla)

## Fine-tuning with pi0

[`lerobot/pi0_base`](https://huggingface.co/lerobot/pi0_base) is a PaLiGemma-based ~3B parameter model. Unlike `smolvla_base`, it dynamically reads camera names from the dataset features, so `--rename_map` is not needed.

> **Known issue**: Fine-tuning `lerobot/pi0_base` is currently known to hang at `Loading model from: lerobot/pi0_base` → `model.safetensors`. We are investigating this issue; running pi0 is not recommended at this time.

### Run fine-tuning

```bash
pixi run train \
  --policy-path lerobot/pi0_base \
  --repo-id lerobot/svla_so101_pickplace \
  --batch-size 4 \
  --steps 200 \
  --device cuda
```

- Use a small `--batch-size` (4–8) due to the large model size. Gradient checkpointing may be needed even on an A100 80GB — add `-- --policy.gradient_checkpointing=true` if required.
- `--rename_map` is not needed (pi0 uses the dataset's camera names as-is).

### Verify with offline inference (no robot needed)

```bash
pixi run policy-test \
  --policy outputs/train/pi0_base/svla_so101_pickplace/<timestamp>/checkpoints/last/pretrained_model \
  --repo-id lerobot/svla_so101_pickplace
```

## MuJoCo simulation: async RTC rollout

A SO-ARM100 MuJoCo model is bundled at `assets/so_arm100/` (no clone needed), with a `sim_so101` robot adapter (in the `lerobot` fork) so the hardware-only `lerobot-rollout --inference.type=rtc` (async Real-Time Chunking) path can be exercised end-to-end without a physical robot — including offscreen rendering and episode recording. See [docs/rtc-sim-rollout.md](docs/rtc-sim-rollout.md) for the full setup/train/rollout walkthrough and RTC parameter reference.

## Roadmap: extending to conveyor-belt grasping (design phase)

> This section describes planned, not-yet-implemented work. For shipped functionality, see [Features](#features) above.

### Target task

- Grasp an object moving on a belt conveyor and place it in a box.
- The belt speed should vary across multiple settings.
- A new detector, built from image input, flags when an object is approaching; on detection it requests the VLA to regenerate its Action Chunk, reacting faster than the default queue-size-based replanning.
- Both the VLA (assumed `smolvla_base`) and the detector need training.
- The detector's implementation is undecided; the goal is a pluggable design so any detector implementation can be swapped in.

### Current pipeline (recap)

1. SO-101 hardware setup → `set-port` → `setup-motors` → `calibrate` → `set-camera` (one-time).
2. `record` to teleoperate and record a dataset (currently static pick & place only).
3. `train` (or a PBS job) to fine-tune `smolvla_base`.
4. `policy-test` for an offline inference sanity check.
5. `eval` to run the policy on the follower — internally this is `lerobot-rollout --strategy.type=episodic --inference.type=sync`, i.e. **synchronous** inference, not RTC. Eval datasets must use the `rollout_` repo-id prefix (e.g. `rollout_test`), not `eval_`.
6. Async RTC rollout (`lerobot-rollout --inference.type=rtc`) is currently **MuJoCo-sim only** ([docs/rtc-sim-rollout.md](docs/rtc-sim-rollout.md)). Switching to `--robot.type=so101_follower` should work in principle (same `Robot` abstraction), but it has never been run on real hardware, and `cli/so101.py` has no wrapper for it.

### Step-by-step for the next real-hardware session

1. `pixi run set-port leader` / `pixi run set-port follower` (one-time)
2. `pixi run setup-motors leader` / `pixi run setup-motors follower` (one-time)
3. `pixi run calibrate leader` / `pixi run calibrate follower`
4. `pixi run set-camera <name> --index N` (attach a camera to the follower)
5. `pixi run check leader` / `pixi run check follower` (optional pre-flight diagnostic)
6. `pixi run teleop` to verify motion
7. `pixi run record --task "pick the object from the belt and place it in the box" --repo-id <name> --episodes <N>` to collect data
8. `pixi run train --policy-path lerobot/smolvla_base --repo-id <name> ...` to fine-tune (use [`jobs/train_smolvla.pbs`](jobs/train_smolvla.pbs) for long runs)
9. `pixi run policy-test --policy <checkpoint> --repo-id <name>` for an offline check
10. `pixi run eval --policy <checkpoint> --task "..." --repo-id rollout_<name>` for an on-robot, synchronous evaluation
11. To try RTC on real hardware, there's no wrapper yet, so `lerobot-rollout` must be assembled by hand (`--robot.type=so101_follower --robot.port=... --robot.id=... --robot.cameras='{...}'`, etc. — equivalent to swapping `--robot.type` in the sim commands in [docs/rtc-sim-rollout.md](docs/rtc-sim-rollout.md)).

### What's missing

1. **The conveyor itself**: no variable-speed belt conveyor, and no way to record/reproduce its speed setting.
2. **A task dataset**: the existing `lerobot/svla_so101_pickplace` is static pick & place. A new dataset covering pickup-from-belt → place-in-box needs to be collected.
3. **No detector implementation exists yet**: inputs (image only, or also joint state?) and outputs (approach flag / distance / bbox) are undecided. A pluggable design needs an abstract detector interface (a swappable protocol) added on the `lerobot` fork side.
4. **No event-driven trigger path from detector to RTC**: the current RTC engine (`rollout/inference/rtc.py`) only replans based on `queue_threshold` (remaining queue size). There's no hook yet for "force an immediate replan the moment the detector fires" (e.g. a `force_replan()` method).
5. **No training data for the detector**: no pipeline exists for collecting/labeling "object is approaching" data.
6. **No evaluation protocol for variable belt speed**: no tooling to compare success rate across different belt speeds (the existing `eval` only records episodes; it doesn't auto-score success/failure).
7. **RTC itself is unverified on real hardware**: it has only been exercised in simulation, never run against `so101_follower`.

## Troubleshooting

Use `qstat` to check running jobs.
