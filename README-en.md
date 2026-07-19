# reactive-vla

English | [日本語](README.md)

## Overview

reactive-vla is a research project on reactive VLA (Vision-Language-Action) control for dynamic pick tasks, such as grasping an object moving on a conveyor belt and placing it into a box. Built on `lerobot`, it provides a CLI/workflow for recording, fine-tuning, and evaluating SmolVLA / pi0 action-chunking imitation learning, on both real hardware (a SO-101 robot arm) and MuJoCo simulation.

Its reactive control escalates in stages: a baseline replan triggered by the existing action-queue threshold, an event-driven early replan from a camera supervisor, and a predictive replan that derives a dynamic effective horizon from the cube's position, velocity, and predicted grasp timing. See [CLAUDE.md](https://github.com/Octpus-VLA/reactive-vla/blob/main/CLAUDE.md) for how this project refers to these stages (Tier 1-3).

📖 **Documentation:** <https://octpus-vla.github.io/reactive-vla/> — step-by-step guides (setup, SmolVLA fine-tuning, editable lerobot, RTC sim rollout). This README is the project overview; detailed usage lives in each directory's own docs.

## Simulation environment

A bundled MuJoCo model of the SO-101 (the same CAD-derived model as the real robot, DeepMind Menagerie's `robotstudio_so101`) is set up with a conveyor belt, cube, and drop-off box for the dynamic pick-and-place scene. You can exercise policy evaluation and RTC async rollout without physical hardware.

| External camera view (`overview`, for recording) | Wrist camera view (`wrist_cam`, policy input) |
|---|---|
| ![sim-eval scene: SO-101 arm, green conveyor belt with a red cube, and a white drop-off box](docs/sim-eval-scene.png) | ![cube and belt from wrist_cam](docs/sim-eval-wrist.png) |

See [docs/rtc-sim-rollout.md](docs/rtc-sim-rollout.md) for details.

## Features

- **SO-101 hardware CLI** (`cli/so101.py`, exposed as `pixi run <command>`) — register the leader/follower arms once, then calibrate, teleoperate, record/replay/visualize/edit datasets, and push them to the Hub. See the [command table in cli/README.md](cli/README-en.md#so-101-commands-pixi-run-command). The arm registration/teleop flow (`set-port` → `setup-motors` → `calibrate` → `teleop`) follows the pattern in [Adwaver4157/lecture_lerobot_teleop](https://github.com/Adwaver4157/lecture_lerobot_teleop).
- **Imitation-learning fine-tuning** — fine-tune `smolvla_base` or `pi0_base` on a SO-101 dataset (or train a policy from scratch), with optional W&B logging and Hugging Face Hub push. See [Fine-tuning in cli/README-en.md](cli/README-en.md#fine-tuning).
- **HPC batch training** — submit fine-tuning as a PBS job instead of running `pixi run train` interactively. The PBS scripts themselves aren't included in this repo (they bake in site-specific queue/`group_list` settings) — see the template in [cli/README-en.md](cli/README-en.md#fine-tuning) and drop your own under `jobs/` (gitignored).
- **MuJoCo simulation** — a bundled SO-101 model + `sim_so101` robot adapter let you exercise the async RTC rollout path without physical hardware. See [docs/rtc-sim-rollout.md](docs/rtc-sim-rollout.md).
- **Sim success-rate evaluation** (`pixi run sim-eval`) — run a trained policy in the MuJoCo sim and score task success rate / success step (a Lift-style criterion: did the cube get picked up). Add `--repo-id rollout_<name>` to also record video/dataset. See [Inference in cli/README-en.md](cli/README-en.md#inference).
- **Sim demo collection** (`pixi run sim-collect`) — a scripted IK expert (driven by privileged sim state) performs pick-and-place in the MuJoCo sim and writes the (observation, action) pairs to a `LeRobotDataset` in the same schema `record` produces. A policy trained on real-hardware images is out of distribution on the sim's rendered observations (the real→sim visual gap); fine-tuning on this sim-rendered data is how you close it. See [docs/sim-scripted-collect.md](docs/sim-scripted-collect.md).

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

For registering and calibrating the SO-101 arms themselves, see [cli/README-en.md](cli/README-en.md#registering-the-so-101-arms).

## Further documentation

- **Full CLI command reference** (detailed flags for recording/training/inference, PBS job template) → [cli/README-en.md](cli/README-en.md)
- **Step-by-step guides** (setup, fine-tuning, RTC sim rollout, latency experiments, and more) → [documentation site](https://octpus-vla.github.io/reactive-vla/)
- **SO-101 robot & simulation assets** (MJCF model, photo, license) → [assets/so101/README.md](assets/so101/README.md)

## License

This repository is released under the [Apache License 2.0](LICENSE). The SO-101 robot description (MJCF) under `assets/so101` is a separate distribution (also Apache License 2.0) — see [assets/so101/LICENSE](assets/so101/LICENSE) for details.
