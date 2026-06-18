# reactive-vla

English | [日本語](README-ja.md)

first octpus vla project repository

📖 **Documentation:** <https://octpus-vla.github.io/reactive-vla/>

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

## Troubleshooting

Use `qstat` to check running jobs.
