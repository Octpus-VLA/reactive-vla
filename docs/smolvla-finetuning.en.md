# Trial run: Fine-tuning SmolVLA on a SO-101 dataset

This is an example of fine-tuning the pretrained [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) model (450M) on the SO-101 pick & place demo dataset [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace). It's meant as a smoke test, so no physical robot or camera is required.

## 1. (On HPC) Move to a GPU node

On HPC systems, move from a CPU node (login/interactive node) to a GPU node before running this. As long as `/work` is on a shared filesystem such as Lustre, the `.pixi` environment and the submodule can be used as-is.

```bash
qsub -I -q interact-g -W group_list=gw13 -l select=1 -l walltime=00:15:00
```

Once on the GPU node, `cd` back into the project directory and run the following.

## 2. Run fine-tuning

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

- `lerobot/svla_so101_pickplace`'s camera names (`observation.images.up` / `observation.images.side`) differ from what `smolvla_base` expects (3 cameras: `camera1`-`camera3`), so `--rename_map` remaps them (`camera3` is left unused). Arguments after `--` are forwarded verbatim to `lerobot-train`.
- A GPU is recommended (about 4 hours for 20k steps on an A100). For a quick smoke test, use `--steps 2000`.
- `--device` accepts `cuda` / `mps` / `cpu`. Omit it for auto-detection.
- Training output is written to `outputs/` (gitignored).

!!! note "About the output path"
    `output_dir` is always `outputs/train/<policy>/<dataset>/<timestamp>` (`MMDD_HHMM`). `--job-name` only sets the W&B display name and is never folded into the directory, so rerunning with the same `--job-name` won't collide with an existing directory. Check the training log (`--output_dir=...`) for the actual path.

For W&B logging or pushing to the Hugging Face Hub, see the [README](https://github.com/Octpus-VLA/reactive-vla#trial-run-fine-tuning-smolvla-on-a-so-101-dataset).

## 3. Verify with offline inference (no robot needed)

```bash
pixi run policy-test \
  --policy outputs/train/smolvla_base/svla_so101_pickplace/<timestamp>/checkpoints/last/pretrained_model \
  --repo-id lerobot/svla_so101_pickplace
```

This feeds recorded dataset frames into the fine-tuned policy and reports inference latency and the deviation from the recorded actions.

!!! note "Reference"
    [SmolVLA fine-tuning guide](https://huggingface.co/docs/lerobot/en/smolvla)
