# Trial run: Fine-tuning SmolVLA on a SO-101 dataset

This is an example of fine-tuning the pretrained [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) model (450M) on the SO-101 pick & place demo dataset [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace). It's meant as a smoke test, so no physical robot or camera is required.

## 1. (On HPC) Move to a GPU node

On HPC systems, move from a CPU node (login/interactive node) to a GPU node before running this. As long as `/work` is on a shared filesystem such as Lustre, the `.pixi` environment and the submodule can be used as-is.

```bash
qsub -I -q interact-g -W group_list=gw13 -l select=1 -l walltime=00:15:00
```

Once on the GPU node, `cd` back into the project directory and run the following.

## 2. Run fine-tuning

`pixi run train` (`cli/so101.py train`) only supports `--policy.type` (training from scratch) and doesn't support `--policy.path` (resuming from a pretrained model), so run `lerobot-train` directly.

```bash
pixi run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/smolvla_so101_pickplace \
  --job_name=smolvla_so101_pickplace \
  --policy.device=cuda \
  --wandb.enable=false
```

- `lerobot/svla_so101_pickplace`'s camera names (`observation.images.up` / `observation.images.side`) differ from what `smolvla_base` expects (3 cameras: `camera1`-`camera3`), so `--rename_map` maps them accordingly (`camera3` is left unused).
- A GPU is recommended (about 4 hours for 20k steps on an A100). For a quick smoke test, reduce `--steps` to something like `200`.
- Set `--policy.device` to `cuda` / `mps` / `cpu` depending on your hardware.
- Training output is written to `outputs/` (gitignored).

## 3. Verify with offline inference (no robot needed)

```bash
pixi run policy-test \
  --policy outputs/train/smolvla_so101_pickplace/checkpoints/last/pretrained_model \
  --repo-id lerobot/svla_so101_pickplace
```

This feeds recorded dataset frames into the fine-tuned policy and reports inference latency and the deviation from the recorded actions.

!!! note "Reference"
    [SmolVLA fine-tuning guide](https://huggingface.co/docs/lerobot/en/smolvla)
