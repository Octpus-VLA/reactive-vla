# お試し学習: SmolVLA を SO-101 データセットでファインチューニング

事前学習済みモデル [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)（450M）を、SO-101 の pick & place デモデータセット [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) でファインチューニングする例です。動作確認用なので自前のロボット・カメラは不要です。

## 1. (HPC 利用時) GPU ノードへの移動

HPC 環境では CPU ノード（ログイン / インタラクティブノード）から GPU ノードに移動してから実行してください。`/work` 配下が Lustre などの共有ファイルシステムであれば、`.pixi` 環境や submodule はそのまま使えます。

```bash
qsub -I -q interact-g -W group_list=gw13 -l select=1 -l walltime=00:15:00
```

GPU ノードに入ったら、再度プロジェクトディレクトリに `cd` してから以下を実行します。

## 2. ファインチューニング実行

`pixi run train`（`cli/so101.py train`）は `--policy.type`（新規学習）専用で、事前学習済みモデルからの再開を表す `--policy.path` には未対応のため、`lerobot-train` を直接実行します。

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

- `lerobot/svla_so101_pickplace` のカメラ名（`observation.images.up` / `observation.images.side`）は `smolvla_base` が期待する名前（`camera1`〜`camera3` の 3 カメラ）と異なるため `--rename_map` でマッピングします（`camera3` は未使用のまま）。
- GPU 推奨（A100 で 20k ステップ約 4 時間）。動作確認だけしたい場合は `--steps=200` 程度に減らすと短時間で完走します。
- `--policy.device` は `cuda` / `mps` / `cpu` から実機に合わせて指定。
- 学習結果は `outputs/`（gitignore 済み）に出力されます。

## 3. オフライン推論で確認（ロボット不要）

```bash
pixi run policy-test \
  --policy outputs/train/smolvla_so101_pickplace/checkpoints/last/pretrained_model \
  --repo-id lerobot/svla_so101_pickplace
```

データセットに記録済みのフレームを入力し、ファインチューニング済みポリシーの推論レイテンシと、記録された実際の行動とのズレを確認できます。

!!! note "参考"
    [SmolVLA ファインチューニングガイド](https://huggingface.co/docs/lerobot/en/smolvla)
