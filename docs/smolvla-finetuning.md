# お試し学習: SmolVLA を SO-101 データセットでファインチューニング

事前学習済みモデル [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)（450M）を、SO-101 の pick & place デモデータセット [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) でファインチューニングする例です。動作確認用なので自前のロボット・カメラは不要です。

## 1. (HPC 利用時) GPU ノードへの移動

HPC 環境では CPU ノード（ログイン / インタラクティブノード）から GPU ノードに移動してから実行してください。`/work` 配下が Lustre などの共有ファイルシステムであれば、`.pixi` 環境や submodule はそのまま使えます。

```bash
qsub -I -q interact-g -W group_list=gw13 -l select=1 -l walltime=00:15:00
```

GPU ノードに入ったら、再度プロジェクトディレクトリに `cd` してから以下を実行します。

## 2. ファインチューニング実行

`pixi run train` は `--policy-path`（事前学習モデルからのファインチューニング）と `--policy`（スクラッチ学習）の二者択一で動作します。

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

- `lerobot/svla_so101_pickplace` のカメラ名（`observation.images.up` / `observation.images.side`）は `smolvla_base` が期待する名前（`camera1`〜`camera3` の 3 カメラ）と異なるため `--rename_map` でマッピングします（`camera3` は未使用のまま）。`--` 以降の引数はそのまま `lerobot-train` に転送されます。
- GPU 推奨（A100 で 20k ステップ約 4 時間）。動作確認だけしたい場合は `--steps 2000` 程度に減らすと短時間で完走します。
- `--device` は `cuda` / `mps` / `cpu` から実機に合わせて指定。省略すると自動検出されます。
- 学習結果は `outputs/`（gitignore 済み）に出力されます。

!!! note "出力先について"
    `output_dir` は常に `outputs/train/<policy>/<dataset>/<タイムスタンプ>`（`MMDD_HHMM`）です。`--job-name` は W&B 上の表示名だけに使われ、ディレクトリ名には含まれないので、同じ `--job-name` で再実行しても既存ディレクトリと衝突しません。実際のパスは学習実行時のログ（`--output_dir=...`）で確認してください。

W&B ロギングや Hugging Face Hub への push を行う場合は、[README](https://github.com/Octpus-VLA/reactive-vla#trial-run-fine-tuning-smolvla-on-a-so-101-dataset) を参照してください。

## 3. オフライン推論で確認（ロボット不要）

```bash
pixi run policy-test \
  --policy outputs/train/smolvla_base/svla_so101_pickplace/<タイムスタンプ>/checkpoints/last/pretrained_model \
  --repo-id lerobot/svla_so101_pickplace
```

データセットに記録済みのフレームを入力し、ファインチューニング済みポリシーの推論レイテンシと、記録された実際の行動とのズレを確認できます。

!!! note "参考"
    [SmolVLA ファインチューニングガイド](https://huggingface.co/docs/lerobot/en/smolvla)
