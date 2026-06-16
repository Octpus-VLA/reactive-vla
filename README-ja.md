# reactive-vla

[English](README.md) | 日本語

first octpus vla project repository

## セットアップ

このリポジトリは `lerobot` を `third_party/lerobot` に git submodule として取り込み、pixi の editable install で利用します。

### 1. submodule の取得

```bash
git submodule update --init --recursive
```

submodule は HTTPS (`https://github.com/Octpus-VLA/lerobot.git`) で参照しているため、SSH鍵の設定は不要です。

### 2. 環境構築

```bash
pixi install
```

- [pixi.toml](pixi.toml) の `platforms` には `osx-arm64` / `linux-64` / `linux-aarch64` を登録しています。利用するマシンのアーキテクチャがこれら以外の場合は `pixi workspace platform add <platform>` で追加してください。
- 動画デコード（`lerobot[dataset]` / torchcodec）に必要な `ffmpeg` も conda 依存として含めています。

### 3. Lint / Format

```bash
pixi run lint   # ruff check
pixi run fmt    # ruff format
pixi run fix    # check --fix + format
```

詳細な構成・カスタムポリシー追加手順は [docs/lerobot-editable-setup.md](docs/lerobot-editable-setup.md) を参照してください。

## お試し学習: SmolVLA を SO-101 データセットでファインチューニング

事前学習済みモデル [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)（450M）を、SO-101のpick & placeデモデータセット [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) でファインチューニングする例です。動作確認用なので自前のロボット・カメラは不要です。

### 1. (HPC利用時) GPUノードへの移動

HPC環境ではCPUノード（ログイン/インタラクティブノード）からGPUノードに移動してから実行してください。`/work` 配下がLustreなどの共有ファイルシステムであれば、`.pixi` 環境やsubmoduleはそのまま使えます。

```bash
qsub -I -q interact-g -W group_list=gw13 -l select=1 -l walltime=00:15:00
```

GPUノードに入ったら、再度プロジェクトディレクトリに `cd` します。

```bash
cd /work/gw13/$USER/$PROJECT
```

### 2. ファインチューニング実行

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

- `lerobot/svla_so101_pickplace` のカメラ名（`observation.images.up` / `observation.images.side`）は `smolvla_base` が期待する名前（`camera1`〜`camera3` の3カメラ）と異なるため `--rename_map` でマッピングします（`camera3` は未使用のまま）。`--` 以降の引数はそのまま `lerobot-train` に転送されます。
- GPU推奨（A100で20kステップ約4時間）。動作確認だけしたい場合は `--steps 2000` 程度に減らすと短時間で完走します。
- `--device` は `cuda` / `mps` / `cpu` から実機に合わせて指定。省略すると自動検出されます。
- 学習結果は `outputs/`（gitignore済み）に出力されます。

#### W&B でトレーニングを記録する

```bash
pixi run wandb-login   # 初回のみ（既にログイン済みならスキップ）

pixi run train \
  --policy-path lerobot/smolvla_base \
  --repo-id lerobot/svla_so101_pickplace \
  --batch-size 64 \
  --steps 20000 \
  --job-name smolvla_so101_pickplace \
  --device cuda \
  --wandb \
  --wandb-project <プロジェクト名> \
  --wandb-entity <チーム名> \
  -- --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}'
```

- `--wandb-project` を省略するとプロジェクト名は `lerobot` になります。
- `--wandb-entity` を省略すると個人アカウントに記録されます。チームに記録する場合は W&B のチーム名（URL の `wandb.ai/<チーム名>` 部分）を指定してください。
- W&B には `train/loss`・`train/lr`・`train/grad_norm` などが標準で記録されます。
- lerobot のデフォルトは `log_freq=200`（200ステップごとに1回ログ）です。ログ頻度を変えたい場合は `-- --log_freq=50` のように転送引数で上書きできます。

#### 学習済みポリシーを Hugging Face Hub にアップロードする

**学習中にそのままプッシュ**（`--push-repo-id` を追加するだけ）:

```bash
pixi run hf-login   # 初回のみ（既にログイン済みならスキップ）

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

`--push-repo-id` に bare name（スラッシュなし）を渡すと、ログイン中の HF ユーザー名が自動でプレフィックスされます（例: `smolvla_so101_pickplace` → `<HFユーザー>/smolvla_so101_pickplace`）。

**学習後に手動でプッシュ**:

```bash
pixi run push-policy \
  --checkpoint outputs/train/smolvla_so101_pickplace/checkpoints/last \
  --repo-id smolvla_so101_pickplace
```

`--checkpoint` にはチェックポイントディレクトリを渡します（`pretrained_model/` サブディレクトリがあれば自動検出します）。`--private` を付けるとプライベートリポジトリとして作成されます。

### 3. オフライン推論で確認（ロボット不要）

```bash
pixi run policy-test \
  --policy outputs/train/smolvla_so101_pickplace/checkpoints/last/pretrained_model \
  --repo-id lerobot/svla_so101_pickplace
```

データセットに記録済みのフレームを入力し、ファインチューニング済みポリシーの推論レイテンシと、記録された実際の行動とのズレを確認できます。

参考: [SmolVLAファインチューニングガイド](https://huggingface.co/docs/lerobot/en/smolvla)

## pi0 でのファインチューニング

[`lerobot/pi0_base`](https://huggingface.co/lerobot/pi0_base) は PaLiGemma ベースの〜3B パラメータモデルです。smolvla_base と異なり、カメラ名をデータセットの特徴量から動的に受け取るため `--rename_map` は不要です。

### ファインチューニング実行

```bash
pixi run train \
  --policy-path lerobot/pi0_base \
  --repo-id lerobot/svla_so101_pickplace \
  --batch-size 4 \
  --steps 20000 \
  --device cuda
```

- モデルが大きいため `--batch-size` は小さく（4〜8 程度）。A100 80GB でも勾配チェックポイントが必要になる場合があります。その場合は `-- --policy.gradient_checkpointing=true` を追加してください。
- `--rename_map` は不要です（pi0 はデータセットのカメラ名をそのまま使います）。

### オフライン推論で確認（ロボット不要）

```bash
pixi run policy-test \
  --policy outputs/train/pi0_base/svla_so101_pickplace/<タイムスタンプ>/checkpoints/last/pretrained_model \
  --repo-id lerobot/svla_so101_pickplace
```