# reactive-vla

[English](README.md) | 日本語

first octpus vla project repository

📖 **ドキュメント:** <https://octpus-vla.github.io/reactive-vla/> — セットアップ・SmolVLAファインチューニング・lerobot editable構成・RTC simロールアウトの手順ガイドはこちら。この README はコマンド/機能のリファレンス、ドキュメントサイトは読み物形式の手順ガイドという役割分担です。

## 機能一覧

- **SO-101 実機操作 CLI**（`cli/so101.py`、`pixi run <command>` として公開）— leader/follower アームを一度登録すれば、以降はキャリブレーション・テレオペ・データセットの記録/再生/可視化/編集・Hubへのアップロードまで行えます。詳細は下記の[コマンド一覧](#so-101-コマンド-pixi-run-command)を参照。アーム登録・テレオペの流れ（`set-port` → `setup-motors` → `calibrate` → `teleop`）は [Adwaver4157/lecture_lerobot_teleop](https://github.com/Adwaver4157/lecture_lerobot_teleop) を参考にしています。
- **模倣学習ファインチューニング** — SO-101 データセットで `smolvla_base` / `pi0_base` をファインチューニング（またはスクラッチ学習）。W&Bロギング・Hugging Face Hubへのpushにも対応。詳細は下記の[お試し学習: SmolVLA](#お試し学習-smolvla-を-so-101-データセットでファインチューニング)・[pi0 でのファインチューニング](#pi0-でのファインチューニング)を参照。
- **HPCバッチ学習** — `pixi run train` をインタラクティブに実行する代わりに、PBSジョブ（[`jobs/train_smolvla.pbs`](jobs/train_smolvla.pbs)）として投入できます。
- **MuJoCoシミュレーション** — 同梱の SO-ARM100 モデルと `sim_so101` ロボットアダプタにより、実機無しで RTC 非同期ロールアウト経路を検証できます。詳細は下記の[MuJoCo シミュレーション](#mujoco-シミュレーション-rtc-非同期ロールアウト)を参照。

### SO-101 コマンド （`pixi run <command>`）

| コマンド | 用途 |
|---|---|
| `set-port leader\|follower` | アームのシリアルポートを検出して保存 |
| `arms` | 登録済みのアーム/カメラを表示 |
| `check leader\|follower` | 保存済みポートでのモーター単位の診断 |
| `set-camera <name> --index N` | follower にカメラを割り当て（削除も可） |
| `setup-motors leader\|follower` | Feetech モーターIDを割り当て |
| `calibrate leader\|follower` | 保存済みポート/idで `lerobot-calibrate` を実行 |
| `teleop` | 保存済みの両アームでテレオペ（`lerobot-teleoperate`） |
| `record --task "..." --repo-id name` | テレオペしながらデータセットを記録 |
| `replay --repo-id name --episode N` | 記録済みエピソードを follower で再生 |
| `viz --repo-id name --episode N` | エピソード（フレーム/状態/行動）を Rerun で可視化 |
| `drop --repo-id name --episodes 0,2` | ローカルデータセットから不良エピソードを削除 |
| `upload --repo-id name` | ローカルデータセットを Hugging Face Hub にアップロード |
| `train --repo-id name [--policy act \| --policy-path ...]` | ポリシーをファインチューニング/学習（詳細は下記） |
| `push-policy --checkpoint ... --repo-id name` | 学習済みチェックポイントを Hub にアップロード |
| `policy-test --policy ... --repo-id ...` | オフライン推論の動作確認（ロボット不要） |
| `eval --policy ... --task "..." --repo-id rollout_name` | 学習済みポリシーを follower で実行し評価エピソードを記録 |
| `hf-login` / `wandb-login` | push/ロギング前の初回ログイン |

各コマンドの全フラグは `pixi run <command> --help` で確認できます。転送系コマンド（`teleop`・`record`・`train`・`eval`・`replay`）の後に置いた引数は、対応する `lerobot-*` CLI にそのまま渡されます。

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
- PBSスケジューラのHPCでは、上記をインタラクティブに実行する代わりに [`jobs/train_smolvla.pbs`](jobs/train_smolvla.pbs) を投入できます: `qsub jobs/train_smolvla.pbs`（内部では同じ `pixi run train` を呼んでいます。`STEPS`/`BATCH_SIZE`/`RESUME` などは `qsub -v` で上書き可能。詳細はスクリプト内のコメント参照）。

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
  --checkpoint outputs/train/smolvla_base/svla_so101_pickplace/<タイムスタンプ>/checkpoints/last \
  --repo-id smolvla_so101_pickplace
```

`--checkpoint` にはチェックポイントディレクトリを渡します（`pretrained_model/` サブディレクトリがあれば自動検出します）。`--private` を付けるとプライベートリポジトリとして作成されます。

> `output_dir` は常に `outputs/train/<policy>/<dataset>/<タイムスタンプ>`（`MMDD_HHMM`）です。`--job-name` は W&B 上の表示名だけに使われ、ディレクトリ名には含まれないので、同じ `--job-name` で再実行しても既存ディレクトリと衝突しません。実際のパスは学習実行時のログ（`--output_dir=...`）で確認してください。

### 3. オフライン推論で確認（ロボット不要）

```bash
pixi run policy-test \
  --policy outputs/train/smolvla_base/svla_so101_pickplace/<タイムスタンプ>/checkpoints/last/pretrained_model \
  --repo-id lerobot/svla_so101_pickplace
```

データセットに記録済みのフレームを入力し、ファインチューニング済みポリシーの推論レイテンシと、記録された実際の行動とのズレを確認できます。

参考: [SmolVLAファインチューニングガイド](https://huggingface.co/docs/lerobot/en/smolvla)

## pi0 でのファインチューニング

[`lerobot/pi0_base`](https://huggingface.co/lerobot/pi0_base) は PaLiGemma ベースの〜3B パラメータモデルです。smolvla_base と異なり、カメラ名をデータセットの特徴量から動的に受け取るため `--rename_map` は不要です。

> **既知の問題**: 現在、`lerobot/pi0_base` のファインチューニングは `Loading model from: lerobot/pi0_base` → `model.safetensors` のダウンロード中に処理が止まる問題が確認されています。調査中のため、現時点では pi0 の実行は推奨しません。

### ファインチューニング実行

```bash
pixi run train \
  --policy-path lerobot/pi0_base \
  --repo-id lerobot/svla_so101_pickplace \
  --batch-size 4 \
  --steps 200 \
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

## MuJoCo シミュレーション: RTC 非同期ロールアウト

SO-ARM100 の MuJoCo モデルを `assets/so_arm100/` に同梱済み（clone不要）で、`sim_so101` ロボットアダプタ（`lerobot` フォーク側）と組み合わせることで、実機前提の `lerobot-rollout --inference.type=rtc`（非同期 Real-Time Chunking）経路をハードウェア無しで一通り検証できます（オフスクリーン描画・エピソード録画を含む）。セットアップ→学習→ロールアウトの手順と RTC パラメータの解説は [docs/rtc-sim-rollout.md](docs/rtc-sim-rollout.md) を参照してください。

## ロードマップ: コンベア把持タスクへの拡張（設計検討中）

> このセクションは未実装の今後の計画です。すでに動く機能は [機能一覧](#機能一覧) を参照してください。

### 目標タスク

- ベルトコンベアで流れてくる物体を把持し、箱に入れる。
- コンベアの速度は複数パターンに変化させる。
- 画像情報から物体の接近を検出する detector を新規実装し、検出時に VLA へ Action Chunk の再生成を要求することで、既定の（キュー残量ベースの）再計画より速い反応を実現する。
- VLA（`smolvla_base` を想定）と detector の両方の学習が必要。
- detector の実装方式は未確定。任意の実装に差し替えられる構成にしたい。

### 現状の実装フロー（再掲）

1. SO-101 実機セットアップ → `set-port` → `setup-motors` → `calibrate` → `set-camera`（初回のみ）。
2. `record` でテレオペしながらデータセットを記録（現状は据え置きの pick & place のみ）。
3. `train`（またはPBSジョブ）で `smolvla_base` をファインチューニング。
4. `policy-test` でオフライン推論を確認。
5. `eval` で実機上のポリシーを評価 — 内部は `lerobot-rollout --strategy.type=episodic --inference.type=sync` による**同期推論**で、RTC ではない。評価データセットの repo-id は `eval_` ではなく `rollout_` プレフィックスが必須（例: `rollout_test`）。
6. RTC 非同期ロールアウト（`lerobot-rollout --inference.type=rtc`）は現状 **MuJoCo シム限定**（[docs/rtc-sim-rollout.md](docs/rtc-sim-rollout.md)）。`--robot.type=so101_follower` への切り替えは `Robot` 抽象上は可能なはずだが、実機での検証はまだ無く、`cli/so101.py` にもラッパーが無い。

### 次回実機を触るときの手順

1. `pixi run set-port leader` / `pixi run set-port follower`（初回のみ）
2. `pixi run setup-motors leader` / `pixi run setup-motors follower`（初回のみ）
3. `pixi run calibrate leader` / `pixi run calibrate follower`
4. `pixi run set-camera <name> --index N`（follower にカメラを割り当て）
5. `pixi run check leader` / `pixi run check follower`（任意の事前診断）
6. `pixi run teleop` で動作確認
7. `pixi run record --task "pick the object from the belt and place it in the box" --repo-id <name> --episodes <N>` でデータ収集
8. `pixi run train --policy-path lerobot/smolvla_base --repo-id <name> ...` でファインチューニング（長時間なら [`jobs/train_smolvla.pbs`](jobs/train_smolvla.pbs)）
9. `pixi run policy-test --policy <checkpoint> --repo-id <name>` でオフライン確認
10. `pixi run eval --policy <checkpoint> --task "..." --repo-id rollout_<name>` で実機・同期推論評価
11. RTC を実機で試す場合は、ラッパーが無いため `lerobot-rollout` を手動で組み立てる必要がある（`--robot.type=so101_follower --robot.port=... --robot.id=... --robot.cameras='{...}'` など。[docs/rtc-sim-rollout.md](docs/rtc-sim-rollout.md) のシム向けコマンドの `--robot.type` を差し替えたものに相当）。

### 不足している要素

1. **コンベア（実機）**: 可変速度のベルトコンベア自体・その速度設定の記録/再現手段が無い。
2. **タスク用データセット**: 既存の `lerobot/svla_so101_pickplace` は据え置きの pick & place。コンベアからの取得 → 箱への配置を含む新規データセットの収集が必要。
3. **detector の実装が存在しない**: 入力（画像のみ／関節角度も使うか）・出力（接近フラグ／距離／bbox）が未決定。「なんでも挟める」構成にするなら、detector 用の抽象インターフェース（差し替え可能なプロトコル）を `lerobot` フォーク側に新設する設計が必要。
4. **detector → RTC のイベント駆動トリガー経路が無い**: 現在の RTC（`rollout/inference/rtc.py`）は `queue_threshold`（キュー残量）でのみ再計画する。「detector が近づいたと判定した瞬間に強制リプランする」というイベント駆動の差し込み口（例: `force_replan()` の追加）はまだ実装されていない。
5. **detector の学習データが無い**: 「物体が接近した」をラベル付けした学習データの収集手段が未整備。
6. **可変速度に対する評価手段が無い**: 異なるコンベア速度での成功率を比較する評価プロトコル・集計ツールが無い（既存の `eval` は録画のみで成功/失敗の自動判定をしない）。
7. **実機での RTC 自体が未検証**: シムでの動作確認のみで、実機（`so101_follower`）に対しては一度も流していない。

