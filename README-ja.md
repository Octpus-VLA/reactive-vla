# reactive-vla

[English](README.md) | 日本語

first octpus vla project repository

📖 **ドキュメント:** <https://octpus-vla.github.io/reactive-vla/> — セットアップ・SmolVLAファインチューニング・lerobot editable構成・RTC simロールアウトの手順ガイドはこちら。この README はコマンド/機能のリファレンス、ドキュメントサイトは読み物形式の手順ガイドという役割分担です。

## 機能一覧

- **SO-101 実機操作 CLI**（`cli/so101.py`、`pixi run <command>` として公開）— leader/follower アームを一度登録すれば、以降はキャリブレーション・テレオペ・データセットの記録/再生/可視化/編集・Hubへのアップロードまで行えます。詳細は下記の[コマンド一覧](#so-101-コマンド-pixi-run-command)を参照。アーム登録・テレオペの流れ（`set-port` → `setup-motors` → `calibrate` → `teleop`）は [Adwaver4157/lecture_lerobot_teleop](https://github.com/Adwaver4157/lecture_lerobot_teleop) を参考にしています。
- **模倣学習ファインチューニング** — SO-101 データセットで `smolvla_base` / `pi0_base` をファインチューニング（またはスクラッチ学習）。W&Bロギング・Hugging Face Hubへのpushにも対応。詳細は下記の[ファインチューニング](#ファインチューニング)を参照。
- **HPCバッチ学習** — `pixi run train` をインタラクティブに実行する代わりに、PBSジョブとして投入できます。PBSスクリプト自体はキュー名・`group_list` などサイト固有の設定を含むため、このリポジトリには含めていません。[ファインチューニング](#ファインチューニング)節のテンプレートを自分のサイト向けに調整して `jobs/` 以下に置いてください（`jobs/` は `.gitignore` 済みです）。
- **MuJoCoシミュレーション** — 同梱の SO-ARM100 モデルと `sim_so101` ロボットアダプタにより、実機無しで RTC 非同期ロールアウト経路を検証できます。詳細は [docs/rtc-sim-rollout.md](docs/rtc-sim-rollout.md) を参照。

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

### 4. Lerobot(SO-101)の調整
詳しくは [Adwaver4157/lecture_lerobot_teleop](https://github.com/Adwaver4157/lecture_lerobot_teleop)を参照
1. `pixi run set-port leader` / `pixi run set-port follower`（初回のみ）
2. `pixi run setup-motors leader` / `pixi run setup-motors follower`（これは基本的にやる必要なし）
3. `pixi run calibrate leader` / `pixi run calibrate follower`
4. `pixi run set-camera front --index 6`（follower にカメラを割り当て）
5. `pixi run set-camera overall --index 4`
6. `pixi run check leader` / `pixi run check follower`（事前診断）
7. `pixi run teleop` で動作確認

## データ収集

`pixi run record` は leader でテレオペしながら follower + カメラの観測を記録し、`lerobot-record` を呼んでデータセットを作成します（`cli/so101.py` の `record` コマンド）。

```bash
pixi run record \
  --task "pick up the red cube and place it in the box" \
  --repo-id lift_red_cube_50episodes \
  --episodes 50 \
  --push
```

### 主なオプション

| フラグ | デフォルト | 用途 |
|---|---|---|
| `--task "<prompt>"` | (必須) | データセットに保存する自然言語のタスク説明 |
| `--repo-id <name>` | 省略可（`--resume` 時は必須） | データセットid。省略すると `<taskのslug>/<MMDD_HHMM>` を自動生成（例: `--task "pick up the red cube"` → `pick_up_the_red_cube/0620_2015`）。`outputs/train/<policy>/<dataset>/<timestamp>` と同じ命名規則です。`/` を含むため `_resolve_repo` は明示的な namespace/name として扱い、HFユーザー名は前置されません — Hubにpushする場合は `<taskのslug>` という名前のnamespace（実際のHFユーザー/組織）が必要になる点に注意してください |
| `--episodes N` | 5 | 記録するエピソード数 |
| `--episode-time SEC` | 20 | 1エピソードの自動停止までの秒数（右矢印キーで早期終了可） |
| `--reset-time SEC` | 5 | エピソード間でシーンをリセットする秒数 |
| `--fps N` | 30 | 記録フレームレート |
| `--push` / `--no-push` | `--no-push` | 記録後にHugging Face Hubへアップロード（事前に `pixi run hf-login` が必要） |
| `--max-rel DEG` | None | followerの1ステップあたりの最大移動角度（安全策） |
| `--display` / `--no-display` | `--display` | Rerunビューアでの可視化 |
| `--keep-viewer` | off | 終了後もRerunビューアを開いたままにする |
| `--cameras` / `--no-cameras` | `--cameras` | カメラ観測の記録有無 |


### 操作方法

記録は自動的に開始します。フォーカスされたターミナル上で矢印キーで制御します。

- **→ (右矢印)**: 現在のエピソードを停止して次へ進む
- **← (左矢印)**: 現在のエピソードを再記録
- **Esc**: セッション全体を停止

### 記録済みデータセットを後からHugging Face Hubへアップロードしたいとき

`--no-push`（デフォルト）で記録した場合や `record` 実行後に気が変わった場合は、ローカルデータセットを後から `upload` でアップロードできます。

```bash
pixi run upload --repo-id <name>
```

`--private` でプライベートリポジトリとして作成、`--tags tag1,tag2` でデータセットカードにタグを付けられます。


## ファインチューニング

事前学習済みモデル [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)（450M）を SO-101 データセットでファインチューニングします。

### 1. (HPC利用時) GPUノードへの移動

```bash
qsub -I -q interact-g -W group_list=gw13 -l select=1 -l walltime=02:00:00
```

### 2. 実行

```bash
pixi run train \
  --policy-path lerobot/smolvla_base \
  --repo-id Octpus-VLA/<dataset> \
  --batch-size 64 --steps 10000 --save-freq 2000 \
  --job-name smolvla_so101_pickplace --device cuda \
  -- --rename_map='{"observation.images.<camera>": "observation.images.camera1"}'
```

- カメラ名がデータセット側で `smolvla_base` の期待する名前（`camera1`〜`camera3`）と異なる場合は `--rename_map` でマッピングします。マップに含めなかったキーは自動的に学習から除外されます。
- 学習結果は `outputs/train/<policy>/<dataset>/<タイムスタンプ>`（gitignore済み）に出力されます。`--job-name` はW&B上の表示名のみに使われ、ディレクトリ名には影響しません。

**HPCで長時間バッチ投入したい場合** は、上記コマンドを包んだPBSスクリプトを自分で用意してください（キュー名・`group_list`・walltimeはサイトに合わせて変更。`jobs/` は `.gitignore` 済みなので、ここに置いたファイルはリポジトリにはpushされません）。テンプレート例:

```bash
#!/bin/bash
#PBS -q short-g
#PBS -W group_list=<your-group>
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -N train_smolvla
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}"

pixi run train \
  --policy-path lerobot/smolvla_base \
  --repo-id Octpus-VLA/<dataset> \
  --batch-size 64 --steps 10000 --save-freq 2000 \
  --job-name smolvla_so101_pickplace --device cuda \
  -- --rename_map='{"observation.images.<camera>": "observation.images.camera1"}'
```

`qsub -v` で値を渡したい場合は、`DATASET_REPO="${DATASET_REPO:?DATASET_REPO is required}"` のように環境変数を読む形にラップしてください。

### 3. W&B ロギング / Hugging Face Hub へのアップロード（任意）

```bash
pixi run wandb-login   # W&B 初回のみ
pixi run hf-login      # Hub push 初回のみ

pixi run train \
  --policy-path lerobot/smolvla_base --repo-id Octpus-VLA/<dataset> \
  --wandb --wandb-project <プロジェクト名> \
  --push-repo-id <名前> \
  -- --rename_map='{"observation.images.<camera>": "observation.images.camera1"}'
```

`--wandb-project`/`--wandb-entity` を省略すると既定のプロジェクト/個人アカウントに記録されます。`--push-repo-id` に bare name を渡すとHFユーザー名が自動で前置されます。学習後にまとめてpushしたい場合は `pixi run push-policy --checkpoint <checkpoint-dir> --repo-id <名前>`。

### 4. オフライン推論で確認（ロボット不要）

```bash
pixi run policy-test \
  --policy outputs/train/smolvla_base/<dataset>/<タイムスタンプ>/checkpoints/last/pretrained_model \
  --repo-id Octpus-VLA/<dataset> \
  --rename_map='{"observation.images.<camera>": "observation.images.camera1"}'
```

データセットに記録済みのフレームを入力し、ファインチューニング済みポリシーの推論レイテンシと、記録された実際の行動とのズレ（`mean |action - recorded|`、単位は度）を確認できます。学習時に `--rename_map` でカメラ名を変換した場合は、ここでも同じ `--rename_map` を渡してください。省略するとデータセット側のキー（例: `front`/`overall`）とチェックポイントが期待するキー（`camera1`〜`camera3`）が食い違い、`Feature mismatch between dataset/environment and policy config` エラーになります。

HPCでバッチ実行したい場合も、学習と同様に上記コマンドを包んだPBSスクリプトを自分で `jobs/` 以下に用意してください（`.gitignore` 済み）。

参考: [SmolVLAファインチューニングガイド](https://huggingface.co/docs/lerobot/en/smolvla)

### pi0 (`lerobot/pi0_base`)

`--policy-path lerobot/smolvla_base` を `--policy-path lerobot/pi0_base` に変えるだけで同じ手順が使えますが、2点異なります。

- **カメラ名も固定です。** `pi0_base` は `smolvla_base` と同様に、入力特徴量が `observation.images.base_0_rgb`・`left_wrist_0_rgb`・`right_wrist_0_rgb`（OpenPI/DROID由来のbase + wrist×2のカメラ構成）に固定されています。「データセットのカメラ名をそのまま動的に使う」わけではないので、データセット側のキー名が異なる場合は `--rename_map` が必要です（例: `'{"observation.images.front": "observation.images.base_0_rgb"}'`）。マップしなかったキーは無視され、マップされなかった残りの期待カメラはマスク付きのダミー画像で自動的に埋められます。
- モデルが大きいため `--batch-size` は4〜8程度に下げてください。`pi0_base` は既定で `train_expert_only=false`・`freeze_vision_encoder=false`・`use_amp=false`（全4Bパラメータをfp32でフル学習）なので、パラメータ・勾配・AdamWのオプティマイザ状態（m, v）だけで **固定約64GB**（4.03B × 4byte × 4）がバッチサイズに関係なく乗ります。つまり「1バッチあたり何GB」という線形の見積もりは成立せず、活性化メモリ（バッチサイズに比例する部分）だけが追加コストです。GPUのVRAM次第なので、目安が欲しい場合は短いステップ数で試し打ちしてください: `pixi run train --policy-path lerobot/pi0_base --repo-id Octpus-VLA/<dataset> --batch-size 6 --steps 10 --device cuda -- --rename_map='{"observation.images.front": "observation.images.base_0_rgb"}'`。さらに大きいバッチを通したい場合は次のフラグが効きます（メモリ削減効果が大きい順）: `-- --policy.train_expert_only=true`（VLM本体を凍結しaction expertのみ学習）、`-- --policy.freeze_vision_encoder=true`、`-- --policy.gradient_checkpointing=true`、`-- --policy.use_amp=true`。

> **事前準備が必要**: `pi0_base` のトークナイザーは Google の Gated リポジトリ [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) を使います。そのページでライセンスに同意した上で、HFトークンが **fine-grained** タイプの場合は、個別リポジトリのスコープ設定とは別に、トークン全体の **Global** 設定で "Read access to contents of all public gated repos you can access" を有効にしてください（個別リポジトリへの `scoped` 権限だけでは他人の名前空間のGatedリポジトリには効きません）。設定が面倒な場合は fine-grained ではない通常の **Read** タイプのトークンでも構いません。上記の設定でファインチューニングが正常に完走することを確認済みです。

## 推論

```bash
pixi run eval --policy <checkpoint> --task "..." --repo-id rollout_<name>
```

実機上でポリシーを実行し、評価エピソードを記録します（内部は `lerobot-rollout --strategy.type=episodic --inference.type=sync` の同期推論）。評価データセットの repo-id は `eval_` ではなく **`rollout_` プレフィックスが必須**です（例: `rollout_test`）。

RTC（非同期 Real-Time Chunking）の非同期ロールアウトは現状 **MuJoCoシム限定**（[docs/rtc-sim-rollout.md](docs/rtc-sim-rollout.md)）です。実機で試す場合は `cli/so101.py` にラッパーが無いため、`lerobot-rollout --robot.type=so101_follower --robot.port=... --robot.id=... --robot.cameras='{...}'` のように手動で組み立てる必要があります（シム向けコマンドの `--robot.type` を差し替えたものに相当）。

## ロードマップ

### 目標タスク

- ベルトコンベアで流れてくる物体を把持し、箱に入れる。
- コンベアの速度は複数パターンに変化させる。
- 画像情報から物体の接近を検出する detector を新規実装し、検出時に VLA へ Action Chunk の再生成を要求することで、既定の（キュー残量ベースの）再計画より速い反応を実現する。
- VLA（`smolvla_base` を想定）と detector の両方の学習が必要。
- detector の実装方式は未確定。任意の実装に差し替えられる構成にしたい。

### 不足している要素

1. **コンベア（実機）**: 可変速度のベルトコンベア自体・その速度設定の記録/再現手段が無い。
2. **タスク用データセット**: 既存の `lerobot/svla_so101_pickplace` は据え置きの pick & place。コンベアからの取得 → 箱への配置を含む新規データセットの収集が必要。
3. **detector の実装が存在しない**: 入力（画像のみ／関節角度も使うか）・出力（接近フラグ／距離／bbox）が未決定。「なんでも挟める」構成にするなら、detector 用の抽象インターフェース（差し替え可能なプロトコル）を `lerobot` フォーク側に新設する設計が必要。
4. **detector → RTC のイベント駆動トリガー経路が無い**: 現在の RTC（`rollout/inference/rtc.py`）は `queue_threshold`（キュー残量）でのみ再計画する。「detector が近づいたと判定した瞬間に強制リプランする」というイベント駆動の差し込み口（例: `force_replan()` の追加）はまだ実装されていない。
5. **detector の学習データが無い**: 「物体が接近した」をラベル付けした学習データの収集手段が未整備。
6. **可変速度に対する評価手段が無い**: 異なるコンベア速度での成功率を比較する評価プロトコル・集計ツールが無い（既存の `eval` は録画のみで成功/失敗の自動判定をしない）。
7. **実機での RTC 自体が未検証**: シムでの動作確認のみで、実機（`so101_follower`）に対しては一度も流していない。

## トラブルシューティング

実行中のジョブは `qstat` で確認できます。
