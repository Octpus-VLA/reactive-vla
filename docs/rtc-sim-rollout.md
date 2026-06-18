# SmolVLA + RTC 非同期ロールアウトをシミュレーションで動かす

実機の `lerobot-rollout --inference.type=rtc`（非同期 Real-Time Chunking）経路を、**ハードウェア無しで MuJoCo シミュレーションで動かす**ための一通り（セットアップ → 学習 → ロールアウト/録画）。最終ゴールは実機 SO-101。

ポイント：RTC 経路は `Robot` 抽象を要求する実機前提の作りなので、MuJoCo を裏に持つ `Robot`（`sim_so101`）を1個用意し、`--robot.type=sim_so101` で同じパイプラインをシムで流す。アダプタを実機ドライバ（`so101_follower`）に差し替えれば実機へ移行できる。

GH200 ノードで end-to-end の動作を確認済み（学習 → sim 接続 → オフスクリーン描画 → RTC 非同期ロールアウト → 録画）。

---

## 全体フロー

```
1. セットアップ   assets 同梱モデル + pixi install + MUJOCO_GL=egl
2. 学習           lerobot-train で普通の SmolVLA を作る（RTC は学習に不要）
3. ロールアウト   lerobot-rollout --inference.type=rtc でシムを走らせる
   └ 録画する場合は --strategy.type=episodic で MP4 を保存
```

---

## 1. セットアップ

### MuJoCo モデル（リポジトリ同梱・clone 不要）

DeepMind Menagerie の SO-ARM100（Apache-2.0）を `assets/so_arm100/` に同梱済み。

- `scene_cameras.xml` … SmolVLA 用カメラ（`cam1`/`cam2`）を足したもの。**ロールアウトではこれを使う**
- `scene.xml` … upstream そのまま（カメラなし）
- `so_arm100.xml` + `assets/` … モデル本体。actuator/joint 名（`Rotation/Pitch/Elbow/Wrist_Pitch/Wrist_Roll/Jaw`）は `SimSO101Config.joint_map` の既定と一致

> カメラ位置・向きは `scene_cameras.xml` 内の暫定値。実機の撮影アングルに合わせて要調整（`mujoco.viewer` で当たりを付けると早い）。

### 環境とレンダラ

```bash
pixi install                 # pixi.toml の mujoco を導入
export MUJOCO_GL=egl         # ヘッドレスGPU描画。ダメなら osmesa（CPU描画）にフォールバック
```

---

## 2. 学習: SmolVLA チェックポイントを作る

**RTC は推論時のテクニックなので、学習側に RTC 用の処理は一切不要**（普通の SmolVLA ファインチューニングでよい）。RTC の有無はロールアウト時に切り替える。

GPU ノードでの学習は PBS ジョブ [`jobs/train_smolvla.pbs`](https://github.com/Octpus-VLA/reactive-vla/blob/main/jobs/train_smolvla.pbs) を使う（リポジトリ root から投入）：

```bash
qsub jobs/train_smolvla.pbs                       # 既定: 20000 steps, save_freq=2000, short-g(<=8h)
qsub -v STEPS=10000 jobs/train_smolvla.pbs        # 配線確認なら steps を減らして高速に
qsub -v RESUME=true jobs/train_smolvla.pbs        # walltime で切れた後、最後のckptから再開
qsub -l walltime=06:00:00 -q small-g jobs/train_smolvla.pbs   # 長時間ジョブ
```

監視 `qstat -u $USER`、ログは投入ディレクトリの `train_smolvla.o<jobid>`。完了すると `outputs/train/smolvla_so101_pickplace/checkpoints/last/pretrained_model` が生成され、ロールアウトの `--policy.path` にそのまま渡せる。

インタラクティブに直接回すなら（ジョブ不要）：

```bash
pixi run lerobot-train \
  --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  --batch_size=64 --steps=10000 --save_freq=2000 \
  --output_dir=outputs/train/smolvla_so101_pickplace --wandb.enable=false
```

**学習のハマりどころ（実測）**

- **`--save_freq` を小さく刻む**。既定は `20000` ＝途中保存ゼロで、walltime に殺されると全ロスト。`2000` 推奨。
- **walltime はステップ数に対し十分に**。GH200 で約 3.45 step/s（20000 steps ≈ 1h40m + データロード）。`interact-g`（上限2h）はギリギリなので 20000 はバッチ（`short-g`）が無難。
- **`--policy.push_to_hub=false` が必須**（Hub に上げないなら）。付け忘れると `repo_id` 不足で検証エラー。上げる場合は `--policy.repo_id=<user>/<name>` + `HF_TOKEN`。
- **再開**は `--resume=true` + `--config_path=.../checkpoints/last/pretrained_model/train_config.json` の2点セット。非 resume 実行で `output_dir` が既存だとエラー。

---

## 3. ロールアウト

### 動作確認（録画なし・base 戦略）

```bash
export MUJOCO_GL=egl
pixi run lerobot-rollout \
  --policy.path=outputs/train/smolvla_so101_pickplace/checkpoints/last/pretrained_model \
  --policy.rtc_config.enabled=true --policy.rtc_config.execution_horizon=10 \
  --robot.type=sim_so101 \
  --robot.mjcf_path=$PWD/assets/so_arm100/scene_cameras.xml \
  --robot.cameras='{camera1: {mujoco_name: cam1, width: 320, height: 240}, camera2: {mujoco_name: cam2, width: 320, height: 240}}' \
  --robot.control_fps=30 \
  --inference.type=rtc --inference.rtc.execution_horizon=10 --inference.queue_threshold=30 \
  --fps=30 --task="Grab the cube" --duration=30
```

### 録画する（episodic 戦略 → MP4）

カメラ映像を LeRobotDataset に保存する。base との差分は太字の4つ：

```bash
export MUJOCO_GL=egl
pixi run lerobot-rollout \
  --policy.path=outputs/train/smolvla_so101_pickplace/checkpoints/last/pretrained_model \
  --policy.rtc_config.enabled=true --policy.rtc_config.execution_horizon=10 \
  --robot.type=sim_so101 \
  --robot.mjcf_path=$PWD/assets/so_arm100/scene_cameras.xml \
  --robot.cameras='{camera1: {mujoco_name: cam1, width: 320, height: 240}, camera2: {mujoco_name: cam2, width: 320, height: 240}}' \
  --robot.control_fps=30 \
  --inference.type=rtc --inference.rtc.execution_horizon=10 --inference.queue_threshold=30 \
  --fps=30 --task="Grab the cube" \
  --strategy.type=episodic \
  --dataset.repo_id=local/rollout_sim_rtc_eval \
  --dataset.root=outputs/rollout/rollout_sim_rtc_eval \
  --dataset.num_episodes=1 --dataset.episode_time_s=30 \
  --dataset.push_to_hub=false \
  --play_sounds=false
```

出力 → `outputs/rollout/rollout_sim_rtc_eval/videos/.../observation.images.camera1/episode_000000.mp4`（camera2 も同様）。

**ロールアウトのハマりどころ（実測）**

- **`--play_sounds=false` が必須**（HPC）。既定 `true` だと `spd-say`（音声合成）を起動しようとして `FileNotFoundError` でクラッシュする。
- **`--dataset.repo_id` は `rollout_` で始める**必要がある（例 `local/rollout_*`）。違うと検証エラー。
- **`--dataset.push_to_hub=false`** を明示（既定 `true`）。
- `pynput` 無し（`Headless environment detected`）はそのまま。`episode_time_s` 経過で自動終了する。

---

## オプション解説

| オプション | 意味 |
|---|---|
| `--policy.path` | チェックポイント（ローカルdir）または HF repo id |
| `--policy.rtc_config.enabled` | **RTC ガイダンスの ON/OFF**。`false` で素のチャンク実行 |
| `--policy.rtc_config.execution_horizon` | ガイダンスが及ぶ終端＝前チャンク(leftover)をブレンドする長さ |
| `--robot.type` | ロボットドライバ。シムは `sim_so101`、実機は `so101_follower` |
| `--robot.mjcf_path` | MuJoCo シーン定義。カメラ入り `scene_cameras.xml` を使う |
| `--robot.cameras` | `{学習時のカメラキー: {mujoco_name: シーン内カメラ名, width, height}}`。**キー名は学習データと一致必須** |
| `--robot.control_fps` | sim の制御ステップ周期。MJCF の timestep から substeps を算出（ログの "17 substeps"） |
| `--inference.type` | 推論エンジン。`rtc`=非同期RTC / `sync`=同期（毎ステップ素直に） |
| `--inference.rtc.execution_horizon` | エンジン側の horizon。**ポリシー側と同値にする**（下記） |
| `--inference.queue_threshold` | キュー残量がこれ以下でリプラン（再推論）発火 |
| `--fps` | メインループがキューから1 action を pop する周期 |
| `--task` | VLA への言語指示 |
| `--duration` | base 戦略の実行秒数（`0`=無限） |
| `--strategy.type` | `base`=記録なし / `episodic`=エピソード録画 / `sentry`・`highlight`・`dagger`=各種記録 |
| `--dataset.repo_id` | 録画データセット名（**`rollout_` 接頭辞必須**） |
| `--dataset.root` | 保存先ディレクトリ（接頭辞不要） |
| `--dataset.num_episodes` / `--dataset.episode_time_s` | episodic の本数 / 1本の最大秒数 |
| `--dataset.push_to_hub` | Hub アップロード。HPC ローカル検証では `false` |
| `--play_sounds` | 音声読み上げ。HPC では `false` 必須 |

> **RTC の `execution_horizon` が2系統**ある点に注意。ポリシー側 `--policy.rtc_config.*`（ガイダンス発火）とエンジン側 `--inference.rtc.*`（キュー管理）で、**両方を一致**させる（`predict_action_chunk` に horizon を渡していないため、ガイダンスはポリシー側 config を見る）。

---

## RTC パラメータの仕組み

![RTC async mechanism](./rtc-async-mechanism.png)

### 4パラメータの役割（混同しやすい）

| パラメータ | 既定 | 何を決めるか | 出どころ |
|---|---|---|---|
| `chunk_size` | 50 | キューに積む実行ステップ数（`predict_action_chunk` の返り長） | smolvla config |
| `queue_threshold` | 30 | **いつリプランするか**。実効リプラン間隔 ≈ `chunk_size − queue_threshold` | `--inference.queue_threshold` |
| `execution_horizon` | 10 | **leftover を使う長さ**＝ガイダンスのブレンド終端。キュー長は決めない | `RTCConfig.execution_horizon` |
| `delay`（≒inference_delay） | 実測 | leftover の先頭を何ステップ凍結するか = `ceil(latency / (1/fps))` | レイテンシ実測 |

### 2スレッドの流れ

- **メインループ**（`rollout/strategies/base.py`）: `1/fps` ごとにキューから 1 action を pop → `robot.send_action`。
- **RTCスレッド**（`rollout/inference/rtc.py`）: `qsize ≤ queue_threshold` で `predict_action_chunk` を実行し、`ActionQueue.merge` でキュー差し替え（先頭 `delay` ステップは破棄）。

→ qsize は「リプラン直後 ≈ chunk_size − delay」から `queue_threshold` まで減って再補充、を繰り返す。

### 1チャンク内の重み付け（LINEAR）

```
timestep:  0 .. delay │ delay .. execution_horizon │ execution_horizon .. chunk_size
weight:      1.0       │      1.0 → 0.0（線形）       │            0.0
領域:        凍結       │        ブレンド               │            自由
```

- **凍結（w=1.0）**: 推論中に実機が実行してしまう先頭 `delay` ステップ。前チャンク(leftover)に完全一致させる。
- **ブレンド（1→0）**: 旧→新チャンクへ滑らかに移行し接合部の不連続を防ぐ。
- **自由（w=0）**: `execution_horizon` 以降。前チャンクを無視して新観測で自由に生成。

各 denoising ステップで `correction = grad((leftover − x1_t) · weights)` を計算し、`v_t − guidance_weight·correction` で前チャンクへ引き寄せる（`policies/rtc/modeling_rtc.py`）。

---

## 検証で分かったこと

- **RTC は実際に効いている**: GH200 でも `real_delay ≈ 18 frames`（30fps で約 0.6s）。つまり推論レイテンシで凍結区間が実際に発生しており、レイテンシ注入なしで RTC の検証になっている。
- ログの `Indexes diff is not equal to real delay (indexes_diff=17, real_delay=18)` は **off-by-1 の bookkeeping ズレで無害**（RTC は `real_delay` を採用して継続）。毎ステップ出てうるさいだけ。
- ロールアウト終了時の `EGLError`（renderer の `__del__`）は後始末の雑音で、正常完走には無関係。

---

## 残課題・留意点

- **カメラアングルが暫定**: `scene_cameras.xml` の `pos`/`xyaxes` は当たり値。MP4 がアームを捉えていなければ調整して撮り直し。
- **単位変換**: body 関節は deg↔rad 線形、gripper は MJCF の joint range と [0,100] を線形対応。符号・オフセットが学習データの action 空間とズレる可能性 → 収録1エピソードを `send_action` に流して qpos の妥当性を確認すると良い。
- **把持性能は別物**: sim ゼロショットで意味ある把持はしない。この sim の主目的は **action/observation 配線と RTC 非同期挙動の検証**。性能評価には sim 収録データでの再学習が要る。
- **RTC 効果をさらに強調したい場合（任意）**: `RTCInferenceConfig` に `simulated_latency_s` を足して `_rtc_loop` に sleep を挟む小改修で、凍結区間を意図的に伸ばせる（未実装）。

---

## 実装メモ / コミット運用

`sim_so101`（submodule = Octpus-VLA/lerobot フォーク）の構成：

- `src/lerobot/robots/sim_so101/` — `SimSO101(Robot)` + `SimSO101Config`。action/observation キーは `so101_follower` と完全一致。`connect` で MJCF ロード、`send_action` で `mj_step`、`get_observation` で qpos 読み + オフスクリーン描画。`mujoco` は遅延 import。
- `robots/utils.py` の `make_robot_from_config` と `scripts/lerobot_rollout.py` の import に `sim_so101` 分岐を追加（draccus 登録）。
- 後で gym env ラップに変えても、`SimSO101` の `connect/get_observation/send_action` 内部だけ差し替えれば登録・キー整合・rollout 配線はそのまま使える。

コミット：

- submodule 側の変更 → フォークにコミット・push（ブランチ `feat/sim-so101-rtc` 推奨）
- 親リポジトリ → submodule ポインタ bump + `pixi.toml`・`jobs/`・`docs/` の変更
