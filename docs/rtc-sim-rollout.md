# SmolVLA + RTC 非同期ロールアウト（実機 / シミュレーション）

`lerobot-rollout --inference.type=rtc`（非同期 Real-Time Chunking）でファインチューニング済み SmolVLA を動かすための一通り。**最終ゴールは実機 SO-101** で、同じパイプラインを MuJoCo シミュレーションでも流せる。

- **実機 SO-101**: 登録済みアームに対し `pixi run eval --rtc` で起動するのが最短。[簡易実行（実機 SO-101・最短）](#最短)を参照。
- **シミュレーション**: ハードウェア無しで配線と RTC 非同期挙動を検証する。`--robot.type=sim_so101` で同じパイプラインをシムで流す（[3. ロールアウト](#3-rollout)以降）。

RTC 経路は `Robot` 抽象を要求する実機前提の作りで、シムは MuJoCo を裏に持つ `Robot`（`sim_so101`）を1個用意して同じ経路を流している。アダプタを実機ドライバ（`so101_follower`）に差し替えれば実機へ移行できる。GH200 ノードで sim の end-to-end 動作を確認済み（学習 → sim 接続 → オフスクリーン描画 → RTC 非同期ロールアウト → 録画）。実機 RTC 経路はコード上は通っているが、実機での動作検証は未実施。

---

## 簡易実行（実機 SO-101・最短） {#最短}

アームとカメラを一度登録しておけば（[セットアップ](setup.md) / `pixi run set-port follower` / `pixi run calibrate follower` / `pixi run set-camera ...`）、RTC 非同期ロールアウトは **`--rtc` を付けるだけ**で起動できる。`pixi run eval` は登録済み follower の port / id / cameras と安全上限・データセット命名を自動で組み立てる（[`cli/so101.py`](https://github.com/Octpus-VLA/reactive-vla/blob/main/cli/so101.py) の `evaluate`）。

```bash
# 同期推論（従来どおり）
pixi run eval --policy <ckpt> --task "Grab the cube" --repo-id rollout_test

# 非同期 RTC（--rtc を足すだけ。横で再推論しながら実行）
pixi run eval --rtc --policy <ckpt> --task "Grab the cube" --repo-id rollout_rtc_test

# horizon / 再推論しきい値を変える + 急な動きを抑える安全上限（推奨）
pixi run eval --rtc --execution-horizon 10 --queue-threshold 30 --max-rel 5 \
  --policy <ckpt> --task "Grab the cube" --repo-id rollout_rtc_test

# detector を足すと replan タイミングが動的制御される（赤 cube の速度に追従）
pixi run eval --rtc --detector red_cube_speed \
  --policy <ckpt> --task "Grab the cube" --repo-id rollout_rtc_detector
```

- `<ckpt>` は学習で出た `outputs/train/.../checkpoints/last`（`pretrained_model` は自動補完）または Hub の `user/name`。
- `--rtc` を付けると内部で `--inference.type=rtc --inference.rtc.execution_horizon=… --inference.queue_threshold=… --policy.rtc_config.enabled=true --policy.rtc_config.execution_horizon=…` に展開される（horizon はポリシー側・エンジン側を自動で一致させる。理由は[オプション解説](#opts)の注記）。
- `--repo-id` は `rollout_` で始める必要がある（lerobot の制約）。エピソード間に follower は自動で初期姿勢へ戻る（leader 不要）。
- 自律実行では **`--max-rel`（1ステップの最大移動量 deg）を付けて急な動きを抑える**のを推奨。
- `--detector`（`motion` / `red_cube_speed`）は **`--rtc` と併用が前提**。`--inference.supervisor.enabled=true --inference.supervisor.detector.type=… --inference.supervisor.camera=…` に展開される。監視カメラは `--detector-camera` で指定（既定は登録済みカメラの `overall`、無ければ先頭）。細かいしきい値は passthrough で渡す（例 `--inference.supervisor.detector.urgent_speed_px_s=300`）。detector の仕組みは [Supervisor トリガによる動的 replan](supervisor-trigger.md) を参照。

下層の生コマンドや引数の意味は [実機 SO-101 で動かす（生コマンド）](#raw) と [オプション解説](#opts) を参照。

---

## 全体フロー

```
1. セットアップ   実機: アーム/カメラ登録 ・ sim: assets 同梱モデル + pixi install + MUJOCO_GL=egl
2. 学習           lerobot-train で普通の SmolVLA を作る（RTC は学習に不要）
3. ロールアウト   lerobot-rollout --inference.type=rtc を実機 / シムで走らせる
   └ 録画する場合は --strategy.type=episodic で MP4 / データセットを保存
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
qsub -v RESUME=outputs/train/smolvla_base/svla_so101_pickplace/<タイムスタンプ>/checkpoints/last jobs/train_smolvla.pbs
                                                   # walltime で切れた後、最後のckptから再開
qsub -l walltime=06:00:00 -q small-g jobs/train_smolvla.pbs   # 長時間ジョブ
```

監視 `qstat -u $USER`、ログは投入ディレクトリの `train_smolvla.o<jobid>`。完了すると `outputs/train/smolvla_base/svla_so101_pickplace/<タイムスタンプ>/checkpoints/last/pretrained_model` が生成され、ロールアウトの `--policy.path` にそのまま渡せる（実際のパスはログの `--output_dir=...` で確認）。

インタラクティブに直接回すなら（ジョブ不要、内部で `lerobot-train` を呼ぶ CLI ラッパー `pixi run train` を使う）：

```bash
pixi run train \
  --policy-path lerobot/smolvla_base \
  --repo-id lerobot/svla_so101_pickplace \
  --batch-size 64 --steps 10000 --save-freq 2000 \
  --device cuda \
  -- --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}'
```

**学習のハマりどころ（実測）**

- **`--save-freq` を小さく刻む**。既定は `20000` ＝途中保存ゼロで、walltime に殺されると全ロスト。`2000` 推奨。
- **walltime はステップ数に対し十分に**。GH200 で約 3.45 step/s（20000 steps ≈ 1h40m + データロード）。`interact-g`（上限2h）はギリギリなので 20000 はバッチ（`short-g`）が無難。
- **Hub に上げない場合は何も指定しなくてよい**（`pixi run train` は `--push-repo-id` を渡さない限り `--policy.push_to_hub=false` を自動付与）。上げる場合は `--push-repo-id <name>` + `HF_TOKEN`（`pixi run hf-login`）。
- **再開**は `pixi run train --resume <output_dir>/checkpoints/last` の1コマンドでよい（内部で `--config_path=.../pretrained_model/train_config.json --resume=true` に展開される）。非 resume 実行で同じ `output_dir` が既存だとエラーになるが、`output_dir` は実行ごとにタイムスタンプが付くため通常は衝突しない。

---

## 3. ロールアウト {#3-rollout}

実機・シムとも同じ `lerobot-rollout` を使い、`--robot.type` だけを差し替える（実機 `so101_follower` / シム `sim_so101`）。実機は `pixi run eval --rtc` が最短（[簡易実行](#最短)）。以下はその下層で何が走っているかと、シムでの動かし方。

### 実機 SO-101 で動かす（生コマンド） {#raw}

`pixi run eval --rtc` を使わず素の `lerobot-rollout` を直接叩く場合。前提として follower のキャリブレーション（`pixi run calibrate follower`、`~/.cache/huggingface/lerobot/calibration/so_follower/<id>.json` に保存）とシリアルポート権限（`/dev/ttyACM*`、`dialout`/`uucp` グループ）が必要。

```bash
lerobot-rollout \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=<follower-id> \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}' \
  --robot.max_relative_target=5 \
  --policy.path=outputs/train/.../checkpoints/last/pretrained_model \
  --policy.rtc_config.enabled=true --policy.rtc_config.execution_horizon=10 \
  --inference.type=rtc --inference.rtc.execution_horizon=10 --inference.queue_threshold=30 \
  --strategy.type=episodic \
  --dataset.repo_id=local/rollout_rtc_real --dataset.num_episodes=10 --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --fps=30 --task="Grab the cube"
```

**実機のハマりどころ・前提**

- **カメラキー名（`front`/`side` など）は学習データと一致必須**。`--robot.cameras` の各キーは学習時の `observation.images.<key>` に対応する（不一致だと観測が埋まらない）。`pixi run eval --rtc` は登録済みカメラからこれを自動生成する。
- **`--robot.max_relative_target`（=`--max-rel`）で1ステップの最大移動量を制限**。自律実行で急なモーションを防ぐ安全策。
- **`--robot.id` は `pixi run calibrate follower` で作った ID**。未キャリブレーションだと接続時に自動キャリブが走る。
- エピソード間 follower は自動で初期姿勢へ戻る（`--return_to_initial_position` 既定 true）。切断時はトルク解放（`disable_torque_on_disconnect` 既定 true）。
- 実機 RTC の凍結区間は推論レイテンシで自然に発生する（sim 同様）。`execution_horizon` はポリシー側・エンジン側を一致させる（[オプション解説](#opts)の注記）。

### 動作確認（シミュレーション・録画なし・base 戦略）

```bash
export MUJOCO_GL=egl
pixi run lerobot-rollout \
  --policy.path=outputs/train/smolvla_base/svla_so101_pickplace/<タイムスタンプ>/checkpoints/last/pretrained_model \
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
  --policy.path=outputs/train/smolvla_base/svla_so101_pickplace/<タイムスタンプ>/checkpoints/last/pretrained_model \
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

## オプション解説 {#opts}

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

## 測定シナリオ（RTC / replan step / detector）

反応性の比較実験は、`pixi run eval --rtc` を起点に3パターンで回せる。定量データは専用の計測コードを足さず、まず既存ログ（RTC スレッドの `real_delay` / queue サイズ、detector の `reason` / `speed_px_s`）を使う。

| シナリオ | コマンド | 振るパラメータ | 観察ログ |
|---|---|---|---|
| ① RTC 単体 | `pixi run eval --rtc ...` | `--execution-horizon` | `RTC inference latency=…, queue=…`（debug）, `real_delay` |
| ② replan step スイープ | `pixi run eval --rtc --queue-threshold {10,20,30,40} ...` | `--queue-threshold` | リプラン間隔 ≈ `chunk_size − queue_threshold`、queue 推移 |
| ③ detector 動的 replan | `pixi run eval --rtc --detector red_cube_speed ...` | `--inference.supervisor.detector.{slow,fast,urgent}_speed_px_s` 等 | `RTC early replan (detector): reason=… speed_px_s=…` |

- ③ では detector が `effective_chunk_size_threshold`（0–1）を返し、エンジンが `× chunk_size` で **動的な queue しきい値**に変換する。cube が速いほど早く（queue を多く残して）リプランし、`urgent_speed_px_s` 超で queue 残量に関係なく即時リプランする。
- detector が無効（既定）なら ① のゲート（`qsize ≤ queue_threshold`）のみで動き、既存挙動と完全に同じ。
- 同一指標で横並び比較したい場合、①②の `queue_threshold`（絶対ステップ）と③の `chunk_size_threshold`（割合）は `絶対 = 割合 × chunk_size` で換算する。

> detector・supervisor の実装は async inference 経路と共有（`lerobot/detectors/`）。同じ detector を PolicyServer + RobotClient 構成でも使える。詳細は [Supervisor トリガによる動的 replan](supervisor-trigger.md)。

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
