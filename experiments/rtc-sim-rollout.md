# SmolVLA + RTC 非同期ロールアウトをシミュレーションで動かす

## 目的・方針

実機の `lerobot-rollout --inference.type=rtc`（非同期 Real-Time Chunking）経路を、**ハードウェア無しでシミュレーションで動かす**。最終ゴールは実機 SO-101。

非同期RTCロールアウト経路は `Robot` 抽象を要求し実機前提（`RolloutConfig` に env フィールドは無い）。そこで **MuJoCo を裏に持つ `Robot`（`sim_so101`）を1個実装**し、`--robot.type=sim_so101` で同じパイプラインをシムで流す。アダプタを実機ドライバ（`so101_follower`）に差し替えれば実機へ移行できる。

前例: `unitree_g1` robot が `is_simulation` モードで MuJoCo gym env をラップしており、このパターンは公式に存在する。`sim_so101` はそれの SO-101 版（DDS 等は不要なのでより薄い）。

RTC の仕組みの早見表は [docs/rtc-async-mechanism.md](../docs/rtc-async-mechanism.md) を参照。

## 実装済みのもの（submodule: Octpus-VLA/lerobot）

- `src/lerobot/robots/sim_so101/` — `SimSO101(Robot)` + `SimSO101Config` + `SimCameraConfig`
  - action/observation キーは `so101_follower` と完全一致（`{shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper}.pos` + カメラ）
  - `connect` で MJCF をロードし motor→actuator/joint を解決、`send_action` で `mj_step`、`get_observation` で qpos 読み + オフスクリーンレンダリング
  - `mujoco` は遅延 import（未導入でも robot 登録 import は通る）
- `src/lerobot/robots/utils.py` — `make_robot_from_config` に `sim_so101` 分岐を追加
- `src/lerobot/scripts/lerobot_rollout.py` — robot import 群に `sim_so101` を追加（draccus 登録のため）
- 親リポジトリ `pixi.toml` — `mujoco >= 3.0` を追加

ローカル確認済み: ruff 通過 / `sim_so101` が draccus に登録 / インスタンス化成功 / action_features が so101 と一致。
**未検証（HPC で要確認）**: MuJoCo の実 step・オフスクリーン描画、単位変換の妥当性、RTC 非同期挙動。

## セットアップ手順（HPC GPU/Linux ノード）

### 1. MuJoCo モデルを取得

DeepMind Menagerie の SO-ARM100 モデルを使う（Apache-2.0）。

```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
# 必要なのは mujoco_menagerie/trs_so_arm100/ 配下（scene.xml + so_arm100.xml + assets/）
```

`scene.xml` には `<camera>` が無いので、SmolVLA 用にカメラを追記する。`<worldbody>` 内に例えば：

```xml
<camera name="cam1" pos="0.4 0.0 0.4" xyaxes="0 -1 0 0.3 0 1"/>
<camera name="cam2" pos="0.0 0.5 0.4" xyaxes="-1 0 0 0 -0.5 1"/>
```

（位置・向きは要調整。`mujoco.viewer` で当たりを付けると早い。）

### 2. mujoco を環境に入れる

```bash
pixi install   # pixi.toml に追加済みの mujoco を解決・導入
```

### 3. オフスクリーン描画のバックエンド（ヘッドレス必須）

GPU ノードでヘッドレス描画するため：

```bash
export MUJOCO_GL=egl     # ヘッドレスGPU。ダメなら osmesa（CPU描画）にフォールバック
```

## 実行コマンド（雛形）

```bash
pixi run lerobot-rollout \
  --policy.path=outputs/train/smolvla_so101_pickplace/checkpoints/last/pretrained_model \
  --policy.rtc_config.enabled=true \
  --policy.rtc_config.execution_horizon=10 \
  --robot.type=sim_so101 \
  --robot.mjcf_path=$PWD/mujoco_menagerie/trs_so_arm100/scene.xml \
  --robot.cameras='{camera1: {mujoco_name: cam1, width: 320, height: 240}, camera2: {mujoco_name: cam2, width: 320, height: 240}}' \
  --robot.control_fps=30 \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.queue_threshold=30 \
  --fps=30 \
  --task="Grab the cube" \
  --duration=30
```

## HPC で検証すべき項目（要対応）

1. **MJCF の joint/actuator 名**: `SimSO101Config.joint_map` の既定（`Rotation/Pitch/Elbow/Wrist_Pitch/Wrist_Roll/Jaw`）が実際の `so_arm100.xml` と一致するか。`connect` で不一致なら fail-fast するので、エラーが出たら joint_map を実名に合わせる。
2. **RTCConfig が2系統**: ポリシー側 `--policy.rtc_config.*`（ガイダンス発火）とエンジン側 `--inference.rtc.*`（キュー/execution_horizon）。**両方の `execution_horizon` を一致**させる。`predict_action_chunk` に execution_horizon kwarg を渡していないため、ガイダンスはポリシー側 config を見る点に注意。
3. **カメラ名の一致**: `--robot.cameras` のキー（`camera1` 等）を、ファインチューニング時のカメラ名に合わせる（README の rename_map 参照。学習で使ったキーと不一致だと画像入力が噛み合わない）。
4. **単位変換の妥当性**: body 関節は deg↔rad 線形、gripper は MJCF の joint range と [0,100] を線形対応させているだけ。符号・オフセット・キャリブが学習データの action 空間とズレる可能性。記録データ1エピソードを `send_action` に流して qpos が妥当に動くか確認すると良い。
5. **レイテンシ注入（任意）**: 速い GPU だと `inference_delay≈0` で RTC の凍結区間がほぼ効かない。RTC の効果を観測したい場合のみ、`RTCInferenceConfig` に `simulated_latency_s` を足して `_rtc_loop` に sleep を挟む小改修を行う（前提として今回は未実装）。

## 留意点

- 既存チェックポイントは sim ではゼロショットで意味ある把持をしない。今回の sim の主目的は**「action/observation 配線と RTC 非同期挙動の検証」**。タスク性能評価には sim 収録データでの再学習が別途必要。
- バックエンドは Menagerie 直ラップを採用。後で gym env ラップ（gym-so100-c 等）に変えたくなっても、`SimSO101` の `connect/get_observation/send_action` 内部だけ差し替えれば、登録・キー整合・rollout 配線はそのまま使える。

## コミット運用

- submodule（`third_party/lerobot` = Octpus-VLA/lerobot フォーク）側の変更 → フォークにコミット・push（ブランチ `feat/sim-so101-rtc` 推奨）
- 親リポジトリ → submodule ポインタ bump + `pixi.toml`・`docs/`・`experiments/` の変更をコミット
