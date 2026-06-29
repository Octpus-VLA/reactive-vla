# スクリプトエキスパートによる sim デモ収集（ファインチューニング用）

実機データだけで学習した SmolVLA は MuJoCo レンダリング観測に対して分布外（テクスチャ・ライティングが別物）で、`sim-eval` ではほとんど動かない。これを埋めるには **観測が sim レンダリングである学習データ** が要る。本ツールはそれを生成する: 特権状態（cube の真の位置・速度）を使う pick-and-place コントローラが `SimSO101` を駆動し、各制御ステップの (観測, アクション) を `lerobot-record` と同一スキーマの `LeRobotDataset` に書き出す。出力はそのまま `pixi run train` に渡せる。

> 実装は [`cli/sim_collect.py`](https://github.com/Octpus-VLA/reactive-vla/blob/main/cli/sim_collect.py)、CLI ラッパは [`cli/so101.py`](https://github.com/Octpus-VLA/reactive-vla/blob/main/cli/so101.py) の `sim-collect`。シム本体の作りは [SmolVLA + RTC 非同期ロールアウト](rtc-sim-rollout.md) を参照。

## 特権情報とデータセットの分離（最重要）

エキスパートは特権情報（cube 姿勢・速度を MuJoCo state から直接読む、IK を解く）を使ってよいが、**データセットに残すのは実機でも観測できるものだけ**:

- **保存する**: 手首カメラ `camera1`（= `wrist_cam`）+ 固定外部視点 `overview`、関節状態 `observation.state`（6 自由度）、コマンドした目標関節角 `action`（6 自由度）。
- **保存しない**: cube の位置・速度などの特権状態。`check_success()` の判定も録らない。

学習するポリシーは画像と関節しか見ない。お手本を作ったエキスパートが特権情報を持っていたことは、学習には漏れない。

## 動かし方

```bash
# 静止 cube（ベルト停止）。cube は機体正面 (y=0) に置かれ、±3cm の xy ジッタで把持位置を散らす。
pixi run sim-collect --episodes 20 --repo-id sim_pickplace --task "Grab the cube"

# 動くベルト（暫定）。cube を -y 端から供給し、等速インターセプトを先読みして掴む。
pixi run sim-collect --episodes 20 --repo-id sim_pickplace_belt --belt-speed 0.05

# 既存データセットを作り直す / Hub に上げる
pixi run sim-collect --episodes 20 --repo-id sim_pickplace --overwrite
pixi run sim-collect --episodes 20 --repo-id sim_pickplace --push
```

主なオプション（`--help` に全量）:

- `--episodes` 収集エピソード数 / `--max-steps` 1 エピソードの制御ステップ上限。
- `--belt-speed` ベルト速度 m/s（0=静止）/ `--belt-distance` 機体からベルト近縁までの距離 m。
- `--jitter` cube 開始 xy の ±ランダム化幅 m（デモを 1 姿勢に固定しないため）。
- `--seed` 乱数シード（再現可能なデータセット）/ `--fps` 制御レート（= データセット fps、学習と一致させる）。
- レンダラは既定 `MUJOCO_GL=egl`（`sim-eval` と違い推論を挟まず GPU 競合がないので egl が安全かつ高速）。

出力は `$HF_LEROBOT_HOME/<repo_id>` に `lerobot-record` と同形式で書かれる。確認:

```bash
pixi run viz --repo-id sim_pickplace --episode 0     # Rerun で観測/状態/アクションを再生
```

## 仕組み

### 状態機械（[`PickPlaceExpert`](https://github.com/Octpus-VLA/reactive-vla/blob/main/cli/sim_collect.py)）

`approach → descend → grasp → lift → carry → place → release → done` の各相で TCP（グリッパ間の `gripperframe` site）の目標 xyz とグリッパ開閉を決める。把持後（lift 以降）の目標高さは **掴む前の cube 静止 z を固定参照** する（held 中の cube 自身の z を参照すると正のフィードバックで腕が暴走し full reach まで跳ね上がるため）。grasp は閉じ命令で 3cm の cube を物理的に挟む（グリッパは `0=閉 / 100=開` の実機スケール）。

### 閉ループ積分サーボ + ステップ制限

MuJoCo の位置アクチュエータは重力負荷で droop する（肘で約 8°、開ループの絶対角指令だと TCP が目標へ届かない）。そこで **絶対 IK 角を送らず**、TCP 誤差から Jacobian ステップを毎制御ステップ「実行中の関節指令」に積分する。指令が droop 分を超えて伸び、実 TCP が目標に到達するまで収束する（積分制御）。さらに 1 ステップの TCP 移動を上限クランプし、相切替で目標が大きく飛んでも腕が急振りして把持 cube を弾き飛ばさないようにしている（有界速度で滑らかに移動）。

### 動くベルトのインターセプト（暫定）

ベルトは等速・直線・+y 方向で速度既知なので、cube 現在 xy を「腕が到達するまでの時間 × ベルト速度」だけ +y に先読みした点を狙う（定数速度モデル）。`belt_speed=0` ではこれが現在 xy に縮退する。先読み時間は接近+下降相ぶんの保守的な見積りで、**まだ十分にチューニングしていない**（基本把持が安定したら refine する）。これは [CLAUDE.md](../CLAUDE.md) の Tier 3（predictive replan）のオフライン・特権情報版に相当する。

## 現状（実測）

- **静止 cube**: ±3cm ジッタで 4/4 = 100% 箱入れ成功を確認（`MUJOCO_GL=osmesa` / GH200）。生成データセットは `LeRobotDataset` で読み込め、`observation.state (6)` / `action (6)` / `observation.images.{camera1,overview} (240,320,3) video` を持ち、`pixi run train` 互換。
- **動くベルト**: 経路は通るが成功率は未検証・要チューニング。

## 次の段階

1. 静止 cube で 50〜100 エピソード収集 → SmolVLA をファインチューニング（`pixi run train`）→ `pixi run sim-eval` で「動くか」を検証。これが real→sim ギャップを実際に埋められるかの最初の確認。
2. 動くベルトの先読み時間・把持タイミングを詰める（Tier 3 の predictor 設計と接続）。
3. ドメインランダム化（cube 色・ライティング・開始位置範囲）を増やしてロバスト性を上げる。
