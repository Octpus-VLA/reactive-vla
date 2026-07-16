# スクリプトエキスパートによる sim デモ収集（ファインチューニング用）

実機データだけで学習した SmolVLA は MuJoCo レンダリング観測に対して分布外（テクスチャ・ライティングが別物）で、`sim-eval` ではほとんど動かない。これを埋めるには **観測が sim レンダリングである学習データ** が要る。本ツールはそれを生成する: 特権状態（cube の真の位置・速度）を使う pick-and-place コントローラが `SimSO101` を駆動し、各制御ステップの (観測, アクション) を `lerobot-record` と同一スキーマの `LeRobotDataset` に書き出す。出力はそのまま `pixi run train` に渡せる。

> 実装は [`cli/sim_collect.py`](https://github.com/Octpus-VLA/reactive-vla/blob/main/cli/sim_collect.py)、CLI ラッパは [`cli/so101.py`](https://github.com/Octpus-VLA/reactive-vla/blob/main/cli/so101.py) の `sim-collect`。シム本体の作りは [SmolVLA + RTC 非同期ロールアウト](rtc-sim-rollout.md) を参照。

## 特権情報とデータセットの分離（最重要）

エキスパートは特権情報（cube 姿勢・速度を MuJoCo state から直接読む、IK を解く）を使ってよいが、**データセットに残すのは実機でも観測できるもの＋（実機転送には使わない）記録専用ビューだけ**:

- **保存する**: 関節状態 `observation.state`（6 自由度）、コマンドした目標関節角 `action`（6 自由度）、および複数カメラ画像（下表）。
- **保存しない**: cube の位置・速度などの特権状態。`check_success()` の判定も録らない。

学習するポリシーは画像と関節しか見ない。お手本を作ったエキスパートが特権情報を持っていたことは、学習には漏れない。

### 記録するカメラ

`sim_collect.collect` の既定で以下を毎ステップ描画して保存する（`cameras` 引数で差し替え可能）。定義は [`assets/so101/scene_cube.xml`](https://github.com/Octpus-VLA/reactive-vla/blob/main/assets/so101/scene_cube.xml)。

| キー | MuJoCo カメラ | 用途 | ポリシー入力 |
|---|---|---|---|
| `front` | `wrist_cam` | 手首 eye-in-hand（実機 SO-101 の唯一の視覚入力） | 現行の標準ではない（下記参照） |
| `overview` | `overview`（旧 `box_top`） | 箱側からロボットを見る固定外部視点 | ✅ これのみ |

> 以前は `belt_top` / `front_high` / `corner` も記録していたが、実際には使っていなかったため削除した（`overview`＝旧 `box_top` に一本化）。

> **記録専用ビューは実機に存在しない固定外部カメラ**（手首カメラのみの実機に転送する把持ポリシーの入力には使わない）。cube 位置・速度の predictor（Tier 3）学習、配置成否の確認、可視化・解析のために残している。学習時は `--rename_map` / policy の入力カメラ指定で使うカメラだけを選べば、他は自動的に学習から除外される。`overview` は `mode="targetbody"` でロボット base を自動追尾するため `--belt-distance` を変えても画角を保つ。

> **現在の標準ワークフローは `overview` を入力カメラにする**（`front=wrist_cam` は収集データには引き続き入っているが、学習時に使わない。実機データが素の状態で使う `front` キーに揃えてあるだけで、`camera1` という名前ではなくなった）。`jobs/train/smolvla.pbs` の既定 `CAMERA=overview` が `observation.images.overview → observation.images.camera1` に rename して学習する（`sim-eval` 側も `--policy-camera` の既定が `overview`）。理由は下記「掴む向きの調整」を参照。

## 動かし方

```bash
# 静止 cube（ベルト停止）。cube は機体正面 (y=0) に置かれ、±3cm の xy ジッタで把持位置を散らす。
pixi run sim-collect --episodes 20 --repo-id sim_pickplace --task "Grab the red cube"

# 動くベルト（固定速度）。cube を -y 端から供給し、リアクティブに追従して掴む。
pixi run sim-collect --episodes 20 --repo-id sim_pickplace_belt --belt-speed 0.05

# 動くベルト（速度をエピソード毎にランダム化）。[0.03, 0.12] m/s から毎回サンプル。
pixi run sim-collect --episodes 40 --repo-id sim_pickplace_belt --belt-speed 0.03 --belt-speed-max 0.12

# 既存データセットを作り直す / 収集後に Hugging Face Hub へ上げる（要 hf-login）
pixi run sim-collect --episodes 20 --repo-id sim_pickplace --overwrite
pixi run hf-login
pixi run sim-collect --episodes 20 --repo-id <hf-user>/sim_pickplace --push
```

主なオプション（`--help` に全量）:

- `--episodes` 収集エピソード数 / `--max-steps` 1 エピソードの制御ステップ上限（低速ベルトほど cube が到達するまで長く、約 320 で 0.03 m/s までカバー。それ以下なら増やす）。
- `--belt-speed` ベルト速度 m/s（0=静止）/ `--belt-speed-max` を併用するとエピソード毎に `[--belt-speed, --belt-speed-max]` から一様サンプル（**速度を変化させた収集**）。
- `--belt-distance` 機体からベルト近縁までの距離 m。
- `--jitter` cube 開始 xy の ±ランダム化幅 m（デモを 1 姿勢に固定しないため）。**既定は小さめの値だが、標準収集では `0.01` を推奨**（下記「掴む向きの調整」参照。`0.03` だと現在の把持姿勢では到達限界に近い個体が失敗する）。
- `--seed` 乱数シード（再現可能なデータセット）/ `--fps` 制御レート（= データセット fps、学習と一致させる）。
- `--push` 収集後に Hub へアップロード（未ログイン or `local/` の id だと**実行前に**エラーで止まる）。
- レンダラは既定 `MUJOCO_GL=egl`（`sim-eval` と違い推論を挟まず GPU 競合がないので egl が安全かつ高速）。

出力は `$HF_LEROBOT_HOME/<repo_id>` に `lerobot-record` と同形式で書かれる。確認:

```bash
pixi run viz --repo-id sim_pickplace --episode 0     # Rerun で観測/状態/アクションを再生
```

## 仕組み

### 状態機械（[`PickPlaceExpert`](https://github.com/Octpus-VLA/reactive-vla/blob/main/cli/sim_collect.py)）

`approach → descend → grasp → lift → carry → place → release → done` の各相で TCP（グリッパ間の `gripperframe` site）の目標 xyz とグリッパ開閉を決める。把持後（lift 以降）の目標高さは **掴む前の cube 静止 z を固定参照** する（held 中の cube 自身の z を参照すると正のフィードバックで腕が暴走し full reach まで跳ね上がるため）。grasp は閉じ命令で 3cm の cube を物理的に挟む（グリッパは `0=閉 / 100=開` の実機スケール）。

### 閉ループ積分サーボ + ステップ制限

MuJoCo の位置アクチュエータは重力負荷で droop する（肘で約 8°、開ループの絶対角指令だと TCP が目標へ届かない）。そこで **絶対 IK 角を送らず**、TCP 誤差から Jacobian ステップを毎制御ステップ「実行中の関節指令」に積分する。指令が droop 分を超えて伸び、実 TCP が目標に到達するまで収束する（積分制御）。さらに 1 ステップの TCP 移動を上限クランプ（`IKConfig.max_tcp_step`）し、相切替で目標が大きく飛んでも腕が急振りして把持 cube を弾き飛ばさないようにしている（有界速度で滑らかに移動）。

> **`max_tcp_step` は掴む向きに依存する**。当初 `0.015` で運用していたが、掴む向きをベルト進行方向（Y軸）に回転した後（下記参照）は保持力が横滑りに弱くなり、`carry→place` の移行で cube がグリッパから滑り落ちる事例が発生（cube 位置が1ステップで11cmジャンプ＝TCPの移動量5.6cmを大きく超える＝滑っている証拠）。さらに cube 開始位置の到達限界が「非単調でカオス的」になっていた（例: x_offset=-0.0125 は成功するのに -0.005 と -0.015 は失敗、という物理的にありえない挙動）。`max_tcp_step=0.008` に下げたところ両方解消し、到達限界も綺麗な単調カットオフに戻った（速度域 0.03-0.12 で 56/56 = 100%）。掴む向きを変えたら、この値も再検証すること。

### 動くベルトのリアクティブ把持（スイートスポット待ち受け）

ベルト上の cube は等速・直線・+y で進む。固定の先読み点を一発で狙うのではなく、**毎ステップ cube の実位置を読んで追従**する（特権情報なので可能）ので、速度が事前に分からなくても、**エピソード毎に速度が変わっても**チューニング無しで対応できる:

1. **待ち受け**: グリッパを機体正面の固定スイートスポット（`grasp_y≈0`、home 姿勢で把持が一番強い位置）の上空でホバリングし、cube がベルトで運ばれてくるのを待つ。
2. **降下開始**: cube が「スイートスポットの `descend_lead_s` 手前」に来たら降下を始める（閉動作が中央付近で完了するように）。
3. **追従把持**: 降下・grasp 中も cube の実 xy を追い続ける（到達窓 `±reach_window_y` でクランプ）。cube が動いていてもグリッパが一緒に動きながら閉じるので、後ろ側でなく cube を**囲んで**掴める。

> 当初の「固定先読み点」方式は低速で失敗していた（cube が到達窓の端 y≈-0.12 = 腕が伸び切った姿勢で掴むことになり、lift で滑って落とす）。スイートスポット待ち受けにしたことで把持姿勢が全速度で一定になり解決。これは [CLAUDE.md](../CLAUDE.md) の Tier 3（predictive replan）のオフライン・特権情報版に相当する。

> **把持の瞬間に cube が減速する**のは正常な物理挙動。計測では approach/descend 中は cube はベルト速度ちょうど（例 0.08 m/s）で進み、ジャウが閉じた瞬間に速度が急減する（cube はベルト摩擦で動いていたが、ベルトより遅いグリッパに掴まれて速度を上書きされ、lift でベルト面を離れると以降はアームだけが支配するため）。現速度域（≤0.12 m/s）では把持成功に影響しないが、もっと速くする場合は把持の瞬間にグリッパを +y にベルト速度で動かす velocity matching が有効。

### 掴む向きの調整（ベルト進行方向に整列）

home キーフレームの `wrist_roll` を `0 → +90°`（[`assets/so101/scene_cube.xml`](https://github.com/Octpus-VLA/reactive-vla/blob/main/assets/so101/scene_cube.xml) の `<key name="home">`）に変更し、グリッパの jaw 開閉軸を**ベルトの長辺（cube の進行方向 = 世界 Y 軸）に整列**させた。ロボット・ベルト・箱の位置は不変、関節角のみの変更。実測: 変更前は jaw 軸がほぼ X 軸（`(0.83, 0.07, 0.55)`）、変更後はほぼ Y 軸（`(-0.10, 0.99, 0.02)`）。あわせて、見た目だけの飾りモーター（`belt_motor_box`/`belt_motor_knob`、物理には無関係）も削除した。

この回転にともない、上記の `max_tcp_step` 調整が必要になった（保持力が変わったため）。

### wrist_cam の待ち受け高さ調整（ベルトが見えるように）

`GraspConfig.approach_height`（待ち受け・持ち上げ・運搬時の TCP 高さ、cube/box 中心からの高さ）は元々 `0.10` m だったが、この高さでは手首カメラ（下向きに固定角で搭載、実機 CAD 由来なのでマウント自体は変更不可）が待ち受け中にベルトのほぼ真上を素通りして奥の市松模様の床（地平線側）を映してしまい、ベルトは画面上端にわずかに映る程度だった。`approach_height=0.06` に下げることで、待ち受け中も画面の大半にベルトが映るようになる（cube が近づくと画面いっぱいに映る）。速度 0.03-0.12 m/s・jitter 0.01 で再検証し 56/56 = 100% 維持を確認済み（機能面のコストなし）。

## 現状（実測、掴む向き調整後の最新版）

- **標準収集レシピ**（`jobs/collect/sim.pbs` の既定に反映済み）:
  ```bash
  pixi run sim-collect --episodes 150 --repo-id OctpusVLA/sim_pickplace_standard \
    --belt-speed 0.03 --belt-speed-max 0.12 --jitter 0.01 --max-steps 400
  ```
  速度域 0.03-0.12 m/s ランダム化・jitter 0.01・150 エピソードで **150/150 = 100%** 箱入れ成功（GH200）。
- 個別確認: `max_tcp_step=0.008` にした上で、速度域 0.03/0.05/0.07/0.09/0.10/0.11/0.12 各 8 回 = 56/56 = 100%、静止 cube も 4/4 = 100%。
- 生成データセットは `LeRobotDataset` で読み込め、`observation.state (6)` / `action (6)` / `observation.images.{front,overview} video` を持ち、`pixi run train` 互換（学習時は `CAMERA=overview` で overview を camera1 にリネーム）。

> 過去の実測値（掴む向き調整前・`max_tcp_step=0.015`・jitter 0.03 の版）: 静止 4/4、固定速度各 4/4、jitter=0.03 のまま速度ランダム化すると 150ep で 59% まで低下（原因が上記の掴む向き変更起因の不安定性）。調査の経緯は [docs/latency-experiments.md](latency-experiments.md) 隣接のコミット履歴も参照。

## 次の段階

1. `OctpusVLA/sim_pickplace_standard`（150ep, 速度ランダム化, jitter 0.01）で SmolVLA をファインチューニング（`pixi run train` / `jobs/train/smolvla.pbs`）→ `pixi run sim-eval` で「動くか」を検証。
2. ドメインランダム化（cube 色・ライティング・開始位置範囲）を増やしてロバスト性を上げる。
3. オンライン化（特権情報を使わず観測のみで予測する Tier 3 predictor）への接続。
