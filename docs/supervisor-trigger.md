# Supervisor トリガによる動的 replan

このメモは、ベルトコンベア上を移動する cube を SO-101 で pick するための反応性を、段階的に整理するものです。ここで使う Tier 1/2/3 は LeRobot の既存 API 名ではなく、このプロジェクト内で replan 機構を説明するための設計上の分類です。

## 背景

SmolVLA のような VLA policy は、観測画像と言語指示から action chunk を生成します。通常の async inference では、action queue の残量が少なくなったタイミングで次の observation を送って replan します。

静的な pick task ではこの方式で十分ですが、ベルトコンベア上の cube は policy が古い chunk を実行している間にも移動します。queue がまだ残っている場合、cube が把持位置からずれても robot は古い chunk を実行し続けるため、反応が遅れます。

## 反応性の3階層

### Tier 1: queue-based replan

既存の baseline です。

```text
action queue の残量 <= chunk_size_threshold
-> 現在の observation を policy server に送る
-> 新しい action chunk を受け取る
```

機能:

- action chunk が尽きる前に次の chunk を補充する
- camera の変化や object の動きは見ない
- 安定しているが、動く cube への反応は queue threshold に制限される

今回の作業では Tier 1 の実装は変更しません。

### Tier 2: event-triggered early replan

今回の実装対象です。camera を supervisor として別スレッドで監視し、検出イベントが起きたら queue 残量とは独立に replan を先制発火させます。

```text
supervisor thread が camera の latest frame を読む
-> detector がイベントを検出する
-> trigger flag を立てる
-> RobotClient が queue 条件と trigger 条件を OR する
-> queue が十分残っていても observation を送る
```

replan 条件は概念的に次の形になります。

```python
should_replan = queue_below_threshold or supervisor_triggered
```

v1 の detector は frame difference による motion detector です。連続フレームを grayscale 化し、輝度差が一定以上の pixel 比率が `supervisor.detector.motion_threshold` を超えたら trigger します。

機能:

- camera の急な変化に反応して早期 replan する
- `chunk_size_threshold` は固定のまま使う
- 既存挙動を壊さないため、デフォルトは無効
- 物体の種類や速度はまだ推定しない

実装は `third_party/lerobot` 側にあります。detector 本体は **async inference 経路と RTC rollout 経路で共有**するため、`src/lerobot/detectors/` に切り出されています。

- `src/lerobot/detectors/`: `MotionDetector` / `RedCubeSpeedDetector` / `DetectorOutput` と `DetectorConfig`・`SupervisorConfig`・`make_detector`（transport 非依存の純粋ロジック）
- `src/lerobot/async_inference/supervisor.py`: `SupervisorMonitor`（async 用にカメラを別スレッドで監視する wrapper）
- `src/lerobot/async_inference/robot_client.py`: queue threshold と supervisor trigger の統合（async 経路、`config.supervisor`）
- `src/lerobot/rollout/inference/rtc.py`: RTC エンジンが制御ループの観測フレームに detector を当てて replan ゲートを動的化（RTC 経路、`--inference.supervisor`）

### Tier 3 v1: speed-adaptive replan

次の段階として、detector が単に「画面が動いた」ことを返すのではなく、赤い cube の位置と画像平面上の速度を推定し、速度に応じて replan timing を動的に調整します。

v1 では、`overall` camera の RGB frame に対して HSV の red mask を作り、mask centroid の移動量から `speed_px_s` を推定します。赤は hue の 0/360 度境界をまたぐため、red mask は両端の hue range を受け入れます。

実装済みの detector output:

```python
DetectorOutput(
    replan_now=True,
    target_visible=True,
    center_px=(x, y),
    speed_px_s=180.0,
    effective_chunk_size_threshold=0.7,
    reason="red_cube_speed",
)
```

機能:

- 赤い cube を HSV mask で追跡する
- 赤い cube が見えていない場合は `target_visible=False` を返す
- mask centroid の移動から `speed_px_s` を推定する
- cube が速いほど `effective_chunk_size_threshold` を大きくし、queue が多めに残っている段階で observation を送る
- `supervisor.detector.urgent_speed_px_s` を超えた場合は、queue 残量に関係なく即時 replan を発火する
- `supervisor.require_target_visible=true` の場合、detector が target を見るまで queue-based RTC replan を待たせる
- 既存挙動を壊さないため、デフォルトは従来の `motion` detector のまま

概念例:

```text
cube が遅い -> 低めの chunk_size_threshold で安定寄り
cube が速い -> 高めの chunk_size_threshold で早めに replan
target が見えない -> 必要時は queue-based RTC replan を待たせる
cube が urgent speed を超える -> queue 残量に関係なく urgent replan
```

ただし、v1 はまだ grasp zone への到達時刻予測や本当の dynamic action horizon 変更は行いません。まずは camera speed から replan timing を動的に変える最小閉ループです。

## 2026-06-20 時点の確定要件

- `overall` camera を supervisor 用 camera として監視できること
- 検出イベントが発生したら、queue 残量に依存せず observation 送信を発火できること
- 既存の async inference 挙動に影響しないよう、デフォルトでは無効であること
- v1 は frame difference のみを使い、YOLO や cube speed predictor は含めないこと
- Tier 3 の dynamic chunk / horizon は将来課題として明確に分離すること

## 2026-06-21 時点の Tier 3 v1 追加

- 赤い cube を HSV mask で検出する `red_cube_speed` detector を追加
- `DetectorOutput` に `center_px`、`speed_px_s`、`effective_chunk_size_threshold`、`replan_now` を持たせる
- `RobotClient` が detector output を読み、速度に応じて一時的な replan threshold を使う
- 速度が urgent threshold を超えた場合は event-triggered replan として即発火する
- dynamic horizon、queue flush、YOLO、grasp-zone 到達予測はまだ future work

## 設定例

detector は **async inference 経路（PolicyServer + RobotClient）と RTC rollout 経路の両方**で、同じ設定キーで有効化できます。違いは prefix だけです。

- async client: `--supervisor.*`
- RTC rollout: `--inference.supervisor.*`（`pixi run eval --rtc --detector ...` が最短形）

detector の種類は `--supervisor.detector.type=motion|red_cube_speed` で選び、その種類固有のパラメータは `--supervisor.detector.<field>` で渡します（draccus の choice registry。選んだ種類のフィールドだけが有効）。

motion detector（既定）を有効化（async client の例）:

```bash
--supervisor.enabled=true \
--supervisor.camera=overall \
--supervisor.poll_fps=20 \
--supervisor.cooldown_s=1.0 \
--supervisor.detector.type=motion \
--supervisor.detector.motion_threshold=0.02
```

赤い cube の速度に応じて replan timing を調整する場合:

```bash
--supervisor.enabled=true \
--supervisor.camera=overall \
--supervisor.cooldown_s=0.5 \
--supervisor.detector.type=red_cube_speed \
--supervisor.detector.slow_speed_px_s=40 \
--supervisor.detector.fast_speed_px_s=200 \
--supervisor.detector.urgent_speed_px_s=250 \
--supervisor.detector.min_chunk_size_threshold=0.25 \
--supervisor.detector.max_chunk_size_threshold=0.75
```

RTC rollout で同じことをする最短形:

```bash
pixi run eval --rtc --detector red_cube_speed --detector-camera overall \
  --policy <ckpt> --task "Grab the cube" --repo-id rollout_rtc_detector
# 細かいしきい値は passthrough: --inference.supervisor.detector.urgent_speed_px_s=300
```

front camera を target visibility gate として使う場合:

```bash
pixi run eval --rtc --detector red_cube_speed --detector-camera front --require-target-visible \
  --policy <ckpt> --task "Grab the cube" --repo-id rollout_front_gate
```

パラメータ（supervisor の配線）:

| オプション | 意味 |
|---|---|
| `supervisor.enabled` | supervisor を有効化する（既定 false） |
| `supervisor.camera` | 監視する camera key（観測の画像キーと一致必須） |
| `supervisor.poll_fps` | camera を読む頻度（**async 経路のみ**。RTC は制御ループのフレームを使うため未使用） |
| `supervisor.cooldown_s` | trigger の連発を抑える秒数 |
| `supervisor.require_target_visible` | RTC のみ: detector が `target_visible=false` を返している間は queue-based replan を抑制する |
| `supervisor.detector.type` | `motion` または `red_cube_speed` |

パラメータ（`detector.type=motion`）:

| オプション | 意味 |
|---|---|
| `supervisor.detector.motion_threshold` | frame 内で変化した pixel 比率の閾値 |

パラメータ（`detector.type=red_cube_speed`）:

| オプション | 意味 |
|---|---|
| `supervisor.detector.slow_speed_px_s` | 低い replan threshold に対応する cube 速度 |
| `supervisor.detector.fast_speed_px_s` | 高い replan threshold に対応する cube 速度 |
| `supervisor.detector.urgent_speed_px_s` | queue 残量に関係なく即時 replan する速度 |
| `supervisor.detector.min_chunk_size_threshold` | cube が遅いときの adaptive threshold |
| `supervisor.detector.max_chunk_size_threshold` | cube が速いときの adaptive threshold |
| `supervisor.detector.hue_tolerance_deg` | red mask の hue 許容幅 |
| `supervisor.detector.saturation_min` | red mask の最小 saturation |
| `supervisor.detector.value_min` | red mask の最小 value |
| `supervisor.detector.min_area_ratio` | red mask として認める最小面積比 |

> RTC 経路では `effective_chunk_size_threshold`（0–1 割合）をエンジンが `× chunk_size` で絶対 queue しきい値に変換します。async 経路では `chunk_size_threshold`（割合）として直接使われます。

## 既存挙動への影響

`supervisor.enabled=false` がデフォルトなので、既存の CLI・async inference・RTC rollout は従来どおり queue threshold だけで replan します。supervisor を有効にした場合だけ、queue threshold に event trigger（と speed-adaptive なしきい値）が追加されます。

## テスト方針

- docs build で MkDocs nav と本文を確認する
- `MotionDetector` / `SupervisorMonitor` / `RedCubeSpeedDetector` の import を確認する
- 赤い cube mask の centroid と速度推定を確認する
- 速度から adaptive threshold への mapping を確認する
- supervisor 無効時に既存 async inference config が作れることを確認する
- supervisor 有効時は camera key、poll fps、cooldown、motion threshold の config validation を確認する

## 今後の課題

- grasp zone 到達時刻の予測
- `DetectorOutput` による `replan_now` と `effective_horizon` の分離
- policy/server 側まで含む本当の dynamic horizon
- action queue の flush / partial replacement
- YOLO + IoU など object detection ベースの trigger
