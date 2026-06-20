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

v1 の detector は frame difference による motion detector です。連続フレームを grayscale 化し、輝度差が一定以上の pixel 比率が `supervisor_motion_threshold` を超えたら trigger します。

機能:

- camera の急な変化に反応して早期 replan する
- `chunk_size_threshold` は固定のまま使う
- 既存挙動を壊さないため、デフォルトは無効
- 物体の種類や速度はまだ推定しない

実装は `third_party/lerobot` 側にあります。

- `src/lerobot/async_inference/supervisor.py`: `MotionDetector` と `SupervisorMonitor`
- `src/lerobot/async_inference/configs.py`: supervisor 用 config
- `src/lerobot/async_inference/robot_client.py`: queue threshold と supervisor trigger の統合

### Tier 3: predictive/adaptive replan

次の研究・実装対象です。detector が単に「画面が動いた」ことを返すのではなく、cube の位置・速度・予測位置を推定し、replan timing と実効 horizon を動的に決めます。

想定する detector output:

```python
DetectorOutput(
    cube_visible=True,
    cube_center_px=(x, y),
    cube_velocity_px_s=(vx, vy),
    predicted_center_px=(px, py),
    time_to_grasp_zone_s=t,
    replan_now=True,
    effective_horizon=8,
)
```

機能:

- cube の色や物体検出結果から target を追跡する
- 速度から grasp zone への到達時刻を予測する
- cube が速いほど短い horizon で再推論する
- cube が遅いときは長めに chunk を実行して安定性を保つ

概念例:

```text
cube が遅い -> effective_horizon を長くする
cube が速い -> effective_horizon を短くする
cube が grasp zone に近い -> urgent replan
```

今回の作業では Tier 3 は実装しません。Tier 2 の supervisor 経路を先に立て、将来 `MotionDetector` を `CubeMotionDetector` に置き換えられるようにします。

## 2026-06-20 時点の確定要件

- `overall` camera を supervisor 用 camera として監視できること
- 検出イベントが発生したら、queue 残量に依存せず observation 送信を発火できること
- 既存の async inference 挙動に影響しないよう、デフォルトでは無効であること
- v1 は frame difference のみを使い、YOLO や cube speed predictor は含めないこと
- Tier 3 の dynamic chunk / horizon は将来課題として明確に分離すること

## 設定例

async inference client 側で supervisor を有効化します。

```bash
--supervisor_enabled=true \
--supervisor_camera=overall \
--supervisor_poll_fps=20 \
--supervisor_cooldown_s=1.0 \
--supervisor_motion_threshold=0.02
```

パラメータ:

| オプション | 意味 |
|---|---|
| `supervisor_enabled` | supervisor monitor を有効化する |
| `supervisor_camera` | 監視する camera key |
| `supervisor_poll_fps` | camera を読む頻度 |
| `supervisor_cooldown_s` | trigger の連発を抑える秒数 |
| `supervisor_motion_threshold` | frame 内で変化した pixel 比率の閾値 |

## 既存挙動への影響

`supervisor_enabled=false` がデフォルトなので、既存の CLI と async inference は従来どおり queue threshold だけで replan します。supervisor を有効にした場合だけ、queue threshold に event trigger が追加されます。

## テスト方針

- docs build で MkDocs nav と本文を確認する
- `MotionDetector` / `SupervisorMonitor` の import を確認する
- supervisor 無効時に既存 async inference config が作れることを確認する
- supervisor 有効時は camera key、poll fps、cooldown、motion threshold の config validation を確認する

## 今後の課題

- HSV color mask による `CubeMotionDetector`
- cube 速度推定と grasp zone 到達時刻の予測
- `DetectorOutput` による `replan_now` と `effective_horizon` の分離
- action queue の flush / partial replacement
- YOLO + IoU など object detection ベースの trigger
