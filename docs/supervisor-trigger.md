# Supervisor トリガによる動的 replan

`overall` カメラを supervisor として監視し、検出イベントが起きた瞬間に
キュー残量に関係なく再推論（replan）を割り込み発火させる機構の設計・実装メモ。

!!! success "ステータス"
    実装済み（v1）。`third_party/lerobot`（submodule）に実装。動き検出（フレーム差分）+ CLI 引数 + 早期 replan のみ。

## 背景と課題

SO-101 は real-time action chunking（RTC）+ async inference で動かしている。
標準の LeRobot async inference では、再推論の発火タイミングは
**キュー残量ベースの固定しきい値 `chunk_size_threshold`** だけで決まる
（[`robot_client.py`](https://github.com/Octpus-VLA/reactive-vla/blob/main/third_party/lerobot/src/lerobot/async_inference/robot_client.py) の `_ready_to_send_observation()`）。

```python
self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold
```

この固定カデンスは「キューがどれだけ減ったか」で発火するため、
**世界が変化した瞬間とは無相関**。ベルトコンベアの速度急変や物体の突発的な
移動のような動的タスクでは、変化への反応が一拍遅れる。

しきい値を上げれば全区間で頻繁に replan できるが、

- replan レートは 1 チャンクの推論時間 `d` で頭打ち（`d` より速くできない）
- 一様に頻繁化するとチャンク境界が増えて mode-jumping（ガタつき）が悪化

という代償がある。そこで **定常時は既存しきい値のまま滑らかに、イベント時だけ
即発火** という非対称な制御（後述 Tier 2）を入れる。

## 反応性の 3 階層（位置づけ）

| Tier | 機構 | レイテンシ | 本機構の対象 |
|------|------|-----------|--------------|
| 1 | 毎ステップの軽量補正（replan しない） | ~1 制御ステップ | 対象外（将来） |
| 2 | **イベント先制 replan（本機構）** | ~`d` | **対象** |
| 3 | サーバ側 cancellable 推論 | 残り `d` を節約 | 対象外（将来） |

サーバは現状シングルスレッドのブロッキング推論で、生成中チャンクの中断は不可
（Tier 3 にはサーバ改造が必要）。本機構は **クライアント側だけで完結する Tier 2**。

## 決定した要件

2026-06-20 にユーザーと確定。

| 項目 | 決定 | 備考 |
|------|------|------|
| 検出ロジック（v1） | **動き検出（フレーム差分）** | 追加依存なし。`detect_fn` として差し替え可能な設計にする |
| パラメータの扱い | **CLI 引数で設定可能** | `RobotClientConfig` に追加。デフォルト無効で既存コマンドに非干渉 |
| トリガ発火時の既存 chunk | **通常通り（早期 replan のみ）** | 既存 `aggregate_fn` でマージ。flush はしない |
| 監視カメラ | `overall` | `front` は policy 入力、`overall` を supervisor に使う |
| 発火経路 | ゲートに `or supervisor.consume_trigger()` を OR | しきい値は書き換えない |

### スコープ外（将来課題）

- YOLO + IoU 等の物体検出ベース trigger（重い依存を伴うため v2 以降）
- 大擾乱時の chunk flush（RTC 凍結 prefix の温存が必要で複雑）
- Tier 1 の毎ステップ補正ヘッド / Tier 3 のサーバ側 cancellable 推論

## アーキテクチャ概要

```
overall cam ──背景スレッドで常時更新──► バッファ
                                          │ read_latest()（ノンブロッキング）
                                          ▼
                          SupervisorMonitor._loop（独立周期, 既定 20Hz）
                            detect_fn(frame) == True かつ cooldown 経過
                                          │ trigger.set()
                                          ▼
control_loop（fps）── _ready_to_send_observation()
        = (qsize/chunk <= threshold)  OR  supervisor.consume_trigger()
                                          │ True
                                          ▼
        control_loop_observation() → send_observation → サーバ推論 → 新 chunk
```

ポイント:

- supervisor は **制御ループ・policy パイプラインから独立したスレッド**で動き、
  `camera.read_latest()` で `overall` の最新フレームをノンブロッキングに覗くだけ。
  制御ループの fps 予算もカメラ HW も奪わない。
- 検出が連続しても **cooldown により 1 イベント 1 発火**に制限し、観測フラッディングを防ぐ。
- しきい値は不変なので、定常時の挙動は従来どおり。

## 実装箇所

すべて `third_party/lerobot/src/lerobot/async_inference/` 配下。

### `supervisor.py`（新規）

- `MotionDetector` — フレーム差分の動き検出器。連続フレームをグレースケール化し、
  画素強度が `PIXEL_DIFF_THRESHOLD`（=25）以上変化した画素の割合が
  `motion_area_threshold` を超えたら `True`。numpy のみで追加依存なし。
  `detect_fn: Callable[[NDArray], bool]` として差し替え可能。
- `SupervisorMonitor` — 独立スレッドで `camera.read_latest()` を `poll_fps` 周期で覗き、
  `detect_fn` が発火しかつ `cooldown_s` 経過していれば `threading.Event` を立てる。
  `consume_trigger()` は立っていれば `True` を返してクリア（1 検出 1 発火）。

### `configs.py`

`RobotClientConfig` に `supervisor_*` フィールドを追加（draccus CLI 引数として露出）。
`__post_init__` で `supervisor_enabled` 時のみ範囲バリデーション。`to_dict()` にも反映。

### `robot_client.py`

| 箇所 | 変更 |
|------|------|
| `__init__` | `supervisor_enabled` なら `SupervisorMonitor` を生成（`self.robot.cameras[camera]` を渡す）。無効時は `self.supervisor = None` |
| `_ready_to_send_observation()` | `queue_ready` を early return し、`return self.supervisor is not None and self.supervisor.consume_trigger()` を追加。**しきい値は不変** |
| `stop()` | `self.supervisor.stop()` を追加 |
| `async_client()` | receiver スレッド起動の直後に `client.supervisor.start()` |

## 検証結果

- `ruff check` / `ruff format --check`：全ファイル pass。
- `MotionDetector`：初回フレーム=False、静止=False、大きな変化=True を単体確認。
- `mkdocs build --strict`：成功（ja/en 両ビルド、本ページを nav 追加済み）。
- 注：オフラインのテスト環境は `async` extra（grpcio）未導入かつ torch/torchvision の
  ABI 不一致があり、`async_inference` パッケージのフル import は不可。これは既存コードにも
  共通する環境要因で、本変更とは無関係（`py_compile` は全ファイル通過）。実機・実行時環境での
  通し確認は下記手順で行う。

## 設定（CLI 引数）

| 引数 | 既定値 | 説明 |
|------|--------|------|
| `--supervisor_enabled` | `False` | supervisor 監視の有効化 |
| `--supervisor_camera` | `overall` | 監視に使うカメラキー |
| `--supervisor_poll_fps` | `20` | 監視ループの周期（Hz） |
| `--supervisor_cooldown_s` | `1.0` | 連続発火を防ぐ最小間隔（秒） |
| `--supervisor_motion_threshold` | `0.02` | 動きと判定する画素割合のしきい値 |

起動例（`overall` を supervisor として有効化）:

```bash
lerobot-async-client \
    ... \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --supervisor_enabled=True \
    --supervisor_camera=overall \
    --supervisor_poll_fps=20 \
    --supervisor_cooldown_s=1.0 \
    --supervisor_motion_threshold=0.02
```

## 検証手順

1. `--supervisor_enabled=True` で起動し、ログに `Supervisor monitor started` が出ることを確認。
2. `overall` の前で大きく動く → ログに `Re-inference triggered by supervisor` が出ることを確認。
3. 静止時はトリガが出ない（cooldown と threshold が機能している）ことを確認。

## 既知の限界

- 発火しても新チャンク到着まで推論遅延 `d` は残る（Tier 2 の原理的下限）。
- `d` 中に起きた変化は生成完了まで不可視（Tier 3 でしか縮められない）。

## 参考文献

- [Real-Time Execution of Action Chunking Flow Policies (RTC)](https://arxiv.org/abs/2506.07339)
- [Adaptive Action Chunking (AAC)](https://arxiv.org/abs/2604.04161)
- [Denoising Tells When to Replan](https://arxiv.org/abs/2606.03847)
- [A2C2: Leave No Observation Behind](https://arxiv.org/abs/2509.23224)
