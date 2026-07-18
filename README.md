# reactive-vla

[English](README-en.md) | 日本語

## 概要

reactive-vla は、ベルトコンベア上を動く物体を掴んで箱に入れる、といった動的なピック&プレースタスクに取り組む VLA（Vision-Language-Action）制御の研究プロジェクトです。`lerobot` をベースに、SmolVLA / pi0 のアクションチャンク方式の模倣学習を、実機（SO-101 ロボットアーム）と MuJoCo シミュレーションの両方で記録・学習・評価できる CLI/ワークフローをこのリポジトリで提供します。

研究の方向性は、3段階の再計画（reactivity）です。

- **Tier 1（実装済み）**: 既存のアクションキュー残量に基づく replan。
- **Tier 2（設計段階・未実装）**: カメラ supervisor によるイベント駆動の早期 replan。
- **Tier 3（設計段階・未実装）**: cube の位置・速度・推定把持タイミングから動的に有効ホライズンを決める予測的 replan。

現状の実装範囲と未着手の項目は [ロードマップ](#ロードマップ) にまとめています。

📖 **ドキュメント:** <https://octpus-vla.github.io/reactive-vla/> — セットアップ・SmolVLAファインチューニング・lerobot editable構成・RTC simロールアウトの手順ガイドはこちら。この README はプロジェクト概要、詳しい使い方は各ディレクトリのドキュメントという役割分担です。

## シミュレーション環境

MuJoCo 上に実機 SO-101 と同じCAD由来のモデル（DeepMind Menagerieの`robotstudio_so101`）を組み込み、コンベアベルト・cube・配置先の箱からなる動的ピック&プレースのシーンを用意しています。実機無しでポリシー評価や RTC 非同期ロールアウトを試せます。

| 外部カメラ視点（`overview`、記録用） | 手首カメラ視点（`wrist_cam`、ポリシー入力） |
|---|---|
| ![sim-eval シーン: SO-101アーム・赤いcubeを載せた緑のベルトコンベア・白い配置先の箱](docs/sim-eval-scene.png) | ![wrist_cam から見た cube とベルト](docs/sim-eval-wrist.png) |

詳細は [docs/rtc-sim-rollout.md](docs/rtc-sim-rollout.md) を参照してください。

## 機能一覧

- **SO-101 実機操作 CLI**（`cli/so101.py`、`pixi run <command>` として公開）— leader/follower アームを一度登録すれば、以降はキャリブレーション・テレオペ・データセットの記録/再生/可視化/編集・Hubへのアップロードまで行えます。詳細は [cli/README.md](cli/README.md#so-101-コマンド-pixi-run-command) のコマンド一覧を参照。
- **模倣学習ファインチューニング** — SO-101 データセットで `smolvla_base` / `pi0_base` をファインチューニング（またはスクラッチ学習）。W&Bロギング・Hugging Face Hubへのpushにも対応。詳細は [cli/README.md](cli/README.md#ファインチューニング) を参照。
- **HPCバッチ学習** — `pixi run train` をインタラクティブに実行する代わりに、PBSジョブとして投入できます。PBSスクリプト自体はキュー名・`group_list` などサイト固有の設定を含むため、このリポジトリには含めていません。[cli/README.md](cli/README.md#ファインチューニング) のテンプレートを自分のサイト向けに調整して `jobs/` 以下に置いてください（`jobs/` は `.gitignore` 済みです）。
- **MuJoCoシミュレーション** — 同梱の SO-101 モデルと `sim_so101` ロボットアダプタにより、実機無しで RTC 非同期ロールアウト経路を検証できます。詳細は [docs/rtc-sim-rollout.md](docs/rtc-sim-rollout.md) を参照。
- **シム上での成功率評価**（`pixi run sim-eval`）— 学習済みポリシーをMuJoCoシム上で実行し、タスク成功率・成功ステップ数を計測（Lift基準: cubeを持ち上げたか）。`--repo-id rollout_<name>` を付ければ動画/データセットも録画できます。詳細は [cli/README.md](cli/README.md#推論) を参照。
- **シム上でのデモ収集**（`pixi run sim-collect`）— 特権状態を使うスクリプトIKエキスパートがMuJoCoシム内で pick-and-place を実演し、(観測, アクション) を `record` と同形式の `LeRobotDataset` に書き出します。実機データで学習したポリシーは sim レンダリング観測に対して分布外（real→sim 視覚ギャップ）なので、このシム観測データでファインチューニングしてギャップを埋めるのが狙いです。詳細は [docs/sim-scripted-collect.md](docs/sim-scripted-collect.md) を参照。

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

SO-101 アーム自体の初期登録・キャリブレーション手順は [cli/README.md](cli/README.md#so-101-アームの初期登録) を参照してください。

## 詳細ドキュメント

- **CLIコマンドの全リファレンス**（記録・学習・推論の詳しいフラグ、PBSテンプレート等）→ [cli/README.md](cli/README.md)
- **手順ガイド**（セットアップ・ファインチューニング・RTC simロールアウト・レイテンシ実験など読み物形式）→ [ドキュメントサイト](https://octpus-vla.github.io/reactive-vla/)
- **SO-101 ロボット本体・シミュレーション資産**（MJCFモデル、写真、ライセンス）の詳しい説明 → [assets/so101/README.md](assets/so101/README.md)

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

## ライセンス

このリポジトリは [Apache License 2.0](LICENSE) の下で公開されています。`assets/so101` 配下の SO-101 ロボット記述（MJCF）は別配布物（同じく Apache License 2.0）です — 詳細は [assets/so101/LICENSE](assets/so101/LICENSE) を参照してください。
