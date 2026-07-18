# reactive-vla ドキュメント

Octpus VLA プロジェクトのドキュメントサイトです。`lerobot` をベースに、SO-101 アームを使った模倣学習（記録 → 学習 → 評価）のワークフローを扱います。

## はじめに

- [セットアップ](setup.md) — submodule の取得、pixi 環境構築、Lint / Format
- [lerobot editable 構成](lerobot-editable-setup.md) — lerobot 本体を直接編集してカスタムポリシーを追加する構成

## 使い方ガイド

- [SmolVLA ファインチューニング](smolvla-finetuning.md) — 事前学習済み SmolVLA を SO-101 データセットでお試し学習
- [RTC sim ロールアウト](rtc-sim-rollout.md) — SmolVLA + RTC 非同期ロールアウトを MuJoCo シミュレーションで動かす
- [sim スクリプトエキスパート収集](sim-scripted-collect.md) — スクリプトIKエキスパートで MuJoCo シム内の pick-and-place デモを収集する

## 研究ノート

- [非同期レイテンシ実験（sync vs RTC）](latency-experiments.md) — 同期推論と RTC 非同期推論のレイテンシを比較する
- [overhead カメラ予測器による動的 pick](overhead-predictor.md) — overhead カメラで cube を推論レイテンシ分だけ時間前進させ VLA に入力する

## リンク

- リポジトリ: [Octpus-VLA/reactive-vla](https://github.com/Octpus-VLA/reactive-vla)
- lerobot fork: [Octpus-VLA/lerobot](https://github.com/Octpus-VLA/lerobot)
