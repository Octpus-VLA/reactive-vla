# lerobot を editable 管理する構成

## 目的
lerobot 本体を直接編集してカスタムモデル（policy）を追加し、本リポジトリ（reactive-vla）から呼び出せるようにする。

## 決定事項（要件）

| 項目 | 決定 |
|---|---|
| lerobot の取り込み方法 | 自分の fork を **git submodule** として `third_party/lerobot` に配置 |
| fork 作成 | ユーザーが手動で作成（`Higashi-Masafumi/lerobot` を想定） |
| カスタムモデル追加方式 | **lerobot 本体を直接編集**（`src/lerobot/policies/` に追加し factory に登録） |
| パッケージ管理 | pixi（0.70.2） / platform: osx-arm64 |
| Python | conda-forge の `python 3.12`（lerobot は `requires-python>=3.12`） |
| lerobot のインストール | pixi の `pypi-dependencies` で `{ path = "third_party/lerobot", editable = true }` |
| 依存（torch 等） | lerobot の pyproject 経由で PyPI から解決 |
| conda/pypi 衝突対策 | conda が pin する `setuptools` / `packaging` が lerobot の制約と衝突するため、conda 依存で範囲指定（`setuptools>=71,<81` / `packaging>=24.2,<26.0`） |

## セットアップ手順

### 1. fork（ユーザー作業）
GitHub 上で `huggingface/lerobot` を `Higashi-Masafumi/lerobot` に fork。

### 2. submodule 追加
```bash
git submodule add https://github.com/Higashi-Masafumi/lerobot.git third_party/lerobot
cd third_party/lerobot
# upstream を追跡用に登録（後で fetch して merge）
git remote add upstream https://github.com/huggingface/lerobot.git
cd ../..
```

### 3. pixi で editable install
`pixi.toml` 設定済み。以下で環境構築：
```bash
pixi install
```

### 4. clone 時の注意
他環境で clone する際は submodule も取得：
```bash
git clone --recurse-submodules <reactive-vla-url>
# もしくは clone 後に
git submodule update --init --recursive
```

## カスタムモデルの追加方法（lerobot 直接編集）

lerobot は `draccus.ChoiceRegistry` でポリシー config を登録し、`src/lerobot/policies/factory.py` の `get_policy_class` / `get_policy_config_class` が名前から動的解決する。新モデル追加は以下：

1. `third_party/lerobot/src/lerobot/policies/<your_model>/` を作成
   - `configuration_<your_model>.py`: `PreTrainedConfig` を継承し `@PreTrainedConfig.register_subclass("<your_model>")` を付与
   - `modeling_<your_model>.py`: `PreTrainedPolicy` を継承
2. `factory.py` の `get_policy_class` / `get_policy_config_class` に分岐を追加
3. 編集は `third_party/lerobot` 内の commit → fork に push。reactive-vla 側は submodule の SHA を bump して commit

## upstream 追従
```bash
cd third_party/lerobot
git fetch upstream
git merge upstream/main   # or rebase
git push origin <branch>
cd ../..
git add third_party/lerobot && git commit -m "Bump lerobot submodule"
```
