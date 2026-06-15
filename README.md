# reactive-vla
first octpus vla project repository

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

[pixi.toml](pixi.toml) の `platforms` には `osx-arm64` / `linux-64` / `linux-aarch64` を登録しています。利用するマシンのアーキテクチャがこれら以外の場合は `pixi workspace platform add <platform>` で追加してください。

### 3. Lint / Format

```bash
pixi run lint   # ruff check
pixi run fmt    # ruff format
pixi run fix    # check --fix + format
```

詳細な構成・カスタムポリシー追加手順は [docs/lerobot-editable-setup.md](docs/lerobot-editable-setup.md) を参照してください。
