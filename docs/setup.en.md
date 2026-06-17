# Setup

This repository pulls in `lerobot` as a git submodule at `third_party/lerobot` and uses pixi's editable install.

## 1. Fetch the submodule

```bash
git submodule update --init --recursive
```

The submodule is referenced over HTTPS (`https://github.com/Octpus-VLA/lerobot.git`), so no SSH key setup is required.

## 2. Set up the environment

```bash
pixi install
```

- [pixi.toml](https://github.com/Octpus-VLA/reactive-vla/blob/main/pixi.toml) registers `osx-arm64` / `linux-64` / `linux-aarch64` under `platforms`. If your machine uses a different architecture, add it with `pixi workspace platform add <platform>`.
- `ffmpeg` is included as a conda dependency, which is required for video decoding (`lerobot[dataset]` / torchcodec).

## 3. Lint / Format

```bash
pixi run lint   # ruff check
pixi run fmt    # ruff format
pixi run fix    # check --fix + format
```

For detailed configuration and how to add custom policies, see [lerobot Editable Setup](lerobot-editable-setup.md).
