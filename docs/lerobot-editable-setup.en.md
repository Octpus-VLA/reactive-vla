# Managing lerobot as an editable install

## Goal
Edit lerobot itself to add custom models (policies) and call them from this repository (reactive-vla).

## Decisions (requirements)

| Item | Decision |
|---|---|
| How lerobot is pulled in | Place your own fork as a **git submodule** at `third_party/lerobot` |
| Creating the fork | Created manually by the user (assumed `Higashi-Masafumi/lerobot`) |
| How custom models are added | **Edit lerobot directly** (add under `src/lerobot/policies/` and register in the factory) |
| Package management | pixi (0.70.2) / platform: osx-arm64 |
| Python | conda-forge `python 3.12` (lerobot requires `requires-python>=3.12`) |
| Installing lerobot | pixi `pypi-dependencies` with `{ path = "third_party/lerobot", editable = true }` |
| Dependencies (torch etc.) | Resolved from PyPI via lerobot's pyproject |
| conda/pypi conflict workaround | conda pins `setuptools` / `packaging` that conflict with lerobot's constraints, so the conda deps are range-pinned (`setuptools>=71,<81` / `packaging>=24.2,<26.0`) |

## Setup steps

### 1. Fork (user action)
On GitHub, fork `huggingface/lerobot` into `Higashi-Masafumi/lerobot`.

### 2. Add the submodule
```bash
git submodule add https://github.com/Higashi-Masafumi/lerobot.git third_party/lerobot
cd third_party/lerobot
# register upstream for tracking (fetch & merge later)
git remote add upstream https://github.com/huggingface/lerobot.git
cd ../..
```

### 3. editable install with pixi
`pixi.toml` is already configured. Build the environment with:
```bash
pixi install
```

### 4. Note when cloning
When cloning in another environment, fetch the submodule too:
```bash
git clone --recurse-submodules <reactive-vla-url>
# or after cloning
git submodule update --init --recursive
```

## How to add a custom model (editing lerobot directly)

lerobot registers policy configs via `draccus.ChoiceRegistry`, and `get_policy_class` / `get_policy_config_class` in `src/lerobot/policies/factory.py` resolve them dynamically by name. To add a new model:

1. Create `third_party/lerobot/src/lerobot/policies/<your_model>/`
   - `configuration_<your_model>.py`: subclass `PreTrainedConfig` and decorate with `@PreTrainedConfig.register_subclass("<your_model>")`
   - `modeling_<your_model>.py`: subclass `PreTrainedPolicy`
2. Add branches to `get_policy_class` / `get_policy_config_class` in `factory.py`
3. Commit edits inside `third_party/lerobot` → push to the fork. On the reactive-vla side, bump the submodule SHA and commit.

## Tracking upstream
```bash
cd third_party/lerobot
git fetch upstream
git merge upstream/main   # or rebase
git push origin <branch>
cd ../..
git add third_party/lerobot && git commit -m "Bump lerobot submodule"
```
