# reactive-vla Documentation

Documentation site for the Octpus VLA project. Built on `lerobot`, it covers the imitation-learning workflow (record → train → eval) for the SO-101 arm.

## Getting Started

- [Setup](setup.md) — fetch the submodule, set up the pixi environment, lint / format
- [lerobot Editable Setup](lerobot-editable-setup.md) — editing lerobot directly to add custom policies

## How-to Guides

- [SmolVLA Fine-tuning](smolvla-finetuning.md) — a trial run fine-tuning the pretrained SmolVLA on a SO-101 dataset
- [RTC Sim Rollout](rtc-sim-rollout.md) — run the SmolVLA + RTC async rollout in a MuJoCo simulation
- [Sim Scripted Expert Collection](sim-scripted-collect.md) — collect MuJoCo pick-and-place demos with a scripted IK expert

## Research Notes

- [Latency Experiments (sync vs RTC)](latency-experiments.md) — compare sync vs. RTC async inference latency
- [Overhead Camera Predictor for Dynamic Pick](overhead-predictor.md) — advance the cube forward by the inference latency on the overhead camera and feed it to the VLA

## Links

- Repository: [Octpus-VLA/reactive-vla](https://github.com/Octpus-VLA/reactive-vla)
- lerobot fork: [Octpus-VLA/lerobot](https://github.com/Octpus-VLA/lerobot)
