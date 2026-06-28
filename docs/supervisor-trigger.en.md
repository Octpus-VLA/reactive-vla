# Supervisor-Triggered Adaptive Replan

This note organizes the reactivity needed for picking a moving cube on a conveyor belt with SO-101. The Tier 1/2/3 names are project-level design terms, not existing LeRobot API names.

## Background

VLA policies such as SmolVLA generate action chunks from camera observations and language instructions. In the regular async inference path, the client sends a new observation when the action queue falls below a threshold.

That works for static pick tasks, but a conveyor cube can move while the robot is still executing an old chunk. If the queue is not yet below threshold, the robot keeps executing stale actions and reacts late.

## Three Levels Of Reactivity

### Tier 1: queue-based replan

The existing baseline.

```text
remaining action queue <= chunk_size_threshold
-> send the current observation to the policy server
-> receive a new action chunk
```

Capabilities:

- Refills the action queue before it runs dry
- Does not inspect camera events or object motion
- Stable, but limited by the queue threshold for moving objects

This work does not change Tier 1.

### Tier 2: event-triggered early replan

This is the implementation target for the current work. A camera supervisor runs on a separate thread. When it detects an event, it triggers an early replan independently of queue level.

```text
supervisor thread reads the camera latest frame
-> detector finds an event
-> trigger flag is set
-> RobotClient ORs the queue condition with the trigger condition
-> an observation is sent even if the queue still has enough actions
```

Conceptually:

```python
should_replan = queue_below_threshold or supervisor_triggered
```

The v1 detector is a frame-difference motion detector. It converts consecutive frames to grayscale and fires when the fraction of pixels with large intensity changes exceeds `supervisor.detector.motion_threshold`.

Capabilities:

- Reacts early to visual changes
- Keeps `chunk_size_threshold` unchanged
- Is disabled by default so existing behavior is preserved
- Does not yet infer object identity or speed

The implementation lives in `third_party/lerobot`. The detectors are shared by the async-inference path and the RTC rollout path, so they live in `src/lerobot/detectors/`:

- `src/lerobot/detectors/`: `MotionDetector` / `RedCubeSpeedDetector` / `DetectorOutput` plus `DetectorConfig`, `SupervisorConfig`, and `make_detector` (pure, transport-agnostic logic)
- `src/lerobot/async_inference/supervisor.py`: `SupervisorMonitor` (async wrapper that polls a camera on its own thread)
- `src/lerobot/async_inference/robot_client.py`: queue threshold plus supervisor trigger integration (async path, `config.supervisor`)
- `src/lerobot/rollout/inference/rtc.py`: the RTC engine runs the detector on the control-loop observation frame to drive the replan gate (RTC path, `--inference.supervisor`)

### Tier 3 v1: speed-adaptive replan

The next step is to return more than "something moved". The detector estimates the red cube position and image-plane speed, then dynamically adjusts replan timing from that speed.

In v1, the `overall` camera frame is segmented with an HSV red mask, and `speed_px_s` is estimated from mask-centroid motion. Red wraps around the 0/360 degree hue boundary, so the mask accepts both ends of the hue range.

Implemented detector output:

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

Capabilities:

- Tracks the red cube with an HSV mask
- Reports `target_visible=False` when the red cube is not visible
- Estimates `speed_px_s` from mask-centroid motion
- Raises `effective_chunk_size_threshold` when the cube is fast, so observations are sent while more queued actions remain
- Fires an immediate replan when speed exceeds `supervisor.detector.urgent_speed_px_s`
- If `supervisor.require_target_visible=true`, holds queue-based RTC replanning until the detector sees the target
- Keeps existing behavior unchanged by default, because `motion` remains the default detector

Conceptually:

```text
slow cube -> lower chunk_size_threshold, stability-biased
fast cube -> higher chunk_size_threshold, earlier replan
target not visible -> hold queue-based RTC replan when required
cube exceeds urgent speed -> urgent replan independent of queue level
```

This v1 does not yet predict grasp-zone arrival or change the true action horizon. It is the smallest closed loop from camera speed to adaptive replan timing.

## Requirements Fixed On 2026-06-20

- Monitor the `overall` camera as a supervisor camera
- Trigger observation sending independently of action queue level
- Keep existing async inference behavior unchanged by default
- Use only frame difference in v1
- Keep YOLO, cube speed prediction, and dynamic horizon as future work

## Tier 3 v1 Added On 2026-06-21

- Added a `red_cube_speed` detector for red-cube HSV masking
- Added structured `DetectorOutput` fields: `center_px`, `speed_px_s`, `effective_chunk_size_threshold`, and `replan_now`
- Let `RobotClient` use detector output to choose a temporary adaptive replan threshold
- Fire an event-triggered replan when speed exceeds the urgent threshold
- Keep dynamic horizon, queue flushing, YOLO, and grasp-zone arrival prediction as future work

## Configuration Example

The detector works in **both** the async-inference path (PolicyServer + RobotClient) and the RTC rollout path, with the same config keys; only the prefix differs:

- async client: `--supervisor.*`
- RTC rollout: `--inference.supervisor.*` (shortest form: `pixi run eval --rtc --detector ...`)

Pick the detector with `--supervisor.detector.type=motion|red_cube_speed` and set its
backend-specific fields with `--supervisor.detector.<field>` (a draccus choice registry, so
only the selected backend's fields are exposed).

Enable the motion detector (default) on the async inference client:

```bash
--supervisor.enabled=true \
--supervisor.camera=overall \
--supervisor.poll_fps=20 \
--supervisor.cooldown_s=1.0 \
--supervisor.detector.type=motion \
--supervisor.detector.motion_threshold=0.02
```

Enable speed-adaptive replanning for the red cube:

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

Shortest form on the RTC rollout path:

```bash
pixi run eval --rtc --detector red_cube_speed --detector-camera overall \
  --policy <ckpt> --task "Grab the cube" --repo-id rollout_rtc_detector
# fine-tune via passthrough: --inference.supervisor.detector.urgent_speed_px_s=300
```

Use the front camera as a target-visibility gate:

```bash
pixi run eval --rtc --detector red_cube_speed --detector-camera front --require-target-visible \
  --policy <ckpt> --task "Grab the cube" --repo-id rollout_front_gate
```

Supervisor wiring:

| Option | Meaning |
|---|---|
| `supervisor.enabled` | Enable the supervisor (default false) |
| `supervisor.camera` | Camera key to watch (must match an observation image key) |
| `supervisor.poll_fps` | Camera polling rate (**async path only**; the RTC path uses the control-loop frame) |
| `supervisor.cooldown_s` | Minimum seconds between triggers |
| `supervisor.require_target_visible` | RTC only: suppress queue-based replanning while the detector reports `target_visible=false` |
| `supervisor.detector.type` | `motion` or `red_cube_speed` |

`detector.type=motion`:

| Option | Meaning |
|---|---|
| `supervisor.detector.motion_threshold` | Fraction of changed pixels required to trigger |

`detector.type=red_cube_speed`:

| Option | Meaning |
|---|---|
| `supervisor.detector.slow_speed_px_s` | Cube speed mapped to the low replan threshold |
| `supervisor.detector.fast_speed_px_s` | Cube speed mapped to the high replan threshold |
| `supervisor.detector.urgent_speed_px_s` | Cube speed that fires immediate replanning |
| `supervisor.detector.min_chunk_size_threshold` | Adaptive threshold used when the cube is slow |
| `supervisor.detector.max_chunk_size_threshold` | Adaptive threshold used when the cube is fast |
| `supervisor.detector.hue_tolerance_deg` | Hue tolerance for the red mask |
| `supervisor.detector.saturation_min` | Minimum saturation for the red mask |
| `supervisor.detector.value_min` | Minimum value for the red mask |
| `supervisor.detector.min_area_ratio` | Minimum red-mask area ratio |

> On the RTC path the engine maps `effective_chunk_size_threshold` (a 0-1 fraction) to an absolute
> queue threshold via `× chunk_size`. On the async path it is used directly as `chunk_size_threshold`.

## Effect On Existing Behavior

`supervisor.enabled=false` by default, so existing CLI, async inference, and RTC rollout runs continue to replan only from the queue threshold. When the supervisor is enabled, event triggers (and speed-adaptive thresholds) are added to the queue condition.

## Test Plan

- Build the MkDocs site to check nav and content
- Import `MotionDetector`, `SupervisorMonitor`, and `RedCubeSpeedDetector`
- Confirm red-cube mask centroid and speed estimation
- Confirm speed-to-adaptive-threshold mapping
- Confirm async inference config works with supervisor disabled
- Confirm config validation for camera supervisor parameters

## Future Work

- Grasp-zone arrival prediction
- Separate `replan_now` from `effective_horizon` through `DetectorOutput`
- True dynamic horizon across policy/server behavior
- Action queue flush or partial replacement
- Object detection triggers such as YOLO plus IoU
