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

The v1 detector is a frame-difference motion detector. It converts consecutive frames to grayscale and fires when the fraction of pixels with large intensity changes exceeds `supervisor_motion_threshold`.

Capabilities:

- Reacts early to visual changes
- Keeps `chunk_size_threshold` unchanged
- Is disabled by default so existing behavior is preserved
- Does not yet infer object identity or speed

The implementation lives in `third_party/lerobot`:

- `src/lerobot/async_inference/supervisor.py`: `MotionDetector` and `SupervisorMonitor`
- `src/lerobot/async_inference/configs.py`: supervisor config
- `src/lerobot/async_inference/robot_client.py`: queue threshold plus supervisor trigger integration

### Tier 3 v1: speed-adaptive replan

The next step is to return more than "something moved". The detector estimates the red cube position and image-plane speed, then dynamically adjusts replan timing from that speed.

In v1, the `overall` camera frame is segmented with an HSV red mask, and `speed_px_s` is estimated from mask-centroid motion. Red wraps around the 0/360 degree hue boundary, so the mask accepts both ends of the hue range.

Implemented detector output:

```python
DetectorOutput(
    replan_now=True,
    center_px=(x, y),
    speed_px_s=180.0,
    effective_chunk_size_threshold=0.7,
    reason="red_cube_speed",
)
```

Capabilities:

- Tracks the red cube with an HSV mask
- Estimates `speed_px_s` from mask-centroid motion
- Raises `effective_chunk_size_threshold` when the cube is fast, so observations are sent while more queued actions remain
- Fires an immediate replan when speed exceeds `supervisor_urgent_speed_px_s`
- Keeps existing behavior unchanged by default, because `motion` remains the default detector

Conceptually:

```text
slow cube -> lower chunk_size_threshold, stability-biased
fast cube -> higher chunk_size_threshold, earlier replan
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

Enable the supervisor on the async inference client:

```bash
--supervisor_enabled=true \
--supervisor_camera=overall \
--supervisor_poll_fps=20 \
--supervisor_cooldown_s=1.0 \
--supervisor_motion_threshold=0.02
```

Enable speed-adaptive replanning for the red cube:

```bash
--supervisor_enabled=true \
--supervisor_detector_type=red_cube_speed \
--supervisor_camera=overall \
--supervisor_poll_fps=20 \
--supervisor_cooldown_s=0.5 \
--supervisor_slow_speed_px_s=40 \
--supervisor_fast_speed_px_s=200 \
--supervisor_urgent_speed_px_s=250 \
--supervisor_min_chunk_size_threshold=0.25 \
--supervisor_max_chunk_size_threshold=0.75
```

| Option | Meaning |
|---|---|
| `supervisor_enabled` | Enable the supervisor monitor |
| `supervisor_camera` | Camera key to watch |
| `supervisor_poll_fps` | Camera polling rate |
| `supervisor_cooldown_s` | Minimum seconds between triggers |
| `supervisor_motion_threshold` | Fraction of changed pixels required to trigger |
| `supervisor_detector_type` | `motion` or `red_cube_speed` |
| `supervisor_slow_speed_px_s` | Cube speed mapped to the low replan threshold |
| `supervisor_fast_speed_px_s` | Cube speed mapped to the high replan threshold |
| `supervisor_urgent_speed_px_s` | Cube speed that fires immediate replanning |
| `supervisor_min_chunk_size_threshold` | Adaptive threshold used when the cube is slow |
| `supervisor_max_chunk_size_threshold` | Adaptive threshold used when the cube is fast |
| `supervisor_red_hue_tolerance_deg` | Hue tolerance for the red mask |
| `supervisor_red_saturation_min` | Minimum saturation for the red mask |
| `supervisor_red_value_min` | Minimum value for the red mask |
| `supervisor_red_min_area_ratio` | Minimum red-mask area ratio |

## Effect On Existing Behavior

`supervisor_enabled=false` by default, so existing CLI and async inference runs continue to replan only from the queue threshold. When the supervisor is enabled, event triggers are added to the queue condition.

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
