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

### Tier 3: predictive/adaptive replan

This is the next research and implementation target. Instead of returning only "something moved", the detector estimates cube position, velocity, predicted position, replan timing, and effective horizon.

Expected detector output:

```python
DetectorOutput(
    cube_visible=True,
    cube_center_px=(x, y),
    cube_velocity_px_s=(vx, vy),
    predicted_center_px=(px, py),
    time_to_grasp_zone_s=t,
    replan_now=True,
    effective_horizon=8,
)
```

Capabilities:

- Tracks the target cube from color masks or object detection
- Predicts when it reaches the grasp zone
- Shortens the effective horizon when the cube is fast
- Uses a longer horizon when the cube is slow to preserve stability

Conceptually:

```text
slow cube -> longer effective_horizon
fast cube -> shorter effective_horizon
cube near grasp zone -> urgent replan
```

Tier 3 is not implemented in this work. The immediate goal is to establish the Tier 2 supervisor path so `MotionDetector` can later be replaced by `CubeMotionDetector`.

## Requirements Fixed On 2026-06-20

- Monitor the `overall` camera as a supervisor camera
- Trigger observation sending independently of action queue level
- Keep existing async inference behavior unchanged by default
- Use only frame difference in v1
- Keep YOLO, cube speed prediction, and dynamic horizon as future work

## Configuration Example

Enable the supervisor on the async inference client:

```bash
--supervisor_enabled=true \
--supervisor_camera=overall \
--supervisor_poll_fps=20 \
--supervisor_cooldown_s=1.0 \
--supervisor_motion_threshold=0.02
```

| Option | Meaning |
|---|---|
| `supervisor_enabled` | Enable the supervisor monitor |
| `supervisor_camera` | Camera key to watch |
| `supervisor_poll_fps` | Camera polling rate |
| `supervisor_cooldown_s` | Minimum seconds between triggers |
| `supervisor_motion_threshold` | Fraction of changed pixels required to trigger |

## Effect On Existing Behavior

`supervisor_enabled=false` by default, so existing CLI and async inference runs continue to replan only from the queue threshold. When the supervisor is enabled, event triggers are added to the queue condition.

## Test Plan

- Build the MkDocs site to check nav and content
- Import `MotionDetector` and `SupervisorMonitor`
- Confirm async inference config works with supervisor disabled
- Confirm config validation for camera supervisor parameters

## Future Work

- HSV color-mask `CubeMotionDetector`
- Cube speed estimation and grasp-zone arrival prediction
- Separate `replan_now` from `effective_horizon` through `DetectorOutput`
- Action queue flush or partial replacement
- Object detection triggers such as YOLO plus IoU
