# PX4 Preflight Failure After EKF Reset — Root Cause & Fix Plan

## Problem Summary

After hard reset → EKF reset sequence, PX4 arm attempts fail with `TEMPORARILY_REJECTED` even when position/velocity flags clear. Training blocks indefinitely.

## Timeline from Log (env=0, ep=13)

```
11:44:06  EPS END → rescue_then_continuous
11:44:06  [RESET] rescue failed, falling back to hard reset
11:44:09  Disarm complete, preflight=False
11:44:10  [EKF PRIME] Gazebo pose injected
11:44:11  [EKF RESET] counter incremented
          preflight=False, local_velocity_invalid + offboard_control_signal_lost
11:44:14  local_velocity_invalid + offboard_control_signal_lost persist
11:44:18  Flags clear (only global_position_invalid remains)
          preflight=False STILL persists
11:44:19  ARM attempt → TEMPORARILY_REJECTED
11:44:23  ARM retry → TEMPORARILY_REJECTED  
11:44:25  [ARM STARTUP FAIL] exhausted
```

## Root Cause Analysis

### Primary Bug: `force=False` in arm_and_takeoff()

**File:** `utils/bridge_factory.py:1836`

```python
# 3. ARM (Dùng force=True vì môi trường RL không có tín hiệu RC)
if not self.is_armed:
    self.arm(force=False)  # ← BUG: comment says force=True, code uses False
```

PX4 `pre_flight_checks_pass` includes:
- `manual_control_signal_lost` — always fails in SITL RL (no RC)
- `gcs_connection_lost` — always fails (no MAVLink GCS)
- `global_position_invalid` — GPS not fused in VIO-only mode

Without `force=True` (param2=21196.0), PX4 rejects arm for SITL safety checks irrelevant to RL.

### Secondary Issue: No Offboard Stream During Settle

After EKF reset (11:44:10), velocity spikes to [0.875, 4.75, 4.34] m/s. Between 11:44:11 and 11:44:18, `offboard_control_signal_lost` flag appears.

**Gap:** Code waits passively after EKF reset without streaming offboard setpoints. PX4 offboard timeout (~500ms) triggers `offboard_control_signal_lost`.

**Current flow:**
```
hard_reset_fallback_episode_reset()
  → disarm()
  → teleport
  → notify_ekf_pose()  [publishes VIO once]
  → sleep(2.0)  [passive wait, no setpoint stream]
  → arm_and_takeoff()
      → streams 50 setpoints
      → set OFFBOARD
      → arm(force=False)  ← fails due to preflight=False
```

## Fix Plan

### Phase 1: Fix `force=False` Bug

**File:** `utils/bridge_factory.py:1836`

Change:
```python
self.arm(force=False)
```

To:
```python
self.arm(force=True)
```

**Rationale:** Comment already states this is correct for SITL RL. PX4 SITL uses param2=21196.0 to bypass preflight checks designed for real hardware with RC/GCS.

**Verification:**
- Grep for other `arm(force=False)` calls → ensure none in reset paths
- Run training stage 1, monitor `[ARM]` logs for ACCEPTED vs REJECTED

---

### Phase 2: Stream Offboard Setpoints During EKF Settle

**Problem:** After `notify_ekf_pose()`, code sleeps 2.0s with no setpoint stream. PX4 marks offboard lost.

**Solution:** Replace passive sleep with active setpoint streaming.

**File:** `envs/manager/reset_manager.py` (or `utils/bridge_factory.py` if `notify_ekf_pose` owns settle logic)

Find the hard reset flow around `notify_ekf_pose()` + `time.sleep(2.0)`.

**Current pattern:**
```python
self.bridge.notify_ekf_pose(gz_pos_enu)
time.sleep(2.0)  # wait for EKF to settle
```

**Replacement pattern:**
```python
self.bridge.notify_ekf_pose(gz_pos_enu)
self._active_settle_after_ekf_reset(duration=2.0)

def _active_settle_after_ekf_reset(self, duration=2.0):
    """Stream zero-velocity setpoints while EKF settles."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        self.bridge.send_velocity(0.0, 0.0, 0.0, 0.0)
        self.bridge._spin_once()
```

**Rationale:**
- Prevents `offboard_control_signal_lost` flag
- Maintains offboard heartbeat during EKF settle
- Zero-velocity setpoint is safe (drone on ground, disarmed)

**Verification:**
- Grep logs for `offboard_control_signal_lost` after EKF reset
- Confirm flag no longer appears between EKF RESET and ARM attempt

---

### Phase 3: Add Preflight Wait Loop (Defensive)

Even with fix 1+2, add explicit wait for `preflight_ok=True` before arm attempt.

**File:** `utils/bridge_factory.py`, before line 1836

**Add:**
```python
# 2.5. Wait for preflight_ok (defensive: ensure EKF fully settled)
if not self.preflight_ok:
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        self.send_velocity(0.0, 0.0, 0.0, 0.0)
        self._spin_once()
        if self.preflight_ok:
            break
    if not self.preflight_ok:
        self.logger.warning(
            "[ARM] preflight_ok still False after 3s wait. "
            "Proceeding with force=True anyway."
        )
```

**Rationale:**
- Defense-in-depth: even if velocity spike clears, wait for PX4 internal preflight state
- Does not block indefinitely (3s timeout)
- With `force=True`, arm should succeed even if preflight stays False

**Verification:**
- Confirm `preflight_ok=True` before arm attempt in logs
- If timeout, confirm arm still succeeds with force flag

---

## Implementation Order

1. **Fix force=False first** (5 min, high impact)
   - Single line change
   - Unblocks training immediately
   
2. **Add offboard stream during settle** (15 min, prevents flag noise)
   - Find `notify_ekf_pose()` call site in reset manager
   - Replace sleep with active loop
   
3. **Add preflight wait loop** (10 min, defensive polish)
   - Insert before arm() call
   - Non-blocking, logs if timeout

## Success Criteria

- Zero `TEMPORARILY_REJECTED` arm failures in training logs
- Zero `offboard_control_signal_lost` between EKF reset and arm
- Hard reset completes in <20s (no multi-minute hangs)
- `preflight_ok=True` appears before arm attempt (or logged warning if not)

## Rollback Plan

If issues arise:
1. Revert `force=True` → `force=False` (restores original behavior)
2. Keep offboard stream changes (strictly improves state hygiene)
3. Remove preflight wait loop (was optional defensive measure)
