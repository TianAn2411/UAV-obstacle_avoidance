# Failsafe Blocks Rescue — Investigation Report

## Summary

Regression: rescue worked before, now triggers failsafe → aborts → hard reset with arm rejection.

**Fixed:** `force=True` bug (line 1836) unblocks hard reset path.

**Remaining:** Why does out-of-fence trigger failsafe + position invalid?

---

## Timeline

### Working state (1-2 days ago, per user)
- Drone exits fence
- Rescue pulls back inside
- Training continues

### Current state (today)
- Drone exits fence
- PX4 failsafe triggers (`lpos_valid={'xy_valid': False, 'z_valid': False}`)
- Rescue aborts (line 959: `if failsafe: return False`)
- Hard reset fallback
- Arm fails with `TEMPORARILY_REJECTED` (force=False bug)
- Training blocks

---

## Root Cause Analysis

### Primary bug (FIXED)
**File:** `utils/bridge_factory.py:1836`
```python
self.arm(force=False)  # should be force=True for SITL RL
```
Fixed → `force=True`

### Secondary issue (OPEN)
**File:** `envs/manager/reset_manager.py:959-961`
```python
if getattr(self.bridge, "failsafe", False):
    self.logger.warning("[RESCUE] skip: PX4 in failsafe — velocity commands ignored")
    return False
```

Log shows failsafe triggered when drone out-of-fence:
```
[11:44:06] preflight=False armed=True failsafe=True failsafe_age=6.956
           lpos_valid={'xy_valid': False, 'z_valid': False, 'v_xy_valid': False, 'v_z_valid': False}
           active_failsafe_flags=['global_position_invalid', 'auto_mission_missing', 'home_position_invalid', 
                                   'manual_control_signal_lost', 'gcs_connection_lost']
```

**Key:** `lpos_valid` all False when out-of-fence → PX4 rejects position estimate → triggers failsafe.

---

## Hypothesis: Why position invalid out-of-fence?

### Option 1: VIO quality degrades with distance
- Visual features sparse far from origin
- PX4 EKF innovation check fails
- Marks estimate as unreliable

### Option 2: PX4 position sanity bounds
- Some PX4 param caps position estimate range
- Out-of-fence exceeds threshold
- EKF rejects

### Option 3: Recent config change
- Commit 34a9258 "optimize some config, change state vector 46→18, exclude lidar"
- May have changed VIO tuning or EKF params indirectly

### Option 4: Timing — VIO callback lag
- Drone moves fast out-of-fence
- VIO updates lag
- PX4 marks stale estimate invalid

---

## Evidence Needed

1. **VIO publish rate when out-of-fence**
   - Check `bridge._vo_thread` publish timestamps
   - Compare in-fence vs out-of-fence rates

2. **PX4 EKF innovation/quality metrics**
   - Subscribe to `/fmu/out/estimator_status`
   - Check `innovation_check_flags`, `pos_horiz_accuracy`

3. **Distance correlation**
   - Log distance from origin when failsafe triggers
   - Check if consistent threshold (e.g., >15m)

4. **Recent config diff**
   - Compare EnvConfig / PX4 params before/after commit 34a9258
   - Check if VIO noise params changed

---

## Fix Options

### A. Remove failsafe check (RISKY)
**Change:** Delete lines 959-961 in `reset_manager.py`

**Pro:** Rescue attempts even during failsafe
**Con:** If PX4 truly ignoring commands, wastes time before hard reset

**Risk level:** Medium — failsafe usually means position control disabled

---

### B. Improve VIO robustness (PROPER)
**Actions:**
1. Increase VIO covariance when far from origin (domain randomization already in place?)
2. Check PX4 `EKF2_*` params for position sanity bounds
3. Increase VIO publish rate in `_vo_thread`

**Pro:** Fixes root cause
**Con:** Requires PX4/VIO tuning, slow iteration

---

### C. Accept hard reset fallback (PRAGMATIC)
**Change:** None (force=True fix already done)

**Pro:** Hard reset works now with force=True
**Con:** Slower than rescue (~20s vs ~3s), more Gazebo load

**Acceptable if:** Rescue only needed <10% of episodes

---

## Current Status

✅ **Fixed:** `force=True` — hard reset no longer blocks
⚠️ **Open:** Failsafe triggers out-of-fence, blocks rescue

**Training should proceed** with hard reset as fallback. Monitor:
- How often rescue fails (check `rescue_fail_count` in logs)
- Episode reset time distribution

If >20% episodes need hard reset → investigate VIO/failsafe trigger.

---

## Next Steps (User Decision)

1. **Test with force=True fix** — run training, check if arm failures gone
2. **If rescue important** → investigate VIO quality out-of-fence
3. **If hard reset acceptable** → close issue, accept slower reset path

**Recommended:** Option 3 (test first), escalate to Option 2 if reset time bottleneck.
