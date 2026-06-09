# Investigation Plan: Multi-Env Cross-Contamination During Rescue

## Problem Statement

User reports: "When env0 goes out-of-fence and triggers rescue, env1 gets safety flight check fail"

**Symptom:** Rescue operation in one environment affects another environment's safety checks.

**Expected:** Each env isolated via ROS_DOMAIN_ID + GZ_PARTITION + separate PX4 instances.

---

## Phase 0: Reproduce and Capture Evidence

**Goal:** Get exact error message and timing.

### Tasks

1. **Run training with debug logging**
   ```bash
   ./run_train.sh --1  # stage 1, n_envs=2
   ```

2. **Monitor logs in parallel**
   ```bash
   # Terminal 1: main training log
   tail -f runs/train_stage0_*.log
   
   # Terminal 2: env-specific debug logs (if exist)
   tail -f runs/env_logs/env_0/*.log runs/env_logs/env_1/*.log
   ```

3. **Grep for pattern after run**
   ```bash
   grep -i "safety\|flight.*check\|fail" runs/train_stage0_*.log
   grep -A10 -B10 "RESCUE" runs/train_stage0_*.log | grep "env=1"
   ```

4. **Capture timestamps**
   - When env0 triggers rescue
   - When env1 reports safety check fail
   - Time delta between events

### Evidence Checklist

- [ ] Exact error message text
- [ ] Which component reports error (PX4 log? Python? Gazebo?)
- [ ] Env0 and env1 episode numbers when it happens
- [ ] Does env1 crash or just log warning?
- [ ] Does error repeat every time env0 rescues?

---

## Phase 1: Verify Isolation Configuration

**Goal:** Confirm each env has separate ROS domain, Gazebo partition, PX4 instance.

### Tasks

1. **Check env creation in train.py**
   - Read `train.py:61-81` — env_vars setup
   - Verify `ROS_DOMAIN_ID = 30 + rank`
   - Verify `GZ_PARTITION = drone_rl_{rank}`
   - Verify `model_name = x500_depth_{rank}`

2. **Verify ROSBridge uses correct domain**
   - Read `utils/bridge_factory.py` constructor
   - Check if ROS_DOMAIN_ID inherited from process env
   - Confirm no hardcoded domain IDs

3. **Check PX4 instance ports**
   - Read `utils/px4_manager.py:98-100`
   - Verify `UXRCE_DDS_PORT = 8888 + rank`
   - Verify `MAV_SYS_ID = 1 + rank` (if used)

4. **Inspect Gazebo world files**
   ```bash
   ls -la worlds/default_*.sdf
   # Check if world names inside files are unique per rank
   grep "world name=" worlds/default_*.sdf
   ```

### Verification

```bash
# During training, check process env vars
ps aux | grep px4 | grep -E "ROS_DOMAIN_ID|GZ_PARTITION"
ps aux | grep gz | grep -E "GZ_PARTITION"
```

Expected:
- env0: `ROS_DOMAIN_ID=30 GZ_PARTITION=drone_rl_0`
- env1: `ROS_DOMAIN_ID=31 GZ_PARTITION=drone_rl_1`

---

## Phase 2: Trace Rescue Command Flow

**Goal:** Ensure rescue velocity commands don't leak across envs.

### Tasks

1. **Read rescue implementation**
   - File: `envs/manager/reset_manager.py:951-1025`
   - Line 1019: `self.bridge.send_velocity(vx, vy, vz, 0.0)`
   - Confirm uses `self.bridge` (per-env instance)

2. **Trace send_velocity through ROSBridge**
   - File: `utils/bridge_factory.py`
   - Find `send_velocity` method
   - Check ROS topic name — does it include model_name or namespace?
   - Verify publisher uses correct ROS_DOMAIN_ID

3. **Check PX4 subscription**
   - PX4 subscribes to `/fmu/in/offboard_control_mode`, `/fmu/in/trajectory_setpoint`
   - Are these namespaced per model? (e.g., `/x500_depth_0/fmu/in/...`)
   - If not namespaced → potential cross-talk

4. **Inspect ROS topic at runtime**
   ```bash
   # During training, list active topics per domain
   ROS_DOMAIN_ID=30 ros2 topic list
   ROS_DOMAIN_ID=31 ros2 topic list
   # Should see separate topics or namespaced topics
   ```

### Evidence

- [ ] Topic names include model-specific namespace?
- [ ] Each ROS_DOMAIN_ID has separate topic list?
- [ ] Any shared topics across domains?

---

## Phase 3: Check Gazebo Physics Isolation

**Goal:** Verify drones in separate Gazebo worlds (or same world but no collision).

### Tasks

1. **Determine world architecture**
   ```bash
   # Check if using separate Gazebo servers per env
   ps aux | grep "gz sim" | wc -l
   # Should be 2 if separate servers, 1 if shared server
   ```

2. **If shared Gazebo server:**
   - Check if drones have unique model names (`x500_depth_0`, `x500_depth_1`)
   - Check if they spawn at different XY positions
   - Verify collision layers don't overlap

3. **If separate Gazebo servers:**
   - Verify `GZ_PARTITION` correctly isolates each server
   - Check Gazebo Transport isolation (each partition = separate network)

4. **Test collision isolation**
   - Manually spawn both drones at same XY position
   - Check if they collide or pass through each other
   - If collision → physics cross-talk possible during rescue

---

## Phase 4: Investigate "Safety Flight Check" Source

**Goal:** Find which component reports the error.

### Possible Sources

#### A. PX4 Commander Module
- PX4 has preflight/in-flight safety checks
- Checks: position valid, velocity sane, geofence, RC signal, etc.
- Error would appear in PX4 console log

**Check:**
```bash
# During training, tail PX4 logs
tail -f /tmp/px4_instance_0.log /tmp/px4_instance_1.log
# Look for "Commander" or "safety" or "preflight" messages
```

#### B. Python-side Safety Check
- `ResetManager` or `TrainManager` may have safety validators
- Search codebase:

```bash
grep -rn "safety.*check\|flight.*check" envs/ utils/ configs/
```

#### C. Gazebo Physics Plugin
- Some plugins enforce flight boundaries
- Check SDF files for safety plugins:

```bash
grep -i "safety\|geofence" worlds/*.sdf models/*.sdf
```

---

## Phase 5: Add Diagnostic Logging

**Goal:** Instrument code to capture exact failure sequence.

### Changes

1. **Add rescue event logging in reset_manager.py**

Before line 951 (`_rescue_to_fence_interior_by_velocity`):
```python
self.logger.info(
    f"[RESCUE START] env={self.env_id} model={self.bridge.model_name} "
    f"pos={self.bridge.get_gazebo_position()} "
    f"ROS_DOMAIN={os.environ.get('ROS_DOMAIN_ID')} "
    f"GZ_PARTITION={os.environ.get('GZ_PARTITION')}"
)
```

After line 1025:
```python
self.logger.info(
    f"[RESCUE END] env={self.env_id} success={result}"
)
```

2. **Add cross-env state logging in train_manager.py**

In `step_process()`, log if other env's bridge state changes unexpectedly:
```python
# Defensive: check if our bridge state corrupted
if hasattr(self, '_last_known_model_name'):
    if self.bridge.model_name != self._last_known_model_name:
        self.logger.error(
            f"[CROSS-ENV BUG] model_name changed! "
            f"expected={self._last_known_model_name} got={self.bridge.model_name}"
        )
self._last_known_model_name = self.bridge.model_name
```

3. **Log SubprocVecEnv barrier waits**

If using `stable-baselines3.common.vec_env.SubprocVecEnv`, it has a barrier where all envs must finish `step()` before continuing. If env0 rescue takes long, env1 blocks.

Check if timeout/deadline logic exists — if not, add it.

---

## Phase 6: Test Hypotheses

### Hypothesis A: SubprocVecEnv Barrier Timeout

**Test:**
1. Force env0 to trigger rescue (set fence very small)
2. Artificially slow rescue (add `time.sleep(10)` in rescue loop)
3. Check if env1 reports timeout or "safety check fail"

**If confirmed:** Add timeout handling in SubprocVecEnv wrapper.

---

### Hypothesis B: ROS Topic Cross-Talk

**Test:**
1. During rescue, subscribe to env1's offboard topic:
   ```bash
   ROS_DOMAIN_ID=31 ros2 topic echo /x500_depth_1/fmu/in/trajectory_setpoint
   ```
2. Check if env0's rescue commands appear on env1's topic

**If confirmed:** Fix topic namespacing in ROSBridge publisher.

---

### Hypothesis C: Gazebo Collision Cross-Talk

**Test:**
1. Teleport env0 drone to env1's position during rescue
2. Check if env1 reports collision or physics disruption

**If confirmed:** Verify GZ_PARTITION isolation or add collision filtering.

---

### Hypothesis D: Shared PX4 EKF State

**Test:**
1. Check if EKF reset in env0 affects env1
2. Monitor `/fmu/out/estimator_status` on both ROS domains during rescue

**If confirmed:** Verify UXRCE_DDS_PORT isolation or PX4_INSTANCE separation.

---

## Phase 7: Implement Fix (Depends on Root Cause)

### If Barrier Timeout:
- Add deadline to rescue (`rescue_timeout_max_s` already exists)
- Ensure rescue returns False on timeout → hard reset fallback

### If Topic Cross-Talk:
- Fix ROSBridge publisher to use namespaced topics:
  ```python
  topic = f"/{self.model_name}/fmu/in/trajectory_setpoint"
  ```

### If Gazebo Collision:
- Verify GZ_PARTITION env var set before `gz sim`
- Or spawn drones at opposite corners (env0 at (-10,-10), env1 at (10,10))

### If PX4 EKF Shared State:
- Verify `PX4_INSTANCE` env var set before PX4 launch
- Check UXRCE_DDS_PORT unique per instance

---

## Success Criteria

- [ ] Reproduce "safety flight check fail" with exact error message
- [ ] Identify which component reports error
- [ ] Confirm env0 rescue does not affect env1 state
- [ ] Run 100 episodes with multiple rescues, zero cross-env errors
- [ ] Document isolation boundary that was violated

---

## Rollback Plan

If fix introduces new issues:
1. Revert to current state (force=True already applied)
2. Disable rescue entirely: set `use_rescue_after_out_of_fence=False` in EnvConfig
3. Accept hard reset fallback for all out-of-fence events

---

## Notes

- User says this worked 1-2 days ago → recent regression
- Commit 34a9258 (Jun 6) changed configs — may have affected isolation
- Current workaround: force=True fixes arm rejection in hard reset path
- If rescue unreliable, hard reset is acceptable fallback (slower but safe)
