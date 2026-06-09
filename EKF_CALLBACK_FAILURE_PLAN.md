# EKF Callback Staleness Fix Plan

**Problem**: EKF `estimator_flags` callback stops publishing mid-episode → hard reset can't converge → 90s timeout → worker crash → training halt

**Root Cause Chain**:
1. Mid-episode: `est_flags_age` grows 5s → 20s → terminal `ekf_callbacks_dead`
2. Hard reset: EKF PRIME/RESET executed, but `estimator_flags` callback never resumes
3. Convergence: `[EKF SYNC]` waits for position delta < threshold, but callbacks stale → position diverges → `delta=infm`
4. ARM rejects: PX4 refuses ARM with `TEMPORARILY_REJECTED` due to stale estimator state
5. Deadline: 90s hard_reset timeout → `RuntimeError` → `EOFError` → training crash

**Evidence** (from log 06/08/26 19:12-19:16):
- Callback age grows unbounded: `0.678s → 2.051s → 6.808s → 9.123s` after reset
- EKF sync failures: `delta=0.597m → delta=1.513m → delta=infm`
- ARM rejects persist despite multiple EKF resets (counter 1→2→3→4)
- Hard reset spans 19:12:50 → 19:16:09 = 199s before deadline kill

---

## Phase 0: Investigation & Root Cause Isolation

### Objectives
1. Identify why `estimator_flags` ROS callback stops firing
2. Determine if PX4 still publishing or if ROS bridge/executor broken
3. Locate callback subscription & executor pattern in `ROSBridge`

### Tasks

**Task 0.1**: Map ROS callback subscription chain
- Read `obstacle_avoidance/utils/bridge_factory.py` — `ROSBridge.__init__` subscriptions
- Grep for `estimator_flags` subscription setup
- Identify executor pattern: which thread calls `rclpy.spin_once`?
- Check if `_spin_thread` background executor handles `estimator_flags` or if it's polled in main thread

**Task 0.2**: Review threading from INVESTIGATION_REPORT.md findings
- Finding [379]: "Dangling thread in ROSBridge.close() — _spin_stop never set, _spin_thread never joined"
- Finding [417]: "VIO state mutated by both VIO thread and main thread without synchronization"
- Finding [421]: "step_process() blocks 50-105ms per step, entirely due to bridge.tick(dt)"
- Finding [422]: "bridge.tick() can hang indefinitely if _spin_lock is held by blocked background thread"
- Check if `_spin_thread` death or `_spin_lock` deadlock could starve `estimator_flags` callback

**Task 0.3**: Determine if PX4 still publishing during failure
- Check if PX4 SITL logs show continued `estimator_status` publication timestamps
- OR: instrument `ROSBridge` to log raw ROS topic message rate (via `ros2 topic hz` equivalent)
- Distinguish: "PX4 stopped publishing" vs "ROS executor stopped spinning" vs "callback dropped"

**Verification**:
- Clear statement: callback death is due to [executor starvation | thread death | PX4 publisher halt]
- File + line number where fix should be applied

---

## Phase 1: Immediate Mitigation — Detect & Bail Early

**Goal**: Prevent 90s+ hang by detecting unrecoverable callback death within 10s

### Tasks

**Task 1.1**: Add early callback liveness check in `hard_reset_fallback_episode_reset`
- **File**: `obstacle_avoidance/envs/manager/reset_manager.py:234`
- After each EKF reset attempt, check if callbacks resumed within 5s window
- If `est_flags_age` still growing after EKF PRIME/RESET, abort attempt immediately
- Log: `[RESET ABORT] estimator_flags callback unrecoverable after EKF reset`

**Task 1.2**: Fail-fast on callback death detection
- Replace current 90s outer timeout with 3 attempts × 10s per-attempt timeout
- After 3 consecutive callback-death aborts, raise `RuntimeError` immediately (no more retries)
- Prevents 3-minute blocked reset — fail at 30s max

**Task 1.3**: Add diagnostic logging
- Log ROS executor thread state: `_spin_thread.is_alive()`
- Log lock contention: timestamp of last successful `_spin_lock` acquire
- Dump to `runs/ekf_callback_death_debug_{timestamp}.log` for post-mortem

**Verification**:
- Trigger same failure mode (manually kill `estimator_flags` publishing)
- Confirm reset aborts within 10s, not 90s
- Confirm diagnostic log captures thread/lock state

---

## Phase 2: ROS Executor Resilience — Prevent Callback Starvation

**Goal**: Fix root cause — ensure `estimator_flags` callback never starves

### Approach A: Dedicated Executor for Critical Callbacks

**Task 2A.1**: Create separate executor for `estimator_flags`
- Split `ROSBridge` into two executors:
  - `_critical_executor` (single-threaded): `estimator_flags`, `vehicle_status`, `vehicle_local_position`
  - `_bulk_executor` (existing `_spin_thread`): depth, pose, other sensors
- Run `_critical_executor.spin_once(timeout_sec=0.01)` in main `tick()` loop (never blocks > 10ms)
- Run `_bulk_executor` in background thread (can block, but doesn't affect critical callbacks)

**Task 2A.2**: Remove `_spin_lock` from critical path
- Critical executor runs in main thread → no lock needed
- Bulk executor publishes to thread-safe queue → main thread dequeues without lock

**Verification**:
- Inject 100ms delay in depth callback → verify `estimator_flags` still fires every 10ms
- Grep logs: no `[PX4 CALLBACK FRESHNESS] estimator_flags stale` warnings during 1000-step episode

### Approach B: Watchdog + Executor Restart

**Task 2B.1**: Add executor watchdog
- Track last `estimator_flags` callback timestamp
- If stale > 2s, kill `_spin_thread` and restart executor
- Log: `[ROS EXECUTOR RESTART] estimator_flags stale for {age}s, restarting spin thread`

**Task 2B.2**: Make executor restartable
- Ensure `rclpy.spin_once` doesn't corrupt state on thread interrupt
- Clear subscription queues before restart
- Re-subscribe to all topics after restart

**Verification**:
- Manually pause PX4 publication → verify watchdog restarts executor within 3s
- Verify no message loss after restart (test with counter in `estimator_flags` data)

**Decision Point**: Choose Approach A (cleaner, avoids restart complexity) unless threading model prevents it

---

## Phase 3: EKF Convergence Robustness

**Goal**: Handle case where callbacks resume but position diverged beyond recovery

### Tasks

**Task 3.1**: Add EKF reset escalation strategy
- Current: repeat same EKF PRIME/RESET on failure
- New: escalate through 3 levels:
  1. Standard EKF reset (existing)
  2. Full PX4 EKF reboot via MAVLink command `MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN` param1=3 (EKF only)
  3. Full disarm → kill PX4 process → respawn → rearm (nuclear option)

**Task 3.2**: Relax convergence threshold for stage 0
- Current: `delta < 0.1m` for 5 consecutive checks
- Stage 0: no pillars, precision less critical
- Relax to `delta < 0.5m` for stage 0, `delta < 0.2m` for stage 1+
- Prevents infinite retry on minor drift

**Task 3.3**: Detect unbounded divergence early
- If `delta` goes `NaN` or `inf`, immediately escalate to level 2 (don't retry level 1)
- If `delta` increases 3 checks in a row, abort current level

**Verification**:
- Inject position noise → verify escalation triggers level 2 within 15s
- Verify stage 0 accepts `delta=0.4m` as converged

---

## Phase 4: Hard Reset Deadline Protection (Already Implemented)

**Status**: Already added in findings [426, 427]
- Thread-based timeout wrapper prevents ROS executor starvation
- Outer 90s wall-clock deadline prevents SubprocVecEnv barrier blocking

**Tasks**:
- Verify timeout wrapper works with Phase 1 early-abort
- Ensure wrapper logs timeout reason clearly: `[TIMEOUT] EKF convergence exceeded {elapsed}s`

---

## Phase 5: Multi-Env Isolation Audit (From INVESTIGATION_REPORT.md)

**Context**: Findings [408-425] identified cross-env risks

### Tasks

**Task 5.1**: Verify ROS domain ID isolation
- Finding [409]: "ROS domain ID and topic collision risks across ranks"
- Grep for `ROS_DOMAIN_ID` assignment in `process_utils.py` or `bridge_factory.py`
- Confirm each env rank sets unique domain: `ROS_DOMAIN_ID = 30 + rank`
- Verify env vars set BEFORE rclpy.init()

**Task 5.2**: Verify Gazebo partition isolation
- Finding [408]: "Python logging not multiprocess-safe across SubprocVecEnv workers"
- Confirm `GZ_PARTITION = drone_rl_{rank}` set per env
- Check if Gazebo services scoped to partition (pose, reset, spawn)

**Task 5.3**: Fix Python logging race condition
- Finding [410]: "Python logging not multiprocess-safe"
- Replace `logging.getLogger(__name__)` with per-rank logger
- Use `logging.getLogger(f"env_{rank}")`  or configure `multiprocessing.get_logger()`

**Verification**:
- Run 4-env training, grep logs for cross-env message leakage (wrong env_id in message)
- Verify no `[Errno 98] Address already in use` errors (port collision)

---

## Phase 6: Full Integration Test

### Test Cases

**Test 6.1**: Nominal multi-env training (4 envs, 5000 steps)
- No `ekf_callbacks_dead` terminals
- No hard reset timeouts
- No `EOFError` crashes

**Test 6.2**: Injected callback failure
- Manually kill `estimator_flags` publication in one env mid-episode
- Verify: early abort within 10s, diagnostic log captured, worker doesn't crash
- Other envs continue unaffected

**Test 6.3**: Injected position divergence
- Teleport drone 10m during EKF sync window
- Verify: escalation to level 2 (EKF reboot), convergence within 30s

**Test 6.4**: Stage 0 long training (50k steps)
- Verify: no accumulation of hard reset failures over time
- Verify: `estimator_flags` callback never stale > 2s

---

## Success Criteria

1. ✅ No `ekf_callbacks_dead` terminals in 50k-step stage 0 run
2. ✅ Hard reset completes within 15s average, 30s max (not 90s+)
3. ✅ No worker crashes (`EOFError`) due to reset timeout
4. ✅ Diagnostic logs clearly identify failure mode when abort occurs
5. ✅ Multi-env training (4 envs) stable for 100k steps

---

## Appendix: Key File Locations

- **ROSBridge**: `obstacle_avoidance/utils/bridge_factory.py`
- **ResetManager**: `obstacle_avoidance/envs/manager/reset_manager.py`
- **TrainManager**: `obstacle_avoidance/envs/manager/train_manager.py`
- **Process utils**: `obstacle_avoidance/utils/process_utils.py`
- **Investigation report**: `obstacle_avoidance/INVESTIGATION_REPORT.md`
