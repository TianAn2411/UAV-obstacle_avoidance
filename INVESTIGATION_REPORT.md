# Investigation Report: Drone RL Multi-Environment Training

**Generated**: 2026-06-09 09:52:25
**Period**: June 6, 2026
**Total Findings**: 50

## Executive Summary

- **BUGFIX**: 7 findings
- **CHANGE**: 4 findings
- **DISCOVERY**: 31 findings
- **FEATURE**: 5 findings
- **REFACTOR**: 2 findings
- **SECURITY_NOTE**: 1 findings

---

## Discoveries

### [378] Burst VIO publishing pattern after teleport interleaves lock acquisitions

**Summary**: publish_vo_burst alternates _publish_visual_odometry and _spin_once calls, creating repeated lock chain: _gz_lock → _ros_pub_lock → _spin_lock

**Facts**:
- publish_vo_burst method iterates count times (default 10) publishing VO messages and spinning (lines 1059-1064)
- Each iteration calls _publish_visual_odometry (acquires _gz_lock then _ros_pub_lock) followed by _spin_once (acquires _spin_lock)
- Used after teleport to "prime" EKF convergence, called from notify_ekf_teleport (line 1068)
- Default burst behavior: 10 iterations at 0.01s interval per burst (lines 1059, 1062)
- With N concurrent environments, N threads may simultaneously execute publish_vo_burst, each repeatedly acquiring the same three locks in identical order

---

### [379] Thread Lock and Concurrency Issues in ROSBridge Identified

**Summary**: Three root causes documented: dangling threads, blocking lock hold, nested lock deadlock risk.

**Facts**:
- Dangling thread in ROSBridge.close() (line 610): _spin_stop never set and _spin_thread never joined, causing crashes on vectorized env reset.
- Blocking lock pattern in _spin_once (line 423): _spin_lock held during rclpy.spin_once(timeout_sec) for up to 50ms, starving other threads.
- Nested locks in _publish_visual_odometry() (lines 966/1047): _vo_publish_lock wraps both _gz_lock and _ros_pub_lock, creating deadlock risk.
- Three-phase fix plan created: Phase 1 (thread cleanup), Phase 2 (decouple blocking), Phase 3 (flatten locks).

---

### [380] Action is not included in policy observation vector

**Summary**: Raw action stored in StepState for logging but excluded from observation fed to policy

**Facts**:
- ObservationManager._build_obs() signature: (depth_raw, ekf_pos, vel, yaw, ang_vel) — no action parameter
- StepState includes action field (line 336) for reward computation and logging, not for policy input
- Observation built separately at line 363 via _build_obs() call, decoupled from action handling
- Policy receives obs from _build_obs() which contains: normalized depth sectors (36-dim), velocity (3), angular velocity (3), altitude (1), goal in body frame (3) — total 46-dim state vector
- raw_action parameter enters step_process as PPO policy output with tanh activation (already [-1, 1] range), then scaled to velocity commands by ActionManager.process()

---

### [381] 46-dim state vector architecture with sensor fusion and domain randomization

**Summary**: Observation built from noisy EKF/IMU data, LiDAR sectors, depth camera; body-centric frames for sim-to-real

**Facts**:
- State vector layout fixed at 46-dim: vel_FLU (3) + ang_vel_FLU (3) + altitude (1) + goal_body (3) + lidar_sectors (36)
- Sensor noise injection (configurable std) applied to: velocity, angular velocity, EKF position, yaw — enables sim-to-real domain randomization
- Goal vector transformed to body-FLU frame using noisy EKF position and yaw (no ground truth leakage)
- LiDAR: 270° sweep (1080 samples) divided into 36 sectors (7.5° each, 30 samples per sector); features are min-range per sector
- Depth camera: 84x84 frame normalized to [0, 10.0] with NaN/inf handling; 3 horizontal sectors yield 9 features (min/mean/free-frac per sector)
- All coordinate transformations use body-FLU frame (Forward-Left-Up) with yaw-based rotation matrix, not ENU/NED ground truth

---

### [382] Velocity action limits and state bounds in training environment

**Summary**: Max speed 8 m/s horizontal, max vertical 5 m/s, yaw rate limit 0.5 rad/s; raw angular velocity from FRD body frame

**Facts**:
- Action space: 4-dimensional [vx, vy, vz, yaw_rate] from PPO with tanh activation, scaled by ActionManager
- Hard limits: max_px4_speed = 8.0 m/s (horizontal), max_px4_vel_z_abs = 5.0 m/s (vertical), yaw_rate_limit = 0.5 rad/s
- Angular velocity sourced from PX4 sensor_combined ROS message, stored as FRD body frame rates in rad/s
- get_angular_velocity() returns NaN when EKF not ready; used raw in observation without filtering
- Sensor noise parameters calibrated to real hardware: velocity 0.05–0.15 m/s, angular velocity 0.005–0.02 rad/s, configurable in env_config

---

### [383] No observation normalization wrapper in training pipeline

**Summary**: VecNormalize not used; PPO receives raw 46-dim state vector without normalizer

**Facts**:
- Grep search for VecNormalize, normalize_obs, obs_rms, clip_obs in train.py and ppo_config.yaml returned zero matches
- Raw 46-dimensional state vector (sensor data from EKF/IMU/LiDAR) fed directly to PPO without stable-baselines3 VecNormalize or equivalent
- Policy must learn to handle natural scale of observations: velocity ≤8 m/s, angular velocity ≤0.5 rad/s, depth 0–10m, LiDAR min-range 0–30m

---

### [384] EnvConfig defines non-uniform velocity limits and multi-env fast reset infrastructure

**Summary**: Velocity limits differ per axis (vx 2.2, vy 1.8, vz_up 1.2, vz_down 0.6 m/s); multi-env fast reset disabled but fully configured

**Facts**:
- State vector 46-dim: 3 proprioception (vel) + 3 (ang_vel) + 1 (altitude) + 3 (goal body) + 36 (LiDAR sectors)
- Velocity limits are asymmetric: vx_limit=2.2 m/s, vy_limit=1.8 m/s, vz_up_limit=1.2 m/s, vz_down_limit=0.6 m/s, yaw_rate_limit=0.5 rad/s
- Observation noise injection currently disabled (all std=0.0); reference real sensor noise documented: velocity 0.05–0.15 m/s, angular velocity 0.005–0.02 rad/s, position 0.1–0.3 m
- Multi-env fast reset infrastructure fully implemented but disabled (multi_env_fast_reset_enabled=False, total_envs=1); supports idle-settle-ekf timeout sequence with per-reset-reason triggers
- Curriculum learning: goal acceptance radius shrinks 1.5→0.5m over 80k steps; goal distance annulus expands 8→16m over 350k steps with 2m minimum bandwidth
- Reset strategies: takeoff assist (80 steps), rescue protocol (4-25s timeout), collision anti-sink (0.2s duration), hard reset periodicity configurable

---

### [389] LiDAR initialization and nan/inf handling patterns in bridge_factory.py

**Summary**: LiDAR raw data initialized to 30.0 max-range default; nan/inf values replaced with safe bounds during processing.

**Facts**:
- LiDAR raw data buffer initialized as 1080-element array filled with 30.0 (max range, no obstacle)
- LiDAR callback _lidar_cb subscribes to /lidar/scan ROS2 topic
- Depth image processing uses np.nan_to_num with nan→0.0, posinf→10.0, neginf→0.0 at line 771
- Position data validated with np.isnan() checks before use (line 807)
- Invalid/missing measurements represented as float('inf') in distance calculations across multiple methods (lines 891, 896, 901, 906)

---

### [390] LiDAR callback sanitizes sensor errors with defensive binning strategy

**Summary**: Positive infinity→30.0 (clear), NaN/negative-infinity/≤0→0.1 (conservative obstacle), then clips to [0.1,30.0].

**Facts**:
- _lidar_cb converts ROS LaserScan msg.ranges to float32 array
- Positive infinity values (beam exceeded max range) replaced with 30.0 (clear space)
- NaN, negative infinity, and values ≤0 (signal error) replaced with 0.1 (conservative close obstacle)
- Final array clipped to [0.1, 30.0] range to enforce bounds
- get_lidar_scan() returns copy of lidar_raw: 1080-element float32 in meters

---

### [391] Depth camera capped at 10.0m while LiDAR capped at 30.0m — different sensor modalities

**Summary**: Depth image max-range ceiling (10.0m) differs from LiDAR (30.0m) due to sensor characteristics.

**Facts**:
- Depth image processing at line 771 uses np.nan_to_num with posinf→10.0m ceiling
- LiDAR processing uses 30.0m ceiling in callback and clip operations
- Goal max distance set to 20.0m (e.goal_xy_norm) as middle ground between sensor ranges
- Different sensor modalities have distinct max-range characteristics: camera depth ≤ 10.0m, LiDAR ≤ 30.0m

---

### [394] Front depth used in stage2 obstacle visibility and slowdown rewards

**Summary**: Reward manager checks front_depth against visibility bounds and slowdown thresholds for obstacle avoidance.

**Facts**:
- reward_manager.py line 186 stores previous minimum depth state via self._prev_min_depth = state.min_depth
- reward_manager.py line 502 validates front_depth within stage2_obstacle_visible_depth_min/max range for visibility reward
- reward_manager.py line 516 applies slowdown penalty when front_depth drops below stage2_obstacle_slowdown_depth_thresh

---

### [395] Obstacle visibility and slowdown reward conditions with multi-gate validation

**Summary**: Both rewards require front_depth threshold checks plus additional safety gates: progress toward goal, pillar distance, speed conditions.

**Facts**:
- Visibility reward (lines 497-509) requires stage >= 2, front_depth in [visible_depth_min, visible_depth_max], pillar distance > safe_distance, and xy-progress > gate threshold
- Slowdown reward (lines 511-524) requires front_depth below threshold AND nearest_pillar_dist below threshold AND speed_drop > gate, all during forward progress with speed > 0.4 m/s and positive goal-direction velocity
- Both rewards use compound boolean gates to prevent triggering on incomplete obstacle scenarios; visibility gate checks depth range is safe, slowdown gate confirms drone is both close to obstacle (front_depth) and close to pillar (nearest_pillar_dist)

---

### [396] Depth array shape mismatch in drone RL environment reset

**Summary**: PPO training crashed when stacking depth observations with inconsistent shapes during environment initialization.

**Facts**:
- ValueError raised in train_manager.py line 514 during np.stack() on depth_history arrays with mismatched shapes
- ARM system failed to arm before environment reset, indicating incomplete drone initialization
- Error occurred in subprocess during vectorized environment reset in stable_baselines3 training loop
- Training checkpoint saved to ppo_drone_stage0_interrupted.zip before subprocess crashed with EOFError
- Broken pipe error on environment close suggests subprocess terminated due to uncaught exception

---

### [397] Root cause identified: depth frame shape inconsistency in history initialization

**Summary**: _reset_depth_history fills deque with copies of normalized depth, but get_depth_84() may return different shapes.

**Facts**:
- _depth_history initialized as deque with maxlen=ecfg.depth_stack_size at line 73
- _reset_depth_history() (line 657) calls self._normalize_depth_frame(self.bridge.get_depth_84()) once and replicates that frame depth_stack_size times
- During _build_obs (line 511-514), depth_history is reset if empty, then a new depth frame is appended before stacking
- np.stack at line 514 fails if deque contains arrays with different shapes, indicating get_depth_84() returned different dimensions on subsequent calls
- reset() at line 185 calls _reset_depth_history, then later _build_obs may call it again (line 512) if empty check triggers

---

### [398] Race condition in reset sequence: depth history filled then immediately overwritten

**Summary**: _reset_depth_history called once, then _build_obs called with a fresh get_depth_84 call that may return different shape.

**Facts**:
- Line 185: _reset_depth_history() fills deque with copies of depth frame A via get_depth_84()
- Line 188-193: _build_obs() called immediately after with a NEW get_depth_84() call, potentially returning frame B
- Inside _build_obs at line 513, new depth frame is appended to the deque without validation
- At line 514, np.stack() called on deque containing mixed frames (copies of frame A + new frame B)
- If get_depth_84() returns different dimensions on successive calls, deque will contain shape-mismatched arrays

---

### [402] Current curriculum-based policy lacks CNN freezing mechanism for early stages

**Summary**: Stages 0 and 1 train CNN on depth despite user intent to learn only from state vector.

**Facts**:
- ppo_config.yaml defines 6 curriculum stages with env_overrides (num_pillars, freeze_vz, noise params) but no CNN freezing control.
- DepthStateExtractor in envs/policy.py combines CNN-processed depth features with state_fc features; no require_grad masking exists.
- Stages 0 and 1 have num_pillars=0 and freeze_vz=true (horizontal navigation only) but CNN still trains on depth input.
- User goal: learn only from state_dim vector in stages 0 and 1 while keeping CNN weights frozen until stage 2+.

---

### [403] CNN freezing not implemented in train.py training pipeline

**Summary**: Grep for "freeze" finds only freeze_vz (environment constraint), no CNN-level require_grad control.

**Facts**:
- train.py iterates through ppo_cfg["curriculum"] entries to load stage-specific config (line 273).
- policy_kwargs dict assembled at line 390 and passed to model initialization (line 455).
- Grep search for "freeze" keyword in train.py returns only "freeze_vz" references (environment-level vz velocity constraint).
- No require_grad masking, CNN freezing, or per-stage feature extractor disabling found in visible training code paths.

---

### [404] Stage transfer uses learning_rate reduction but no CNN parameter freezing

**Summary**: Model loading chain has stage-specific lr adjustment but policy_kwargs remains static across stages.

**Facts**:
- policy_kwargs dict (lines 390–393) specifies DepthStateExtractor with static features_dim=256; no stage-dependent variation.
- Stage transfer logic (line 442) loads previous stage's final model with reduced learning_rate=1e-4 for stage > 1.
- Model loading priority chain: interrupted > latest_ckpt > final_model > prev_stage_transfer > fresh (lines 404–457).
- Fresh model initialization (lines 449–458) uses "MultiInputPolicy" with fixed policy_kwargs regardless of curriculum stage.

---

### [408] Multi-env process isolation audit: environment variable and port offset strategy

**Summary**: Audited rank-based environment variable assignment for ROS domain IDs, Gazebo partitions, PX4 instances, and port offsets across SubprocVecEnv workers.

**Facts**:
- ROS_DOMAIN_ID formula: 30 + rank; set at train.py:79, px4_manager.py:98, process_utils.py:17,93,124,160,197
- GZ_PARTITION formula: "drone_rl_{rank}"; set at train.py:81, px4_manager.py:100, process_utils.py:94,125,161,198
- UXRCE_DDS_PORT formula: 8888 + rank; set at train.py:132-133, process_utils.py:19,27
- PX4_INSTANCE formula: rank integer; set at train.py:80, px4_manager.py:99; PX4 binary invoked with -i flag at px4_manager.py:147
- Each rank creates isolated rootfs directory at build/px4_sitl_default/rootfs/{rank}/ with os.makedirs at train.py:87
- World SDF file per rank: default_{rank}.sdf; no cross-rank write collision but writes occur synchronously before PX4 launch at train.py:104-117
- SubprocVecEnv rank closure captured in make_env() list comprehension at train.py:384-387; rank baked into each worker's environment

---

### [410] Python logging not multiprocess-safe across SubprocVecEnv workers

**Summary**: Parent and child worker processes share same logger instance; output from multiple ranks can interleave in single log file.

**Facts**:
- Logger created in parent at train.py:295-298 with FileHandler to shared log file train_stage{N}_{ts}.log
- Python logging module uses GIL-protected lock per handler within process but not multiprocess-safe across SubprocVecEnv workers
- Per-env step logs are rank-isolated at env_logs/env_{rank}/stage{N}_{ts}/ but main training log is not
- If child processes log via inherited logger, output interleaves with parent and other children in same log file

---

### [411] Configuration fields multi_env_fast_reset_enabled and total_envs never populated from n_envs

**Summary**: EnvConfig fields defined with default values but never set from actual n_envs argument; multi_env_fast reset path unreachable with default False.

**Facts**:
- multi_env_fast_reset_enabled dataclass default: False at env_config.py:109
- total_envs dataclass default: 1 at env_config.py:110
- make_env() receives n_envs as parameter at train.py:385 but never writes to ecfg.total_envs at train.py:237-241
- stage_conf curriculum dict does not populate multi_env_fast_reset_enabled or total_envs based on ppo_config.yaml
- ResetManager.classify_reset() consumes multi_env_fast_reset_enabled; fast reset path unreachable with default False

---

### [412] Startup stagger disabled: STARTUP_STAGGER_S set to 0

**Summary**: Inter-rank startup stagger is effectively disabled; all ranks begin Gazebo/PX4 initialization concurrently despite commented intent for 15-20s spacing.

**Facts**:
- STARTUP_STAGGER_S = 0 at train.py:67 with comment "Gazebo needs ~15-20s"
- Stagger loop at train.py:68-74 calls time.sleep(rank * STARTUP_STAGGER_S) for all ranks > 0; with stagger=0, sleep is 0 seconds
- Log message still fires for rank > 0 and total_envs > 1 (train.py:71-74) even when no actual sleep occurs
- Separate within-rank startup_sleep=18.0 at train.py:212 is a different mechanism: per-rank wait inside PX4InstanceManager.start() for Gazebo/PX4 readiness

---

### [413] Teleport drone implementation uses GZ transport API with CLI subprocess fallback

**Summary**: teleport_drone method at line 1702 validates poses via both transport API and CLI subprocess with wall-clock timing.

**Facts**:
- teleport_drone method exists at line 1702 in bridge_factory.py
- Implementation attempts GZ transport API first (self.gz_client.set_pose) before falling back to subprocess.run with gz service CLI command
- Pose validation uses wait_for_gazebo_pose with timeout=3.0, tolerance=0.5, and wall-clock time tracking (min_stamp=call_time)
- Subprocess calls for gz service at lines 1775 and 1868 (and 2248) use timeout=5.0 and capture_output=True
- Bridge uses wall-clock monotonic time tracking via self._last_status_wall, self._last_local_pos_wall, self._last_estimator_flags_wall for callback freshness validation
- ROS2 tick method (line 1933) uses rclpy.spin_once with timeout_sec management and tracks spin overruns with wall-clock monitoring

---

### [417] Lock contention on _spin_lock and _ros_pub_lock during concurrent thread operations

**Summary**: _spin_lock contended by ROSSpin thread (20Hz), main tick() busy-spin, and direct _spin_once calls; _ros_pub_lock contended by VIO (30Hz), keepalive (20Hz), and main thread.

**Facts**:
- _spin_lock (line 155) acquired at line 422 in _spin_once() which calls rclpy.spin_once(timeout_sec=0.0)
- Background ROSSpin_* thread (line 417, daemon=True) calls _spin_once at 20Hz
- Main thread tick() method (lines 2163-2189) busy-spins in tight loop calling _spin_once() with no sleep, holds _spin_lock on every iteration
- Main thread also calls _spin_once() directly in arm_and_takeoff, wait_until_armed, and other blocking wait methods
- _ros_pub_lock (line 312) acquired at lines 1056, 1645, 1688 for publish() calls
- VIO thread (line 218, 30Hz) calls _publish_visual_odometry which acquires _ros_pub_lock at line 1056
- Keepalive thread (line 574, 20Hz) calls _publish_velocity_setpoint (line 515) and _publish_position_setpoint_ned (lines 530, 556) which acquire _ros_pub_lock
- Main thread calls send_velocity() and send_position_setpoint_ned() which also acquire _ros_pub_lock

---

### [418] tick() method busy-spins without sleep between executor calls

**Summary**: Lines 2163-2189: tick() spins in tight while loop with no time.sleep, acquires _spin_lock on every iteration.

**Facts**:
- tick(dt, max_wall_s=None) at line 2163 starts wall-clock timer and spins until end_time is reached
- Inner loop (lines 2176-2189) has no sleep between _spin_once() calls
- Each _spin_once() acquires _spin_lock, calls rclpy.spin_once(timeout_sec=0.0), and releases
- max_wall_s defaults to min(dt + 0.05, 0.20) — for dt=0.02 the ceiling is 0.07s, for dt=0.05 the ceiling is 0.10s
- If wall-clock time exceeds max_wall_s, logs warning at line 2181 and breaks
- tick() is called during reset and initialization sequences and in main control loop

---

### [419] os.environ mutation in spawn_random_field creates race condition

**Summary**: Lines 2724, 2747: Spawner.spawn_random_field mutates process-global os.environ["GZ_PARTITION"] without synchronization.

**Facts**:
- spawn_random_field (line 2689) temporarily sets os.environ["GZ_PARTITION"] = self.gz_partition at line 2724
- After spawning field, restores original GZ_PARTITION in finally block at line 2747
- os.environ is a process-global dictionary shared by all threads
- No lock serializes this mutation; if multiple Spawner instances run concurrently, they will race to set/restore GZ_PARTITION
- Between line 2724 and 2747, any other thread that reads os.environ["GZ_PARTITION"] will see the temporarily modified value

---

### [420] Comment claims _vo_publish_lock serializes VIO state but lock does not exist

**Summary**: Line 352 docstring references _vo_publish_lock for serialization between VIO thread and burst callers, but _vo_publish_lock is never defined.

**Facts**:
- Line 352 docstring states: "Serialized with explicit burst callers via _vo_publish_lock"
- No _vo_publish_lock is defined anywhere in bridge_factory.py
- Only four locks exist: _gz_lock (line 138), _spin_lock (line 155), _last_setpoint_lock (line 311), _ros_pub_lock (line 312)
- The VIO fields referenced in that docstring (_last_gz_pos_for_vo_vel, _teleport_zero_vel_countdown, _teleport_reset_countdown, _vo_reset_counter) are not protected by any lock
- This is documentation mismatch: the comment claims synchronization that does not exist in the code

---

### [421] step_process() blocks 50-105ms per step, entirely due to bridge.tick(dt) call

**Summary**: step_process executes in sequence: failsafe checks, send_velocity, tick(dt=0.05), six get_* calls, geometry updates, terminal check, reward. Total: ~50-105ms.

**Facts**:
- step_process() at train_manager.py:214 follows strict sequence with timing breakdown
- Failsafe checks, action processing, reward calculation are negligible (&lt;1ms each)
- send_velocity() acquires _last_setpoint_lock and _ros_pub_lock, publishes two ROS messages; ~0.1-1ms typical
- bridge.tick(self.ecfg.dt) at train_manager.py:269 with typical dt=0.05s dominates wall-clock time
- tick() loops calling _spin_once() until dt seconds elapse, with max_wall_s guard of min(dt+0.05, 0.20) seconds
- tick() contributes approximately dt to dt+0.05 seconds wall-clock time; for dt=0.05, contributes 50-100ms
- No additional sleeps or blocking I/O in step_process() itself beyond tick()
- Observation queue, pillar geometry, collision detection are in-memory operations with no I/O

---

### [423] hard_reset_fallback_episode_reset can block up to 300 seconds with no outer timeout

**Summary**: 6-attempt loop with ~50s per attempt; each attempt includes up to 19s arm_and_takeoff + teleport + EKF convergence waits.

**Facts**:
- hard_reset_fallback_episode_reset at reset_manager.py:210 runs outer loop with up to 6 attempts
- No total wall-clock deadline on the outer loop; only per-step timeouts exist
- Each attempt includes: land/disarm (1-2s), idle (2-2.5s), teleport (8.5-22s), pose wait (3.1s), VIO prime (1s), PX4 callbacks wait (3s), settle wait (config-bounded), EKF convergence (3-9s), arm sanity (2s), idle (1.5-2s), arm_and_takeoff (up to 19s)
- Worst case per attempt: ~45-55 seconds
- Six attempts × 50s = 300 seconds maximum before raising RuntimeError at reset_manager.py:249
- startup_arm mode also unbounded: 15s gz_pose_ready wait + up to 57s from 3-attempt arm_with_startup_retry with 2s idle between attempts
- SubprocVecEnv has no timeout enforced on individual reset() calls; if reset() blocks 300s, subprocess is completely unresponsive

---

### [424] Three threads now contend for _ros_pub_lock; old code had at most two

**Summary**: Main thread, VIO thread (30Hz), and keepalive thread (20Hz) all acquire _ros_pub_lock on every step. Old code had no VIO thread.

**Facts**:
- New code added _vo_thread at bridge_factory.py:340-345 which calls _publish_visual_odometry at 30Hz (line 335)
- _publish_visual_odometry acquires _ros_pub_lock at bridge_factory.py:1056 to call vo_pub.publish(msg)
- Keepalive thread at bridge_factory.py:574 calls _publish_velocity_setpoint at lines 515, 530, 556 which acquire _ros_pub_lock at line 1645
- Main thread send_velocity calls _publish_velocity_setpoint (line 1645) acquiring _ros_pub_lock
- Main thread send_position_setpoint_ned calls _publish_position_setpoint_ned (line 1688) acquiring _ros_pub_lock
- Old code at obstacle_avoidance_mission/envs/bridge_factory.py had no background VIO thread; VO publishing was synchronous, no lock contention on publish
- Three concurrent callers (main, vo_thread, keepalive_thread) vs two in old code (main, keepalive_thread) or one if vo was sync

---

### [425] New background _spin_thread (20Hz) introduced _spin_lock contention not present in old code

**Summary**: Old code had no background spin thread; main thread called rclpy.spin_once directly with no lock. New code adds lock contention on every tick().

**Facts**:
- New code added _start_ros_spin_thread() at bridge_factory.py:372 which creates _spin_thread (daemon=True) at line 417
- _spin_thread calls _spin_once() in loop at 20Hz (interval 0.05s from line 414)
- _spin_once() at line 421 acquires _spin_lock, calls rclpy.spin_once(self, timeout_sec=0.0), releases
- Old code at obstacle_avoidance_mission/envs/bridge_factory.py:1954 called rclpy.spin_once(self, timeout_sec=...) directly on main thread with no _spin_lock
- Old code had no _spin_lock, no _spin_thread, no background spinning
- Main thread's bridge.tick() now must compete with background _spin_thread for _spin_lock on every _spin_once() call inside tick()

---
## Bug Fixes

### [386] Fixed yaw rate sign inversion in angular velocity frame conversion

**Summary**: Corrected double negation: -float(-ang_vel[2]) → -float(ang_vel[2]) in FRD→FLU conversion

**Facts**:
- Fixed line 576 in _build_state_vector(): yaw_rate now correctly inverted from FRD to FLU frame
- Double negation bug caused yaw_rate to have wrong sign, policy would learn inverted yaw control
- FRD→FLU frame conversion now correct: roll unchanged, pitch negated, yaw negated (not double-negated)
- Bug affected all episodes during training; policy would develop inverted yaw response

---

### [414] Found undefined variable bug: cur_alt used before definition in keepalive stale hover log

**Summary**: Line 521 logs cur_alt but variable is only defined at line 561, causing NameError at runtime.

**Facts**:
- Line 521 in /home/sw_an/PX4-Autopilot/obstacle_avoidance/utils/bridge_factory.py contains active log statement: `f"age={age:.3f}s mode={mode} cur_alt={cur_alt:.2f}m"`
- Variable cur_alt is not defined until line 561: `cur_alt = -cur_z_ned if np.isfinite(cur_z_ned) else 0.0`
- Original altitude safety logic (lines 483-516) is commented out, including cur_alt calculation and Z-safety checks
- New code path at line 561 attempts to define cur_alt but only after it's referenced in logging at line 521
- Keepalive stale hover path (mode == "velocity") will crash with NameError when logging threshold at line 521

---

### [415] Spawner class calls undefined _gz_spin_wrap method

**Summary**: Spawner.spawn_pillar, move_pillar, move_pillars_batch, _scene_entity_names all call self._gz_spin_wrap which is not defined on Spawner class.

**Facts**:
- Spawner class at line 2389 does not inherit from ROSBridge and does not define _gz_spin_wrap method
- Spawner.spawn_pillar (line 2407) calls self._gz_spin_wrap with max_wait_s=18.0
- Spawner.move_pillar (line 2422) calls self._gz_spin_wrap with max_wait_s=4.0
- Spawner.move_pillars_batch (line 2446) calls self._gz_spin_wrap with variable max_wait_s
- Spawner._scene_entity_names (lines 2458, 2491) calls self._gz_spin_wrap with variable max_wait_s
- _gz_spin_wrap is defined only on ROSBridge class at line 1898
- Runtime error will be AttributeError: 'Spawner' object has no attribute '_gz_spin_wrap'

---

### [416] VIO state mutated by both VIO thread and main thread without synchronization

**Summary**: _last_gz_pos_for_vo_vel, _teleport_zero_vel_countdown, _teleport_reset_countdown, _vo_reset_counter shared between threads with no lock.

**Facts**:
- _publish_visual_odometry (VIO thread, 30Hz) mutates _last_gz_pos_for_vo_vel at line 1004, _teleport_zero_vel_countdown at line 1008, _teleport_reset_countdown at line 1018, _vo_reset_counter at line 1020
- notify_ekf_teleport (main thread) mutates _teleport_zero_vel_countdown at line 1175 and _teleport_reset_countdown at line 1176
- prime_visual_odometry_after_reset (main thread) mutates _last_gz_pos_for_vo_vel at line 1228 and _vo_reset_counter at line 1230
- No lock guards these four fields; only _gz_lock and _ros_pub_lock exist
- These are non-atomic read-modify-write operations in Python, protected only by GIL which is not sufficient for multi-threaded safety
- Line 352 falsely claims these fields are \\\"Serialized with explicit burst callers via _vo_publish_lock\\\" but _vo_publish_lock does not exist

---

### [422] bridge.tick() can hang indefinitely if _spin_lock is held by blocked background thread

**Summary**: tick() calls _spin_once() which acquires _spin_lock with NO TIMEOUT; if spin_thread is stalled, main thread hangs forever.

**Facts**:
- tick() at bridge_factory.py:2163 loops calling _spin_once() until wall time end_time is reached
- _spin_once() at bridge_factory.py:421 acquires _spin_lock with `with self._spin_lock:` (blocking.acquire() with no timeout)
- threading.Lock.acquire() blocks indefinitely if lock holder does not release
- Background _spin_thread at bridge_factory.py:417 (daemon=True) calls _spin_once() at 20Hz (interval 0.05s)
- If _spin_thread is stalled inside rclpy.spin_once (backed-up ROS queue) or any other blocking call, it holds _spin_lock
- Main thread's tick() call will block indefinitely waiting for _spin_lock.acquire()
- The max_wall_s guard at bridge_factory.py:2179 cannot fire because it only checks AFTER _spin_once() returns; if _spin_once() never returns, guard never runs
- Old code at obstacle_avoidance_mission/envs/bridge_factory.py:1954 called rclpy.spin_once directly on main thread with no lock, no contention hazard

---

### [426] Thread-based timeout wrapper to prevent ROS executor starvation during multi-env Gazebo RPCs

**Summary**: Added _gz_spin_wrap method to decouple blocking gz transport calls from ROS executor thread.

**Facts**:
- ROSBridge._gz_spin_wrap runs any function in a background thread with wall-clock timeout (default 3s)
- Executor continues spinning during fn execution, preventing PX4 callbacks and VIO thread from timing out
- Spawner class also implements _gz_spin_wrap with identical pattern for consistency
- Timeout returns None instead of blocking; exceptions are logged but re-raised to caller
- Used by teleport_drone, spawn_pillar, move_pillar, and batch entity operations

---

### [427] Outer wall-clock deadline for hard reset to prevent SubprocVecEnv barrier blocking

**Summary**: Hard reset now fails with RuntimeError if exceeds 90s, preventing one env from stalling entire multi-env training loop.

**Facts**:
- Added class constant _HARD_RESET_MAX_S = 90.0 to ResetManager
- hard_reset_fallback_episode_reset computes wall_deadline = time.monotonic() + 90.0 at start
- Deadline check inserted at top of attempt loop, raises RuntimeError before retrying if exceeded
- Error message includes attempt count, deadline, and original failure reason
- 90s limit covers full reset sequence: disarm → teleport → EKF convergence → arm → lift

---

## Features

### [387] Implemented soft observation normalization in state vector

**Summary**: Added element-wise normalization by physical limits to standardize feature scales for policy learning

**Facts**:
- Velocity normalized: vel_flu / [vx_limit, vy_limit, vz_up_limit] = [2.2, 1.8, 1.2] m/s
- Angular velocity normalized: ang_vel_flu / [π, π, yaw_rate_limit] = [π, π, 0.5] rad/s
- Altitude normalized: ekf_pos[2] / fence_z_max = 8.0 m
- Goal position normalized: goal_body / [goal_xy_norm, goal_xy_norm, goal_z_norm] = [20.0, 20.0, 8.0] m
- LiDAR ranges normalized: lidar_features / lidar_max_range = 30.0 m
- Soft normalization (no hard clipping) preserves gradient signals for values exceeding physical bounds
- All normalization divisors sourced from EnvConfig, making tuning independent of code changes

---

### [399] Added quaternion observation noise parameter to EnvConfig

**Summary**: Enables sim-to-real training with quaternion noise injection for IMU simulation

**Facts**:
- obs_noise_quat_std parameter added to EnvConfig dataclass in configs/env_config.py
- Parameter controls per-component quaternion noise with subsequent renormalization
- Default value 0.0 (disabled); typical real IMU quaternion noise ranges 0.01–0.03
- Part of sim-to-real observation noise pipeline applied during state vector construction in train_manager.py

---

### [400] Implemented quaternion noise injection in state vector construction

**Summary**: Adds Gaussian noise to quaternion observations during policy training when enabled

**Facts**:
- Quaternion noise injection added in _build_state_vector method of TrainManager class
- Gaussian noise sampled per-component with magnitude from obs_noise_quat_std config parameter
- Quaternion renormalized via L2 norm division after noise injection to maintain unit vector property
- Noise application is conditional: only when ecfg.obs_noise_quat_std > 0.0 (disabled by default)
- Applied to policy observation in _build_state_vector before state concatenation at line 614

---

### [406] Add freeze_cnn configuration flag to curriculum stages 0 and 1

**Summary**: New ppo_config.yaml parameter enables per-stage CNN parameter freezing in early training.

**Facts**:
- Added `freeze_cnn: true` to stage 0 with comment "CNN branch frozen, train state MLP only" (line 34).
- Added `freeze_cnn: true` to stage 1 with comment "CNN branch frozen, train state MLP only" (line 43).
- Stages 2–5 remain unchanged; CNN trains normally when freeze_cnn is absent or false.
- freeze_cnn flag mirrors the curriculum pattern already used for freeze_vz (environment-level constraint).

---

### [407] Implement CNN parameter freezing in train.py based on curriculum config

**Summary**: New logic reads freeze_cnn flag and applies require_grad masking to CNN layers during training.

**Facts**:
- Added CNN freezing block immediately after model creation (lines 464–470) to apply regardless of model source (fresh/checkpoint/stage-transfer).
- Reads `freeze_cnn` flag from curriculum stage config: `bool(conf.get("freeze_cnn", False))` defaults to False for backward compatibility.
- Iterates through `model.policy.features_extractor.named_parameters()` and disables gradients for layers starting with "cnn" (covers cnn.* and cnn_fc.*).
- Logic: `param.requires_grad = not freeze_cnn` — when freeze_cnn=true, CNN parameters are frozen; when false, all parameters train normally.
- Logs freeze status for debugging: "[CNN] freeze_cnn=true — cnn+cnn_fc frozen" or "[CNN] freeze_cnn=false — all params trainable".

---

## Changes

### [385] Added observation normalization parameters to EnvConfig

**Summary**: Introduced goal_xy_norm and goal_z_norm to separate normalization divisors from curriculum parameters

**Facts**:
- Added goal_xy_norm: float = 20.0 to env_config.py; larger than fence_x_max (15.0) to cover drone drift edge cases
- Added goal_z_norm: float = 8.0 to env_config.py; matches fence_z_max for altitude normalization
- These parameters will be used in _build_state_vector() for soft normalization of goal_body vector
- Design decouples normalization divisors from curriculum parameters (goal_dist_high_end, fence bounds)
- 20.0 m chosen as compromise: larger than fence_x_max (15.0) to handle drone-at-boundary scenarios but conservative vs fence diagonal (42m)

---

### [388] Goal sampling bounded with 20m maximum distance in post-warmup phase

**Summary**: Phase B now enforces max_goal_dist=e.goal_xy_norm (20m) alongside min_goal_dist constraint.

**Facts**:
- Goal sampling condition changed from `if dist > min_goal_dist:` to `if min_goal_dist <= dist <= max_goal_dist:`
- max_goal_dist set to e.goal_xy_norm (documented as 20m) in post-ramp-up phase (else branch)
- Maximum distance value matches observation normalizer for consistency between sampling and normalization
- Applies only when ramp_ratio >= 1.0 (post-warmup / phase B goal generation)

---

### [401] Enforce quaternion canonical form (qw >= 0) in get_quaternion

**Summary**: Normalizes quaternion representation by negating when real part is negative for training consistency

**Facts**:
- get_quaternion method in ROSBridge class now enforces canonical quaternion form
- Quaternions with qw (real part) < 0.0 are negated to enforce qw >= 0.0
- Canonicalization applied outside the Gazebo lock after reading gz_quat array
- Ensures mathematically equivalent rotations (q and -q) resolve to single canonical representation
- Critical for consistent quaternion noise injection downstream in state vector construction

---

### [405] Relax DepthStateExtractor state dimension assertion for flexibility

**Summary**: Changed from fixed n_state==46 to flexible n_state > 0 to enable future configuration.

**Facts**:
- DepthStateExtractor assertion changed from `assert n_state == 46` to `assert n_state > 0` (line 12).
- Allows state dimension to vary without rejecting valid configurations.
- Enables feature extractor to work with different state vector sizes in future iterations.

---
## Refactors

### [392] Eliminated redundant goal distance norm calculation in step_process

**Summary**: Replaced duplicate norm(rel_goal_xy) with reuse of dist_xy variable computed 2 lines prior.

**Facts**:
- Removed line: `goal_dir_norm = float(np.linalg.norm(rel_goal_xy))`
- Changed condition from `if goal_dir_norm > 1e-6:` to `if dist_xy > 1e-6:`
- dist_xy already computed at line 317 as `float(np.linalg.norm(rel_goal_xy))` — same calculation
- No behavioral change — both compute identical value, now reused instead of recalculated

---

### [393] Shifted depth normalization responsibility from _build_obs to callers

**Summary**: Changed _build_obs parameter from raw depth to pre-normalized depth_frame; removed internal normalization step.

**Facts**:
- Renamed _build_obs parameter from `depth: np.ndarray` to `depth_frame: np.ndarray`
- Removed line: `depth_frame = self._normalize_depth_frame(depth)` from inside _build_obs
- Updated docstring: "depth_frame must already be normalized" (clarifies new contract)
- Eliminates double normalization: depth normalized once at call site (step_process) instead of twice (step_process line 309 + _build_obs internal)

---

## Security Notes

### [409] ROS domain ID and topic collision risks across ranks

**Summary**: Hardcoded ROS topic names (/clock, /depth_camera/image_raw, /lidar/scan) lack rank isolation; partitioned only by environment variable.

**Facts**:
- ROS topic /clock bridged by all ranks; isolated only by ROS_DOMAIN_ID environment variable at process_utils.py:127
- /depth_camera ROS topic and /lidar_2d_v2/scan Gazebo topic names hardcoded; isolated only by GZ_PARTITION at process_utils.py:163,200
- /camera/depth/image_raw and /lidar/scan ROS topics hardcoded; isolated only by ROS_DOMAIN_ID at process_utils.py:174,210
- ROS domain IDs range 30..30+N-1 for N ranks; no guard against external ROS processes using same range
- If environment variable isolation fails (partition not respected by ros_gz_bridge), all ranks would collide on identical topic names

---

## Complete Timeline

| ID | Time | Type | Title |
|---|---|---|---|
| 378 | 03:27:19 | discovery | Burst VIO publishing pattern after teleport interleaves lock acquisitions |
| 379 | 03:56:19 | discovery | Thread Lock and Concurrency Issues in ROSBridge Identified |
| 380 | 04:16:54 | discovery | Action is not included in policy observation vector |
| 381 | 04:18:06 | discovery | 46-dim state vector architecture with sensor fusion and domain randomization |
| 382 | 04:18:46 | discovery | Velocity action limits and state bounds in training environment |
| 383 | 04:18:46 | discovery | No observation normalization wrapper in training pipeline |
| 384 | 04:22:10 | discovery | EnvConfig defines non-uniform velocity limits and multi-env fast reset infrastructure |
| 385 | 04:27:25 | change | Added observation normalization parameters to EnvConfig |
| 386 | 04:27:45 | bugfix | Fixed yaw rate sign inversion in angular velocity frame conversion |
| 387 | 04:27:45 | feature | Implemented soft observation normalization in state vector |
| 388 | 04:31:26 | change | Goal sampling bounded with 20m maximum distance in post-warmup phase |
| 389 | 04:33:09 | discovery | LiDAR initialization and nan/inf handling patterns in bridge_factory.py |
| 390 | 04:33:20 | discovery | LiDAR callback sanitizes sensor errors with defensive binning strategy |
| 391 | 04:35:17 | discovery | Depth camera capped at 10.0m while LiDAR capped at 30.0m — different sensor modalities |
| 392 | 04:36:53 | refactor | Eliminated redundant goal distance norm calculation in step_process |
| 393 | 04:37:37 | refactor | Shifted depth normalization responsibility from _build_obs to callers |
| 394 | 04:56:25 | discovery | Front depth used in stage2 obstacle visibility and slowdown rewards |
| 395 | 04:56:35 | discovery | Obstacle visibility and slowdown reward conditions with multi-gate validation |
| 396 | 05:37:57 | discovery | Depth array shape mismatch in drone RL environment reset |
| 397 | 05:38:07 | discovery | Root cause identified: depth frame shape inconsistency in history initialization |
| 398 | 05:38:15 | discovery | Race condition in reset sequence: depth history filled then immediately overwritten |
| 399 | 08:38:33 | feature | Added quaternion observation noise parameter to EnvConfig |
| 400 | 08:38:44 | feature | Implemented quaternion noise injection in state vector construction |
| 401 | 08:39:51 | change | Enforce quaternion canonical form (qw >= 0) in get_quaternion |
| 402 | 08:43:24 | discovery | Current curriculum-based policy lacks CNN freezing mechanism for early stages |
| 403 | 08:43:35 | discovery | CNN freezing not implemented in train.py training pipeline |
| 404 | 08:43:47 | discovery | Stage transfer uses learning_rate reduction but no CNN parameter freezing |
| 405 | 08:43:56 | change | Relax DepthStateExtractor state dimension assertion for flexibility |
| 406 | 08:44:14 | feature | Add freeze_cnn configuration flag to curriculum stages 0 and 1 |
| 407 | 08:44:27 | feature | Implement CNN parameter freezing in train.py based on curriculum config |
| 408 | 09:05:23 | discovery | Multi-env process isolation audit: environment variable and port offset strategy |
| 409 | 09:05:23 | security_note | ROS domain ID and topic collision risks across ranks |
| 410 | 09:05:23 | discovery | Python logging not multiprocess-safe across SubprocVecEnv workers |
| 411 | 09:05:23 | discovery | Configuration fields multi_env_fast_reset_enabled and total_envs never populated from n_envs |
| 412 | 09:05:23 | discovery | Startup stagger disabled: STARTUP_STAGGER_S set to 0 |
| 413 | 09:06:42 | discovery | Teleport drone implementation uses GZ transport API with CLI subprocess fallback |
| 414 | 09:06:52 | bugfix | Found undefined variable bug: cur_alt used before definition in keepalive stale hover log |
| 415 | 09:07:49 | bugfix | Spawner class calls undefined _gz_spin_wrap method |
| 416 | 09:07:49 | bugfix | VIO state mutated by both VIO thread and main thread without synchronization |
| 417 | 09:07:49 | discovery | Lock contention on _spin_lock and _ros_pub_lock during concurrent thread operations |
| 418 | 09:07:49 | discovery | tick() method busy-spins without sleep between executor calls |
| 419 | 09:07:49 | discovery | os.environ mutation in spawn_random_field creates race condition |
| 420 | 09:07:49 | discovery | Comment claims _vo_publish_lock serializes VIO state but lock does not exist |
| 421 | 09:09:11 | discovery | step_process() blocks 50-105ms per step, entirely due to bridge.tick(dt) call |
| 422 | 09:09:11 | bugfix | bridge.tick() can hang indefinitely if _spin_lock is held by blocked background thread |
| 423 | 09:09:11 | discovery | hard_reset_fallback_episode_reset can block up to 300 seconds with no outer timeout |
| 424 | 09:09:11 | discovery | Three threads now contend for _ros_pub_lock; old code had at most two |
| 425 | 09:09:11 | discovery | New background _spin_thread (20Hz) introduced _spin_lock contention not present in old code |
| 426 | 09:43:37 | bugfix | Thread-based timeout wrapper to prevent ROS executor starvation during multi-env Gazebo RPCs |
| 427 | 09:43:47 | bugfix | Outer wall-clock deadline for hard reset to prevent SubprocVecEnv barrier blocking |
