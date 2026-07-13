from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    # Observation / action
    depth_shape: tuple = (3, 84, 84)       # depth obs: 3 stacked 84x84 frames
    depth_min: float = -1.0   # Ch1/Ch2 (delta_v, delta_a) are signed [-1, 1]
    depth_max: float = 1.0    # Ch0 (EDT distance) normalized to [0, 1]
    bev_max_range_m: float = 10.0  # physical range for Ch0 de-normalization; must match ExtractorConfig.max_depth_range
    state_dim: int = 31   # 3 vel + 3 ang_vel + 1 alt + 3 goal + 4 orientation + 4 last_action + 4 delta_A1 + 4 delta_A2 + 4 fence_flu + 1 dfa_progress
    action_dim: int = 4                    # [vx, vy, vz, yaw_rate]
    # Track A vs Track B (SAGE) — "raw": ActorCriticPolicy outputs [vx,vy,vz,yaw_rate]
    # directly in [-1,1] via ActionManager. "symbolic": SymbolicActorCriticPolicy outputs
    # symbolic_num_gates Beta-distributed gates in [0,1], blended by SAGEPipeline's
    # primitives (goal/vertical/dodge/face) + CBFShield instead of ActionManager.
    # Default "raw" — Track A code path is byte-identical when this stays unset.
    policy_mode: str = "raw"               # "raw" | "symbolic"
    symbolic_num_gates: int = 4            # goal, vertical, dodge, face (positional order) — must match SAGEPipeline's primitives list, 1:1 (no extra +1, p_boost removed)
    sage_face_gate_min: float = 0.2        # p_face floor — see PrimitivesConfig.face_gate_min
    cbf_enabled: bool = False              # symbolic mode only — gates CBFShield's real QP vs identity passthrough
    # CBFShield is minimal-intervention by design (QP finds the SMALLEST
    # correction satisfying grad_h.u >= -cbf_alpha*h, h = EDT(x)-margin) --
    # not "actively flee the obstacle". Correction only grows once h goes
    # negative (drone already inside the margin), scaled by cbf_alpha. Raise
    # cbf_safety_margin to trigger the constraint farther out, cbf_alpha to
    # make the response steeper once triggered -- both were previously
    # hardcoded (0.5m / 2.0) in train_manager._build_sage_pipeline with no
    # way to tune via config.
    cbf_safety_margin: float = 0.5         # metres, h(x) = EDT(x) - this
    cbf_alpha: float = 2.0                 # class-K linear gain
    # Symbolic-only reset curriculum: randomize post-reset altitude (scripted vz
    # climb/descend, not teleport) so VerticalPrimitive's descend branch (z >
    # alt_optimal_high) gets exercised too -- the shared _pre_episode_climb_if_low
    # always lands at a fixed low altitude (1.5m) for both tracks, so without this
    # symbolic mode never trained the "fly back down" case. No effect on Track A.
    sage_random_start_alt_min: float = 2.0
    sage_random_start_alt_max: float = 6.5  # margin below alt_max=7.0 -- scripted climb can overshoot target by ~0.15m
    # Sim-to-real observation noise: x_noisy = x + N(0, σ²), applied in
    # train_manager.py:_build_state_vector — only the policy observation is noised,
    # GT position is still used for rewards/collision/fence. 0.0 = disabled.
    lidar_num_sectors: int = 36        # angular sectors across 45° sweep (1.25° each)
    lidar_min_range: float = 0.1       # clip floor (m) — also used for signal-error fill
    lidar_max_range: float = 30.0      # clip ceiling (m)
    goal_xy_norm: float = 20.0         # normalizer for body-frame goal xy (m) — > goal_dist_high_end to cover drone drift
    goal_z_norm: float = 8.0           # normalizer for body-frame goal z (m) — matches fence_z_max
    # Sim-to-real observation noise — injected into policy obs only, NOT reward/collision GT
    # Stage 0 (freeze_vz, no pillars): 0.0 — drone học bay cơ bản, không cần nhiễu
    # Stage 1 (no pillars):            nhỏ — pos=0.05, vel=0.03, yaw=0.01, ang_vel=0.005, quat=0.005
    # Stage 2+ (có cột):               trung — pos=0.10, vel=0.05, yaw=0.02, ang_vel=0.008, quat=0.010
    # Stage 4–5 (nhiều cột, full sim-to-real): pos=0.15, vel=0.08, yaw=0.03, ang_vel=0.012, quat=0.015
    obs_noise_pos_std: float = 0.0      # σ position (m)                — real VIO/GPS: 0.1–0.3
    obs_noise_vel_std: float = 0.0      # σ velocity (m/s)              — real EKF:     0.05–0.15
    obs_noise_yaw_std: float = 0.0      # σ yaw (rad)                   — real IMU:     0.02–0.08
    obs_noise_ang_vel_std: float = 0.0  # σ angular rate (rad/s)        — real IMU gyro: 0.005–0.02
    obs_noise_quat_std: float = 0.0     # σ quaternion per-component + renorm — real IMU: 0.01–0.03

    # Sim-to-real action randomization
    # Stage 0:   action_noise_std=0.0, action_delay_steps=0 — không nhiễu, học hành vi sạch
    # Stage 1:   action_noise_std=0.02, action_delay_steps=0 — nhiễu nhẹ, chưa cần delay
    # Stage 2–3: action_noise_std=0.03, action_delay_steps=1 — thêm 1-step delay (50ms)
    # Stage 4–5: action_noise_std=0.05, action_delay_steps=1 — full sim-to-real, gần thực tế nhất
    # Ghi chú: action_noise_std đơn vị m/s (vx/vy/vz) và rad/s (yaw_rate) — scale theo vx_limit
    action_noise_std: float = 0.0       # σ actuator noise added to sent cmd — real ESC jitter: 0.02–0.05
    action_delay_steps: int = 0         # minimum delay steps (fixed when action_delay_steps_max=0)
    action_delay_steps_max: int = 0     # if > action_delay_steps: sample uniform int per-episode in [min, max]

    # Per-episode domain randomisation
    # mass_scale: uniform in [min, max] each episode — models mass/motor/battery variation
    #   1.0 = no randomisation; <1 = heavier (less effective); >1 = lighter
    #   Stage 0-1: disabled (1.0, 1.0) — learn clean dynamics first
    #   Stage 2-3: ±5-8% — mild uncertainty
    #   Stage 4-5: ±10% — full sim-to-real (Kaufmann 2023 used ±15%)
    mass_scale_min: float = 1.0
    mass_scale_max: float = 1.0
    # wind_speed_max: max horizontal wind speed (m/s) sampled uniform per episode
    #   0.0 = disabled; direction random ENU; wz=0 (horizontal only)
    #   Stage 0-1: 0.0; Stage 2: 0.5; Stage 3: 1.0; Stage 4-5: 2.0
    wind_speed_max: float = 0.0

    # Mission geometry
    start: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    goal: list = field(default_factory=lambda: [8.0, 0.0, 2.0])
    goal_z_min: float = 1.0
    goal_z_max: float = 6.5

    # Goal acceptance radius curriculum (shrinks over training)
    goal_xy_radius_schedule_values: list = field(default_factory=lambda: [1.5, 1.2, 1.0, 0.5])
    goal_xy_radius_schedule_steps: list = field(default_factory=lambda: [0, 40_000, 60_000, 80_000])

    # Goal sampling ramp (annulus inner/outer edge over training steps)
    goal_dist_low_start: float = 8.0
    goal_dist_low_end: float = 16.0
    goal_dist_high_start: float = 13.0
    goal_dist_high_end: float = 16.0
    goal_dist_ramp_steps: int = 180_000
    goal_dist_ramp_min_band: float = 2.0   # minimum annulus width during ramp phase

    # Fence boundaries
    fence_x_min: float = -15.0
    fence_x_max: float = 15.0
    fence_y_min: float = -15.0
    fence_y_max: float = 15.0
    fence_z_min: float = 0.3
    fence_z_max: float = 8.0

    # Velocity limits (applied by ActionManager)
    vx_limit: float = 3.0
    vy_limit: float = 2.5
    vz_up_limit: float = 1.6
    vz_down_limit: float = 0.6
    yaw_rate_limit: float = 1.35
    action_smoothing: float = 0.35
    freeze_vz: bool = False  # Stage 0: soft-band vz constraint
    freeze_vz_band_low: float = 2.5  # if z < this AND vz_cmd < 0 → override climb
    freeze_vz_band_high: float = 6.5  # if z > this → P-controller descend to band_high
    freeze_vz_hold_alt: float = 3.2   # legacy, unused by soft band

    # Symbolic Extractor (BEV pipeline)
    use_symbolic_extractor: bool = False  # True: 3-channel kinematic BEV; False: single-channel depth (legacy)
    freeze_vz_kp: float = 2.0         # P-gain for upper boundary enforcement

    # Altitude safe band
    alt_min: float = 0.8
    alt_max: float = 7.0

    # Episode limits
    max_steps: int = 500
    dt: float = 0.1

    # Takeoff assist
    takeoff_assist_steps: int = 80
    takeoff_assist_alt: float = 0.7
    takeoff_assist_vz: float = 1.5
    airborne_z: float = 0.8

    # PX4 sanity thresholds
    max_px4_lpos_z_abs: float = 20.0
    max_px4_vel_z_abs: float = 5.0
    max_px4_speed: float = 8.0

    # Teleport settle
    teleport_settle_timeout: float = 3.0
    teleport_settle_max_speed: float = 2.0
    teleport_settle_max_vz: float = 1.5

    # Depth processing
    depth_sector_feature_dim: int = 9
    depth_stack_size: int = 3
    center_crop_h_lo: float = 0.35   # fraction of rows to crop top
    center_crop_h_hi: float = 0.75   # fraction of rows to crop bottom
    center_crop_w_lo: float = 0.35   # fraction of cols to crop left
    center_crop_w_hi: float = 0.65   # fraction of cols to crop right

    # Reset — lift warmup (from old drone_env.py L751-753)
    lift_warmup_time: float = 0.3
    lift_vz: float = 1.5

    # Reset — rescue (from old drone_env.py L78)
    rescue_margin_m: float = 10.0
    rescue_jitter_m: float = 3.0   # random XY offset added to rescue target to break same-position loops

    # Reset — pre-episode yaw alignment (from old drone_env.py L640-650)
    pre_episode_auto_yaw_enabled: bool = False
    pre_episode_auto_yaw_timeout_s: float = 4.0
    pre_episode_auto_yaw_tol_deg: float = 8.0
    pre_episode_auto_yaw_gain: float = 1.2
    # Fraction of this stage's min_steps over which yaw-assist min_assist ramps
    # 1.0->0.0, scaling the fixed relative schedule in train_manager.py's
    # reset() (_YAW_SCHEDULE_FRACS). Expressed as a fraction (not a raw step
    # count) so the ramp duration auto-scales if min_steps is changed later --
    # e.g. 0.375 always means "ramp over the first 37.5% of this stage",
    # regardless of what min_steps is set to. Can exceed 1.0 (ramp intentionally
    # outlives this stage, finishing via yaw_curriculum_carryover_steps in the
    # next one -- see stage0/stage1 in ppo_config.yaml). <=0 = binary on/off
    # (full assist, no fade). Set per-stage in ppo_config.yaml.
    yaw_curriculum_frac: float = 0.0
    # This stage's curriculum step budget (mirrors ppo_config.yaml curriculum
    # entry's min_steps) -- forwarded here only so yaw_curriculum_frac above
    # has something to scale against at runtime.
    min_steps: int = 0
    # Manual carryover for the _YAW_SCHEDULE_FRACS assist-ratio lookup
    # (train_manager.py reset()) so the ramp-down doesn't reset to 100% assist at a stage boundary.
    # _stage_start_step is per-stage (stage{N}_start_step.json), so
    # _steps_in_stage alone would jump back to 0 the instant a later stage also
    # enables pre_episode_auto_yaw_enabled. Set this in a later stage's yaml
    # block to the earlier stage's actual run length (e.g. stage1's block sets
    # this to stage0's min_steps) to continue the same ramp instead of
    # restarting it. 0 = no carryover (default, single-stage schedule).
    yaw_curriculum_carryover_steps: int = 0

    # Reset — multi-env fast reset (from old drone_env.py L616-636)
    multi_env_fast_reset_enabled: bool = False
    total_envs: int = 1
    multi_env_fast_reset_idle_before_teleport_s: float = 0.25
    multi_env_fast_reset_fresh_timeout_s: float = 1.5
    multi_env_fast_reset_settle_timeout_s: float = 1.0
    multi_env_fast_reset_ekf_timeout_s: float = 1.5

    # Reset — strategy thresholds (from old drone_env.py L66-67)
    continuous_reset_fence_margin_thresh: float = 3.0

    # Start zone clearance (used by _pose_near_start)
    start_clearance_xy: float = 0.8   # m, radius within which pose is "near start"

    # Reset — descend before disarm (from old drone_env.py L746-748)
    reset_descend_alt: float = 0.25
    reset_descend_timeout: float = 4.0
    reset_descend_vz: float = -2.0

    # Reset — collision anti-sink during continuous reset (from old drone_env.py L604-614)
    collision_continuous_reset_anti_sink_enabled: bool = True
    collision_continuous_reset_anti_sink_duration_s: float = 0.20
    collision_continuous_reset_anti_sink_gain: float = 0.70
    collision_continuous_reset_anti_sink_max_vz: float = 0.20

    # Reset — hard reset periodicity (from old drone_env.py L36)
    hard_reset_every_episodes: int = 0


    use_rescue_after_out_of_fence: bool = True
    rescue_timeout_base_s: float = 4.0
    rescue_timeout_min_s: float = 8.0
    rescue_timeout_max_s: float = 25.0
    rescue_timeout_buffer_s: float = 2.0
    rescue_expected_speed_factor: float = 0.5
    rescue_xy_speed_max: float = 2.5
    rescue_target_alt_m: float = 3.0
    rescue_xy_kp: float = 1.5

    # Reset — multi-env fast reset trigger reasons (from old drone_env.py L634-638)
    multi_env_fast_reset_reasons: tuple = ("fell_to_ground", "ground", "flipped")
