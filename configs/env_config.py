from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    # Observation / action
    depth_shape: tuple = (3, 84, 84)       # depth obs: 3 stacked 84x84 frames
    depth_min: float = 0.0
    depth_max: float = 10.0
    state_dim: int = 46   # 7 proprioception + 3 body-frame goal + 36 LiDAR sectors
    action_dim: int = 4                    # [vx, vy, vz, yaw_rate]
    # Sim-to-real observation noise: x_noisy = x + N(0, σ²), applied in
    # train_manager.py:_build_state_vector — only the policy observation is noised,
    # GT position is still used for rewards/collision/fence. 0.0 = disabled.
    lidar_num_sectors: int = 36        # angular sectors across 270° sweep (7.5° each)
    lidar_min_range: float = 0.1       # clip floor (m) — also used for signal-error fill
    lidar_max_range: float = 30.0      # clip ceiling (m)
    obs_noise_pos_std: float = 0.0    # σ position (m)   — real VIO/GPS: 0.1–0.3
    obs_noise_vel_std: float = 0.0    # σ velocity (m/s) — real EKF:     0.05–0.15
    obs_noise_yaw_std: float = 0.0    # σ yaw (rad)      — real IMU:     0.02–0.08
    obs_noise_ang_vel_std: float = 0.0  # σ angular rate (rad/s) — real IMU gyro: 0.005–0.02

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
    goal_dist_high_start: float = 10.0
    goal_dist_high_end: float = 16.0
    goal_dist_ramp_steps: int = 350_000
    goal_dist_ramp_min_band: float = 2.0   # minimum annulus width during ramp phase

    # Fence boundaries
    fence_x_min: float = -15.0
    fence_x_max: float = 15.0
    fence_y_min: float = -15.0
    fence_y_max: float = 15.0
    fence_z_min: float = -0.3
    fence_z_max: float = 8.0

    # Velocity limits (applied by ActionManager)
    vx_limit: float = 2.2
    vy_limit: float = 1.8
    vz_up_limit: float = 1.2
    vz_down_limit: float = 0.6
    yaw_rate_limit: float = 0.5
    action_smoothing: float = 0.35
    freeze_vz: bool = False  # Stage 0: lock vz=0, drone learns horizontal nav only

    # Altitude safe band
    alt_min: float = 0.8
    alt_max: float = 5.0

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
    rescue_margin_m: float = 5.5

    # Reset — pre-episode yaw alignment (from old drone_env.py L640-650)
    pre_episode_auto_yaw_enabled: bool = True
    pre_episode_auto_yaw_timeout_s: float = 4.0
    pre_episode_auto_yaw_tol_deg: float = 8.0
    pre_episode_auto_yaw_gain: float = 1.2

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

    # Reset — rescue parameters (from old drone_env.py L78-92)
    use_rescue_after_out_of_fence: bool = True
    rescue_timeout_base_s: float = 4.0
    rescue_timeout_min_s: float = 8.0
    rescue_timeout_max_s: float = 25.0
    rescue_timeout_buffer_s: float = 2.0
    rescue_expected_speed_factor: float = 0.5
    rescue_xy_speed_max: float = 2.5
    rescue_target_alt_m: float = 2.8
    rescue_xy_kp: float = 1.5

    # Reset — multi-env fast reset trigger reasons (from old drone_env.py L634-638)
    multi_env_fast_reset_reasons: tuple = ("fell_to_ground", "ground", "flipped", "out_of_fence")
