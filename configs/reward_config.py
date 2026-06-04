from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    # Terminal rewards
    goal_xy_terminal_reward: float = 130.0
    max_steps_penalty_pillars_near_goal: float = -200.0
    max_steps_penalty_pillars_far_goal: float = -190.0
    max_steps_penalty_no_pillars: float = -90.0
    near_goal_threshold_factor: float = 3.5  # multiplied by goal_xy_radius

    # Progress shaping
    progress_pos_coef: float = 8.0
    progress_neg_coef: float = 6.0
    stage2_progress_pos_coef: float = 2.2

    # Time penalty (escalating)
    time_penalty_base: float = -0.08
    time_penalty_step170: float = -0.12
    time_penalty_step250: float = -0.13
    time_penalty_step350: float = -0.15
    time_penalty_step450: float = -0.20

    # Altitude penalty (linear outside [alt_min, alt_max] from env_config)
    alt_below_min_coef: float = -2.5
    alt_above_max_coef: float = -1.5

    # Action smoothness
    smooth_penalty_coef: float = -0.04

    # Velocity toward goal
    stage1_velocity_goal_coef: float = 0.60
    stage2_velocity_goal_coef_far: float = 0.175
    stage2_velocity_goal_coef_near: float = 0.12
    stage2_velocity_goal_coef_danger: float = 0.03
    stage2_velocity_goal_clip: float = 1.8
    stage2_velocity_goal_far_dist: float = 2.5
    stage2_velocity_goal_near_dist: float = 1.9

    # Heading / yaw alignment
    stage1_face_goal_coef: float = 0.40
    stage1_yaw_error_penalty_coef: float = 0.26
    stage1_backwards_yaw_penalty_coef: float = 0.20
    stage1_backwards_yaw_speed_gate: float = 0.50
    stage2_yaw_align_gain: float = 1.5
    stage2_yaw_align_penalty_coef: float = 0.05
    stage2_backwards_yaw_penalty_coef: float = 0.25
    stage2_backwards_yaw_dot_thresh: float = -0.20
    stage2_backwards_yaw_speed_to_goal_thresh: float = 0.35
    stage2_backwards_yaw_horizontal_speed_thresh: float = 0.50
    yaw_rate_penalty_coef: float = -0.175

    # Goal yaw terminal bonus
    stage1_goal_yaw_good_dot: float = 0.85
    stage1_goal_yaw_ok_dot: float = 0.65
    stage1_goal_yaw_bad_dot: float = 0.0
    stage1_goal_yaw_bonus_good: float = 15.0
    stage1_goal_yaw_bonus_ok: float = 7.0
    stage1_goal_yaw_penalty_bad: float = -15.0

    # Lateral / fence penalty
    stage1_lateral_thresh: float = 3.0
    stage1_lateral_coef: float = 0.03
    stage1_fence_thresh: float = 2.5
    stage1_fence_coef: float = 0.25
    stage2_lateral_thresh: float = 2.0
    stage2_lateral_coef: float = 0.08
    stage2_fence_thresh: float = 3.0
    stage2_fence_coef: float = 0.60
    stage2_near_fence_penalty_coefs: list = field(default_factory=lambda: [-0.03, -0.25])

    # Start zone penalty
    start_zone_penalty_coef: float = -0.2
    start_penalty_after_steps: int = 80
    start_xy_radius: float = 1.5

    # Speed / fall penalties
    speed_penalty_thresh: float = 2.5
    speed_penalty_coef: float = -0.05
    fall_vz_thresh: float = 0.5
    fall_penalty_coef: float = -0.2

    # Pillar clearance / collision
    pillar_safe_clearance: float = 2.0
    pillar_clearance_penalty_weight: float = 0.60
    pillar_danger_radius: float = 1.1   # too-close XY penalty
    pillar_zone_radius: float = 1.85    # entered_pillar_zone flag
    pillar_too_close_penalty_coef: float = -0.8

    # Pillar pass reward
    passed_pillar_reward: float = 7.0
    clearance_progress_reward: float = 0.03

    # Near-miss
    stage2_near_miss_clearance_min: float = 1.2
    stage2_near_miss_clearance_good: float = 2.5
    stage2_near_miss_reward_good: float = 0.50
    stage2_near_miss_reward_ok: float = 0.20

    # Obstacle approach penalty
    obstacle_approach_depth_thresh: float = 2.0
    obstacle_approach_speed_thresh: float = 1.0
    obstacle_approach_coef: float = -0.3

    # Obstacle slowdown reward
    stage2_obstacle_slowdown_reward: float = 0.05
    stage2_obstacle_slowdown_depth_thresh: float = 1.2
    stage2_obstacle_slowdown_speed_drop: float = 0.22
    stage2_obstacle_slowdown_min_goal_speed: float = 0.225

    # Obstacle visibility reward
    stage2_obstacle_visibility_reward: float = 0.15
    stage2_obstacle_visible_depth_min: float = 1.2
    stage2_obstacle_visible_depth_max: float = 5.0
    stage2_obstacle_visibility_safe_pillar_dist: float = 1.4
    stage2_obstacle_visibility_progress_gate: float = -0.1
    stage2_obstacle_visibility_end_step: int = 75_000

    # Bypass subgoals
    stage2_pillar_bypass_offset_m: float = 2.4
    stage2_pillar_bypass_near_radius: float = 1.2
    stage2_pillar_bypass_near_reward: float = 0.5
    stage2_pillar_bypass_reach_reward: float = 10.75
    stage2_bypass_decision_reward: float = 0.3
    stage2_bypass_decision_min_align: float = 0.5
    stage2_bypass_decision_min_speed: float = 0.2
    stage2_bypass_progress_reward: float = 0.95
    stage2_bypass_progress_min_clearance_gain: float = 0.05

    # Ring subgoals
    stage2_pillar_ring_radius_margin: float = 1.20
    stage2_pillar_ring_near_radius: float = 0.65
    stage2_pillar_ring_near_reward: float = 0.35
    stage2_pillar_ring_reach_reward: float = 4.25
    stage2_pillar_ring_max_active_pillars: int = 3

    # Post-pillar reorientation
    stage2_post_pillar_window_steps: int = 35
    stage2_post_pillar_fixation_penalty: float = 0.025
    stage2_post_pillar_goal_realign_reward: float = 0.05
    stage2_post_pillar_look_pillar_dot_thresh: float = 0.60
    stage2_post_pillar_look_goal_dot_thresh: float = 0.75
    stage2_post_pillar_speed_goal_thresh: float = 0.20

    # Stage 1 subgoals
    stage1_subgoal_near_radius: float = 1.8
    stage1_subgoal_reach_radius: float = 1.2
    stage1_subgoal_near_reward: float = 2.5
    stage1_subgoal_reach_reward: float = 5.0
    stage1_subgoal_dist_min: float = 8.0
    stage1_subgoal_dist_max: float = 25.0

    # Collision course / clearance
    clearance_body_safe: float = 0.45
    clearance_body_danger: float = 0.20
    clearance_soft_penalty_coef: float = -1.75
    clearance_danger_penalty_coef: float = -4.0
    collision_course_coef: float = -5.25

    # Too-slow penalty
    reward_too_slow_after_steps: int = 35
    reward_too_slow_min_dist: float = 1.7
    reward_too_slow_min_speed: float = 0.7
    reward_too_slow_coef: float = -0.15

    # Pillar attention reward (from old drone_env.py L155-228)
    stage2_pillar_attention_geom_reward: float = 0.03
    stage2_pillar_attention_geom_dot_thresh: float = 0.35
    stage2_pillar_attention_geom_min_dist: float = 1.4
    stage2_pillar_attention_geom_max_dist: float = 4.5
    stage2_pillar_attention_geom_min_progress: float = -0.10
    stage2_pillar_attention_normalize_by_count: bool = True
    stage2_pillar_attention_reference_count: int = 5
    stage2_pillar_first_look_reward: float = 2.5
    stage2_pillar_first_look_dot_thresh: float = 0.30
    stage2_pillar_first_look_min_dist: float = 1.5
    stage2_pillar_first_look_max_dist: float = 6.0
    stage2_pillar_first_look_min_progress: float = -0.05
    stage2_pillar_avoidance_track_reward: float = 1.40
    stage2_pillar_avoidance_track_max_steps: int = 25
    stage2_pillar_avoidance_track_dot_thresh: float = 0.20
    stage2_pillar_avoidance_track_min_dist: float = 1.4
    stage2_pillar_avoidance_track_max_dist: float = 4.8
    stage2_pillar_avoidance_track_min_progress: float = -0.10
    # Post-pillar (max steps per pillar caps, from old drone_env.py L415-419)
    stage2_post_pillar_reward_max_steps_per_pillar: int = 12
    stage2_post_pillar_penalty_max_steps_per_pillar: int = 20
