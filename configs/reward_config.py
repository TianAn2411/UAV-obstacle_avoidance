from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    # ------------------------------------------------------------------ #
    # Terminal rewards — one-shot at episode end                          #
    # ------------------------------------------------------------------ #
    # Applied once when episode ends. Near-fence episodes stack an extra
    # penalty: base + out_of_fence_penalty * near_fence_terminal_penalty_factor
    #
    # Example: max_steps, stage0, near fence → -90 + (-150*0.8) = -210
    # Example: collision anywhere           →  -280
    goal_xy_terminal_reward: float = 130.0
    collision_penalty: float = -280.0
    out_of_fence_penalty: float = -150.0
    near_fence_terminal_penalty_factor: float = 0.8
    fell_to_ground_penalty: float = -200.0
    flipped_penalty: float = -200.0
    max_steps_penalty_pillars_near_goal: float = -200.0
    max_steps_penalty_pillars_far_goal: float = -190.0
    max_steps_penalty_no_pillars: float = -90.0
    near_goal_threshold_factor: float = 3.5  # near_goal if dist_xy <= goal_radius * this

    # ------------------------------------------------------------------ #
    # Progress shaping — per step                                         #
    # ------------------------------------------------------------------ #
    # r = coef * (prev_dist_xy - dist_xy)
    # Example: drone moves 0.1m closer → +8.0*0.1 = +0.8/step
    # Example: drone drifts 0.05m away  → -6.0*0.05 = -0.3/step
    progress_pos_coef: float = 8.0
    progress_neg_coef: float = 6.0
    stage2_progress_pos_coef: float = 2.2  # reduced in pillar stages (pillar detours expected)

    # ------------------------------------------------------------------ #
    # Time penalty — escalating per step                                  #
    # ------------------------------------------------------------------ #
    # Accumulates each step. Steps 0-170: base only. After 170: base+step170, etc.
    # Example: at step 200 → -0.08 + -0.12 = -0.20/step
    # Example: at step 400 → -0.08 + -0.12 + -0.13 + -0.15 = -0.48/step
    # Only escalates when num_pillars > 0 (stage 0-1: base only).
    time_penalty_base: float = -0.08
    time_penalty_step170: float = -0.12
    time_penalty_step250: float = -0.13
    time_penalty_step350: float = -0.15
    time_penalty_step450: float = -0.20

    # ------------------------------------------------------------------ #
    # Altitude penalty — per step, linear outside safe band               #
    # ------------------------------------------------------------------ #
    # r = coef * (alt_min - alt)  if alt < alt_min  (too low)
    # r = coef * (alt - alt_max)  if alt > alt_max  (too high)
    # alt_min / alt_max come from EnvConfig.
    # Example: drone at 1.0m, alt_min=2.0 → -2.5 * 1.0 = -2.5/step
    alt_below_min_coef: float = -2.5
    alt_above_max_coef: float = -1.5

    # ------------------------------------------------------------------ #
    # Action smoothness — per step                                        #
    # ------------------------------------------------------------------ #
    # r = coef * ||action - prev_action||
    # Example: jerk of 0.5 on one axis → -0.04 * 0.25 = -0.01/step (mild)
    # Example: full reversal (delta=2.0) → -0.04 * 4.0 = -0.16/step
    smooth_penalty_coef: float = -0.04

    # ------------------------------------------------------------------ #
    # Velocity toward goal — per step                                     #
    # ------------------------------------------------------------------ #
    # r = coef * clip(dot(vel_xy, goal_dir), -1, clip_max)
    # Example stage1: flying at 1.5m/s toward goal → 0.50 * 1.5 = +0.75/step
    # Example stage1: drifting 0.5m/s away         → 1.0 * (-0.5) clamped = -0.5/step
    # Stage2: coef scales down near pillars to avoid rushing into obstacles.
    stage1_velocity_goal_coef: float = 0.50
    stage2_velocity_goal_coef_far: float = 0.175    # pillar dist > far_dist
    stage2_velocity_goal_coef_near: float = 0.12    # pillar dist > near_dist
    stage2_velocity_goal_coef_danger: float = 0.03  # pillar dist <= near_dist
    stage2_velocity_goal_clip: float = 1.8
    stage2_velocity_goal_far_dist: float = 2.5
    stage2_velocity_goal_near_dist: float = 1.9

    # ------------------------------------------------------------------ #
    # Yaw alignment — per step                                            #
    # ------------------------------------------------------------------ #
    # Progress amplification: yaw reward is amplified when drone is moving
    # toward goal while facing it, penalized when moving toward goal facing away.
    # r_progress = stage1_yaw_progress_amp * progress * cos(yaw_error)
    # (added on top of base yaw signal; only stage0/1, num_pillars==0)
    # Example: +0.1m progress, cos=0.95 → 2.0*0.1*0.95 = +0.19 bonus
    # Example: +0.1m progress, cos=0.5  → 2.0*0.1*0.5  = +0.10 bonus (smaller)
    # Example: +0.1m progress, cos=-0.3 → 2.0*0.1*(-0.3) = -0.06 penalty
    stage1_yaw_progress_amp: float = 2.0        # amplification coef for progress × cos term
    stage1_yaw_progress_gate: float = 0.0       # min progress (m) to trigger amplificationo
    # r = face_goal_coef * cos(yaw_error)  +  forward_bonus_or_penalty
    # Example stage1: perfectly aligned (yaw_error=0) → 0.20*1.0 + 0.40 = +0.60/step
    # Example stage1: 90° off (cos=0)                 → 0.20*0.0 - 1.23 = -1.23/step
    # backwards_yaw_penalty triggers when flying fast while facing away.
    stage1_face_goal_coef: float = 0.20
    stage1_backwards_yaw_penalty_coef: float = 0.20
    stage1_backwards_yaw_speed_gate: float = 0.50   # min speed (m/s) to trigger

    # Forward-goal bonus/penalty — piecewise linear ramp around good_thresh
    # cos(yaw_error) >= good_thresh → bonus (drone facing goal)
    # cos(yaw_error) <  good_thresh → penalty (drone facing away)
    #
    # Bonus:   +coef * (cos - thresh) / (1 - thresh)       — ramps 0→coef as cos goes thresh→1.0
    # Penalty: -coef * (thresh - cos) / (thresh + 1)       — ramps 0→coef as cos goes thresh→-1.0
    #
    # Stage1 (no pillars): tight thresh=0.98 ≈ yaw_error < 11.5° for bonus
    #   cos=1.00 (0°)   → +0.40 * 1.0 = +0.40 bonus
    #   cos=0.98 (11°)  →  0.00 (boundary)
    #   cos=0.00 (90°)  → -2.5 * 0.98/1.98 = -1.24 penalty
    #   cos=-1.0 (180°) → -3.5 penalty (max)
    stage1_yaw_good_thresh: float = 0.92       # cos threshold ≈ 11.5° — above = bonus, below = penalty
    stage1_yaw_forward_bonus_coef: float = 0.75   # max bonus at perfect alignment (cos=1.0)
    stage1_yaw_forward_penalty_coef: float = 3.5  # max penalty at full misalignment (cos=-1.0)

    # Stage2+ (with pillars): looser thresh=0.9275 ≈ yaw_error < 21.9° for bonus
    #   cos=1.00 (0°)   → +0.20 * 1.0 = +0.20 bonus
    #   cos=0.93 (22°)  →  0.00 (boundary)
    #   cos=0.00 (90°)  → -1.2 * 0.9275/1.9275 = -0.578 penalty
    #   cos=-1.0 (180°) → -1.2 penalty (max)
    stage2_yaw_good_thresh: float = 0.9275        # cos threshold ≈ 21.9° — looser than stage1
    stage2_yaw_forward_bonus_coef: float = 0.20
    stage2_yaw_forward_penalty_coef: float = 1.2

    stage2_backwards_yaw_penalty_coef: float = 0.25
    stage2_backwards_yaw_dot_thresh: float = -0.20
    stage2_backwards_yaw_speed_to_goal_thresh: float = 0.35
    stage2_backwards_yaw_horizontal_speed_thresh: float = 0.50
    yaw_rate_penalty_coef: float = -0.475  # r = coef * |yaw_rate|, e.g. 0.5 rad/s → -0.088/step

    # ------------------------------------------------------------------ #
    # Goal yaw terminal bonus — at goal arrival                           #
    # ------------------------------------------------------------------ #
    # One-shot bonus/penalty added on top of goal_xy_terminal_reward.
    # Encourages arriving while facing the goal direction.
    # Example: arrive facing goal (cos=0.90 > good_dot=0.85) → extra +15
    # Example: arrive facing away (cos=-0.1 < bad_dot=0.0)   → extra -15
    stage1_goal_yaw_good_dot: float = 0.85
    stage1_goal_yaw_ok_dot: float = 0.65
    stage1_goal_yaw_bad_dot: float = 0.0
    stage1_goal_yaw_bonus_good: float = 15.0
    stage1_goal_yaw_bonus_ok: float = 7.0
    stage1_goal_yaw_penalty_bad: float = -15.0

    # ------------------------------------------------------------------ #
    # Lateral / fence penalty — per step                                  #
    # ------------------------------------------------------------------ #
    # Quadratic penalty when drone approaches fence.
    # r = -coef * (thresh - fence_margin)^2  when fence_margin < thresh
    # Example stage1: 1.5m from fence, thresh=2.5 → -0.25*(2.5-1.5)^2 = -0.25/step
    # Example stage1: 0.5m from fence            → -0.25*(2.5-0.5)^2 = -1.0/step
    stage1_lateral_thresh: float = 3.0
    stage1_lateral_coef: float = 0.03
    stage1_fence_thresh: float = 2.5
    stage1_fence_coef: float = 0.25
    stage2_lateral_thresh: float = 2.0
    stage2_lateral_coef: float = 0.08
    stage2_fence_thresh: float = 3.0
    stage2_fence_coef: float = 0.60
    stage2_near_fence_penalty_coefs: list = field(default_factory=lambda: [-0.03, -0.25])

    # ------------------------------------------------------------------ #
    # Start zone penalty — per step                                       #
    # ------------------------------------------------------------------ #
    # Penalizes hovering near start position after start_penalty_after_steps.
    # r = coef * (radius - dist_to_start)  when dist < radius AND step > gate
    # Example: 0.5m from start at step 100 → -0.5 * (1.5-0.5) = -0.5/step
    start_zone_penalty_coef: float = -0.5
    start_penalty_after_steps: int = 80
    start_xy_radius: float = 1.5

    # ------------------------------------------------------------------ #
    # Speed / fall penalties — per step                                   #
    # ------------------------------------------------------------------ #
    # Speed: r = coef * (speed - thresh)  when ||vel|| > thresh
    # Example: flying at 3.5m/s, thresh=2.5 → -0.05*(3.5-2.5) = -0.05/step
    #
    # Fall: r = coef * (vz - thresh)  when vz_down > thresh (drone dropping fast)
    # Example: falling at 1.0m/s down, thresh=0.5 → -0.2*(1.0-0.5) = -0.1/step
    speed_penalty_thresh: float = 2.5
    speed_penalty_coef: float = -0.05
    fall_vz_thresh: float = 0.5
    fall_penalty_coef: float = -0.2

    # ------------------------------------------------------------------ #
    # Pillar clearance / collision — per step, stage2+                    #
    # ------------------------------------------------------------------ #
    # too_close: r = coef  when XY dist to nearest pillar < danger_radius
    # clearance_soft: scales from 0 at safe_clearance to full at 0 body clearance
    # collision_course: r = coef * risk * time_weight * speed_weight
    # Example: 0.8m from pillar center, danger_radius=1.1 → -0.8/step (flat)
    pillar_safe_clearance: float = 2.0
    pillar_danger_radius: float = 1.1
    pillar_zone_radius: float = 1.85

    # ------------------------------------------------------------------ #
    # Pillar pass reward — one-shot per pillar                            #
    # ------------------------------------------------------------------ #
    # passed_pillar_reward fires once when drone moves past a pillar.
    # clearance_progress_reward fires each step while actively gaining clearance.
    # Example: pass 1 pillar cleanly → +7.0 once
    passed_pillar_reward: float = 7.0
    clearance_progress_reward: float = 0.03

    # ------------------------------------------------------------------ #
    # Near-miss reward — one-shot                                         #
    # ------------------------------------------------------------------ #
    # Rewards passing close to a pillar without collision (skilled dodge).
    # clearance in [min, good]: ok reward. clearance >= good: full reward.
    # Example: pass at 1.5m clearance (between 1.2 and 2.5) → +0.20 once
    stage2_near_miss_clearance_min: float = 1.2
    stage2_near_miss_clearance_good: float = 2.5
    stage2_near_miss_reward_good: float = 0.50
    stage2_near_miss_reward_ok: float = 0.20

    # ------------------------------------------------------------------ #
    # Obstacle approach penalty — per step                                #
    # ------------------------------------------------------------------ #
    # Penalizes flying fast toward a nearby obstacle.
    # Triggers when depth < depth_thresh AND speed_toward_obs > speed_thresh.
    # Example: depth=1.5m, speed_toward=1.2m/s → -0.3/step
    obstacle_approach_depth_thresh: float = 2.0
    obstacle_approach_speed_thresh: float = 1.0

    # ------------------------------------------------------------------ #
    # Obstacle slowdown reward — per step                                 #
    # ------------------------------------------------------------------ #
    # Rewards decelerating when close to an obstacle.
    # Example: depth=1.0m, speed dropped 0.3m/s → +0.05/step
    stage2_obstacle_slowdown_reward: float = 0.05
    stage2_obstacle_slowdown_depth_thresh: float = 1.2
    stage2_obstacle_slowdown_speed_drop: float = 0.22
    stage2_obstacle_slowdown_min_goal_speed: float = 0.225

    # ------------------------------------------------------------------ #
    # Obstacle visibility reward — per step, early training only          #
    # ------------------------------------------------------------------ #
    # Rewards keeping obstacle in camera FOV at moderate distance.
    # Decays to 0 after visibility_end_step to avoid fixation later.
    # Example: obstacle at 2m depth, visible → +0.15/step (up to step 75k)
    stage2_obstacle_visibility_reward: float = 0.15
    stage2_obstacle_visible_depth_min: float = 1.2
    stage2_obstacle_visible_depth_max: float = 5.0
    stage2_obstacle_visibility_safe_pillar_dist: float = 1.4
    stage2_obstacle_visibility_progress_gate: float = -0.1
    stage2_obstacle_visibility_end_step: int = 75_000

    # ------------------------------------------------------------------ #
    # Bypass subgoals — one-shot per pillar, stage2+                      #
    # ------------------------------------------------------------------ #
    # Virtual waypoints placed offset_m to the side of each pillar.
    # Guides the drone to commit to a side early rather than head-on.
    # near: drone enters near_radius  → small reward once
    # reach: drone passes through     → large reward once
    # Example: reach bypass waypoint → +10.75 once per pillar
    stage2_pillar_bypass_offset_m: float = 2.4
    stage2_pillar_bypass_near_radius: float = 1.2
    stage2_pillar_bypass_near_reward: float = 0.5
    stage2_pillar_bypass_reach_reward: float = 10.75
    stage2_bypass_decision_reward: float = 0.3
    stage2_bypass_decision_min_align: float = 0.5
    stage2_bypass_decision_min_speed: float = 0.2
    stage2_bypass_progress_reward: float = 0.95
    stage2_bypass_progress_min_clearance_gain: float = 0.05

    # ------------------------------------------------------------------ #
    # Ring subgoals — one-shot per pillar, stage2+                        #
    # ------------------------------------------------------------------ #
    # Tighter ring around each pillar. Rewards passing through the ring
    # cleanly (closer = harder = bigger reward signal).
    # Example: pass through ring at 0.5m from edge → +4.25 once
    stage2_pillar_ring_radius_margin: float = 1.20
    stage2_pillar_ring_near_radius: float = 0.65
    stage2_pillar_ring_near_reward: float = 0.35
    stage2_pillar_ring_reach_reward: float = 4.25
    stage2_pillar_ring_max_active_pillars: int = 3

    # ------------------------------------------------------------------ #
    # Post-pillar reorientation — per step, brief window after passing    #
    # ------------------------------------------------------------------ #
    # After passing a pillar, encourages drone to stop fixating on it and
    # reorient toward the goal. Window = window_steps after pillar pass.
    # fixation_penalty: drone still looking at pillar → -penalty/step
    # goal_realign_reward: drone turns back to goal   → +reward/step
    stage2_post_pillar_window_steps: int = 35
    stage2_post_pillar_fixation_penalty: float = 0.025
    stage2_post_pillar_goal_realign_reward: float = 0.05
    stage2_post_pillar_look_pillar_dot_thresh: float = 0.60
    stage2_post_pillar_look_goal_dot_thresh: float = 0.75
    stage2_post_pillar_speed_goal_thresh: float = 0.20

    # ------------------------------------------------------------------ #
    # Stage1 subgoals — one-shot waypoints, stage0/1 only                 #
    # ------------------------------------------------------------------ #
    # 2-4 equidistant waypoints placed along start→goal line.
    # near: drone within near_radius  → +2.5 once
    # reach: drone within reach_radius → +10.0 once
    # Example: goal at 12m, 3 waypoints → max +37.5 bonus per episode
    stage1_subgoal_near_radius: float = 1.8
    stage1_subgoal_reach_radius: float = 1.2
    stage1_subgoal_near_reward: float = 2.5
    stage1_subgoal_reach_reward: float = 10.0
    stage1_subgoal_dist_min: float = 8.0   # only spawn subgoals when goal is far enough
    stage1_subgoal_dist_max: float = 25.0
    stage1_subgoal_count: int = 4        # fixed number of waypoints along start→goal line

    # ------------------------------------------------------------------ #
    # Collision course / body clearance — per step, stage2+               #
    # ------------------------------------------------------------------ #
    # clearance_soft: penalty proportional to proximity (0 at safe, max at 0)
    # clearance_danger: extra steep penalty below danger threshold
    # collision_course: r = coef * risk * time_weight * speed_weight
    # Example: body clearance 0.1m (below danger=0.2) → -4.0/step
    clearance_body_safe: float = 0.45
    clearance_body_danger: float = 0.20
    clearance_soft_penalty_coef: float = -1.75
    clearance_danger_penalty_coef: float = -4.0
    collision_course_coef: float = -5.25

    # ------------------------------------------------------------------ #
    # Too-slow penalty — per step, stage2+                                #
    # ------------------------------------------------------------------ #
    # Penalizes hovering near a pillar without making progress.
    # Triggers after reward_too_slow_after_steps steps near a pillar.
    # Example: stuck 1.5m from pillar for 40 steps → -0.15/step from step 35
    reward_too_slow_after_steps: int = 35
    reward_too_slow_min_dist: float = 1.7
    reward_too_slow_min_speed: float = 0.7
    reward_too_slow_coef: float = -0.15

    # ------------------------------------------------------------------ #
    # Pillar attention reward — per step, stage2+                         #
    # ------------------------------------------------------------------ #
    # geom: reward for keeping nearest pillar in camera FOV while close.
    # first_look: one-shot reward first time drone looks at a pillar.
    # avoidance_track: reward for tracking pillar while actively dodging.
    # Example: pillar at 3m, camera facing it (dot>0.35) → +0.03/step
    # Example: first time looking at pillar 4m away → +2.5 once
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
    # Post-pillar caps
    stage2_post_pillar_reward_max_steps_per_pillar: int = 12
    stage2_post_pillar_penalty_max_steps_per_pillar: int = 20
