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
    out_of_fence_penalty: float = -230.0
    near_fence_terminal_penalty_factor: float = 0.8  # unused — kept for reference
    near_fence_terminal_base_penalty: float = -120.0  # flat stack added to max_steps/collision when near fence
    fell_to_ground_penalty: float = -200.0
    flipped_penalty: float = -200.0
    max_steps_penalty_pillars_near_goal: float = -200.0
    max_steps_penalty_pillars_far_goal: float = -190.0
    max_steps_penalty_no_pillars: float = -90.0
    near_goal_threshold_factor: float = 3.5  # near_goal if dist_xy <= goal_radius * this

    # ------------------------------------------------------------------ #
    # PBRS Hybrid Potential — replaces progress + subgoal rewards         #
    # ------------------------------------------------------------------ #
    # Φ(s,q) = pbrs_dfa_coef*(q/N) + pbrs_dist_coef*(1 - dist/dist_start)
    # dist_start set dynamically per episode → Φ=0 at start, Φ=α+β at goal.
    # F(s,q,s',q') = pbrs_gamma*Φ(s',q') - Φ(s,q)  — applied at ALL steps (no terminal correction;
    # Grzes 2017 correction was for negative Φ and penalises goal arrival with positive Φ).
    # Episode init: _prev_phi = Φ(s0, q=0) to avoid potential shock at step 1.
    pbrs_gamma: float = 1.0
    pbrs_dist_coef: float = 15.0       # α: weight for distance component
    pbrs_dfa_coef: float = 10.0        # β: weight for DFA progress component

    # ------------------------------------------------------------------ #
    # Reward Machine bonus — hard bonus at DFA state transitions          #
    # ------------------------------------------------------------------ #
    rm_subgoal_bonus: float = 8.0         # fired once per q → q+1 advance

    # ------------------------------------------------------------------ #
    # Time penalty — flat rate per step                                   #
    # ------------------------------------------------------------------ #
    time_penalty_base: float = -0.09

    # ------------------------------------------------------------------ #
    # Altitude reward/penalty — piecewise linear per step                 #
    # ------------------------------------------------------------------ #
    # z < alt_min                          → heavy penalty (below_min_coef)
    # alt_min ≤ z < alt_suboptimal_low_thresh → light penalty (suboptimal_low)
    # alt_suboptimal_low_thresh ≤ z < alt_optimal_low → 0
    # alt_optimal_low ≤ z ≤ alt_optimal_high → flat reward (alt_optimal_reward)
    # alt_optimal_high < z ≤ alt_suboptimal_high_thresh → 0
    # alt_suboptimal_high_thresh < z ≤ alt_max → light penalty (above_suboptimal)
    # z > alt_max                          → heavy penalty (above_max_coef)
    alt_below_min_coef: float = -2.5
    alt_above_max_coef: float = -1.5

    alt_suboptimal_low_thresh: float = 2.5   # 0.8–2.5m: light penalty zone
    alt_suboptimal_low_coef: float = -0.7

    alt_optimal_low: float = 4.0             # flat reward zone lower bound
    alt_optimal_high: float = 5.5            # flat reward zone upper bound
    alt_optimal_reward: float = 0.30         # +0.30/step when in [4.0, 5.5]m
    alt_ramp_coef: float = 0.10              # ramp peak: 0→0.10 below, 0.10→0 above optimal zone

    alt_suboptimal_high_thresh: float = 6.0  # 6.0–7.0m: light penalty zone
    alt_above_suboptimal_coef: float = -0.5

    # ------------------------------------------------------------------ #
    # Action smoothness — per step                                        #
    # ------------------------------------------------------------------ #
    # r = coef * ||action - prev_action||
    # Example: jerk of 0.5 on one axis → -0.04 * 0.25 = -0.01/step (mild)
    # Example: full reversal (delta=2.0) → -0.04 * 4.0 = -0.16/step
    smooth_penalty_coef: float = -0.04

    # ------------------------------------------------------------------ #
    # Yaw alignment — per step                                            #
    # ------------------------------------------------------------------ #
    # r = face_goal_coef * cos(yaw_error)  +  forward_bonus_or_penalty
    # Example stage1: perfectly aligned (yaw_error=0) → 0.20*1.0 + 0.40 = +0.60/step
    # Example stage1: 90° off (cos=0)                 → 0.20*0.0 - 1.23 = -1.23/step
    # backwards_yaw_penalty triggers when flying fast while facing away.
    stage1_face_goal_coef: float = 0.025
    stage1_backwards_yaw_penalty_coef: float = 0.05
    stage1_backwards_yaw_speed_gate: float = 0.5   # min speed (m/s) to trigger (0.5)

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
    stage1_yaw_good_thresh: float = 0.92       # cos threshold ≈ 36.8° — above = bonus, below = penalty
    stage1_yaw_forward_bonus_coef: float = 0.10   # max bonus at perfect alignment (cos=1.0)
    stage1_yaw_forward_penalty_coef: float = 0.35  # max penalty at full misalignment (cos=-1.0)

    # Stage2+ (with pillars): looser thresh=0.9275 ≈ yaw_error < 21.9° for bonus
    #   cos=1.00 (0°)   → +0.20 * 1.0 = +0.20 bonus
    #   cos=0.93 (22°)  →  0.00 (boundary)
    #   cos=0.00 (90°)  → -1.2 * 0.9275/1.9275 = -0.578 penalty
    #   cos=-1.0 (180°) → -1.2 penalty (max)
    stage2_yaw_good_thresh: float = 0.9275        # cos threshold ≈ 21.9° — looser than stage1
    stage2_yaw_forward_bonus_coef: float = 0.20
    stage2_yaw_forward_penalty_coef: float = 0.5

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
    # Near-miss reward — one-shot                                         #
    # ------------------------------------------------------------------ #
    # Rewards passing close to a pillar without collision (skilled dodge).
    # clearance in [min, good]: ok reward. clearance >= good: full reward.
    # Example: pass at 1.5m clearance (between 1.2 and 2.5) → +0.20 once
    stage2_near_miss_clearance_min: float = 1.2
    stage2_near_miss_clearance_good: float = 2.5
    stage2_near_miss_reward_good: float = 0.0   # removed: redundant with rm_subgoal_bonus
    stage2_near_miss_reward_ok: float = 0.0     # removed: redundant with rm_subgoal_bonus

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
    # Bypass subgoals — geometry kept, rewards REPLACED BY DFA+RM        #
    # ------------------------------------------------------------------ #
    stage2_pillar_bypass_offset_m: float = 2.4
    stage2_pillar_bypass_near_radius: float = 1.2   # still used as DFA trigger radius
    stage2_pillar_bypass_near_reward: float = 3.0       # spatial pull toward safe bypass path
    stage2_pillar_bypass_reach_reward: float = 0.0     # covered by rm_subgoal_bonus at pillar pass

    # ------------------------------------------------------------------ #
    # Ring subgoals — geometry kept for logging, rewards REMOVED          #
    # ------------------------------------------------------------------ #
    stage2_pillar_ring_radius_margin: float = 1.20
    stage2_pillar_ring_near_radius: float = 0.65
    stage2_pillar_ring_near_reward: float = 0.0    # REMOVED (was 0.35)
    stage2_pillar_ring_reach_reward: float = 0.0   # REMOVED (was 4.25)
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
    # Stage1 subgoals — geometry kept as DFA checkpoints, rewards → RM   #
    # ------------------------------------------------------------------ #
    stage1_subgoal_near_radius: float = 3.2    # kept for DFA near-trigger
    stage1_subgoal_reach_radius: float = 1.5   # kept as DFA transition trigger
    stage1_subgoal_near_reward: float = 0.0    # REPLACED by rm_subgoal_bonus
    stage1_subgoal_reach_reward: float = 0.0   # REPLACED by rm_subgoal_bonus
    stage1_subgoal_dist_min: float = 8.0
    stage1_subgoal_dist_max: float = 25.0
    stage1_subgoal_count: int = 6
    stage1_sg_bonus_pillar_stage: float = 0.0  # zeroed: redundant with PBRS phi_dist; straight-line waypoints conflict with bypass trajectories

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
    clearance_reward_coef: float = 0.0   # removed: noise (0.05 << clearance_soft -1.75)
    near_pillar_speed_clearance: float = 0.60  # body clearance threshold to trigger speed penalty
    near_pillar_speed_safe: float = 0.70       # speed below this: no penalty
    near_pillar_speed_coef: float = 2.5        # max -2.0/step at clearance→0, speed=1.5m/s

    # ------------------------------------------------------------------ #
    # Pillar attention reward — per step, stage2+                         #
    # ------------------------------------------------------------------ #
    # geom: reward for keeping nearest pillar in camera FOV while close.
    # first_look: one-shot reward first time drone looks at a pillar.
    # avoidance_track: reward for tracking pillar while actively dodging.
    # Example: pillar at 3m, camera facing it (dot>0.35) → +0.03/step
    # Example: first time looking at pillar 4m away → +2.5 once
    stage2_pillar_attention_geom_reward: float = 0.0   # disabled: dead code, too weak to justify wiring
    stage2_pillar_attention_geom_dot_thresh: float = 0.35
    stage2_pillar_attention_geom_min_dist: float = 1.4
    stage2_pillar_attention_geom_max_dist: float = 4.5
    stage2_pillar_attention_geom_min_progress: float = -0.10
    stage2_pillar_attention_normalize_by_count: bool = True
    stage2_pillar_attention_reference_count: int = 5
    stage2_pillar_first_look_reward: float = 0.0   # disabled: dead code, revisit if stage2 struggles with detection
    stage2_pillar_first_look_dot_thresh: float = 0.30
    stage2_pillar_first_look_min_dist: float = 1.5
    stage2_pillar_first_look_max_dist: float = 6.0
    stage2_pillar_first_look_min_progress: float = -0.05
    stage2_pillar_avoidance_track_reward: float = 0.0
    stage2_pillar_avoidance_track_max_steps: int = 25
    stage2_pillar_avoidance_track_dot_thresh: float = 0.20
    stage2_pillar_avoidance_track_min_dist: float = 1.4
    stage2_pillar_avoidance_track_max_dist: float = 4.8
    stage2_pillar_avoidance_track_min_progress: float = -0.10
    # Post-pillar caps
    stage2_post_pillar_reward_max_steps_per_pillar: int = 12
    stage2_post_pillar_penalty_max_steps_per_pillar: int = 20
