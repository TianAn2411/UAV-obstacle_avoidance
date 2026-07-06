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
    # One-shot fast-finish bonus, stacks on goal_xy_terminal_reward.
    # Piecewise in step_count at goal arrival (physical floor ~60 steps, so
    # no ramp needed below early_step):
    #   steps <= 80        → +20.0 (flat max)
    #   80 < steps <= 120   → linear ramp 20.0 → 10.0
    #   steps > 120         → +0.0
    # ALSO: on goal arrival with steps <= fast_finish_late_step, the whole
    # episode's accumulated time_penalty is refunded (net time cost = 0) —
    # see _reward_terminal(). Only fires on actual goal success; collision /
    # max_steps / out_of_fence / etc keep paying time_penalty normally, since
    # we only know the outcome — and can only fairly refund — at episode end.
    fast_finish_early_step: int = 80
    fast_finish_late_step: int = 120
    fast_finish_max_bonus: float = 20.0
    fast_finish_min_bonus: float = 10.0
    collision_penalty: float = -200.0  # was -280.0; brought down to fell_to_ground/flipped_penalty's tier
                                        # (still clearly worse than goal_xy_terminal_reward=+130) --
                                        # -280 was the single largest-magnitude terminal penalty by a wide
                                        # margin, producing heavy-tailed episode returns that correlated
                                        # with poor explained_variance + policy-update instability in
                                        # stage2 training (training_progress_stage2.csv). Does NOT apply
                                        # retroactively to an in-progress run's checkpoint -- only takes
                                        # effect on a fresh restart/new stage.
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
    # Time penalty — tiered by step_count, harsher the longer an episode  #
    # runs. Tier 1 (<120 steps) matches fast_finish_late_step -- episodes #
    # that finish within the refund window pay this and get it refunded  #
    # at goal arrival (_reward_terminal), so it's the only tier that ever #
    # nets to ~0. Tiers 2-4 penalize episodes that drag on, independent   #
    # of whether they eventually succeed.                                 #
    # ------------------------------------------------------------------ #
    # -0.13 (was -0.09): SAGE primitives already make reaching the goal
    # "easy" (zero-shot competent nav/hold-altitude), so PPO's remaining
    # differentiator is speed/efficiency -- pushed up alongside shrinking
    # alt_optimal_reward below (0.30 -> 0.04) since the two combined used to
    # make lingering in-band net +0.21/step (0.30 - 0.09), a real incentive
    # to dawdle rather than rush. New net: 0.04 - 0.13 = -0.09/step idle.
    time_penalty_base: float = -0.13          # step_count < 120
    time_penalty_tier2: float = -0.18         # 120 <= step_count < 200
    time_penalty_tier3: float = -0.23         # 200 <= step_count < 300
    time_penalty_tier4: float = -0.28         # step_count >= 300

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
    # 0.04 (was 0.30): flat in-band bonus was 3.3x time_penalty_base's old
    # magnitude, making it net +0.21/step to linger in-band regardless of
    # progress (0.30 - 0.09) -- a real anti-speed incentive stacked directly
    # against fast_finish/time_penalty's intent. Kept small and positive
    # (not 0) rather than removed: still a mild backstop against
    # gate_vertical atrophying to 0 during easy stages (see
    # Primitive.gate_min's own floor for the harder safety guarantee).
    alt_optimal_reward: float = 0.04         # +0.04/step when in [4.0, 5.5]m
    # 0.03 (was 0.10): must stay below alt_optimal_reward (0.04) -- otherwise
    # the ramp zone right outside the band pays MORE than the flat "optimal"
    # zone inside it, inverting the incentive (reward drops the instant you
    # actually settle in-band instead of rising toward it).
    alt_ramp_coef: float = 0.03              # ramp peak: 0→0.03 below, 0.03→0 above optimal zone

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
    # Example stage1: perfectly aligned (yaw_error=0) → 0.15*1.0 + 0.10 = +0.25/step
    # Example stage1: 90° off (cos=0)                 → 0.15*0.0 - 0.29 = -0.29/step
    # backwards_yaw_penalty triggers when flying fast while facing away
    # (raw/Track A only -- gated off for symbolic, see _reward_yaw_align).
    # face_goal_coef raised 0.025->0.08->0.15 (P-NSRL symbolic redesign):
    # even after the first bump, per-step yaw reward was still drowned out
    # by terminal+pbrs+rm (~150+21+50/episode) once the policy converged --
    # observed goal_xy episodes arriving with yaw_align swinging -28..+8
    # with zero visible learning pressure. Paired with the one-shot
    # stage1_goal_yaw_* arrival bonus/penalty now wired into
    # _reward_terminal (see reward_manager.py) for a second, cleaner signal.
    stage1_face_goal_coef: float = 0.15
    stage1_backwards_yaw_penalty_coef: float = 0.05
    stage1_backwards_yaw_speed_gate: float = 0.5   # min speed (m/s) to trigger (0.5)

    # Forward-goal bonus/penalty — piecewise linear ramp around good_thresh
    # cos(yaw_error) >= good_thresh → bonus (drone facing goal)
    # cos(yaw_error) <  good_thresh → penalty (drone facing away)
    #
    # Bonus:   +coef * (cos - thresh) / (1 - thresh)       — ramps 0→coef as cos goes thresh→1.0
    # Penalty: -coef * (thresh - cos) / (thresh + 1)       — ramps 0→coef as cos goes thresh→-1.0
    #
    # Stage1 (no pillars): thresh=0.935 ≈ yaw_error < 20.8° for bonus
    #   cos=1.000 (0°)     → +0.10 * 1.0 = +0.10 bonus
    #   cos=0.935 (20.8°)  →  0.00 (boundary)
    #   cos=0.000 (90°)    → -0.6 * 0.935/1.935 = -0.29 penalty
    #   cos=-1.0 (180°)    → -0.6 penalty (max) -- goal directly behind, the
    #                        case that most needs p_face to turn fast
    stage1_yaw_good_thresh: float = 0.935      # cos threshold ≈ 20.8° (arccos(0.935), was 0.92≈23° -- tightened), above = bonus, below = penalty -- fallback when ramp is off (schedule list empty); overridden by _update_yaw_good_thresh() when stage1_yaw_good_thresh_schedule_* below is set
    stage1_yaw_forward_bonus_coef: float = 0.10   # max bonus at perfect alignment (cos=1.0)
    stage1_yaw_forward_penalty_coef: float = 0.6   # max penalty at full misalignment (cos=-1.0), was 0.35 -- raised alongside face_goal_coef
    # Ramp good_thresh tighter over training (same piecewise-linear style as
    # EnvConfig.goal_xy_radius_schedule): loosest at step 0 so bonus fires
    # often while p_face is untrained, tightening to 0.982 (~11 deg) by 300k
    # steps once the policy should already be hitting the looser bar
    # reliably. Empty list = ramp disabled, stage1_yaw_good_thresh above used
    # as a flat value instead (train_manager.py's _update_yaw_good_thresh()).
    stage1_yaw_good_thresh_schedule_values: list = field(default_factory=lambda: [0.935, 0.982])
    stage1_yaw_good_thresh_schedule_steps: list = field(default_factory=lambda: [0, 300_000])

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
    # No neutral zone -- anything below ok_dot is penalized (_reward_terminal).
    stage1_goal_yaw_good_dot: float = 0.982    # ~11°
    stage1_goal_yaw_ok_dot: float = 0.93       # ~21.6°
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
    # Disabled: was penalizing ANY high total speed unconditionally
    # (no goal-direction/obstacle gate) — conflicts with wanting fast,
    # direct flight to goal. coef=0.0 kills the term; vx/vy/vz_limit
    # already bound top speed physically.
    #
    # Fall: r = coef * (vz - thresh)  when vz_down > thresh (drone dropping fast)
    # Example: falling at 1.0m/s down, thresh=0.5 → -0.2*(1.0-0.5) = -0.1/step
    speed_penalty_thresh: float = 2.5
    speed_penalty_coef: float = 0.0
    fall_vz_thresh: float = 0.5
    fall_penalty_coef: float = -0.2

    pillar_zone_radius: float = 1.85  # "near a pillar" gate for the clearance shaping below

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
    # Pillar clearance — unified single tier each side, per step stage2+  #
    # ------------------------------------------------------------------ #
    # Redesigned this session: was 3 overlapping mechanisms (collision_course
    # velocity-gated @2.0m + clearance_soft quadratic @0.45m + too_close
    # quadratic @0.20m) causing wide, "scared" avoidance. Actual collision is
    # already handled by terminal.collision_penalty -- this is light shaping
    # only, both sides of a target clearance band. Gated by pillar_zone_radius
    # (only active while a pillar is actually nearby).
    #
    # too-close side: linear penalty below close_thresh (risk of collision).
    # Example: clearance=0.1m (close_thresh=0.5) → -0.8/step
    pillar_clearance_close_thresh: float = 0.5
    pillar_clearance_close_coef: float = -1.0

    # too-wide side: linear penalty above wide_thresh (sloppy/wide detour) --
    # this is the new term: "khuyến khích đường bay gọn" (encourage a tight
    # flight path), nothing discouraged excess avoidance margin before.
    # Example: clearance=2.5m (wide_thresh=1.5) → -0.15/step, capped at -0.6
    pillar_clearance_wide_thresh: float = 1.5
    pillar_clearance_wide_coef: float = -0.15
    pillar_clearance_wide_penalty_cap: float = -0.6

    # One-shot bonus for a successfully completed pass whose minimum body
    # clearance landed in the "tight but clean" band -- rewards skillful close
    # avoidance specifically, on top of the existing rm_subgoal_bonus (which
    # fires on any pass regardless of tightness). Below pass_tight_min is
    # still risky/no bonus; above pass_tight_max already got the base
    # subgoal bonus, no extra needed.
    pillar_tight_pass_min: float = 0.5
    pillar_tight_pass_max: float = 1.2
    pillar_tight_pass_bonus: float = 3.0

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

    # Track B (symbolic_action) only, gated in RewardManager.compute() on
    # ecfg.policy_mode=="symbolic" -- penalizes CBFShield's correction
    # magnitude itself (||u_safe - u_intent||) each step, independent of
    # whether the resulting corrected trajectory stayed safe. Purpose: if
    # PPO picks gates that require a big CBF correction, it must still see a
    # cost for that choice -- otherwise CBF silently bails it out and the
    # dodge gate's marginal effect on outcome (and thus its gradient signal)
    # vanishes exactly at the states where learning to avoid matters most.
    # 0.0 = opt-in default, no behavior change until empirically tuned.
    cbf_intervention_coef: float = 0.0
