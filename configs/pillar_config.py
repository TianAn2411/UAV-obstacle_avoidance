from dataclasses import dataclass


@dataclass
class PillarConfig:
    # Main (blocking-eligible) pillar budget -- the real obstacles the
    # curriculum difficulty knobs (e.g. ppo_config.yaml "Stage 2: 3 pillars")
    # refer to. uniform_pillar randomizes BOTH this and the decor budget
    # below between their own min/max each episode.
    num_pillars: int = 0
    min_pillars: int = 0
    max_pillars: int = 0
    uniform_pillar: bool = False

    # Decor pillar budget -- non-blocking, no bypass subgoal / dfa_N credit.
    # Previously carved out of the main budget probabilistically (p_blocking,
    # removed) so main-pillar count was never exact and not independently
    # tunable. Explicit budget now: a wide detour that avoids every main
    # pillar can still run into a decor one (see decor_lateral_max below),
    # so "just fly wide around everything" isn't a free pass.
    num_decor_pillars: int = 0
    min_decor_pillars: int = 0
    max_decor_pillars: int = 0

    # Pillar geometry
    radius_range: tuple = (0.1, 0.2)
    height_range: tuple = (6.0, 8.0)
    # Grid spacing band for the dense field spawn (sample_corridor_pillars) --
    # any two occupied cells are never closer than min_dist, row/column step
    # drawn uniform[min_dist, max_dist].
    min_dist: float = 2.2
    max_dist: float = 2.4

    # Spawn field. Fence is +-15 (EnvConfig.fence_x/y_min/max) -- 2m margin here
    # keeps pillars clear of near_fence penalty zone / fence collision.
    spawn_bounds: tuple = (-13.0, -13.0, 13.0, 13.0)
    corridor_half_width: float = 4.5
    start_clearance: float = 2.5
    goal_clearance: float = 1.5
    t_min: float = 0.25
    t_max: float = 0.85

    # Drone physical dimensions (for collision checks)
    drone_trigger_radius: float = 0.25
    drone_trigger_half_height: float = 0.25
    pillar_collision_margin: float = 0.09
    pillar_hard_overlap_margin: float = -0.05

    # Heading-into-pillar check
    pillar_heading_horizon_s: float = 0.8
    pillar_heading_min_speed: float = 0.1

    # Hard collision gates (used in pre-step check)
    hard_collision_front_depth: float = 0.45
    hard_collision_forward_speed: float = 0.2
    hard_collision_pillar_dist: float = 0.70
    hard_collision_pillar_fwd_align: float = 0.45
    hard_collision_closing_speed: float = 0.35

    # Pool parking (pillars parked far away when not in use)
    pillar_pool_radius: float = 0.25
    pillar_pool_height: float = 7.0
    pillar_pool_parking_x: float = 1000.0
    pillar_pool_parking_y: float = 50.0
    pillar_pool_parking_z: float = 20.0

    # Spawn parameters (from old drone_env.py L1303-1390). Was 11.0 but the
    # formula (t*(goal-start)_unjittered + lateral*normal_jittered) makes this
    # shift pillars <15cm regardless of angle -- not the corridor-angle variety
    # it was meant to add, just dead complexity. Dropped to 0.
    corridor_jitter_deg: float = 0.0

    # Near-miss margin (from old drone_env.py L345)
    near_miss_margin: float = 0.50

    # Pass-detection engagement-circle radius (raw center-to-center; from old
    # drone_env.py L116). PillarManager.update() derives
    # engage_thresh = bypass_enter_radius - drone_trigger_radius and compares
    # it against CLEARANCE (body-surface distance), not raw distance, per
    # pillar -- keeps this radius's nominal magnitude meaningful across the
    # full radius_range (0.2-0.4m) instead of silently favoring small
    # pillars. A pillar counts as "passed" when the drone enters this circle
    # (clearance_body < engage_thresh) then later exits it -- see
    # PillarManager.update()'s pass-detection block.
    bypass_enter_radius: float = 1.5

    # Dense jittered-grid spawn (sample_corridor_pillars). Main pillars fill
    # the WHOLE +-corridor_half_width band (not a narrow near-centerline
    # slalom) at min_dist..max_dist row/column spacing -- every row reserves
    # exactly 1 randomly-positioned gap so a flyable lane always exists at
    # every cross-section, with the rest of that row's cells randomly filled
    # to hit num_pillars/min_pillars/max_pillars. Real collision footprint is
    # only radius_range_max(0.4)+drone_trigger_radius(0.25) ~= 0.65m, well
    # under the grid spacing, so gaps are always drone-passable.
    #
    # Decor pillars' lateral spread -- outside +-corridor_half_width, out to
    # decor_lateral_max, sparse (decor_fill_prob) and with no gap guarantee
    # (they never need to be threadable, only present enough that a wide
    # detour around the whole main field still risks running into one). None
    # -> falls back to corridor_half_width (old behaviour).
    decor_lateral_max: float = 9.0
    decor_fill_prob: float = 0.35

    # UNUSED (from old drone_env.py L394) -- pass-detection no longer uses a
    # corridor-t-crossing margin, replaced by an enter/exit engagement-circle
    # check (see bypass_enter_radius above). Kept for reference.
    post_pillar_pass_margin: float = 0.8

    # Bypass subgoal (geometry params only — reward values in RewardConfig)
    bypass_max_active: int = 22        # max simultaneous bypass subgoals (covers all corridor pillars up to stage 5)
    gap_detection_enabled: bool = True
    required_gap_width: float = 1.6

    # Ring subgoal (geometry params — reward values in RewardConfig)
    ring_points_per_pillar: int = 9
    ring_min_fence_margin: float = 1.0
    ring_max_claim_per_pillar: int = 1

    # Dynamic (moving) pillars -- a random subset of this episode's sampled
    # pillar positions get hosted by a SEPARATE pool of non-static, Gazebo
    # VelocityControl-driven entities (PillarManager._setup_dynamic_pillars)
    # instead of the static pool. Each picks a random heading, drifts at a
    # random speed, and re-picks a new random heading every
    # dynamic_pillar_dir_change_interval_s; if it drifts past
    # dynamic_pillar_move_radius from its own spawn point it re-heads back
    # toward that point (jittered) so it wanders within its own zone rather
    # than escaping across the arena. A velocity command is only sent to
    # Gazebo on a heading change (not every step) -- physics integrates the
    # rest on its own. Off by default.
    dynamic_pillar_enabled: bool = False
    num_dynamic_pillars: int = 0             # size of the reserved dynamic-pool; clamped to actual pillar count each episode
    dynamic_pillar_speed_min: float = 0.2    # m/s
    dynamic_pillar_speed_max: float = 0.6    # m/s
    dynamic_pillar_move_radius: float = 2.0  # m, radius of allowed wander zone around spawn point
    dynamic_pillar_dir_change_interval_s: float = 2.0
