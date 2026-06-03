from dataclasses import dataclass


@dataclass
class PillarConfig:
    num_pillars: int = 0

    # Pillar geometry
    radius_range: tuple = (0.2, 0.4)
    height_range: tuple = (4.0, 6.0)
    min_dist: float = 2.0

    # Spawn field
    spawn_bounds: tuple = (-10.0, -9.0, 10.0, 9.0)
    corridor_half_width: float = 2.5
    start_clearance: float = 2.5
    goal_clearance: float = 2.5
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
    pillar_pool_height: float = 5.0
    pillar_pool_parking_x: float = 1000.0
    pillar_pool_parking_y: float = 50.0
    pillar_pool_parking_z: float = 20.0

    # Spawn parameters (from old drone_env.py L1303-1390)
    corridor_jitter_deg: float = 11.0

    # Near-miss margin (from old drone_env.py L345)
    near_miss_margin: float = 0.50

    # Pillar bypass zone enter radius (from old drone_env.py L116)
    bypass_enter_radius: float = 2.5

    # Post-pillar pass detection margin (from old drone_env.py L394)
    post_pillar_pass_margin: float = 0.8

    # Bypass subgoal (geometry params only — reward values in RewardConfig)
    bypass_max_active: int = 1         # max simultaneous bypass subgoals
    gap_detection_enabled: bool = True
    required_gap_width: float = 1.6

    # Ring subgoal (geometry params — reward values in RewardConfig)
    ring_points_per_pillar: int = 9
    ring_min_fence_margin: float = 1.0
    ring_max_claim_per_pillar: int = 1
