from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from obstacle_avoidance.configs.env_config import EnvConfig
from obstacle_avoidance.configs.pillar_config import PillarConfig
from obstacle_avoidance.configs.reward_config import RewardConfig
from obstacle_avoidance.utils.bridge_factory import Spawner


@dataclass
class PillarSnapshot:
    nearest_dist: float
    nearest_xy: Optional[np.ndarray]
    all_positions: list[np.ndarray]    # xy for each pillar
    all_radii: list[float]
    is_collision: bool
    collision_type: str                # "hard", "soft", "none"


class PillarManager:
    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        spawner: Spawner,
        pcfg: PillarConfig,
        ecfg: EnvConfig,
        rcfg: RewardConfig,
        env_id: int = 0,
    ) -> None:
        self._spawner = spawner
        self._p = pcfg
        self._e = ecfg
        self._r = rcfg
        self._env_id = env_id

        self._pool_initialized: bool = False
        self._pool_names: list[str] = []

        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        """Reset all per-episode tracking state."""
        self.last_pillar_metadata: list[dict] = []
        self.pillar_pass_states: dict[str, dict] = {}
        self.prev_nearest_dist: Optional[float] = None
        self.entered_pillar_zone: bool = False
        self.entered_pillar_zone_count: int = 0
        self.avoided_pillar_count: int = 0

        self._episode_start_xy: np.ndarray = np.zeros(2, dtype=np.float32)
        self._episode_goal_xy: np.ndarray = np.zeros(2, dtype=np.float32)

        # Bypass subgoal state
        self._bypass_sgs_xy: list[np.ndarray] = []
        self._bypass_reached: list[bool] = []
        self._bypass_near_rewarded: list[bool] = []
        self._bypass_idx: int = 0

        # Ring subgoal state
        self._ring_sgs: list[dict] = []
        self._ring_claimed: dict[int, int] = {}

        # Stage1 subgoal state (straight-line waypoints, num_pillars==0 only)
        self._stage1_sgs_xy: list[np.ndarray] = []
        self._stage1_reached: list[bool] = []
        self._stage1_near_rewarded: list[bool] = []
        self._stage1_idx: int = 0
        self._stage1_reward_scale: float = 1.0

        # Step output caches
        self._last_collision_snap: dict = {}
        self._last_bypass_info: dict = {"reward": 0.0, "active_subgoal": None, "clearance_gain": 0.0}
        self._last_ring_info: dict = {"reward": 0.0}
        self._last_attention_info: dict = {"attention_reward": 0.0, "post_pillar_reward": 0.0}

    # ------------------------------------------------------------------ #
    # Main interface                                                      #
    # ------------------------------------------------------------------ #

    def reset_episode(self, start: np.ndarray, goal: np.ndarray) -> None:
        """Set up pillars for a new episode."""
        self._reset_episode_state()

        self._episode_start_xy = np.array(start[:2], dtype=np.float32)
        self._episode_goal_xy = np.array(goal[:2], dtype=np.float32)

        # Stage1 subgoals always set up (guards num_pillars==0 internally)
        self._setup_stage1_subgoals(start, goal)

        p = self._p
        if p.num_pillars == 0:
            self.last_pillar_metadata = []
            return

        self._ensure_pool_initialized()

        prefix = f"pillar_env{self._env_id}__"
        sampled = self._sample_pillar_metadata(prefix)
        candidate_metadata = sampled["metadata"]

        # Build candidate_metadata with pool names
        named_metadata: list[dict] = []
        for i, m in enumerate(candidate_metadata):
            named_metadata.append({
                "name": f"pillar_env{self._env_id}__{i}",
                "x": float(m["x"]),
                "y": float(m["y"]),
                "radius": float(m.get("radius", p.pillar_pool_radius)),
                "height": float(m.get("height", p.pillar_pool_height)),
            })

        # Build batch poses: active pillars + park unused pool slots
        max_pool = max(p.num_pillars, 5)
        batch_poses: list[dict] = []
        for m in named_metadata:
            batch_poses.append({
                "name": m["name"],
                "x": float(m["x"]),
                "y": float(m["y"]),
                "z": float(0.5 * m["height"]),
                "yaw": 0.0,
            })
        for i in range(len(named_metadata), max_pool):
            batch_poses.append({
                "name": f"pillar_env{self._env_id}__{i}",
                "x": float(p.pillar_pool_parking_x + i),
                "y": float(p.pillar_pool_parking_y),
                "z": float(0.5 * p.pillar_pool_height),
                "yaw": 0.0,
            })

        try:
            self._spawner.move_pillars_batch(batch_poses)
        except Exception:
            pass

        self.last_pillar_metadata = named_metadata
        self._setup_bypass_subgoals(start, goal)
        self._setup_ring_subgoals()

    def update(
        self,
        pos: np.ndarray,
        vel_xy: np.ndarray,
        yaw: float,
        goal_xy_radius: float,
        step_count: int,
    ) -> PillarSnapshot:
        """Per-step pillar logic. Returns PillarSnapshot."""
        p = self._p
        r = self._r

        # Reset per-step accumulators
        nearest_dist: float = float("inf")
        nearest_xy: Optional[np.ndarray] = None
        nearest_meta: Optional[dict] = None
        nearest_state: Optional[dict] = None

        all_positions: list[np.ndarray] = []
        all_radii: list[float] = []

        pos_xy = np.asarray(pos[:2], dtype=np.float32)

        # Corridor direction for pass-detection t-param
        start_xy = self._episode_start_xy
        goal_xy = self._episode_goal_xy
        corridor_vec = goal_xy - start_xy
        corridor_norm = float(np.linalg.norm(corridor_vec)) + 1e-8
        corridor_dir = corridor_vec / corridor_norm
        pos_t = float(np.dot(pos_xy - start_xy, corridor_dir))

        # Goal distance (needed for pass detection)
        dist_xy = float(np.linalg.norm(goal_xy - pos_xy))

        for m in self.last_pillar_metadata:
            name = str(m.get("name", ""))
            if not name:
                continue
            pillar_xy = np.array([m["x"], m["y"]], dtype=np.float32)
            pillar_r = float(m.get("radius", p.pillar_pool_radius))
            pillar_h = float(m.get("height", p.pillar_pool_height))
            dist_to_pillar = float(np.linalg.norm(pos_xy - pillar_xy))

            all_positions.append(pillar_xy)
            all_radii.append(pillar_r)

            # Init pass state if needed
            if name not in self.pillar_pass_states:
                self.pillar_pass_states[name] = {
                    "entered": False,
                    "rewarded": False,
                    "entry_goal_dist": None,
                    "min_dist": float("inf"),
                    "pillar_t": float(np.dot(pillar_xy - start_xy, corridor_dir)),
                    "pass_step": None,
                    "post_pass_steps": 0,
                    "goal_realign_reward_steps": 0,
                    "fixation_penalty_steps": 0,
                    "last_look_pillar_dot": float("nan"),
                    "last_look_goal_dot": float("nan"),
                    "last_speed_toward_goal": float("nan"),
                    "first_look_rewarded": False,
                    "first_look_step": None,
                    "first_look_dot": float("nan"),
                    "first_look_dist": float("nan"),
                    "avoidance_track_reward_steps": 0,
                }

            state = self.pillar_pass_states[name]

            # Track min dist
            state["min_dist"] = min(float(state["min_dist"]), dist_to_pillar)

            # Mark entered when drone enters bypass enter radius
            if (not state["entered"]) and dist_to_pillar < p.bypass_enter_radius:
                state["entered"] = True
                state["entry_goal_dist"] = float(dist_xy)
                self.entered_pillar_zone_count += 1

            # Track nearest
            if dist_to_pillar < nearest_dist:
                nearest_dist = dist_to_pillar
                nearest_xy = pillar_xy
                nearest_meta = m
                nearest_state = state

        # Update entered_pillar_zone flag
        if np.isfinite(nearest_dist) and nearest_dist <= r.pillar_zone_radius:
            self.entered_pillar_zone = True

        # Clearance body
        nearest_pillar_r = float(nearest_meta.get("radius", p.pillar_pool_radius)) if nearest_meta is not None else p.pillar_pool_radius
        if np.isfinite(nearest_dist):
            clearance_body = nearest_dist - p.drone_trigger_radius - nearest_pillar_r
        else:
            clearance_body = float("inf")

        # Pillar pass detection
        reward_pillar_passed: float = 0.0
        for state in self.pillar_pass_states.values():
            if not state["entered"] or state.get("rewarded", False):
                continue
            if float(state["min_dist"]) <= p.pillar_collision_margin:
                continue
            entry_goal_dist = state.get("entry_goal_dist")
            if entry_goal_dist is None or dist_xy >= float(entry_goal_dist):
                continue
            if pos_t <= float(state.get("pillar_t", 0.0)) + p.post_pillar_pass_margin:
                continue
            reward_pillar_passed += r.passed_pillar_reward
            state["rewarded"] = True
            state["pass_step"] = int(step_count)
            self.avoided_pillar_count += 1

        # Clearance progress reward
        reward_clearance_progress: float = 0.0
        in_pillar_zone = np.isfinite(nearest_dist) and nearest_dist < p.bypass_enter_radius
        clearance_improved = (
            self.prev_nearest_dist is not None
            and np.isfinite(nearest_dist)
            and nearest_dist > self.prev_nearest_dist
        )
        # Get prev_dist_xy proxy — we don't have it, so approximate with entry_goal_dist vs dist_xy
        # Use clearance progress like old code: in zone + clearance improved + goal dist improved
        # We track prev_nearest_dist but don't have prev_dist_xy here;
        # we signal clearance_progress via snap and let reward_manager use bypass_progress for the rest
        if in_pillar_zone and clearance_improved:
            reward_clearance_progress = r.clearance_progress_reward

        # Near-miss reward
        reward_near_miss: float = 0.0
        if np.isfinite(nearest_dist):
            for state in self.pillar_pass_states.values():
                if not state.get("rewarded"):
                    continue
                min_d = float(state.get("min_dist", float("inf")))
                if min_d < r.stage2_near_miss_clearance_min:
                    continue
                if not state.get("near_miss_rewarded", False):
                    state["near_miss_rewarded"] = True
                    if min_d >= r.stage2_near_miss_clearance_good:
                        reward_near_miss += float(r.stage2_near_miss_reward_good)
                    else:
                        reward_near_miss += float(r.stage2_near_miss_reward_ok)

        # Collision check
        is_collision, coll_type, heading_info = self._check_collision(
            pos, vel_xy, nearest_meta, nearest_xy, nearest_dist
        )

        # Stage1 subgoal update (open-field stages, num_pillars==0)
        stage1_reward = self._update_stage1_subgoals(pos_xy)

        # Bypass subgoal update
        bypass_reward, bypass_near_count, bypass_reach_count = self._update_bypass_subgoals(pos_xy)
        active_bypass = self._get_active_bypass_subgoal()
        clearance_gain = 0.0
        if self.prev_nearest_dist is not None and np.isfinite(nearest_dist):
            clearance_gain = nearest_dist - self.prev_nearest_dist

        self._last_bypass_info = {
            "reward": bypass_reward,
            "active_subgoal": active_bypass,
            "clearance_gain": float(clearance_gain),
        }

        # Ring subgoal update
        ring_reward, ring_near_count, ring_reach_count = self._update_ring_subgoals(pos_xy)
        self._last_ring_info = {"reward": ring_reward}

        # Progress proxy for attention reward
        # We don't have prev_dist_xy here, approximate as 0 (attention gated by dist/dot, not progress)
        progress_proxy = 0.0

        # Clearance improving flag for attention
        clearance_improving = clearance_improved

        # Attention reward
        attention_reward = self._compute_attention_reward(
            pos, vel_xy, yaw, nearest_state, nearest_meta, nearest_dist,
            progress_proxy, step_count,
            bypass_reward=bypass_reward,
            bypass_subgoal_reward=bypass_reward,
            ring_subgoal_reward=ring_reward,
            clearance_improving=clearance_improving,
        )

        # Post-pillar reward
        post_pillar_reward = self._compute_post_pillar_reward(
            pos, vel_xy, yaw, goal_xy_radius, step_count
        )

        self._last_attention_info = {
            "attention_reward": attention_reward,
            "post_pillar_reward": post_pillar_reward,
        }

        # Build collision snap
        heading_into = heading_info.get("heading_into", False)
        d_closest = heading_info.get("d_closest", float("inf"))
        t_closest = heading_info.get("t_closest", float("nan"))
        speed = heading_info.get("speed", float(np.linalg.norm(vel_xy)))

        self._last_collision_snap = {
            "clearance_body": clearance_body,
            "heading_into": heading_into,
            "d_closest": d_closest,
            "t_closest": t_closest if t_closest is not None else float("nan"),
            "collision_radius": (nearest_pillar_r + p.drone_trigger_radius + p.pillar_collision_margin),
            "speed": speed,
            "stage1_subgoal_reward": stage1_reward,
            "pillar_passed_reward": reward_pillar_passed,
            "clearance_progress_reward": reward_clearance_progress,
            "near_miss_reward": reward_near_miss,
            "entered_pillar_zone": self.entered_pillar_zone,
            "nearest_dist": nearest_dist,
        }

        # Update prev state
        self.prev_nearest_dist = nearest_dist if np.isfinite(nearest_dist) else None

        return PillarSnapshot(
            nearest_dist=nearest_dist,
            nearest_xy=nearest_xy,
            all_positions=all_positions,
            all_radii=all_radii,
            is_collision=is_collision,
            collision_type=coll_type,
        )

    def get_collision_snap(self) -> dict:
        return self._last_collision_snap

    def get_bypass_info(self) -> dict:
        return self._last_bypass_info

    def get_ring_info(self) -> dict:
        return self._last_ring_info

    def get_attention_info(self) -> dict:
        return self._last_attention_info

    # ------------------------------------------------------------------ #
    # Spawn helpers                                                       #
    # ------------------------------------------------------------------ #

    def _pillar_pool_name(self, index: int) -> str:
        return f"pillar_env{self._env_id}__{index}"

    def _ensure_pool_initialized(self) -> None:
        """Spawn pool pillars at parking positions if not already done."""
        if self._pool_initialized:
            return
        p = self._p
        max_pool = max(p.num_pillars, 5)
        pool_names: list[str] = []
        for i in range(max_pool):
            name = self._pillar_pool_name(i)
            pool_names.append(name)
            park_x = float(p.pillar_pool_parking_x + i)
            park_y = float(p.pillar_pool_parking_y)
            try:
                if hasattr(self._spawner, "spawn_pillar"):
                    self._spawner.spawn_pillar(
                        name=name,
                        x=park_x,
                        y=park_y,
                        radius=p.pillar_pool_radius,
                        height=p.pillar_pool_height,
                    )
            except Exception:
                pass
        self._pool_names = pool_names
        self._pool_initialized = True

    def _sample_pillar_metadata(self, prefix: str) -> dict:
        """Sample pillar placement metadata with retry logic. Port from old drone_env.py L1320."""
        p = self._p
        start = self._episode_start_xy
        goal = self._episode_goal_xy

        # Base config (matches old _pillar_spawn_config)
        corridor_half_width = p.corridor_half_width
        min_pillar_dist = p.min_dist
        fallback_used = False
        accepted_partial_spawn = False
        actual_count = 0
        metadata: list[dict] = []

        min_required_pillars = max(2, p.num_pillars - 2)

        for attempt in range(1, 4):
            attempt_corridor = corridor_half_width
            attempt_min_dist = min_pillar_dist
            if attempt == 2:
                attempt_min_dist = 2.0
                fallback_used = True
            elif attempt == 3:
                attempt_corridor = 3.2
                attempt_min_dist = 1.6
                fallback_used = True

            try:
                metadata = self._spawner.sample_random_field_metadata(
                    num_pillars=p.num_pillars,
                    start=start,
                    goal=goal,
                    name_prefix=prefix,
                    corridor_half_width=attempt_corridor,
                    start_clearance=p.start_clearance,
                    goal_clearance=p.goal_clearance,
                    t_min=p.t_min,
                    t_max=p.t_max,
                    spawn_bounds=p.spawn_bounds,
                    pillar_radius_range=p.radius_range,
                    pillar_height_range=p.height_range,
                    min_dist=attempt_min_dist,
                    corridor_jitter_deg=p.corridor_jitter_deg,
                )
            except Exception:
                metadata = []

            actual_count = len(metadata)
            if actual_count >= min_required_pillars:
                accepted_partial_spawn = actual_count < p.num_pillars
                return {
                    "ok": True,
                    "metadata": metadata,
                    "actual_count": actual_count,
                    "fallback_used": fallback_used,
                    "accepted_partial_spawn": accepted_partial_spawn,
                }

        return {
            "ok": False,
            "metadata": metadata,
            "actual_count": actual_count,
            "fallback_used": fallback_used,
            "accepted_partial_spawn": accepted_partial_spawn,
            "min_required_pillars": min_required_pillars,
        }

    # ------------------------------------------------------------------ #
    # Subgoal setup                                                       #
    # ------------------------------------------------------------------ #

    def _setup_bypass_subgoals(self, start: np.ndarray, goal: np.ndarray) -> None:
        """Set up bypass subgoals. Port from old drone_env.py L1503-1613."""
        self._bypass_sgs_xy = []
        self._bypass_reached = []
        self._bypass_near_rewarded = []
        self._bypass_idx = 0

        p = self._p
        r = self._r

        if p.num_pillars == 0 or not self.last_pillar_metadata:
            return

        start_xy = np.array(start[:2], dtype=np.float32)
        goal_xy = np.array(goal[:2], dtype=np.float32)
        corridor = goal_xy - start_xy
        corridor_len = float(np.linalg.norm(corridor)) + 1e-8
        corridor_dir = corridor / corridor_len
        lateral_dir = np.array([-corridor_dir[1], corridor_dir[0]], dtype=np.float32)

        metadata = list(self.last_pillar_metadata)
        candidates: list[dict] = []

        for m in metadata:
            pillar_xy = np.array([m["x"], m["y"]], dtype=np.float32)
            pillar_r = float(m.get("radius", p.pillar_pool_radius))

            t = float(np.dot(pillar_xy - start_xy, corridor_dir))
            if t <= 2.0 or t >= corridor_len - 2.0:
                continue

            cross = float(np.dot(pillar_xy - start_xy, lateral_dir))
            if abs(cross) > 3.5:
                continue

            side_candidates: list[dict] = []
            for side_name, side_sign in (("left", 1.0), ("right", -1.0)):
                sg = pillar_xy + lateral_dir * (side_sign * float(r.stage2_pillar_bypass_offset_m))
                margin = self._fence_margin_xy(sg)
                if margin < 1.0:
                    continue

                gap_penalty = 0.0
                if p.gap_detection_enabled:
                    blocked_gap = False
                    for other in metadata:
                        if other is m:
                            continue
                        other_xy = np.array([other["x"], other["y"]], dtype=np.float32)
                        other_r = float(other.get("radius", p.pillar_pool_radius))
                        center_dist = float(np.linalg.norm(other_xy - pillar_xy))
                        gap_width = center_dist - pillar_r - other_r
                        if gap_width >= p.required_gap_width:
                            continue
                        other_cross = float(np.dot(other_xy - pillar_xy, lateral_dir))
                        if side_sign * other_cross <= 0.0:
                            continue
                        blocked_gap = True
                        break
                    if blocked_gap:
                        gap_penalty = 1000.0

                side_candidates.append({
                    "score": margin - gap_penalty,
                    "xy": sg.astype(np.float32),
                    "side": side_name,
                    "margin": margin,
                })

            if not side_candidates:
                continue

            side_candidates.sort(key=lambda item: item["score"], reverse=True)
            best = side_candidates[0]
            if best["score"] < 0.0:
                continue

            candidates.append({
                "t": t,
                "xy": best["xy"],
                "side": best["side"],
                "pillar_xy": pillar_xy.astype(np.float32),
                "margin": best["margin"],
            })

        candidates.sort(key=lambda c: c["t"])

        for cand in candidates[: p.bypass_max_active]:
            self._bypass_sgs_xy.append(cand["xy"])
            self._bypass_reached.append(False)
            self._bypass_near_rewarded.append(False)

    def _setup_ring_subgoals(self) -> None:
        """Set up ring subgoals around pillars. Port from old drone_env.py L2205-2252."""
        self._ring_sgs = []
        self._ring_claimed = {}

        p = self._p
        r = self._r

        if p.num_pillars == 0 or not self.last_pillar_metadata:
            return

        metadata = list(self.last_pillar_metadata)
        n_active = min(len(metadata), r.stage2_pillar_ring_max_active_pillars)

        for pillar_idx, m in enumerate(metadata[:n_active]):
            pillar_xy = np.array([m["x"], m["y"]], dtype=np.float32)
            pillar_r = float(m.get("radius", p.pillar_pool_radius))
            ring_r = pillar_r + float(r.stage2_pillar_ring_radius_margin)

            n_pts = int(p.ring_points_per_pillar)
            for k in range(n_pts):
                angle = 2.0 * math.pi * k / n_pts
                pt = pillar_xy + ring_r * np.array(
                    [math.cos(angle), math.sin(angle)], dtype=np.float32
                )
                if self._fence_margin_xy(pt) < p.ring_min_fence_margin:
                    continue
                self._ring_sgs.append({
                    "xy": pt,
                    "pillar_idx": pillar_idx,
                    "near_rewarded": False,
                    "reach_rewarded": False,
                })
            self._ring_claimed[pillar_idx] = 0

    # ------------------------------------------------------------------ #
    # Per-step subgoal updates                                            #
    # ------------------------------------------------------------------ #

    def _setup_stage1_subgoals(self, start: np.ndarray, goal: np.ndarray) -> None:
        """Place 2-4 equidistant waypoints along start→goal line (num_pillars==0 only)."""
        self._stage1_sgs_xy = []
        self._stage1_reached = []
        self._stage1_near_rewarded = []
        self._stage1_idx = 0
        self._stage1_reward_scale = 1.0

        r = self._r
        if self._p.num_pillars != 0:
            return  # only for open-field stages

        start_xy = np.asarray(start[:2], dtype=np.float32)
        goal_xy  = np.asarray(goal[:2],  dtype=np.float32)
        dist_xy  = float(np.linalg.norm(goal_xy - start_xy))

        if not (r.stage1_subgoal_dist_min <= dist_xy <= r.stage1_subgoal_dist_max):
            return

        # Reward scale decays as distance increases (easier = more reward)
        dist_span  = max(1e-6, r.stage1_subgoal_dist_max - r.stage1_subgoal_dist_min)
        dist_alpha = float(np.clip((dist_xy - r.stage1_subgoal_dist_min) / dist_span, 0.0, 1.0))
        self._stage1_reward_scale = 1.0  # fixed scale; configurable via reward_config if needed

        n = 2 if dist_xy < 10.0 else (3 if dist_xy < 11.0 else 4)
        for i in range(1, n + 1):
            t = float(i) / float(n + 1)
            sg_xy = (1.0 - t) * start_xy + t * goal_xy
            self._stage1_sgs_xy.append(sg_xy)

        self._stage1_reached      = [False] * len(self._stage1_sgs_xy)
        self._stage1_near_rewarded = [False] * len(self._stage1_sgs_xy)

    def _update_stage1_subgoals(self, pos_xy: np.ndarray) -> float:
        """Check current subgoal, award near/reach rewards, advance index. Returns reward."""
        if not self._stage1_sgs_xy or self._stage1_idx >= len(self._stage1_sgs_xy):
            return 0.0

        r      = self._r
        reward = 0.0
        i      = self._stage1_idx
        sg_xy  = self._stage1_sgs_xy[i]
        d      = float(np.linalg.norm(np.asarray(pos_xy, dtype=np.float32) - sg_xy))

        if not self._stage1_near_rewarded[i] and d <= r.stage1_subgoal_near_radius:
            self._stage1_near_rewarded[i] = True
            reward += r.stage1_subgoal_near_reward * self._stage1_reward_scale

        if d <= r.stage1_subgoal_reach_radius:
            self._stage1_reached[i] = True
            reward += r.stage1_subgoal_reach_reward * self._stage1_reward_scale
            self._stage1_idx = min(len(self._stage1_sgs_xy), self._stage1_idx + 1)

        return reward
    def _get_active_bypass_subgoal(self) -> Optional[np.ndarray]:
        if not self._bypass_sgs_xy:
            return None
        i = self._bypass_idx
        if i < 0 or i >= len(self._bypass_sgs_xy):
            return None
        return self._bypass_sgs_xy[i]

    def _update_bypass_subgoals(
        self, pos_xy: np.ndarray
    ) -> tuple[float, int, int]:
        """Update bypass subgoal states and return (reward, hit_near, hit_reach)."""
        if not self._bypass_sgs_xy:
            return 0.0, 0, 0

        i = self._bypass_idx
        if i < 0 or i >= len(self._bypass_sgs_xy):
            return 0.0, 0, 0

        r = self._r
        sg = self._bypass_sgs_xy[i]
        d = float(np.linalg.norm(np.asarray(pos_xy, dtype=np.float32) - sg))

        reward = 0.0
        hit_near = 0
        hit_reach = 0

        if d <= float(r.stage2_pillar_bypass_near_radius):
            if not self._bypass_near_rewarded[i]:
                reward += float(r.stage2_pillar_bypass_near_reward)
                self._bypass_near_rewarded[i] = True
                hit_near = 1

            if not self._bypass_reached[i]:
                reward += float(r.stage2_pillar_bypass_reach_reward)
                self._bypass_reached[i] = True
                self._bypass_idx = i + 1
                hit_reach = 1

        return reward, hit_near, hit_reach

    def _update_ring_subgoals(
        self, pos_xy: np.ndarray
    ) -> tuple[float, int, int]:
        """Update ring subgoal states and return (reward, hit_near, hit_reach)."""
        if not self._ring_sgs:
            return 0.0, 0, 0

        r = self._r
        p = self._p
        pos = np.asarray(pos_xy, dtype=np.float32)
        total_reward = 0.0
        hit_near = 0
        hit_reach = 0

        for sg in self._ring_sgs:
            d = float(np.linalg.norm(pos - sg["xy"]))
            pillar_idx = int(sg["pillar_idx"])
            claim_count = int(self._ring_claimed.get(pillar_idx, 0))

            if d > float(r.stage2_pillar_ring_near_radius):
                continue

            if not sg["near_rewarded"]:
                sg["near_rewarded"] = True
                hit_near += 1
                total_reward += float(r.stage2_pillar_ring_near_reward)

            if (
                not sg["reach_rewarded"]
                and claim_count < p.ring_max_claim_per_pillar
            ):
                sg["reach_rewarded"] = True
                self._ring_claimed[pillar_idx] = claim_count + 1
                hit_reach += 1
                total_reward += float(r.stage2_pillar_ring_reach_reward)

        return total_reward, hit_near, hit_reach

    # ------------------------------------------------------------------ #
    # Attention / post-pillar rewards                                     #
    # ------------------------------------------------------------------ #

    def _compute_attention_reward(
        self,
        pos: np.ndarray,
        vel_xy: np.ndarray,
        yaw: float,
        nearest_state: Optional[dict],
        nearest_meta: Optional[dict],
        nearest_dist: float,
        progress: float,
        step_count: int,
        bypass_reward: float = 0.0,
        bypass_subgoal_reward: float = 0.0,
        ring_subgoal_reward: float = 0.0,
        clearance_improving: bool = False,
    ) -> float:
        """Compute pillar attention reward. Port from old drone_env.py L1771-1944."""
        r = self._r
        p = self._p

        if self.avoided_pillar_count >= p.num_pillars:
            return 0.0
        if p.num_pillars <= 0:
            return 0.0
        if nearest_state is None or nearest_meta is None:
            return 0.0
        if not np.isfinite(nearest_dist):
            return 0.0

        state = nearest_state

        # Assume obstacle visible if dist is finite and within attention range
        obstacle_visible = (
            float(r.stage2_pillar_attention_geom_min_dist)
            <= nearest_dist
            <= float(r.stage2_pillar_attention_geom_max_dist)
        )
        if not obstacle_visible:
            # Also allow first_look range
            obstacle_visible = (
                float(r.stage2_pillar_first_look_min_dist)
                <= nearest_dist
                <= float(r.stage2_pillar_first_look_max_dist)
            )
        if not obstacle_visible:
            return 0.0

        if state.get("rewarded", False):
            return 0.0

        # Attention scale
        attention_scale = 1.0
        if r.stage2_pillar_attention_normalize_by_count:
            attention_scale *= min(
                1.0,
                float(r.stage2_pillar_attention_reference_count)
                / max(1.0, float(p.num_pillars)),
            )

        pos_xy = np.asarray(pos[:2], dtype=np.float32)
        pillar_xy = np.array([nearest_meta["x"], nearest_meta["y"]], dtype=np.float32)
        forward_xy = np.array(
            [math.cos(float(yaw)), math.sin(float(yaw))],
            dtype=np.float32,
        )
        to_pillar = pillar_xy - pos_xy
        to_pillar_norm = max(float(np.linalg.norm(to_pillar)), 1e-6)
        to_pillar_dir = to_pillar / to_pillar_norm
        look_pillar_dot = float(np.dot(forward_xy, to_pillar_dir))

        reward_geom_attention = 0.0
        reward_first_look = 0.0
        reward_avoidance_track = 0.0
        first_look_fired = False

        # Geometric attention
        geom_gate_ok = bool(
            r.stage2_pillar_attention_geom_min_dist
            <= float(nearest_dist)
            <= r.stage2_pillar_attention_geom_max_dist
            and look_pillar_dot >= r.stage2_pillar_attention_geom_dot_thresh
            and progress >= r.stage2_pillar_attention_geom_min_progress
        )
        if geom_gate_ok:
            geom_dot_span = max(
                1e-6,
                1.0 - float(r.stage2_pillar_attention_geom_dot_thresh),
            )
            geom_scale = float(
                np.clip(
                    (look_pillar_dot - float(r.stage2_pillar_attention_geom_dot_thresh))
                    / geom_dot_span,
                    0.0,
                    1.0,
                )
            )
            reward_geom_attention = (
                float(r.stage2_pillar_attention_geom_reward)
                * attention_scale
                * geom_scale
            )

        # First-look reward
        first_gate_ok = bool(
            (not state.get("first_look_rewarded", False))
            and r.stage2_pillar_first_look_min_dist
            <= float(nearest_dist)
            <= r.stage2_pillar_first_look_max_dist
            and look_pillar_dot >= r.stage2_pillar_first_look_dot_thresh
            and progress >= r.stage2_pillar_first_look_min_progress
        )
        if first_gate_ok:
            reward_first_look = float(r.stage2_pillar_first_look_reward) * attention_scale
            state["first_look_rewarded"] = True
            state["first_look_step"] = int(step_count)
            state["first_look_dot"] = float(look_pillar_dot)
            state["first_look_dist"] = float(nearest_dist)
            first_look_fired = True

        # Avoidance track reward
        avoidance_progress_ok = (
            bypass_reward > 0.0
            or bypass_subgoal_reward > 0.0
            or ring_subgoal_reward > 0.0
            or bool(clearance_improving)
        )
        track_gate_ok = bool(
            state.get("first_look_rewarded", False)
            and (not first_look_fired)
            and int(state.get("avoidance_track_reward_steps", 0))
            < int(r.stage2_pillar_avoidance_track_max_steps)
            and r.stage2_pillar_avoidance_track_min_dist
            <= float(nearest_dist)
            <= r.stage2_pillar_avoidance_track_max_dist
            and look_pillar_dot >= r.stage2_pillar_avoidance_track_dot_thresh
            and progress >= r.stage2_pillar_avoidance_track_min_progress
        )
        if track_gate_ok and avoidance_progress_ok:
            reward_avoidance_track = float(r.stage2_pillar_avoidance_track_reward) * attention_scale
            state["avoidance_track_reward_steps"] = (
                int(state.get("avoidance_track_reward_steps", 0)) + 1
            )

        return reward_geom_attention + reward_first_look + reward_avoidance_track

    def _compute_post_pillar_reward(
        self,
        pos: np.ndarray,
        vel_xy: np.ndarray,
        yaw: float,
        goal_xy_radius: float,
        step_count: int,
    ) -> float:
        """Compute post-pillar reorientation reward/penalty. Port from old drone_env.py L1656-1769."""
        r = self._r
        p = self._p

        if p.num_pillars <= 0:
            return 0.0
        if not self.last_pillar_metadata:
            return 0.0

        pos_xy = np.asarray(pos[:2], dtype=np.float32)
        goal_xy = self._episode_goal_xy
        dist_xy = float(np.linalg.norm(goal_xy - pos_xy))

        if dist_xy < max(float(goal_xy_radius) * 1.5, 1.0):
            return 0.0

        forward_xy = np.array(
            [math.cos(float(yaw)), math.sin(float(yaw))],
            dtype=np.float32,
        )
        goal_dir_xy = goal_xy - pos_xy
        goal_norm = float(np.linalg.norm(goal_dir_xy))
        if goal_norm <= 1e-6:
            return 0.0
        goal_dir_xy = goal_dir_xy / goal_norm
        speed_toward_goal = float(np.dot(np.asarray(vel_xy, dtype=np.float32), goal_dir_xy))
        look_goal_dot = float(np.dot(forward_xy, goal_dir_xy))

        # Find most recently rewarded pillar still in window
        active_name: Optional[str] = None
        active_meta: Optional[dict] = None
        active_state: Optional[dict] = None
        active_t: float = float("-inf")

        for meta in self.last_pillar_metadata:
            name = str(meta.get("name", ""))
            if not name:
                continue
            state = self.pillar_pass_states.get(name)
            if not state or not state.get("rewarded", False):
                continue
            if state.get("pass_step") is None:
                state["pass_step"] = int(step_count)
            post_pass_steps = int(step_count - int(state["pass_step"]))
            if post_pass_steps < 0 or post_pass_steps > r.stage2_post_pillar_window_steps:
                continue
            pillar_t = float(state.get("pillar_t", float("-inf")))
            if pillar_t > active_t:
                active_t = pillar_t
                active_name = name
                active_meta = meta
                active_state = state

        if active_name is None or active_meta is None or active_state is None:
            return 0.0

        pillar_xy = np.array([active_meta["x"], active_meta["y"]], dtype=np.float32)
        to_pillar = pillar_xy - pos_xy
        pillar_norm = float(np.linalg.norm(to_pillar))
        if pillar_norm <= 1e-6:
            return 0.0
        to_pillar_dir = to_pillar / pillar_norm
        look_pillar_dot = float(np.dot(forward_xy, to_pillar_dir))

        reward_goal_realign = 0.0
        penalty_fixation = 0.0

        if (
            look_pillar_dot > r.stage2_post_pillar_look_pillar_dot_thresh
            and int(active_state.get("fixation_penalty_steps", 0))
            < r.stage2_post_pillar_penalty_max_steps_per_pillar
        ):
            penalty_fixation = -float(r.stage2_post_pillar_fixation_penalty)
            active_state["fixation_penalty_steps"] = (
                int(active_state.get("fixation_penalty_steps", 0)) + 1
            )

        if (
            look_goal_dot > r.stage2_post_pillar_look_goal_dot_thresh
            and speed_toward_goal > r.stage2_post_pillar_speed_goal_thresh
            and int(active_state.get("goal_realign_reward_steps", 0))
            < r.stage2_post_pillar_reward_max_steps_per_pillar
        ):
            reward_goal_realign = float(r.stage2_post_pillar_goal_realign_reward)
            active_state["goal_realign_reward_steps"] = (
                int(active_state.get("goal_realign_reward_steps", 0)) + 1
            )

        active_state["last_look_pillar_dot"] = look_pillar_dot
        active_state["last_look_goal_dot"] = look_goal_dot
        active_state["last_speed_toward_goal"] = speed_toward_goal

        return reward_goal_realign + penalty_fixation

    # ------------------------------------------------------------------ #
    # Collision geometry helpers                                          #
    # ------------------------------------------------------------------ #

    def _check_collision(
        self,
        pos: np.ndarray,
        vel_xy: np.ndarray,
        nearest_meta: Optional[dict],
        nearest_xy: Optional[np.ndarray],
        nearest_dist: float,
    ) -> tuple[bool, str, dict]:
        """Check collision against nearest pillar. Simplified port from old drone_env.py L2050-2203."""
        p = self._p

        default_heading: dict = {
            "heading_into": False,
            "d_closest": float("inf"),
            "t_closest": float("nan"),
            "speed": float(np.linalg.norm(vel_xy[:2])),
            "trigger_radius": float("nan"),
        }

        if nearest_meta is None or nearest_xy is None or not np.isfinite(nearest_dist):
            return False, "none", default_heading

        pos_arr = np.asarray(pos, dtype=np.float32)
        if pos_arr.shape[0] < 3 or not np.all(np.isfinite(pos_arr[:3])):
            return False, "none", default_heading

        pillar_r = float(nearest_meta.get("radius", p.pillar_pool_radius))
        pillar_h = float(nearest_meta.get("height", p.pillar_pool_height))
        drone_r = p.drone_trigger_radius
        drone_half_h = p.drone_trigger_half_height
        collision_margin = p.pillar_collision_margin
        hard_overlap_margin = p.pillar_hard_overlap_margin

        drone_z = float(pos_arr[2])
        drone_z_min = drone_z - drone_half_h
        drone_z_max = drone_z + drone_half_h
        z_overlap = bool(drone_z_max >= 0.0 and drone_z_min <= pillar_h)

        base_radius = pillar_r + drone_r
        clearance = nearest_dist - base_radius

        heading_into, heading_info = self._is_heading_into_pillar(
            drone_xy=pos_arr[:2],
            vel_xy=np.asarray(vel_xy, dtype=np.float32)[:2],
            pillar_xy=nearest_xy,
            pillar_radius=pillar_r,
        )

        combined_heading: dict = {
            "heading_into": bool(heading_into),
            "d_closest": heading_info.get("d_closest", float("inf")),
            "t_closest": heading_info.get("t_closest", float("nan")),
            "speed": heading_info.get("speed", float(np.linalg.norm(vel_xy[:2]))),
            "trigger_radius": heading_info.get("trigger_radius", base_radius + collision_margin),
        }

        if z_overlap:
            hard_overlap = bool(clearance <= hard_overlap_margin)
            if hard_overlap:
                return True, "hard", combined_heading

            collision_radius = base_radius + collision_margin
            inside_collision_margin = bool(nearest_dist <= collision_radius)
            if inside_collision_margin and heading_into:
                return True, "soft", combined_heading

        return False, "none", combined_heading

    def _is_heading_into_pillar(
        self,
        drone_xy: np.ndarray,
        vel_xy: np.ndarray,
        pillar_xy: np.ndarray,
        pillar_radius: float,
    ) -> tuple[bool, dict]:
        """Check if velocity ray intersects pillar trigger zone. Port from old drone_env.py L1946-2048."""
        p = self._p
        try:
            pos = np.asarray(drone_xy, dtype=np.float32)
            v = np.asarray(vel_xy, dtype=np.float32)
            c = np.asarray(pillar_xy, dtype=np.float32)

            if pos.shape[0] < 2 or v.shape[0] < 2 or c.shape[0] < 2:
                return False, {
                    "reason": "invalid_input", "speed": float("nan"),
                    "t_closest": float("nan"), "d_closest": float("inf"),
                    "trigger_radius": float("nan"),
                }

            if not (
                np.all(np.isfinite(pos[:2]))
                and np.all(np.isfinite(v[:2]))
                and np.all(np.isfinite(c[:2]))
            ):
                return False, {
                    "reason": "invalid_input", "speed": float("nan"),
                    "t_closest": float("nan"), "d_closest": float("inf"),
                    "trigger_radius": float("nan"),
                }

            drone_radius = p.drone_trigger_radius
            threshold = p.pillar_collision_margin
            horizon_s = p.pillar_heading_horizon_s
            min_speed = p.pillar_heading_min_speed

            speed = float(np.linalg.norm(v[:2]))
            trigger_radius = float(pillar_radius) + float(drone_radius) + float(threshold)

            if speed < float(min_speed):
                return False, {
                    "reason": "too_slow", "speed": speed,
                    "t_closest": float("nan"), "d_closest": float("inf"),
                    "trigger_radius": trigger_radius,
                }

            rel = c[:2] - pos[:2]
            speed_sq = max(speed * speed, 1e-6)
            t_closest = float(np.dot(rel, v[:2]) / speed_sq)

            if t_closest < 0.0 or t_closest > float(horizon_s):
                t_clip = float(np.clip(t_closest, 0.0, float(horizon_s)))
                closest = pos[:2] + v[:2] * t_clip
                d_closest = float(np.linalg.norm(c[:2] - closest))
                return False, {
                    "reason": "behind_or_outside_horizon", "speed": speed,
                    "t_closest": t_closest, "d_closest": d_closest,
                    "trigger_radius": trigger_radius,
                }

            closest = pos[:2] + v[:2] * t_closest
            d_closest = float(np.linalg.norm(c[:2] - closest))
            heading_into = bool(d_closest <= trigger_radius)

            return heading_into, {
                "reason": "heading_into_zone" if heading_into else "misses_zone",
                "speed": speed, "t_closest": t_closest, "d_closest": d_closest,
                "trigger_radius": trigger_radius,
            }

        except Exception as exc:
            return False, {
                "reason": f"exception:{exc}", "speed": float("nan"),
                "t_closest": float("nan"), "d_closest": float("inf"),
                "trigger_radius": float("nan"),
            }

    # ------------------------------------------------------------------ #
    # Geometry utilities                                                  #
    # ------------------------------------------------------------------ #

    def _fence_margin_xy(self, pos_xy: np.ndarray) -> float:
        """Minimum distance from pos to XY fence boundary."""
        e = self._e
        x, y = float(pos_xy[0]), float(pos_xy[1])
        return min(
            x - e.fence_x_min,
            e.fence_x_max - x,
            y - e.fence_y_min,
            e.fence_y_max - y,
        )
