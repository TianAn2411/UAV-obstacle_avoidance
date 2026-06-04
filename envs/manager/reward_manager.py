from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from obstacle_avoidance.configs.env_config import EnvConfig
from obstacle_avoidance.configs.reward_config import RewardConfig
import math

@dataclass
class RewardComponents:
    # Navigation
    progress: float = 0.0
    time: float = 0.0
    velocity_goal: float = 0.0
    heading_goal: float = 0.0
    # Altitude
    ground: float = 0.0
    altitude: float = 0.0
    # Action quality
    smooth: float = 0.0
    speed_penalty: float = 0.0
    fall_penalty: float = 0.0
    yaw_rate_penalty: float = 0.0
    # Goal alignment
    yaw_align: float = 0.0
    face_progress: float = 0.0
    backwards_yaw: float = 0.0
    # Spatial penalties
    lateral: float = 0.0
    near_fence: float = 0.0
    start_zone: float = 0.0
    # Pillar-specific
    pillar_too_close: float = 0.0
    pillar_clearance_soft: float = 0.0
    pillar_passed: float = 0.0
    clearance_progress: float = 0.0
    near_miss: float = 0.0
    obstacle_approach: float = 0.0
    collision_course: float = 0.0
    clearance_body: float = 0.0
    # Subgoal rewards
    stage1_subgoal: float = 0.0
    bypass_subgoal: float = 0.0
    ring_subgoal: float = 0.0
    bypass_decision: float = 0.0
    bypass_progress: float = 0.0
    # Pillar attention / behavior
    obstacle_visibility: float = 0.0
    obstacle_slowdown: float = 0.0
    pillar_attention: float = 0.0
    post_pillar: float = 0.0
    near_pillar_speed: float = 0.0
    goal_tunnel_penalty: float = 0.0
    clearance_escape: float = 0.0
    fast_clean_dodge: float = 0.0
    too_slow_penalty: float = 0.0
    # Terminal
    terminal: float = 0.0
    # Meta
    total: float = 0.0


@dataclass
class StepState:
    """All inputs RewardManager needs — passed by TrainManager."""
    pos: np.ndarray                         # [x, y, z]
    vel: np.ndarray                         # [vx, vy, vz]
    yaw: float
    prev_pos: np.ndarray
    prev_action: np.ndarray
    action: np.ndarray
    step_count: int
    global_step: int
    stage_index: int
    dist_xy: float
    prev_dist_xy: float
    front_depth: float
    depth_sector: np.ndarray               # shape (9,)
    nearest_pillar_dist: Optional[float]
    nearest_pillar_xy: Optional[np.ndarray]
    pillar_collision_snap: dict             # from PillarManager
    bypass_subgoal_info: dict              # from PillarManager
    ring_subgoal_info: dict                # from PillarManager
    attention_info: dict                   # from PillarManager
    reset_info: dict                       # from ResetManager
    done_reason: str                       # empty if not terminal
    goal_xy_radius: float
    is_terminal: bool
    is_truncated: bool
    goal: np.ndarray                       # current episode goal [x, y, z]
    start: np.ndarray                      # episode start [x, y, z]
    num_pillars: int                       # active pillars this episode
    horizontal_speed: float               # precomputed ||vel_xy||
    min_depth: float                       # global min depth this step
    final_yaw_rate: float                 # actual yaw rate sent to bridge
    yaw_error: float                       # wrap_pi(desired_yaw - current_yaw)


class RewardManager:
    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def __init__(self, rcfg: RewardConfig, ecfg: EnvConfig) -> None:
        self._r = rcfg
        self._e = ecfg
        # Per-episode tracked state (reset by reset_episode)
        self._has_been_airborne: bool = False
        self._prev_min_depth: float = float("inf")
        self._prev_horizontal_speed: float = 0.0
        self._was_near_obstacle: bool = False

    def reset_episode(self) -> None:
        self._has_been_airborne = False
        self._prev_min_depth = float("inf")
        self._prev_horizontal_speed = 0.0
        self._was_near_obstacle = False

    # ------------------------------------------------------------------ #
    # Main interface                                                      #
    # ------------------------------------------------------------------ #

    def compute(self, state: StepState) -> tuple[float, RewardComponents]:
        c = RewardComponents()
        obstacle_enabled = state.num_pillars > 0

        # --- Navigation ---
        progress = state.prev_dist_xy - state.dist_xy
        c.progress = self._reward_progress(state, progress)
        c.time = self._reward_time(state.step_count, state)
        c.velocity_goal = self._reward_velocity_goal(state)
        c.heading_goal = self._reward_heading_goal(state)

        # --- Altitude ---
        c.ground, c.altitude = self._reward_altitude(state)

        # --- Action quality ---
        c.smooth = self._reward_smooth(state.action, state.prev_action)
        c.yaw_rate_penalty = float(self._r.yaw_rate_penalty_coef) * abs(state.final_yaw_rate)

        # --- Yaw alignment ---
        c.yaw_align = self._reward_yaw_align(state)

        # --- Spatial penalties ---
        c.lateral, c.near_fence, c.start_zone = self._reward_spatial_penalties(state)

        # --- Speed / fall ---
        speed = float(np.linalg.norm(state.vel))
        if speed > self._r.speed_penalty_thresh:
            c.speed_penalty = self._r.speed_penalty_coef * (speed - self._r.speed_penalty_thresh)
        if state.vel[2] > self._r.fall_vz_thresh:
            c.fall_penalty = self._r.fall_penalty_coef * (float(state.vel[2]) - self._r.fall_vz_thresh)

        # --- Pillar clearance / collision course ---
        if obstacle_enabled:
            (
                c.pillar_too_close,
                c.pillar_clearance_soft,
                c.collision_course,
                c.clearance_body,
                c.near_pillar_speed,
            ) = self._reward_pillar_clearance(state)
            # Scale progress/yaw/velocity by danger proximity
            clearance_body = state.pillar_collision_snap.get("clearance_body", float("inf"))
            if np.isfinite(clearance_body) and clearance_body < self._r.clearance_body_safe:
                danger_scale = max(0.0, min(1.0, clearance_body / self._r.clearance_body_safe))
                c.progress *= max(0.25, danger_scale)
                c.yaw_align *= max(0.30, danger_scale)
                c.velocity_goal *= max(0.50, danger_scale)
            c.pillar_passed = float(state.pillar_collision_snap.get("pillar_passed_reward", 0.0))
            c.clearance_progress = float(state.pillar_collision_snap.get("clearance_progress_reward", 0.0))
            c.near_miss = float(state.pillar_collision_snap.get("near_miss_reward", 0.0))

        # --- Subgoals ---
        c.stage1_subgoal, c.bypass_subgoal, c.ring_subgoal = self._reward_subgoals(state)
        c.bypass_decision = self._reward_bypass_decision(state)
        c.bypass_progress = self._reward_bypass_progress(state)

        # --- Pillar behavior ---
        if obstacle_enabled:
            beh = self._reward_pillar_behavior(state)
            c.obstacle_visibility = beh.get("visibility", 0.0)
            c.obstacle_slowdown = beh.get("slowdown", 0.0)
            c.pillar_attention = beh.get("attention", 0.0)
            c.post_pillar = beh.get("post_pillar", 0.0)
            c.too_slow_penalty = beh.get("too_slow", 0.0)

        # --- Terminal ---
        c.terminal = self._reward_terminal(state)

        # --- Update tracked state ---
        self._prev_min_depth = state.min_depth
        self._prev_horizontal_speed = state.horizontal_speed
        if float(state.pos[2]) > self._e.airborne_z:
            self._has_been_airborne = True

        c.total = (
            c.progress + c.time + c.velocity_goal + c.heading_goal
            + c.ground + c.altitude
            + c.smooth + c.yaw_rate_penalty + c.speed_penalty + c.fall_penalty
            + c.yaw_align
            + c.lateral + c.near_fence + c.start_zone
            + c.pillar_too_close + c.pillar_clearance_soft + c.collision_course
            + c.clearance_body + c.near_pillar_speed
            + c.pillar_passed + c.clearance_progress + c.near_miss
            + c.stage1_subgoal + c.bypass_subgoal + c.ring_subgoal
            + c.bypass_decision + c.bypass_progress
            + c.obstacle_visibility + c.obstacle_slowdown + c.pillar_attention
            + c.post_pillar + c.too_slow_penalty
            + c.terminal
        )
        return c.total, c

    # ------------------------------------------------------------------ #
    # Reward computation helpers                                          #
    # ------------------------------------------------------------------ #

    def _reward_progress(self, state: StepState, progress: float) -> float:
        # Source: old drone_env.py L5015-5021
        if progress >= 0.0:
            coef = self._r.stage2_progress_pos_coef if state.num_pillars > 0 else self._r.progress_pos_coef
            return coef * progress
        return self._r.progress_neg_coef * progress

    def _reward_time(self, step_count: int, state: StepState) -> float:
        # Escalating penalty. Source: old drone_env.py L5024-5032
        r = self._r.time_penalty_base
        if state.num_pillars > 0:
            if step_count > 170:
                r += self._r.time_penalty_step170
            if step_count > 250:
                r += self._r.time_penalty_step250
            if step_count > 350:
                r += self._r.time_penalty_step350
            if step_count > 450:
                r += self._r.time_penalty_step450
        return r

    def _reward_altitude(self, state: StepState) -> tuple[float, float]:
        z = float(state.pos[2])
        if z < self._e.alt_min:
            return 0.0, self._r.alt_below_min_coef * (self._e.alt_min - z)
        if z > self._e.alt_max:
            return 0.0, self._r.alt_above_max_coef * (z - self._e.alt_max)
        return 0.0, 0.0

    def _reward_smooth(self, action: np.ndarray, prev_action: np.ndarray) -> float:
        # Source: old drone_env.py L5066
        return self._r.smooth_penalty_coef * float(np.linalg.norm(action - prev_action))

    def _reward_velocity_goal(self, state: StepState) -> float:
        # Source: old drone_env.py L5278-5308
        r = self._r
        goal_vec = np.asarray(state.goal[:2], dtype=np.float32) - np.asarray(state.pos[:2], dtype=np.float32)
        goal_dir = goal_vec / (float(np.linalg.norm(goal_vec)) + 1e-8)
        vel_xy = np.asarray(state.vel[:2], dtype=np.float32)
        speed_toward_goal = float(np.dot(vel_xy, goal_dir))

        if state.num_pillars > 0:
            pil_dist = state.nearest_pillar_dist if state.nearest_pillar_dist is not None else float("inf")
            if pil_dist > r.stage2_velocity_goal_far_dist:
                coef = r.stage2_velocity_goal_coef_far
                if state.dist_xy < 6.0 and pil_dist > 4.0:
                    coef *= 1.8
            elif pil_dist > r.stage2_velocity_goal_near_dist:
                coef = r.stage2_velocity_goal_coef_near
            else:
                coef = r.stage2_velocity_goal_coef_danger
            clipped = float(np.clip(speed_toward_goal, -1.0, r.stage2_velocity_goal_clip))
            return coef * clipped
        else:
            clipped = float(np.clip(speed_toward_goal, -1.0, 2.5))
            return r.stage1_velocity_goal_coef * clipped

    def _reward_heading_goal(self, state: StepState) -> float:
        # Stage2 heading supplement. Source: old drone_env.py L5304
        if state.num_pillars == 0:
            return 0.0
        vel_xy = np.asarray(state.vel[:2], dtype=np.float32)
        goal_vec = np.asarray(state.goal[:2], dtype=np.float32) - np.asarray(state.pos[:2], dtype=np.float32)
        goal_dir = goal_vec / (float(np.linalg.norm(goal_vec)) + 1e-8)
        vel_xy_norm = float(np.linalg.norm(vel_xy))
        if vel_xy_norm < 1e-6:
            return 0.0
        heading_align = float(np.dot(vel_xy / vel_xy_norm, goal_dir))
        return 0.10 * heading_align * min(vel_xy_norm, 2.0)

    def _reward_yaw_align(self, state: StepState) -> float:
        # Source: old drone_env.py L5127-5270

        r = self._r
        yaw_error = state.yaw_error
        yaw_align_cos = float(math.cos(float(yaw_error)))
        camera_fwd_dot = float(np.clip(math.cos(float(yaw_error)), -1.0, 1.0))

        if state.stage_index <= 1:
            reward_face = r.stage1_face_goal_coef * yaw_align_cos
        else:
            reward_face = 0.10 * yaw_align_cos

        total = reward_face

        # Forward-goal bonus/penalty
        if state.stage_index == 1 and state.num_pillars == 0:
            good_thresh = 0.98
            reward_coef = 0.40
            penalty_coef = 2.5
        else:
            good_thresh = 0.9275
            reward_coef = 0.2
            penalty_coef = 1.2

        if camera_fwd_dot >= good_thresh:
            total += reward_coef * (camera_fwd_dot - good_thresh) / max(1e-6, 1.0 - good_thresh)
        else:
            total += -penalty_coef * (good_thresh - camera_fwd_dot) / max(1e-6, good_thresh + 1.0)

        # Stage1 backwards yaw penalty
        if state.num_pillars == 0:
            if (
                state.horizontal_speed > r.stage1_backwards_yaw_speed_gate
                and camera_fwd_dot < 0.0
            ):
                total += (
                    -r.stage1_backwards_yaw_penalty_coef
                    * state.horizontal_speed
                    * abs(camera_fwd_dot)
                )

        # Stage2 backwards yaw penalty
        if state.stage_index >= 2 and state.num_pillars > 0:
            goal_vec = np.asarray(state.goal[:2], dtype=np.float32) - np.asarray(state.pos[:2], dtype=np.float32)
            goal_dir = goal_vec / (float(np.linalg.norm(goal_vec)) + 1e-8)
            vel_xy = np.asarray(state.vel[:2], dtype=np.float32)
            speed_to_goal = float(np.dot(vel_xy, goal_dir))
            if (
                speed_to_goal > r.stage2_backwards_yaw_speed_to_goal_thresh
                and state.horizontal_speed > r.stage2_backwards_yaw_horizontal_speed_thresh
                and camera_fwd_dot < r.stage2_backwards_yaw_dot_thresh
            ):
                total += (
                    -r.stage2_backwards_yaw_penalty_coef
                    * speed_to_goal
                    * abs(camera_fwd_dot)
                )

        return total

    def _reward_spatial_penalties(self, state: StepState) -> tuple[float, float, float]:
        # Source: old drone_env.py L5537-5578
        r = self._r
        pos_xy = np.asarray(state.pos[:2], dtype=np.float32)
        cross_track = self._cross_track_error_xy(pos_xy, state.start[:2], state.goal[:2])
        fence_margin = self._fence_margin_xy(pos_xy)

        if state.stage_index <= 1:
            lat_thresh = r.stage1_lateral_thresh
            lat_coef = r.stage1_lateral_coef
            fence_thresh = r.stage1_fence_thresh
            fence_coef = r.stage1_fence_coef
        else:
            lat_thresh = r.stage2_lateral_thresh
            lat_coef = r.stage2_lateral_coef
            fence_thresh = r.stage2_fence_thresh
            fence_coef = r.stage2_fence_coef

        lateral = 0.0
        if cross_track > lat_thresh:
            lateral = -lat_coef * (cross_track - lat_thresh)

        near_fence = 0.0
        if state.stage_index >= 2:
            if fence_margin < 1.0:
                near_fence = r.stage2_near_fence_penalty_coefs[0] * (2.0 - 1.0)
                near_fence += r.stage2_near_fence_penalty_coefs[1] * (1.0 - fence_margin)
            elif fence_margin < 2.0:
                near_fence = r.stage2_near_fence_penalty_coefs[0] * (2.0 - fence_margin)
        elif fence_margin < fence_thresh:
            near_fence = -fence_coef * (fence_thresh - fence_margin) ** 2

        start_zone = 0.0
        dist_from_start = float(np.linalg.norm(pos_xy - np.asarray(state.start[:2], dtype=np.float32)))
        if state.step_count > r.start_penalty_after_steps and dist_from_start < r.start_xy_radius:
            start_zone = r.start_zone_penalty_coef * (r.start_xy_radius - dist_from_start)

        return lateral, near_fence, start_zone

    def _reward_pillar_clearance(
        self, state: StepState
    ) -> tuple[float, float, float, float, float]:
        # Source: old drone_env.py L5700-5723, L5667-5699
        r = self._r
        snap = state.pillar_collision_snap
        clearance_body = snap.get("clearance_body", float("inf"))
        heading_into = snap.get("heading_into", False)
        d_closest = snap.get("d_closest", float("inf"))
        t_closest = snap.get("t_closest", float("inf"))
        collision_radius = snap.get("collision_radius", float("nan"))
        speed_col = snap.get("speed", state.horizontal_speed)

        too_close = 0.0
        clearance_soft = 0.0
        near_speed = 0.0

        if np.isfinite(clearance_body):
            if clearance_body < r.clearance_body_safe:
                x = (r.clearance_body_safe - clearance_body) / max(r.clearance_body_safe, 1e-6)
                clearance_soft = r.clearance_soft_penalty_coef * (x ** 2)
            if clearance_body < r.clearance_body_danger:
                x = (r.clearance_body_danger - clearance_body) / max(r.clearance_body_danger, 1e-6)
                too_close = r.clearance_danger_penalty_coef * (x ** 2)
            near_speed_clearance = 0.60
            near_speed_safe_speed = 0.70
            near_speed_coef = 2.5
            if clearance_body < near_speed_clearance:
                speed_excess = max(0.0, state.horizontal_speed - near_speed_safe_speed)
                proximity = (near_speed_clearance - clearance_body) / max(near_speed_clearance, 1e-6)
                near_speed = -near_speed_coef * proximity * speed_excess

        collision_course = 0.0
        if (
            heading_into
            and np.isfinite(d_closest)
            and np.isfinite(collision_radius)
            and d_closest < collision_radius
            and 0.0 < t_closest < 1.0
        ):
            risk = max(0.0, (collision_radius - d_closest) / max(collision_radius, 1e-6))
            time_w = max(0.0, 1.0 - t_closest)
            speed_w = min(max(speed_col, 0.0) / 2.0, 1.5)
            collision_course = r.collision_course_coef * risk * time_w * speed_w

        clearance_body_reward = clearance_body if np.isfinite(clearance_body) else 0.0
        return too_close, clearance_soft, collision_course, clearance_body_reward, near_speed

    def _reward_subgoals(self, state: StepState) -> tuple[float, float, float]:
        # Source: delegates to PillarManager info in StepState
        stage1 = float(state.pillar_collision_snap.get("stage1_subgoal_reward", 0.0))
        bypass = float(state.bypass_subgoal_info.get("reward", 0.0))
        ring = float(state.ring_subgoal_info.get("reward", 0.0))
        return stage1, bypass, ring

    def _reward_bypass_decision(self, state: StepState) -> float:
        # Source: old drone_env.py L5368-5384
        r = self._r
        if state.num_pillars == 0:
            return 0.0
        active_bypass = state.bypass_subgoal_info.get("active_subgoal")
        if active_bypass is None:
            return 0.0
        vel_xy = np.asarray(state.vel[:2], dtype=np.float32)
        vel_xy_norm = float(np.linalg.norm(vel_xy))
        if vel_xy_norm < r.stage2_bypass_decision_min_speed:
            return 0.0
        pos_xy = np.asarray(state.pos[:2], dtype=np.float32)
        to_sg = np.asarray(active_bypass, dtype=np.float32) - pos_xy
        to_sg_norm = float(np.linalg.norm(to_sg))
        if to_sg_norm < 1e-6:
            return 0.0
        bypass_dir = to_sg / to_sg_norm
        vel_dir = vel_xy / (vel_xy_norm + 1e-8)
        align = float(np.dot(vel_dir, bypass_dir))
        if align > r.stage2_bypass_decision_min_align:
            return r.stage2_bypass_decision_reward * align * min(vel_xy_norm, 1.5)
        return 0.0

    def _reward_bypass_progress(self, state: StepState) -> float:
        # Source: old drone_env.py L5402-5417
        r = self._r
        if state.num_pillars == 0:
            return 0.0
        pil_dist = state.nearest_pillar_dist
        if pil_dist is None or not np.isfinite(pil_dist):
            return 0.0
        clearance_gain = state.bypass_subgoal_info.get("clearance_gain", 0.0)
        progress = state.prev_dist_xy - state.dist_xy
        in_bypass_zone = pil_dist < r.stage2_pillar_bypass_offset_m
        if (
            in_bypass_zone
            and float(clearance_gain) > r.stage2_bypass_progress_min_clearance_gain
            and progress > 0.0
        ):
            return r.stage2_bypass_progress_reward
        return 0.0

    def _reward_pillar_behavior(self, state: StepState) -> dict:
        # Delegates to pre-computed attention_info from PillarManager
        # Source: old drone_env.py L5314-5365, L5467-5516
        r = self._r
        out: dict = {}
        ai = state.attention_info

        out["attention"] = float(ai.get("attention_reward", 0.0))
        out["post_pillar"] = float(ai.get("post_pillar_reward", 0.0))

        # Obstacle visibility
        visibility = 0.0
        if (
            state.stage_index >= 2
            and r.stage2_obstacle_visible_depth_min <= state.front_depth <= r.stage2_obstacle_visible_depth_max
            and (state.nearest_pillar_dist or float("inf")) > r.stage2_obstacle_visibility_safe_pillar_dist
            and (state.prev_dist_xy - state.dist_xy) > r.stage2_obstacle_visibility_progress_gate
        ):
            visibility = r.stage2_obstacle_visibility_reward
        out["visibility"] = visibility

        # Obstacle slowdown
        slowdown = 0.0
        speed_drop = self._prev_horizontal_speed - state.horizontal_speed
        if (
            state.front_depth < r.stage2_obstacle_slowdown_depth_thresh
            and (state.nearest_pillar_dist or float("inf")) < r.stage2_obstacle_slowdown_depth_thresh
            and speed_drop > r.stage2_obstacle_slowdown_speed_drop
            and (state.prev_dist_xy - state.dist_xy) > 0.0
            and state.horizontal_speed > 0.4
        ):
            slowdown = r.stage2_obstacle_slowdown_reward
        out["slowdown"] = slowdown

        # Too-slow penalty
        too_slow = 0.0
        if (
            state.step_count > r.reward_too_slow_after_steps
            and state.dist_xy > r.reward_too_slow_min_dist
            and state.horizontal_speed < r.reward_too_slow_min_speed
        ):
            too_slow = r.reward_too_slow_coef
        out["too_slow"] = too_slow

        return out

    def _reward_terminal(self, state: StepState) -> float:
        r = self._r
        # Goal success — no near-fence addition
        if state.done_reason in ("goal_xy", "goal_3d", "success", "goal_reached"):
            return r.goal_xy_terminal_reward
        # out_of_fence — full penalty, skip near-fence addition (already violated)
        if state.is_terminal and state.done_reason == "out_of_fence":
            return r.out_of_fence_penalty

        base = 0.0
        if state.is_truncated and state.done_reason == "max_steps":
            near_goal = state.dist_xy <= max(state.goal_xy_radius * r.near_goal_threshold_factor, 1.0)
            if state.num_pillars == 0:
                base = r.max_steps_penalty_no_pillars
            elif near_goal:
                base = r.max_steps_penalty_pillars_near_goal
            else:
                base = r.max_steps_penalty_pillars_far_goal
        elif state.is_terminal and state.done_reason == "collision":
            base = r.collision_penalty
        elif state.is_terminal and state.done_reason == "fell_to_ground":
            base = r.fell_to_ground_penalty
        elif state.is_terminal and state.done_reason == "flipped":
            base = r.flipped_penalty

        # Near-fence penalty — only at episode end
        if (state.is_terminal or state.is_truncated):
            fence_margin = self._fence_margin_xy(state.pos[:2])
            if fence_margin < self._e.continuous_reset_fence_margin_thresh:
                base += r.out_of_fence_penalty * r.near_fence_terminal_penalty_factor

        return base

    # ------------------------------------------------------------------ #
    # Geometry utilities                                                  #
    # ------------------------------------------------------------------ #

    def _cross_track_error_xy(self, pos_xy: np.ndarray, start_xy: np.ndarray, goal_xy: np.ndarray) -> float:
        # Perpendicular distance from pos to the start→goal line segment.
        # Source: old drone_env.py L7302
        a = np.asarray(start_xy[:2], dtype=np.float64)
        b = np.asarray(goal_xy[:2], dtype=np.float64)
        p = np.asarray(pos_xy[:2], dtype=np.float64)
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq < 1e-8:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(np.dot(p - a, ab) / ab_len_sq, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + t * ab)))

    def _fence_margin_xy(self, pos_xy: np.ndarray) -> float:
        # Minimum distance from pos to XY fence boundary.
        # Source: old drone_env.py L7319
        e = self._e
        x, y = float(pos_xy[0]), float(pos_xy[1])
        return min(x - e.fence_x_min, e.fence_x_max - x, y - e.fence_y_min, e.fence_y_max - y)
