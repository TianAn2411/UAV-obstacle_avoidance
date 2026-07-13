from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from obstacle_avoidance.configs.env_config import EnvConfig
from obstacle_avoidance.configs.reward_config import RewardConfig
import math

@dataclass
class RewardComponents:
    # PBRS + RM
    pbrs: float = 0.0
    rm_bonus: float = 0.0
    # Navigation
    time: float = 0.0
    # Altitude
    altitude: float = 0.0
    # Action quality
    smooth: float = 0.0
    speed_penalty: float = 0.0
    fall_penalty: float = 0.0
    yaw_rate_penalty: float = 0.0
    # Goal alignment
    yaw_align: float = 0.0
    # Spatial penalties
    lateral: float = 0.0
    near_fence: float = 0.0
    start_zone: float = 0.0
    # Pillar-specific
    pillar_too_close: float = 0.0
    pillar_too_wide: float = 0.0
    pillar_tight_pass: float = 0.0
    bypass_reward: float = 0.0
    stage1_waypoint_bonus: float = 0.0
    # Pillar behavior
    post_pillar: float = 0.0
    # Terminal
    terminal: float = 0.0
    # Symbolic-mode only (Track B) — see cbf_intervention_coef's docstring
    cbf_intervention: float = 0.0
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
    nearest_pillar_dist: Optional[float]
    nearest_pillar_xy: Optional[np.ndarray]
    pillar_collision_snap: dict             # from PillarManager
    bypass_subgoal_info: dict              # from PillarManager
    ring_subgoal_info: dict                # from PillarManager
    attention_info: dict                   # from PillarManager
    reset_info: dict                       # from ResetManager
    done_reason: str                       # empty if not terminal
    goal_xy_radius: float
    yaw_good_thresh: float                 # ramped stage1_yaw_good_thresh, see train_manager.py's _update_yaw_good_thresh()
    is_terminal: bool
    is_truncated: bool
    goal: np.ndarray                       # current episode goal [x, y, z]
    start: np.ndarray                      # episode start [x, y, z]
    num_pillars: int                       # active pillars this episode
    horizontal_speed: float               # precomputed ||vel_xy||
    final_yaw_rate: float                 # actual yaw rate sent to bridge
    yaw_error: float                       # wrap_pi(desired_yaw - current_yaw)
    # DFA state (P-NSRL)
    dfa_q: int = 0                         # current DFA state index (0..N)
    dfa_N: int = 1                         # total subgoal checkpoints this episode
    dfa_q_prev: int = 0                    # DFA state at previous step (detect transition)
    stage1_waypoint_advance: int = 0       # waypoints newly crossed this step (pillar stages only)
    # Track B (symbolic_action) only — CBFShield.last_correction_norm, 0.0
    # whenever no CBF shield exists (Track A) or it didn't correct this step.
    cbf_correction_norm: float = 0.0


class RewardManager:
    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def __init__(self, rcfg: RewardConfig, ecfg: EnvConfig) -> None:
        self._r = rcfg
        self._e = ecfg
        # Per-episode tracked state (reset by reset_episode)
        self._has_been_airborne: bool = False
        self._prev_horizontal_speed: float = 0.0
        self._prev_phi: float = 0.0        # Φ(s_{t-1}, q_{t-1}) for PBRS
        self._dist_norm: float = 1.0  # set per episode in reset_episode()

    def _phi(self, dist_xy: float, dfa_q: int, dfa_N: int) -> float:
        """Hybrid potential: Φ(s,q) = β*(q/N) + α*(1 - dist/dist_norm)
        dist_norm = dist_start of this episode → Φ=0 at start, Φ=α+β at goal.
        Hover bonus F=(γ-1)*Φ ≤ 0 everywhere (Φ ≥ 0) → no free reward.
        """
        r = self._r
        phi_dfa  = float(r.pbrs_dfa_coef) * (float(dfa_q) / max(float(dfa_N), 1.0))
        phi_dist = float(r.pbrs_dist_coef) * (1.0 - float(dist_xy) / self._dist_norm)
        return phi_dfa + phi_dist

    def reset_episode(self, dist_start: float = 0.0, dfa_N: int = 1) -> None:
        self._has_been_airborne = False
        self._prev_horizontal_speed = 0.0
        # Dynamic norm = dist_start → Φ=0 at start regardless of goal distance.
        # Floor 4.0 guards against near-zero dist_start on startup/edge cases.
        self._dist_norm = max(float(dist_start), 4.0)
        # Init at Φ(s0, q=0) — avoids potential shock at step 1
        self._prev_phi = self._phi(dist_xy=dist_start, dfa_q=0, dfa_N=dfa_N)

    # ------------------------------------------------------------------ #
    # Main interface                                                      #
    # ------------------------------------------------------------------ #

    def compute(self, state: StepState) -> tuple[float, RewardComponents]:
        c = RewardComponents()
        obstacle_enabled = state.num_pillars > 0

        # --- PBRS (Hybrid Potential) ---
        # Standard PBRS: F = γΦ(s') - Φ(s) at every step.
        # No Grzes terminal correction — that was designed for negative Φ and would
        # give F = -Φ_prev = -25 at the goal terminal step, penalising success.
        phi_current = self._phi(state.dist_xy, state.dfa_q, state.dfa_N)
        c.pbrs = float(self._r.pbrs_gamma) * phi_current - self._prev_phi
        self._prev_phi = phi_current

        # --- Reward Machine bonus --- two independent additive sources:
        # (a) waypoint dfa_q advance, gated off the goal-arrival transition --
        # dfa_N's reserved final slot bumps dfa_q by 1 the instant goal is
        # reached (see train_manager.py's DFA snap), which isn't a real
        # waypoint crossing and is already fully paid by terminal
        # (goal_xy_terminal_reward + fast_finish + yaw bonus). Without this
        # gate, every successful episode got a free extra rm_subgoal_bonus
        # for simply arriving.
        # (b) pillar pass count this step (PillarManager.update(), unconditional
        # on blocking/decor status) -- independent of dfa_q entirely, so a
        # pillar pass and a waypoint crossing can both pay out the same step.
        rm = 0.0
        if state.dfa_q > state.dfa_q_prev and state.done_reason not in ("goal_xy", "goal_3d", "success"):
            rm += float(self._r.rm_subgoal_bonus)
        rm += float(self._r.rm_subgoal_bonus) * float(state.pillar_collision_snap.get("pillars_passed_count", 0))
        c.rm_bonus = rm

        # --- Time ---
        c.time = self._reward_time(state)

        # --- Altitude ---
        c.altitude = self._reward_altitude(state)

        # --- Action quality ---
        c.smooth = self._reward_smooth(state.action, state.prev_action)
        # Penalize yaw_rate only when it's turning the WRONG way (away from
        # reducing yaw_error) or spinning while already aligned -- never while
        # actively correcting toward the goal, however large the misalignment.
        # Old weight (abs(cos(yaw_error))) penalized correction hardest right
        # at yaw_error=+-180deg (goal behind, cos=-1, weight=1) -- fighting
        # FacePrimitive's own p_face gate at exactly the state that most needs
        # a fast, large yaw_rate correction (sign(yaw_rate)==sign(yaw_error)
        # is the "same rotational direction as the error" test, since both
        # are wrap_pi(desired_yaw - current_yaw)-style signed quantities).
        _he = float(state.yaw_error)
        _yr = float(state.final_yaw_rate)
        if abs(_he) < 1e-3:
            _yaw_align_w = 1.0 if abs(_yr) > 1e-3 else 0.0  # aligned but still spinning -> penalize
        else:
            _yaw_align_w = 0.0 if (_yr * _he) > 0.0 else 1.0  # correcting toward goal -> free
        c.yaw_rate_penalty = float(self._r.yaw_rate_penalty_coef) * abs(_yr) * _yaw_align_w

        # --- Yaw alignment (no progress amplification — progress replaced by PBRS) ---
        c.yaw_align = self._reward_yaw_align(state)

        # --- Spatial penalties ---
        c.lateral, c.near_fence, c.start_zone = self._reward_spatial_penalties(state)

        # --- Speed / fall ---
        speed = float(np.linalg.norm(state.vel))
        if speed > self._r.speed_penalty_thresh:
            c.speed_penalty = self._r.speed_penalty_coef * (speed - self._r.speed_penalty_thresh)
        if state.vel[2] < -self._r.fall_vz_thresh:
            c.fall_penalty = self._r.fall_penalty_coef * (-float(state.vel[2]) - self._r.fall_vz_thresh)

        # --- Pillar clearance (unified: too-close + too-wide + tight-pass bonus) ---
        if obstacle_enabled:
            (
                c.pillar_too_close,
                c.pillar_too_wide,
            ) = self._reward_pillar_clearance(state)
            c.pillar_tight_pass = float(state.pillar_collision_snap.get("pillar_tight_pass_bonus", 0.0))
            # Danger scale: only yaw_align (progress/velocity_goal already removed)
            clearance_body = state.pillar_collision_snap.get("clearance_body", float("inf"))
            if np.isfinite(clearance_body) and clearance_body < self._r.pillar_clearance_close_thresh:
                danger_scale = max(0.0, min(1.0, clearance_body / self._r.pillar_clearance_close_thresh))
                c.yaw_align *= max(0.30, danger_scale)
            c.bypass_reward = float(state.bypass_subgoal_info.get("reward", 0.0))
            c.stage1_waypoint_bonus = int(state.stage1_waypoint_advance) * float(self._r.stage1_sg_bonus_pillar_stage)

        # --- Pillar behavior ---
        if obstacle_enabled:
            beh = self._reward_pillar_behavior(state)
            c.post_pillar = beh.get("post_pillar", 0.0)

        # --- Terminal ---
        c.terminal = self._reward_terminal(state)

        # --- CBF intervention penalty (Track B / symbolic_action only) ---
        # Gated on policy_mode explicitly (not just relying on
        # cbf_correction_norm defaulting to 0.0 for Track A) so this term can
        # never leak into raw-mode reward even if a future bug sets the field
        # elsewhere. Penalizes the CORRECTION MAGNITUDE itself, independent of
        # whether the resulting (CBF-corrected) trajectory was safe --
        # otherwise PPO gets bailed out by CBF with zero learning signal for
        # the risky gate choice that required the correction in the first
        # place. coef defaults 0.0 (opt-in, no behavior change until tuned).
        if self._e.policy_mode == "symbolic":
            c.cbf_intervention = -self._r.cbf_intervention_coef * state.cbf_correction_norm

        # Fast-finish time_penalty refund — displayed in `time` (not `terminal`)
        # so the log shows net time cost ≈0 on a fast goal arrival. Only fires
        # on actual goal success within fast_finish_late_step; collision /
        # max_steps / out_of_fence / etc keep paying time_penalty normally.
        if state.done_reason in ("goal_xy", "goal_3d", "success", "goal_reached"):
            if state.step_count <= self._r.fast_finish_late_step:
                c.time += -state.step_count * self._r.time_penalty_base

        # --- Update tracked state ---
        self._prev_horizontal_speed = state.horizontal_speed
        if float(state.pos[2]) > self._e.airborne_z:
            self._has_been_airborne = True

        c.total = (
            c.pbrs + c.rm_bonus
            + c.time
            + c.altitude
            + c.smooth + c.yaw_rate_penalty + c.speed_penalty + c.fall_penalty
            + c.yaw_align
            + c.lateral + c.near_fence + c.start_zone
            + c.pillar_too_close + c.pillar_too_wide + c.pillar_tight_pass
            + c.bypass_reward
            + c.stage1_waypoint_bonus
            + c.post_pillar
            + c.terminal
            + c.cbf_intervention
        )
        return c.total, c

    # ------------------------------------------------------------------ #
    # Reward computation helpers                                          #
    # ------------------------------------------------------------------ #

    def _reward_time(self, state: StepState) -> float:
        r = self._r
        steps = state.step_count
        if steps < 120:
            return r.time_penalty_base
        if steps < 200:
            return r.time_penalty_tier2
        if steps < 300:
            return r.time_penalty_tier3
        return r.time_penalty_tier4

    def _reward_altitude(self, state: StepState) -> float:
        z = float(state.pos[2])
        r = self._r
        e = self._e

        # Altitude positive reward: low ramp → flat → high ramp
        if r.alt_optimal_low <= z <= r.alt_optimal_high:
            optimal_reward = r.alt_optimal_reward
        elif r.alt_suboptimal_low_thresh <= z < r.alt_optimal_low:
            ramp = (z - r.alt_suboptimal_low_thresh) / (r.alt_optimal_low - r.alt_suboptimal_low_thresh)
            optimal_reward = r.alt_ramp_coef * ramp
        elif r.alt_optimal_high < z <= r.alt_suboptimal_high_thresh:
            ramp = (r.alt_suboptimal_high_thresh - z) / (r.alt_suboptimal_high_thresh - r.alt_optimal_high)
            optimal_reward = r.alt_ramp_coef * ramp
        else:
            optimal_reward = 0.0

        # Piecewise penalty
        penalty = 0.0
        if z < e.alt_min:
            boundary_penalty = r.alt_suboptimal_low_coef * (r.alt_suboptimal_low_thresh - e.alt_min)
            extra_penalty = r.alt_below_min_coef * (e.alt_min - z)
            penalty = boundary_penalty + extra_penalty
        elif z < r.alt_suboptimal_low_thresh:
            penalty = r.alt_suboptimal_low_coef * (r.alt_suboptimal_low_thresh - z)
        elif z > e.alt_max:
            penalty = r.alt_above_max_coef * (z - e.alt_max)
        elif z > r.alt_suboptimal_high_thresh:
            penalty = r.alt_above_suboptimal_coef * (z - r.alt_suboptimal_high_thresh)

        return optimal_reward + penalty

    def _reward_smooth(self, action: np.ndarray, prev_action: np.ndarray) -> float:
        # Source: old drone_env.py L5066
        return self._r.smooth_penalty_coef * float(np.linalg.norm(action - prev_action))

    def _reward_yaw_align(self, state: StepState) -> float:
        # Source: old drone_env.py L5127-5270

        r = self._r
        yaw_error = state.yaw_error
        yaw_align_cos = float(math.cos(float(yaw_error)))
        camera_fwd_dot = float(np.clip(math.cos(float(yaw_error)), -1.0, 1.0))

        reward_face = r.stage1_face_goal_coef * yaw_align_cos

        total = reward_face

        # Forward-goal bonus/penalty (thresholds/coefs configurable in RewardConfig)
        good_thresh = state.yaw_good_thresh
        reward_coef = r.stage1_yaw_forward_bonus_coef
        penalty_coef = r.stage1_yaw_forward_penalty_coef

        if camera_fwd_dot >= good_thresh:
            total += reward_coef * (camera_fwd_dot - good_thresh) / max(1e-6, 1.0 - good_thresh)
        else:
            if state.stage_index == 0:
                total += 0.0
            else:
                total += -penalty_coef * (good_thresh - camera_fwd_dot) / max(1e-6, good_thresh + 1.0)

        # Stage1/Stage2 backwards yaw penalty -- Track A (raw) only. Written
        # under the assumption translation is coupled to heading (ActionManager
        # has no such coupling either, but this penalty is what taught Track A's
        # policy not to fly fast while facing away from the goal). Track B
        # (symbolic) breaks that assumption on purpose: GoalPrimitive is
        # holonomic (goal.py's [cos,sin](heading_error) for [vx,vy]) --
        # translating at speed while yaw is still catching up (FacePrimitive,
        # own gate) is the intended, efficient behavior there, not a fault to
        # penalize. Gating this off for symbolic avoids fighting that design.
        if self._e.policy_mode != "symbolic":
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

    def _reward_pillar_clearance(self, state: StepState):
        """Light 2-sided shaping around a target clearance band, only while a
        pillar is actually nearby (gated by pillar_zone_radius). Single linear
        tier each side -- too-close (collision risk) and too-wide (sloppy
        detour, encourages a tight flight path). Actual collision is handled
        separately by terminal.collision_penalty; this is shaping only.
        """
        r = self._r
        snap = state.pillar_collision_snap
        clearance_body = snap.get("clearance_body", float("inf"))
        nearest_dist = snap.get("nearest_dist", float("inf"))

        too_close = 0.0
        too_wide = 0.0

        in_zone = np.isfinite(nearest_dist) and nearest_dist <= r.pillar_zone_radius
        if in_zone and np.isfinite(clearance_body):
            if clearance_body < r.pillar_clearance_close_thresh:
                x = 1.0 - max(0.0, clearance_body) / max(r.pillar_clearance_close_thresh, 1e-6)
                too_close = r.pillar_clearance_close_coef * x
            elif clearance_body > r.pillar_clearance_wide_thresh:
                over = clearance_body - r.pillar_clearance_wide_thresh
                too_wide = max(r.pillar_clearance_wide_penalty_cap, r.pillar_clearance_wide_coef * over)

        return too_close, too_wide

    def _reward_pillar_behavior(self, state: StepState) -> dict:
        # Delegates to pre-computed attention_info from PillarManager
        # Source: old drone_env.py L5314-5365, L5467-5516
        r = self._r
        out: dict = {}
        ai = state.attention_info

        out["post_pillar"] = float(ai.get("post_pillar_reward", 0.0))

        return out

    def _reward_terminal(self, state: StepState) -> float:
        r = self._r
        # Goal success — no near-fence addition
        if state.done_reason in ("goal_xy", "goal_3d", "success", "goal_reached"):
            steps = state.step_count
            if steps <= r.fast_finish_early_step:
                bonus = r.fast_finish_max_bonus
            elif steps <= r.fast_finish_late_step:
                span = r.fast_finish_late_step - r.fast_finish_early_step
                frac = (steps - r.fast_finish_early_step) / span
                bonus = r.fast_finish_max_bonus - frac * (r.fast_finish_max_bonus - r.fast_finish_min_bonus)
            else:
                bonus = 0.0
            # One-shot yaw-at-arrival bonus/penalty (stage1_goal_yaw_* was
            # defined in RewardConfig but never wired in -- per-step
            # yaw_align alone gets drowned out by terminal+pbrs+rm (~150+21+
            # 50) over a whole episode, too weak a gradient to teach p_face
            # to actually arrive facing the goal even after policy converges
            # (observed: terminal stayed a flat goal_xy_terminal_reward+
            # fast_finish_bonus regardless of arrival yaw_align swinging
            # -28..+8 across episodes). This ties a clean, undiluted signal
            # directly to the one state that matters: heading at the moment
            # of success.
            # No neutral zone: anything below ok_dot is penalized (was a
            # 0.0..ok_dot dead-zone with no bonus/penalty before).
            yaw_dot = float(np.clip(math.cos(float(state.yaw_error)), -1.0, 1.0))
            if yaw_dot >= r.stage1_goal_yaw_good_dot:
                yaw_bonus = r.stage1_goal_yaw_bonus_good
            elif yaw_dot >= r.stage1_goal_yaw_ok_dot:
                yaw_bonus = r.stage1_goal_yaw_bonus_ok
            else:
                yaw_bonus = r.stage1_goal_yaw_penalty_bad
            return r.goal_xy_terminal_reward + bonus + yaw_bonus
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
                base += r.near_fence_terminal_base_penalty

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
