import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from obstacle_avoidance.configs.env_config import EnvConfig
from obstacle_avoidance.configs.pillar_config import PillarConfig
from obstacle_avoidance.configs.reward_config import RewardConfig
from obstacle_avoidance.envs.manager.action_manager import ActionManager, ActionOutput
from obstacle_avoidance.envs.manager.logging_manager import LoggingManager
from obstacle_avoidance.envs.manager.noise_manager import NoiseManager
from obstacle_avoidance.envs.manager.pillar_manager import PillarManager, PillarSnapshot
from obstacle_avoidance.envs.manager.reset_manager import ResetManager, ResetDecision
from obstacle_avoidance.envs.manager.reward_manager import RewardManager, StepState, RewardComponents
from obstacle_avoidance.utils.bridge_factory import ROSBridge, Spawner
from obstacle_avoidance.utils.logger import setup_logger


@dataclass
class StepResult:
    obs: dict           # {"depth": np.ndarray, "state": np.ndarray}
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class TrainManager:
    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        bridge: ROSBridge,
        spawner: Spawner,
        ecfg: EnvConfig,
        rcfg: RewardConfig,
        pcfg: PillarConfig,
        env_id: int = 0,
        log_dir=None,
        start_step: int = 0,
    ) -> None:
        self.bridge = bridge
        self.ecfg = ecfg
        self.env_id = env_id
        self.logger = setup_logger(f"TRAIN_{env_id}")

        self._action_manager = ActionManager(ecfg)
        self._pillar_manager = PillarManager(spawner, pcfg, ecfg, rcfg, env_id)
        self._reset_manager = ResetManager(bridge, ecfg, env_id)
        self._reward_manager = RewardManager(rcfg, ecfg)
        self._logging_manager = LoggingManager(env_id=env_id, log_dir=log_dir, log_every_steps=50)

        # Episode state
        self._step_count: int = 0
        self._global_step: int = int(start_step)
        self._stage_start_step: int = int(start_step)
        self._goal_xy_radius: float = float(ecfg.goal_xy_radius_schedule_values[0])
        self._last_goal_xy_radius_logged: Optional[float] = None

        # Start / goal
        start_arr = np.asarray(ecfg.start, dtype=np.float32)
        self._start: np.ndarray = start_arr.copy()
        self._goal: np.ndarray = np.asarray(ecfg.goal, dtype=np.float32).copy()

        # Carry state between steps
        self._prev_pos: np.ndarray = start_arr.copy()
        self._prev_dist_xy: float = 0.0
        self._ep_dist_start: float = 0.0
        self._prev_action: np.ndarray = np.zeros(4, dtype=np.float32)
        self._prev_smoothed_action_n: np.ndarray = np.zeros(4, dtype=np.float32)
        self._prev_prev_smoothed_action_n: np.ndarray = np.zeros(4, dtype=np.float32)
        self._prev_delta_a1: np.ndarray = np.zeros(4, dtype=np.float32)

        # Episode done-reason tracking
        self._last_done_reason: Optional[str] = None
        self._ep_reward_sum: float = 0.0

        # RNG for goal sampling (seeded fresh each episode via np.random)
        self._np_random = np.random.default_rng()

        # Airborne tracking for reward manager
        self._has_been_airborne: bool = False

        # All sim-to-real noise: obs noise, action delay/noise, mass scale, wind
        self._noise_manager = NoiseManager(ecfg, self._np_random)

        # DFA state (P-NSRL)
        self._dfa_q: int = 0            # current DFA state index
        self._dfa_N: int = 1            # total subgoal checkpoints this episode
        self._dfa_q_prev: int = 0       # DFA state at previous step
        self._dfa_use_corridor: bool = False  # True → dfa_q=corridor_avoided_count; False → stage1_idx
        self._prev_stage1_idx: int = 0  # stage1_idx at previous step (for waypoint advance delta)
        self._dfa_segment_t_boundaries: list = [0.0, 1.0]  # corridor t-boundaries per DFA segment

    def close(self) -> None:
        """Clean up bridge and flush logs. Source: old drone_env.py L8134."""
        try:
            self.bridge.close()
        except Exception as e:
            self.logger.debug(f"[TRAIN] bridge close error: {e}")

    # ------------------------------------------------------------------ #
    # Public entry points (called by drone_env.py)                       #
    # ------------------------------------------------------------------ #

    def reset(self) -> tuple[dict, dict]:
        """Orchestrate reset: classify → dispatch → finalize. Source: old drone_env.py L4314."""
        reset_t0 = time.monotonic()
        self._update_goal_xy_radius()

        reason = self._last_done_reason or "startup"

        # Compute fence margin for classify decision
        pos_now = self.bridge.get_gazebo_position()
        fence_margin = self._fence_margin_xy(pos_now[:2]) if np.all(np.isfinite(pos_now)) else 999.0

        decision = self._reset_manager.classify_reset(reason, fence_margin=fence_margin)

        self.logger.debug(
            f"[RESET][START] env={self.env_id} reason={reason} "
            f"mode={decision.mode} fence_margin={fence_margin:.2f}"
        )

        # Yaw curriculum: assist_ratio ~ Uniform(min_assist, 1.0), step-wise schedule.
        # (threshold_steps, min_assist): at each milestone min_assist drops.
        _assist_ratio = 1.0
        if self.ecfg.pre_episode_auto_yaw_enabled:
            _steps_in_stage = max(0, self._global_step - self._stage_start_step)
            _YAW_SCHEDULE = [
                (     0, 1.0),  # 0–50k:   Uniform(1.0, 1.0) — full assist
                ( 50_000, 0.7),  # 50–80k:  Uniform(0.7, 1.0)
                ( 80_000, 0.6),  # 80–130k: Uniform(0.6, 1.0)
                (130_000, 0.4),  # 130–200k:Uniform(0.4, 1.0)
                (200_000, 0.2),  # 200–300k:Uniform(0.2, 1.0)
                (300_000, 0.0),  # 300k+:   Uniform(0.0, 1.0)
            ]
            _min_assist = 0.0
            for _thresh, _val in reversed(_YAW_SCHEDULE):
                if _steps_in_stage >= _thresh:
                    _min_assist = _val
                    break
            _assist_ratio = float(self._np_random.uniform(_min_assist, 1.0))

        if decision.mode == "startup_arm":
            # For grounded reasons, drone may be far from self._start — use current pos so
            # _startup_arm_sanity pose check doesn't fail before even trying to arm.
            if decision.reason in {"fell_to_ground", "ground", "flipped"} and np.all(np.isfinite(pos_now)):
                arm_start = np.array([pos_now[0], pos_now[1], 0.0], dtype=np.float32)
            else:
                arm_start = self._start
            self._reset_manager.startup_arm_episode_reset(arm_start)
            new_pos = self.bridge.get_gazebo_position()
            if np.all(np.isfinite(new_pos)):
                self._start = np.array([new_pos[0], new_pos[1], 0.0], dtype=np.float32)
            self._goal = self._generate_new_goal(override_start=self._start)
            self._reset_manager.pre_episode_auto_yaw_to_goal(self._goal, _assist_ratio)
            self._pillar_manager.reset_episode(self._start, self._goal)

        elif decision.mode == "continuous":
            new_start_xy = self._reset_manager.continuous_episode_reset(reason)
            self._start = np.array([new_start_xy[0], new_start_xy[1], 0.0], dtype=np.float32)
            self._goal = self._generate_new_goal(override_start=self._start)
            self._reset_manager.pre_episode_auto_yaw_to_goal(self._goal, _assist_ratio)
            self._pillar_manager.reset_episode(self._start, self._goal)

        elif decision.mode == "rescue_then_continuous":
            new_start_xy = self._reset_manager.rescue_then_continuous_reset(reason)
            if new_start_xy is not None:
                self._start = np.array([new_start_xy[0], new_start_xy[1], 0.0], dtype=np.float32)
            else:
                # Rescue failed → fall back to hard reset
                self.logger.warning(
                    f"[RESET] rescue failed, falling back to hard reset reason={reason}"
                )
                self._reset_manager.hard_reset_fallback_episode_reset(reason, reset_t0, self._start)
                new_pos = self.bridge.get_gazebo_position()
                self._start = np.array([new_pos[0], new_pos[1], 0.0], dtype=np.float32)

            self._goal = self._generate_new_goal(override_start=self._start)
            self._reset_manager.pre_episode_auto_yaw_to_goal(self._goal, _assist_ratio)
            self._pillar_manager.reset_episode(self._start, self._goal)

        elif decision.mode == "multi_env_fast":
            fast_ok = self._reset_manager.multi_env_fast_recover_episode_reset(
                reason, reset_t0, self._start
            )
            if not fast_ok:
                self.logger.warning(
                    f"[RESET] multi_env_fast failed, falling back to hard reset reason={reason}"
                )
                self._reset_manager.hard_reset_fallback_episode_reset(reason, reset_t0, self._start)
                new_pos = self.bridge.get_gazebo_position()
                self._start = np.array([new_pos[0], new_pos[1], 0.0], dtype=np.float32)
            self._goal = self._generate_new_goal(override_start=self._start)
            self._reset_manager.pre_episode_auto_yaw_to_goal(self._goal, _assist_ratio)
            self._pillar_manager.reset_episode(self._start, self._goal)

        else:  # hard
            self._reset_manager.hard_reset_fallback_episode_reset(reason, reset_t0, self._start)
            new_pos = self.bridge.get_gazebo_position()
            self._start = np.array([new_pos[0], new_pos[1], 0.0], dtype=np.float32)
            self._goal = self._generate_new_goal(override_start=self._start)
            self._reset_manager.pre_episode_auto_yaw_to_goal(self._goal, _assist_ratio)
            self._pillar_manager.reset_episode(self._start, self._goal)

        # If armed but still below 1m after reset (e.g. continuous reset landed low),
        # climb to safe altitude before episode begins.
        _pos_pre = self.bridge.get_gazebo_position()
        if (
            getattr(self.bridge, "is_armed", False)
            and np.all(np.isfinite(_pos_pre))
            and float(_pos_pre[2]) < 1.0
        ):
            self._pre_episode_climb_if_low(_pos_pre)

        # Common finalize
        self._action_manager.reset(
            hold_alt=self.ecfg.freeze_vz_hold_alt if self.ecfg.freeze_vz else None
        )

        # DFA reset — Option B: dfa_q = corridor_avoided_count, dfa_N = active corridor pillars.
        # Fallback to stage1 linear waypoints when no active corridor pillars exist.
        self._dfa_q = 0
        self._dfa_q_prev = 0
        if self._pillar_manager._actual_num_pillars == 0:
            # +1: goal is the final DFA state (dfa=1.0 only at goal, not at last waypoint)
            self._dfa_N = max(len(self._pillar_manager._stage1_sgs_xy), 1) + 1
            self._dfa_use_corridor = False
        else:
            active_n = self._pillar_manager._active_corridor_pillar_count
            if active_n > 0:
                # +1: goal is the final DFA state (dfa=1.0 only at goal, not after last pillar)
                self._dfa_N = active_n + 1
                self._dfa_use_corridor = True
            else:
                # All pillars outside drone's path — fall back to linear progress
                self._dfa_N = max(len(self._pillar_manager._stage1_sgs_xy), 1) + 1
                self._dfa_use_corridor = False

        self._dfa_segment_t_boundaries = self._compute_dfa_segment_boundaries()

        dist_start = float(np.linalg.norm(self._goal[:2] - self._start[:2]))
        self._ep_dist_start = dist_start
        self._reward_manager.reset_episode(dist_start=dist_start, dfa_N=self._dfa_N)
        self._reset_manager.on_episode_end(reason)

        synced_pos = self.bridge.get_gazebo_position()
        self._step_count = 0
        self._prev_stage1_idx = 0
        self._prev_pos = synced_pos.copy() if np.all(np.isfinite(synced_pos)) else self._start.copy()
        self._prev_dist_xy = float(np.linalg.norm(self._goal[:2] - self._prev_pos[:2]))
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._prev_smoothed_action_n = np.zeros(4, dtype=np.float32)
        self._prev_prev_smoothed_action_n = np.zeros(4, dtype=np.float32)
        self._prev_delta_a1 = np.zeros(4, dtype=np.float32)
        self._noise_manager.reset_episode(self.bridge)
        self._has_been_airborne = bool(float(synced_pos[2]) > self.ecfg.airborne_z)
        self._last_done_reason = None

        self.bridge.reset_extractor()

        self.bridge.show_debug_markers(self._goal[:2])

        obs = self._build_obs(
            self.bridge.get_perception(),
            self.bridge.get_ekf_position_enu(),
            self.bridge.get_linear_velocity(),
            self.bridge.get_yaw()[0],
            self.bridge.get_angular_velocity(),
        )

        self.logger.debug(
            f"[RESET][DONE] env={self.env_id} mode={decision.mode} "
            f"total={time.monotonic() - reset_t0:.3f}s "
            f"start_xy={np.round(self._start[:2], 3).tolist()} "
            f"goal_xy={np.round(self._goal[:2], 3).tolist()} "
            f"dist_xy={self._prev_dist_xy:.2f}"
        )

        self._ep_reward_sum = 0.0
        dist_to_goal = float(np.linalg.norm(self._goal[:2] - synced_pos[:2]))
        self._logging_manager.log_episode_start(
            reset_mode=decision.mode,
            reason=reason,
            pos=synced_pos,
            dist_xy=dist_to_goal,
        )

        return obs, {"reset_mode": decision.mode, "reason": reason}

    def step_process(self, raw_action: np.ndarray) -> StepResult:
        """
        Main step loop. Source: old drone_env.py L4379.
        1. ActionManager.process → velocity cmd
        2. bridge.send_velocity + tick
        3. Read new state
        4. PillarManager.update
        5. Build StepState
        6. RewardManager.compute
        7. _check_terminal
        8. Return StepResult
        """
        self._update_goal_xy_radius()

        # Early exit: failsafe active before action is sent
        if self.bridge.failsafe and self._step_count > 5:
            self.logger.warning(
                f"[STEP] env={self.env_id} px4_failsafe=True at step start "
                f"(step={self._step_count}) — terminating episode"
            )
            obs = self._build_obs(
                self.bridge.get_perception(),
                self.bridge.get_ekf_position_enu(),
                self.bridge.get_linear_velocity(),
                self.bridge.get_yaw()[0],
                self.bridge.get_angular_velocity(),
            )
            pos = self.bridge.get_gazebo_position().astype(np.float32)
            if not np.all(np.isfinite(pos)):
                pos = self._prev_pos.copy()
            dist_xy = float(np.linalg.norm(self._goal[:2] - pos[:2]))
            self._last_done_reason = "px4_failsafe"
            self._reset_manager.on_episode_end("px4_failsafe")
            self._logging_manager.record_episode_end(
                done_reason="px4_failsafe",
                episode_reward=self._ep_reward_sum,
                steps=self._step_count + 1,
                pos=pos,
            )
            return StepResult(obs=obs, reward=-1.0, terminated=True, truncated=False,
                              info={"done_reason": "px4_failsafe", "pos": pos,
                                    "dist_xy": dist_xy, "step_count": self._step_count})

        # 1. Compute smoothed velocity command
        pos_now = self.bridge.get_gazebo_position()
        alt = float(pos_now[2]) if np.all(np.isfinite(pos_now)) else 0.0
        is_takeoff = not self._has_been_airborne

        action_out: ActionOutput = self._action_manager.process(
            raw_action=raw_action,
            step_count=self._step_count,
            altitude=alt,
            is_takeoff_phase=is_takeoff,
        )

        # 2. Domain randomisation pipeline → send velocity and tick
        computed_cmd = np.array(
            [action_out.vx, action_out.vy, action_out.vz, action_out.yaw_rate], dtype=np.float32
        )
        # mass scale + delay buffer + ESC noise; delayed_cmd goes into state[14:18]
        delayed_cmd, sent_cmd = self._noise_manager.step_action(computed_cmd)

        self.bridge.send_velocity(sent_cmd[0], sent_cmd[1], sent_cmd[2], sent_cmd[3])
        self.bridge.tick(self.ecfg.dt)

        if not self.bridge.is_px4_callbacks_healthy(max_est_flags_age=5.0):
            age = time.monotonic() - self.bridge._last_estimator_flags_wall
            self.logger.warning(
                f"[STEP] env={self.env_id} estimator-flags stale for {age:.1f}s "
                f"(step={self._step_count})"
            )

        # 3. Read new state
        pos = self.bridge.get_gazebo_position().astype(np.float32)
        vel = self.bridge.get_linear_velocity().astype(np.float32)
        yaw, _ = self.bridge.get_yaw()
        perception = self.bridge.get_perception()  # (3,84,84) BEV or (1,84,84) depth
        ekf_pos = self.bridge.get_ekf_position_enu().astype(np.float32)
        ang_vel = self.bridge.get_angular_velocity().astype(np.float32)

        if not np.all(np.isfinite(pos)):
            pos = self._prev_pos.copy()
        if not np.all(np.isfinite(vel)):
            vel = np.zeros(3, dtype=np.float32)

        # Update airborne tracking
        if float(pos[2]) > self.ecfg.airborne_z:
            self._has_been_airborne = True

        # 4. PillarManager.update
        vel_xy = vel[:2]
        pillar_snap: PillarSnapshot = self._pillar_manager.update(
            pos=pos,
            vel_xy=vel_xy,
            yaw=float(yaw),
            goal_xy_radius=self._goal_xy_radius,
            step_count=self._step_count,
        )

        # 4b. DFA advance — Option B: corridor_avoided_count for pillar stages
        self._dfa_q_prev = self._dfa_q
        if self._dfa_use_corridor:
            new_idx = self._pillar_manager._corridor_avoided_count
        else:
            new_idx = self._pillar_manager._stage1_idx
        self._dfa_q = min(new_idx, self._dfa_N)

        # 4c. Stage1 waypoint advance (pillar stages only — fires flat bonus, not DFA)
        _cur_stage1_idx = self._pillar_manager._stage1_idx
        _stage1_advance = max(0, _cur_stage1_idx - self._prev_stage1_idx)
        self._prev_stage1_idx = _cur_stage1_idx

        # 5. Compute obs-derived features
        rel_goal_xy = (self._goal[:2] - pos[:2]).astype(np.float32)
        dist_xy = float(np.linalg.norm(rel_goal_xy))
        horizontal_speed = float(np.linalg.norm(vel_xy))

        if dist_xy > 1e-6:
            desired_yaw = math.atan2(float(rel_goal_xy[1]), float(rel_goal_xy[0]))
        else:
            desired_yaw = float(yaw)
        yaw_error = self._wrap_pi(desired_yaw - float(yaw))

        # Pillar info dicts for StepState
        pillar_collision_snap = self._pillar_manager.get_collision_snap()
        bypass_info = self._pillar_manager.get_bypass_info()
        ring_info = self._pillar_manager.get_ring_info()
        attention_info = self._pillar_manager.get_attention_info()

        # Check terminal BEFORE reward (need done_reason for terminal reward)
        terminated, truncated, done_reason = self._check_terminal(
            pos=pos,
            pillar_snap=pillar_snap,
            step_count=self._step_count,
            goal_xy_radius=self._goal_xy_radius,
        )

        # Snap dfa_q to dfa_N when goal reached — ensures dfa=1.0 only at actual goal,
        # not after last pillar/waypoint (fixes false dfa=1.0 saturation before goal).
        if done_reason in ("goal_xy", "goal_3d", "success"):
            self._dfa_q = self._dfa_N

        # 6. Build StepState and compute reward
        step_state = StepState(
            pos=pos,
            vel=vel,
            yaw=float(yaw),
            prev_pos=self._prev_pos,
            prev_action=self._prev_action,
            action=np.asarray(raw_action, dtype=np.float32),
            step_count=self._step_count,
            global_step=self._global_step,
            stage_index=(0 if self.ecfg.freeze_vz else (1 if self._pillar_manager._actual_num_pillars == 0 else 2)),
            dist_xy=dist_xy,
            prev_dist_xy=self._prev_dist_xy,
            nearest_pillar_dist=pillar_snap.nearest_dist if np.isfinite(pillar_snap.nearest_dist) else None,
            nearest_pillar_xy=pillar_snap.nearest_xy,
            pillar_collision_snap=pillar_collision_snap,
            bypass_subgoal_info=bypass_info,
            ring_subgoal_info=ring_info,
            attention_info=attention_info,
            reset_info=self._reset_manager.get_reset_helpers(),
            done_reason=done_reason,
            goal_xy_radius=self._goal_xy_radius,
            is_terminal=terminated,
            is_truncated=truncated,
            goal=self._goal,
            start=self._start,
            num_pillars=self._pillar_manager._actual_num_pillars,
            horizontal_speed=horizontal_speed,
            final_yaw_rate=action_out.yaw_rate,
            yaw_error=yaw_error,
            dfa_q=self._dfa_q,
            dfa_N=self._dfa_N,
            dfa_q_prev=self._dfa_q_prev,
            stage1_waypoint_advance=_stage1_advance,
        )

        reward, components = self._reward_manager.compute(step_state)

        self._ep_reward_sum += float(reward)
        self._logging_manager.record_step(
            step_count=self._step_count,
            reward=float(reward),
            components=components,
            done_reason=done_reason,
            pos=pos,
            dist_xy=dist_xy,
        )
        ep_comp_snapshot: dict | None = None
        if terminated or truncated:
            ep_comp_snapshot = dict(self._logging_manager._ep_component_sums)
            self._logging_manager.record_episode_end(
                done_reason=done_reason,
                episode_reward=self._ep_reward_sum,
                steps=self._step_count + 1,
                pos=pos,
            )

        # 7. Build obs
        obs = self._build_obs(perception, ekf_pos, vel, float(yaw), ang_vel)

        # Update carry state
        self._prev_pos = pos.copy()
        self._prev_dist_xy = dist_xy
        self._prev_action = np.asarray(raw_action, dtype=np.float32)
        # State vector [14:18] = cmd actually sent to PX4 this step (delayed, pre-noise).
        # With delay=N: delayed_cmd = computed_cmd_{t-N} — what drone executed this step.
        # ESC noise is unobservable on real hardware → store pre-noise, not sent_cmd.
        _vz_delayed_norm = self.ecfg.vz_up_limit if delayed_cmd[2] >= 0 else self.ecfg.vz_down_limit
        _ao_limits = np.array([self.ecfg.vx_limit, self.ecfg.vy_limit, _vz_delayed_norm, self.ecfg.yaw_rate_limit], dtype=np.float32)
        self._prev_smoothed_action_n = np.clip(delayed_cmd / _ao_limits, -1.0, 1.0)
        self._step_count += 1
        self._global_step += 1

        if terminated or truncated:
            self._last_done_reason = done_reason
            self._reset_manager.on_episode_end(done_reason)

        info = {
            "pos": pos,
            "start": self._start.copy(),
            "dist_start": self._ep_dist_start,
            "vel": vel,
            "dist_xy": dist_xy,
            "done_reason": done_reason,
            "step_count": self._step_count,
            "reward_components": components,
            "fence_margin": self._fence_margin_xy(pos[:2]),
            "dfa_q": self._dfa_q,
            "dfa_N": self._dfa_N,
        }
        if ep_comp_snapshot is not None:
            info["ep_reward_components"] = ep_comp_snapshot

        return StepResult(
            obs=obs,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    # ------------------------------------------------------------------ #
    # Episode management                                                  #
    # ------------------------------------------------------------------ #

    def _generate_new_goal(self, override_start: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample a new goal position. Annulus ramp during warmup, legacy random after.
        Source: old drone_env.py L1172.
        """
        base_pos = override_start if override_start is not None else self.bridge.get_gazebo_position()
        base_pos = np.asarray(base_pos, dtype=np.float32)

        e = self.ecfg
        if e.goal_dist_ramp_steps > 0:
            ramp_ratio = min(1.0, max(0.0, float(self._global_step) / float(e.goal_dist_ramp_steps)))
        else:
            ramp_ratio = 1.0

        goal_dist_low = e.goal_dist_low_start + (e.goal_dist_low_end - e.goal_dist_low_start) * ramp_ratio
        goal_dist_high = e.goal_dist_high_start + (e.goal_dist_high_end - e.goal_dist_high_start) * ramp_ratio

        if ramp_ratio < 1.0:
            goal_dist_high = max(goal_dist_high, goal_dist_low + float(e.goal_dist_ramp_min_band))
        if goal_dist_high < goal_dist_low:
            goal_dist_high = goal_dist_low

        _gbuf = 3.5  # buffer inside fence wall (fence=±15 → goal_bounds=±11.5)
        goal_bounds = (
            e.fence_x_min + _gbuf,
            e.fence_x_max - _gbuf,
            e.fence_y_min + _gbuf,
            e.fence_y_max - _gbuf,
        )

        if ramp_ratio < 1.0:
            for _ in range(120):
                theta = self._np_random.uniform(-math.pi, math.pi)
                u = self._np_random.uniform(0.0, 1.0)
                r = math.sqrt(
                    goal_dist_low ** 2
                    + u * (goal_dist_high ** 2 - goal_dist_low ** 2)
                )
                gx = float(base_pos[0] + r * math.cos(theta))
                gy = float(base_pos[1] + r * math.sin(theta))
                if not (goal_bounds[0] <= gx <= goal_bounds[1] and goal_bounds[2] <= gy <= goal_bounds[3]):
                    continue
                gz = float(self._np_random.uniform(e.goal_z_min, e.goal_z_max - 1.0))
                dist = float(np.linalg.norm(np.array([gx, gy], dtype=np.float32) - base_pos[:2]))
                if goal_dist_low <= dist <= goal_dist_high:
                    return np.array([gx, gy, gz], dtype=np.float32)
        else:
            min_goal_dist = goal_dist_low
            max_goal_dist = e.goal_xy_norm  # 20m — matches observation normalizer
            for _ in range(50):
                gx = float(self._np_random.uniform(goal_bounds[0], goal_bounds[1]))
                gy = float(self._np_random.uniform(goal_bounds[2], goal_bounds[3]))
                gz = float(self._np_random.uniform(e.goal_z_min, e.goal_z_max - 1.0))
                dist = float(np.linalg.norm(np.array([gx, gy], dtype=np.float32) - base_pos[:2]))
                if min_goal_dist <= dist <= max_goal_dist:
                    return np.array([gx, gy, gz], dtype=np.float32)

        # Fallback to corner farthest from base
        _gcx = (e.fence_x_min + e.fence_x_max) / 2
        _gcy = (e.fence_y_min + e.fence_y_max) / 2
        _grx = e.fence_x_max - _gcx - _gbuf
        _gry = e.fence_y_max - _gcy - _gbuf
        candidates = [
            np.array([_gcx + _grx, _gcy + _gry, 2.0], dtype=np.float32),
            np.array([_gcx + _grx, _gcy - _gry, 2.0], dtype=np.float32),
            np.array([_gcx - _grx, _gcy + _gry, 2.0], dtype=np.float32),
            np.array([_gcx - _grx, _gcy - _gry, 2.0], dtype=np.float32),
        ]
        return max(candidates, key=lambda c: float(np.linalg.norm(c[:2] - base_pos[:2])))

    def _update_goal_xy_radius(self) -> None:
        """Lookup schedule in ecfg and update goal_xy_radius. Source: old drone_env.py L7281."""
        e = self.ecfg
        step = max(0, int(self._global_step) - int(self._stage_start_step))
        idx = 0
        for i, start_step in enumerate(e.goal_xy_radius_schedule_steps):
            if step >= int(start_step):
                idx = i
            else:
                break
        target_radius = float(e.goal_xy_radius_schedule_values[idx])
        self._goal_xy_radius = target_radius

    def _pre_episode_climb_if_low(self, pos_now: np.ndarray) -> None:
        """Climb to safe altitude if armed and below 1m at episode start. Timeout 6s.
        When freeze_vz active, target = band_low + 0.3 (inside the protected band).
        """
        target_z = (self.ecfg.freeze_vz_band_low + 0.3) if self.ecfg.freeze_vz else 1.5
        timeout = 6.0
        t0 = time.monotonic()
        self.logger.info(
            f"[RESET] pre_episode_climb env={self.env_id} "
            f"z={float(pos_now[2]):.2f}m → climbing to {target_z}m"
        )
        while time.monotonic() - t0 < timeout:
            pos = self.bridge.get_gazebo_position()
            if not np.all(np.isfinite(pos)) or float(pos[2]) >= target_z:
                break
            self.bridge.send_velocity(0.0, 0.0, self.ecfg.takeoff_assist_vz, 0.0)
            self.bridge.tick(self.ecfg.dt)
        self.bridge.send_velocity(0.0, 0.0, 0.0, 0.0)
        self.bridge.tick(self.ecfg.dt)

    # ------------------------------------------------------------------ #
    # Observation / depth processing                                      #
    # ------------------------------------------------------------------ #

    def _build_obs(self, bev: np.ndarray, ekf_pos: np.ndarray, vel: np.ndarray, yaw: float, ang_vel: np.ndarray) -> dict:
        """Build obs dict from (3,84,84) BEV tensor + 31-dim state vector."""
        # bev: (3, 84, 84) meters [0, depth_max] from symbolic_extractor
        depth_stack = (bev / self.ecfg.depth_max).astype(np.float32)  # → [0, 1] for CNN

        state_vec = self._build_state_vector(ekf_pos, vel, yaw, ang_vel)

        if not np.all(np.isfinite(state_vec)):
            state_vec = np.nan_to_num(state_vec, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(depth_stack)):
            depth_stack = np.nan_to_num(depth_stack, nan=0.0, posinf=1.0, neginf=0.0)

        return {"depth": depth_stack, "state": state_vec}

    def _build_state_vector(
        self,
        ekf_pos: np.ndarray,
        vel: np.ndarray,
        yaw: float,
        ang_vel: np.ndarray,
    ) -> np.ndarray:
        """
        31-dim ego-centric state vector (all from EKF/IMU, no GT leaks).
        Layout:
          [0:3]   linear vel  [vx, vy, vz]  FLU  (+noise)
          [3:6]   angular vel [rx, py, yaw_r]  FLU  (+noise)
          [6]     altitude z  ENU            (+noise)
          [7:10]  goal in body-FLU [body_x, body_y, body_z]  (using noisy EKF pos+yaw)
          [10:14] orientation [sin_yaw, cos_yaw, pitch/45°, roll/45°]  (no sign ambiguity)
          [14:18] last action A_t [vx, vy, vz, yaw_rate] (in [-1,1])
          [18:22] delta_A1 = A_t - A_{t-1}  (kinematic velocity), clip(·/2)→[-1,1]
          [22:26] delta_A2 = ΔA1_t - ΔA1_{t-1}  (kinematic accel), clip(·/2)→[-1,1]
          [26:30] fence margins body-FLU [fwd, back, left, right] normalized
          [30]    DFA progress q/N ∈ [0,1]
        """
        ekf_pos = np.asarray(ekf_pos, dtype=np.float32)
        vel     = np.asarray(vel,     dtype=np.float32)
        ang_vel = np.asarray(ang_vel, dtype=np.float32)

        # Fetch quaternion here so all obs noise is applied together below
        quat = self.bridge.get_quaternion()
        vel, ang_vel, ekf_pos, yaw, quat = self._noise_manager.apply_obs_noise(
            vel, ang_vel, ekf_pos, yaw, quat
        )

        # Body-frame goal vector (no GT — uses noisy EKF pos + yaw)
        rel_goal = (self._goal[:3] - ekf_pos[:3]).astype(np.float32)
        cy, sy = math.cos(yaw), math.sin(yaw)
        body_x = float( cy * rel_goal[0] + sy * rel_goal[1])
        body_y = float(-sy * rel_goal[0] + cy * rel_goal[1])
        body_z = float(rel_goal[2])

        # ENU velocity → FLU body frame
        vel_flu = np.array([
            float( cy * vel[0] + sy * vel[1]),
            float(-sy * vel[0] + cy * vel[1]),
            float(vel[2])
        ], dtype=np.float32)

        # FRD angular velocity → FLU (flip y and z)
        ang_vel_flu = np.array([
            float(ang_vel[0]),
            -float(ang_vel[1]),
            -float(ang_vel[2])
        ], dtype=np.float32)

        # Normalize to roughly [-1, 1] using known physical bounds.
        # vz uses asymmetric limits: vz_up_limit for climb, vz_down_limit for descent.
        _vz_vel_norm = self.ecfg.vz_up_limit if vel_flu[2] >= 0 else self.ecfg.vz_down_limit
        vel_flu_n = vel_flu / np.array([self.ecfg.vx_limit, self.ecfg.vy_limit, _vz_vel_norm], dtype=np.float32)
        ang_vel_flu_n = ang_vel_flu / np.array([math.pi, math.pi, self.ecfg.yaw_rate_limit], dtype=np.float32)
        alt_n = float(ekf_pos[2]) / self.ecfg.fence_z_max
        goal_n = np.array([
            body_x / self.ecfg.goal_xy_norm,
            body_y / self.ecfg.goal_xy_norm,
            body_z / self.ecfg.goal_z_norm,
        ], dtype=np.float32)

        # Orientation as [sin_yaw, cos_yaw, pitch_n, roll_n] — no quaternion sign ambiguity
        # quat already fetched + noised above via apply_obs_noise
        qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
        roll  = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        _MAX_TILT = math.pi / 4  # 45° normalizer — typical multirotor max tilt
        orientation = np.array([
            math.sin(yaw),
            math.cos(yaw),
            pitch / _MAX_TILT,
            roll  / _MAX_TILT,
        ], dtype=np.float32)

        # [14:18] A_t — smoothed cmd actually sent to drone, normalized to [-1, 1]
        last_action = self._prev_smoothed_action_n.copy()

        # [18:22] delta_A1 = A_t - A_{t-1}  (kinematic velocity of command)
        delta_a1 = last_action - self._prev_prev_smoothed_action_n
        delta_a1_norm = np.clip(delta_a1 / 2.0, -1.0, 1.0)

        # [22:26] delta_A2 = ΔA1_t - ΔA1_{t-1}  (kinematic accel of command)
        # Uses self._prev_delta_a1 stored from PREVIOUS step before rolling
        delta_a2 = delta_a1 - self._prev_delta_a1
        delta_a2_norm = np.clip(delta_a2 / 2.0, -1.0, 1.0)

        # Roll kinematic buffers AFTER computing both deltas
        self._prev_delta_a1 = delta_a1.copy()
        self._prev_prev_smoothed_action_n = last_action.copy()

        # Fence proximity sensor — sparse warning in body-FLU frame.
        # Uses signed wall distances so virtual fence exits are handled correctly:
        # d < 0 means drone already crossed that wall → clamp warning to 1.0.
        px, py = float(ekf_pos[0]), float(ekf_pos[1])
        _fx_min = float(self.ecfg.fence_x_min)
        _fx_max = float(self.ecfg.fence_x_max)
        _fy_min = float(self.ecfg.fence_y_min)
        _fy_max = float(self.ecfg.fence_y_max)
        _WARNING_ZONE = 2.0  # metres — sensor active inside this distance from wall

        _d_xp = _fx_max - px   # positive when inside +x wall, negative when past it
        _d_xn = px - _fx_min   # positive when inside -x wall
        _d_yp = _fy_max - py
        _d_yn = py - _fy_min

        def _fence_warn(dx: float, dy: float) -> float:
            dists = []
            if dx > 1e-6:
                dists.append(_d_xp / dx)
            elif dx < -1e-6:
                dists.append(_d_xn / (-dx))
            if dy > 1e-6:
                dists.append(_d_yp / dy)
            elif dy < -1e-6:
                dists.append(_d_yn / (-dy))
            d = min(dists) if dists else math.inf
            return max(0.0, min(1.0, (_WARNING_ZONE - d) / _WARNING_ZONE))

        fence_flu_n = np.array([
            _fence_warn( cy,  sy),  # forward
            _fence_warn(-cy, -sy),  # back
            _fence_warn(-sy,  cy),  # left
            _fence_warn( sy, -cy),  # right
        ], dtype=np.float32)

        # obs[30]: hybrid DFA progress — discrete milestone + continuous within-segment
        # (dfa_q + t_segment) / dfa_N: smooth for MLP, still encodes which pillar region
        if self._dfa_q >= self._dfa_N:
            _dfa_val = 1.0
        else:
            _q = self._dfa_q
            _cvec = self._goal[:2] - self._start[:2]
            _cnorm = float(np.linalg.norm(_cvec)) + 1e-8
            _cdir = _cvec / _cnorm
            _t_drone = float(np.dot(ekf_pos[:2] - self._start[:2], _cdir))
            _bounds = self._dfa_segment_t_boundaries
            _t_s = _bounds[_q] if _q < len(_bounds) else _cnorm
            _t_e = _bounds[_q + 1] if _q + 1 < len(_bounds) else _cnorm
            _seg_len = _t_e - _t_s
            _t_cap = 1.0 - 1.0 / float(max(self._dfa_N, 2))
            _t_seg = max(0.0, min(_t_cap, (_t_drone - _t_s) / _seg_len)) if _seg_len > 1e-6 else 0.0
            _dfa_val = min(1.0, float(_q + _t_seg) / float(max(self._dfa_N, 1)))
        dfa_progress = np.array([_dfa_val], dtype=np.float32)

        state = np.concatenate([
            vel_flu_n,                                  # [0:3]
            ang_vel_flu_n,                              # [3:6]
            np.array([alt_n], dtype=np.float32),        # [6]
            goal_n,                                     # [7:10]
            orientation,                                # [10:14]
            last_action,                                # [14:18]
            delta_a1_norm,                              # [18:22] kinematic vel
            delta_a2_norm,                              # [22:26] kinematic accel
            fence_flu_n,                                # [26:30]
            dfa_progress,                               # [30] ∈ [0.0, 1.0]
        ]).astype(np.float32)

        state[0:6]   = np.clip(state[0:6],   -1.0, 1.0)   # vel + ang_vel
        state[7:10]  = np.clip(state[7:10],  -3.0, 3.0)   # goal_n
        state[12:14] = np.clip(state[12:14], -1.0, 1.0)   # pitch/roll
        assert state.shape == (31,), f"state.shape expected (31,), got {state.shape}"
        return state

    # ------------------------------------------------------------------ #
    # Terminal / fence checks                                             #
    # ------------------------------------------------------------------ #

    def _check_terminal(
        self,
        pos: np.ndarray,
        pillar_snap: PillarSnapshot,
        step_count: int,
        goal_xy_radius: float,
    ) -> tuple[bool, bool, str]:
        """
        Returns (terminated, truncated, done_reason).
        Source: old drone_env.py L6480-6534.
        """
        dist_xy = float(np.linalg.norm(self._goal[:2] - pos[:2]))

        # Goal reached
        if dist_xy <= goal_xy_radius:
            return True, False, "goal_xy"

        # Physical collision (pillar)
        if pillar_snap.is_collision:
            return True, False, "collision"

        # Out of fence
        if self._out_of_fence(pos):
            return True, False, "out_of_fence"

        # Altitude too low (fell to ground after being airborne)
        if self._has_been_airborne and float(pos[2]) < self.ecfg.fence_z_min + 0.2:
            return True, False, "fell_to_ground"

        # Flipped
        if self.bridge.is_flipped():
            return True, False, "flipped"

        # PX4 in failsafe — episode corrupted, terminate immediately
        if self.bridge.failsafe and step_count > 5:
            return True, False, "px4_failsafe"

        # EKF callbacks dead — DDS stalled, hard reset needed
        if not self.bridge.is_px4_callbacks_healthy(max_est_flags_age=15.0):
            self.logger.warning(
                f"[TERMINAL] env={self.env_id} ekf_callbacks_dead "
                f"est_flags_age={time.monotonic() - self.bridge._last_estimator_flags_wall:.1f}s"
            )
            return True, False, "ekf_callbacks_dead"

        # Max steps (truncation)
        if step_count >= self.ecfg.max_steps - 1:
            return False, True, "max_steps"

        return False, False, ""

    def _out_of_fence(self, pos: np.ndarray) -> bool:
        """Source: old drone_env.py L7874."""
        e = self.ecfg
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        return (
            x < e.fence_x_min or x > e.fence_x_max
            or y < e.fence_y_min or y > e.fence_y_max
            or z > e.fence_z_max
        )

    def _px4_estimator_insane(self) -> bool:
        """Source: old drone_env.py L8091."""
        px4_lpos = getattr(self.bridge, "px4_lpos", np.zeros(3, dtype=np.float32))
        vel = self.bridge.get_linear_velocity()
        if not np.all(np.isfinite(px4_lpos)):
            return True
        if not np.all(np.isfinite(vel)):
            return True
        e = self.ecfg
        if abs(float(px4_lpos[2])) > e.max_px4_lpos_z_abs:
            return True
        if abs(float(vel[2])) > e.max_px4_vel_z_abs:
            return True
        if float(np.linalg.norm(vel)) > e.max_px4_speed:
            return True
        return False

    # ------------------------------------------------------------------ #
    # Misc utilities                                                      #
    # ------------------------------------------------------------------ #

    def _compute_dfa_segment_boundaries(self) -> list:
        """Corridor t-boundaries for each DFA segment: [0, t_p1, t_p2, ..., corridor_norm].
        Used to compute smooth hybrid obs[30] = (dfa_q + t_within_segment) / dfa_N.
        """
        corridor_vec = self._goal[:2] - self._start[:2]
        corridor_norm = float(np.linalg.norm(corridor_vec)) + 1e-8
        corridor_dir = corridor_vec / corridor_norm
        if self._dfa_use_corridor:
            active_names = self._pillar_manager._active_corridor_pillar_names
            t_vals = [
                float(np.dot(
                    np.array([float(m["x"]), float(m["y"])], dtype=np.float32) - self._start[:2],
                    corridor_dir,
                ))
                for m in self._pillar_manager.last_pillar_metadata
                if str(m.get("name", "")) in active_names
            ]
        else:
            t_vals = [
                float(np.dot(wp - self._start[:2], corridor_dir))
                for wp in self._pillar_manager._stage1_sgs_xy
            ]
        t_vals.sort()
        return [0.0] + t_vals + [corridor_norm]

    def _fence_margin_xy(self, pos_xy: np.ndarray) -> float:
        """Minimum distance to XY fence boundary."""
        e = self.ecfg
        x, y = float(pos_xy[0]), float(pos_xy[1])
        return min(
            x - e.fence_x_min,
            e.fence_x_max - x,
            y - e.fence_y_min,
            e.fence_y_max - y,
        )

    def _pose_near_start(self, pos: np.ndarray) -> bool:
        """Source: old drone_env.py L7766."""
        if not np.all(np.isfinite(pos)):
            return False
        xy_err = float(np.linalg.norm(pos[:2] - self._start[:2]))
        z = float(pos[2])
        return xy_err < self.ecfg.start_clearance_xy and -0.25 <= z <= 0.35

    def _wrap_pi(self, angle: float) -> float:
        """Source: old drone_env.py L7870."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def get_isolation_info(self) -> dict:
        """Return env metadata for multi-env logging. Source: old drone_env.py L3591."""
        model_name = getattr(self.bridge, "model_name", f"x500_depth_{self.env_id}")
        gz_partition = getattr(self.bridge, "gz_partition", "")
        return {
            "env_id": self.env_id,
            "model_name": model_name,
            "gz_partition": gz_partition,
            "num_pillars": self._pillar_manager._p.num_pillars,
        }
