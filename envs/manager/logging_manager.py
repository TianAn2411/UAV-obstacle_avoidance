"""
LoggingManager — per-environment training telemetry.

Responsibilities:
  - Per-step debug logging (_write_step_log)
  - Episode start/end block logging
  - Reward component summaries
  - k-step aggregate metrics
  - CSV episode summary rows
  - Log-file rotation (size-based, >50 MB)
  - Reset timing / warning logs

No ROS imports. Only stdlib: logging, csv, os, time, math.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import time
from dataclasses import asdict, fields
from typing import Optional

import numpy as np

from obstacle_avoidance.envs.manager.reward_manager import RewardComponents
from obstacle_avoidance.utils.logger import setup_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_ROTATE_MAX_BYTES: int = 50 * 1024 * 1024  # 50 MB
_CSV_HEADER: list[str] = [
    "episode",
    "steps",
    "done_reason",
    "success",
    "ep_reward",
    "start_dist_xy",
    "final_dist_xy",
    "min_dist_xy",
    "path_eff",
    # reward components (all fields from RewardComponents)
    *[f.name for f in fields(RewardComponents)],
]


class LoggingManager:
    """Manages all training telemetry for a single environment instance."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, env_id: int, log_dir: str | None, log_every_steps: int, enable_csv: bool = False) -> None:
        self.env_id: int = env_id
        self.log_dir: str | None = log_dir
        self.log_every_steps: int = max(1, log_every_steps)

        # ---- log-rotation state ----------------------------------------
        self._log_path: str | None = None           # current info log file path
        self._debug_log_path: str | None = None     # current debug log file path
        self._log_rotate_suffix: int = 0            # numeric suffix for rotated files

        # ---- logger -------------------------------------------------------
        self.logger: logging.Logger = self._init_logger()

        # ---- CSV writer ---------------------------------------------------
        self._csv_file = None
        self._csv_writer = None
        if log_dir is not None and enable_csv:
            self._init_csv()

        # ---- episode-level accumulators -----------------------------------
        self.episode_count: int = 0
        self.global_env_step: int = 0    # incremented each record_step call

        self._ep_reward_sum: float = 0.0
        self._ep_step_count: int = 0
        self._ep_start_dist_xy: float = 0.0
        self._ep_final_dist_xy: float = 0.0
        self._ep_min_dist_xy: float = math.inf
        self._ep_path_len_xy: float = 0.0
        self._ep_prev_pos_xy: np.ndarray | None = None
        self._ep_start_wall: float = time.time()

        # per-component reward sums (keyed by RewardComponents field name)
        self._ep_component_sums: dict[str, float] = {
            f.name: 0.0 for f in fields(RewardComponents)
        }

        # ---- k-step aggregate metrics ------------------------------------
        self._kstep_interval: int = max(1, log_every_steps * 20)  # ~20 log windows
        self._kstep_last_report_block: int = -1
        self._kstep_total_episodes: int = 0
        self._kstep_goal_success_count: int = 0
        self._kstep_max_steps_count: int = 0
        self._kstep_out_of_fence_count: int = 0
        self._kstep_sum_final_minus_start_dist: float = 0.0
        self._kstep_sum_min_dist: float = 0.0

        # ---- wall-time gap tracking (for reset phase monitoring) ----------
        self.last_step_wall_time: float = time.monotonic()
        self.first_step_after_reset: bool = True

    # ------------------------------------------------------------------
    # Public entry points (called by TrainManager)
    # ------------------------------------------------------------------

    def record_step(
        self,
        step_count: int,
        reward: float,
        components: RewardComponents,
        done_reason: str,
        pos: np.ndarray,
        dist_xy: float,
    ) -> None:
        """Accumulate per-step data; emit step log at every log_every_steps boundary."""
        self.global_env_step += 1
        self._ep_step_count += 1
        self._ep_reward_sum += float(reward)

        # accumulate per-component sums
        comp_dict = asdict(components)
        for key in self._ep_component_sums:
            self._ep_component_sums[key] += float(comp_dict.get(key, 0.0))

        # track distance stats
        d = float(dist_xy)
        self._ep_final_dist_xy = d
        if d < self._ep_min_dist_xy:
            self._ep_min_dist_xy = d

        # accumulate XY path length
        p = np.asarray(pos, dtype=np.float64)
        if self._ep_prev_pos_xy is not None:
            self._ep_path_len_xy += float(np.linalg.norm(p[:2] - self._ep_prev_pos_xy))
        self._ep_prev_pos_xy = p[:2].copy()

        # log-file rotation check
        self._maybe_rotate_env_log()

        # periodic step log
        if self.global_env_step % self.log_every_steps == 0:
            self._write_step_log(
                step_count=step_count,
                reward=reward,
                components=components,
                pos=pos,
                dist_xy=dist_xy,
            )

        # k-step metrics
        self._maybe_log_kstep_metrics()

    def record_episode_end(self, done_reason: str, episode_reward: float, steps: int, pos: np.ndarray | None = None) -> None:
        """Write episode end logs and CSV row, then reset accumulators."""
        self._ep_reward_sum = float(episode_reward)  # authoritative value from caller
        self._ep_step_count = int(steps)
        wall_time = time.time() - self._ep_start_wall

        self._log_eps_end_block(
            done_reason=done_reason,
            pos=pos if pos is not None else np.zeros(3, dtype=np.float32),
            dist_xy=self._ep_final_dist_xy,
            reward=episode_reward,
            wall_time=wall_time,
            reset_action="unknown",
        )
        self._log_eps_reward_summary()
        self._write_ep_summary(done_reason)
        self._update_kstep_episode_metrics(done_reason)

        # reset accumulators
        self.episode_count += 1
        self._ep_reward_sum = 0.0
        self._ep_step_count = 0
        self._ep_start_dist_xy = self._ep_final_dist_xy
        self._ep_final_dist_xy = 0.0
        self._ep_min_dist_xy = math.inf
        self._ep_path_len_xy = 0.0
        self._ep_prev_pos_xy = None
        self._ep_start_wall = time.time()
        for key in self._ep_component_sums:
            self._ep_component_sums[key] = 0.0

    def log_episode_start(
        self, reset_mode: str, reason: str, pos: np.ndarray, dist_xy: float
    ) -> None:
        """Log episode header block (ported from drone_env._log_eps_start_block)."""
        self._ep_start_dist_xy = float(dist_xy)
        self._ep_final_dist_xy = float(dist_xy)
        self._ep_min_dist_xy = float(dist_xy)
        self._ep_path_len_xy = 0.0
        self._ep_prev_pos_xy = None
        self._ep_start_wall = time.time()
        self._reset_step_wall_gap_tracking()

        self.logger.info("==================== EPS START ====================")
        self.logger.info(
            f"env={self.env_id} ep={self.episode_count} "
            f"mode={reset_mode} reason={reason}"
        )
        arr = np.asarray(pos, dtype=np.float32).reshape(-1)
        self.logger.info(
            f"cur_pos={self._fmt_xyz(pos)} dist_xy={float(dist_xy):.2f}"
        )
        if arr.size >= 3 and np.isfinite(arr[2]) and float(arr[2]) > 0.3:
            self.logger.info(f"[DRONE IS AIRBORNE] z={float(arr[2]):.2f}")
        self.logger.info("===================================================")

    def log_reset_timing(
        self,
        phase: str,
        duration: float,
        last_reason: str | None = None,
        total_reset_round: int | None = None,
        attempt: int | None = None,
    ) -> None:
        """Log reset phase timing (ported from drone_env._log_reset_timing)."""
        self.logger.debug(
            "[RESET TIMING] "
            f"mono={time.monotonic():.3f} "
            f"env_id={self.env_id} "
            f"phase={phase} "
            f"duration={float(duration):.3f} "
            f"last_reason={last_reason} "
            f"reset_count={self.episode_count} "
            f"total_reset_round={total_reset_round} "
            f"attempt={attempt}"
        )
        _major_phases = {
            "out_of_fence_velocity_rescue",
            "terminal_relaunch_attempt",
            "continuous_reset_sync_frame",
            "continuous_reset_spawn",
            "force_disarm",
            "teleport",
            "settle_after_teleport",
            "arm_and_takeoff",
            "lift_warmup",
            "sync_frame",
            "spawn_pillars",
            "hard_reset_unsafe_pose_recovery",
            "hard_reset_reset_round_failed",
        }
        if phase in _major_phases:
            self.logger.info(
                "[RESET][PHASE] "
                f"env={self.env_id} "
                f"phase={phase} "
                "status=ok "
                f"duration={float(duration):.3f}s"
            )

    def log_reset_warn(self, phase: str, reason: str, fallback: str) -> None:
        """Log warning when reset falls back to unexpected path (ported from drone_env._log_reset_warn)."""
        self.logger.warning(
            "[RESET][WARN] "
            f"env={self.env_id} "
            f"phase={phase} "
            "status=fail "
            f"reason={reason} "
            f"fallback={fallback}"
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _fmt_xyz(self, vec: np.ndarray) -> str:
        """Format [x, y, z] as '(x.xx,y.xx,z.xx)'."""
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return "(nan,nan,nan)"
        return f"({arr[0]:.2f},{arr[1]:.2f},{arr[2]:.2f})"

    def _fmt_xy(self, vec: np.ndarray) -> str:
        """Format [x, y] as '(x.xx,y.xx)'."""
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
            return "(nan,nan)"
        return f"({arr[0]:.2f},{arr[1]:.2f})"

    # ------------------------------------------------------------------
    # Internal log writers
    # ------------------------------------------------------------------

    def _write_step_log(
        self,
        step_count: int,
        reward: float,
        components: RewardComponents,
        pos: np.ndarray,
        dist_xy: float,
    ) -> None:
        """Write per-step log entry (simplified, debug-level)."""
        comp = asdict(components)
        self.logger.debug(
            "[STEP] "
            f"env={self.env_id} "
            f"global_step={self.global_env_step} "
            f"ep_step={step_count} "
            f"pos={self._fmt_xyz(pos)} "
            f"dist_xy={float(dist_xy):.2f} "
            f"reward={float(reward):.3f} "
            f"ep_reward={self._ep_reward_sum:.3f} "
            f"progress={comp.get('progress', 0.0):.3f} "
            f"terminal={comp.get('terminal', 0.0):.3f} "
            f"velocity_goal={comp.get('velocity_goal', 0.0):.3f} "
            f"obstacle_visibility={comp.get('obstacle_visibility', 0.0):.3f} "
            f"pillar_attention={comp.get('pillar_attention', 0.0):.3f}"
        )

    def _log_eps_end_block(
        self,
        done_reason: str,
        pos: np.ndarray,
        dist_xy: float,
        reward: float,
        wall_time: float,
        reset_action: str,
    ) -> None:
        """Write episode end summary block (ported from drone_env._log_eps_end_block)."""
        success = done_reason in ("goal_xy", "goal_3d", "success")
        self.logger.info("==================== EPS END ======================")
        self.logger.info(
            f"env={self.env_id} ep={self.episode_count} reason={done_reason} "
            f"success={str(success).lower()} steps={self._ep_step_count}"
        )
        self.logger.info(
            f"final_pos={self._fmt_xyz(pos)}"
        )
        self.logger.info(
            f"final_dist_xy={float(dist_xy):.2f} total_reward={float(self._ep_reward_sum):.2f}"
        )
        self.logger.info(
            f"duration_s={float(wall_time):.2f} reset_next={reset_action}"
        )
        self.logger.info("===================================================")

    def _log_eps_reward_summary(self) -> None:
        """Write per-component reward totals for the episode (ported from drone_env._log_eps_reward_summary)."""
        s = self._ep_component_sums
        self.logger.info(
            "[EPS REWARD] "
            f"total={s.get('total', 0.0):.2f} "
            f"terminal={s.get('terminal', 0.0):.2f} "
            f"progress={s.get('progress', 0.0):.2f} "
            f"velocity_goal={s.get('velocity_goal', 0.0):.2f} "
            f"heading_goal={s.get('heading_goal', 0.0):.2f} "
            f"yaw_align={s.get('yaw_align', 0.0):.2f} "
            f"face_progress={s.get('face_progress', 0.0):.2f} "
            f"obstacle_visibility={s.get('obstacle_visibility', 0.0):.2f} "
            f"pillar_attention={s.get('pillar_attention', 0.0):.2f} "
            f"obstacle_slowdown={s.get('obstacle_slowdown', 0.0):.2f} "
            f"bypass_decision={s.get('bypass_decision', 0.0):.2f} "
            f"bypass_progress={s.get('bypass_progress', 0.0):.2f} "
            f"stage1_subgoal={s.get('stage1_subgoal', 0.0):.2f} "
            f"bypass_subgoal={s.get('bypass_subgoal', 0.0):.2f} "
            f"ring_subgoal={s.get('ring_subgoal', 0.0):.2f} "
            f"pillar_passed={s.get('pillar_passed', 0.0):.2f} "
            f"pillar_clearance_soft={s.get('pillar_clearance_soft', 0.0):.2f} "
            f"pillar_too_close={s.get('pillar_too_close', 0.0):.2f} "
            f"near_pillar_speed={s.get('near_pillar_speed', 0.0):.2f} "
            f"collision_course={s.get('collision_course', 0.0):.2f} "
            f"near_miss={s.get('near_miss', 0.0):.2f} "
            f"backwards_yaw={s.get('backwards_yaw', 0.0):.2f} "
            f"smooth={s.get('smooth', 0.0):.2f} "
            f"speed_penalty={s.get('speed_penalty', 0.0):.2f} "
            f"fall_penalty={s.get('fall_penalty', 0.0):.2f} "
            f"lateral={s.get('lateral', 0.0):.2f} "
            f"near_fence={s.get('near_fence', 0.0):.2f} "
            f"start_zone={s.get('start_zone', 0.0):.2f} "
            f"ground={s.get('ground', 0.0):.2f} "
            f"altitude={s.get('altitude', 0.0):.2f} "
            f"time={s.get('time', 0.0):.2f}"
        )

    def _write_ep_summary(self, done_reason: str) -> None:
        """Write episode summary: structured log block + CSV row (ported from drone_env._write_ep_summary)."""
        try:
            dist_delta = self._ep_start_dist_xy - self._ep_final_dist_xy
            path_eff = max(0.0, dist_delta) / max(self._ep_path_len_xy, 1e-6)
            goal_success = done_reason in ("goal_xy", "goal_3d", "success")
            min_dist = self._ep_min_dist_xy if math.isfinite(self._ep_min_dist_xy) else self._ep_final_dist_xy
            s = self._ep_component_sums

            self.logger.info(
                "[EP SUMMARY][BASIC] "
                f"env={self.env_id} ep={self.episode_count} steps={self._ep_step_count} "
                f"reason={done_reason} success={goal_success} "
                f"ep_reward={self._ep_reward_sum:.1f} "
                f"start_dist={self._ep_start_dist_xy:.2f} "
                f"final_dist={self._ep_final_dist_xy:.2f} "
                f"min_dist={min_dist:.2f} "
                f"path_eff={path_eff:.4f}"
            )
            self.logger.info(
                "[EP SUMMARY][REWARD_DETAIL] "
                f"progress={s.get('progress', 0.0):.2f} "
                f"velocity_goal={s.get('velocity_goal', 0.0):.2f} "
                f"time={s.get('time', 0.0):.2f} "
                f"altitude={s.get('altitude', 0.0):.2f} "
                f"ground={s.get('ground', 0.0):.2f} "
                f"smooth={s.get('smooth', 0.0):.2f} "
                f"speed_penalty={s.get('speed_penalty', 0.0):.2f} "
                f"fall_penalty={s.get('fall_penalty', 0.0):.2f} "
                f"lateral={s.get('lateral', 0.0):.2f} "
                f"near_fence={s.get('near_fence', 0.0):.2f} "
                f"pillar_clearance_soft={s.get('pillar_clearance_soft', 0.0):.2f} "
                f"pillar_too_close={s.get('pillar_too_close', 0.0):.2f} "
                f"collision_course={s.get('collision_course', 0.0):.2f} "
                f"near_miss={s.get('near_miss', 0.0):.2f} "
                f"terminal={s.get('terminal', 0.0):.2f} "
                f"total={s.get('total', 0.0):.2f}"
            )

            # CSV row
            if self._csv_writer is not None:
                row: dict[str, object] = {
                    "episode": self.episode_count,
                    "steps": self._ep_step_count,
                    "done_reason": done_reason,
                    "success": int(goal_success),
                    "ep_reward": round(self._ep_reward_sum, 4),
                    "start_dist_xy": round(self._ep_start_dist_xy, 4),
                    "final_dist_xy": round(self._ep_final_dist_xy, 4),
                    "min_dist_xy": round(min_dist, 4),
                    "path_eff": round(path_eff, 6),
                }
                for fname in self._ep_component_sums:
                    row[fname] = round(self._ep_component_sums[fname], 4)
                self._csv_writer.writerow(row)
                if self._csv_file is not None:
                    self._csv_file.flush()
        except Exception as exc:
            try:
                self.logger.debug(f"[EP SUMMARY] failed: {exc}")
            except Exception:
                pass

    def _update_kstep_episode_metrics(self, done_reason: str) -> None:
        """Accumulate metrics at k-step boundaries (ported from drone_env._update_kstep_episode_metrics)."""
        self._kstep_total_episodes += 1
        if done_reason in ("goal_xy", "goal_3d", "success"):
            self._kstep_goal_success_count += 1
        if done_reason == "max_steps":
            self._kstep_max_steps_count += 1
        if done_reason == "out_of_fence":
            self._kstep_out_of_fence_count += 1

        min_dist = self._ep_min_dist_xy if math.isfinite(self._ep_min_dist_xy) else self._ep_final_dist_xy
        self._kstep_sum_final_minus_start_dist += float(
            self._ep_final_dist_xy - self._ep_start_dist_xy
        )
        self._kstep_sum_min_dist += float(min_dist)

    def _maybe_log_kstep_metrics(self) -> None:
        """Emit k-step metric summary if at block boundary (ported from drone_env._maybe_log_kstep_metrics)."""
        if self._kstep_interval <= 0:
            return

        cur_block = self.global_env_step // self._kstep_interval
        if cur_block <= self._kstep_last_report_block:
            return

        total_eps = self._kstep_total_episodes
        if total_eps > 0:
            out_of_fence_ratio = float(self._kstep_out_of_fence_count) / float(total_eps)
            mean_final_minus_start_dist = (
                self._kstep_sum_final_minus_start_dist / float(total_eps)
            )
            mean_min_dist = self._kstep_sum_min_dist / float(total_eps)
        else:
            out_of_fence_ratio = 0.0
            mean_final_minus_start_dist = 0.0
            mean_min_dist = 0.0

        self.logger.info(
            "[KSTEP METRICS] "
            f"env_id={self.env_id} "
            f"global_env_step={self.global_env_step} "
            f"window={self._kstep_interval} "
            f"goal_success_count={self._kstep_goal_success_count} "
            f"max_steps_count={self._kstep_max_steps_count} "
            f"out_of_fence_count={self._kstep_out_of_fence_count} "
            f"total_episodes={total_eps} "
            f"out_of_fence_ratio={out_of_fence_ratio:.4f} "
            f"mean_final_minus_start_dist={mean_final_minus_start_dist:.3f} "
            f"mean_min_dist={mean_min_dist:.3f}"
        )

        self._kstep_last_report_block = cur_block
        self._kstep_goal_success_count = 0
        self._kstep_max_steps_count = 0
        self._kstep_out_of_fence_count = 0
        self._kstep_total_episodes = 0
        self._kstep_sum_final_minus_start_dist = 0.0
        self._kstep_sum_min_dist = 0.0

    def _log_collision_check_block(self) -> None:
        """Log collision check state for debug (stub — no collision state in manager)."""
        self.logger.debug(
            f"[COLLISION CHECK] env={self.env_id} ep={self.episode_count} "
            "(no collision detail available in LoggingManager)"
        )

    def _current_log_block_path(self) -> str:
        """Return current log file path; rotates by size (>50 MB)."""
        if self.log_dir is None:
            return ""
        base = os.path.join(
            self.log_dir,
            f"env{self.env_id}_ep{self.episode_count:06d}",
        )
        if self._log_rotate_suffix == 0:
            return f"{base}.txt"
        return f"{base}_{self._log_rotate_suffix:03d}.txt"

    def _maybe_rotate_env_log(self) -> None:
        """Rotate log file if the current one exceeds 50 MB."""
        if self._log_path is None:
            return
        try:
            size = os.path.getsize(self._log_path)
        except OSError:
            return
        if size < _LOG_ROTATE_MAX_BYTES:
            return

        self._log_rotate_suffix += 1
        new_info = self._build_log_path(suffix=self._log_rotate_suffix)
        new_debug = self._build_debug_log_path(suffix=self._log_rotate_suffix)
        self._log_path = new_info
        self._debug_log_path = new_debug
        self.logger = setup_logger(
            f"ENV_{self.env_id}",
            info_log_file=new_info,
            debug_log_file=new_debug,
        )
        try:
            self.logger.info(
                f"[LOG ROTATE] env_id={self.env_id} "
                f"global_env_step={self.global_env_step} "
                f"suffix={self._log_rotate_suffix} "
                f"new_log_file={new_info}"
            )
        except Exception:
            pass

    def _reset_step_wall_gap_tracking(self) -> None:
        """Reset wall-clock gap tracker between steps (ported from drone_env._reset_step_wall_gap_tracking)."""
        self.last_step_wall_time = time.monotonic()
        self.first_step_after_reset = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_log_path(self, suffix: int = 0) -> str:
        assert self.log_dir is not None
        base = os.path.join(self.log_dir, f"env_{self.env_id}_main")
        return f"{base}.txt" if suffix == 0 else f"{base}_{suffix:03d}.txt"

    def _build_debug_log_path(self, suffix: int = 0) -> str:
        assert self.log_dir is not None
        base = os.path.join(self.log_dir, f"env_{self.env_id}_debug")
        return f"{base}.txt" if suffix == 0 else f"{base}_{suffix:03d}.txt"

    def _init_logger(self) -> logging.Logger:
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            info_path = self._build_log_path(0)
            debug_path = self._build_debug_log_path(0)
            self._log_path = info_path
            self._debug_log_path = debug_path
            return setup_logger(
                f"ENV_{self.env_id}",
                info_log_file=info_path,
                debug_log_file=debug_path,
            )
        return setup_logger(f"ENV_{self.env_id}")

    def _init_csv(self) -> None:
        assert self.log_dir is not None
        os.makedirs(self.log_dir, exist_ok=True)
        csv_path = os.path.join(self.log_dir, f"env_{self.env_id}_episodes.csv")
        write_header = not os.path.exists(csv_path)
        self._csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=_CSV_HEADER,
            extrasaction="ignore",
        )
        if write_header:
            self._csv_writer.writeheader()
            self._csv_file.flush()
