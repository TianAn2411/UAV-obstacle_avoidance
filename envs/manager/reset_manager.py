import math
import time
from dataclasses import dataclass

import numpy as np

from obstacle_avoidance.configs.env_config import EnvConfig
from obstacle_avoidance.utils.bridge_factory import ROSBridge
from obstacle_avoidance.utils.logger import setup_logger


@dataclass
class ResetDecision:
    mode: str            # "continuous", "hard", "multi_env_fast", "rescue_then_continuous", "startup_arm"
    reason: str
    do_respawn_pillars: bool


class ResetManager:
    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def __init__(self, bridge: ROSBridge, ecfg: EnvConfig, env_id: int = 0) -> None:
        self.bridge = bridge
        self.ecfg = ecfg
        self.env_id = env_id
        self.logger = setup_logger(f"RESET_{env_id}")

        self.reset_count = 0
        self.consecutive_failures = 0
        self.rescue_fail_count = 0
        self.rescue_success_count = 0

    # ------------------------------------------------------------------ #
    # Main interface                                                      #
    # ------------------------------------------------------------------ #

    def classify_reset(self, done_reason: str, fence_margin: float) -> ResetDecision:
        """Port of _classify_reset_action from old drone_env.py L3684."""
        reason = str(done_reason)

        # Shield terminal reasons excluded — future real-life only
        if self._is_shield_terminal_reason(reason):
            return ResetDecision(mode="hard", reason=reason, do_respawn_pillars=True)

        base = self._classify_reset_action(reason)

        if base == "continuous":
            if reason in {"fell_to_ground", "ground", "flipped"}:
                # Drone is already on ground — no teleport or EKF re-notify needed.
                # Inside fence: rearm + takeoff (startup_arm). Near fence: hard reset to teleport first.
                if fence_margin < self.ecfg.continuous_reset_fence_margin_thresh:
                    return ResetDecision(mode="hard", reason=reason, do_respawn_pillars=True)
                return ResetDecision(mode="startup_arm", reason=reason, do_respawn_pillars=True)
            if fence_margin < self.ecfg.continuous_reset_fence_margin_thresh:
                return ResetDecision(mode="rescue_then_continuous", reason=reason, do_respawn_pillars=True)
            # Upgrade to multi_env_fast if eligible
            if self._should_use_multi_env_fast_reset(reason):
                return ResetDecision(mode="multi_env_fast", reason=reason, do_respawn_pillars=True)
            return ResetDecision(mode="continuous", reason=reason, do_respawn_pillars=True)

        if base == "rescue_then_continuous":
            return ResetDecision(mode="rescue_then_continuous", reason=reason, do_respawn_pillars=True)

        if base == "startup_arm":
            return ResetDecision(mode="startup_arm", reason=reason, do_respawn_pillars=True)

        return ResetDecision(mode="hard", reason=reason, do_respawn_pillars=True)

    def on_episode_end(self, done_reason: str) -> None:
        self.reset_count += 1
        if done_reason in {"collision", "out_of_fence", "flipped", "fell_to_ground", "ground"}:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

    def get_reset_helpers(self) -> dict:
        return {
            "reset_count": self.reset_count,
            "consecutive_failures": self.consecutive_failures,
            "rescue_fail_count": self.rescue_fail_count,
            "rescue_success_count": self.rescue_success_count,
        }

    # ------------------------------------------------------------------ #
    # Reset execution                                                     #
    # ------------------------------------------------------------------ #

    def continuous_episode_reset(self, reason: str) -> np.ndarray:
        """
        NO TELEPORT. Drone stays at current position.
        Returns new start XY = current gz_pos XY.
        Source: old drone_env.py L3771-3818.
        """
        gz_pos0 = self.bridge.get_gazebo_position()
        self.logger.debug(
            f"[RESET] Continuous episode reset reason={reason} "
            f"pos={np.round(gz_pos0, 3).tolist()}"
        )

        if (
            reason == "collision"
            and self.ecfg.collision_continuous_reset_anti_sink_enabled
        ):
            self._continuous_reset_collision_anti_sink(
                duration=self.ecfg.collision_continuous_reset_anti_sink_duration_s
            )
        else:
            self._zero_velocity_command(duration=0.2)

        gz_pos = self.bridge.get_gazebo_position()
        if gz_pos[2] > self.ecfg.alt_max + 2.0:
            self._descend_to_safe_altitude(target_z=3.0, timeout=4.5)
            gz_pos = self.bridge.get_gazebo_position()

        new_start_xy = np.array([gz_pos[0], gz_pos[1]], dtype=np.float32)
        return new_start_xy

    def multi_env_fast_recover_episode_reset(
        self, reason: str, reset_t0_mono: float, start: np.ndarray
    ) -> bool:
        """
        Fast reset for multi-env: teleport but skip full settle/EKF.
        Source: old drone_env.py L3820-3940.
        Returns True on success.
        """
        self.logger.warning(
            f"[RESET POLICY] multi_env_fast_reset env={self.env_id} "
            f"reason={reason} total_envs={self.ecfg.total_envs} "
            "action=avoid_blocking_hard_reset"
        )

        if getattr(self.bridge, "is_armed", False):
            for _ in range(4):
                self.bridge.disarm(force=True)
                self.bridge.tick(0.05)

        self._idle_spin(duration=self.ecfg.multi_env_fast_reset_idle_before_teleport_s)

        tp_wall = time.monotonic()
        teleport_ok, teleport_target = self._teleport_to_target(start)
        if not teleport_ok:
            self.logger.warning(
                "[RESET POLICY] multi_env_fast_reset failed "
                "reason=teleport_failed fallback=hard_reset_fallback"
            )
            return False

        self._wait_gz_pose_near(teleport_target, tol=0.10, stable_hits=1, timeout=0.8)
        self.bridge.prime_visual_odometry_after_reset(
            duration=0.5, zero_velocity=True, reset_counter=True
        )

        fresh_ok = self.bridge.wait_for_fresh_px4_callbacks(
            after_wall=tp_wall,
            timeout=self.ecfg.multi_env_fast_reset_fresh_timeout_s,
            max_age=0.5,
            require_status=True,
            require_local_pos=True,
            prefer_estimator_flags=True,
        )
        if not fresh_ok:
            self.logger.warning(
                "[RESET POLICY] multi_env_fast_reset stale_px4_callbacks "
                "fallback=hard_reset_fallback"
            )
            return False

        self._wait_after_teleport_settle(
            timeout=self.ecfg.multi_env_fast_reset_settle_timeout_s,
            max_speed=self.ecfg.teleport_settle_max_speed,
            max_vz=self.ecfg.teleport_settle_max_vz,
        )

        ekf_synced = self._wait_for_ekf_convergence(
            timeout=self.ecfg.multi_env_fast_reset_ekf_timeout_s,
            after_wall=tp_wall,
            stable_hits_required=3,
        )
        if not ekf_synced:
            self.logger.debug(
                "[RESET POLICY] multi_env_fast_reset "
                "ekf_not_fully_synced_but_continuing_if_arm_sanity_ok=true"
            )

        coarse_sanity_ok, info = self._wait_for_startup_arm_sanity(
            start=start, timeout=1.0
        )
        if not coarse_sanity_ok:
            self.logger.warning(
                "[RESET POLICY] multi_env_fast_reset startup_arm_sanity_not_ready "
                f"pose_near_start={info.get('pose_near_start')} "
                f"speed={info.get('speed', float('nan')):.2f} "
                f"vz_abs={info.get('vz_abs', float('nan')):.2f} "
                "fallback=hard_reset_fallback"
            )
            return False

        arm_ok = self._arm_with_startup_retry(
            start=start,
            context="[FAST RESET ARM STARTUP]",
            max_attempts=1,
            retry_idle_s=1.0,
        )
        if not arm_ok:
            self.logger.warning(
                "[RESET POLICY] multi_env_fast_reset arm_failed "
                "fallback=hard_reset_fallback"
            )
            return False

        self._lift_warmup_before_episode(duration=self.ecfg.lift_warmup_time)
        return True

    _HARD_RESET_MAX_S = 90.0  # outer wall-clock deadline; prevents one env blocking SubprocVecEnv barrier

    def hard_reset_fallback_episode_reset(
        self, reason: str, reset_t0_mono: float, start: np.ndarray
    ) -> None:
        """
        Full hard reset: disarm → teleport → EKF → arm → lift.
        6-attempt retry loop. Raises RuntimeError if all attempts fail or outer deadline exceeded.
        Source: old drone_env.py L3975-4265.
        """
        last_reason = None if reason == "startup" else reason

        reset_success = False
        arm_fail_count = 0
        max_attempts = 6
        wall_deadline = time.monotonic() + self._HARD_RESET_MAX_S

        for attempt in range(1, max_attempts + 1):
            if time.monotonic() > wall_deadline:
                err_msg = (
                    f"[ENV {self.env_id}] hard_reset deadline exceeded after {attempt - 1} attempts "
                    f"({self._HARD_RESET_MAX_S:.0f}s) reason={reason}"
                )
                self.logger.error(err_msg)
                raise RuntimeError(err_msg)
            self.logger.debug(f"[RESET] Full reset attempt {attempt}/{max_attempts}")

            if last_reason is None and attempt == 1:
                coarse_sanity_ok, info = self._wait_for_startup_arm_sanity(
                    start=start, timeout=2.0
                )
                if not coarse_sanity_ok:
                    self.logger.warning(
                        "[RESET] cold start startup arm sanity not ready "
                        f"pose_near_start={info.get('pose_near_start')} "
                        f"speed={info.get('speed', float('nan')):.2f} "
                        f"vz_abs={info.get('vz_abs', float('nan')):.2f} "
                        f"estimator_insane={info.get('estimator_insane')} "
                        f"flipped={info.get('flipped')}"
                    )
                    self._idle_spin(duration=1.0)
                    continue
                arm_ok = self._arm_with_startup_retry(
                    start=start,
                    context="[COLD START ARM STARTUP]",
                    max_attempts=2,
                    retry_idle_s=2.0,
                )
                if arm_ok:
                    self._lift_warmup_before_episode(duration=self.ecfg.lift_warmup_time)
                    reset_success = True
                    break
                self.logger.warning(
                    f"[RESET] Cold start arm_and_takeoff failed attempt={attempt}"
                )
                arm_fail_count += 1
                self._idle_spin(duration=1.0)
                continue

            self._reset_land_force_disarm()
            # Extra wait for failure_detector to clear: FD resets when drone is disarmed + stable.
            # If fd_critical_failure is active, arm will be TEMPORARILY_REJECTED until FD clears.
            fd_status = int(getattr(self.bridge, "failure_detector_status", 0))
            post_disarm_wait = 4.0 if fd_status != 0 else 2.0
            if fd_status != 0:
                self.logger.warning(
                    f"[RESET] failure_detector_status={fd_status} — extending post-disarm wait to {post_disarm_wait}s for FD to clear"
                )
            self._idle_spin(duration=post_disarm_wait)

            tp_wall = time.monotonic()
            tp_min_stamp = self.bridge.get_clock().now().nanoseconds * 1e-9
            teleport_ok, teleport_target = self._teleport_to_target(start)
            if not teleport_ok:
                self.logger.error("[RESET] Teleport failed, model may be missing")
                continue

            self._wait_gz_pose_near(teleport_target, tol=0.15, stable_hits=3, timeout=3.0, min_stamp=tp_min_stamp)
            # Gazebo set_pose does NOT reset physics velocity — drone may still tumble after teleport.
            # Wait 4s for ground contact + friction to zero out residual velocity before EKF burst.
            self._idle_spin(duration=4.0)
            self._wait_gz_pose_near(teleport_target, tol=0.10, stable_hits=1, timeout=0.6)

            self.bridge.prime_visual_odometry_after_reset(
                duration=1.0, zero_velocity=True, reset_counter=True
            )

            fresh_ok = self.bridge.wait_for_fresh_px4_callbacks(
                after_wall=tp_wall,
                timeout=3.0,
                max_age=0.5,
                require_status=True,
                require_local_pos=True,
                prefer_estimator_flags=True,
            )
            if not fresh_ok:
                self.logger.warning(
                    "[RESET] PX4 callbacks stale after teleport/VO reset; retry reset"
                )
                continue

            settled = self._wait_after_teleport_settle(
                timeout=self.ecfg.teleport_settle_timeout,
                max_speed=self.ecfg.teleport_settle_max_speed,
                max_vz=self.ecfg.teleport_settle_max_vz,
            )
            if not settled:
                if self._skip_ekf_renotify_if_estimator_unstable("[RESET][SETTLE RETRY]"):
                    arm_fail_count += 1
                    continue

                retry_wall = time.monotonic()
                self.bridge.prime_visual_odometry_after_reset(
                    duration=1.5, zero_velocity=True, reset_counter=True
                )
                retry_fresh_ok = self.bridge.wait_for_fresh_px4_callbacks(
                    after_wall=retry_wall,
                    timeout=3.0,
                    max_age=0.5,
                    require_status=True,
                    require_local_pos=True,
                    prefer_estimator_flags=True,
                )
                if not retry_fresh_ok:
                    self.logger.warning(
                        "[RESET][SETTLE RETRY] PX4 callbacks stale after re-notify"
                    )
                    continue

                ekf_retry_ok = self._wait_for_ekf_convergence(
                    timeout=9.0, after_wall=retry_wall, stable_hits_required=5
                )
                if not ekf_retry_ok:
                    self.logger.warning(
                        "[RESET][SETTLE RETRY] recovered=False action=hard_reset"
                    )
                    continue

            ekf_synced = self._wait_for_ekf_convergence(
                timeout=3.0, after_wall=tp_wall, stable_hits_required=5
            )
            if not ekf_synced:
                self.logger.warning(
                    "[RESET][EKF RETRY] first convergence failed, re-notify EV reset"
                )
                if self._skip_ekf_renotify_if_estimator_unstable("[RESET][EKF RETRY]"):
                    arm_fail_count += 1
                    if arm_fail_count >= 2:
                        self._idle_spin(duration=3.0)
                    continue

                retry_wall = time.monotonic()
                self.bridge.prime_visual_odometry_after_reset(
                    duration=1.0, zero_velocity=True, reset_counter=True
                )
                retry_fresh_ok = self.bridge.wait_for_fresh_px4_callbacks(
                    after_wall=retry_wall,
                    timeout=3.0,
                    max_age=0.5,
                    require_status=True,
                    require_local_pos=True,
                    prefer_estimator_flags=True,
                )
                if retry_fresh_ok:
                    ekf_synced = self._wait_for_ekf_convergence(
                        timeout=3.0, after_wall=retry_wall, stable_hits_required=5
                    )
                else:
                    ekf_synced = False

                if not ekf_synced:
                    coarse_sanity_ok, info = self._startup_arm_sanity(start=start)
                    if not coarse_sanity_ok:
                        self.logger.error(
                            "[RESET] EKF convergence failed after teleport"
                        )
                        arm_fail_count += 1
                        if arm_fail_count >= 2:
                            self._idle_spin(duration=3.0)
                        continue

            pos_after = self.bridge.get_gazebo_position()
            if not self._pose_near_start(pos_after, start=start):
                self.logger.warning(
                    f"[RESET] Pose not near start after teleport "
                    f"pos={np.round(pos_after, 3).tolist()}"
                )
                continue

            coarse_sanity_ok, info = self._wait_for_startup_arm_sanity(
                start=start, timeout=2.0
            )
            if not coarse_sanity_ok:
                self.logger.warning(
                    "[RESET] startup arm sanity not ready before arm "
                    f"pose_near_start={info.get('pose_near_start')} "
                    f"speed={info.get('speed', float('nan')):.2f} "
                    f"vz_abs={info.get('vz_abs', float('nan')):.2f} "
                    f"estimator_insane={info.get('estimator_insane')} "
                    f"flipped={info.get('flipped')}"
                )
                arm_fail_count += 1
                if arm_fail_count >= 2:
                    self._idle_spin(duration=2.0)
                continue

            self._idle_spin(duration=1.5)
            arm_ok = self._arm_with_startup_retry(
                start=start,
                context="[RESET ARM STARTUP]",
                max_attempts=2,
                retry_idle_s=2.0,
            )
            if not arm_ok:
                arm_fail_count += 1
                if arm_fail_count >= 5:
                    self._idle_spin(duration=2.0)
                continue

            self._lift_warmup_before_episode(duration=self.ecfg.lift_warmup_time)

            for _ in range(5):
                self.bridge.tick(0.05)

            current_pos = self.bridge.get_gazebo_position()
            if self._out_of_fence(current_pos) or self.bridge.is_flipped():
                self.logger.warning("[RESET] Invalid pose after arm/takeoff, retry")
                continue

            reset_success = True
            break

        if not reset_success:
            err_msg = (
                f"[ENV {self.env_id}] Reset failed after {max_attempts} attempts. "
                f"last_reason={last_reason}"
            )
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)

    def rescue_then_continuous_reset(self, reason: str) -> np.ndarray | None:
        """
        Velocity-drive drone back inside fence, then continuous reset.
        Returns new start XY on success, None on failure.
        Source: old drone_env.py L3946-3975.
        """
        if reason == "out_of_fence":
            if not self.ecfg.use_rescue_after_out_of_fence:
                self.logger.warning("[RESET POLICY] rescue disabled for out_of_fence")
                return None

            rescue_ok = self._rescue_to_fence_interior_by_velocity()
            if not rescue_ok:
                self.rescue_fail_count += 1
                return None

            self.rescue_success_count += 1
            return self.continuous_episode_reset(reason="rescue_success")

        if reason == "max_steps_near_fence_recentered":
            return self.continuous_episode_reset(reason=reason)

        # Near-fence downgrade from continuous (e.g. max_steps close to boundary)
        rescue_ok = self._rescue_to_fence_interior_by_velocity()
        if not rescue_ok:
            self.rescue_fail_count += 1
            return None
        self.rescue_success_count += 1
        return self.continuous_episode_reset(reason=reason)

    def pre_episode_auto_yaw_to_goal(self, goal: np.ndarray) -> None:
        """
        Rotate drone in-place toward goal before episode starts.
        Source: old drone_env.py L3710.
        """
        if not self.ecfg.pre_episode_auto_yaw_enabled:
            return

        pos = self.bridge.get_gazebo_position()
        if not np.all(np.isfinite(pos)):
            return

        goal_vec_xy = goal[:2] - np.asarray(pos[:2], dtype=np.float32)
        goal_norm = float(np.linalg.norm(goal_vec_xy))
        if goal_norm <= 1e-6:
            return

        desired_yaw = math.atan2(float(goal_vec_xy[1]), float(goal_vec_xy[0]))
        tol_rad = math.radians(float(self.ecfg.pre_episode_auto_yaw_tol_deg))
        t0 = time.monotonic()

        while time.monotonic() - t0 < float(self.ecfg.pre_episode_auto_yaw_timeout_s):
            current_yaw, _ = self.bridge.get_yaw()
            yaw_error = self._wrap_pi(desired_yaw - float(current_yaw))
            if abs(yaw_error) <= tol_rad:
                break
            yaw_rate_cmd = float(np.clip(
                self.ecfg.pre_episode_auto_yaw_gain * yaw_error,
                -self.ecfg.yaw_rate_limit - 0.7,
                self.ecfg.yaw_rate_limit + 0.7,
            ))
            self.bridge.send_velocity(0.0, 0.0, 0.0, yaw_rate_cmd)
            self.bridge.tick(0.05)

        self._zero_velocity_command(duration=0.05)

    # ------------------------------------------------------------------ #
    # Teleport / settle helpers                                          #
    # ------------------------------------------------------------------ #

    def _teleport_to_target(self, target_pos: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Double-pass teleport to clear old state.
        Source: old drone_env.py L3050.
        """
        target = np.asarray(target_pos, dtype=np.float32).copy()
        teleport_ok = False

        for _ in range(1, 4):
            teleport_ok = self.bridge.teleport_drone(target)
            if teleport_ok:
                break
            self._idle_spin(duration=0.5)

        if not teleport_ok:
            return False, target

        self._idle_spin(duration=0.4)
        teleport_ok_2 = self.bridge.teleport_drone(target)

        if teleport_ok_2:
            return True, target

        pose_confirmed = self._wait_gz_pose_near(target, tol=0.20, stable_hits=1, timeout=0.8)
        if pose_confirmed:
            self.logger.warning(
                "[RESET] teleport settle second pass failed but pose already near target; continuing"
            )
            return True, target

        return False, target

    def _wait_gz_pose_near(
        self,
        target_pos: np.ndarray,
        tol: float = 0.08,
        stable_hits: int = 3,
        timeout: float = 2.0,
        min_stamp: float = 0.0,
    ) -> bool:
        """Poll gz position until within tol for stable_hits consecutive reads. Source: old drone_env.py L2979."""
        t0 = time.time()
        hits = 0
        target = np.asarray(target_pos, dtype=np.float32)
        last_pos = None
        last_age = float("inf")

        while time.time() - t0 < timeout:
            self.bridge.tick(0.05)
            pos = self.bridge.get_gazebo_position()
            last_pos = pos
            stamp = getattr(self.bridge, "gz_pose_stamp", 0.0)

            if hasattr(self.bridge, "get_clock"):
                now_s = self.bridge.get_clock().now().nanoseconds * 1e-9
            else:
                now_s = time.time()

            last_age = now_s - float(stamp) if stamp else float("inf")
            stamp_ok = (float(stamp) >= min_stamp) if min_stamp > 0.0 else (float(stamp) > 0.0)

            ok = (
                np.all(np.isfinite(pos))
                and float(np.linalg.norm(np.asarray(pos, dtype=np.float32) - target)) <= float(tol)
                and stamp_ok
            )

            if ok:
                hits += 1
                if hits >= int(stable_hits):
                    return True
            else:
                hits = 0

        self.logger.warning(
            f"[RESET][GZ POSE STABLE TIMEOUT] target={np.round(target, 3).tolist()} "
            f"pos={np.round(last_pos, 3).tolist() if last_pos is not None else None} "
            f"age={last_age:.3f} stamp_ok={stamp_ok} min_stamp={min_stamp:.3f} hits={hits}/{stable_hits}"
        )
        return False

    def _wait_after_teleport_settle(
        self,
        timeout: float = 5.0,
        max_speed: float = 1.5,
        max_vz: float = 1.0,
    ) -> bool:
        """Poll velocity until settled after teleport. Source: old drone_env.py L3113."""
        t0 = time.time()
        best_speed = float("inf")
        best_vz = float("inf")

        while time.time() - t0 < timeout:
            self.bridge.tick(0.05)
            pos = self.bridge.get_gazebo_position()
            vel = self.bridge.get_linear_velocity()

            if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
                continue

            speed = float(np.linalg.norm(vel))
            vz_abs = abs(float(vel[2]))
            best_speed = min(best_speed, speed)
            best_vz = min(best_vz, vz_abs)

            if speed < max_speed and vz_abs < max_vz:
                self._idle_spin(duration=1.0)
                return True

        self.logger.debug(
            f"[RESET] settle after teleport not perfect "
            f"best_speed={best_speed:.2f} best_vz={best_vz:.2f}"
        )
        return True  # non-blocking: continue anyway

    def _wait_for_ekf_convergence(
        self,
        timeout: float = 5.0,
        after_wall: float | None = None,
        stable_hits_required: int = 5,
    ) -> bool:
        """Poll until PX4 EKF aligns with Gazebo. Source: old drone_env.py L3488."""
        t0 = time.monotonic()
        iterations = 0
        delta = float("inf")
        xy_err = float("inf")
        z_err = float("inf")
        stable_hits = 0
        fresh_ok = after_wall is None

        while time.monotonic() - t0 < timeout:
            self.bridge.tick(0.05)

            if after_wall is not None:
                fresh_ok = self.bridge.has_fresh_px4_callbacks_after(
                    after_wall=after_wall,
                    max_age=0.5,
                    require_status=True,
                    require_local_pos=True,
                    prefer_estimator_flags=True,
                )
                if not fresh_ok:
                    stable_hits = 0
                    iterations += 1
                    continue

            gz_pos = self.bridge.get_gazebo_position()
            px4_lpos = getattr(self.bridge, "px4_lpos", np.zeros(3, dtype=np.float32))

            gz_ned = np.array([gz_pos[1], gz_pos[0], -gz_pos[2]], dtype=np.float32)
            err = px4_lpos - gz_ned
            xy_err = float(np.linalg.norm(err[:2]))
            z_err = float(abs(err[2]))
            delta = float(np.linalg.norm(err))

            ekf_ev_pos = getattr(self.bridge, "_ekf_ev_pos", False)
            ekf_ev_hgt = getattr(self.bridge, "_ekf_ev_hgt", False)
            ekf_ev_yaw = getattr(self.bridge, "_ekf_ev_yaw", False)

            ekf_ok = xy_err < 0.30 and z_err < 0.50 and ekf_ev_pos and ekf_ev_hgt and ekf_ev_yaw
            if ekf_ok:
                stable_hits += 1
                if stable_hits >= stable_hits_required:
                    return True
            else:
                stable_hits = 0

            iterations += 1

        self.logger.warning(
            f"[EKF SYNC] Timeout. delta={delta:.3f}m xy={xy_err:.3f} z={z_err:.3f} "
            f"stable_hits={stable_hits}/{stable_hits_required}"
        )
        return False

    # ------------------------------------------------------------------ #
    # Arm / hard reset helpers                                           #
    # ------------------------------------------------------------------ #

    def _arm_with_startup_retry(
        self,
        start: np.ndarray,
        context: str,
        max_attempts: int = 2,
        retry_idle_s: float = 2.0,
    ) -> bool:
        """bridge.arm_and_takeoff() with retry. Source: old drone_env.py L7829."""
        for arm_attempt in range(1, max_attempts + 1):
            coarse_sanity_ok, info = self._startup_arm_sanity(start=start)
            if not coarse_sanity_ok:
                self.logger.warning(
                    f"{context} coarse startup sanity failed "
                    f"pose_near_start={info.get('pose_near_start')} "
                    f"speed={info.get('speed', float('nan')):.2f} "
                    f"vz_abs={info.get('vz_abs', float('nan')):.2f} "
                    f"estimator_insane={info.get('estimator_insane')} "
                    f"flipped={info.get('flipped')}"
                )
                return False

            arm_ok = self.bridge.arm_and_takeoff()
            if arm_ok:
                return True

            if arm_attempt < max_attempts:
                self.logger.warning(
                    "[ARM STARTUP RETRY] arm_and_takeoff failed; idle_then_retry"
                )
                self._idle_spin(duration=retry_idle_s)

        self.logger.error("[ARM STARTUP FAIL] arm attempts exhausted")
        return False

    def startup_arm_episode_reset(self, start: np.ndarray) -> None:
        """Lightweight cold-start: wait for gz_pose_ready, then arm and lift.
        Skips the full hard-reset loop (disarm/teleport/EKF re-notify) since
        PX4 just booted with VIO streaming already running.
        """
        t0 = time.monotonic()
        while not getattr(self.bridge, "gz_pose_ready", False):
            if time.monotonic() - t0 > 15.0:
                self.logger.warning("[STARTUP ARM] gz_pose_ready never True after 15s, proceeding anyway")
                break
            self._idle_spin(duration=0.1)

        arm_ok = self._arm_with_startup_retry(
            start=start,
            context="[STARTUP ARM]",
            max_attempts=3,
            retry_idle_s=2.0,
        )
        if not arm_ok:
            self.logger.error("[STARTUP ARM] arm_and_takeoff failed, falling back to hard reset")
            self.hard_reset_fallback_episode_reset("startup_arm_fallback", time.monotonic(), start)
            self._idle_spin(0.3)  # give pose listener time to catch up before caller reads position
            return

        self._lift_warmup_before_episode(duration=self.ecfg.lift_warmup_time)

    def _mark_bridge_state_stale_after_hard_reset(self) -> None:
        """Invalidate cached arm/offboard flags. Source: old drone_env.py L3241."""
        if hasattr(self.bridge, "is_armed"):
            self.bridge.is_armed = False
        if hasattr(self.bridge, "offboard_enabled"):
            self.bridge.offboard_enabled = False
        self.logger.debug("[RESET] Cleared cached bridge armed/offboard state")

    def _lift_warmup_before_episode(self, duration: float = 0.3) -> None:
        """Send upward velocity until z>=1.5m or timeout. Source: old drone_env.py L3181."""
        start_pos = self.bridge.get_gazebo_position()
        start_z = float(start_pos[2])
        best_z = start_z
        t0 = time.time()

        while time.time() - t0 < duration:
            pos = self.bridge.get_gazebo_position()
            z = float(pos[2])
            best_z = max(best_z, z)
            if z >= 1.5:
                break
            self.bridge.send_velocity(0.0, 0.0, self.ecfg.lift_vz, 0.0)
            self.bridge.tick(0.05)

        pos = self.bridge.get_gazebo_position()
        self.logger.debug(
            f"[LIFT WARMUP] start_z={start_z:.2f} best_z={best_z:.2f} "
            f"pos={np.round(pos, 3).tolist()}"
        )

    def _startup_arm_sanity(self, start: np.ndarray) -> tuple[bool, dict]:
        """Check sane state before arming. Source: old drone_env.py L7775."""
        pos = self.bridge.get_gazebo_position()
        vel = self.bridge.get_linear_velocity()
        pos_finite = bool(np.all(np.isfinite(pos)))
        vel_finite = bool(np.all(np.isfinite(vel)))
        pose_near_start = bool(pos_finite and self._pose_near_start(pos, start=start))
        speed = float(np.linalg.norm(vel[:2])) if vel_finite else float("inf")
        vz_abs = abs(float(vel[2])) if vel_finite else float("inf")
        settled_speed_ok = bool(
            vel_finite
            and speed < float(self.ecfg.teleport_settle_max_speed)
            and vz_abs < float(self.ecfg.teleport_settle_max_vz)
        )
        estimator_insane = bool(self._px4_estimator_insane())
        flipped = bool(self.bridge.is_flipped())
        coarse_sanity_ok = bool(
            pos_finite
            and vel_finite
            and pose_near_start
            and (not flipped)
            and (not estimator_insane)
            and (
                settled_speed_ok
                or (
                    speed < float(self.ecfg.max_px4_speed)
                    and vz_abs < float(self.ecfg.max_px4_vel_z_abs)
                )
            )
        )
        return coarse_sanity_ok, {
            "pos": pos,
            "vel": vel,
            "pose_near_start": pose_near_start,
            "speed": speed,
            "vz_abs": vz_abs,
            "settled_speed_ok": settled_speed_ok,
            "estimator_insane": estimator_insane,
            "flipped": flipped,
            "preflight_ok": bool(getattr(self.bridge, "preflight_ok", False)),
        }

    def _wait_for_startup_arm_sanity(
        self, start: np.ndarray, timeout: float = 2.0
    ) -> tuple[bool, dict]:
        """Poll _startup_arm_sanity until True or timeout. Source: old drone_env.py L7816."""
        t0 = time.monotonic()
        last_info = None
        while time.monotonic() - t0 < timeout:
            coarse_sanity_ok, info = self._startup_arm_sanity(start=start)
            last_info = info
            if coarse_sanity_ok:
                return True, info
            self.bridge.tick(0.05)
        if last_info is None:
            return self._startup_arm_sanity(start=start)
        return False, last_info

    def _skip_ekf_renotify_if_estimator_unstable(self, context: str) -> bool:
        """Log estimator state but NEVER skip re-notify — skipping when insane creates a deadlock
        (insane → skip → stays insane → arm fails → repeat until deadline)."""
        if not self._px4_estimator_insane():
            return False
        px4_lpos = getattr(self.bridge, "px4_lpos", np.zeros(3, dtype=np.float32))
        vel = self.bridge.get_linear_velocity()
        self.logger.warning(
            f"{context} estimator unstable — will still re-notify EKF "
            f"px4_lpos={np.round(px4_lpos, 3).tolist()} "
            f"vel={np.round(vel, 3).tolist()} "
            f"preflight={getattr(self.bridge, 'preflight_ok', False)}"
        )
        return False  # never skip — insane is exactly when re-notify is needed most

    # ------------------------------------------------------------------ #
    # Velocity / motion helpers                                          #
    # ------------------------------------------------------------------ #

    def _idle_spin(self, duration: float = 0.5) -> None:
        """Spin ROS callbacks only, no velocity publish. Source: old drone_env.py L2649."""
        t0 = time.time()
        while time.time() - t0 < duration:
            self.bridge.tick(0.05)

    def _zero_velocity_command(self, duration: float = 0.5) -> None:
        """Send (0,0,0,0) velocity loop. Source: old drone_env.py L2658."""
        t0 = time.time()
        while time.time() - t0 < duration:
            self.bridge.send_velocity(0.0, 0.0, 0.0, 0.0)
            self.bridge.tick(0.05)

    def _continuous_reset_collision_anti_sink(self, duration: float = 0.2) -> None:
        """Brief upward vz burst to prevent sinking after collision. Source: old drone_env.py L3751."""
        if not getattr(self.bridge, "offboard_enabled", False):
            self._zero_velocity_command(duration=duration)
            return

        t0 = time.time()
        gain = float(self.ecfg.collision_continuous_reset_anti_sink_gain)
        max_vz = max(0.0, float(self.ecfg.collision_continuous_reset_anti_sink_max_vz))

        while time.time() - t0 < duration:
            vel = self.bridge.get_linear_velocity()
            vz_now = float(vel[2]) if np.all(np.isfinite(vel)) else 0.0
            vz_cmd = 0.0
            if vz_now < 0.0:
                vz_cmd = float(np.clip((-vz_now) * gain, 0.0, max_vz))
            self.bridge.send_velocity(0.0, 0.0, vz_cmd, 0.0)
            self.bridge.tick(0.05)

    def _descend_to_safe_altitude(self, target_z: float = 1.0, timeout: float = 4.0) -> None:
        """Send vz=-2.0 until z <= target_z. Source: old drone_env.py L2900."""
        if not getattr(self.bridge, "is_armed", False):
            return
        if not getattr(self.bridge, "offboard_enabled", False):
            return

        start_pos = self.bridge.get_gazebo_position()
        if not np.all(np.isfinite(start_pos)):
            return
        if float(start_pos[2]) <= target_z:
            return

        t0 = time.time()
        while time.time() - t0 < timeout:
            pos = self.bridge.get_gazebo_position()
            if np.all(np.isfinite(pos)) and float(pos[2]) <= target_z:
                break
            self.bridge.send_velocity(0.0, 0.0, -2.0, 0.0)
            self.bridge.tick(0.05)

    def _descend_before_disarm(self) -> None:
        """Descend to reset_descend_alt before disarm. Source: old drone_env.py L2842."""
        if not getattr(self.bridge, "is_armed", False):
            return

        start_pos = self.bridge.get_gazebo_position()
        if not np.all(np.isfinite(start_pos)):
            return
        if float(start_pos[2]) <= self.ecfg.reset_descend_alt:
            return

        t0 = time.time()
        while time.time() - t0 < self.ecfg.reset_descend_timeout:
            pos = self.bridge.get_gazebo_position()
            if np.all(np.isfinite(pos)) and float(pos[2]) <= self.ecfg.reset_descend_alt:
                break
            self.bridge.send_velocity(0.0, 0.0, self.ecfg.reset_descend_vz, 0.0)
            self.bridge.tick(0.05)

    def _reset_land_force_disarm(self) -> None:
        """Descend then force disarm. Source: old drone_env.py L2941."""
        self._descend_before_disarm()
        for _ in range(10):
            self.bridge.disarm(force=True)
            self.bridge.tick(0.1)
            if hasattr(self.bridge, "is_armed") and not self.bridge.is_armed:
                break
        self._idle_spin(duration=1.0)

    # ------------------------------------------------------------------ #
    # Rescue helpers                                                      #
    # ------------------------------------------------------------------ #

    def _compute_rescue_target_xy(self, pos_xy: np.ndarray) -> np.ndarray:
        """Clamp pos inside fence - rescue_margin, then add random jitter to break loop."""
        x_min = self.ecfg.fence_x_min + self.ecfg.rescue_margin_m
        x_max = self.ecfg.fence_x_max - self.ecfg.rescue_margin_m
        y_min = self.ecfg.fence_y_min + self.ecfg.rescue_margin_m
        y_max = self.ecfg.fence_y_max - self.ecfg.rescue_margin_m
        tx = float(np.clip(float(pos_xy[0]), x_min, x_max))
        ty = float(np.clip(float(pos_xy[1]), y_min, y_max))
        jitter = float(getattr(self.ecfg, "rescue_jitter_m", 3.0))
        if jitter > 0.0:
            tx = float(np.clip(tx + np.random.uniform(-jitter, jitter), x_min, x_max))
            ty = float(np.clip(ty + np.random.uniform(-jitter, jitter), y_min, y_max))
        return np.array([tx, ty], dtype=np.float32)

    def _rescue_to_fence_interior_by_velocity(self) -> bool:
        """Velocity-drive drone back inside fence. Source: old drone_env.py L2736."""
        if not getattr(self.bridge, "is_armed", False):
            self.logger.debug("[RESCUE] skip: drone is not armed")
            return False
        if not getattr(self.bridge, "offboard_enabled", False):
            self.logger.debug("[RESCUE] skip: offboard is not enabled")
            return False
        # if getattr(self.bridge, "failsafe", False):
        #     self.logger.warning("[RESCUE] skip: PX4 in failsafe — velocity commands ignored")
        #     return False

        pos0 = self.bridge.get_gazebo_position()
        if not np.all(np.isfinite(pos0)):
            self.logger.warning("[RESCUE] invalid start pose, cannot rescue")
            return False

        target_xy = self._compute_rescue_target_xy(pos0[:2])
        target_alt = float(np.clip(self.ecfg.rescue_target_alt_m, 1.0, self.ecfg.alt_max))
        start_to_target_xy = float(np.linalg.norm(target_xy - pos0[:2]))
        expected_speed = max(
            0.1,
            float(self.ecfg.rescue_xy_speed_max) * float(self.ecfg.rescue_expected_speed_factor),
        )
        dynamic_timeout_s = (
            float(self.ecfg.rescue_timeout_base_s)
            + (start_to_target_xy / expected_speed)
            + float(self.ecfg.rescue_timeout_buffer_s)
        )
        rescue_timeout_used_s = float(
            np.clip(dynamic_timeout_s, self.ecfg.rescue_timeout_min_s, self.ecfg.rescue_timeout_max_s)
        )

        t0 = time.monotonic()
        while time.monotonic() - t0 < rescue_timeout_used_s:
            pos = self.bridge.get_gazebo_position()
            if not np.all(np.isfinite(pos)):
                self.bridge.tick(0.05)
                continue

            fence_margin = min(
                pos[0] - self.ecfg.fence_x_min,
                self.ecfg.fence_x_max - pos[0],
                pos[1] - self.ecfg.fence_y_min,
                self.ecfg.fence_y_max - pos[1],
            )
            inside_safe = (not self._out_of_fence(pos)) and (
                fence_margin >= max(0.2, self.ecfg.rescue_margin_m * 0.8)
            )

            err_xy = target_xy - np.asarray(pos[:2], dtype=np.float32)
            err_z = target_alt - float(pos[2])

            if inside_safe and float(np.linalg.norm(err_xy)) < 1.0:
                self._zero_velocity_command(duration=0.3)
                return True

            vx = float(np.clip(
                self.ecfg.rescue_xy_kp * float(err_xy[0]),
                -self.ecfg.rescue_xy_speed_max,
                self.ecfg.rescue_xy_speed_max,
            ))
            vy = float(np.clip(
                self.ecfg.rescue_xy_kp * float(err_xy[1]),
                -self.ecfg.rescue_xy_speed_max,
                self.ecfg.rescue_xy_speed_max,
            ))
            vz = float(np.clip(err_z * 0.5, -1.0, 1.0))
            self.bridge.send_velocity(vx, vy, vz, 0.0)
            self.bridge.tick(0.05)

        self.logger.warning(
            f"[RESCUE] timeout after {rescue_timeout_used_s:.1f}s"
        )
        return False

    # ------------------------------------------------------------------ #
    # Classification helpers                                             #
    # ------------------------------------------------------------------ #

    def _classify_reset_action(self, reason: str) -> str:
        """Source: old drone_env.py L3684."""
        if reason == "startup":
            return "startup_arm"
        if reason in {"goal_xy", "max_steps", "collision", "flipped", "fell_to_ground", "goal_xy_wrong_altitude", "goal_xy_near_boundary",  "px4_failsafe"}:
            return "continuous"
        if reason in {"out_of_fence", "max_steps_near_fence_recentered"}:
            return "rescue_then_continuous"
        if reason in {"ekf_callbacks_dead"}:
            return "hard_reset_fallback"
        return "hard_reset_fallback"

    def _is_continuous_reset_reason(self, reason: str) -> bool:
        return self._classify_reset_action(reason) == "continuous"

    def _should_use_multi_env_fast_reset(self, reason: str) -> bool:
        if not self.ecfg.multi_env_fast_reset_enabled:
            return False
        if int(self.ecfg.total_envs) <= 1:
            return False
        return str(reason) in set(self.ecfg.multi_env_fast_reset_reasons)

    def _is_shield_terminal_reason(self, reason: str) -> bool:
        return reason in {
            "bbox_shield_collision_prevented",
            "shield_collision_prevented",
            "shield_ground_prevented",
            "shield_out_of_fence_prevented",
            "shield_unstable_action_prevented",
        }

    # ------------------------------------------------------------------ #
    # Geometry / PX4 helpers                                            #
    # ------------------------------------------------------------------ #

    def _pose_near_start(self, pos: np.ndarray, start: np.ndarray) -> bool:
        if not np.all(np.isfinite(pos)):
            return False
        xy_err = float(np.linalg.norm(pos[:2] - start[:2]))
        z = float(pos[2])
        return xy_err < self.ecfg.start_clearance_xy and -0.25 <= z <= 0.60

    def _out_of_fence(self, pos: np.ndarray) -> bool:
        x, y = float(pos[0]), float(pos[1])
        return (
            x < self.ecfg.fence_x_min
            or x > self.ecfg.fence_x_max
            or y < self.ecfg.fence_y_min
            or y > self.ecfg.fence_y_max
        )

    def _px4_estimator_insane(self) -> bool:
        """Source: old drone_env.py L8091."""
        px4_lpos = getattr(self.bridge, "px4_lpos", np.zeros(3, dtype=np.float32))
        vel = self.bridge.get_linear_velocity()
        if not np.all(np.isfinite(px4_lpos)):
            return True
        if not np.all(np.isfinite(vel)):
            return True
        if abs(float(px4_lpos[2])) > self.ecfg.max_px4_lpos_z_abs:
            return True
        if abs(float(vel[2])) > self.ecfg.max_px4_vel_z_abs:
            return True
        if float(np.linalg.norm(vel)) > self.ecfg.max_px4_speed:
            return True
        return False

    def _wrap_pi(self, angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
