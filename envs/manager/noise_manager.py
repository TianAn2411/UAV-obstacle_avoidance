"""Centralised domain randomisation — obs noise, action delay/noise, mass scale, wind."""
from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Tuple

import numpy as np

from obstacle_avoidance.configs.env_config import EnvConfig

if TYPE_CHECKING:
    from obstacle_avoidance.utils.bridge_factory import ROSBridge


class NoiseManager:
    """
    All sim-to-real noise in one place:
      - Observation noise   (pos / vel / yaw / ang_vel / quat Gaussian, per-step)
      - Action delay buffer (latency simulation, fixed size from ecfg)
      - Action noise        (ESC jitter, per-step, unobservable)
      - Mass scale          (per-episode ±N% actuation effectiveness)
      - Wind                (per-episode random horizontal wind via Gazebo Transport)

    Usage:
        noise_mgr = NoiseManager(ecfg, np.random.default_rng())

        # --- at episode start ---
        noise_mgr.reset_episode(bridge)         # samples mass/wind, clears delay buf

        # --- in _build_state_vector ---
        vel, ang_vel, pos, yaw, quat = noise_mgr.apply_obs_noise(vel, ang_vel, pos, yaw, quat)

        # --- in step_process ---
        delayed_cmd, sent_cmd = noise_mgr.step_action(computed_cmd)
        # state[14:18] ← delayed_cmd  (pre-noise, observable on real hw)
        # bridge.send_velocity(*sent_cmd)
    """

    def __init__(self, ecfg: EnvConfig, rng: np.random.Generator) -> None:
        self.ecfg = ecfg
        self._rng = rng

        self._current_delay: int = max(0, int(ecfg.action_delay_steps))
        self._delay_buf: deque = deque(
            [np.zeros(4, dtype=np.float32)] * (self._current_delay + 1),
            maxlen=self._current_delay + 1,
        )
        self._mass_scale: float = 1.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mass_scale(self) -> float:
        """Per-episode actuation scale sampled in reset_episode()."""
        return self._mass_scale

    # ------------------------------------------------------------------
    # Episode boundary
    # ------------------------------------------------------------------

    def reset_episode(self, bridge: "ROSBridge | None" = None) -> None:
        """Sample per-episode randomisation. Call once at the start of every episode."""
        # Sample delay for this episode
        min_d = max(0, int(self.ecfg.action_delay_steps))
        max_d = max(min_d, int(self.ecfg.action_delay_steps_max))
        self._current_delay = int(self._rng.integers(min_d, max_d + 1)) if max_d > min_d else min_d

        # Recreate delay buffer with sampled size (cheap at episode boundary)
        self._delay_buf = deque(
            [np.zeros(4, dtype=np.float32)] * (self._current_delay + 1),
            maxlen=self._current_delay + 1,
        )

        # Mass scale: uniform in [mass_scale_min, mass_scale_max]
        lo, hi = float(self.ecfg.mass_scale_min), float(self.ecfg.mass_scale_max)
        self._mass_scale = float(self._rng.uniform(lo, hi)) if hi > lo else 1.0

        # Wind: random horizontal direction, speed uniform in [0, wind_speed_max]
        if self.ecfg.wind_speed_max > 0.0 and bridge is not None:
            speed = float(self._rng.uniform(0.0, self.ecfg.wind_speed_max))
            angle = float(self._rng.uniform(0.0, 2.0 * math.pi))
            wx = speed * math.cos(angle)
            wy = speed * math.sin(angle)
            ok = bridge.set_wind(wx, wy, 0.0)
            if not ok:
                import logging
                logging.getLogger(__name__).debug(
                    "[NOISE] set_wind skipped (Gazebo transport unavailable)"
                )

    # ------------------------------------------------------------------
    # Per-step: observation noise
    # ------------------------------------------------------------------

    def apply_obs_noise(
        self,
        vel: np.ndarray,
        ang_vel: np.ndarray,
        pos: np.ndarray,
        yaw: float,
        quat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Inject per-step Gaussian noise into policy observations.
        GT values (used for rewards / collision / fence) are NOT touched here —
        callers must pass already-separated noisy copies.
        Returns noisy (vel, ang_vel, pos, yaw, quat).
        """
        e = self.ecfg
        if e.obs_noise_vel_std > 0.0:
            vel = vel + self._rng.standard_normal(3).astype(np.float32) * e.obs_noise_vel_std
        if e.obs_noise_ang_vel_std > 0.0:
            ang_vel = ang_vel + self._rng.standard_normal(3).astype(np.float32) * e.obs_noise_ang_vel_std
        if e.obs_noise_pos_std > 0.0:
            pos = pos + self._rng.standard_normal(3).astype(np.float32) * e.obs_noise_pos_std
        if e.obs_noise_yaw_std > 0.0:
            yaw = float(yaw) + float(self._rng.standard_normal()) * e.obs_noise_yaw_std
        if e.obs_noise_quat_std > 0.0:
            quat = quat + self._rng.standard_normal(4).astype(np.float32) * e.obs_noise_quat_std
            quat /= np.linalg.norm(quat)
        return vel, ang_vel, pos, yaw, quat

    # ------------------------------------------------------------------
    # Per-step: action pipeline
    # ------------------------------------------------------------------

    def step_action(
        self, computed_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full action pipeline for one step:
          1. Apply mass_scale  — per-episode actuation effectiveness
          2. Push into delay buffer
          3. Pop delayed_cmd   — pre-noise; goes into state vector [14:18]
          4. Add ESC noise     — unobservable on real hw; not stored in state

        Returns (delayed_cmd, sent_cmd).
        """
        e = self.ecfg

        # 1. Mass scale (per-episode, constant within episode)
        scaled = computed_cmd * self._mass_scale
        scaled[0] = float(np.clip(scaled[0], -e.vx_limit,       e.vx_limit))
        scaled[1] = float(np.clip(scaled[1], -e.vy_limit,       e.vy_limit))
        scaled[2] = float(np.clip(scaled[2], -e.vz_down_limit,  e.vz_up_limit))
        scaled[3] = float(np.clip(scaled[3], -e.yaw_rate_limit, e.yaw_rate_limit))

        # 2+3. Delay buffer
        self._delay_buf.append(scaled)
        delayed_cmd = self._delay_buf[0].copy()

        # 4. ESC noise (unobservable — NOT stored in state vector)
        sent_cmd = delayed_cmd.copy()
        if e.action_noise_std > 0.0:
            sent_cmd += self._rng.standard_normal(4).astype(np.float32) * e.action_noise_std
            sent_cmd[0] = float(np.clip(sent_cmd[0], -e.vx_limit,       e.vx_limit))
            sent_cmd[1] = float(np.clip(sent_cmd[1], -e.vy_limit,       e.vy_limit))
            sent_cmd[2] = float(np.clip(sent_cmd[2], -e.vz_down_limit,  e.vz_up_limit))
            sent_cmd[3] = float(np.clip(sent_cmd[3], -e.yaw_rate_limit, e.yaw_rate_limit))

        return delayed_cmd, sent_cmd
