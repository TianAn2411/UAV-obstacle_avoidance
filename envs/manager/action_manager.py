from dataclasses import dataclass
import numpy as np
from obstacle_avoidance.configs.env_config import EnvConfig


@dataclass
class ActionOutput:
    vx: float
    vy: float
    vz: float
    yaw_rate: float


class ActionManager:
    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: EnvConfig) -> None:
        self._cfg = cfg
        self._cmd_vel = np.zeros(4, dtype=np.float32)  # smoothed command state

    def reset(self) -> None:
        self._cmd_vel = np.zeros(4, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Main interface                                                      #
    # ------------------------------------------------------------------ #

    def process(
        self,
        raw_action: np.ndarray,      # shape (4,) from PPO, range [-1, 1]
        step_count: int,
        altitude: float,
        is_takeoff_phase: bool,
        num_pillars: int = 0,
    ) -> ActionOutput:
        cfg = self._cfg
        raw = np.asarray(raw_action, dtype=np.float32)

        # --- Velocity limits (pillar mode caps XY speed) ---
        max_vx = min(cfg.vx_limit, 1.4) if num_pillars > 0 else cfg.vx_limit
        max_vy = min(cfg.vy_limit, 1.2) if num_pillars > 0 else cfg.vy_limit

        # --- Scale raw [-1,1] → velocity space ---
        target_vx = float(raw[0]) * max_vx
        target_vy = float(raw[1]) * max_vy
        # action[2] positive = fly up (ENU); bridge converts to PX4 NED internally
        if cfg.freeze_vz:
            target_vz = 0.0
            self._cmd_vel[2] = 0.0  # flush EMA buffer — prevents spike when vz unlocks
        else:
            target_vz = float(raw[2]) * (cfg.vz_up_limit if raw[2] >= 0.0 else cfg.vz_down_limit)
        target_yr = float(raw[3]) * cfg.yaw_rate_limit

        target_cmd = np.array([target_vx, target_vy, target_vz, target_yr], dtype=np.float32)

        # --- EMA smoothing: blend toward new target ---
        # cmd_vel = (1 - α) * prev + α * target  (α = action_smoothing = 0.35)
        α = cfg.action_smoothing
        self._cmd_vel = (1.0 - α) * self._cmd_vel + α * target_cmd

        # --- Yaw deadband: suppress tiny residual spin ---
        if abs(float(self._cmd_vel[3])) < 0.05:
            self._cmd_vel[3] = 0.0

        # --- Clip all channels to hard limits ---
        self._cmd_vel[0] = float(np.clip(self._cmd_vel[0], -max_vx, max_vx))
        self._cmd_vel[1] = float(np.clip(self._cmd_vel[1], -max_vy, max_vy))
        self._cmd_vel[2] = float(np.clip(self._cmd_vel[2], -cfg.vz_down_limit, cfg.vz_up_limit))
        self._cmd_vel[3] = float(np.clip(self._cmd_vel[3], -cfg.yaw_rate_limit, cfg.yaw_rate_limit))

        vx = float(self._cmd_vel[0])
        vy = float(self._cmd_vel[1])
        vz = float(self._cmd_vel[2])
        yr = float(self._cmd_vel[3])

        # --- Takeoff assist: lock XY/yaw and command climb until airborne ---
        # Applied after smoothing; overrides cmd but does NOT update _cmd_vel state
        # so smoothing resumes correctly once airborne.
        if is_takeoff_phase and step_count < cfg.takeoff_assist_steps and altitude < cfg.takeoff_assist_alt:
            vx = 0.0
            vy = 0.0
            yr = 0.0
            if altitude < 0.25:
                vz = cfg.takeoff_assist_vz * 0.7
            elif altitude < 0.55:
                vz = cfg.takeoff_assist_vz * 0.5
            else:
                vz = cfg.takeoff_assist_vz * 0.2

        return ActionOutput(vx=vx, vy=vy, vz=vz, yaw_rate=yr)
