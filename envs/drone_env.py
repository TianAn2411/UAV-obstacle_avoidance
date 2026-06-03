import gymnasium as gym
from gymnasium import spaces
import numpy as np

from obstacle_avoidance.configs.env_config import EnvConfig
from obstacle_avoidance.configs.pillar_config import PillarConfig
from obstacle_avoidance.configs.reward_config import RewardConfig
from obstacle_avoidance.envs.manager.train_manager import TrainManager


class DroneObstacleEnv(gym.Env):
    def __init__(
        self,
        bridge,
        spawner,
        ecfg: EnvConfig = None,
        rcfg: RewardConfig = None,
        pcfg: PillarConfig = None,
        env_id: int = 0,
        log_dir: str | None = None,
    ):
        super().__init__()
        ecfg = ecfg or EnvConfig()
        rcfg = rcfg or RewardConfig()
        pcfg = pcfg or PillarConfig()

        self.observation_space = spaces.Dict({
            "depth": spaces.Box(
                ecfg.depth_min, ecfg.depth_max,
                shape=ecfg.depth_shape, dtype=np.float32,
            ),
            "state": spaces.Box(-np.inf, np.inf, shape=(ecfg.state_dim,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ecfg.action_dim,), dtype=np.float32)

        self._manager = TrainManager(bridge, spawner, ecfg, rcfg, pcfg, env_id=env_id, log_dir=log_dir)

    def reset(self, seed=None, options=None):
        return self._manager.reset()

    def step(self, action):
        result = self._manager.step_process(action)
        return result.obs, result.reward, result.terminated, result.truncated, result.info

    def close(self):
        self._manager.close()
