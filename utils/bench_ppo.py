#!/usr/bin/env python3
"""Benchmark PPO inference latency. Run from obstacle_avoidance/ dir."""
import sys, time, os, glob
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv


def _is_symbolic() -> bool:
    """Mirrors train.py's policy_mode toggle -- must match or this silently
    finds a stale/wrong-mode checkpoint (or none at all)."""
    try:
        import yaml
        with open("configs/ppo_config.yaml") as f:
            return str((yaml.safe_load(f) or {}).get("policy_mode", "raw")) == "symbolic"
    except Exception:
        return False


def find_model(stage: int) -> str:
    is_symbolic = _is_symbolic()
    model_prefix = f"ppo_drone_stage{stage}" if not is_symbolic else f"ppo_drone_symbolic_stage{stage}"
    mode_subdir = "symbolics" if is_symbolic else "raws"

    candidates = [
        f"interrupt/{mode_subdir}/{model_prefix}_interrupted.zip",
        f"{model_prefix}_final.zip",
    ]
    ckpts = sorted(glob.glob(f"ckpts/stage{stage}/{mode_subdir}/stage{stage}_*_steps.zip"))
    if ckpts:
        candidates.insert(1, ckpts[-1])
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for stage {stage} (policy_mode={'symbolic' if is_symbolic else 'raw'})")


stage = int(sys.argv[1]) if len(sys.argv) > 1 else 2
path = find_model(stage)
print(f"Model: {path}")

# Load metadata + policy weights directly — bypasses optimizer group mismatch
data, params, _ = load_from_zip_file(path, device="cpu", verbose=0)

obs_space = data["observation_space"]
act_space = data["action_space"]

class _DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = obs_space
        self.action_space = act_space
    def reset(self, **kw):
        return {k: np.zeros(v.shape, v.dtype) for k, v in obs_space.spaces.items()}, {}
    def step(self, a):
        return self.reset()[0], 0.0, False, False, {}

model = PPO(
    policy=data["policy_class"],
    env=DummyVecEnv([_DummyEnv]),
    policy_kwargs=data.get("policy_kwargs", {}),
    device="cpu",
)
model.policy.load_state_dict(params["policy"])
model.policy.eval()

n_params = sum(p.numel() for p in model.policy.parameters())
print(f"Params: {n_params:,} ({n_params/1e6:.2f}M)")

obs = {
    "depth": np.random.rand(1, 3, 84, 84).astype(np.float32),
    "state": np.random.rand(1, 31).astype(np.float32),
}

N = 500

def bench(device_name: str):
    m = model if device_name == "cpu" else PPO(
        policy=data["policy_class"],
        env=DummyVecEnv([_DummyEnv]),
        policy_kwargs=data.get("policy_kwargs", {}),
        device=device_name,
    )
    if device_name != "cpu":
        m.policy.load_state_dict(params["policy"])
    m.policy.eval()

    # warmup
    for _ in range(30):
        m.policy.predict(obs, deterministic=True)
    if device_name == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        m.policy.predict(obs, deterministic=True)
        if device_name == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times)
    print(f"\nPPO inference (N={N}, {device_name.upper()}):")
    print(f"  mean  : {t.mean():.3f} ms")
    print(f"  median: {np.median(t):.3f} ms")
    print(f"  p95   : {np.percentile(t, 95):.3f} ms")
    print(f"  p99   : {np.percentile(t, 99):.3f} ms")
    print(f"  max   : {t.max():.3f} ms")
    print(f"  → Max control freq: {1000/t.mean():.0f} Hz")

bench("cpu")
if torch.cuda.is_available():
    bench("cuda")
