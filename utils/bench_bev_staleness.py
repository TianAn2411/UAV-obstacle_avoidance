#!/usr/bin/env python3
"""Standalone BEV staleness probe under real CPU load (PX4 + Gazebo + ROS).

Spins up ONE real drone instance via the same make_env() machinery train.py
uses, loads the latest stage checkpoint for realistic inference-driven
flight, and polls bridge._latest_bev from the OUTSIDE (no edits to
bridge_factory.py or any training file) to measure how stale the BEV
tensor is when the policy reads it via get_perception().

Read-only: never calls model.save() / model.learn() — the checkpoint used
for inference is untouched. Does not modify any existing training file.

Run this only when the real training process is stopped (Ctrl+C on
run_train.sh — triggers a normal graceful interrupted-model save, safe to
resume afterwards). Reuses rank 0's ROS domain / GZ partition / PX4 slot
by default so the measurement matches real training conditions exactly.

Usage (from obstacle_avoidance/, venv active):
    python3 -m obstacle_avoidance.utils.bench_bev_staleness --stage 2 --steps 500
"""

import argparse
import os
import threading
import time

import numpy as np
import yaml

from obstacle_avoidance.train import make_env, _ppo_load
from obstacle_avoidance.utils.checkpoint_utils import find_latest_checkpoint
from obstacle_avoidance.utils.process_utils import (
    start_microxrce_agent_single,
    stop_bridge_process,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--poll-hz", type=float, default=200.0, help="BEV identity poll rate")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", "ppo_config.yaml")
    with open(config_path) as f:
        ppo_cfg = yaml.safe_load(f)
    conf = next(c for c in ppo_cfg["curriculum"] if c["stage"] == args.stage)

    ckpt_dir = os.path.join(project_root, "ckpts", f"stage{args.stage}")
    ckpt_path = find_latest_checkpoint(ckpt_dir, f"stage{args.stage}")
    if ckpt_path is None:
        raise RuntimeError(f"No checkpoint found in {ckpt_dir}")
    print(f"[BENCH] loading checkpoint (read-only, inference only): {ckpt_path}")

    xrce_agent = start_microxrce_agent_single(port=8888)
    env = None
    try:
        env = make_env(
            rank=args.rank,
            num_pillars=conf["num_pillars"],
            curriculum_stage=args.stage,
            run_id="bev_staleness_bench",
            total_envs=1,
            stage_conf=conf,
        )()

        obs, _ = env.reset()

        # _ppo_load merges cold-start multi-group optimizer -> single group,
        # same fix train.py uses for resuming interrupted/checkpoint models.
        # env=None skips SB3's check_for_correct_spaces (the live training's own
        # VecNormalize pickle carries a STALE declared observation_space frozen
        # from before the depth_min/max scheme change in env_config.py — see
        # note below). Not needed for inference-only anyway.
        model = _ppo_load(ckpt_path, env=None, device="cpu")

        bridge = env.unwrapped._manager.bridge  # ROSBridge — read-only access, no edits
        control_dt_ms = env.unwrapped._manager.ecfg.dt * 1000.0

        update_timestamps: list = []
        stop_poll = False

        def _poller() -> None:
            last_ref = None
            poll_dt = 1.0 / args.poll_hz
            while not stop_poll:
                cur = bridge._latest_bev
                if cur is not last_ref:
                    update_timestamps.append(time.monotonic())
                    last_ref = cur
                time.sleep(poll_dt)

        poll_thread = threading.Thread(target=_poller, daemon=True)
        poll_thread.start()

        step_timestamps: list = []
        for i in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            step_timestamps.append(time.monotonic())
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            if (i + 1) % 50 == 0:
                print(f"[BENCH] step {i + 1}/{args.steps}  bev_updates_so_far={len(update_timestamps)}")

        stop_poll = True
        poll_thread.join(timeout=2.0)

        if len(update_timestamps) < 2:
            print("[BENCH] not enough BEV updates captured — check use_symbolic_extractor / sensor topics")
        else:
            gaps_ms = np.diff(np.array(update_timestamps)) * 1000.0
            print(f"\n[BENCH] BEV frame-to-frame gaps over {args.steps} control steps "
                  f"({len(update_timestamps)} frames captured):")
            print(f"  mean  : {gaps_ms.mean():.1f} ms")
            print(f"  median: {np.median(gaps_ms):.1f} ms")
            print(f"  p95   : {np.percentile(gaps_ms, 95):.1f} ms")
            print(f"  max   : {gaps_ms.max():.1f} ms")
            print(f"  control dt = {control_dt_ms:.1f} ms -> BEV updates "
                  f"{'KEEP UP with' if gaps_ms.mean() <= control_dt_ms else 'LAG BEHIND'} control loop")

            staleness_ms = []
            for t_step in step_timestamps:
                prior = [t for t in update_timestamps if t <= t_step]
                if prior:
                    staleness_ms.append((t_step - prior[-1]) * 1000.0)
            if staleness_ms:
                staleness_ms = np.array(staleness_ms)
                print(f"\n[BENCH] BEV staleness AT the moment policy reads it (age of frame used):")
                print(f"  mean  : {staleness_ms.mean():.1f} ms")
                print(f"  median: {np.median(staleness_ms):.1f} ms")
                print(f"  p95   : {np.percentile(staleness_ms, 95):.1f} ms")
                print(f"  max   : {staleness_ms.max():.1f} ms")
                print(f"  frames older than 1 control step ({control_dt_ms:.0f}ms): "
                      f"{100.0 * float((staleness_ms > control_dt_ms).mean()):.1f}%")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as e:
                print(f"[BENCH] env close error: {e}")
        try:
            stop_bridge_process(xrce_agent)
        except Exception as e:
            print(f"[BENCH] xrce agent stop error: {e}")


if __name__ == "__main__":
    main()
