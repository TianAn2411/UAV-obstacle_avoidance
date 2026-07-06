#!/usr/bin/env python3
"""
Discord training notifier — polls training_progress_stageN.csv and sends
a webhook message every ROW_INTERVAL rows (~50k steps).

Usage (standalone, no venv needed beyond stdlib + urllib):
  python3 discord_notify.py --stage 1 &

Or launched from run_train.sh:
  python3 obstacle_avoidance/discord_notify.py --stage $STAGE &
  NOTIFY_PID=$!
  ...
  kill $NOTIFY_PID 2>/dev/null
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

WEBHOOK_URL = os.environ.get(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1521010393890164811/"
    "89Zey-jgCwjViR0cR2vVTkbV8wlka7bDZFvU1IUDb4tLIZafGbE8v_OlOpMcO4_ECbdC",
)

# Send one Discord message every this many CSV rows (each row ≈ 10k steps)
ROW_INTERVAL = 2   # ~50k steps

# How often to check for new rows when waiting
POLL_INTERVAL_S = 30


def _post(payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "DiscordBot (discord_notify, 1.0)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 204):
                print(f"[discord] HTTP {resp.status}", file=sys.stderr)
    except Exception as e:
        print(f"[discord] send failed: {e}", file=sys.stderr)


def _send(content: str) -> None:
    _post({"content": content})


def _load_ppo_config(base: Path) -> dict:
    cfg_path = base / "configs" / "ppo_config.yaml"
    result = {}
    try:
        import yaml  # optional
        with open(cfg_path) as f:
            result = yaml.safe_load(f) or {}
    except Exception:
        pass
    return result


def _stage_info(ppo_cfg: dict, stage: int) -> dict:
    curriculum = ppo_cfg.get("curriculum", [])
    for item in curriculum:
        if isinstance(item, dict) and item.get("stage") == stage:
            return item
    return {}


def _load_stage_start_step(base: Path, stage: int, ppo_cfg: dict) -> int:
    is_symbolic = str(ppo_cfg.get("policy_mode", "raw")) == "symbolic"
    path = base / "ckpts" / f"stage{stage}" / ("symbolics" if is_symbolic else "raws") / f"stage{stage}_start_step.json"
    try:
        with open(path) as f:
            return int(json.load(f).get("step", 0))
    except Exception:
        return 0


def _read_zip_step(zip_path: Path) -> int:
    """Read num_timesteps from SB3 zip without loading weights (data entry only)."""
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            if "data" in z.namelist():
                data = json.loads(z.read("data"))
                return int(data.get("num_timesteps", 0))
    except Exception:
        pass
    return 0


def _get_actual_step(base: Path, stage: int, ppo_cfg: dict) -> int:
    """Best-effort: read actual model num_timesteps from zip (interrupted > latest ckpt).

    Mirrors train.py's raw/symbolic namespace split (model_prefix, ckpt_dir,
    interrupt/{raws,symbolics}/) -- must match exactly or this silently reads
    the wrong mode's (or a stale/nonexistent) checkpoint.
    """
    is_symbolic = str(ppo_cfg.get("policy_mode", "raw")) == "symbolic"
    model_prefix = f"ppo_drone_stage{stage}" if not is_symbolic else f"ppo_drone_symbolic_stage{stage}"
    mode_subdir = "symbolics" if is_symbolic else "raws"

    candidates = [
        base / "interrupt" / mode_subdir / f"{model_prefix}_interrupted.zip",
    ]
    # Also check latest checkpoint in ckpts/stage{N}/{raws,symbolics}/
    ckpt_dir = base / "ckpts" / f"stage{stage}" / mode_subdir
    if ckpt_dir.exists():
        zips = sorted(ckpt_dir.glob(f"stage{stage}_*_steps.zip"))
        if zips:
            candidates.append(zips[-1])
    for p in candidates:
        if p.exists():
            step = _read_zip_step(p)
            if step > 0:
                return step
    return 0


def _fmt_start(stage: int, stage_cfg: dict, ppo_cfg: dict, prev_window: list,
               stage_start_step: int = 0, actual_step: int = 0) -> str:
    pillars   = stage_cfg.get("num_pillars", "?")
    min_steps = stage_cfg.get("min_steps", "?")
    lr        = ppo_cfg.get("learning_rate", "?")
    n_steps   = ppo_cfg.get("n_steps", "?")
    batch     = ppo_cfg.get("batch_size", "?")
    freeze_vz = stage_cfg.get("freeze_vz", "?")
    freeze_cn = stage_cfg.get("freeze_cnn", "?")
    mode      = str(ppo_cfg.get("policy_mode", "raw"))
    cbf_on    = bool(ppo_cfg.get("cbf_enabled", False))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    min_steps_fmt = f"{min_steps:,}" if isinstance(min_steps, int) else str(min_steps)
    if isinstance(min_steps, int) and min_steps > 0:
        steps_in_stage = max(0, actual_step - stage_start_step)
        remaining = max(0, min_steps - steps_in_stage)
        step_fmt = f"{actual_step:,}  (còn {remaining:,})" if actual_step > 0 else f"0  (còn {min_steps_fmt})"
    else:
        step_fmt = f"{actual_step:,}" if actual_step > 0 else "0"
    lines = [
        f"🚀 **Training started — Stage {stage} [{mode.upper()}{' +CBF' if mode == 'symbolic' and cbf_on else ''}]**",
        f"```",
        f"time       : {ts}",
        f"mode       : {mode}{' (cbf_enabled)' if mode == 'symbolic' and cbf_on else ''}",
        f"step       : {step_fmt}",
        f"pillars    : {pillars}",
        f"min_steps  : {min_steps_fmt}",
        f"lr         : {lr}",
        f"n_steps    : {n_steps}",
        f"batch_size : {batch}",
        f"freeze_vz  : {freeze_vz}",
        f"freeze_cnn : {freeze_cn}",
        f"```",
    ]

    if prev_window:
        last = prev_window[-1]
        n    = len(prev_window)
        step = int(float(last.get("train_step", 0)))
        min_s_v = stage_cfg.get("min_steps", 0)
        min_s_v = min_s_v if isinstance(min_s_v, int) else 0
        if min_s_v > 0:
            pct = min(100.0, 100.0 * max(0, step - stage_start_step) / min_s_v)
        else:
            pct = float(last.get("progress_pct", 0))
        lines += [
            f"📋 **Tổng kết {n} checkpoint trước — {pct:.1f}%** `[{_bar(pct)}]`",
            f"```",
            f"step       : {step:,}",
            f"{'metric':<12} {'latest':>8}  {'avg(×' + str(n) + ')':>8}",
            f"{'-'*32}",
            f"{'rew_mean':<12} {float(last.get('ep_rew_mean',0)):>8.2f}  {_avg(prev_window,'ep_rew_mean'):>8.2f}",
            f"{'ep_len':<12} {float(last.get('ep_len_mean',0)):>8.1f}  {_avg(prev_window,'ep_len_mean'):>8.1f}",
            f"{'success':<12} {float(last.get('success_rate',0)):>7.1f}%  {_avg(prev_window,'success_rate'):>7.1f}%",
            f"{'collision':<12} {float(last.get('collision_rate',0)):>7.1f}%  {_avg(prev_window,'collision_rate'):>7.1f}%",
            f"{'out_fence':<12} {float(last.get('out_fence_rate',0)):>7.1f}%  {_avg(prev_window,'out_fence_rate'):>7.1f}%",
            f"```",
        ]

    return "\n".join(lines)


def _avg(rows: list, key: str) -> float:
    vals = [float(r.get(key, 0)) for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else 0.0


def _bar(pct: float, width: int = 20) -> str:
    filled = int(max(0.0, min(100.0, pct)) / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_progress(row: dict, window: list, stage: int,
                  min_steps: int = 0, stage_start_step: int = 0,
                  actual_step: int = 0, mode: str = "raw") -> str:
    """window = last N rows (including row) for rolling averages."""
    csv_step = int(float(row.get("train_step", 0)))
    step     = actual_step if actual_step > csv_step else csv_step
    episodes = int(float(row.get("episodes", 0)))
    pillars  = row.get("num_pillars", "0")
    n        = len(window)

    if min_steps > 0:
        steps_in_stage = max(0, step - stage_start_step)
        pct       = min(100.0, 100.0 * steps_in_stage / min_steps)
        remaining = max(0, min_steps - steps_in_stage)
    else:
        pct       = float(row.get("progress_pct", 0))
        remaining = 0

    # Latest (current row) — CSV stores rates already as % (0–100)
    rew_now  = float(row.get("ep_rew_mean", 0))
    len_now  = float(row.get("ep_len_mean", 0))
    succ_now = float(row.get("success_rate", 0))
    coll_now = float(row.get("collision_rate", 0))
    fence_now= float(row.get("out_fence_rate", 0))

    # Rolling average
    rew_avg  = _avg(window, "ep_rew_mean")
    len_avg  = _avg(window, "ep_len_mean")
    succ_avg = _avg(window, "success_rate")
    coll_avg = _avg(window, "collision_rate")
    fence_avg= _avg(window, "out_fence_rate")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    remaining_fmt = f"{remaining:,}" if remaining > 0 else "done"
    lines = [
        f"📊 **Stage {stage} [{mode.upper()}] — {pct:.1f}%** `[{_bar(pct)}]`",
        f"```",
        f"time       : {ts}",
        f"step       : {step:,}  (còn {remaining_fmt})",
        f"episodes   : {episodes:,}",
        f"pillars    : {pillars}",
        f"",
        f"{'metric':<12} {'latest':>8}  {'avg(×' + str(n) + ')':>8}",
        f"{'-'*32}",
        f"{'rew_mean':<12} {rew_now:>8.2f}  {rew_avg:>8.2f}",
        f"{'ep_len':<12} {len_now:>8.1f}  {len_avg:>8.1f}",
        f"{'success':<12} {succ_now:>7.1f}%  {succ_avg:>7.1f}%",
        f"{'collision':<12} {coll_now:>7.1f}%  {coll_avg:>7.1f}%",
        f"{'out_fence':<12} {fence_now:>7.1f}%  {fence_avg:>7.1f}%",
        f"```",
    ]
    return "\n".join(lines)


def watch(stage: int, base: Path) -> None:
    # Load PPO config + stage start step
    ppo_cfg          = _load_ppo_config(base)
    is_symbolic      = str(ppo_cfg.get("policy_mode", "raw")) == "symbolic"
    # Must match envs/monitor.py's TrainingMonitor.progress_csv_path split —
    # raw/symbolic runs each write their own runs/{raws,symbolics}/ CSV so
    # this poll loop reads the right one instead of waiting on a stale flat path.
    csv_path = base / "runs" / ("symbolics" if is_symbolic else "raws") / f"training_progress_stage{stage}.csv"
    stage_cfg        = _stage_info(ppo_cfg, stage)
    min_s            = stage_cfg.get("min_steps", 0)
    min_s            = min_s if isinstance(min_s, int) else 0
    stage_start_step = _load_stage_start_step(base, stage, ppo_cfg)

    print(f"[discord] watching {csv_path}")
    print(f"[discord] notify every {ROW_INTERVAL} rows (~{ROW_INTERVAL * 10}k steps)")

    # Wait for CSV to appear, then send start message immediately
    while not csv_path.exists():
        print(f"[discord] waiting for {csv_path} ...")
        time.sleep(POLL_INTERVAL_S)

    # CSV appeared — send start notification right away (no data rows needed)
    _start_actual_step = _get_actual_step(base, stage, ppo_cfg)
    _send(_fmt_start(stage, stage_cfg, ppo_cfg, [], stage_start_step=stage_start_step,
                     actual_step=_start_actual_step))
    print(f"[discord] start message sent (actual_step={_start_actual_step})")

    rows_seen    = 0
    last_notified_row = -1   # index of last row we sent a notification for

    while True:
        try:
            with open(csv_path, newline="") as f:
                reader = list(csv.DictReader(f))
        except Exception as e:
            print(f"[discord] read error: {e}", file=sys.stderr)
            time.sleep(POLL_INTERVAL_S)
            continue

        current_rows = len(reader)
        if current_rows > rows_seen:
            rows_seen = current_rows

        # Determine which rows warrant a notification
        # Notify at row indices: ROW_INTERVAL-1, 2*ROW_INTERVAL-1, ...
        # i.e. every time we cross a multiple of ROW_INTERVAL
        latest_notify_idx = (rows_seen // ROW_INTERVAL) * ROW_INTERVAL - 1
        if rows_seen > 0 and latest_notify_idx > last_notified_row and latest_notify_idx < rows_seen:
            row      = reader[-1]
            window   = reader[-ROW_INTERVAL:]
            actual_step = _get_actual_step(base, stage, ppo_cfg)
            _send(_fmt_progress(row, window, stage, min_steps=min_s,
                                stage_start_step=stage_start_step,
                                actual_step=actual_step,
                                mode=str(ppo_cfg.get("policy_mode", "raw"))))
            last_notified_row = latest_notify_idx
            print(f"[discord] notified at row {latest_notify_idx} "
                  f"(csv_step={row.get('train_step','?')} actual={actual_step or 'n/a'})")

        time.sleep(POLL_INTERVAL_S)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument(
        "--base",
        default=str(Path(__file__).resolve().parent),
        help="Path to obstacle_avoidance/ directory",
    )
    args = parser.parse_args()
    watch(args.stage, Path(args.base))


if __name__ == "__main__":
    main()
