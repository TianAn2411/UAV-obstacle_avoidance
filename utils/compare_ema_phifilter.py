#!/usr/bin/env python3
"""Offline comparison: current EMA kinematic stack vs. phifilter.PhiFilter.

Standalone — no ROS/Gazebo/PX4 needed. Feeds a synthetic distance-to-obstacle
signal (the same kind of scalar the HALO pipeline's M_t channel produces) through:

  1. "ema"       - the exact recursion used today in symbolic_extractor/mapper.py
                   HALO._build_bev(): m_ema = alpha*z + (1-alpha)*prev,
                   dv/da = finite-difference of m_ema (alpha = bev_ema_alpha = 0.618).
  2. "phifilter" - PhiFilter() default (no lead).
  3. "phifilter_lead" - PhiFilter(lead=True) (negative-group-delay output).

Synthetic signal mimics realistic BEV M_t behavior at ~25 Hz (bev_out rate
observed in the real pipeline):
  - hover_noisy:   flat distance + small voxel-quantization jitter (obstacle far)
  - approach_ramp: linear closing distance (drone flying at constant obstacle
                   closing speed) -> tests LAG
  - occlusion_step: sudden distance drop (pillar swings into view around a
                   corner / occlusion clears) -> tests OVERSHOOT
  - near_miss_osc: sinusoidal distance (yaw sweep near an obstacle) -> tests
                   tracking quality during real "motion" (not just noise)
  - stale_glitch:  occasional big one-frame outlier, modeling the 50-700ms
                   BEV staleness spikes measured on this box (bench_bev_staleness.py)
                   showing up as a stale/wrong distance value for one frame.

Ground truth (the noise-free piecewise signal) is known analytically, so we
can score RMSE / lag / overshoot / noise-reduction / velocity-RMSE directly,
instead of eyeballing it.

Usage:
    source ~/drone_rl_env/bin/activate
    python3 -m obstacle_avoidance.utils.compare_ema_phifilter
    python3 -m obstacle_avoidance.utils.compare_ema_phifilter --no-plot
"""
from __future__ import annotations

import argparse
import os

import numpy as np

try:
    from phifilter import PhiFilter
except ImportError as exc:
    raise SystemExit(
        "phifilter not importable — activate ~/drone_rl_env "
        "(pip show phifilter should list it)."
    ) from exc

# Matches symbolic_extractor/configs/extractor_config.py:141
BEV_EMA_ALPHA = 0.618
DT = 1.0 / 25.0  # seconds/sample, matches observed bev_out rate


# ── Synthetic ground-truth signal ───────────────────────────────────────────

def make_signal(seed: int = 0):
    """Return (truth, noisy, true_velocity, segment_bounds) all shape (N,)."""
    rng = np.random.default_rng(seed)
    segs = []

    # 1. hover_noisy: flat at 8m, 100 samples (~4s)
    n1 = 100
    segs.append(np.full(n1, 8.0))

    # 2. approach_ramp: 8m -> 2m over 100 samples, constant closing speed
    n2 = 100
    segs.append(np.linspace(8.0, 2.0, n2))

    # 3. occlusion_step: instant drop 2m -> 0.8m (new pillar revealed), hold
    n3 = 60
    segs.append(np.full(n3, 0.8))

    # 4. near_miss_osc: oscillate 0.8m +/- 0.5m (yaw sweep near obstacle)
    n4 = 150
    t = np.arange(n4) * DT
    segs.append(0.8 + 0.5 * np.sin(2 * np.pi * 0.5 * t))

    truth = np.concatenate(segs)
    bounds = np.cumsum([0, n1, n2, n3, n4])

    # True velocity: analytic derivative of the piecewise truth (m/s)
    true_vel = np.zeros_like(truth)
    true_vel[bounds[1]:bounds[2]] = (segs[1][-1] - segs[1][0]) / ((n2 - 1) * DT)
    t4 = np.arange(n4) * DT
    true_vel[bounds[3]:bounds[4]] = 0.5 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t4)

    # Sensor/pipeline noise: voxel-quantization jitter + occasional stale-frame glitch
    noise = rng.normal(0.0, 0.05, size=truth.shape)  # ~1 voxel (0.2m grid, sub-voxel jitter)
    noisy = truth + noise

    # Stale-glitch: ~1 in 40 frames a whole-different (older) distance value leaks
    # through, modeling the 50-700ms BEV staleness spikes measured on this box.
    glitch_idx = rng.choice(len(noisy), size=len(noisy) // 40, replace=False)
    glitch_lag = rng.integers(3, 15, size=glitch_idx.shape)  # "frames ago" value
    for i, lag in zip(glitch_idx, glitch_lag):
        src = max(0, i - lag)
        noisy[i] = truth[src] + rng.normal(0.0, 0.05)

    return truth, noisy, true_vel, bounds


# ── Harder / adversarial signal ─────────────────────────────────────────────

def default_drone_max_speed() -> float:
    """Read the REAL configured max drone speed from EnvConfig (not hardcoded).

    Combined horizontal speed bound sqrt(vx_limit^2 + vy_limit^2). Stages can
    override vx_limit/vy_limit via stage_conf (see train.py _ecfg_fields loop),
    so this is read live from the dataclass default, not a hardcoded number.
    """
    from obstacle_avoidance.configs.env_config import EnvConfig
    c = EnvConfig()
    return float(np.hypot(c.vx_limit, c.vy_limit))


def make_hard_signal(seed: int = 0, drone_max_speed: float | None = None,
                      obstacle_max_speed: float = 0.0):
    """Harder test battery, on top of the base 4-segment signal.

    drone_max_speed: real configured max drone horizontal speed (m/s). If None,
        read live from EnvConfig (not hardcoded -- stages can scale vx/vy_limit
        via stage_conf, so this must track the actual config, not a constant).
    obstacle_max_speed: max speed of the OBSTACLE itself (m/s). 0.0 = static
        pillar (current stages). >0 models a future moving-obstacle stage --
        the real closing rate is then drone_max_speed + obstacle_max_speed,
        NOT bounded by drone speed alone.

    Segments:
    - burst_glitch: 2-3 CONSECUTIVE stale samples (not just isolated single
      frames) -- stress-tests whether the rolling median/MAD used for the
      Hampel gate gets poisoned when several bad samples sit in the window.
    - fast_real_approach: a REAL, abrupt, sustained close event at the actual
      physically-possible closing speed (drone_max_speed + obstacle_max_speed),
      that then HOLDS at the close value (does not revert). Tests whether the
      filter distinguishes a genuine emergency from a glitch of similar shape.
    - moving_obstacle_only: drone treated as ~stationary; the OBSTACLE alone
      closes at obstacle_max_speed. Only meaningful when obstacle_max_speed>0
      -- isolates obstacle-only motion from drone-motion-driven approach, so a
      drone-speed-only physical bound would NOT catch this case.
    - held_stale: resamples the already-noisy signal by repeating the last
      computed sample for a random hold length (matching the bench-measured
      staleness distribution: ~80% within 1 control step, tail up to 7 steps).

    Returns (truth, noisy, noisy_held, true_vel, bounds, seg_names).
    """
    if drone_max_speed is None:
        drone_max_speed = default_drone_max_speed()
    close_speed = drone_max_speed + obstacle_max_speed

    rng = np.random.default_rng(seed)
    segs = []
    seg_names = []

    n1 = 100
    segs.append(np.full(n1, 8.0)); seg_names.append("hover_noisy")
    n2 = 100
    segs.append(np.linspace(8.0, 2.0, n2)); seg_names.append("approach_ramp")
    n3 = 60
    segs.append(np.full(n3, 0.8)); seg_names.append("occlusion_step")
    n4 = 150
    t4 = np.arange(n4) * DT
    segs.append(0.8 + 0.5 * np.sin(2 * np.pi * 0.5 * t4)); seg_names.append("near_miss_osc")

    # 5. fast_real_approach: real abrupt close event at the ACTUAL physical
    # closing speed, distance dropped = close_speed * duration.
    n5 = 5
    drop5 = close_speed * (n5 - 1) * DT
    start5 = min(3.0, 0.1 + drop5 + 0.5)
    segs.append(np.linspace(start5, max(0.1, start5 - drop5), n5)); seg_names.append("fast_real_approach")
    n5b = 40
    segs.append(np.full(n5b, max(0.1, start5 - drop5))); seg_names.append("fast_real_approach_hold")

    # 6. moving_obstacle_only: drone ~stationary, obstacle alone closes at
    # obstacle_max_speed -- only non-trivial when obstacle_max_speed > 0.
    n6 = 6
    drop6 = obstacle_max_speed * (n6 - 1) * DT
    start6 = min(3.0, 0.2 + drop6 + 0.5)
    segs.append(np.linspace(start6, max(0.2, start6 - drop6), n6)); seg_names.append("moving_obstacle_only")
    n6b = 40
    segs.append(np.full(n6b, max(0.2, start6 - drop6))); seg_names.append("moving_obstacle_only_hold")

    truth = np.concatenate(segs)
    bounds = np.cumsum([0] + [len(s) for s in segs])

    true_vel = np.zeros_like(truth)
    true_vel[bounds[1]:bounds[2]] = (segs[1][-1] - segs[1][0]) / ((n2 - 1) * DT)
    true_vel[bounds[3]:bounds[4]] = 0.5 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t4)
    true_vel[bounds[4]:bounds[5]] = (segs[4][-1] - segs[4][0]) / ((n5 - 1) * DT)
    true_vel[bounds[6]:bounds[7]] = (segs[6][-1] - segs[6][0]) / ((n6 - 1) * DT)

    noise = rng.normal(0.0, 0.05, size=truth.shape)
    noisy = truth + noise

    # Burst glitches: 2-3 CONSECUTIVE corrupted samples (not isolated single frames)
    n_bursts = max(1, len(noisy) // 120)
    starts = rng.choice(len(noisy) - 4, size=n_bursts, replace=False)
    for s in starts:
        burst_len = int(rng.integers(2, 4))  # 2 or 3 consecutive bad samples
        lag = int(rng.integers(5, 15))
        src = max(0, s - lag)
        noisy[s:s + burst_len] = truth[src] + rng.normal(0.0, 0.05, size=min(burst_len, len(noisy) - s))

    # held_stale: repeat the current (already noisy) sample for a random hold
    # length, matching the bench-measured staleness distribution.
    hold_vals = np.array([1, 2, 3, 4, 5, 6, 7])
    hold_probs = np.array([0.80, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01])
    noisy_held = np.empty_like(noisy)
    i = 0
    while i < len(noisy):
        k = int(rng.choice(hold_vals, p=hold_probs))
        k = min(k, len(noisy) - i)
        noisy_held[i:i + k] = noisy[i]
        i += k

    return truth, noisy, noisy_held, true_vel, bounds, seg_names


# ── Filters under test ──────────────────────────────────────────────────────

def ema_kinematic(z: np.ndarray, alpha: float = BEV_EMA_ALPHA):
    """Exact recursion from HALO._build_bev() (mapper.py), unclipped for fair RMSE scoring."""
    m_ema = np.empty_like(z)
    dv = np.empty_like(z)
    m_ema[0] = z[0]
    dv[0] = 0.0
    for i in range(1, len(z)):
        m_ema[i] = alpha * z[i] + (1.0 - alpha) * m_ema[i - 1]
        dv[i] = m_ema[i] - m_ema[i - 1]
    return m_ema, dv / DT  # dv/DT -> m/s, comparable to true_vel


def phifilter_kinematic(z: np.ndarray, **kwargs):
    f = PhiFilter(**kwargs)
    level = np.empty_like(z)
    vel = np.empty_like(z)
    for i, zi in enumerate(z):
        level[i] = f.update(float(zi))
        vel[i] = f.vel  # PhiFilter's own alpha-beta velocity state
    return level, vel / DT


# ── Scoring ──────────────────────────────────────────────────────────────────

def score(name, level, vel, truth, true_vel, bounds):
    rmse = float(np.sqrt(np.mean((level - truth) ** 2)))

    # Lag on the ramp segment (samples where filter trails a known linear ramp)
    r0, r1 = bounds[1], bounds[2]
    ramp_lag_m = float(np.mean((truth - level)[r0 + 20:r1]))  # skip transient

    # Overshoot after the step (bounds[2] is the step edge)
    s0, s1 = bounds[2], bounds[3]
    step_true = truth[s0]
    overshoot = float(np.min(level[s0:s1]) - step_true)  # negative = undershoot past truth

    # Noise std during flat hover segment
    h0, h1 = bounds[0], bounds[1]
    noise_std = float(np.std(level[h0 + 10:h1] - truth[h0 + 10:h1]))

    vel_rmse = float(np.sqrt(np.mean((vel - true_vel) ** 2)))
    vel_peak_abs = float(np.max(np.abs(vel)))  # worst single-frame spike a policy could see

    print(f"{name:18s} level_RMSE={rmse:6.3f}m  ramp_lag={ramp_lag_m:+6.3f}m  "
          f"step_overshoot={overshoot:+6.3f}m  hover_noise_std={noise_std:6.3f}m  "
          f"vel_RMSE={vel_rmse:6.3f}m/s  vel_peak_abs={vel_peak_abs:6.2f}m/s")
    return dict(rmse=rmse, ramp_lag=ramp_lag_m, overshoot=overshoot,
                noise_std=noise_std, vel_rmse=vel_rmse, vel_peak_abs=vel_peak_abs)


# ── Hard-signal metrics + param sweep ───────────────────────────────────────

def _attenuation(vel, true_vel, a0, a1):
    true_peak = float(np.max(np.abs(true_vel[a0:a1])))
    est_peak = float(np.max(np.abs(vel[a0:a1])))
    return float(1.0 - est_peak / true_peak) if true_peak > 1e-9 else 0.0


def hard_score(vel, truth, true_vel, bounds):
    """Metrics specific to the adversarial scenarios in make_hard_signal()."""
    vel_rmse = float(np.sqrt(np.mean((vel - true_vel) ** 2)))
    vel_peak_abs = float(np.max(np.abs(vel)))  # worst spike anywhere (glitches included)

    # fast_real_approach: bounds[4]:bounds[5] is the drone-driven real emergency ramp.
    attenuation = _attenuation(vel, true_vel, bounds[4], bounds[5])
    # moving_obstacle_only: bounds[6]:bounds[7] isolates OBSTACLE-driven motion
    # (drone ~stationary) -- only meaningful when obstacle_max_speed > 0.
    obstacle_attenuation = _attenuation(vel, true_vel, bounds[6], bounds[7])

    return dict(vel_rmse=vel_rmse, vel_peak_abs=vel_peak_abs,
                real_attenuation=attenuation, obstacle_attenuation=obstacle_attenuation)


def run_scenario_sweep(seed: int = 0, plot: bool = True):
    """Real numbers (not guesses) for: does PhiFilter still work once (a) velocity
    limits are scaled up for a faster stage, and (b) the obstacle itself moves?
    """
    drone_default = default_drone_max_speed()
    scenarios = [
        ("default (static pillar)", drone_default, 0.0),
        ("2x drone speed (fast stage)", drone_default * 2.0, 0.0),
        ("moving obstacle (+1.5 m/s)", drone_default, 1.5),
        ("2x speed + moving obstacle", drone_default * 2.0, 1.5),
    ]
    print(f"\n[SCENARIO SWEEP] drone_max_speed default = {drone_default:.2f} m/s "
          f"(read live from EnvConfig.vx_limit/vy_limit, not hardcoded)\n")

    plot_data = None
    for label, dspeed, ospeed in scenarios:
        truth, noisy, noisy_held, true_vel, bounds, seg_names = make_hard_signal(
            seed=seed, drone_max_speed=dspeed, obstacle_max_speed=ospeed)
        print(f"-- {label}: drone={dspeed:.2f}m/s obstacle={ospeed:.2f}m/s "
              f"(closing={dspeed + ospeed:.2f}m/s) --")

        _, ema_vel = ema_kinematic(noisy)
        r = hard_score(ema_vel, truth, true_vel, bounds)
        print(f"  {'ema (current)':20s} vel_peak_abs={r['vel_peak_abs']:7.2f}m/s  "
              f"drone_event_attenuation={r['real_attenuation']:+6.3f}  "
              f"obstacle_event_attenuation={r['obstacle_attenuation']:+6.3f}")

        _, phi_vel = phifilter_kinematic(noisy)
        r = hard_score(phi_vel, truth, true_vel, bounds)
        print(f"  {'phifilter (default)':20s} vel_peak_abs={r['vel_peak_abs']:7.2f}m/s  "
              f"drone_event_attenuation={r['real_attenuation']:+6.3f}  "
              f"obstacle_event_attenuation={r['obstacle_attenuation']:+6.3f}")

        _, phi_tuned_vel = phifilter_kinematic(noisy, window=3, gamma=1.0, k_out=1.5)
        r = hard_score(phi_tuned_vel, truth, true_vel, bounds)
        print(f"  {'phifilter (tuned)':20s} vel_peak_abs={r['vel_peak_abs']:7.2f}m/s  "
              f"drone_event_attenuation={r['real_attenuation']:+6.3f}  "
              f"obstacle_event_attenuation={r['obstacle_attenuation']:+6.3f}")
        print()

        if label == "2x speed + moving obstacle":
            plot_data = dict(truth=truth, noisy=noisy, true_vel=true_vel, bounds=bounds,
                              seg_names=seg_names, label=label,
                              ema_vel=ema_vel, phi_vel=phi_vel, phi_tuned_vel=phi_tuned_vel)

    print("[READ] attenuation near 0.0 = real event preserved (safe). Near 1.0 = filter\n"
          "       smoothed away a real closing event (dangerous, missed obstacle).")

    if plot and plot_data is not None:
        _plot_scenario(plot_data)


def _plot_scenario(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth, noisy, true_vel, bounds = d["truth"], d["noisy"], d["true_vel"], d["bounds"]
    t = np.arange(len(truth)) * DT

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(t, truth, "k--", lw=1, label="truth")
    ax1.plot(t, noisy, ".", color="0.8", ms=2, label="noisy input (glitches + held-stale)")
    for b in bounds[1:-1]:
        ax1.axvline(b * DT, color="gray", lw=0.5, ls=":")
    ax1.set_ylabel("distance (m)")
    ax1.legend(fontsize=8)
    ax1.set_title(f"Worst-case scenario: {d['label']} -- distance signal")

    ax2.plot(t, true_vel, "k--", lw=1.3, label="true velocity (ground truth)")
    ax2.plot(t, d["ema_vel"], label="ema dv (current, diff-of-diff)", alpha=0.8)
    ax2.plot(t, d["phi_vel"], label="phifilter (default params)", alpha=0.8)
    ax2.plot(t, d["phi_tuned_vel"], label="phifilter (window=3,gamma=1,k_out=1.5)", alpha=0.9)
    for b in bounds[1:-1]:
        ax2.axvline(b * DT, color="gray", lw=0.5, ls=":")
    ax2.set_ylabel("velocity (m/s)")
    ax2.set_xlabel("time (s)")
    ax2.legend(fontsize=8)
    ax2.set_title("dv/velocity channel -- note fast_real_approach and moving_obstacle_only segments")

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "runs", "compare_ema_phifilter_hard.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\n[PLOT] saved to {out_path}")


def run_param_sweep(seed: int = 0):
    truth, noisy, noisy_held, true_vel, bounds, seg_names = make_hard_signal(seed=seed)

    print(f"\n[HARD TEST] N={len(truth)} samples "
          f"(hover|ramp|step|osc|fast_real_approach), burst-glitches + held-stale reads\n")

    print("-- baseline: EMA (current) on burst-glitch signal, and on held-stale signal --")
    _, ema_vel = ema_kinematic(noisy)
    r = hard_score(ema_vel, truth, true_vel, bounds)
    print(f"{'ema (glitch)':22s} vel_RMSE={r['vel_rmse']:6.3f}m/s  "
          f"vel_peak_abs={r['vel_peak_abs']:7.2f}m/s  real_attenuation={r['real_attenuation']:+6.3f}")
    _, ema_vel_held = ema_kinematic(noisy_held)
    r = hard_score(ema_vel_held, truth, true_vel, bounds)
    print(f"{'ema (held-stale)':22s} vel_RMSE={r['vel_rmse']:6.3f}m/s  "
          f"vel_peak_abs={r['vel_peak_abs']:7.2f}m/s  real_attenuation={r['real_attenuation']:+6.3f}")

    print("\n-- sweep k_out (Hampel outlier threshold), window=8, use_snr=True --")
    for k_out in (2.0, 2.5, 3.0, 3.5, 4.0):
        _, vel = phifilter_kinematic(noisy, k_out=k_out)
        r = hard_score(vel, truth, true_vel, bounds)
        _, vel_h = phifilter_kinematic(noisy_held, k_out=k_out)
        r_h = hard_score(vel_h, truth, true_vel, bounds)
        print(f"k_out={k_out:3.1f}          vel_RMSE={r['vel_rmse']:6.3f}m/s  "
              f"vel_peak_abs={r['vel_peak_abs']:7.2f}m/s  real_attenuation={r['real_attenuation']:+6.3f}  "
              f"|  held-stale: vel_RMSE={r_h['vel_rmse']:6.3f}m/s  real_attenuation={r_h['real_attenuation']:+6.3f}")

    print("\n-- sweep window (lookback for ER/MAD/variance), k_out=3.0, use_snr=True --")
    for window in (5, 8, 12, 16):
        _, vel = phifilter_kinematic(noisy, window=window)
        r = hard_score(vel, truth, true_vel, bounds)
        _, vel_h = phifilter_kinematic(noisy_held, window=window)
        r_h = hard_score(vel_h, truth, true_vel, bounds)
        print(f"window={window:2d}         vel_RMSE={r['vel_rmse']:6.3f}m/s  "
              f"vel_peak_abs={r['vel_peak_abs']:7.2f}m/s  real_attenuation={r['real_attenuation']:+6.3f}  "
              f"|  held-stale: vel_RMSE={r_h['vel_rmse']:6.3f}m/s  real_attenuation={r_h['real_attenuation']:+6.3f}")

    print("\n-- use_snr on/off (k_out=3.0, window=8) --")
    for use_snr in (True, False):
        _, vel = phifilter_kinematic(noisy, use_snr=use_snr)
        r = hard_score(vel, truth, true_vel, bounds)
        print(f"use_snr={str(use_snr):5s}     vel_RMSE={r['vel_rmse']:6.3f}m/s  "
              f"vel_peak_abs={r['vel_peak_abs']:7.2f}m/s  real_attenuation={r['real_attenuation']:+6.3f}")

    print("\n[READ] real_attenuation close to 0.0 = genuine emergency signal preserved (safe).")
    print("       real_attenuation close to 1.0 = filter smoothed away a real close-obstacle event (dangerous).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hard", action="store_true", help="Run the harder adversarial test + param sweep instead")
    args = p.parse_args()

    if args.hard:
        run_param_sweep(seed=args.seed)
        run_scenario_sweep(seed=args.seed, plot=not args.no_plot)
        return

    truth, noisy, true_vel, bounds = make_signal(seed=args.seed)

    ema_level, ema_vel = ema_kinematic(noisy)
    phi_level, phi_vel = phifilter_kinematic(noisy)
    phi_lead_level, phi_lead_vel = phifilter_kinematic(noisy, lead=True)

    print(f"\n[COMPARE] N={len(truth)} samples, dt={DT*1000:.0f}ms "
          f"(hover|ramp|step|oscillation)\n")
    results = {}
    results["ema (current)"] = score("ema (current)", ema_level, ema_vel, truth, true_vel, bounds)
    results["phifilter"] = score("phifilter", phi_level, phi_vel, truth, true_vel, bounds)
    results["phifilter_lead"] = score("phifilter_lead", phi_lead_level, phi_lead_vel, truth, true_vel, bounds)

    print("\n[VERDICT]")
    if results["phifilter"]["noise_std"] < 0.8 * results["ema (current)"]["noise_std"]:
        print(" - phifilter reduces hover noise leak vs current EMA (adaptive gain working).")
    if abs(results["phifilter"]["ramp_lag"]) < abs(results["ema (current)"]["ramp_lag"]):
        print(" - phifilter tracks the ramp with less lag than current EMA.")
    if results["phifilter_lead"]["ramp_lag"] < results["phifilter"]["ramp_lag"]:
        print(" - lead=True further cuts ramp lag (trades a bit of hover noise for it).")
    if results["phifilter"]["vel_rmse"] < results["ema (current)"]["vel_rmse"]:
        print(" - phifilter's alpha-beta velocity state is more accurate than diff-of-diff dv.")
    ema_peak = results["ema (current)"]["vel_peak_abs"]
    phi_peak = results["phifilter"]["vel_peak_abs"]
    if phi_peak < 0.5 * ema_peak:
        print(f" - stale-frame glitches spike EMA dv up to {ema_peak:.1f}m/s (diff-of-diff has no "
              f"outlier gate) vs phifilter's {phi_peak:.1f}m/s (Hampel/MAD gate absorbs them). "
              f"This directly correlates with the BEV staleness spikes measured earlier this session.")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        t = np.arange(len(truth)) * DT
        ax1.plot(t, truth, "k--", lw=1, label="truth")
        ax1.plot(t, noisy, ".", color="0.75", ms=2, label="noisy input")
        ax1.plot(t, ema_level, label="ema (current)")
        ax1.plot(t, phi_level, label="phifilter")
        ax1.plot(t, phi_lead_level, label="phifilter (lead=True)")
        for b in bounds[1:-1]:
            ax1.axvline(b * DT, color="gray", lw=0.5, ls=":")
        ax1.set_ylabel("distance (m)")
        ax1.legend(fontsize=8)
        ax1.set_title("M_t level: EMA vs PhiFilter")

        ax2.plot(t, true_vel, "k--", lw=1, label="true velocity")
        ax2.plot(t, ema_vel, label="ema dv (diff-of-diff)")
        ax2.plot(t, phi_vel, label="phifilter vel (alpha-beta state)")
        ax2.set_ylabel("velocity (m/s)")
        ax2.set_xlabel("time (s)")
        ax2.legend(fontsize=8)
        ax2.set_title("dv/velocity channel")

        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "runs", "compare_ema_phifilter.png",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        print(f"\n[PLOT] saved to {out_path}")


if __name__ == "__main__":
    main()
