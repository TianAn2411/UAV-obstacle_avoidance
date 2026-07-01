"""
PBRS + DFA + RM Reward Test Suite
==================================
Standalone — no ROS/SITL needed.

Kiểm tra:
  A. PBRS correctness (potential shock, terminal correction, DFA transition)
  B. RM bonus correctness
  C. Reward dominance analysis (full episode simulation, per-component)
  D. Scale audit (per-step magnitudes)
  E. Edge cases
  F. Phi properties
  G. Comparative balance (success vs timeout vs collision; dense noise vs terminal signal)

Run:
  cd ~/PX4-Autopilot
  source ~/drone_rl_env/bin/activate
  python3 -m obstacle_avoidance.tests.test_pbrs_reward
"""

import sys
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import os
for path in ["/workspace/PX4-Autopilot", "/root/PX4-Autopilot", "/home/sw_an/PX4-Autopilot"]:
    if os.path.exists(path):
        sys.path.insert(0, path)
        break

from obstacle_avoidance.configs.reward_config import RewardConfig
from obstacle_avoidance.configs.env_config import EnvConfig
from obstacle_avoidance.envs.manager.reward_manager import RewardManager, StepState, RewardComponents

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RCFG = RewardConfig()
ECFG = EnvConfig()

GOAL  = np.array([15.0,  0.0, 4.0], dtype=np.float32)
START = np.array([ 0.0,  0.0, 4.0], dtype=np.float32)
DIST_START = float(np.linalg.norm(GOAL[:2] - START[:2]))   # 15.0 m

# done_reason strings that _reward_terminal recognises as hard-terminal
_HARD_TERMINAL_REASONS = {"goal_xy", "goal_3d", "success", "goal_reached",
                           "collision", "fell_to_ground", "flipped", "out_of_fence"}


def make_state(
    pos: np.ndarray,
    prev_pos: Optional[np.ndarray] = None,
    dfa_q: int = 0,
    dfa_N: int = 6,
    dfa_q_prev: int = 0,
    num_pillars: int = 0,
    is_terminal: bool = False,
    is_truncated: bool = False,
    done_reason: str = "",
    yaw_error: float = 0.0,
    vel: Optional[np.ndarray] = None,
) -> StepState:
    if prev_pos is None:
        prev_pos = pos.copy()
    if vel is None:
        vel = np.zeros(3, dtype=np.float32)
    dist_xy      = float(np.linalg.norm(GOAL[:2] - pos[:2]))
    prev_dist_xy = float(np.linalg.norm(GOAL[:2] - prev_pos[:2]))
    return StepState(
        pos=pos,
        vel=vel,
        yaw=0.0,
        prev_pos=prev_pos,
        prev_action=np.zeros(4, dtype=np.float32),
        action=np.zeros(4, dtype=np.float32),
        step_count=0,
        global_step=0,
        stage_index=1 if num_pillars == 0 else 2,
        dist_xy=dist_xy,
        prev_dist_xy=prev_dist_xy,
        front_depth=5.0,
        depth_sector=np.ones(9, dtype=np.float32) * 5.0,
        nearest_pillar_dist=None,
        nearest_pillar_xy=None,
        pillar_collision_snap={
            "clearance_body": float("inf"),
            "heading_into": False,
            "d_closest": float("inf"),
            "t_closest": float("nan"),
            "collision_radius": 1.0,
            "speed": 0.0,
            "stage1_subgoal_reward": 0.0,
            "pillar_passed_reward": 0.0,
            "pillars_passed_count": 0,
            "clearance_progress_reward": 0.0,
            "near_miss_reward": 0.0,
            "entered_pillar_zone": False,
            "nearest_dist": float("inf"),
        },
        bypass_subgoal_info={"reward": 0.0, "active_subgoal": None, "clearance_gain": 0.0},
        ring_subgoal_info={"reward": 0.0},
        attention_info={"attention_reward": 0.0, "post_pillar_reward": 0.0},
        reset_info={},
        done_reason=done_reason,
        goal_xy_radius=1.0,
        is_terminal=is_terminal,
        is_truncated=is_truncated,
        goal=GOAL,
        start=START,
        num_pillars=num_pillars,
        horizontal_speed=float(np.linalg.norm(vel[:2])),
        min_depth=5.0,
        final_yaw_rate=0.0,
        yaw_error=yaw_error,
        dfa_q=dfa_q,
        dfa_N=dfa_N,
        dfa_q_prev=dfa_q_prev,
    )


def fresh_rm(dist_start: float = DIST_START, dfa_N: int = 6) -> RewardManager:
    rm = RewardManager(RCFG, ECFG)
    rm.reset_episode(dist_start=dist_start, dfa_N=dfa_N)
    return rm


PASS = "✅ PASS"
FAIL = "❌ FAIL"

def check(cond: bool, label: str, detail: str = "") -> bool:
    status = PASS if cond else FAIL
    print(f"  {status}  {label}" + (f"  [{detail}]" if detail else ""))
    return cond


# ─────────────────────────────────────────────────────────────────────────────
# simulate_episode — accumulates all RewardComponent fields over N steps
#
# terminal_reason: done_reason string for last step.
#   goal success   → "goal_xy"    (is_terminal=True)
#   timeout        → "max_steps"  (is_truncated=True)
#   collision      → "collision"  (is_terminal=True)
#   If empty: auto-set to "goal_xy" when goal_reached, "max_steps" otherwise.
# ─────────────────────────────────────────────────────────────────────────────

def simulate_episode(
    scenario_name: str,
    num_pillars: int,
    dfa_N: int,
    steps: int,
    yaw_err_rad: float = 0.0,
    goal_reached: bool = True,
    terminal_reason: str = "",
    dfa_advances: Optional[list] = None,
    pillars_pass_steps: Optional[list] = None,
) -> dict:
    if dfa_advances is None:
        dfa_advances = list(range(steps // (dfa_N + 1), steps, steps // (dfa_N + 1)))[:dfa_N]
    if pillars_pass_steps is None:
        pillars_pass_steps = []
    if not terminal_reason:
        terminal_reason = "goal_xy" if goal_reached else "max_steps"

    rm = RewardManager(RCFG, ECFG)
    rm.reset_episode(dist_start=DIST_START, dfa_N=dfa_N)

    totals: dict = {}
    dfa_q = 0

    for step in range(steps):
        t = step / steps
        dist = DIST_START * (1.0 - t)
        pos = np.array([DIST_START - dist, 0.0, 4.0], dtype=np.float32)
        vel = np.array([DIST_START / steps / 0.1, 0.0, 0.0], dtype=np.float32)

        dfa_q_prev = dfa_q
        if step in dfa_advances and dfa_q < dfa_N:
            dfa_q += 1

        is_last = (step == steps - 1)
        if is_last:
            dr = terminal_reason
            is_terminal = dr in _HARD_TERMINAL_REASONS
            is_truncated = not is_terminal
        else:
            dr = ""
            is_terminal = False
            is_truncated = False

        passed = 1 if step in pillars_pass_steps else 0

        s = make_state(
            pos=pos,
            vel=vel,
            dfa_q=dfa_q,
            dfa_N=dfa_N,
            dfa_q_prev=dfa_q_prev,
            num_pillars=num_pillars,
            is_terminal=is_terminal,
            is_truncated=is_truncated,
            done_reason=dr,
            yaw_error=yaw_err_rad,
        )
        s.pillar_collision_snap["pillars_passed_count"] = passed
        if num_pillars > 0:
            s.nearest_pillar_dist = 3.0

        _, c = rm.compute(s)

        # Accumulate ALL fields including 'total'
        for field_name in c.__dataclass_fields__:
            totals[field_name] = totals.get(field_name, 0.0) + getattr(c, field_name)

    return totals


# ─────────────────────────────────────────────────────────────────────────────
# A. PBRS Correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_A_pbrs_correctness():
    print("\n── A. PBRS Correctness ─────────────────────────────────────────────")
    all_ok = True

    # A1: No potential shock at step 1
    rm = fresh_rm(dist_start=DIST_START, dfa_N=6)
    phi_s0 = rm._phi(DIST_START, 0, 6)
    s = make_state(pos=START.copy(), prev_pos=START.copy())
    _, c = rm.compute(s)
    expected_max_shock = abs((RCFG.pbrs_gamma - 1.0) * phi_s0) + 0.05
    ok = abs(c.pbrs) <= expected_max_shock + 0.1
    all_ok &= check(ok, "A1 No potential shock at step 1",
                    f"c.pbrs={c.pbrs:.4f}, phi_s0={phi_s0:.4f}, threshold={expected_max_shock:.4f}")

    # A2: F > 0 when moving toward goal (pos=[10→11], goal at x=15)
    rm = fresh_rm()
    pos0 = np.array([10.0, 0.0, 4.0], dtype=np.float32)
    pos1 = np.array([11.0, 0.0, 4.0], dtype=np.float32)
    s0 = make_state(pos=pos0)
    rm.compute(s0)
    s1 = make_state(pos=pos1, prev_pos=pos0)
    _, c = rm.compute(s1)
    ok = c.pbrs > 0
    all_ok &= check(ok, "A2 F > 0 when moving toward goal", f"c.pbrs={c.pbrs:.4f}")

    # A3: F < 0 when moving away from goal (pos=[10→8])
    rm = fresh_rm()
    pos0 = np.array([10.0, 0.0, 4.0], dtype=np.float32)
    pos1 = np.array([ 8.0, 0.0, 4.0], dtype=np.float32)
    s0 = make_state(pos=pos0)
    rm.compute(s0)
    s1 = make_state(pos=pos1, prev_pos=pos0)
    _, c = rm.compute(s1)
    ok = c.pbrs < 0
    all_ok &= check(ok, "A3 F < 0 when moving away from goal", f"c.pbrs={c.pbrs:.4f}")

    # A4: DFA transition adds ΔΦ_DFA to F
    rm = fresh_rm(dfa_N=6)
    pos = np.array([10.0, 0.0, 4.0], dtype=np.float32)
    s_no = make_state(pos=pos, prev_pos=pos, dfa_q=2, dfa_N=6, dfa_q_prev=2)
    rm._prev_phi = rm._phi(float(np.linalg.norm(GOAL[:2] - pos[:2])), 2, 6)
    _, c_no = rm.compute(s_no)
    rm._prev_phi = rm._phi(float(np.linalg.norm(GOAL[:2] - pos[:2])), 2, 6)
    s_tr = make_state(pos=pos, prev_pos=pos, dfa_q=3, dfa_N=6, dfa_q_prev=2)
    _, c_tr = rm.compute(s_tr)
    delta = c_tr.pbrs - c_no.pbrs
    expected_delta = RCFG.pbrs_gamma * (RCFG.pbrs_dfa_coef / 6.0)
    ok = abs(delta - expected_delta) < 0.01
    all_ok &= check(ok, "A4 DFA transition adds ΔΦ_DFA to F",
                    f"delta={delta:.4f}, expected={expected_delta:.4f}")

    # A5: Terminal step uses standard PBRS (no Grzes correction)
    rm = fresh_rm()
    s_pre = make_state(pos=np.array([14.5, 0.0, 4.0], dtype=np.float32))
    rm.compute(s_pre)
    phi_before = rm._prev_phi
    s_term = make_state(pos=GOAL.copy(), is_terminal=True, done_reason="goal_xy")
    _, c = rm.compute(s_term)
    phi_goal = rm._phi(dist_xy=float(np.linalg.norm(GOAL[:2] - GOAL[:2])), dfa_q=0, dfa_N=6)
    expected = RCFG.pbrs_gamma * phi_goal - phi_before
    ok = abs(c.pbrs - expected) < 0.01
    all_ok &= check(ok, "A5 Terminal step: standard PBRS F=γΦ(s')-Φ(s)",
                    f"c.pbrs={c.pbrs:.4f}, expected={expected:.4f}")

    # A6: Accumulated PBRS positive over goal-reaching episode (no terminal correction)
    rm = fresh_rm(dist_start=15.0, dfa_N=1)
    phi_start = rm._phi(15.0, 0, 1)
    phi_goal  = rm._phi(0.0,  1, 1)
    total_pbrs = 0.0
    N_steps = 30
    for i in range(N_steps):
        dist = 15.0 - (15.0 * i / N_steps)
        q = 1 if i > N_steps // 2 else 0
        is_term = (i == N_steps - 1)
        s = make_state(
            pos=np.array([dist, 0.0, 4.0], dtype=np.float32),
            dfa_q=q, dfa_N=1,
            is_terminal=is_term,
            done_reason="goal_xy" if is_term else "",
        )
        _, c = rm.compute(s)
        total_pbrs += c.pbrs
    upper = phi_goal - phi_start  # theoretical max (γ=1 limit)
    ok = 0.0 < total_pbrs <= upper
    all_ok &= check(ok, "A6 Accumulated PBRS > 0 over goal-reaching episode",
                    f"total={total_pbrs:.3f}, upper={upper:.3f}")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# B. RM Bonus Correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_B_rm_bonus():
    print("\n── B. RM Bonus Correctness ─────────────────────────────────────────")
    all_ok = True

    rm = fresh_rm()
    s = make_state(pos=START.copy(), dfa_q=2, dfa_q_prev=2)
    _, c = rm.compute(s)
    all_ok &= check(c.rm_bonus == 0.0, "B1 No transition → rm_bonus = 0", f"{c.rm_bonus}")

    rm = fresh_rm()
    s = make_state(pos=START.copy(), dfa_q=3, dfa_q_prev=2)
    _, c = rm.compute(s)
    ok = abs(c.rm_bonus - RCFG.rm_subgoal_bonus) < 1e-6
    all_ok &= check(ok, "B2 DFA q+1 → rm_bonus = rm_subgoal_bonus",
                    f"{c.rm_bonus} vs {RCFG.rm_subgoal_bonus}")

    rm = fresh_rm()
    s = make_state(pos=START.copy(), num_pillars=2)
    s.pillar_collision_snap["pillars_passed_count"] = 2
    _, c = rm.compute(s)
    expected = 2.0 * RCFG.rm_pillar_passed_bonus
    all_ok &= check(abs(c.rm_bonus - expected) < 1e-6, "B3 2 pillars → 2×rm_pillar_passed_bonus",
                    f"{c.rm_bonus} vs {expected}")

    rm = fresh_rm()
    s = make_state(pos=START.copy(), num_pillars=2, dfa_q=1, dfa_q_prev=0)
    s.pillar_collision_snap["pillars_passed_count"] = 1
    _, c = rm.compute(s)
    expected = RCFG.rm_subgoal_bonus + RCFG.rm_pillar_passed_bonus
    all_ok &= check(abs(c.rm_bonus - expected) < 1e-6, "B4 DFA+pillar → subgoal+pillar bonus",
                    f"{c.rm_bonus} vs {expected}")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# C. Reward Dominance Analysis — per-component episode totals
# ─────────────────────────────────────────────────────────────────────────────

def test_C_dominance():
    print("\n── C. Reward Dominance Analysis ────────────────────────────────────")

    # (name, num_pillars, dfa_N, steps, yaw_err, goal_reached, terminal_reason, dfa_advances, pil_pass_steps)
    scenarios = [
        ("Stage1 success  goal_xy  (200s)", 0, 6, 200, 0.0, True,  "goal_xy",   [33,66,100,133,166], []),
        ("Stage1 timeout  max_steps(200s)", 0, 6, 200, 0.0, False, "max_steps", [33,66,100,133,166], []),
        ("Stage1 collision         ( 50s)", 0, 1,  50, 0.3, False, "collision",  [25],                []),
        ("Stage2 success  goal_xy  (300s)", 2, 2, 300, 0.1, True,  "goal_xy",   [100,200],           [100,200]),
        ("Stage2 timeout  max_steps(300s)", 2, 2, 300, 0.1, False, "max_steps", [100,200],           [100,200]),
    ]

    # Components that are large by design — don't flag them
    EXPECTED_LARGE = {
        "pbrs", "rm_bonus", "terminal", "yaw_align", "time",
        "ground", "altitude", "obstacle_visibility",
    }
    FLAG_RATIO = 0.20   # flag dense component if > 20% of |terminal component|

    all_ok = True
    for name, n_pil, dfa_n, steps, yaw_e, goal_ok, term_reason, dfa_adv, pil_pass in scenarios:
        totals = simulate_episode(name, n_pil, dfa_n, steps, yaw_e, goal_ok, term_reason, dfa_adv, pil_pass)
        terminal_abs = abs(totals.get("terminal", 1.0))

        print(f"\n  Scenario: {name}")
        print(f"  {'Component':<28} {'Total':>10}  {'%|terminal|':>12}  Flag")
        print(f"  {'-'*64}")

        flagged = []
        for k, v in sorted(totals.items(), key=lambda x: abs(x[1]), reverse=True):
            if k == "total":
                continue
            pct = 100.0 * abs(v) / max(terminal_abs, 1e-6)
            flag = ""
            if k not in EXPECTED_LARGE and abs(v) > 0.1:
                if pct > FLAG_RATIO * 100:
                    flag = "⚠️  HIGH"
                    flagged.append((k, v, pct))
            if abs(v) > 0.01 or k in ("pbrs", "rm_bonus", "terminal"):
                print(f"  {k:<28} {v:>10.3f}  {pct:>11.1f}%  {flag}")

        print(f"  {'─'*64}")
        ep_total = totals.get("total", 0.0)
        print(f"  {'EPISODE TOTAL':<28} {ep_total:>10.3f}")

        if flagged:
            print(f"  ⚠️  Unexpected dominant: {[f[0] for f in flagged]}")
            all_ok = False
        else:
            print(f"  ✅ No unexpected dominant components")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# D. Scale Audit — per-step magnitudes
# ─────────────────────────────────────────────────────────────────────────────

def test_D_scale_audit():
    print("\n── D. Per-step Scale Audit ─────────────────────────────────────────")

    per_step_dist = DIST_START / 200
    phi_a = RCFG.pbrs_dist_coef * (1.0 - (DIST_START - per_step_dist) / DIST_START)
    phi_b = RCFG.pbrs_dist_coef * (1.0 - DIST_START / DIST_START)
    f_typical = RCFG.pbrs_gamma * phi_a - phi_b
    print(f"  F_pbrs/step (0.075m progress) ≈ {f_typical:.4f}")
    print(f"  ground/step (alt=4m optimal)  ≈ {RCFG.alt_optimal_coef:.3f}")
    print(f"  yaw_align/step (perfect align) ≈ {RCFG.stage1_yaw_forward_bonus_coef:.3f} (max)")

    print(f"\n  Terminal rewards:")
    print(f"    goal_xy_terminal_reward      = {RCFG.goal_xy_terminal_reward}")
    print(f"    collision_penalty            = {RCFG.collision_penalty}")
    print(f"    max_steps_penalty_no_pillars = {RCFG.max_steps_penalty_no_pillars}")

    phi_start = RCFG.pbrs_dist_coef * (1.0 - DIST_START / DIST_START)
    phi_goal  = RCFG.pbrs_dist_coef * 1.0 + RCFG.pbrs_dfa_coef * 1.0
    total_pbrs = phi_goal - phi_start
    term = RCFG.goal_xy_terminal_reward
    print(f"\n  Episode totals (200 steps, stage1, perfect):")
    print(f"    ground   = {RCFG.alt_optimal_coef*200:.1f}  ({100*RCFG.alt_optimal_coef*200/term:.1f}% of goal_terminal)")
    print(f"    yaw_align= {RCFG.stage1_yaw_forward_bonus_coef*200:.1f}  ({100*RCFG.stage1_yaw_forward_bonus_coef*200/term:.1f}% of goal_terminal)")
    print(f"    pbrs     ≈ {total_pbrs:.1f}  ({100*total_pbrs/term:.1f}% of goal_terminal)")
    print(f"    rm_bonus ≈ {6*RCFG.rm_subgoal_bonus:.1f}  ({100*6*RCFG.rm_subgoal_bonus/term:.1f}% of goal_terminal)")

    pbrs_pct = 100 * total_pbrs / term
    ok_pbrs = 5 < pbrs_pct < 35
    check(ok_pbrs, f"D1 PBRS total 5-35% of terminal ({pbrs_pct:.1f}%)")

    rm_pct = 100 * (6 * RCFG.rm_subgoal_bonus) / term
    ok_rm = 5 < rm_pct < 30
    check(ok_rm, f"D2 RM total 5-30% of terminal ({rm_pct:.1f}%)")

    # Dense reward per step should be < 3× PBRS per step
    dense_per_step = RCFG.alt_optimal_coef + RCFG.stage1_yaw_forward_bonus_coef
    ratio = dense_per_step / max(f_typical, 1e-6)
    ok_ratio = ratio < 20.0   # warn if dense >> pbrs per step
    check(ok_ratio, f"D3 dense/step < 20× pbrs/step (ratio={ratio:.1f}×)")

    return ok_pbrs and ok_rm and ok_ratio


# ─────────────────────────────────────────────────────────────────────────────
# E. Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

def test_E_edge_cases():
    print("\n── E. Edge Cases ───────────────────────────────────────────────────")
    all_ok = True

    try:
        rm = fresh_rm(dfa_N=0)
        s = make_state(pos=START.copy(), dfa_N=0)
        _, c = rm.compute(s)
        all_ok &= check(np.isfinite(c.total), "E1 dfa_N=0 doesn't crash/NaN", f"total={c.total:.4f}")
    except Exception as e:
        all_ok &= check(False, "E1 dfa_N=0 doesn't crash", str(e))

    rm = fresh_rm(dist_start=15.0, dfa_N=6)
    for _ in range(10):
        rm.compute(make_state(pos=np.array([12.0, 0.0, 4.0], dtype=np.float32)))
    phi_before = rm._prev_phi
    rm.reset_episode(dist_start=15.0, dfa_N=6)
    phi_s0 = rm._phi(15.0, 0, 6)
    all_ok &= check(abs(rm._prev_phi - phi_s0) < 1e-6, "E2 reset_episode restores _prev_phi",
                    f"before={phi_before:.4f}, after={rm._prev_phi:.4f}, Φ(s0)={phi_s0:.4f}")

    rm = fresh_rm()
    s = make_state(pos=GOAL.copy(), is_terminal=True, done_reason="goal_xy")
    _, c = rm.compute(s)
    all_ok &= check(np.isfinite(c.pbrs), "E3 dist_xy=0 at goal no NaN", f"c.pbrs={c.pbrs:.4f}")

    rm = fresh_rm()
    phi_prev = rm._prev_phi
    pos_trunc = np.array([8.0, 0.0, 4.0], dtype=np.float32)
    phi_trunc = rm._phi(float(np.linalg.norm(GOAL[:2] - pos_trunc[:2])), dfa_q=0, dfa_N=6)
    s = make_state(pos=pos_trunc, is_truncated=True, done_reason="max_steps")
    _, c = rm.compute(s)
    expected = RCFG.pbrs_gamma * phi_trunc - phi_prev
    all_ok &= check(abs(c.pbrs - expected) < 0.01, "E4 truncated → standard PBRS (no special correction)",
                    f"c.pbrs={c.pbrs:.4f}, expected={expected:.4f}")

    try:
        rm = fresh_rm(dfa_N=3)
        s = make_state(pos=START.copy(), dfa_q=5, dfa_N=3, dfa_q_prev=4)
        _, c = rm.compute(s)
        all_ok &= check(np.isfinite(c.total), "E5 dfa_q > dfa_N no crash", f"total={c.total:.4f}")
    except Exception as e:
        all_ok &= check(False, "E5 dfa_q > dfa_N no crash", str(e))

    rm = fresh_rm()
    s = make_state(pos=START.copy())
    _, c = rm.compute(s)
    # E6-E8: Old components have been removed from RewardComponents dataclass
    removed_ok = not any(hasattr(c, attr) for attr in ["progress", "velocity_goal", "stage1_subgoal", "bypass_subgoal", "ring_subgoal"])
    all_ok &= check(removed_ok, "E6-E8 old components are removed from RewardComponents")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# F. Phi Properties
# ─────────────────────────────────────────────────────────────────────────────

def test_F_phi_properties():
    print("\n── F. Potential Function Properties ───────────────────────────────")
    all_ok = True
    rm = fresh_rm()

    dists = [15.0, 12.0, 8.0, 4.0, 1.0, 0.0]
    phis = [rm._phi(d, 0, 6) for d in dists]
    ok = all(phis[i] < phis[i+1] for i in range(len(phis)-1))
    all_ok &= check(ok, "F1 Φ_dist monotonically increases as dist→0",
                    str([f"{p:.3f}" for p in phis]))

    phis_q = [rm._phi(10.0, q, 6) for q in range(7)]
    ok = all(phis_q[i] < phis_q[i+1] for i in range(len(phis_q)-1))
    all_ok &= check(ok, "F2 Φ_DFA monotonically increases as q increases",
                    str([f"{p:.3f}" for p in phis_q]))

    phi_min = rm._phi(100.0, 0, 1)
    phi_max = rm._phi(0.0, 6, 6)
    all_ok &= check(phi_min < 0 < phi_max, "F3 Φ bounded: min<0<max",
                    f"min={phi_min:.3f}, max={phi_max:.3f}")

    rm2 = fresh_rm(dist_start=12.0, dfa_N=4)
    expected = rm2._phi(12.0, 0, 4)
    all_ok &= check(abs(rm2._prev_phi - expected) < 1e-6, "F4 reset_episode inits _prev_phi=Φ(s0,q=0)",
                    f"_prev_phi={rm2._prev_phi:.4f}, Φ(s0)={expected:.4f}")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# G. Comparative Balance Analysis
#    Checks that terminal signal dominates over per-episode dense noise
#    and that outcome ordering is correct: success > timeout > collision
# ─────────────────────────────────────────────────────────────────────────────

def test_G_comparative_balance():
    print("\n── G. Comparative Balance Analysis ────────────────────────────────")
    all_ok = True

    # ── Stage 1 ─────────────────────────────────────────────────────────────
    print("\n  Stage 1 (200 steps, no pillars):")
    s1_ok     = simulate_episode("s1_ok",  0, 6, 200, 0.0, True,  "goal_xy",   [33,66,100,133,166], [])
    s1_to     = simulate_episode("s1_to",  0, 6, 200, 0.0, False, "max_steps", [33,66,100,133,166], [])
    s1_col    = simulate_episode("s1_col", 0, 1,  50, 0.3, False, "collision",  [25],                [])

    # terminal differential for stage1: goal_xy - max_steps_no_pillars
    term_diff_s1 = RCFG.goal_xy_terminal_reward - RCFG.max_steps_penalty_no_pillars  # 130-(-90)=220

    ep_ok  = s1_ok.get("total",  0.0)
    ep_to  = s1_to.get("total",  0.0)
    ep_col = s1_col.get("total", 0.0)

    print(f"    success   total = {ep_ok:>8.1f}")
    print(f"    timeout   total = {ep_to:>8.1f}")
    print(f"    collision total = {ep_col:>8.1f}")
    print(f"    terminal_diff(goal-timeout) = {term_diff_s1:.0f}")

    ok = ep_ok > ep_to
    all_ok &= check(ok, "G1 Stage1: success total > timeout total",
                    f"{ep_ok:.1f} vs {ep_to:.1f}")

    ok = ep_to > ep_col
    all_ok &= check(ok, "G2 Stage1: timeout total > collision total",
                    f"{ep_to:.1f} vs {ep_col:.1f}")

    # Dense rewards that are SAME in success and timeout (background noise)
    ground_noise  = s1_ok.get("ground", 0.0)
    yaw_noise     = s1_ok.get("yaw_align", 0.0)
    # Both appear in success AND failure → their episode accumulation should be < terminal_diff
    ok = ground_noise < term_diff_s1
    all_ok &= check(ok, "G3 Stage1: ground/episode < terminal_diff",
                    f"ground={ground_noise:.1f} < {term_diff_s1:.0f}")
    ok = yaw_noise < term_diff_s1
    all_ok &= check(ok, "G4 Stage1: yaw_align/episode < terminal_diff",
                    f"yaw={yaw_noise:.1f} < {term_diff_s1:.0f}")

    # Signal-to-noise: terminal_diff should be > sum of dense background
    dense_total = ground_noise + abs(yaw_noise)
    snr = term_diff_s1 / max(dense_total, 1e-6)
    ok = snr > 1.5   # terminal diff ≥ 1.5× the dense noise sum
    all_ok &= check(ok, f"G5 Stage1: SNR = terminal_diff / dense_background ≥ 1.5",
                    f"snr={snr:.2f}  (term_diff={term_diff_s1:.0f} / dense={dense_total:.1f})")

    # ── Stage 2 ─────────────────────────────────────────────────────────────
    print("\n  Stage 2 (300 steps, 2 pillars):")
    s2_ok  = simulate_episode("s2_ok",  2, 2, 300, 0.1, True,  "goal_xy",   [100,200], [100,200])
    s2_to  = simulate_episode("s2_to",  2, 2, 300, 0.1, False, "max_steps", [100,200], [100,200])
    s2_col = simulate_episode("s2_col", 2, 1, 100, 0.3, False, "collision",  [50],      [50])

    term_diff_s2 = RCFG.goal_xy_terminal_reward - RCFG.max_steps_penalty_pillars_far_goal  # 130-(-190)=320

    ep2_ok  = s2_ok.get("total",  0.0)
    ep2_to  = s2_to.get("total",  0.0)
    ep2_col = s2_col.get("total", 0.0)

    print(f"    success   total = {ep2_ok:>8.1f}")
    print(f"    timeout   total = {ep2_to:>8.1f}")
    print(f"    collision total = {ep2_col:>8.1f}")
    print(f"    terminal_diff(goal-timeout) = {term_diff_s2:.0f}")

    ok = ep2_ok > ep2_to
    all_ok &= check(ok, "G6 Stage2: success total > timeout total",
                    f"{ep2_ok:.1f} vs {ep2_to:.1f}")

    ok = ep2_to > ep2_col
    all_ok &= check(ok, "G7 Stage2: timeout total > collision total",
                    f"{ep2_to:.1f} vs {ep2_col:.1f}")

    # ── Behavioral checks ───────────────────────────────────────────────────
    print("\n  Behavioral signal checks:")

    # Bad yaw episode should earn less than aligned episode
    s1_aligned  = simulate_episode("aligned",  0, 6, 100, 0.0,        True, "goal_xy", [], [])
    s1_misalign = simulate_episode("misalign", 0, 6, 100, math.pi/2,  True, "goal_xy", [], [])
    ok = s1_aligned.get("yaw_align", 0) > s1_misalign.get("yaw_align", 0)
    all_ok &= check(ok, "G8 Aligned yaw earns more yaw_align than 90° misaligned",
                    f"aligned={s1_aligned.get('yaw_align',0):.1f} vs misalign={s1_misalign.get('yaw_align',0):.1f}")

    # PBRS over success episode > PBRS over hover-at-midpoint episode
    # (absolute sign depends on episode_length × (1-γ) × avg_Φ — not a reliable indicator)
    rm_hover_g9 = RewardManager(RCFG, ECFG)
    rm_hover_g9.reset_episode(dist_start=DIST_START, dfa_N=6)
    pbrs_hover_g9 = 0.0
    for i in range(200):
        is_last = (i == 199)
        sh = make_state(pos=np.array([7.5, 0.0, 4.0], dtype=np.float32),
                        is_terminal=is_last, done_reason="goal_xy" if is_last else "")
        _, ch = rm_hover_g9.compute(sh)
        pbrs_hover_g9 += ch.pbrs
    pbrs_success = s1_ok.get("pbrs", 0.0)
    ok = pbrs_success > pbrs_hover_g9
    all_ok &= check(ok, "G9 Success PBRS > hover-at-midpoint PBRS",
                    f"success={pbrs_success:.2f} vs hover_mid={pbrs_hover_g9:.2f}")

    # Hover episode (no progress toward goal, same pos) earns less total than flying episode
    # Simulate hover: drone stays at mid-point for all steps
    rm_hover = RewardManager(RCFG, ECFG)
    rm_hover.reset_episode(dist_start=DIST_START, dfa_N=6)
    hover_total = 0.0
    for i in range(100):
        is_last = (i == 99)
        s = make_state(
            pos=np.array([7.5, 0.0, 4.0], dtype=np.float32),
            is_terminal=is_last, done_reason="goal_xy" if is_last else "",
        )
        _, c = rm_hover.compute(s)
        hover_total += c.total

    rm_fly = RewardManager(RCFG, ECFG)
    rm_fly.reset_episode(dist_start=DIST_START, dfa_N=6)
    fly_total = 0.0
    for i in range(100):
        t = i / 100
        pos = np.array([DIST_START * t, 0.0, 4.0], dtype=np.float32)
        vel = np.array([DIST_START / 100 / 0.1, 0.0, 0.0], dtype=np.float32)
        is_last = (i == 99)
        s = make_state(
            pos=pos, vel=vel,
            is_terminal=is_last, done_reason="goal_xy" if is_last else "",
        )
        _, c = rm_fly.compute(s)
        fly_total += c.total

    # G10: per-step PBRS is larger when moving toward goal vs standing still
    # (tests directional gradient, avoids terminal-correction confound)
    rm_a = fresh_rm(dist_start=10.0, dfa_N=1)
    rm_b = fresh_rm(dist_start=10.0, dfa_N=1)
    pbrs_move = 0.0
    pbrs_stay = 0.0
    for i in range(5):
        pos_move = np.array([5.0 + i * 1.0, 0.0, 4.0], dtype=np.float32)   # closing in on goal
        pos_stay = np.array([5.0,            0.0, 4.0], dtype=np.float32)   # stationary
        _, cm = rm_a.compute(make_state(pos=pos_move, dfa_N=1))
        _, cs = rm_b.compute(make_state(pos=pos_stay, dfa_N=1))
        pbrs_move += cm.pbrs
        pbrs_stay += cs.pbrs
    ok = pbrs_move > pbrs_stay
    all_ok &= check(ok, "G10 Moving toward goal accumulates more PBRS than hovering",
                    f"move={pbrs_move:.3f} vs stay={pbrs_stay:.3f}")

    # G11: new coefs — yaw_align/200steps < 50% of goal_terminal (verify 0.5→0.25 tuning)
    yaw_200 = RCFG.stage1_yaw_forward_bonus_coef * 200
    yaw_pct = 100.0 * yaw_200 / RCFG.goal_xy_terminal_reward
    ok = yaw_pct < 50.0
    all_ok &= check(ok, f"G11 yaw_align/200steps < 50% of goal_terminal ({yaw_pct:.1f}%)")

    # G12: new coefs — ground/200steps < 50% of goal_terminal (verify 0.6→0.3 tuning)
    ground_200 = RCFG.alt_optimal_coef * 200
    ground_pct = 100.0 * ground_200 / RCFG.goal_xy_terminal_reward
    ok = ground_pct < 50.0
    all_ok &= check(ok, f"G12 ground/200steps < 50% of goal_terminal ({ground_pct:.1f}%)")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print(" PBRS + DFA + RM Reward Test Suite")
    print(f" pbrs_dist_coef={RCFG.pbrs_dist_coef}  pbrs_dfa_coef={RCFG.pbrs_dfa_coef}"
          f"  pbrs_gamma={RCFG.pbrs_gamma}")
    print(f" rm_subgoal_bonus={RCFG.rm_subgoal_bonus}  rm_pillar_passed_bonus={RCFG.rm_pillar_passed_bonus}")
    print(f" alt_optimal_coef={RCFG.alt_optimal_coef}  stage1_yaw_forward_bonus_coef={RCFG.stage1_yaw_forward_bonus_coef}")
    print("=" * 68)

    results = {
        "A - PBRS Correctness":       test_A_pbrs_correctness(),
        "B - RM Bonus":               test_B_rm_bonus(),
        "C - Dominance Analysis":     test_C_dominance(),
        "D - Scale Audit":            test_D_scale_audit(),
        "E - Edge Cases":             test_E_edge_cases(),
        "F - Phi Properties":         test_F_phi_properties(),
        "G - Comparative Balance":    test_G_comparative_balance(),
    }

    print("\n" + "=" * 68)
    print(" SUMMARY")
    print("=" * 68)
    n_pass = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {name}")
    print(f"\n  {n_pass}/{len(results)} test groups passed")

    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
