#!/usr/bin/env python3
"""
Phẫu thuật mở rộng state vector 23→31 cho model đã train.

Strategy: patch zip trong memory (data JSON + policy.pth) rồi load bình thường.
  - data["observation_space"]: state (23,) → (31,)
  - data["learning_rate"]: warm-up LR (3e-5)
  - policy.pth: Linear(23,64).weight surgery → Linear(31,64)
  - optimizer: bỏ qua → SB3 tạo Adam mới khi load

Chạy 1 lần:
  cd ~/PX4-Autopilot/obstacle_avoidance
  python3 utils/expand_state_weights.py
"""
import io
import os
import pickle
import shutil
import sys
import zipfile

import numpy as np
import torch
import gymnasium as gym

_repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.insert(0, _repo_root)

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file, data_to_json

MODEL_ZIP_IN = "interrupt/raws/ppo_drone_stage0_interrupted"
PKL_IN       = "interrupt/raws/ppo_drone_stage0_vecnormalize_interrupted"
MODEL_OUT    = "ppo_drone_stage0_expanded_31"
TMP_ZIP      = "_tmp_surgery_31.zip"
WARM_LR      = 3e-5   # 10× nhỏ hơn 3e-4 — warm-up cho 8 cột zero hội tụ

# ── 1. Load raw data + params từ zip cũ ──────────────────────────────────────
print(f"Loading {MODEL_ZIP_IN}.zip ...")
data, params, _ = load_from_zip_file(f"{MODEL_ZIP_IN}.zip", device="cpu")

# ── 2. Patch obs_space: state (23,) → (31,) ──────────────────────────────────
old_obs_space = data["observation_space"]
new_obs_space = gym.spaces.Dict({
    "depth": old_obs_space.spaces["depth"],
    "state": gym.spaces.Box(-np.inf, np.inf, shape=(31,), dtype=np.float32),
})
data["observation_space"] = new_obs_space
print("  obs_space patched: state (23,) → (31,)")

# ── 3. Patch learning rate → warm-up ─────────────────────────────────────────
data["learning_rate"] = WARM_LR
print(f"  learning_rate → {WARM_LR}")

# ── 4. Surgery trọng số Linear(23,64) → Linear(31,64) ────────────────────────
WEIGHT_KEYS = [k for k in params["policy"].keys() if k.endswith("state_fc.0.weight")]
assert len(WEIGHT_KEYS) >= 1, f"No state_fc.0.weight keys found"

sample_w = params["policy"][WEIGHT_KEYS[0]]
if sample_w.shape == (64, 31):
    print("  Weight already 31-dim — skip surgery (model was already copied)")
    NEED_MODEL_SURGERY = False
elif sample_w.shape == (64, 23):
    NEED_MODEL_SURGERY = True
    for key in WEIGHT_KEYS:
        old_w = params["policy"][key]
        new_w = torch.zeros(64, 31, dtype=torch.float32)
        new_w[:, :18]   = old_w[:, :18]    # vel..last_cmd
        new_w[:, 18:26] = 0.0              # delta_A1, delta_A2 — zero init
        new_w[:, 26:31] = old_w[:, 18:23]  # fence + DFA — shift
        params["policy"][key] = new_w
        print(f"  Patched {key}: (64,23) → (64,31)")
else:
    raise AssertionError(f"Unexpected weight shape: {sample_w.shape}")

# ── 5+6+7. Ghi/load/save model ───────────────────────────────────────────────
if NEED_MODEL_SURGERY:
    print(f"Writing patched zip: {TMP_ZIP}")
    with zipfile.ZipFile(f"{MODEL_ZIP_IN}.zip", "r") as old_zip:
        with zipfile.ZipFile(TMP_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as new_zip:
            for item in old_zip.infolist():
                if item.filename == "data":
                    new_zip.writestr("data", data_to_json(data))
                elif item.filename == "policy.pth":
                    buf = io.BytesIO()
                    torch.save(params["policy"], buf)
                    new_zip.writestr("policy.pth", buf.getvalue())
                else:
                    new_zip.writestr(item, old_zip.read(item.filename))

    print("Loading patched zip ...")
    model = PPO.load(TMP_ZIP, device="cpu")
    os.remove(TMP_ZIP)
else:
    print(f"Loading {MODEL_ZIP_IN}.zip (already 31-dim) ...")
    model = PPO.load(f"{MODEL_ZIP_IN}.zip", device="cpu")

fc0 = model.policy.features_extractor.state_fc[0]
assert fc0.in_features == 31, f"Expected 31, got {fc0.in_features}"
print(f"  state_fc[0]: Linear({fc0.in_features}, {fc0.out_features}) ✓")

# Reset Adam → momentum từ checkpoint sai shape; fresh optimizer đúng shape
model.policy.optimizer = torch.optim.Adam(
    model.policy.parameters(),
    lr=WARM_LR,
    eps=1e-5,
)
print(f"  Optimizer reset with warm_lr={WARM_LR}")

model.save(MODEL_OUT)
print(f"Saved: {MODEL_OUT}.zip")

# ── 8. Phẫu thuật VecNormalize pkl ───────────────────────────────────────────
PKL_OUT = f"{MODEL_OUT}.pkl"

with open(f"{PKL_IN}.pkl", "rb") as f:
    vn = pickle.load(f)

# Patch observation_space: state (23,) → (31,)
# SB3 check_shape_equal khi set_venv → phải khớp với env mới
vn.observation_space.spaces["state"] = gym.spaces.Box(
    -np.inf, np.inf, shape=(31,), dtype=np.float32
)

# Patch old_obs cache (không critical — ghi đè ở step đầu, nhưng tránh crash nếu SB3 đọc shape)
old_state = vn.old_obs["state"]   # (1, 23)
new_state = np.zeros((1, 31), dtype=np.float32)
new_state[:, :18]   = old_state[:, :18]
new_state[:, 18:26] = 0.0
new_state[:, 26:31] = old_state[:, 18:23]
vn.old_obs["state"] = new_state

# Fix __getstate__ compat: SB3 mới xóa class_attributes + returns khỏi instance dict
# pkl cũ không có 2 attr này → inject để __getstate__ không KeyError
if "class_attributes" not in vn.__dict__:
    vn.__dict__["class_attributes"] = {}
if "returns" not in vn.__dict__:
    vn.__dict__["returns"] = np.zeros(vn.num_envs, dtype=np.float64)

with open(PKL_OUT, "wb") as f:
    pickle.dump(vn, f)

print(f"Saved pkl: {PKL_OUT}")
print(f"  obs_space['state'] patched: (23,) → (31,)")
print(f"  ret_rms preserved — mean={vn.ret_rms.mean:.4f}  var={vn.ret_rms.var:.4f}  count={vn.ret_rms.count:.0f}")
print()
print("Done. Tiếp theo:")
print(f"  cp {MODEL_OUT}.zip interrupt/raws/ppo_drone_stage0_interrupted.zip")
print(f"  cp {MODEL_OUT}.pkl interrupt/raws/ppo_drone_stage0_vecnormalize_interrupted.pkl")
print("  ./run_train.sh --1")
