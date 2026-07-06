#!/usr/bin/env python3
"""
CNN architecture surgery: k=8,s=4 → k=4,s=2 (flatten 3136 → 5184).

Reinits: features_extractor.cnn.*, features_extractor.cnn_fc.*
  (CNN frozen in stage 0/1 → random weights, no value to preserve)
Transfers: state_fc, fusion_fc, fusion_norm, PPO actor/critic heads
  (500k+ steps of training — fully preserved)

Sets LR = 3e-5 (warm-up) so fusion_fc re-calibrates with new CNN output
over remaining stage 1 steps before stage 2 unfreezes CNN.

Usage:
  cd ~/PX4-Autopilot/obstacle_avoidance
  source ~/drone_rl_env/bin/activate
  python3 utils/swap_cnn_arch.py [--stage N] [--in MODEL_IN] [--out MODEL_OUT]
"""

import argparse
import io
import os
import sys
import zipfile

import torch

_repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.insert(0, _repo_root)

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file, data_to_json

WARM_LR = 3e-5

# SB3 MultiInputActorCriticPolicy stores 3 copies of feature extractor:
#   features_extractor.*, pi_features_extractor.*, vf_features_extractor.*
_FE_PREFIXES = [
    "features_extractor",
    "pi_features_extractor",
    "vf_features_extractor",
]

# Suffix → new shape (shared across all 3 prefixes)
# cnn indices: 0=Conv1, 2=Conv2, 4=Conv3 (odd indices are SiLU, no params)
_CNN_SUFFIX_SHAPES = {
    "cnn.0.weight":    (32,  3, 4, 4),   # was (32, 3, 8, 8)
    "cnn.0.bias":      (32,),
    "cnn.2.weight":    (64, 32, 4, 4),   # same shape, different stride
    "cnn.2.bias":      (64,),
    "cnn.4.weight":    (64, 64, 3, 3),   # same shape, different stride
    "cnn.4.bias":      (64,),
    "cnn_fc.0.weight": (256, 5184),      # was (256, 3136)
    "cnn_fc.0.bias":   (256,),
}

# Expand to all prefixes
NEW_CNN_SHAPES = {
    f"{prefix}.{suffix}": shape
    for prefix in _FE_PREFIXES
    for suffix, shape in _CNN_SUFFIX_SHAPES.items()
}


def _kaiming_init(tensor: torch.Tensor) -> torch.Tensor:
    torch.nn.init.kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu")
    return tensor


def swap_cnn(model_in: str, model_out: str) -> None:
    print(f"Loading {model_in} ...")
    data, params, _ = load_from_zip_file(model_in, device="cpu")

    policy = params["policy"]

    # Report old shapes (only features_extractor prefix to avoid repetition)
    for suffix, new_shape in _CNN_SUFFIX_SHAPES.items():
        key = f"features_extractor.{suffix}"
        old_shape = tuple(policy[key].shape) if key in policy else None
        if old_shape != new_shape:
            print(f"  {suffix}: {old_shape} → {new_shape}")
        else:
            print(f"  {suffix}: {old_shape} → {new_shape}  (same shape, reinit)")
    print(f"  (patching across {len(_FE_PREFIXES)} extractors: {_FE_PREFIXES})")

    # Reinit all CNN keys with new shapes + Kaiming init
    for key, shape in NEW_CNN_SHAPES.items():
        t = torch.zeros(shape, dtype=torch.float32)
        if "weight" in key and len(shape) > 1:
            _kaiming_init(t)
        policy[key] = t

    # Patch warm-up LR
    old_lr = data.get("learning_rate", "?")
    data["learning_rate"] = WARM_LR
    print(f"\n  learning_rate: {old_lr} → {WARM_LR}")

    # Write patched zip (copy all files, replace data + policy.pth)
    tmp = model_out + ".tmp"
    src = model_in if model_in.endswith(".zip") else model_in + ".zip"
    dst = model_out if model_out.endswith(".zip") else model_out + ".zip"

    print(f"\nWriting {dst} ...")
    with zipfile.ZipFile(src, "r") as old_zip:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as new_zip:
            for item in old_zip.infolist():
                if item.filename == "data":
                    new_zip.writestr("data", data_to_json(data))
                elif item.filename == "policy.pth":
                    buf = io.BytesIO()
                    torch.save(policy, buf)
                    new_zip.writestr("policy.pth", buf.getvalue())
                else:
                    new_zip.writestr(item, old_zip.read(item.filename))

    os.replace(tmp, dst)
    print(f"Saved: {dst}")

    # ── Verify: load with new arch code (policy.py must already be updated) ──
    print("\nVerifying load ...")
    model = PPO.load(dst, device="cpu")
    fe = model.policy.features_extractor
    sd = model.policy.state_dict()

    # Architecture checks
    conv0  = fe.cnn[0]
    conv1  = fe.cnn[2]
    conv2  = fe.cnn[4]
    cnn_fc = fe.cnn_fc[0]
    state_fc   = fe.state_fc[0]
    fusion_fc  = fe.fusion_fc[0]
    fusion_norm = fe.fusion_norm

    assert conv0.kernel_size  == (4, 4),  f"conv0 kernel: {conv0.kernel_size}"
    assert conv0.stride       == (2, 2),  f"conv0 stride: {conv0.stride}"
    assert conv1.kernel_size  == (4, 4),  f"conv1 kernel: {conv1.kernel_size}"
    assert conv1.stride       == (2, 2),  f"conv1 stride: {conv1.stride}"
    assert conv2.kernel_size  == (3, 3),  f"conv2 kernel: {conv2.kernel_size}"
    assert conv2.stride       == (2, 2),  f"conv2 stride: {conv2.stride}"
    assert cnn_fc.in_features == 5184,    f"cnn_fc in_features: {cnn_fc.in_features}"
    assert state_fc.in_features  == 31,   f"state_fc in_features: {state_fc.in_features}"
    assert fusion_fc.in_features == 320,  f"fusion_fc in_features: {fusion_fc.in_features}"

    # Weight integrity checks
    # CNN reinitialized → norm should be Kaiming-scale (not near zero, not huge)
    cnn_w_norm   = sd["features_extractor.cnn_fc.0.weight"].norm().item()
    # state_fc transferred → norm should reflect trained weights (typically 0.5–5.0)
    state_w_norm = sd["features_extractor.state_fc.0.weight"].norm().item()
    # fusion_fc transferred
    fusion_w_norm = sd["features_extractor.fusion_fc.0.weight"].norm().item()

    assert cnn_w_norm > 0.01,   f"cnn_fc weight near zero — Kaiming init failed"
    assert state_w_norm > 0.01, f"state_fc weight near zero — transfer failed"

    print()
    print("=" * 55)
    print("  SURGERY REPORT")
    print("=" * 55)
    print(f"  CNN arch (REINITIALIZED — Kaiming):")
    print(f"    cnn[0]: Conv2d(3→32,  k={conv0.kernel_size[0]}, s={conv0.stride[0]})")
    print(f"    cnn[2]: Conv2d(32→64, k={conv1.kernel_size[0]}, s={conv1.stride[0]})")
    print(f"    cnn[4]: Conv2d(64→64, k={conv2.kernel_size[0]}, s={conv2.stride[0]})")
    print(f"    cnn_fc: Linear(5184→256)  weight_norm={cnn_w_norm:.4f}")
    print(f"  State branch (TRANSFERRED — trained weights):")
    print(f"    state_fc[0]: Linear(31→64)  weight_norm={state_w_norm:.4f}")
    print(f"  Fusion (TRANSFERRED):")
    print(f"    fusion_fc[0]: Linear(320→256)  weight_norm={fusion_w_norm:.4f}")
    print(f"  LR: {data['learning_rate']}  (warm-up)")
    print(f"  Output: {dst}")
    print("=" * 55)
    print("  SURGERY SUCCESSFUL — run ./run_train.sh --1 to resume")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--in",  dest="model_in",  default=None)
    parser.add_argument("--out", dest="model_out", default=None)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Mirrors train.py's raw/symbolic namespace split -- must match or this
    # silently reads/writes the wrong mode's checkpoint.
    is_symbolic = False
    try:
        import yaml
        with open(os.path.join(base, "configs", "ppo_config.yaml")) as f:
            is_symbolic = str((yaml.safe_load(f) or {}).get("policy_mode", "raw")) == "symbolic"
    except Exception:
        pass
    model_prefix = f"ppo_drone_stage{args.stage}" if not is_symbolic else f"ppo_drone_symbolic_stage{args.stage}"
    mode_subdir = "symbolics" if is_symbolic else "raws"
    default_path = os.path.join(base, "interrupt", mode_subdir, f"{model_prefix}_interrupted.zip")
    model_in  = args.model_in  or default_path
    model_out = args.model_out or default_path

    if not os.path.exists(model_in):
        sys.exit(f"Model not found: {model_in}")

    swap_cnn(model_in, model_out)


if __name__ == "__main__":
    main()
