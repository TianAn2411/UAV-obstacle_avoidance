#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_train.sh — SB3 PPO Launcher (obstacle_avoidance/)
# ============================================================
#
# Usage:
#   ./run_train.sh --1
#   ./run_train.sh --2
#
# Args:
#   --1 | --2 | --3 | --4 | --5   Curriculum stage (passed as --stage N)
#
# Notes:
#   - Interrupted model always takes resume priority (no flag needed)
#   - N_ENVS and checkpoint_freq are in obstacle_avoidance/configs/ppo_config.yaml
# ============================================================

PROJECT_DIR="/home/sw_an/PX4-Autopilot"
VENV_DIR="/home/sw_an/drone_rl_env"
NEW_TRAIN_PY="$PROJECT_DIR/obstacle_avoidance/train.py"

STAGE=""

# ── Parse args ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --0|--1|--2|--3|--4|--5)
            STAGE="${1#--}"
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: $0 --<stage>"
            echo "Example: $0 --2"
            exit 1
            ;;
    esac
    shift
done

if [ -z "$STAGE" ]; then
    STAGE="1"
    echo "[CONFIG] No stage specified, defaulting to stage $STAGE"
fi

echo "============================================================"
echo "[CONFIG] Backend:   obstacle_avoidance (new modular)"
echo "[CONFIG] Stage:     $STAGE"
echo "[CONFIG] Resume:    interrupted model takes priority automatically"
echo "============================================================"

# ── Clean old processes ──────────────────────────────────────
echo "[CLEAN] Kill old PX4 / Gazebo / ROS-GZ processes"

pkill -TERM -f "MicroXRCEAgent" || true
pkill -TERM -f "parameter_bridge" || true
pkill -TERM -f "ros_gz_bridge" || true
pkill -TERM -f "px4.*-i" || true
pkill -TERM -f "bin/px4" || true
pkill -TERM -f "px4_sitl" || true
pkill -TERM -f "gz sim" || true
pkill -TERM -f "gzserver" || true
pkill -TERM -f "gzclient" || true
pkill -TERM -f "ruby.*gz" || true

sleep 3

pkill -KILL -f "MicroXRCEAgent" || true
pkill -KILL -f "parameter_bridge" || true
pkill -KILL -f "ros_gz_bridge" || true
pkill -KILL -f "px4.*-i" || true
pkill -KILL -f "bin/px4" || true
pkill -KILL -f "px4_sitl" || true
pkill -KILL -f "gz sim" || true
pkill -KILL -f "gzserver" || true
pkill -KILL -f "gzclient" || true
pkill -KILL -f "ruby.*gz" || true

sleep 2

echo "[CHECK] Remaining processes:"
ps aux | grep -E "px4|MicroXRCEAgent|gz sim|gzserver|gzclient|parameter_bridge|ros_gz_bridge" | grep -v grep || echo "  (none)"

# ── Activate venv ────────────────────────────────────────────
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "[ERROR] Virtualenv not found: $VENV_DIR/bin/activate"
    exit 1
fi

# ── Compile check ────────────────────────────────────────────
echo "[CHECK] py_compile"

python3 -m py_compile "$NEW_TRAIN_PY"
python3 -m py_compile "$PROJECT_DIR/obstacle_avoidance/envs/drone_env.py"
python3 -m py_compile "$PROJECT_DIR/obstacle_avoidance/envs/monitor.py"
python3 -m py_compile "$PROJECT_DIR/obstacle_avoidance/envs/policy.py"
python3 -m py_compile "$PROJECT_DIR/obstacle_avoidance/utils/process_utils.py"
python3 -m py_compile "$PROJECT_DIR/obstacle_avoidance/utils/px4_manager.py"
python3 -m py_compile "$PROJECT_DIR/obstacle_avoidance/utils/checkpoint_utils.py"
python3 -m py_compile "$PROJECT_DIR/obstacle_avoidance/utils/bridge_factory.py"

echo "[CHECK] All files OK"

# ── Thread limits ────────────────────────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1
export OPENCV_NUM_THREADS=1

# ── Run training ─────────────────────────────────────────────
echo "============================================================"
echo "[RUN] obstacle_avoidance PPO Training — stage $STAGE"
echo "============================================================"

cd "$PROJECT_DIR"
python3 -m obstacle_avoidance.train --stage "$STAGE"
