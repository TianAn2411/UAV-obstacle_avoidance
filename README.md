# UAV Obstacle Avoidance — PPO + PX4 SITL

Reinforcement learning pipeline for training a quadrotor to navigate through pillar obstacles using depth vision. Runs on PX4 SITL + Gazebo with ROS 2 bridging, designed for sim-to-real transfer.

---

## Prerequisites

### 1. PX4 Autopilot source (release/1.15)

```bash
git clone -b release/1.15 https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
bash Tools/setup/ubuntu.sh
make px4_sitl
```

This repo (`obstacle_avoidance/`) must live inside `~/PX4-Autopilot/`.

### 2. ROS 2 Jazzy

Follow the [official ROS 2 Jazzy install guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html), then:

```bash
sudo apt install ros-jazzy-ros-gz-bridge ros-jazzy-ros-gz-sim
source /opt/ros/jazzy/setup.bash   # add to ~/.bashrc
```

### 3. Gazebo Harmonic

```bash
sudo apt install gz-harmonic
```

### 4. MicroXRCE-DDS Agent

```bash
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
```

### 5. px4_msgs ROS 2 package

> **Important:** `px4_msgs` must match your PX4 firmware branch exactly.
> Using the wrong branch causes topic type mismatches — the bridge connects
> but messages are silently dropped or cause `rcl` errors.

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone the branch that matches PX4 release/1.15
git clone -b release/1.15 https://github.com/PX4/px4_msgs.git

cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select px4_msgs
source install/setup.bash   # add to ~/.bashrc
```

**Common symptom of wrong px4_msgs version:**
```
[ros2topic]: Could not import message definition for type 'px4_msgs/msg/VehicleOdometry'
# or silent: callbacks never fire even though bridge is running
```

If this happens, verify branch alignment:
```bash
# Check PX4 firmware branch
cd ~/PX4-Autopilot && git branch

# Check px4_msgs branch
cd ~/ros2_ws/src/px4_msgs && git branch
# Both must show: release/1.15
```

### 6. Python venv

```bash
python3 -m venv ~/drone_rl_env
source ~/drone_rl_env/bin/activate
pip install stable-baselines3 gymnasium numpy opencv-python
```

---

## Requirements Summary

| Dependency | Version |
|---|---|
| PX4 Autopilot | `release/1.15` at `~/PX4-Autopilot`, SITL built |
| Gazebo | Harmonic (`gz-harmonic`) |
| ROS 2 | Jazzy |
| MicroXRCE-DDS Agent | installed to system |
| px4_msgs | `release/1.15` branch, built in `~/ros2_ws` |
| Python venv | `~/drone_rl_env` with `stable-baselines3`, `gymnasium` |

---

## Quick Start

```bash
# Activate venv + source ROS 2 (or add both to ~/.bashrc)
source ~/drone_rl_env/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# Run training from PX4-Autopilot root
cd ~/PX4-Autopilot

./run_train.sh --0    # Stage 0: learn basic navigation (no pillars, 295k steps)
./run_train.sh --1    # Stage 1: full 4D control (no pillars, 500k steps)
./run_train.sh --2    # Stage 2: 3 pillars  (500k steps)
./run_train.sh --3    # Stage 3: 5 pillars  (500k steps)
./run_train.sh --4    # Stage 4: 10 pillars (500k steps)
./run_train.sh --5    # Stage 5: 20 pillars (open)
```

Interrupted model (`ppo_drone_stage{N}_interrupted.zip`) is always resumed automatically — no flag needed.

**Direct invocation:**
```bash
source ~/drone_rl_env/bin/activate
cd ~/PX4-Autopilot
python3 -m obstacle_avoidance.train --stage 1
```

---

## Architecture

```
obstacle_avoidance/
  train.py                  # Entry point: make_env() + run_training()
  configs/
    env_config.py           # EnvConfig — fence, reset, action params
    reward_config.py        # RewardConfig — all reward coefficients
    pillar_config.py        # PillarConfig — pillar geometry/spawn
    ppo_config.yaml         # PPO hyperparams + 5-stage curriculum table
    machine_config.yaml     # Per-machine CPU/core tuning
  envs/
    drone_env.py            # DroneObstacleEnv (gym.Env wrapper)
    policy.py               # DepthStateExtractor (CNN + MLP feature extractor)
    monitor.py              # TrainingMonitor SB3 callback
    manager/
      train_manager.py      # Step/reset orchestrator (main logic)
      action_manager.py     # Velocity smoothing, takeoff assist
      noise_manager.py      # All domain randomisation (obs/action/mass/wind)
      reward_manager.py     # Per-step reward computation
      reset_manager.py      # Reset strategy classification + dispatch
      pillar_manager.py     # Pillar spawn, subgoals, collision
      logging_manager.py    # Per-env step/episode file logging
  utils/
    bridge_factory.py       # ROSBridge (rclpy Node) + ROS/GZ helpers
    process_utils.py        # Gazebo/PX4/MicroXRCE process launchers
    px4_manager.py          # Per-rank PX4 SITL instance manager
    gz_transport_client.py  # Native Gazebo Transport client
  symbolic_extractor/       # Symbolic feature extraction pipeline (submodule)
```

### symbolic_extractor

`symbolic_extractor/` is a submodule implementing a proprietary symbolic perception pipeline that converts raw sensor data into structured feature representations for the RL policy. Interface: see `symbolic_extractor/pipeline.py`.

### Data flow

```
PPO.learn()
  → SubprocVecEnv (n_envs subprocesses)
    → DroneObstacleEnv.step(action)
      → TrainManager.step_process(action)
            ActionManager            → smoothed velocity cmd
            NoiseManager.step_action → action delay + ESC jitter
            ROSBridge.send_velocity  → PX4 offboard setpoint (ENU→NED)
            ROSBridge.tick(dt)       → rclpy spin loop
            ROSBridge.get_*()        → position, depth, velocity
            PillarManager.update()   → collision, subgoal, attention
            NoiseManager.apply_obs_noise() → inject obs Gaussian noise
            RewardManager.compute()  → scalar reward + components
            _check_terminal()        → (terminated, truncated, reason)
```

---

## Observation & Action Space

**Observation** — dict space:

| Key | Shape | Description |
|---|---|---|
| `depth` | `(3, 84, 84)` float32 | 3 stacked depth frames, clipped to [0, 10] m |
| `state` | `(31,)` float32 | ego-centric state vector (see below) |

State vector layout (all normalized, body-frame FLU):
```
[0:3]   velocity FLU      (vx, vy, vz)
[3:6]   angular velocity  (roll_rate, pitch_rate, yaw_rate)
[6]     altitude
[7:10]  goal body-FLU     (gx, gy, gz)
[10:14] orientation       (sin_yaw, cos_yaw, pitch/45°, roll/45°)
[14:18] last cmd sent to PX4, normalized to [-1,1]
[18:22] delta_A1 = A_t - A_{t-1}           (action kinematic velocity)
[22:26] delta_A2 = ΔA1_t - ΔA1_{t-1}      (action kinematic acceleration)
[26:30] fence proximity body-FLU (forward, back, left, right) ∈ [0,1]
[30]    DFA progress = subgoal_idx / total_subgoals ∈ [0,1]
```

**Action** — `(4,)` float32 in `[-1, 1]`: `[vx, vy, vz, yaw_rate]`
Scaled by velocity limits in `EnvConfig`, smoothed with α=0.35.

---

## Reward Structure

All coefficients live in `configs/reward_config.py`.

The reward uses **Potential-Based Reward Shaping (PBRS)** with a hybrid potential over both XY distance progress and DFA subgoal progress. This guarantees policy-invariance while providing dense shaped signal.

Key reward components:

| Group | Components |
|---|---|
| PBRS | `pbrs` (γΦ(s') − Φ(s)), `rm_bonus` (subgoal + pillar-passed bonuses) |
| Navigation | `yaw_align`, `lateral`, `speed_penalty`, `too_slow_penalty` |
| Safety | `pillar_too_close`, `pillar_clearance_soft`, `collision_course`, `near_miss` |
| Behavior | `smooth`, `yaw_rate_penalty`, `obstacle_slowdown`, `pillar_attention` |
| Penalty | `time`, `ground`, `altitude`, `near_fence`, `start_zone` |
| Terminal | one-shot on episode end |

---

## Curriculum

| Stage | Pillars | Min steps | Notes |
|---|---|---|---|
| 0 | 0 | 295k | Learn basic nav + altitude control, CNN frozen, no noise |
| 1 | 0 | 500k | Full 4D control, CNN frozen, light obs noise |
| 2 | 3 | 500k | CNN unfreezes, action delay + mass/wind noise |
| 3 | 5 | 500k | Increased noise + wind |
| 4 | 10 | 500k | Near real-hardware noise levels |
| 5 | 20 | open | Full sim-to-real noise |

Goal acceptance radius and spawn distance ramp automatically over training steps.

---

## Domain Randomisation

All sim-to-real randomisation is centralised in `envs/manager/noise_manager.py`:

| Type | Frequency | Purpose |
|---|---|---|
| Obs noise (pos/vel/ang_vel/quat) | per-step Gaussian | VIO/IMU drift |
| Action delay | fixed buffer | control latency |
| ESC jitter | per-step Gaussian | actuator uncertainty |
| Mass scale | per-episode uniform | mass/motor/battery variation |
| Wind | per-episode random direction + magnitude | external disturbance |

---

## Reset Strategy

`ResetManager` classifies each episode end into:

| Mode | When |
|---|---|
| `startup_arm` | First episode or drone grounded |
| `continuous` | Normal in-bounds terminal — teleport to new start |
| `rescue_then_continuous` | Drone near/outside fence — velocity-driven back in |
| `hard` | Sim failure (failsafe, EKF dead) — disarm → teleport → rearm |

---

## Multi-Env Isolation

Each of `n_envs` parallel environments gets dedicated:
- `ROS_DOMAIN_ID = 30 + rank`
- `GZ_PARTITION = drone_rl_{rank}`
- `PX4_INSTANCE = rank` (shared UXRCE-DDS agent on port 8888)
- Per-rank SDF world file with patched world name
- Staggered startup to prevent simultaneous Gazebo launches

---

## Coordinate System

| Frame | Convention |
|---|---|
| Gazebo | ENU (East-North-Up) |
| PX4 | NED (North-East-Down) |
| `ROSBridge.send_velocity()` | input ENU → internally converts to NED |
| `ROSBridge.get_linear_velocity()` | returns ENU |

---

## Checkpoints & Resume

Checked in order at startup:

1. `ppo_drone_stage{N}_interrupted.zip` — Ctrl+C save
2. `ckpts/stage{N}/stage{N}_{steps}_steps.zip` — latest periodic
3. `ppo_drone_stage{N}_final.zip`
4. `ppo_drone_stage{N-1}_final.zip` — stage transfer (lr reset to 1e-4)
5. Fresh model

---

## Logs & Artifacts

```
obstacle_avoidance/
  runs/
    train_stage{N}_{timestamp}.log          # training log
    training_progress_stage{N}.csv          # per-checkpoint CSV
    env_logs/env_{rank}/stage{N}_{ts}/      # per-env step logs
  ckpts/stage{N}/
    stage{N}_{steps}_steps.zip
  ppo_drone_stage{N}_interrupted.zip
  ppo_drone_stage{N}_final.zip
```

---

## CPU Tuning (`configs/machine_config.yaml`)

Tune CPU allocation per machine. All pinning disabled by default.

```yaml
pin_processes: false         # true = sched_setaffinity for PX4 + bridges
                             # WARNING: never pin Python worker — contains 5Hz keepalive thread

main_process_cores: "0-3"   # cores for main PPO process (PyTorch)
torch_num_threads: 4         # intra-op threads
torch_interop_threads: 2     # inter-op threads

cores_per_env: 6             # cores per rank when pin_processes=true
                             # rank 0 → cores 4-9, rank 1 → cores 10-15, ...
```

| Cores | Recommended config |
|---|---|
| 8 | `pin_processes: false`, `torch_num_threads: 2` |
| 16 | `torch_num_threads: 4`, optionally `pin_processes: true`, `cores_per_env: 4` |
| 24 | `torch_num_threads: 4`, `pin_processes: true`, `cores_per_env: 6` |

---

## Syntax Check

```bash
source ~/drone_rl_env/bin/activate
python3 -m py_compile obstacle_avoidance/train.py
python3 -m py_compile obstacle_avoidance/envs/drone_env.py
python3 -m py_compile obstacle_avoidance/utils/bridge_factory.py
python3 -m py_compile obstacle_avoidance/envs/manager/noise_manager.py
python3 -m py_compile obstacle_avoidance/envs/manager/train_manager.py
```
