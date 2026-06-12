# UAV Obstacle Avoidance — PPO + PX4 SITL

Reinforcement learning pipeline for training a quadrotor to navigate through pillar obstacles using depth vision. Runs on PX4 SITL + Gazebo with ROS2 bridging, designed for sim-to-real transfer.

---

## Prerequisites

### 1. PX4 Autopilot source

Clone and set up PX4:
```bash
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
bash Tools/setup/ubuntu.sh
```

This repo (`obstacle_avoidance/`) must live inside `~/PX4-Autopilot/`.

### 2. ROS2 Jazzy

Follow the [official ROS2 Jazzy install guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html), then install Gazebo bridge:
```bash
sudo apt install ros-jazzy-ros-gz-bridge ros-jazzy-ros-gz-sim
```

Source ROS2 in your shell (or add to `~/.bashrc`):
```bash
source /opt/ros/jazzy/setup.bash
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

### 5. px4_msgs ROS2 package

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/PX4/px4_msgs.git
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### 6. Python venv

```bash
python3 -m venv ~/drone_rl_env
source ~/drone_rl_env/bin/activate
pip install stable-baselines3 gymnasium numpy
```

---

## Requirements Summary

| Dependency | Version |
|---|---|
| PX4 Autopilot | cloned at `~/PX4-Autopilot`, SITL built |
| Gazebo | Harmonic (`gz-harmonic`) |
| ROS2 | Jazzy |
| MicroXRCE-DDS Agent | installed to system |
| px4_msgs | built in `~/ros2_ws` |
| Python venv | `~/drone_rl_env` with `stable-baselines3`, `gymnasium` |

Build PX4 SITL:
```bash
cd ~/PX4-Autopilot && make px4_sitl
```

---

## Quick Start

```bash
# Stage 0: altitude-hold only (learn horizontal nav, vz frozen)
cd ~/PX4-Autopilot
./run_train.sh --0

# Stage 1–5: progressive pillar curriculum
./run_train.sh --1    # 0 pillars   (500k steps)
./run_train.sh --2    # 2 pillars   (500k steps)
./run_train.sh --3    # 5 pillars   (500k steps)
./run_train.sh --4    # 10 pillars  (500k steps)
./run_train.sh --5    # 20 pillars  (open)
```

An interrupted model (`ppo_drone_stage{N}_interrupted.zip`) is always resumed automatically — no flag needed.

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
    ppo_config.yaml         # PPO hyperparams + curriculum table
  envs/
    drone_env.py            # DroneObstacleEnv (gym.Env wrapper)
    policy.py               # DepthStateExtractor (CNN + MLP)
    monitor.py              # TrainingMonitor SB3 callback
    manager/
      train_manager.py      # Step/reset orchestrator
      action_manager.py     # Velocity smoothing, takeoff assist
      reward_manager.py     # Per-step reward computation
      reset_manager.py      # Reset strategy classification + dispatch
      pillar_manager.py     # Pillar spawn, subgoals, collision
      logging_manager.py    # Per-env step/episode file logging
  utils/
    bridge_factory.py       # ROSBridge (rclpy Node)
    process_utils.py        # Gazebo/PX4/MicroXRCE process launchers
    px4_manager.py          # Per-rank PX4 SITL instance manager
    gz_transport_client.py  # Native Gazebo Transport client
```

### Data flow

```
PPO.learn()
  → SubprocVecEnv (n_envs subprocesses)
    → DroneObstacleEnv.step(action)
      → TrainManager.step_process(action)
            ActionManager        → smoothed velocity cmd
            ROSBridge.send_velocity()  → PX4 offboard setpoint
            ROSBridge.tick(dt)         → rclpy spin loop
            ROSBridge.get_*()          → position, depth, velocity, yaw
            PillarManager.update()     → collision, subgoal, attention
            RewardManager.compute()    → scalar reward
            _check_terminal()          → (terminated, truncated, reason)
```

---

## Observation & Action Space

**Observation** — dict space:

| Key | Shape | Description |
|---|---|---|
| `depth` | `(3, 84, 84)` float32 | 3 stacked depth frames, clipped to [0, 10] m |
| `state` | `(20,)` float32 | ego-centric state vector (see below) |

State vector layout:
```
[0:3]   rel_goal_xy[x,y], rel_goal_z
[3:6]   velocity [vx, vy, vz]
[6]     altitude
[7]     yaw
[8]     dist_xy to goal
[9:18]  depth sector features (3 sectors × min/mean/free_frac)
[18:20] sin/cos of yaw error to goal
```

**Action** — `(4,)` float32 in `[-1, 1]`: `[vx, vy, vz, yaw_rate]`
Scaled by velocity limits in `EnvConfig`, smoothed with α=0.35.

---

## Reward Structure

All coefficients live in `configs/reward_config.py`.

| Component | Description |
|---|---|
| `progress` | XY distance reduction to goal |
| `velocity_goal` | Speed aligned toward goal |
| `yaw_align` | Yaw error penalty |
| `terminal` | One-shot reward at episode end |
| `lateral` | Fence proximity penalty |
| `near_fence` | Per-step soft fence penalty |
| `smooth` | Action smoothness |
| `time` | Escalating step penalty |
| `pillar_clearance_soft` | Soft clearance from pillars |
| `collision_course` | Approach-vector danger |

**Terminal rewards:**

| Reason | Reward |
|---|---|
| `goal_xy` | `+130` |
| `collision` | `-280` |
| `out_of_fence` | `-150` |
| `fell_to_ground` | `-200` |
| `flipped` | `-200` |
| `max_steps` (near fence) | base + `0.8 × out_of_fence_penalty` |

---

## Curriculum

| Stage | Pillars | Min steps | Notes |
|---|---|---|---|
| 0 | 0 | — | vz frozen, horizontal nav only |
| 1 | 0 | 500k | Full 4D control |
| 2 | 2 | 500k | |
| 3 | 5 | 500k | |
| 4 | 10 | 500k | |
| 5 | 20 | open | |

Goal acceptance radius and spawn distance ramp automatically over training steps.

---

## Reset Strategy

`ResetManager` classifies each episode end into:

| Mode | When |
|---|---|
| `continuous` | Normal in-bounds terminal (goal, collision, etc.) |
| `rescue_then_continuous` | Drone near/outside fence — velocity-driven back in |
| `hard` | Sim failure (failsafe, EKF dead) — full disarm → teleport → rearm |
| `startup_arm` | First episode only |

---

## Multi-Env Isolation

Each of `n_envs` parallel environments gets dedicated:
- `ROS_DOMAIN_ID = 30 + rank`
- `GZ_PARTITION = drone_rl_{rank}`
- `PX4_INSTANCE = rank`, `UXRCE_DDS_PORT = 8888 + rank`
- Per-rank SDF world file with patched world name
- 15s startup stagger between ranks

---

## Coordinate System

| Frame | Convention |
|---|---|
| Gazebo | ENU (East-North-Up) |
| PX4 | NED (North-East-Down) |
| `ROSBridge.send_velocity()` | **input ENU** → internally converts to NED |
| `ROSBridge.get_linear_velocity()` | **returns ENU** |

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

File này cho phép tùy chỉnh CPU allocation theo từng máy. Mặc định tắt toàn bộ pinning để đảm bảo ổn định.

```yaml
# configs/machine_config.yaml

pin_processes: false         # true = dùng sched_setaffinity để pin PX4 + bridges
                             # CẢNH BÁO: KHÔNG pin Python worker — chứa keepalive 5Hz

main_process_cores: "0-3"   # cores dành cho main PPO process (PyTorch)
torch_num_threads: 4         # intra-op threads (matrix ops) — luôn active khi > 0
torch_interop_threads: 2     # inter-op threads — luôn active khi > 0

cores_per_env: 6             # số cores cấp cho mỗi rank khi pin_processes=true
                             # rank 0 → cores 4-9, rank 1 → cores 10-15, ...
```

### Các thành phần và tác dụng

| Param | Khi nào có tác dụng | Ghi chú |
|---|---|---|
| `torch_num_threads` | Luôn luôn (nếu > 0) | Giới hạn PyTorch thread của main PPO process |
| `torch_interop_threads` | Luôn luôn (nếu > 0) | Giới hạn inter-op parallelism của PyTorch |
| `pin_processes` | Khi `true` | Pin PX4 + bridge processes vào dedicated core group |
| `main_process_cores` | Khi `pin_processes=true` | Pin main PPO process vào core range này |
| `cores_per_env` | Khi `pin_processes=true` | Cores cấp cho PX4 + bridges mỗi rank |

### Layout core cho máy 24 core

```
cores 0-3   → main PPO process (PyTorch training)
cores 4-9   → rank 0: PX4 + 4 bridges
cores 10-15 → rank 1: PX4 + 4 bridges
cores 16-21 → rank 2: PX4 + 4 bridges  (nếu n_envs=3)
cores 22-23 → OS / processes khác
```

### Tại sao không pin Python worker?

Python worker subprocess chứa:
- **Keepalive thread** — gửi velocity setpoint mỗi 0.2s, nếu miss → PX4 vào failsafe
- **ROS spin thread** (MultiThreadedExecutor) — nhận PX4 telemetry callbacks

Nếu pin worker vào cùng 6 cores với PX4 + 4 bridges, các thread này bị CPU starvation → `KEEPALIVE STALE HOVER` → mất ổn định. Worker luôn chạy tự do theo OS scheduler.

### Khuyến nghị theo máy

| Số core | Cấu hình |
|---|---|
| 8 core | `pin_processes: false`, `torch_num_threads: 2` |
| 16 core | `torch_num_threads: 4`, có thể thử `pin_processes: true`, `cores_per_env: 4` |
| 24 core | `torch_num_threads: 4`, `pin_processes: true` (cấu hình mặc định trong file) |

---

## Syntax Check

```bash
source ~/drone_rl_env/bin/activate
python3 -m py_compile obstacle_avoidance/train.py
python3 -m py_compile obstacle_avoidance/envs/drone_env.py
python3 -m py_compile obstacle_avoidance/utils/bridge_factory.py
```
