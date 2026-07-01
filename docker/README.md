# Docker for UAV Obstacle Avoidance + OpenVINS

Docker nay duoc thiet ke de chay moi thu trong repo UAV, khong can link PX4 vao `/root` hay link `obstacle_avoidance` vao trong PX4.

Mac Apple Silicon luu y: Docker nay mac dinh build/run `linux/amd64` de tranh mismatch image ROS/PX4/Gazebo. Neu muon thu native ARM64, co the set `DOCKER_PLATFORM=linux/arm64`, nhung nen coi day la duong thu nghiem.

Trong container:

```text
/workspace/
└── obstacle_avoidance/                 # repo UAV, mount tu thu muc cha cua docker/
    ├── train.py
    ├── configs/openvins/
    ├── docker/
    └── external/                       # ignored by git
        ├── PX4-Autopilot/
        ├── ros2_ws/
        │   └── src/px4_msgs/
        └── openvins_ws/
            └── src/open_vins/
```

Package Python duoc import bang:

```bash
PYTHONPATH=/workspace
python -m obstacle_avoidance.train --stage 1
```

## Docker Da Cai San Nhung Gi

- PX4 SITL build dependencies: compiler toolchain, `ccache`, `bc`, `protobuf-compiler`, XML tools, GStreamer, Gazebo Harmonic dependencies.
- OpenVINS dependencies: OpenCV + contrib, Eigen, Boost, Ceres, Glog/Gflags, SuiteSparse.
- ROS 2 Jazzy packages: `cv_bridge`, `image_transport`, `message_filters`, `nav_msgs`, `sensor_msgs`, `geometry_msgs`, `tf2_ros`, `ros_gz_bridge`, `ros_gz_sim`, `ros_gz_interfaces`.
- Gazebo Harmonic Python bindings used by this repo: `gz.transport13`, `gz.msgs10`.
- DRL Python environment in `/opt/drone_rl_env`.
- OpenVINS GPS-denied state-estimation defaults under `state_estimation/`.

## Build Container

From macOS:

```bash
cd /Users/nguyendangkhoa/Documents/UAV-Capstone-Project/UAV-obstacle_avoidance/docker
docker compose up --build -d
```

Kiem tra Dockerfile:

```bash
cd /Users/nguyendangkhoa/Documents/UAV-Capstone-Project/UAV-obstacle_avoidance/docker
docker build --check .
```

Khong chay `docker build --check .` tu root `UAV-obstacle_avoidance/`, vi Dockerfile nam trong folder `docker/`.

Open noVNC:

```text
http://localhost:6080/vnc.html
```

Password:

```text
uavcapstone
```

Shell:

```bash
docker exec -it uav_obstacle_openvins_container bash
```

## Clone And Build Runtime Repos Inside UAV Repo

Inside the container:

```bash
cd /workspace/obstacle_avoidance
bash docker/bootstrap_external.sh
```

Mac path sau khi chay script:

```text
/Users/nguyendangkhoa/Documents/UAV-Capstone-Project/UAV-obstacle_avoidance/external/
```

`external/` da duoc them vao `.gitignore`, nen PX4/OpenVINS/build artifacts khong bi Git track.

Co the override versions:

```bash
PX4_REF=main PX4_MSGS_REF=main OPENVINS_REF=master BUILD_JOBS=2 \
  bash docker/bootstrap_external.sh
```

Dung `BUILD_JOBS=1` neu Docker Desktop bi thieu RAM.

## Run Training

Inside the container:

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/obstacle_avoidance/external/ros2_ws/install/setup.bash
source /workspace/obstacle_avoidance/external/openvins_ws/install/setup.bash

export STATE_ESTIMATOR_SOURCE=openvins
export PX4_ROOT=/workspace/obstacle_avoidance/external/PX4-Autopilot
export ROS2_WS=/workspace/obstacle_avoidance/external/ros2_ws
export OPENVINS_WS=/workspace/obstacle_avoidance/external/openvins_ws
export OPENVINS_CONFIG_PATH=/workspace/obstacle_avoidance/configs/openvins/estimator_config.yaml
export OPENVINS_GZ_CAM0_TOPIC=/openvins/cam0/image_raw
export OPENVINS_GZ_CAM1_TOPIC=/openvins/cam1/image_raw
export OPENVINS_USE_STEREO=1
export OPENVINS_MAX_CAMERAS=2
export OPENVINS_PUBLISH_RATE_HZ=30
export OPENVINS_IMU_STAMP_SOURCE=px4
export STATE_ESTIMATOR_ALLOW_GT_FALLBACK=0
export OPENVINS_FALLBACK_TO_GAZEBO_VO=0

cd /workspace
python -m obstacle_avoidance.train --stage 1
```

`train.py` bay gio doc `PX4_ROOT` tu env. Mac dinh neu khong set, no se tim PX4 o:

```text
/workspace/obstacle_avoidance/external/PX4-Autopilot
```

## Model And Camera Topics

Docker chi dam bao dependency va runtime. OpenVINS van can image topics that tu Gazebo/PX4.

Config hien tai mac dinh:

```text
/openvins/imu
/openvins/cam0/image_raw
/openvins/cam1/image_raw
```

`OPENVINS_CAM*` la ROS topic OpenVINS subscribe. `OPENVINS_GZ_CAM*` la Gazebo source topic ma `ros_gz_bridge` doc. Mac dinh hai cap nay trung nhau de tuong thich voi model da publish dung topic.

Neu model dang chay khong publish du hai camera image topics, OpenVINS stereo se khong healthy. Khi do co 3 cach:

- Dung PX4/Gazebo model da co stereo camera topics.
- Doi OpenVINS config sang mono: `OPENVINS_USE_STEREO=0`, `OPENVINS_MAX_CAMERAS=1`, va map `OPENVINS_GZ_CAM0_TOPIC` vao Gazebo camera topic that.
- Sua PX4/Gazebo model ben ngoai Docker folder de them stereo sensors.

Docker nay khong sua PX4 source/model.

## Useful Checks

Check import/runtime:

```bash
python - <<'PY'
mods = [
    "rclpy",
    "cv_bridge",
    "px4_msgs",
    "sensor_msgs",
    "nav_msgs",
    "geometry_msgs",
    "ros_gz_interfaces",
    "gz.transport13",
    "gz.msgs10",
    "stable_baselines3",
    "torch",
    "cv2",
    "yaml",
]
import importlib.util
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing:", missing)
raise SystemExit(1 if missing else 0)
PY
```

Check OpenVINS executable:

```bash
ros2 pkg executables ov_msckf
```

Check topics:

```bash
ros2 topic list | grep -E 'openvins|ov_msckf|fmu|camera|depth|lidar'
```

Check VIO output:

```bash
ros2 topic echo /ov_msckf/odomimu --once
```

Check PX4 external vision input:

```bash
ros2 topic echo /fmu/in/vehicle_visual_odometry --once
```

## Sources Used

- OpenVINS installation docs: https://docs.openvins.com/gs-installing.html
- PX4 external vision estimation docs: https://docs.px4.io/main/en/ros/external_position_estimation
- PX4 Gazebo simulation docs: https://docs.px4.io/main/en/sim_gazebo_gz/
