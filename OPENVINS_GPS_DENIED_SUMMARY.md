# Tổng hợp tích hợp VIO/OpenVINS cho GPS-denied state estimation

## Kết luận thiết kế

Trong repo này, VIO chính là `state_estimation`. OpenVINS được dùng làm backend
hiện tại vì nhẹ, filter-based, có ROS2 support, và phù hợp với HALO proposal:
HALO cần state estimate `x_t = (p_t, q_t, v_t)` để biến depth/LiDAR thành posed
world-frame point set trước khi tạo BEV occupancy, EDT và kinematic tensor.

PX4 không bị sửa. Kết quả OpenVINS được đưa vào PX4 EKF2 qua external vision:

```text
PX4 SensorCombined -> /openvins/imu
Gazebo images      -> /openvins/cam{0,1}/image_raw
OpenVINS ov_msckf  -> /ov_msckf/odomimu
state_estimation   -> /fmu/in/vehicle_visual_odometry
PX4 EKF2           -> /fmu/out/vehicle_local_position, /fmu/out/vehicle_odometry
```

## Module mới trong `state_estimation/`

- `config.py`: `OpenVinsConfig` gom toàn bộ biến môi trường và validate cấu hình.
- `frames.py`: chuyển đổi ROS ENU/FLU <-> PX4 NED/FRD, test được không cần ROS.
- `runtime.py`: launch camera bridge và OpenVINS node.
- `openvins_px4_bridge.py`: adapter ROS runtime, publish IMU cho OpenVINS và
  publish `VehicleOdometry` cho PX4.
- `README.md`: mô tả cách module chạy và các biến môi trường quan trọng.

`utils/process_utils.py` chỉ còn wrapper tương thích cho các script cũ. Logic VIO
thật nằm trong `state_estimation`.

## Runtime flags chính

```bash
export STATE_ESTIMATOR_SOURCE=openvins
export OPENVINS_CONFIG_PATH=/workspace/obstacle_avoidance/configs/openvins/estimator_config.yaml
export OPENVINS_ODOM_TOPIC=/ov_msckf/odomimu
export OPENVINS_IMU_TOPIC=/openvins/imu
export OPENVINS_CAM0_TOPIC=/openvins/cam0/image_raw
export OPENVINS_CAM1_TOPIC=/openvins/cam1/image_raw
export OPENVINS_GZ_CAM0_TOPIC=/openvins/cam0/image_raw
export OPENVINS_GZ_CAM1_TOPIC=/openvins/cam1/image_raw
export OPENVINS_USE_STEREO=1
export OPENVINS_MAX_CAMERAS=2
export OPENVINS_PUBLISH_RATE_HZ=30
export OPENVINS_IMU_STAMP_SOURCE=px4
export STATE_ESTIMATOR_ALLOW_GT_FALLBACK=0
export OPENVINS_FALLBACK_TO_GAZEBO_VO=0
```

`OPENVINS_CAM*` là ROS topic OpenVINS subscribe. `OPENVINS_GZ_CAM*` là Gazebo
source topic để `ros_gz_bridge` đọc. Nếu dùng mono camera, đặt
`OPENVINS_USE_STEREO=0` và `OPENVINS_MAX_CAMERAS=1`.

## Docker

Docker chỉ nằm trong `UAV-obstacle_avoidance/docker/`, không sửa Docker/PX4 ở
ngoài repo này. Dockerfile đã được cập nhật thêm dependency mà OpenVINS ROS2
khai báo: `rclcpp`, `std_msgs`, `tf2`, `tf2_eigen`, `visualization_msgs`, cùng
OpenCV/Eigen/Ceres/Boost đã có.

Script chính:

```bash
cd /workspace/obstacle_avoidance
bash docker/bootstrap_external.sh
```

Script này clone/build PX4, `px4_msgs`, và OpenVINS trong `external/` của chính
repo UAV obstacle avoidance.

## Kiểm thử đã chạy

```bash
PYTHONPYCACHEPREFIX=/private/tmp/uav_pycache \
  python -m py_compile \
  state_estimation/config.py state_estimation/frames.py state_estimation/runtime.py \
  state_estimation/openvins_px4_bridge.py utils/process_utils.py \
  utils/bridge_factory.py train.py tests/test_state_estimation.py
```

Kết quả: pass.

```bash
bash -n docker/bootstrap_external.sh docker/entrypoint.sh
docker build --check docker/
```

Kết quả: pass.

Do runtime Python của host không có `pytest`, mình chạy thêm smoke test trực
tiếp bằng `assert` cho frame/config logic; kết quả pass. Trong Docker, `pytest`
đã được cài và có thể chạy `python -m pytest tests/test_state_estimation.py -q`.

## Giới hạn cần nhớ

- Calibration trong `configs/openvins/` là bản SITL/OAK-D style khởi tạo, phải
  thay bằng Kalibr thật trước khi bay phần cứng.
- Nếu Gazebo model không publish stereo image topic, OpenVINS sẽ không healthy;
  cần map `OPENVINS_GZ_CAM*` đúng topic thật hoặc chuyển sang mono.
- Reset/teleport simulator vẫn dùng Gazebo pose để điều khiển môi trường. Agent,
  reward và navigation position trong GPS-denied mode ưu tiên PX4 EKF/OpenVINS,
  chỉ fallback Gazebo khi bật rõ `STATE_ESTIMATOR_ALLOW_GT_FALLBACK=1`.
