# State Estimation / VIO

This package owns GPS-denied state estimation for HALO.

The current backend is OpenVINS MSCKF because it is lightweight, filter-based,
ROS 2 compatible, and matches the proposal requirement that HALO receives
`x_t = (p_t, q_t, v_t)` from VIO/state filtering before constructing posed point
sets.

## Runtime Flow

```text
PX4 SensorCombined
  -> OpenVinsPx4Bridge publishes /openvins/imu as ROS FLU Imu

Gazebo camera images
  -> ros_gz_bridge publishes /openvins/cam{0,1}/image_raw

OpenVINS ov_msckf
  -> /ov_msckf/odomimu nav_msgs/Odometry in ROS ENU/FLU

OpenVinsPx4Bridge
  -> /fmu/in/vehicle_visual_odometry px4_msgs/VehicleOdometry in PX4 NED/FRD
```

No PX4 source files are modified. PX4 EKF2 fuses the estimate through the
external-vision input topic.

## Main Files

- `config.py`: all env-driven VIO configuration in one dataclass.
- `frames.py`: ROS ENU/FLU <-> PX4 NED/FRD math, testable without ROS.
- `runtime.py`: OpenVINS process and camera bridge launchers.
- `openvins_px4_bridge.py`: in-process ROS adapter used by `ROSBridge`.

## Important Environment Variables

- `STATE_ESTIMATOR_SOURCE=openvins`
- `OPENVINS_CONFIG_PATH=configs/openvins/estimator_config.yaml`
- `OPENVINS_ODOM_TOPIC=/ov_msckf/odomimu`
- `OPENVINS_IMU_TOPIC=/openvins/imu`
- `OPENVINS_CAM0_TOPIC=/openvins/cam0/image_raw`
- `OPENVINS_CAM1_TOPIC=/openvins/cam1/image_raw`
- `OPENVINS_GZ_CAM0_TOPIC` and `OPENVINS_GZ_CAM1_TOPIC` for Gazebo source topics.
- `OPENVINS_PUBLISH_RATE_HZ=30` or higher for PX4 EKF2 external vision.

For hardware runs, replace the Kalibr YAML files in `configs/openvins/` with
real IMU-camera calibration and point the camera topics at the hardware driver.
