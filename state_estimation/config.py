"""Configuration for the GPS-denied VIO state-estimation stack."""

from __future__ import annotations

import os
from dataclasses import dataclass


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class CameraBridgeSpec:
    """One Gazebo Image -> ROS Image bridge used by OpenVINS."""

    label: str
    gz_topic: str
    ros_topic: str
    gz_type: str = "gz.msgs.Image"
    ros_type: str = "sensor_msgs/msg/Image"


@dataclass(frozen=True)
class OpenVinsConfig:
    """All runtime knobs for OpenVINS as this repo's state estimator.

    The defaults are intentionally GPS-denied and SITL-friendly. Hardware runs
    should override the calibration YAML files and, if needed, camera topics.
    """

    project_root: str
    ros_distro: str = "jazzy"
    ros2_ws: str = ""
    openvins_ws: str = ""
    config_path: str = ""

    namespace: str = "/ov_msckf"
    odom_topic: str = "/ov_msckf/odomimu"
    imu_topic: str = "/openvins/imu"
    imu_frame: str = "base_link"
    cam0_topic: str = "/openvins/cam0/image_raw"
    cam1_topic: str = "/openvins/cam1/image_raw"
    gz_cam0_topic: str = "/openvins/cam0/image_raw"
    gz_cam1_topic: str = "/openvins/cam1/image_raw"
    image_gz_type: str = "gz.msgs.Image"

    use_stereo: bool = True
    max_cameras: int = 2
    use_sim_time: bool = True
    publish_rate_hz: float = 30.0
    timeout_s: float = 0.35

    publish_imu_from_px4: bool = True
    imu_stamp_source: str = "px4"
    forward_angular_velocity: bool = False
    gyro_variance: float = 4e-8
    accel_variance: float = 4e-6
    position_variance: float = 0.05
    orientation_variance: float = 0.05
    velocity_variance: float = 0.05

    reset_prime_source: str = "gazebo"
    fallback_to_gazebo_vo: bool = False

    @classmethod
    def from_env(cls, project_root: str | None = None) -> "OpenVinsConfig":
        root = os.path.abspath(
            project_root
            or os.environ.get("UAV_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        )
        external_dir = os.environ.get("EXTERNAL_DIR", os.path.join(root, "external"))
        ros2_ws = os.environ.get("ROS2_WS", os.path.join(external_dir, "ros2_ws"))
        openvins_ws = os.environ.get("OPENVINS_WS", os.path.join(external_dir, "openvins_ws"))
        default_config = os.path.join(root, "configs", "openvins", "estimator_config.yaml")

        use_stereo = env_bool("OPENVINS_USE_STEREO", True)
        max_cameras = env_int("OPENVINS_MAX_CAMERAS", 2 if use_stereo else 1)

        return cls(
            project_root=root,
            ros_distro=os.environ.get("ROS_DISTRO", "jazzy"),
            ros2_ws=ros2_ws,
            openvins_ws=openvins_ws,
            config_path=os.environ.get("OPENVINS_CONFIG_PATH", default_config),
            namespace=os.environ.get("OPENVINS_NAMESPACE", "/ov_msckf"),
            odom_topic=os.environ.get("OPENVINS_ODOM_TOPIC", "/ov_msckf/odomimu"),
            imu_topic=os.environ.get("OPENVINS_IMU_TOPIC", "/openvins/imu"),
            imu_frame=os.environ.get("OPENVINS_IMU_FRAME", "base_link"),
            cam0_topic=os.environ.get("OPENVINS_CAM0_TOPIC", "/openvins/cam0/image_raw"),
            cam1_topic=os.environ.get("OPENVINS_CAM1_TOPIC", "/openvins/cam1/image_raw"),
            gz_cam0_topic=os.environ.get("OPENVINS_GZ_CAM0_TOPIC", "/openvins/cam0/image_raw"),
            gz_cam1_topic=os.environ.get("OPENVINS_GZ_CAM1_TOPIC", "/openvins/cam1/image_raw"),
            image_gz_type=os.environ.get("OPENVINS_GZ_IMAGE_TYPE", "gz.msgs.Image"),
            use_stereo=use_stereo,
            max_cameras=max_cameras,
            use_sim_time=env_bool("OPENVINS_USE_SIM_TIME", True),
            publish_rate_hz=env_float("OPENVINS_PUBLISH_RATE_HZ", 30.0),
            timeout_s=env_float("OPENVINS_TIMEOUT_S", 0.35),
            publish_imu_from_px4=env_bool("OPENVINS_PUBLISH_IMU_FROM_PX4", True),
            imu_stamp_source=os.environ.get("OPENVINS_IMU_STAMP_SOURCE", "px4").strip().lower(),
            forward_angular_velocity=env_bool("OPENVINS_FORWARD_ANGULAR_VELOCITY", False),
            gyro_variance=env_float("OPENVINS_IMU_GYRO_VARIANCE", 4e-8),
            accel_variance=env_float("OPENVINS_IMU_ACCEL_VARIANCE", 4e-6),
            position_variance=env_float("OPENVINS_POSITION_VARIANCE", 0.05),
            orientation_variance=env_float("OPENVINS_ORIENTATION_VARIANCE", 0.05),
            velocity_variance=env_float("OPENVINS_VELOCITY_VARIANCE", 0.05),
            reset_prime_source=os.environ.get("OPENVINS_RESET_PRIME_SOURCE", "gazebo").strip().lower(),
            fallback_to_gazebo_vo=env_bool("OPENVINS_FALLBACK_TO_GAZEBO_VO", False),
        ).validated()

    def validated(self) -> "OpenVinsConfig":
        if self.max_cameras < 1:
            raise ValueError("OPENVINS_MAX_CAMERAS must be >= 1")
        if self.use_stereo and self.max_cameras < 2:
            raise ValueError("OPENVINS_USE_STEREO=true requires OPENVINS_MAX_CAMERAS>=2")
        if self.timeout_s <= 0.0:
            raise ValueError("OPENVINS_TIMEOUT_S must be positive")
        if self.publish_rate_hz < 30.0:
            raise ValueError("PX4 EKF2 expects external vision at 30 Hz or higher")
        if self.imu_stamp_source not in {"px4", "ros"}:
            raise ValueError("OPENVINS_IMU_STAMP_SOURCE must be 'px4' or 'ros'")
        return self

    @property
    def camera_bridge_specs(self) -> list[CameraBridgeSpec]:
        specs = [
            CameraBridgeSpec(
                label="openvins_cam0",
                gz_topic=self.gz_cam0_topic,
                ros_topic=self.cam0_topic,
                gz_type=self.image_gz_type,
            )
        ]
        if self.use_stereo and self.max_cameras >= 2:
            specs.append(
                CameraBridgeSpec(
                    label="openvins_cam1",
                    gz_topic=self.gz_cam1_topic,
                    ros_topic=self.cam1_topic,
                    gz_type=self.image_gz_type,
                )
            )
        return specs
