"""OpenVINS adapter for PX4 external-vision state estimation.

This is the state_estimation module used by the HALO pipeline:

1. PX4 ``SensorCombined`` -> ROS ``sensor_msgs/Imu`` for OpenVINS.
2. OpenVINS ``nav_msgs/Odometry`` -> PX4 ``VehicleOdometry`` for EKF2.
3. A small health/state API for reset logic and HALO world-frame consumers.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from nav_msgs.msg import Odometry
from px4_msgs.msg import SensorCombined, VehicleOdometry
from sensor_msgs.msg import Imu

from .config import OpenVinsConfig
from .frames import (
    covariance3,
    diag3,
    enu_position_to_ned,
    enu_velocity_to_ned,
    frd_vector_to_flu,
    orientation_diag3,
    quat_enu_flu_to_ned_frd,
    quat_normalize_wxyz,
)


@dataclass
class VioStatus:
    healthy: bool
    sample_count: int
    imu_count: int
    last_sample_age_s: float | None
    reset_generation: int
    odom_topic: str
    imu_topic: str


@dataclass
class _VioSample:
    stamp_wall: float
    stamp_ros_us: int
    position_enu: np.ndarray
    q_enu_flu: list[float]
    velocity_enu: np.ndarray
    position_variance: list[float]
    orientation_variance: list[float]
    velocity_variance: list[float]


def _stamp_to_us(stamp) -> int:
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    if sec == 0 and nanosec == 0:
        return 0
    return sec * 1_000_000 + nanosec // 1000


def _set_ros_stamp_from_us(stamp, stamp_us: int) -> None:
    stamp.sec = int(stamp_us // 1_000_000)
    stamp.nanosec = int((stamp_us % 1_000_000) * 1000)


class OpenVinsPx4Bridge:
    """Runtime bridge between OpenVINS and PX4 EKF2.

    The bridge attaches to an existing rclpy node so it shares the same
    executor, ROS_DOMAIN_ID, namespace, and sim-time settings as the training
    environment. It deliberately has no PX4 source edits.
    """

    def __init__(
        self,
        *,
        node,
        px4_topic: Callable[[str], str],
        qos,
        logger,
        config: OpenVinsConfig | None = None,
    ) -> None:
        self.node = node
        self._px4_topic = px4_topic
        self.qos = qos
        self.logger = logger
        self.config = config or OpenVinsConfig.from_env()

        self._lock = threading.Lock()
        self._latest: Optional[_VioSample] = None
        self._reset_generation = 0
        self._sample_count = 0
        self._imu_count = 0
        self._last_status_log_wall = 0.0

        self.odom_sub = self.node.create_subscription(
            Odometry,
            self.config.odom_topic,
            self._odom_cb,
            self.qos,
        )

        self.imu_pub = None
        self.sensor_combined_sub = None
        if self.config.publish_imu_from_px4:
            self.imu_pub = self.node.create_publisher(Imu, self.config.imu_topic, self.qos)
            self.sensor_combined_sub = self.node.create_subscription(
                SensorCombined,
                self._px4_topic("/fmu/out/sensor_combined"),
                self._sensor_combined_cb,
                self.qos,
            )

        self.logger.info(
            "[STATE_ESTIMATION:OPENVINS] enabled "
            f"odom={self.config.odom_topic} imu={self.config.imu_topic} "
            f"stereo={self.config.use_stereo} max_cameras={self.config.max_cameras} "
            f"timeout={self.config.timeout_s:.3f}s "
            f"imu_from_px4={self.config.publish_imu_from_px4}"
        )

    def note_reset(self) -> None:
        """Drop stale VIO samples after a simulator teleport/reset boundary."""

        with self._lock:
            self._latest = None
            self._reset_generation += 1
        self.logger.info(
            "[STATE_ESTIMATION:OPENVINS] reset boundary "
            f"generation={self._reset_generation}"
        )

    def healthy(self, max_age_s: Optional[float] = None) -> bool:
        return self.get_position_enu(max_age_s=max_age_s) is not None

    def status(self) -> VioStatus:
        with self._lock:
            sample = self._latest
            age = None if sample is None else time.monotonic() - sample.stamp_wall
            sample_count = self._sample_count
            imu_count = self._imu_count
            reset_generation = self._reset_generation
        healthy = age is not None and age <= self.config.timeout_s
        return VioStatus(
            healthy=healthy,
            sample_count=sample_count,
            imu_count=imu_count,
            last_sample_age_s=age,
            reset_generation=reset_generation,
            odom_topic=self.config.odom_topic,
            imu_topic=self.config.imu_topic,
        )

    def maybe_log_status(self, period_s: float = 5.0) -> None:
        now = time.monotonic()
        if now - self._last_status_log_wall < period_s:
            return
        self._last_status_log_wall = now
        status = self.status()
        age = "none" if status.last_sample_age_s is None else f"{status.last_sample_age_s:.3f}s"
        self.logger.info(
            "[STATE_ESTIMATION:OPENVINS] "
            f"healthy={status.healthy} samples={status.sample_count} "
            f"imu={status.imu_count} age={age}"
        )

    def get_position_enu(self, max_age_s: Optional[float] = None) -> Optional[np.ndarray]:
        max_age = self.config.timeout_s if max_age_s is None else float(max_age_s)
        with self._lock:
            sample = self._latest
            if sample is None:
                return None
            if time.monotonic() - sample.stamp_wall > max_age:
                return None
            return sample.position_enu.copy()

    def get_velocity_enu(self, max_age_s: Optional[float] = None) -> Optional[np.ndarray]:
        max_age = self.config.timeout_s if max_age_s is None else float(max_age_s)
        with self._lock:
            sample = self._latest
            if sample is None:
                return None
            if time.monotonic() - sample.stamp_wall > max_age:
                return None
            return sample.velocity_enu.copy()

    def build_vehicle_odometry(
        self,
        *,
        timestamp_us: int,
        reset_counter: int,
        force_zero_velocity: bool = False,
        reset_variance: bool = False,
    ) -> Optional[VehicleOdometry]:
        with self._lock:
            sample = self._latest
            if sample is None:
                return None
            if time.monotonic() - sample.stamp_wall > self.config.timeout_s:
                return None
            position_enu = sample.position_enu.copy()
            q_enu_flu = list(sample.q_enu_flu)
            velocity_enu = sample.velocity_enu.copy()
            position_variance = list(sample.position_variance)
            orientation_variance = list(sample.orientation_variance)
            velocity_variance = list(sample.velocity_variance)
            sample_stamp_us = int(sample.stamp_ros_us)

        msg = VehicleOdometry()
        msg.timestamp = int(timestamp_us)
        msg.timestamp_sample = sample_stamp_us if sample_stamp_us > 0 else int(timestamp_us)
        msg.pose_frame = VehicleOdometry.POSE_FRAME_NED
        msg.position = enu_position_to_ned(position_enu)
        msg.q = quat_enu_flu_to_ned_frd(q_enu_flu)

        msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
        msg.velocity = [0.0, 0.0, 0.0] if force_zero_velocity else enu_velocity_to_ned(velocity_enu)
        msg.angular_velocity = [0.0, 0.0, 0.0] if self.config.forward_angular_velocity else [math.nan, math.nan, math.nan]

        if reset_variance:
            msg.position_variance = [
                max(position_variance[0], 0.05),
                max(position_variance[1], 0.05),
                max(position_variance[2], 0.10),
            ]
        else:
            msg.position_variance = position_variance
        msg.orientation_variance = orientation_variance
        msg.velocity_variance = velocity_variance
        msg.reset_counter = int(reset_counter) % 256
        msg.quality = 100
        return msg

    def _sensor_combined_cb(self, msg: SensorCombined) -> None:
        if self.imu_pub is None:
            return

        gyro = np.asarray(msg.gyro_rad, dtype=np.float64)
        accel = np.asarray(msg.accelerometer_m_s2, dtype=np.float64)
        if gyro.shape != (3,) or accel.shape != (3,):
            return
        if not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(accel)):
            return

        imu = Imu()
        stamp_us = int(getattr(msg, "timestamp_sample", 0) or getattr(msg, "timestamp", 0))
        if self.config.imu_stamp_source == "px4" and stamp_us > 0:
            _set_ros_stamp_from_us(imu.header.stamp, stamp_us)
        else:
            imu.header.stamp = self.node.get_clock().now().to_msg()
        imu.header.frame_id = self.config.imu_frame

        gyro_flu = frd_vector_to_flu(gyro)
        accel_flu = frd_vector_to_flu(accel)
        imu.angular_velocity.x = gyro_flu[0]
        imu.angular_velocity.y = gyro_flu[1]
        imu.angular_velocity.z = gyro_flu[2]
        imu.linear_acceleration.x = accel_flu[0]
        imu.linear_acceleration.y = accel_flu[1]
        imu.linear_acceleration.z = accel_flu[2]

        imu.angular_velocity_covariance = covariance3(self.config.gyro_variance)
        imu.linear_acceleration_covariance = covariance3(self.config.accel_variance)
        imu.orientation_covariance[0] = -1.0

        self.imu_pub.publish(imu)
        with self._lock:
            self._imu_count += 1

    def _odom_cb(self, msg: Odometry) -> None:
        pos = np.array(
            [
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                float(msg.pose.pose.position.z),
            ],
            dtype=np.float32,
        )
        vel = np.array(
            [
                float(msg.twist.twist.linear.x),
                float(msg.twist.twist.linear.y),
                float(msg.twist.twist.linear.z),
            ],
            dtype=np.float32,
        )
        q = [
            float(msg.pose.pose.orientation.w),
            float(msg.pose.pose.orientation.x),
            float(msg.pose.pose.orientation.y),
            float(msg.pose.pose.orientation.z),
        ]

        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
            return
        if not np.all(np.isfinite(q)):
            return

        sample = _VioSample(
            stamp_wall=time.monotonic(),
            stamp_ros_us=_stamp_to_us(msg.header.stamp),
            position_enu=pos,
            q_enu_flu=quat_normalize_wxyz(q),
            velocity_enu=vel,
            position_variance=diag3(msg.pose.covariance, self.config.position_variance),
            orientation_variance=orientation_diag3(
                msg.pose.covariance,
                self.config.orientation_variance,
            ),
            velocity_variance=diag3(msg.twist.covariance, self.config.velocity_variance),
        )
        with self._lock:
            self._latest = sample
            self._sample_count += 1
