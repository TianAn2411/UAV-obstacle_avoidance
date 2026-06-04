import time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import numpy as np
import re
import cv2
import subprocess
import math
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logger import setup_logger

from cv_bridge import CvBridge
import threading
import concurrent.futures

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleAttitude,
    VehicleLocalPosition,
    VehicleGlobalPosition,
    VehicleStatus,
    VehicleControlMode,
    VehicleCommandAck,
    VehicleOdometry,
    EstimatorStatusFlags,
    FailsafeFlags,
)
from sensor_msgs.msg import Image, LaserScan
try:
    from geometry_msgs.msg import Pose as GeometryPose
except Exception:
    GeometryPose = None

try:
    from ros_gz_interfaces.msg import EntityPose_V
except Exception:
    EntityPose_V = None
try:
    import gz.transport13 as gz_transport
    from gz.msgs10.pose_v_pb2 import Pose_V as GzPoseV
except Exception:
    gz_transport = None
    GzPoseV = None

try:
    from utils.gz_transport_client import GzTransportClient
except Exception:
    GzTransportClient = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

try:
    import spawn_world
except ImportError:
    spawn_world = None


class NavState:
    MANUAL = 0
    POSCTL = 2
    HOLD = 4
    OFFBOARD = 14
    TAKEOFF = 17
    LAND = 18


class ROSBridge(Node):
    def __init__(
        self,
        gazebo_port,
        world_name="default",
        model_name="x500_depth_0",
        px4_ns="",
        target_system=1,
        gz_partition=None,
    ):
        super().__init__(
            f"drone_bridge_{gazebo_port}",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True),
            ],
        )
        self._use_sim_time = True

        self.world_name = world_name
        self.model_name = model_name
        self.px4_ns = px4_ns.rstrip("/")
        self.target_system = int(target_system)
        self.gz_partition = gz_partition or os.environ.get("GZ_PARTITION", "")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        rank = int(self.target_system) - 1
        log_path = os.path.join(project_root, "runs", "env_logs", f"env_{rank}.txt")
        self.logger = setup_logger(f"BRIDGE_{self.model_name}", log_file=log_path)
        self.gz_client = (
            GzTransportClient(gz_partition=self.gz_partition, use_lock=True, logger=self.logger)
            if GzTransportClient is not None
            else None
        )


        self.logger.info(
            f"[BRIDGE INIT] model={self.model_name} "
            f"world={self.world_name} "
            f"px4_ns={self.px4_ns} "
            f"target_system={self.target_system} "
            f"GZ_PARTITION={self.gz_partition}"
        )

        self.cv_bridge = CvBridge()

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.nav_state = NavState.MANUAL
        self.is_armed = False
        self.preflight_ok = False
        self.offboard_enabled = False
        self.position_enabled = False

        self.current_amsl_alt = 0.0
        self.depth_raw = np.ones((84, 84), dtype=np.float32) * 10.0
        self.lidar_raw = np.ones(1080, dtype=np.float32) * 30.0  # default: max range (no obstacle)

        self.gz_pos = np.zeros(3, dtype=np.float32)
        self.gz_quat = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z] ENU/FLU, identity default
        self.gz_pose_ready = True  # stream spawn-origin pose immediately; overwritten on first callback
        self.gz_pose_stamp = 0.0
        self.gz_pose_source = "default_spawn"
        self._gz_lock = threading.Lock()

        self.px4_lpos = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)  # FRD body rates (rad/s)
        self.px4_vel = np.zeros(3, dtype=np.float32)
        self._px4_vel_ready = False
        self._teleport_zero_vel_countdown = 0
        self._teleport_reset_countdown = 0
        self._vo_reset_counter = 0
        self._last_vo_stamp_us = 0
        self._last_gz_pos_for_vo_vel = None
        self._last_gz_vel_stamp_s = None
        self._vo_velocity_max_m_s = 15.0
        self._vo_publish_lock = threading.Lock()
        self._vo_stop = False
        self._vo_thread = None

        self._last_status_wall = 0.0
        self._last_control_mode_wall = 0.0
        self._last_local_pos_wall = 0.0
        self._last_estimator_flags_wall = 0.0
        self._last_att_wall = 0.0
        self._last_status_px4_ts = 0
        self._last_local_pos_px4_ts = 0
        self._last_estimator_flags_px4_ts = 0
        self._last_estimator_flags_fallback_warn_wall = 0.0
        self._last_failsafe_flags_wall = 0.0
        self._failsafe_flags_msg = None
        self._last_ack_msg = None
        self._last_ack_wall = 0.0
        self._last_preflight_debug_wall = 0.0
        self._last_gz_pose_debug_log = 0.0
        self._last_local_pos_valid = {
            "xy_valid": False,
            "z_valid": False,
            "v_xy_valid": False,
            "v_z_valid": False,
        }

        self._ekf_yaw_align = False
        self._ekf_tilt_align = False
        self._ekf_ev_yaw = False
        self._ekf_ev_pos = False
        self._ekf_ev_hgt = False

        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            self._px4_topic("/fmu/in/offboard_control_mode"),
            self.qos,
        )

        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            self._px4_topic("/fmu/in/trajectory_setpoint"),
            self.qos,
        )

        self.command_pub = self.create_publisher(
            VehicleCommand,
            self._px4_topic("/fmu/in/vehicle_command"),
            self.qos,
        )
        self.vo_pub = self.create_publisher(
            VehicleOdometry,
            self._px4_topic("/fmu/in/vehicle_visual_odometry"),
            self.qos,
        )

        # Dedicated thread streams VIO at 30Hz independent of rclpy spin_once,
        # so PX4 EKF always gets odometry even during blocking gz service calls.
        self._start_vo_thread()

        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self._depth_cb,
            self.qos,
        )
        self.lidar_sub = self.create_subscription(
            LaserScan,
            "/lidar/scan",
            self._lidar_cb,
            self.qos,
        )

        self.att_sub = self.create_subscription(
            VehicleAttitude,
            self._px4_topic("/fmu/out/vehicle_attitude"),
            self._att_cb,
            self.qos,
        )

        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            self._px4_topic("/fmu/out/vehicle_local_position"),
            self._local_pos_cb,
            self.qos,
        )

        self.create_subscription(
            VehicleOdometry,
            f"{self._px4_ns}/fmu/out/vehicle_odometry",
            self._odom_cb,
            self.qos,
        )

        self.global_pos_sub = self.create_subscription(
            VehicleGlobalPosition,
            self._px4_topic("/fmu/out/vehicle_global_position"),
            self._global_pos_cb,
            self.qos,
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            self._px4_topic("/fmu/out/vehicle_status"),
            self._status_cb,
            self.qos,
        )

        self.control_mode_sub = self.create_subscription(
            VehicleControlMode,
            self._px4_topic("/fmu/out/vehicle_control_mode"),
            self._control_mode_cb,
            self.qos,
        )

        self.ack_sub = self.create_subscription(
            VehicleCommandAck,
            self._px4_topic("/fmu/out/vehicle_command_ack"),
            self._ack_cb,
            self.qos,
        )

        self.est_flags_sub = self.create_subscription(
            EstimatorStatusFlags,
            self._px4_topic("/fmu/out/estimator_status_flags"),
            self._estimator_flags_cb,
            self.qos,
        )

        self.failsafe_flags_sub = self.create_subscription(
            FailsafeFlags,
            self._px4_topic("/fmu/out/failsafe_flags"),
            self._failsafe_flags_cb,
            self.qos,
        )

        self.keepalive_enabled = False
        self.keepalive_period = float(os.environ.get("KEEPALIVE_PERIOD", "0.05"))  # 20Hz default
        self.keepalive_stale_time = float(os.environ.get("KEEPALIVE_STALE_TIME", "0.6"))
        self.keepalive_stale_hold_after_s = float(os.environ.get("KEEPALIVE_STALE_HOLD_AFTER_S", "30.0"))
        self.allow_keepalive_position_stale = False
        self.keepalive_target_alt = 2.8
        self.keepalive_alt_deadband = 0.25
        self.keepalive_z_kp = 0.35
        self.keepalive_max_descend_vz = 0.35
        self.keepalive_max_ascend_vz = 0.25

        self._last_setpoint_lock = threading.Lock()
        self._ros_pub_lock = threading.Lock()
        self._last_setpoint_time = 0.0
        self._last_setpoint_mode = "velocity"

        self._last_velocity_cmd = (0.0, 0.0, 0.0, 0.0)
        self._last_position_cmd = (0.0, 0.0, 0.0, math.nan)

        self._stall_pos_locked = False
        self._stall_pos = (0.0, 0.0, 0.0)
        self.last_keepalive_stale_age = 0.0
        self._last_keepalive_log_time = 0.0

        self._keepalive_thread = None
        self._keepalive_stop = False
        self._start_offboard_keepalive_thread()

        self._start_gz_pose_listener()

    def enable_offboard_keepalive(self, enabled=False):
        self.keepalive_enabled = bool(enabled)
        if not enabled:
            self._stall_pos_locked = False
            self.last_keepalive_stale_age = 0.0
        # print(f"[KEEPALIVE] enabled={self.keepalive_enabled} model={self.model_name}")

    def _start_vo_thread(self):
        if self._vo_thread is not None and self._vo_thread.is_alive():
            return
        self._vo_stop = False
        self._vo_thread = threading.Thread(
            target=self._vo_thread_worker,
            name=f"VIOPublisher_{self.model_name}",
            daemon=True,
        )
        self._vo_thread.start()

    def _vo_thread_worker(self):
        """Publish VehicleOdometry at 30Hz from a dedicated thread.

        Independent of rclpy spin_once so VIO never drops during blocking gz
        service RPCs (teleport, pillar spawn) or CPU-starved spin delays.
        Serialized with explicit burst callers via _vo_publish_lock.
        """
        period = 1.0 / 30.0
        next_t = time.monotonic()
        while not self._vo_stop:
            try:
                with self._vo_publish_lock:
                    self._publish_visual_odometry()
            except Exception as exc:
                try:
                    self.logger.warning(f"[VO THREAD] publish failed: {exc}")
                except Exception:
                    pass
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            elif sleep_s < -period:
                next_t = time.monotonic()

    def _start_offboard_keepalive_thread(self):
        if self._keepalive_thread is not None:
            return

        def _worker():
            self.logger.info(f"[KEEPALIVE] thread started model={self.model_name}")
            while not self._keepalive_stop:
                try:
                    if not self.keepalive_enabled:
                        time.sleep(self.keepalive_period)
                        continue

                    if getattr(self, "is_armed", False) is False:
                        time.sleep(self.keepalive_period)
                        continue

                    now = self.get_clock().now().nanoseconds * 1e-9

                    with self._last_setpoint_lock:
                        mode = self._last_setpoint_mode
                        last_t = float(self._last_setpoint_time)
                        vel_cmd = self._last_velocity_cmd
                        pos_cmd = self._last_position_cmd

                    age = now - last_t if last_t > 0.0 else float("inf")

                    # PRIORITY: Action fresh luôn được ưu tiên, reset hold ngay
                    if age <= self.keepalive_stale_time:
                        # Nếu vừa thoát khỏi hold, log resume
                        was_locked = self._stall_pos_locked
                        self._stall_pos_locked = False
                        self.last_keepalive_stale_age = 0.0

                        if was_locked and now - self._last_keepalive_log_time >= 1.0:
                            self.logger.info(
                                f"[KEEPALIVE FRESH RESUME] model={self.model_name} "
                                f"age={age:.3f}s mode={mode}"
                            )
                            self._last_keepalive_log_time = now

                        # DO NOT publish trajectory here - env already published fresh action
                        # Keepalive must not compete with env when action is fresh
                        time.sleep(self.keepalive_period)
                        continue

                    # Chỉ vào stale logic khi age > keepalive_stale_time
                    self.last_keepalive_stale_age = age

                    # Giai đoạn 2: stale nhưng chưa đến threshold hold, stream hover velocity để giữ Offboard
                    if age < self.keepalive_stale_hold_after_s:
                        if mode == "velocity":
                            # Không dùng vel_cmd cũ, thay bằng hover velocity NED
                            vx = 0.0
                            vy = 0.0
                            vz = 0.0
                            yr = 0.0

                            # Z safety: chỉ can thiệp khi altitude thấp nguy hiểm
                            cur_z_ned = float(self.px4_lpos[2])
                            cur_alt = -cur_z_ned if np.isfinite(cur_z_ned) else 0.0

                            target_alt = self.keepalive_target_alt
                            deadband = self.keepalive_alt_deadband
                            kp = self.keepalive_z_kp
                            max_down = self.keepalive_max_descend_vz  # NED: dương là hạ xuống
                            max_up = self.keepalive_max_ascend_vz     # NED: âm là bay lên

                            alt_err = cur_alt - target_alt

                            if cur_alt < 1.2:
                                # Quá thấp: ép leo nhẹ để không rơi.
                                vz = -max_up
                            elif alt_err > deadband:
                                # Quá cao: hạ từ từ.
                                vz = min(max_down, kp * alt_err)
                            elif alt_err < -deadband:
                                # Thấp hơn target: leo nhẹ.
                                vz = -min(max_up, kp * (-alt_err))
                            else:
                                # Trong vùng hợp lý: hover.
                                vz = 0.0
                            self._publish_velocity_setpoint(vx, vy, 0.0, yr)

                            # Log throttle: tối đa 2.5 giây 1 lần
                            if now - self._last_keepalive_log_time >= 3.0:
                                self.logger.warning(
                                    f"[KEEPALIVE STALE HOVER] model={self.model_name} "
                                    f"age={age:.3f}s mode={mode} cur_alt={cur_alt:.2f}m "
                                    f"target_alt={target_alt:.2f}m alt_err={alt_err:.2f} "
                                    f"hover_cmd=({vx:.2f},{vy:.2f},{vz:.2f},{yr:.2f})"
                                )
                                self._last_keepalive_log_time = now
                        else:
                            # Mode position: chỉ publish nếu allow_keepalive_position_stale=True
                            if self.allow_keepalive_position_stale:
                                x, y, z, yaw = pos_cmd
                                self._publish_position_setpoint_ned(x, y, z, yaw)

                    # Giai đoạn 3: stale quá lâu, chuyển position hold
                    else:
                        if not self._stall_pos_locked:
                            hold_x = float(self.px4_lpos[0])
                            hold_y = float(self.px4_lpos[1])
                            hold_z = float(self.px4_lpos[2])  # PX4 local NED: z âm hơn là cao hơn

                            # Nếu local position không hợp lệ hoặc z quá thấp, ép giữ độ cao an toàn.
                            # NED: -1.5 nghĩa là khoảng 1.5m trên mốc local.
                            if (
                                not np.isfinite(hold_x)
                                or not np.isfinite(hold_y)
                                or not np.isfinite(hold_z)
                            ):
                                hold_x = 0.0
                                hold_y = 0.0
                                hold_z = -1.5
                            elif hold_z > -1.0:
                                hold_z = -1.5

                            self._stall_pos = (hold_x, hold_y, hold_z)
                            self._stall_pos_locked = True

                        x, y, z = self._stall_pos
                        self._publish_position_setpoint_ned(x, y, z, math.nan)

                        # Log throttle: tối đa 1 giây 1 lần
                        if now - self._last_keepalive_log_time >= 2.5:
                            cur_z_ned = float(self.px4_lpos[2])
                            cur_alt = -cur_z_ned if np.isfinite(cur_z_ned) else 0.0
                            self.logger.warning(
                                f"[KEEPALIVE STALE HOLD] model={self.model_name} "
                                f"age={age:.3f}s mode={mode} cur_alt={cur_alt:.2f}m "
                                f"hold_pos=({x:.2f},{y:.2f},{z:.2f})"
                            )
                            self._last_keepalive_log_time = now

                except Exception as exc:
                    self.logger.error(f"[KEEPALIVE] exception model={self.model_name}: {exc}")

                time.sleep(self.keepalive_period)

        self._keepalive_thread = threading.Thread(target=_worker, daemon=True)
        self._keepalive_thread.start()


    def prime_visual_odometry_after_reset(
        self,
        duration: float = 2.0,
        zero_velocity: bool = True,
        reset_counter: bool = False,
    ):
        """
        Force-publish Gazebo pose as VehicleOdometry for a short period after
        PX4 reset/teleport so EKF initializes from the current Gazebo pose.
        """
        if reset_counter:
            self.notify_ekf_teleport(prime_count=30, reset_count=100)

        if zero_velocity:
            self._last_gz_pos_for_vo_vel = None
            self._last_gz_vel_stamp_s = None
            self._teleport_zero_vel_countdown = max(
                int(getattr(self, "_teleport_zero_vel_countdown", 0)),
                int(duration * 50),
            )

        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            try:
                with self._vo_publish_lock:
                    self._publish_visual_odometry()
            except Exception as exc:
                self.get_logger().warning(f"[EV PRIME] publish failed: {exc}")
            self.tick(0.02)

    def close(self):
        self.keepalive_enabled = False
        self._keepalive_stop = True
        self._stall_pos_locked = False
        self.last_keepalive_stale_age = 0.0
        self._vo_stop = True

        if self._keepalive_thread is not None:
            try:
                self._keepalive_thread.join(timeout=1.0)
            except Exception:
                pass

        if self._vo_thread is not None:
            try:
                self._vo_thread.join(timeout=0.5)
            except Exception:
                pass

        try:
            self.destroy_node()
        except Exception:
            pass

    # ============================================================
    # Env / topic helpers
    # ============================================================

    def _gz_env(self):
        env = os.environ.copy()

        if self.gz_partition:
            env["GZ_PARTITION"] = str(self.gz_partition)

        return env

    def _invalidate_gz_pose_cache(self):
        # Keep gz_pose_ready=True so VIO never stops during teleport — stream last known pose.
        with self._gz_lock:
            self.gz_pose_stamp = 0.0
            self.gz_pose_source = "stale_pre_teleport"

    def _px4_topic(self, topic):
        if self.px4_ns:
            return f"{self.px4_ns}{topic}"
        return topic

    # ============================================================
    # PX4 callbacks
    # ============================================================

    def _status_cb(self, msg):
        # Debug: log first few callbacks to verify subscription works
        if not hasattr(self, '_status_cb_count'):
            self._status_cb_count = 0
        self._status_cb_count += 1
        if self._status_cb_count <= 3:
            self.logger.info(f"[DEBUG] _status_cb called (count={self._status_cb_count}) model={self.model_name} ns={self.px4_ns}")

        old_preflight_ok = bool(self.preflight_ok)
        self._last_status_wall = time.monotonic()
        self._last_status_px4_ts = int(getattr(msg, "timestamp", 0))
        self.nav_state = msg.nav_state
        self.preflight_ok = bool(msg.pre_flight_checks_pass)
        self.is_armed = msg.arming_state == 2
        self.arming_state = int(getattr(msg, "arming_state", -1))
        self.latest_arming_reason = int(getattr(msg, "latest_arming_reason", -1))
        self.latest_disarming_reason = int(
            getattr(msg, "latest_disarming_reason", -1)
        )
        self.failure_detector_status = int(
            getattr(msg, "failure_detector_status", 0)
        )
        self.failsafe = bool(getattr(msg, "failsafe", False))
        self.failsafe_defer_state = int(getattr(msg, "failsafe_defer_state", -1))
        self.safety_off = bool(getattr(msg, "safety_off", False))
        self.gcs_connection_lost = bool(getattr(msg, "gcs_connection_lost", False))
        if not self.preflight_ok and (
            old_preflight_ok
            or time.monotonic() - self._last_preflight_debug_wall > 2.0
        ):
            self.log_preflight_debug(
                context="[PX4 STATUS PREFLIGHT FALSE]",
                force=True,
            )

    def _control_mode_cb(self, msg):
        self._last_control_mode_wall = time.monotonic()
        self.offboard_enabled = bool(msg.flag_control_offboard_enabled)
        self.position_enabled = bool(msg.flag_control_position_enabled)

    def _ack_cb(self, msg):
        self._last_ack_wall = time.monotonic()
        self._last_ack_msg = msg
        if msg.result != 0:
            self.logger.warning(
                f"[PX4 ACK] "
                f"model={self.model_name} "
                f"ns={self.px4_ns} "
                f"target_system={self.target_system} "
                f"command={msg.command} rejected "
                f"result={msg.result}({self._vehicle_command_result_name(msg.result)}) "
                f"result_param1={getattr(msg, 'result_param1', None)} "
                f"result_param2={getattr(msg, 'result_param2', None)}"
            )
            if int(getattr(msg, "command", -1)) == int(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
            ):
                self.log_preflight_debug(
                    context="[PX4 ACK ARM REJECT DEBUG]",
                    force=True,
                )

    def _att_cb(self, msg):
        self._last_att_wall = time.monotonic()
        q = msg.q
        w, x, y, z = q[0], q[1], q[2], q[3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        self.roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            self.pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            self.pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def _estimator_flags_cb(self, msg):
        self._last_estimator_flags_wall = time.monotonic()
        self._last_estimator_flags_px4_ts = int(getattr(msg, "timestamp", 0))
        self._ekf_yaw_align = bool(getattr(msg, "cs_yaw_align", False))
        self._ekf_tilt_align = bool(getattr(msg, "cs_tilt_align", False))
        self._ekf_ev_yaw = bool(getattr(msg, "cs_ev_yaw", False))
        self._ekf_ev_pos = bool(getattr(msg, "cs_ev_pos", False))
        self._ekf_ev_hgt = bool(getattr(msg, "cs_ev_hgt", False))

    def _failsafe_flags_cb(self, msg):
        self._last_failsafe_flags_wall = time.monotonic()
        self._failsafe_flags_msg = msg

    def _depth_cb(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        img_float = img.astype(np.float32)
        depth = np.nan_to_num(img_float, nan=10.0, posinf=10.0, neginf=10.0)
        depth[depth <= 0.0] = 10.0
        depth = np.clip(depth, 0.0, 10.0)
        if depth.shape == (84, 84):
            self.depth_raw = depth.astype(np.float32, copy=False)
        else:
            self.depth_raw = cv2.resize(
                depth,
                (84, 84),
            ).astype(np.float32)

    def _local_pos_cb(self, msg):
        # Debug: log first few callbacks to verify subscription works
        if not hasattr(self, '_local_pos_cb_count'):
            self._local_pos_cb_count = 0
        self._local_pos_cb_count += 1
        if self._local_pos_cb_count <= 3:
            self.logger.info(f"[DEBUG] _local_pos_cb called (count={self._local_pos_cb_count}) model={self.model_name} ns={self.px4_ns}")

        self._last_local_pos_wall = time.monotonic()
        self._last_local_pos_px4_ts = int(getattr(msg, "timestamp", 0))
        self._last_local_pos_valid = {
            "xy_valid": bool(getattr(msg, "xy_valid", False)),
            "z_valid": bool(getattr(msg, "z_valid", False)),
            "v_xy_valid": bool(getattr(msg, "v_xy_valid", False)),
            "v_z_valid": bool(getattr(msg, "v_z_valid", False)),
            "heading_good_for_control": bool(
                getattr(msg, "heading_good_for_control", False)
            ),
        }
        self.px4_lpos = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.px4_vel = np.array([msg.vx, msg.vy, msg.vz], dtype=np.float32)
        self._px4_vel_ready = True

    def _odom_cb(self, msg):
        # angular_velocity is FRD body frame (rad/s); NaN when EKF not ready
        raw = np.array(msg.angular_velocity, dtype=np.float32)
        if np.any(np.isnan(raw)):
            return  # keep last known-good value
        self.angular_velocity = raw

    def _global_pos_cb(self, msg):
        self.current_amsl_alt = msg.alt

    def _vehicle_command_result_name(self, result):
        names = {
            0: "ACCEPTED",
            1: "TEMPORARILY_REJECTED",
            2: "DENIED",
            3: "UNSUPPORTED",
            4: "FAILED",
            5: "IN_PROGRESS",
            6: "CANCELLED",
        }
        return names.get(int(result), "UNKNOWN")

    def _active_failsafe_flag_names(self):
        msg = self._failsafe_flags_msg
        if msg is None:
            return [], {}

        bool_fields = [
            "angular_velocity_invalid",
            "attitude_invalid",
            "local_altitude_invalid",
            "local_position_invalid",
            "local_position_invalid_relaxed",
            "local_velocity_invalid",
            "global_position_invalid",
            "auto_mission_missing",
            "offboard_control_signal_lost",
            "home_position_invalid",
            "manual_control_signal_lost",
            "gcs_connection_lost",
            "battery_low_remaining_time",
            "battery_unhealthy",
            "geofence_breached",
            "mission_failure",
            "vtol_fixed_wing_system_failure",
            "wind_limit_exceeded",
            "flight_time_limit_exceeded",
            "local_position_accuracy_low",
            "fd_critical_failure",
            "fd_esc_arming_failure",
            "fd_imbalanced_prop",
            "fd_motor_failure",
        ]
        mode_req_fields = [
            "mode_req_angular_velocity",
            "mode_req_attitude",
            "mode_req_local_alt",
            "mode_req_local_position",
            "mode_req_local_position_relaxed",
            "mode_req_global_position",
            "mode_req_mission",
            "mode_req_offboard_signal",
            "mode_req_home_position",
            "mode_req_wind_and_flight_time_compliance",
            "mode_req_prevent_arming",
            "mode_req_manual_control",
            "mode_req_other",
        ]

        active = [name for name in bool_fields if bool(getattr(msg, name, False))]
        mode_req = {
            name: int(getattr(msg, name, 0))
            for name in mode_req_fields
            if int(getattr(msg, name, 0)) != 0
        }
        return active, mode_req

    def log_preflight_debug(self, context="[PREFLIGHT DEBUG]", force=False):
        now = time.monotonic()
        if not force and now - self._last_preflight_debug_wall < 1.0:
            return
        self._last_preflight_debug_wall = now

        active_failsafe, mode_req = self._active_failsafe_flag_names()
        failsafe_age = (
            now - self._last_failsafe_flags_wall
            if self._last_failsafe_flags_wall > 0.0
            else float("inf")
        )
        status_age = (
            now - self._last_status_wall
            if self._last_status_wall > 0.0
            else float("inf")
        )
        lpos_age = (
            now - self._last_local_pos_wall
            if self._last_local_pos_wall > 0.0
            else float("inf")
        )
        flags_age = (
            now - self._last_estimator_flags_wall
            if self._last_estimator_flags_wall > 0.0
            else float("inf")
        )

        ack = self._last_ack_msg
        if ack is not None:
            ack_summary = (
                f"command={getattr(ack, 'command', None)} "
                f"result={getattr(ack, 'result', None)}"
                f"({self._vehicle_command_result_name(getattr(ack, 'result', -1))}) "
                f"result_param1={getattr(ack, 'result_param1', None)} "
                f"result_param2={getattr(ack, 'result_param2', None)}"
            )
        else:
            ack_summary = "none"

        self.logger.warning(
            f"{context} "
            f"model={self.model_name} ns={self.px4_ns} "
            f"preflight={self.preflight_ok} armed={self.is_armed} "
            f"nav_state={self.nav_state} offboard={self.offboard_enabled} "
            f"position_enabled={self.position_enabled} "
            f"status_age={status_age:.3f} "
            f"failsafe_age={failsafe_age:.3f} "
            f"lpos_age={lpos_age:.3f} "
            f"est_flags_age={flags_age:.3f} "
            f"failure_detector_status={getattr(self, 'failure_detector_status', None)} "
            f"latest_arming_reason={getattr(self, 'latest_arming_reason', None)} "
            f"latest_disarming_reason={getattr(self, 'latest_disarming_reason', None)} "
            f"safety_off={getattr(self, 'safety_off', None)} "
            f"failsafe={getattr(self, 'failsafe', None)} "
            f"active_failsafe_flags={active_failsafe or ['none']} "
            f"mode_req_nonzero={mode_req or {'none': 0}} "
            f"lpos_valid={self._last_local_pos_valid} "
            f"px4_lpos={np.round(self.px4_lpos, 3).tolist()} "
            f"px4_vel={np.round(self.px4_vel, 3).tolist()} "
            f"ekf_flags={{'yaw_align': {self._ekf_yaw_align}, "
            f"'tilt_align': {self._ekf_tilt_align}, "
            f"'ev_yaw': {self._ekf_ev_yaw}, "
            f"'ev_pos': {self._ekf_ev_pos}, "
            f"'ev_hgt': {self._ekf_ev_hgt}}} "
            f"last_ack={ack_summary}"
        )

    def _vo_time_us(self, clock_ns=None) -> int:
        if clock_ns is None:
            clock_ns = self.get_clock().now().nanoseconds
        now_us = int(int(clock_ns) / 1000)
        if now_us <= self._last_vo_stamp_us:
            now_us = self._last_vo_stamp_us + 1000
        self._last_vo_stamp_us = now_us
        return now_us

    def _px4_timestamp_us(self) -> int:
        clock_ns = int(self.get_clock().now().nanoseconds)
        if clock_ns > 0:
            return max(1, int(clock_ns / 1000))
        return max(1, int(time.monotonic() * 1_000_000))

    def _publish_visual_odometry(self):
        """Push Ground Truth Position + Orientation từ Gazebo vào PX4 EKF (NED + EV yaw).

        Called from _vo_thread_worker (30Hz) and optionally from burst callers on the
        main thread; _vo_publish_lock serializes concurrent invocations.
        """
        if not self.gz_pose_ready:
            return

        clock_ns = int(self.get_clock().now().nanoseconds)
        if self._use_sim_time and clock_ns <= 0:
            return
        now_s = float(clock_ns) * 1e-9

        with self._gz_lock:
            x_enu, y_enu, z_enu = self.gz_pos
            q_enu_flu = list(self.gz_quat)
        current_pos_enu = np.array(
            [float(x_enu), float(y_enu), float(z_enu)],
            dtype=np.float32,
        )

        # ENU -> NED
        x_ned = float(y_enu)
        y_ned = float(x_enu)
        z_ned = float(-z_enu)

        q_ned_frd = self._convert_quat_enu_flu_to_ned_frd(q_enu_flu)

        msg = VehicleOdometry()
        stamp_us = self._vo_time_us(clock_ns)
        msg.timestamp = stamp_us
        msg.timestamp_sample = stamp_us

        msg.pose_frame = VehicleOdometry.POSE_FRAME_NED
        msg.position = [x_ned, y_ned, z_ned]

        msg.q = [
            float(q_ned_frd[0]),
            float(q_ned_frd[1]),
            float(q_ned_frd[2]),
            float(q_ned_frd[3]),
        ]
        msg.orientation_variance = [0.05, 0.05, 0.05]

        msg.angular_velocity = [float('nan'), float('nan'), float('nan')]

        if self._teleport_zero_vel_countdown > 0:
            msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
            msg.velocity = [0.0, 0.0, 0.0]
            msg.velocity_variance = [0.05, 0.05, 0.05]
            self._teleport_zero_vel_countdown -= 1
            self._last_gz_pos_for_vo_vel = current_pos_enu.copy()
            self._last_gz_vel_stamp_s = now_s
        elif (
            self._last_gz_pos_for_vo_vel is not None
            and self._last_gz_vel_stamp_s is not None
        ):
            dt = max(1e-3, now_s - float(self._last_gz_vel_stamp_s))
            v_enu = (current_pos_enu - self._last_gz_pos_for_vo_vel) / dt
            if not np.all(np.isfinite(v_enu)):
                v_enu = np.zeros(3, dtype=np.float32)

            speed = float(np.linalg.norm(v_enu))
            max_speed = float(self._vo_velocity_max_m_s)
            if speed > max_speed > 0.0:
                v_enu = v_enu * (max_speed / speed)

            msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
            msg.velocity = [
                float(v_enu[1]),
                float(v_enu[0]),
                float(-v_enu[2]),
            ]
            msg.velocity_variance = [0.05, 0.05, 0.05]
            self._last_gz_pos_for_vo_vel = current_pos_enu.copy()
            self._last_gz_vel_stamp_s = now_s
        else:
            msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
            msg.velocity = [0.0, 0.0, 0.0]
            msg.velocity_variance = [0.10, 0.10, 0.10]
            self._last_gz_pos_for_vo_vel = current_pos_enu.copy()
            self._last_gz_vel_stamp_s = now_s

        # Lower variance means EKF trusts EV more. Keep finite, realistic
        # variance during teleport recovery; do not inflate it as an override.
        if self._teleport_reset_countdown > 0:
            msg.position_variance = [0.01, 0.01, 0.04]
            self._teleport_reset_countdown -= 1
        else:
            msg.position_variance = [0.01, 0.01, 0.01]

        msg.reset_counter = int(self._vo_reset_counter)
        msg.quality = 100

        with self._ros_pub_lock:
            self.vo_pub.publish(msg)

    def publish_vo_burst(self, count=10, interval=0.01):
        """Publish nhiều VO messages liên tiếp ngay sau teleport để EKF hội tụ nhanh.
        Velocity cache is reset by notify_ekf_teleport() before these bursts.
        """
        for _ in range(count):
            with self._vo_publish_lock:
                self._publish_visual_odometry()
            rclpy.spin_once(self, timeout_sec=interval)

    def notify_ekf_teleport(self, prime_count=30, reset_count=100, interval=0.01):
        """Báo EKF sau Gazebo teleport dùng NED + EV yaw strategy.

        1. Reset VO velocity cache so pre/post-teleport poses are not differenced.
        2. Gửi prime_count samples với reset_counter cũ để EKF yaw-align trước.
        3. Tăng reset_counter để kích reset position trong EKF.
        4. Gửi reset_count samples với counter mới để EKF chấp nhận reset.

        EKF FIX 2: Tăng burst count từ (10, 50) lên (30, 100) để đảm bảo
        EKF nhận đủ samples hội tụ position.
        """
        prime_count = max(0, int(prime_count))
        reset_count = max(0, int(reset_count))

        self._last_gz_pos_for_vo_vel = None
        self._last_gz_vel_stamp_s = None
        self._teleport_zero_vel_countdown = max(
            self._teleport_zero_vel_countdown,
            prime_count + reset_count,
        )

        self.logger.info(
            f"[EKF PRIME NED+EV_YAW] model={self.model_name} "
            f"old_counter={self._vo_reset_counter} "
            f"yaw_align={self._ekf_yaw_align} ev_yaw={self._ekf_ev_yaw} "
            f"ev_pos={self._ekf_ev_pos} ev_hgt={self._ekf_ev_hgt} "
            f"gz_pos={self.gz_pos.tolist()} px4_lpos={self.px4_lpos.tolist()}"
        )

        # 1. Prime samples với counter cũ để EKF fuse EV yaw trước
        self.publish_vo_burst(count=prime_count, interval=interval)

        # 2. Tăng reset_counter để kích EKF reset position
        self._vo_reset_counter = (self._vo_reset_counter + 1) % 256
        self._teleport_reset_countdown = max(
            self._teleport_reset_countdown,
            reset_count,
        )

        self.logger.info(
            f"[EKF RESET NED+EV_YAW] model={self.model_name} "
            f"new_counter={self._vo_reset_counter} "
            f"yaw_align={self._ekf_yaw_align} ev_yaw={self._ekf_ev_yaw} "
            f"ev_pos={self._ekf_ev_pos} ev_hgt={self._ekf_ev_hgt}"
        )

        # 3. Burst với counter mới để EKF bắt reset
        self.publish_vo_burst(count=reset_count, interval=interval)

    # ============================================================
    # Gazebo pose source
    # ============================================================

    def _set_gz_pose(self, x, y, z, qw, qx, qy, qz, source="unknown"):
        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm > 1e-6:
            qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
        else:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

        with self._gz_lock:
            self.gz_pos = np.array([float(x), float(y), float(z)], dtype=np.float32)
            self.gz_quat = [float(qw), float(qx), float(qy), float(qz)]
            self.gz_pose_ready = True
            self.gz_pose_stamp = self.get_clock().now().nanoseconds * 1e-9
            self.gz_pose_source = source

        gz_yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        now = time.monotonic()
        if now - float(getattr(self, "_last_gz_pose_debug_log", 0.0)) > 2.0:
            self._last_gz_pose_debug_log = now
            self.logger.debug(
                f"[GZ POSE UPDATE] source={source} "
                f"pos=({x:.3f},{y:.3f},{z:.3f}) "
                f"quat=({qw:.4f},{qx:.4f},{qy:.4f},{qz:.4f}) "
                f"gz_yaw_deg={math.degrees(gz_yaw):.1f}"
            )

    def _convert_quat_enu_flu_to_ned_frd(self, q_enu_flu):
        """
        Convert quaternion từ Gazebo/ROS convention sang PX4 convention.

        Input:  q_enu_flu [w, x, y, z]  — world ENU, body FLU
        Output: q_ned_frd [w, x, y, z]  — world NED, body FRD

        Derived from:  q_ned_frd = q_ned_enu * q_enu_flu * q_flu_frd
          q_ned_enu = [0, -√½, -√½, 0]  (ENU→NED frame rotation)
          q_flu_frd = [0,  1,   0,  0]  (FLU→FRD body rotation, 180° around x)
        Result is automatically unit-norm when input is unit-norm.
        """
        w, x, y, z = float(q_enu_flu[0]), float(q_enu_flu[1]), float(q_enu_flu[2]), float(q_enu_flu[3])
        s = 0.70710678118  # 1/√2
        q_out = [
            s * (w + z),
            s * (x + y),
            s * (x - y),
            s * (w - z),
        ]
        # Renormalize for numerical safety
        norm = math.sqrt(sum(v * v for v in q_out))
        if norm > 1e-6:
            q_out = [v / norm for v in q_out]
        else:
            q_out = [1.0, 0.0, 0.0, 0.0]
        return q_out

    def _extract_pose_name_and_pose(self, pose_msg):
        name = ""

        if hasattr(pose_msg, "name"):
            name = str(pose_msg.name)
        elif hasattr(pose_msg, "entity") and hasattr(pose_msg.entity, "name"):
            name = str(pose_msg.entity.name)
        elif hasattr(pose_msg, "header") and hasattr(pose_msg.header, "frame_id"):
            name = str(pose_msg.header.frame_id)

        pos = None
        quat = None

        if hasattr(pose_msg, "pose"):
            pos = getattr(pose_msg.pose, "position", None)
            quat = getattr(pose_msg.pose, "orientation", None)
        elif hasattr(pose_msg, "position"):
            pos = pose_msg.position
            quat = getattr(pose_msg, "orientation", None)

        return name, pos, quat

    def _gz_entity_pose_cb(self, msg):
        poses = getattr(msg, "poses", None)

        if poses is None:
            poses = getattr(msg, "pose", [])

        for p in poses:
            name, pos, quat = self._extract_pose_name_and_pose(p)

            if pos is None:
                continue

            if self.model_name not in name:
                continue

            if quat is not None:
                qw = float(getattr(quat, "w", 1.0))
                qx = float(getattr(quat, "x", 0.0))
                qy = float(getattr(quat, "y", 0.0))
                qz = float(getattr(quat, "z", 0.0))
            else:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

            self._set_gz_pose(
                pos.x, pos.y, pos.z,
                qw, qx, qy, qz,
                source="ros_entity_pose_v",
            )
            return

    def _gz_model_pose_cb(self, msg):
        """Callback for geometry_msgs/msg/Pose from /model/<model_name>/pose."""
        try:
            self._set_gz_pose(
                float(msg.position.x),
                float(msg.position.y),
                float(msg.position.z),
                float(msg.orientation.w),
                float(msg.orientation.x),
                float(msg.orientation.y),
                float(msg.orientation.z),
                source="ros_gz_model_pose",
            )
        except Exception as exc:
            self.logger.debug(
                f"[GZ POSE] ros model pose callback ignored "
                f"model={self.model_name} reason={exc}"
            )

    def _start_pose_health_check(self, source, topic, delay_s=3.0):
        def _worker():
            time.sleep(float(delay_s))
            if not self.gz_pose_ready:
                self.logger.warning(
                    f"[GZ POSE WARN] no pose received after {delay_s:.1f}s. "
                    f"source={source} topic={topic} model={self.model_name}"
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _start_ros_model_pose_listener(self, topic):
        """Subscribe ROS 2 model pose from gz PosePublisher + ros_gz_bridge."""
        if GeometryPose is None:
            self.logger.warning(
                "[GZ POSE] source=ros_gz_model_pose unavailable "
                "reason=geometry_msgs_import_failed"
            )
            return False

        try:
            self.gz_model_pose_sub = self.create_subscription(
                GeometryPose,
                topic,
                self._gz_model_pose_cb,
                self.qos,
            )
        except Exception as exc:
            self.logger.warning(
                f"[GZ POSE] source=ros_gz_model_pose subscribe_failed "
                f"topic={topic} model={self.model_name} reason={exc}"
            )
            return False

        self.logger.info(
            f"[GZ POSE] source=ros_gz_model_pose topic={topic} "
            f"model={self.model_name} GZ_PARTITION={self.gz_partition}"
        )
        self._start_pose_health_check(source="ros_gz_model_pose", topic=topic)
        return True

    def _start_gz_pose_listener(self):
        """
        Ưu tiên theo thứ tự:

        1. ROS model pose /model/<model>/pose (geometry_msgs/Pose).
        2. Native Gazebo Transport Python subscriber:
           /world/<world>/dynamic_pose/info, gz.msgs.Pose_V
        3. ROS EntityPose_V legacy path (optional).
        4. Fallback cuối: gz topic -e text parser.

        Không dùng TFMessage.
        """
        model_pose_topic = f"/model/{self.model_name}/pose"
        dynamic_pose_topic = f"/world/{self.world_name}/dynamic_pose/info"

        if self._start_ros_model_pose_listener(model_pose_topic):
            return

        self.logger.warning(
            f"[GZ POSE] ROS model pose unavailable; trying native gz transport. "
            f"model_pose_topic={model_pose_topic} "
            f"dynamic_pose_topic={dynamic_pose_topic} "
            f"model={self.model_name} GZ_PARTITION={self.gz_partition}"
        )

        if self._start_native_gz_pose_listener(dynamic_pose_topic):
            return

        if EntityPose_V is not None:
            self.gz_pose_sub = self.create_subscription(
                EntityPose_V,
                dynamic_pose_topic,
                self._gz_entity_pose_cb,
                self.qos,
            )

            self.logger.info(
                f"[GZ POSE] source=ros_entity_pose_v "
                f"topic={dynamic_pose_topic} "
                f"model={self.model_name} "
                f"GZ_PARTITION={self.gz_partition}"
            )
            self._start_pose_health_check(
                source="ros_entity_pose_v",
                topic=dynamic_pose_topic,
            )
            return

        self.logger.error(
            f"[GZ POSE] all typed pose sources failed; "
            f"source=text_parser_fallback "
            f"topic={dynamic_pose_topic} "
            f"model={self.model_name} "
            f"GZ_PARTITION={self.gz_partition}"
        )

        self._start_gz_pose_parser_fallback(dynamic_pose_topic)

    def _start_native_gz_pose_listener(self, topic):
        """
        Native Gazebo Transport subscriber for gz.msgs.Pose_V.

        This avoids parsing `gz topic -e` text output when ROS Jazzy does not
        provide ros_gz_interfaces/msg/EntityPose_V.
        """
        if self.gz_client is None or not self.gz_client.available():
            self.logger.warning(
                "[GZ POSE] source=native_gz_transport unavailable "
                "reason=transport_client_unavailable"
            )
            return False

        try:
            ok = self.gz_client.subscribe_pose_v(self.world_name, self._gz_pose_v_cb)
        except Exception as exc:
            self.logger.warning(
                f"[GZ POSE] source=native_gz_transport failed "
                f"topic={topic} model={self.model_name} reason={exc}"
            )
            return False

        if not ok:
            self.logger.warning(
                f"[GZ POSE] source=native_gz_transport subscribe_failed "
                f"topic={topic} model={self.model_name}"
            )
            return False

        self.logger.info(
            f"[GZ POSE] source=native_gz_transport "
            f"topic={topic} "
            f"model={self.model_name} "
            f"GZ_PARTITION={self.gz_partition}"
        )

        self._start_pose_health_check(source="native_gz_transport", topic=topic)
        return True


    def _gz_pose_v_cb(self, msg):
        """
        Callback for gz.msgs.Pose_V from /world/<world>/dynamic_pose/info.
        """
        try:
            for pose in msg.pose:
                if pose.name != self.model_name:
                    continue

                self._set_gz_pose(
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                    float(pose.orientation.w),
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    source="native_gz_transport",
                )
                return

        except Exception as exc:
            self.logger.debug(
                f"[GZ POSE] native callback ignored "
                f"model={self.model_name} reason={exc}"
            )

    def _start_gz_pose_parser_fallback(self, topic):
        def _worker():
            while True:
                cmd = [
                    "gz",
                    "topic",
                    "-e",
                    "-t",
                    topic,
                ]

                self.logger.debug(
                    f"[GZ POSE PARSER] start "
                    f"model={self.model_name} "
                    f"GZ_PARTITION={self.gz_partition} "
                    f"topic={topic}"
                )

                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        bufsize=1,
                        env=self._gz_env(),
                    )
                except Exception as exc:
                    self.logger.error(
                        f"[GZ POSE PARSER] failed to start "
                        f"model={self.model_name}: {exc}"
                    )
                    time.sleep(1.0)
                    continue

                in_target = False
                in_position = False
                in_orientation = False
                x = y = z = None
                qw = qx = qy = qz = None

                try:
                    for raw_line in proc.stdout:
                        line = raw_line.strip()

                        if line.startswith('name: "'):
                            in_target = f'name: "{self.model_name}"' in line
                            in_position = False
                            in_orientation = False
                            x = y = z = None
                            qw = qx = qy = qz = None
                            continue

                        if not in_target:
                            continue

                        if line.startswith("position {"):
                            in_position = True
                            in_orientation = False
                            continue

                        if line.startswith("orientation {"):
                            in_orientation = True
                            in_position = False
                            continue

                        if in_position:
                            if line.startswith("x:"):
                                x = float(line.split(":", 1)[1].strip())
                            elif line.startswith("y:"):
                                y = float(line.split(":", 1)[1].strip())
                            elif line.startswith("z:"):
                                z = float(line.split(":", 1)[1].strip())
                            elif line.startswith("}"):
                                in_position = False
                        elif in_orientation:
                            if line.startswith("x:"):
                                qx = float(line.split(":", 1)[1].strip())
                            elif line.startswith("y:"):
                                qy = float(line.split(":", 1)[1].strip())
                            elif line.startswith("z:"):
                                qz = float(line.split(":", 1)[1].strip())
                            elif line.startswith("w:"):
                                qw = float(line.split(":", 1)[1].strip())
                            elif line.startswith("}"):
                                in_orientation = False
                                if (
                                    x is not None and y is not None and z is not None
                                    and qw is not None and qx is not None
                                    and qy is not None and qz is not None
                                ):
                                    self._set_gz_pose(
                                        x, y, z,
                                        qw, qx, qy, qz,
                                        source="text_parser_fallback",
                                    )

                except Exception as exc:
                    self.logger.error(
                        f"[GZ POSE PARSER] exception "
                        f"model={self.model_name}: {exc}"
                    )

                try:
                    proc.kill()
                except Exception:
                    pass

                self.logger.debug(
                    f"[GZ POSE PARSER] stopped, restart soon. "
                    f"model={self.model_name}"
                )

                time.sleep(1.0)

        threading.Thread(target=_worker, daemon=True).start()

    # ============================================================
    # State getters
    # ============================================================

    def get_gazebo_position(self):
        """
        Return Gazebo raw world pose only.
        Used by reset/teleport/hard reset.
        """
        with self._gz_lock:
            return self.gz_pos.copy()

    def get_linear_velocity(self):
        # PX4 local velocity is in NED frame (North, East, Down).
        # We return it in Gazebo ENU frame (East, North, Up).
        vx_enu = float(self.px4_vel[1])   # East
        vy_enu = float(self.px4_vel[0])   # North
        vz_enu = -float(self.px4_vel[2])  # Up
        return np.array([vx_enu, vy_enu, vz_enu], dtype=np.float32)

    def get_yaw(self):
        # PX4 Yaw is clockwise from North. Gazebo Yaw is counter-clockwise from East.
        yaw_enu = -float(self.yaw) + (math.pi / 2.0)
        return yaw_enu, 0.0

    def get_angular_velocity(self) -> np.ndarray:
        return self.angular_velocity.copy()  # FRD body rates (rad/s)

    def get_ekf_position_enu(self) -> np.ndarray:
        # px4_lpos is NED (North, East, Down) — convert to ENU (East, North, Up)
        return np.array([
            float(self.px4_lpos[1]),   # East  ← NED[1]
            float(self.px4_lpos[0]),   # North ← NED[0]
            -float(self.px4_lpos[2]),  # Up    ← -NED[2]
        ], dtype=np.float32)

    def get_depth_84(self):
        return np.expand_dims(self.depth_raw, axis=0)

    def _lidar_cb(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        # posinf = beam exceeded max range → clear space
        ranges[np.isposinf(ranges)] = 30.0
        # nan / neginf / <=0 = signal error → treat as close obstacle (conservative)
        bad = np.isnan(ranges) | np.isneginf(ranges) | (ranges <= 0.0)
        ranges[bad] = 0.1
        self.lidar_raw = np.clip(ranges, 0.1, 30.0)

    def get_lidar_scan(self) -> np.ndarray:
        return self.lidar_raw.copy()  # (1080,) float32, range [0.1, 30.0] m

    def is_flipped(self):
        roll_deg = abs(math.degrees(self.roll))
        pitch_deg = abs(math.degrees(self.pitch))
        return roll_deg > 90.0 or pitch_deg > 90.0

    # ============================================================
    # PX4 setpoints
    # ============================================================

    def send_velocity(self, vx_enu, vy_enu, vz_enu, yr_enu):
        """
        Gửi vận tốc xuống PX4.
        Input: vx_enu, vy_enu, vz_enu, yr_enu (hệ quy chiếu Gazebo ENU).
        PX4 cần hệ quy chiếu NED (North, East, Down).
        - PX4 North (vx_ned) = ENU North (vy_enu)
        - PX4 East (vy_ned)  = ENU East (vx_enu)
        - PX4 Down (vz_ned)  = ENU Down (-vz_enu)
        - PX4 Yaw Rate (CW)  = ENU Yaw Rate (CCW) * -1
        """
        vx_ned = float(vy_enu)
        vy_ned = float(vx_enu)
        vz_ned = -float(vz_enu)
        yr_ned = -float(yr_enu)

        with self._last_setpoint_lock:
            self._last_setpoint_mode = "velocity"
            self._last_velocity_cmd = (
                vx_ned,
                vy_ned,
                vz_ned,
                yr_ned,
            )
            self._last_setpoint_time = self.get_clock().now().nanoseconds * 1e-9
            # Reset hold ngay khi có action mới
            self._stall_pos_locked = False

        self._publish_velocity_setpoint(vx_ned, vy_ned, vz_ned, yr_ned)

    def _publish_velocity_setpoint(self, vx, vy, vz, yr):
        ts = self._px4_timestamp_us()

        om = OffboardControlMode()
        om.timestamp = ts
        om.position = False
        om.velocity = True
        om.acceleration = False
        om.attitude = False
        om.body_rate = False

        tp = TrajectorySetpoint()
        tp.timestamp = ts
        tp.position = [math.nan, math.nan, math.nan]
        tp.velocity = [float(vx), float(vy), float(vz)]
        tp.yaw = math.nan
        tp.yawspeed = float(yr)

        with self._ros_pub_lock:
            self.offboard_pub.publish(om)
            self.trajectory_pub.publish(tp)

    def send_position_setpoint_ned(self, x, y, z, yaw=math.nan):
        """
        Gửi position setpoint theo PX4 local NED frame.

        PX4 NED:
        - z nhỏ hơn / âm hơn là bay lên.
        """
        with self._last_setpoint_lock:
            self._last_setpoint_mode = "position"
            self._last_position_cmd = (
                float(x),
                float(y),
                float(z),
                float(yaw) if not math.isnan(yaw) else math.nan,
            )
            self._last_setpoint_time = self.get_clock().now().nanoseconds * 1e-9
            # Reset hold ngay khi có action mới
            self._stall_pos_locked = False

        self._publish_position_setpoint_ned(x, y, z, yaw)

    def _publish_position_setpoint_ned(self, x, y, z, yaw=math.nan):
        ts = self._px4_timestamp_us()

        om = OffboardControlMode()
        om.timestamp = ts
        om.position = True
        om.velocity = False
        om.acceleration = False
        om.attitude = False
        om.body_rate = False

        tp = TrajectorySetpoint()
        tp.timestamp = ts
        tp.position = [float(x), float(y), float(z)]
        tp.velocity = [math.nan, math.nan, math.nan]
        tp.yaw = float(yaw) if not math.isnan(yaw) else math.nan
        tp.yawspeed = math.nan

        with self._ros_pub_lock:
            self.offboard_pub.publish(om)
            self.trajectory_pub.publish(tp)

    # ============================================================
    # Arm / disarm / offboard
    # ============================================================

    def arm(self, force=True):
        """
        Force arm cho SITL/mô phỏng.

        PX4 magic number:
        - param2 = 21196.0: force arm/disarm

        Không dùng force=True cho drone thật.
        """
        force_magic = 21196.0 if force else 0.0

        self._send_cmd(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            p1=1.0,
            p2=force_magic,
        )

    def disarm(self, force=False):
        """
        Disarm PX4.

        force=True dùng cho SITL reset khi PX4 reject disarm vì còn trên không.
        """
        self.enable_offboard_keepalive(False)

        force_magic = 21196.0 if force else 0.0

        self._send_cmd(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            p1=0.0,
            p2=force_magic,
        )

    def wait_until_armed(
        self, timeout=0.8, stream_mode="position", target_z_ned=-1.2
):
        """
        Chờ PX4 armed nhưng vẫn stream setpoint liên tục.
        """
        t0 = time.monotonic()

        x_ned = float(self.px4_lpos[0])
        y_ned = float(self.px4_lpos[1])

        while time.monotonic() - t0 < timeout:
            if stream_mode == "position":
                self.send_position_setpoint_ned(x_ned, y_ned, target_z_ned, yaw=math.nan)
            else:
                self.send_velocity(0.0, 0.0, 0.0, 0.0)

            rclpy.spin_once(self, timeout_sec=0.03)

            if self.is_armed:
                return True

        return False

    def wait_until_offboard(
        self, timeout=1.2, stream_mode="position", target_z_ned=-1.2
    ):
        """
        Chờ PX4 vào OFFBOARD nhưng vẫn stream setpoint liên tục.
        """
        t0 = time.monotonic()

        x_ned = float(self.px4_lpos[0])
        y_ned = float(self.px4_lpos[1])

        while time.monotonic() - t0 < timeout:
            if stream_mode == "position":
                self.send_position_setpoint_ned(x_ned, y_ned, target_z_ned, yaw=math.nan)
            else:
                self.send_velocity(0.0, 0.0, 0.0, 0.0)

            rclpy.spin_once(self, timeout_sec=0.03)

            if self.offboard_enabled or self.nav_state == NavState.OFFBOARD:
                return True

        return False

    def arm_and_takeoff(self):
        """
        Sử dụng Offboard Velocity Takeoff (Ép cất cánh bằng vận tốc):
        1. Stream velocity setpoints (vz âm để đi lên).
        2. Set mode sang OFFBOARD.
        3. Force ARM.
        4. Chờ đạt độ cao và ổn định.
        """
        if not np.all(np.isfinite(self.px4_lpos)):
            self.logger.error("[ARM] px4_lpos is not finite, fail fast.")
            return False

        already_ready = (
            self.is_armed
            and (self.offboard_enabled or self.nav_state == NavState.OFFBOARD)
        )

        if already_ready:
            self.logger.info("[ARM] already armed/offboard, continue.")
            for _ in range(10):
                self.send_velocity(0.0, 0.0, 0.0, 0.0)
                rclpy.spin_once(self, timeout_sec=0.05)
            self.enable_offboard_keepalive(True)
            return True

        self.logger.debug(
            "[ARM] start PX4 OFFBOARD takeoff flow (Velocity mode). "
            f"model={self.model_name} "
            f"gz_pos={self.get_gazebo_position()} "
            f"px4_lpos={self.px4_lpos} "
        )

        # 1. Stream velocity setpoint trước khi set OFFBOARD (PX4 yêu cầu stream > 0.5s)
        for _ in range(50):
            self.send_velocity(0.0, 0.0, 0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.03)

        # 2. Chuyển sang OFFBOARD mode
        if not (self.offboard_enabled or self.nav_state == NavState.OFFBOARD):
            self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0)
            offboard_ok = False
            t0 = time.monotonic()
            while time.monotonic() - t0 < 2.0:
                self.send_velocity(0.0, 0.0, 1.8, 0.0)
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.offboard_enabled or self.nav_state == NavState.OFFBOARD:
                    offboard_ok = True
                    break

            if not offboard_ok:
                self.logger.error(f"[OFFBOARD] mode switch failed. offboard={self.offboard_enabled}")
                return False

        # 3. ARM (Dùng force=True vì môi trường RL không có tín hiệu RC)
        if not self.is_armed:
            self.arm(force=False)
            t0 = time.monotonic()
            armed_ok = False
            while time.monotonic() - t0 < 2.0:
                self.send_velocity(0.0, 0.0, 1.5, 0.0)
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.is_armed:
                    armed_ok = True
                    break
            if not armed_ok:
                self.logger.error(f"[ARM] arm failed. armed={self.is_armed}")
                return False

        # 4. Chờ nâng độ cao (Ép lên bằng vận tốc)
        best_alt = float(self.get_gazebo_position()[2])
        takeoff_t0 = time.monotonic()
        last_print_t = takeoff_t0
        while time.monotonic() - takeoff_t0 < 15.0:
            # Gửi liên tục vận tốc đi lên (vz = 1.5 m/s trong hệ quy chiếu ENU)
            self.send_velocity(0.0, 0.0, 1.8, 0.0)
            rclpy.spin_once(self, timeout_sec=0.05)

            gz_alt = float(self.get_gazebo_position()[2])
            best_alt = max(best_alt, gz_alt)

            if time.monotonic() - last_print_t > 1.0:
                self.logger.debug(f"[ARM] forcing takeoff (velocity)... current alt={gz_alt:.2f}m")
                last_print_t = time.monotonic()

            if gz_alt >= 3.0:
                break

        # Settle thêm 1 chút cho ổn định bằng cách phanh lại (vz = 0)
        for _ in range(10):
            self.send_velocity(0.0, 0.0, 0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.03)

        if best_alt < 0.35:
            self.logger.error(f"[ARM] takeoff did not lift enough, fail. best_alt={best_alt:.2f}")
            # Thử thêm 1 lần hích cực mạnh cuối cùng thay vì bỏ cuộc luôn
            self.send_velocity(0.0, 0.0, 3.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.5)
            return False

        self.logger.debug(f"[ARM] OFFBOARD velocity takeoff done. best_alt={best_alt:.2f}")
        self.enable_offboard_keepalive(False)
        return True



    # ============================================================
    # Gazebo helpers
    # ============================================================
    #
    def _gz_spin_wrap(self, fn, *args, max_wait_s=3.0, **kwargs):
        """Run fn(*args, **kwargs) in a background thread while keeping rclpy spinning.

        Prevents gz transport RPCs and subprocess.run calls from starving the ROS
        executor: without this, vo_timer and PX4 callbacks drop during teleport /
        pillar-spawn operations, causing EKF time stalls and KEEPALIVE STALE HOVER
        warnings when training with 2+ envs.

        Returns fn's return value, or None if max_wait_s is exceeded before fn returns.
        Exceptions raised by fn are re-raised in the caller.
        """
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(fn, *args, **kwargs)
        ex.shutdown(wait=False)
        deadline = time.monotonic() + float(max_wait_s)
        while not fut.done():
            rclpy.spin_once(self, timeout_sec=0.005)
            if time.monotonic() > deadline:
                self.logger.warning(
                    f"[GZ SPIN WRAP] {getattr(fn, '__name__', str(fn))} still running "
                    f"after {max_wait_s:.1f}s — proceeding without result"
                )
                return None
        return fut.result()

    def teleport_drone(self, pos):
        """
        Teleport model trong đúng GZ_PARTITION.

        Quan trọng:
        - Không chỉ kiểm tra returncode.
        - Phải kiểm tra stdout có data: true.
        - Sau đó wait pose update mới hơn thời điểm gọi teleport.
        """
        x, y, z = map(float, pos)

        req_str = (
            f'name: "{self.model_name}" '
            f"position {{ x: {x} y: {y} z: {z} }} "
            f"orientation {{ x: 0 y: 0 z: 0 w: 1 }}"
        )

        cmd = [
            "gz",
            "service",
            "-s",
            f"/world/{self.world_name}/set_pose/blocking",
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            req_str,
        ]

        call_time = self.get_clock().now().nanoseconds * 1e-9
        self._invalidate_gz_pose_cache()

        transport_ok = False
        if self.gz_client is not None and self.gz_client.available():
            try:
                _sp_result = self._gz_spin_wrap(
                    self.gz_client.set_pose,
                    world_name=self.world_name,
                    name=self.model_name,
                    x=x,
                    y=y,
                    z=z,
                    yaw=0.0,
                    timeout_ms=2000,
                    max_wait_s=2.5,
                )
                transport_ok = bool(_sp_result) if _sp_result is not None else False
                if transport_ok:
                    self.logger.debug(
                        f"[GZ TRANSPORT] set_pose ok world={self.world_name} model={self.model_name}"
                    )
                else:
                    self.logger.warning(
                        f"[GZ TRANSPORT] set_pose failed; fallback CLI world={self.world_name} model={self.model_name}"
                    )
            except Exception as exc:
                self.logger.warning(f"[GZ TRANSPORT] set_pose failed; fallback CLI: {exc}")
                transport_ok = False

        if transport_ok:
            pose_ok = self.wait_for_gazebo_pose(
                [x, y, z],
                timeout=3.0,
                tol=0.5,
                min_stamp=call_time,
            )
            if pose_ok:
                self._teleport_zero_vel_countdown = 40
            return pose_ok

        try:
            result = self._gz_spin_wrap(
                subprocess.run,
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
                env=self._gz_env(),
                max_wait_s=6.0,
            )
            if result is None:
                self.logger.error(
                    f"[Teleport] CLI spin_wrap timed out "
                    f"model={self.model_name} GZ_PARTITION={self.gz_partition}"
                )
                return False
        except Exception as exc:
            self.logger.error(
                f"[Teleport] exception "
                f"model={self.model_name} "
                f"GZ_PARTITION={self.gz_partition}: {exc}"
            )
            return False

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if stdout:
            self.logger.debug(f"[Teleport] stdout: {stdout}")

        if stderr:
            self.logger.debug(f"[Teleport] stderr: {stderr}")

        service_ok = result.returncode == 0 and ("data: true" in stdout.lower())

        if not service_ok:
            self.logger.error(
                f"[Teleport] failed. "
                f"model={self.model_name} "
                f"world={self.world_name} "
                f"GZ_PARTITION={self.gz_partition} "
                f"returncode={result.returncode} "
                f"stdout={stdout} "
                f"stderr={stderr}"
            )
            return False

        pose_ok = self.wait_for_gazebo_pose(
            [x, y, z],
            timeout=3.0,
            tol=0.5,
            min_stamp=call_time,
        )
        if pose_ok:
            self._teleport_zero_vel_countdown = 40
        return pose_ok

    def wait_for_gazebo_pose(self, target, timeout=3.0, tol=0.5, min_stamp=0.0):
        """
        Chờ pose từ ros_gz/parser.

        min_stamp dùng để tránh pose cache cũ đánh lừa sau hard reset/teleport.
        """
        target = np.array(target, dtype=np.float32)
        t0 = time.monotonic()

        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)

            with self._gz_lock:
                pos = self.gz_pos.copy()
                ready = bool(self.gz_pose_ready)
                stamp = float(self.gz_pose_stamp)

            if ready and stamp >= min_stamp and np.linalg.norm(pos - target) < tol:
                return True

        with self._gz_lock:
            pos = self.gz_pos.copy()
            ready = bool(self.gz_pose_ready)
            stamp = float(self.gz_pose_stamp)
            source = self.gz_pose_source

        self.logger.info(
            f"[Teleport] set_pose returned success but pose bridge did not confirm. "
            f"model={self.model_name} "
            f"GZ_PARTITION={self.gz_partition} "
            f"target={target.tolist()} "
            f"current={pos} "
            f"gz_pose_ready={ready} "
            f"gz_pose_stamp={stamp:.3f} "
            f"min_stamp={min_stamp:.3f} "
            f"source={source}"
        )

        return False

    def debug_gz_partition_topics(self):
        cmd = ["gz", "topic", "-l"]

        try:
            result = subprocess.run(
                cmd,
                env=self._gz_env(),
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except Exception as exc:
            self.logger.error(
                f"[GZ DEBUG] topic list failed "
                f"model={self.model_name} "
                f"GZ_PARTITION={self.gz_partition}: {exc}"
            )
            return

        lines = [
            line for line in result.stdout.splitlines()
            if self.model_name in line or "pose" in line or "state" in line
        ]

        self.logger.debug(
            f"[GZ DEBUG] model={self.model_name} "
            f"GZ_PARTITION={self.gz_partition} "
            f"topics={lines[:30]}"
        )

    def collided(self):
        """
        Depth-based collision proxy.

        Avoid false positives from a single near pixel at image edge.
        Only trigger if a meaningful region in the central/front depth view
        is very close.
        """
        depth = np.asarray(self.depth_raw, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            return False

        h, w = depth.shape

        # Chỉ xét vùng trung tâm, không xét toàn ảnh.
        # Giảm false positive khi cột nằm bên trái/phải camera.
        roi = depth[
            int(h * 0.25): int(h * 0.80),
            int(w * 0.20): int(w * 0.80),
        ]

        valid = np.isfinite(roi)
        if not np.any(valid):
            return False

        roi_valid = roi[valid]

        near_thresh = 0.7
        near_ratio_thresh = 0.02

        min_depth = float(np.min(roi_valid))
        near_ratio = float(np.mean(roi_valid < near_thresh))

        return bool(
            min_depth < near_thresh
            and near_ratio > near_ratio_thresh
        )

    def tick(self, dt, max_wall_s=None):
        dt = float(dt)
        max_wall_s = float(max_wall_s if max_wall_s is not None else min(dt + 0.05, 0.20))

        start = time.monotonic()
        end_time = start + dt

        spin_count = 0
        max_spin = 0.0

        while time.monotonic() < end_time:
            if time.monotonic() - start > max_wall_s:
                self.logger.warning(
                    f"[BRIDGE TICK OVERRUN] requested_dt={dt:.3f} "
                    f"wall={time.monotonic()-start:.3f} "
                    f"spin_count={spin_count} max_spin={max_spin:.3f}"
                )
                break

            remaining = end_time - time.monotonic()
            spin_t0 = time.monotonic()
            rclpy.spin_once(self, timeout_sec=min(0.005, max(0.0, remaining)))
            spin_dt = time.monotonic() - spin_t0
            spin_count += 1
            max_spin = max(max_spin, spin_dt)

    def _px4_callback_freshness(self, after_wall, max_age=0.5):
        now = time.monotonic()
        after_wall = float(after_wall)
        max_age = float(max_age)

        def _age(last_wall):
            last_wall = float(last_wall)
            if last_wall <= 0.0:
                return float("inf")
            return now - last_wall

        def _fresh(last_wall):
            last_wall = float(last_wall)
            return last_wall > after_wall and (now - last_wall) <= max_age

        return {
            "status_ok": _fresh(self._last_status_wall),
            "local_pos_ok": _fresh(self._last_local_pos_wall),
            "estimator_flags_ok": _fresh(self._last_estimator_flags_wall),
            "status_age": _age(self._last_status_wall),
            "local_pos_age": _age(self._last_local_pos_wall),
            "estimator_flags_age": _age(self._last_estimator_flags_wall),
        }

    def has_fresh_px4_callbacks_after(
        self,
        after_wall,
        max_age=0.5,
        require_status=True,
        require_local_pos=True,
        prefer_estimator_flags=True,
        log_estimator_fallback=False,
    ):
        freshness = self._px4_callback_freshness(
            after_wall=after_wall,
            max_age=max_age,
        )

        if require_status and not freshness["status_ok"]:
            return False

        if require_local_pos and not freshness["local_pos_ok"]:
            return False

        if prefer_estimator_flags and not freshness["estimator_flags_ok"]:
            now = time.monotonic()
            if (
                log_estimator_fallback
                and now - self._last_estimator_flags_fallback_warn_wall > 1.0
            ):
                self._last_estimator_flags_fallback_warn_wall = now
                self.logger.warning(
                    "[PX4 CALLBACK FRESHNESS] estimator_flags stale/unavailable; "
                    "continuing with fresh status/local_pos "
                    f"after_wall={float(after_wall):.3f} "
                    f"status_age={freshness['status_age']:.3f} "
                    f"lpos_age={freshness['local_pos_age']:.3f} "
                    f"flags_age={freshness['estimator_flags_age']:.3f}"
                )

        return True

    def wait_for_fresh_px4_callbacks(
        self,
        after_wall,
        timeout=3.0,
        max_age=0.5,
        require_status=True,
        require_local_pos=True,
        prefer_estimator_flags=True,
    ):
        t0 = time.monotonic()
        last_freshness = None

        while time.monotonic() - t0 < float(timeout):
            self.tick(0.05)
            last_freshness = self._px4_callback_freshness(
                after_wall=after_wall,
                max_age=max_age,
            )

            status_ok = (not require_status) or last_freshness["status_ok"]
            local_pos_ok = (not require_local_pos) or last_freshness["local_pos_ok"]

            if status_ok and local_pos_ok:
                if prefer_estimator_flags and last_freshness["estimator_flags_ok"]:
                    self.logger.info(
                        "[PX4 CALLBACK FRESHNESS] fresh status/local_pos/estimator_flags "
                        f"after_wall={float(after_wall):.3f} "
                        f"status_age={last_freshness['status_age']:.3f} "
                        f"lpos_age={last_freshness['local_pos_age']:.3f} "
                        f"flags_age={last_freshness['estimator_flags_age']:.3f}"
                    )
                elif prefer_estimator_flags:
                    self.logger.warning(
                        "[PX4 CALLBACK FRESHNESS] estimator_flags stale/unavailable; "
                        "continuing with fresh status/local_pos "
                        f"after_wall={float(after_wall):.3f} "
                        f"status_age={last_freshness['status_age']:.3f} "
                        f"lpos_age={last_freshness['local_pos_age']:.3f} "
                        f"flags_age={last_freshness['estimator_flags_age']:.3f}"
                    )
                else:
                    self.logger.info(
                        "[PX4 CALLBACK FRESHNESS] fresh status/local_pos "
                        f"after_wall={float(after_wall):.3f} "
                        f"status_age={last_freshness['status_age']:.3f} "
                        f"lpos_age={last_freshness['local_pos_age']:.3f}"
                    )
                return True

        if last_freshness is None:
            last_freshness = self._px4_callback_freshness(
                after_wall=after_wall,
                max_age=max_age,
            )

        self.logger.warning(
            "[PX4 CALLBACK FRESHNESS] timeout "
            f"after_wall={float(after_wall):.3f} "
            f"status_ok={last_freshness['status_ok']} "
            f"lpos_ok={last_freshness['local_pos_ok']} "
            f"flags_ok={last_freshness['estimator_flags_ok']} "
            f"status_age={last_freshness['status_age']:.3f} "
            f"lpos_age={last_freshness['local_pos_age']:.3f} "
            f"flags_age={last_freshness['estimator_flags_age']:.3f}"
        )
        return False

    def print_debug_status(self, prefix="[BRIDGE]"):
        self.logger.info(
            f"{prefix} "
            f"model={self.model_name} "
            f"px4_ns={self.px4_ns} "
            f"target_system={self.target_system} "
            f"GZ_PARTITION={self.gz_partition} "
            f"armed={self.is_armed} "
            f"nav_state={self.nav_state} "
            f"offboard={self.offboard_enabled} "
            f"preflight={self.preflight_ok} "
            f"gz_pos={self.get_gazebo_position()} "
            f"gz_pose_ready={self.gz_pose_ready} "
            f"gz_pose_source={self.gz_pose_source} "
            f"px4_lpos={self.px4_lpos} "
            f"vel={self.get_linear_velocity()}"
        )

    def _send_cmd(self, command, p1=0.0, p2=0.0, p3=0.0, p7=0.0):
        msg = VehicleCommand(
            command=command,
            param1=float(p1),
            param2=float(p2),
            param3=float(p3),
            param7=float(p7),
            target_system=self.target_system,
            target_component=1,
            from_external=True,
            timestamp=self._px4_timestamp_us(),
        )

        msg.param5 = math.nan
        msg.param6 = math.nan

        self.command_pub.publish(msg)

    def request_estimator_reset(self):
        """DEPRECATED: CMD 241 không tồn tại trong PX4 build này. Dùng notify_ekf_teleport()."""
        self.logger.warning(
            f"[EKF RESET] request_estimator_reset DEPRECATED, "
            f"redirecting to notify_ekf_teleport for {self.model_name}"
        )
        self.notify_ekf_teleport()

        self.logger.debug(f"[EKF RESET] done for {self.model_name}")

    def request_ekf_gps_reset(self):
        """
        Alternative: dùng GPS Yaw reset command.

        MAV_CMD_GPS_INPUT (command 220) với reset flags.
        Hoặc dùng VehicleCommand.VEHICLE_CMD_SET_GPS_GLOBAL_ORIGIN.
        """
        self.logger.debug(f"[EKF GPS RESET] alternative reset for {self.model_name}")

        for _ in range(3):
            self._send_cmd(220, p1=0.0, p2=0.0, p7=0.0)
            rclpy.spin_once(self, timeout_sec=0.05)


class Spawner:
    def __init__(self, world_name="default", gz_partition=None, verbose_pillar_verify=False):
        self.world_name = world_name
        self.gz_partition = gz_partition or os.environ.get("GZ_PARTITION", "")
        self.verbose_pillar_verify = bool(verbose_pillar_verify)
        partition_suffix = f"_{self.gz_partition}" if self.gz_partition else ""
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        _log_path = os.path.join(_project_root, "runs", "env_logs", f"spawner{partition_suffix}.txt")
        self.logger = setup_logger(f"SPAWNER{partition_suffix}", log_file=_log_path)
        self.gz_client = (
            GzTransportClient(gz_partition=self.gz_partition, use_lock=True, logger=self.logger)
            if GzTransportClient is not None
            else None
        )


    def _gz_env(self):
        env = os.environ.copy()

        if self.gz_partition:
            env["GZ_PARTITION"] = str(self.gz_partition)

        return env

    def spawn_pillar(self, name, x, y, radius=0.3, height=6.0):
        if spawn_world is not None:
            try:
                ok = self._gz_spin_wrap(
                    spawn_world.spawn_pillar,
                    name=name, x=x, y=y, radius=radius, height=height,
                    world_name=self.world_name, env=self._gz_env(),
                    max_wait_s=18.0,
                )
                return bool(ok) if ok is not None else False
            except Exception as exc:
                self.logger.warning(f"[SPAWNER SPAWN FAIL] name={name} reason={exc}")
                return False
        return False

    def move_pillar(self, name, x, y, z):
        if spawn_world is not None:
            try:
                ok = self._gz_spin_wrap(
                    spawn_world.move_entity,
                    name=name,
                    x=x,
                    y=y,
                    z=z,
                    world_name=self.world_name,
                    env=self._gz_env(),
                    max_wait_s=4.0,
                )
                return bool(ok) if ok is not None else False
            except Exception as exc:
                self.logger.warning(f"[SPAWNER MOVE FAIL] name={name} reason={exc}")
                return False
        return False

    def move_pillars_batch(self, poses):
        """Batch move pillars bằng Gazebo /set_pose_vector.

        poses: list[dict] với keys: name, x, y, z, yaw (optional)
        """
        if spawn_world is None:
            raise RuntimeError("spawn_world module is not available")
        timeout_ms = int(os.environ.get("GZ_SET_POSE_VECTOR_TIMEOUT_MS", "2500"))
        return self._gz_spin_wrap(
            spawn_world.move_entities_batch,
            poses=poses,
            world_name=self.world_name,
            env=self._gz_env(),
            timeout_ms=timeout_ms,
            max_wait_s=float(timeout_ms) / 1000.0 + 1.5,
        )

    def _scene_entity_names(self, timeout_ms=3000):
        if self.gz_client is not None and self.gz_client.available():
            try:
                names = self._gz_spin_wrap(
                    self.gz_client.scene_entity_names,
                    self.world_name,
                    timeout_ms=timeout_ms,
                    max_wait_s=float(timeout_ms) / 1000.0 + 1.0,
                )
                if names is not None:
                    self.logger.debug(
                        f"[GZ TRANSPORT] scene_entity_names ok world={self.world_name} count={len(names)}"
                    )
                    return set(names)
                self.logger.warning(
                    f"[GZ TRANSPORT] scene_entity_names failed; fallback CLI world={self.world_name}"
                )
            except Exception as exc:
                self.logger.warning(
                    f"[GZ TRANSPORT] scene_entity_names failed; fallback CLI world={self.world_name} error={exc}"
                )

        cmd = [
            "gz",
            "service",
            "-s",
            f"/world/{self.world_name}/scene/info",
            "--reqtype",
            "gz.msgs.Empty",
            "--reptype",
            "gz.msgs.Scene",
            "--timeout",
            str(timeout_ms),
            "--req",
            "",
        ]
        result = self._gz_spin_wrap(
            subprocess.run, cmd,
            capture_output=True, text=True, env=self._gz_env(), check=False,
            max_wait_s=float(timeout_ms) / 1000.0 + 1.0,
        )
        if result is None or result.returncode != 0:
            rc = result.returncode if result is not None else "timeout"
            self.logger.warning(f"[SPAWNER VERIFY] scene/info failed returncode={rc}")
            return set()
        return set(re.findall(r'name:\s*"([^"]+)"', result.stdout or ""))

    def _dynamic_pose_map(self, timeout_sec=2.0):
        topic = f"/world/{self.world_name}/dynamic_pose/info"
        return self._topic_pose_map(topic=topic, timeout_sec=timeout_sec)

    def _pose_info_map(self, timeout_sec=2.0):
        topic = f"/world/{self.world_name}/pose/info"
        return self._topic_pose_map(topic=topic, timeout_sec=timeout_sec)

    def _topic_pose_map(self, topic, timeout_sec=2.0):
        cmd = ["gz", "topic", "-e", "-t", topic]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._gz_env(),
        )
        out = ""
        try:
            out, _ = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            out = (exc.output or "") if isinstance(exc.output, str) else (exc.output.decode("utf-8", "ignore") if exc.output else "")
        return self._parse_pose_text_to_map(out)

    def _parse_pose_text_to_map(self, text):
        pose_map = {}
        if not text:
            return pose_map
        pattern = re.compile(
            r'name:\s*"([^"]+)".*?position\s*\{\s*x:\s*([-\d.eE+]+)\s*y:\s*([-\d.eE+]+)\s*z:\s*([-\d.eE+]+)\s*\}',
            re.DOTALL,
        )
        for m in pattern.finditer(text):
            try:
                pose_map[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
            except Exception:
                continue
        return pose_map

    def verify_pillars(self, candidate_metadata, pose_tol=0.5):
        scene_names = self._scene_entity_names()
        pose_info_map = self._pose_info_map()
        dynamic_map = self._dynamic_pose_map()
        return self._verify_pillars_from_snapshots(
            candidate_metadata,
            pose_info_map,
            dynamic_map,
            scene_names,
            pose_tol=pose_tol,
        )

    def _verify_pillars_from_snapshots(self, candidate_metadata, pose_info_map, dynamic_pose_map, scene_names, pose_tol=0.5):
        results = []
        existing_count = 0
        has_missing = False
        has_pose_error = False
        has_pose_missing = False
        for m in candidate_metadata:
            name = m["name"]
            exp_x = float(m["x"])
            exp_y = float(m["y"])
            pose_info_actual = pose_info_map.get(name)
            dynamic_actual = dynamic_pose_map.get(name)

            if pose_info_actual is not None and dynamic_actual is not None:
                pose_info_xy = np.array([float(pose_info_actual[0]), float(pose_info_actual[1])], dtype=np.float32)
                dynamic_xy = np.array([float(dynamic_actual[0]), float(dynamic_actual[1])], dtype=np.float32)
                source_delta = float(np.linalg.norm(pose_info_xy - dynamic_xy))
                if source_delta > 0.5:
                    self.logger.warning(
                        "[SPAWNER POSE SOURCE MISMATCH] "
                        f"name={name} "
                        f"pose_info_xy=({pose_info_xy[0]:.3f},{pose_info_xy[1]:.3f}) "
                        f"dynamic_pose_xy=({dynamic_xy[0]:.3f},{dynamic_xy[1]:.3f}) "
                        f"delta={source_delta:.3f}"
                    )

            if pose_info_actual is not None:
                exists = True
                pose_source = "pose_info"
                existing_count += 1
                actual_xy = (float(pose_info_actual[0]), float(pose_info_actual[1]))
                pose_error = float(np.linalg.norm(np.array([actual_xy[0] - exp_x, actual_xy[1] - exp_y])))
                if pose_error > float(pose_tol):
                    has_pose_error = True
            elif dynamic_actual is not None:
                exists = True
                pose_source = "dynamic_pose"
                existing_count += 1
                actual_xy = (float(dynamic_actual[0]), float(dynamic_actual[1]))
                pose_error = float(np.linalg.norm(np.array([actual_xy[0] - exp_x, actual_xy[1] - exp_y])))
                if pose_error > float(pose_tol):
                    has_pose_error = True
            elif name in scene_names:
                exists = True
                pose_source = "scene_only"
                existing_count += 1
                has_pose_missing = True
                actual_xy = None
                pose_error = None
            else:
                exists = False
                pose_source = "missing"
                has_missing = True
                actual_xy = None
                pose_error = None
            results.append(
                {
                    "name": name,
                    "expected_xy": (exp_x, exp_y),
                    "exists": exists,
                    "pose_source": pose_source,
                    "actual_xy": actual_xy,
                    "pose_error": pose_error,
                }
            )

        status = "ok"
        if has_missing:
            status = "fail_missing_entities"
        elif has_pose_missing:
            status = "inconclusive_pose_missing"
        elif has_pose_error:
            status = "fail_pose_error"

        pose_errors = [float(r["pose_error"]) for r in results if r["pose_error"] is not None]
        max_pose_error = max(pose_errors) if pose_errors else None

        should_log_per_pillar = bool(self.verbose_pillar_verify or status != "ok")
        if should_log_per_pillar:
            for r in results:
                exp_x, exp_y = r["expected_xy"]
                self.logger.info(
                    "[SPAWNER VERIFY] "
                    f"name={r['name']} "
                    f"expected_xy=({exp_x:.3f},{exp_y:.3f}) "
                    f"exists={r['exists']} "
                    f"pose_source={r['pose_source']} "
                    f"actual_xy={r['actual_xy']} "
                    f"pose_error={r['pose_error']}"
                )
        self.logger.info(
            "[SPAWNER VERIFY SUMMARY] "
            f"status={status} "
            f"metadata_count={len(candidate_metadata)} "
            f"actual_existing_count={existing_count} "
            f"pose_info_count={len(pose_info_map)} "
            f"dynamic_pose_count={len(dynamic_pose_map)} "
            f"scene_count={len(scene_names)} "
            f"max_pose_error={max_pose_error}"
        )

        return {
            "ok": status == "ok",
            "status": status,
            "results": results,
            "metadata_count": len(candidate_metadata),
            "actual_existing_count": existing_count,
            "pose_info_count": len(pose_info_map),
            "dynamic_pose_count": len(dynamic_pose_map),
            "scene_count": len(scene_names),
            "max_pose_error": max_pose_error,
        }


    def sample_random_field_metadata(
        self,
        num_pillars=0,
        start=None,
        goal=None,
        name_prefix="pillar",
        corridor_half_width=2.5,
        start_clearance=2.5,
        goal_clearance=2.5,
        t_min=0.25,
        t_max=0.85,
        spawn_bounds=(-10.0, -9.0, 10.0, 9.0),
        pillar_radius_range=(0.2, 0.4),
        pillar_height_range=(4.0, 6.0),
        min_dist=2.0,
        corridor_jitter_deg=0.0,
    ):
        if num_pillars > 0 and spawn_world is not None:
            return spawn_world.sample_random_field_metadata(
                num_pillars=num_pillars,
                start=start,
                goal=goal,
                name_prefix=name_prefix,
                corridor_half_width=corridor_half_width,
                start_clearance=start_clearance,
                goal_clearance=goal_clearance,
                t_min=t_min,
                t_max=t_max,
                spawn_bounds=spawn_bounds,
                pillar_radius_range=pillar_radius_range,
                pillar_height_range=pillar_height_range,
                min_dist=min_dist,
                corridor_jitter_deg=corridor_jitter_deg,
            )
        return []

    def spawn_random_field(
        self,
        num_pillars=0,
        start=None,
        goal=None,
        name_prefix="pillar",
        corridor_half_width=2.5,
        start_clearance=2.5,
        goal_clearance=2.5,
        t_min=0.25,
        t_max=0.85,
        spawn_bounds=(-10.0, -9.0, 10.0, 9.0),
        pillar_radius_range=(0.2, 0.4),
        pillar_height_range=(4.0, 6.0),
        min_dist=2.0
    ):
        if num_pillars > 0 and spawn_world is not None:
            old_partition = os.environ.get("GZ_PARTITION")

            if self.gz_partition:
                os.environ["GZ_PARTITION"] = str(self.gz_partition)

            try:
                return spawn_world.spawn_random_field(
                    num_pillars=num_pillars,
                    world_name=self.world_name,
                    start=start,
                    goal=goal,
                    name_prefix=name_prefix,
                    corridor_half_width=corridor_half_width,
                    start_clearance=start_clearance,
                    goal_clearance=goal_clearance,
                    t_min=t_min,
                    t_max=t_max,
                    spawn_bounds=spawn_bounds,
                    pillar_radius_range=pillar_radius_range,
                    pillar_height_range=pillar_height_range,
                    min_dist=min_dist
                )
            finally:
                if old_partition is None:
                    os.environ.pop("GZ_PARTITION", None)
                else:
                    os.environ["GZ_PARTITION"] = old_partition
        return []

    def clear_pillars(self, num_pillars=50, name_prefix="pillar"):
        t0 = time.monotonic()
        if spawn_world is None:
            raise RuntimeError("spawn_world module is not available")

        spawn_world.clear_pillars(
            num_to_check=num_pillars,
            world_name=self.world_name,
            name_prefix=name_prefix,
            env=self._gz_env(),
        )
        self.logger.debug(
            f"Cleared pillars with prefix '{name_prefix}' in {time.monotonic()-t0:.2f}s"
        )


def make_bridge(
    gazebo_port,
    world="default",
    model_name="x500_depth_0",
    px4_ns="",
    target_system=1,
    gz_partition=None,
):
    if not rclpy.ok():
        rclpy.init()

    return ROSBridge(
        gazebo_port,
        world_name=world,
        model_name=model_name,
        px4_ns=px4_ns,
        target_system=target_system,
        gz_partition=gz_partition,
    )


def make_spawner(world="default", gz_partition=None, verbose_pillar_verify=False):
    return Spawner(
        world_name=world,
        gz_partition=gz_partition,
        verbose_pillar_verify=verbose_pillar_verify,
    )
