"""
Subprocess-launcher utilities for SITL bridge processes.

Ported from obstacle_avoidance_mission/scripts/train.py (L143–290).
"""

import logging
import os
import signal
import subprocess

logger = logging.getLogger(__name__)


def start_microxrce_agent(rank: int, ros_domain_id: int) -> subprocess.Popen:
    """Deprecated: use start_microxrce_agent_single() instead."""
    agent_env = os.environ.copy()
    agent_env["ROS_DOMAIN_ID"] = str(ros_domain_id)

    port = 8888 + rank

    logger.info(
        f"[ENV {rank}] Starting MicroXRCEAgent "
        f"ROS_DOMAIN_ID={ros_domain_id}, port={port}"
    )

    return subprocess.Popen(
        ["MicroXRCEAgent", "udp4", "-p", str(port)],
        env=agent_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def start_microxrce_agent_single(port: int = 8888) -> subprocess.Popen:
    """Start ONE shared MicroXRCEAgent for ALL PX4 instances.

    Multiple PX4 clients distinguish themselves via UXRCE_DDS_KEY (set by
    PX4 rcS as px4_instance+1). Each client still gets its own DDS domain
    via UXRCE_DDS_DOM_ID=ROS_DOMAIN_ID. No ROS_DOMAIN_ID needed on the
    agent itself — the agent is domain-agnostic at transport level.
    """
    logger.info(f"[UXRCE] Starting single shared MicroXRCEAgent port={port}")

    return subprocess.Popen(
        ["MicroXRCEAgent", "udp4", "-p", str(port)],
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop_bridge_process(
    proc: "subprocess.Popen | None",
    timeout_term: float = 3.0,
    timeout_kill: float = 2.0,
) -> None:
    if proc is None:
        return

    if proc.poll() is not None:
        return

    logger.info(f"Stopping bridge process pid={proc.pid}")

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return

    try:
        proc.wait(timeout=timeout_term)
        return
    except subprocess.TimeoutExpired:
        pass

    logger.warning(f"Force killing bridge process pid={proc.pid}")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            return

    try:
        proc.wait(timeout=timeout_kill)
    except Exception:
        pass


def start_gz_pose_bridge(
    model_name: str,
    gz_partition: str = None,
    ros_domain_id: int = 0,
) -> subprocess.Popen:
    """
    Bridge Gazebo model pose (PosePublisher) sang ROS 2 Pose topic.

    Gazebo topic:
        /model/<model_name>/pose, gz.msgs.Pose
    ROS 2 topic:
        /model/<model_name>/pose, geometry_msgs/msg/Pose

    Không dùng TFMessage.
    """
    bridge_env = os.environ.copy()
    bridge_env["ROS_DOMAIN_ID"] = str(ros_domain_id)
    if gz_partition:
        bridge_env["GZ_PARTITION"] = str(gz_partition)

    gz_topic = f"/model/{model_name}/pose"
    pose_gz_type = os.environ.get("GZ_POSE_BRIDGE_TYPE", "gz.msgs.Pose")

    spec = f"{gz_topic}@geometry_msgs/msg/Pose[{pose_gz_type}"

    logger.info(
        f"Starting ros_gz model pose bridge "
        f"ROS_DOMAIN_ID={ros_domain_id}, "
        f"GZ_PARTITION={gz_partition or 'default'}, "
        f"topic={gz_topic}, "
        f"spec={spec}"
    )

    return subprocess.Popen(
        ["ros2", "run", "ros_gz_bridge", "parameter_bridge", spec],
        env=bridge_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def start_gz_clock_bridge(
    gz_partition: str = None,
    ros_domain_id: int = 0,
) -> subprocess.Popen:
    """Bridge Gazebo /clock to ROS 2 /clock for sim-time nodes."""
    bridge_env = os.environ.copy()
    bridge_env["ROS_DOMAIN_ID"] = str(ros_domain_id)
    if gz_partition:
        bridge_env["GZ_PARTITION"] = str(gz_partition)

    spec = "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"

    logger.info(
        f"Starting ros_gz clock bridge "
        f"ROS_DOMAIN_ID={ros_domain_id}, "
        f"GZ_PARTITION={gz_partition or 'default'}, "
        f"spec={spec}"
    )

    return subprocess.Popen(
        ["ros2", "run", "ros_gz_bridge", "parameter_bridge", spec],
        env=bridge_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def start_gz_depth_bridge(
    model_name: str,
    gz_partition: str,
    ros_domain_id: int,
) -> subprocess.Popen:
    """
    Bridge Gazebo depth_camera sang ROS 2 Image topic.

    Gazebo publish:  /depth_camera  (gz.msgs.Image)
    ROS 2 receive:   /camera/depth/image_raw  (sensor_msgs/msg/Image)
    """
    bridge_env = os.environ.copy()
    bridge_env["ROS_DOMAIN_ID"] = str(ros_domain_id)
    bridge_env["GZ_PARTITION"] = str(gz_partition)

    spec = "/depth_camera@sensor_msgs/msg/Image[gz.msgs.Image"

    logger.info(
        f"Starting ros_gz depth bridge "
        f"ROS_DOMAIN_ID={ros_domain_id}, "
        f"GZ_PARTITION={gz_partition}, "
        f"Gazebo: /depth_camera -> ROS 2: /camera/depth/image_raw"
    )

    return subprocess.Popen(
        ["ros2", "run", "ros_gz_bridge", "parameter_bridge", spec,
         "--ros-args", "-r", "/depth_camera:=/camera/depth/image_raw"],
        env=bridge_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def start_gz_lidar_bridge(
    model_name: str,
    gz_partition: str,
    ros_domain_id: int,
) -> subprocess.Popen:
    """
    Bridge Gazebo 2D LiDAR scan to ROS 2 LaserScan topic.

    Gazebo publish:  /lidar_2d_v2/scan  (gz.msgs.LaserScan)
    ROS 2 receive:   /lidar/scan        (sensor_msgs/msg/LaserScan)
    """
    bridge_env = os.environ.copy()
    bridge_env["ROS_DOMAIN_ID"] = str(ros_domain_id)
    bridge_env["GZ_PARTITION"] = str(gz_partition)

    spec = "/lidar_2d_v2/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"

    logger.info(
        f"Starting ros_gz lidar bridge "
        f"ROS_DOMAIN_ID={ros_domain_id}, GZ_PARTITION={gz_partition}, "
        f"Gazebo: /lidar_2d_v2/scan -> ROS 2: /lidar/scan"
    )

    return subprocess.Popen(
        ["ros2", "run", "ros_gz_bridge", "parameter_bridge", spec,
         "--ros-args", "-r", "/lidar_2d_v2/scan:=/lidar/scan"],
        env=bridge_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
