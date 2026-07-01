"""Process launchers for the OpenVINS state-estimation stack."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess

from .config import OpenVinsConfig

logger = logging.getLogger(__name__)


def _source_line(path: str) -> str:
    quoted = shlex.quote(path)
    return f"if [ -f {quoted} ]; then source {quoted}; fi"


def _bridge_args(spec_topic: str, ros_topic: str, ros_type: str, gz_type: str) -> list[str]:
    spec = f"{spec_topic}@{ros_type}[{gz_type}"
    args = ["ros2", "run", "ros_gz_bridge", "parameter_bridge", spec]
    if spec_topic != ros_topic:
        args.extend(["--ros-args", "-r", f"{spec_topic}:={ros_topic}"])
    return args


def start_openvins_camera_bridges(
    *,
    config: OpenVinsConfig,
    gz_partition: str,
    ros_domain_id: int,
) -> list[tuple[str, subprocess.Popen]]:
    """Bridge Gazebo camera image topics into the ROS topics consumed by OpenVINS."""

    bridge_env = os.environ.copy()
    bridge_env["ROS_DOMAIN_ID"] = str(ros_domain_id)
    bridge_env["GZ_PARTITION"] = str(gz_partition)

    processes: list[tuple[str, subprocess.Popen]] = []
    for spec in config.camera_bridge_specs:
        args = _bridge_args(spec.gz_topic, spec.ros_topic, spec.ros_type, spec.gz_type)
        logger.info(
            "Starting state_estimation camera bridge %s ROS_DOMAIN_ID=%s "
            "GZ_PARTITION=%s gz=%s ros=%s",
            spec.label,
            ros_domain_id,
            gz_partition,
            spec.gz_topic,
            spec.ros_topic,
        )
        proc = subprocess.Popen(
            args,
            env=bridge_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        processes.append((spec.label, proc))
    return processes


def start_openvins_node(
    *,
    config: OpenVinsConfig,
    ros_domain_id: int,
) -> subprocess.Popen:
    """Start the OpenVINS ROS2 subscriber node."""

    ov_env = os.environ.copy()
    ov_env["ROS_DOMAIN_ID"] = str(ros_domain_id)

    setup_lines = [
        _source_line(f"/opt/ros/{config.ros_distro}/setup.bash"),
        _source_line(os.path.join(config.ros2_ws, "install", "setup.bash")),
        _source_line(os.path.join(config.openvins_ws, "install", "setup.bash")),
    ]
    ros_args = [
        "exec",
        "ros2",
        "run",
        "ov_msckf",
        "run_subscribe_msckf",
        config.config_path,
        "--ros-args",
        "-r",
        f"__ns:={config.namespace}",
        "-p",
        f"use_sim_time:={'true' if config.use_sim_time else 'false'}",
        "-p",
        f"config_path:={config.config_path}",
        "-p",
        f"topic_imu:={config.imu_topic}",
        "-p",
        f"topic_camera0:={config.cam0_topic}",
        "-p",
        f"topic_camera1:={config.cam1_topic}",
        "-p",
        f"use_stereo:={'true' if config.use_stereo else 'false'}",
        "-p",
        f"max_cameras:={int(config.max_cameras)}",
    ]
    shell_cmd = " && ".join(setup_lines + [" ".join(shlex.quote(arg) for arg in ros_args)])

    logger.info(
        "Starting OpenVINS node ROS_DOMAIN_ID=%s config=%s imu=%s cam0=%s cam1=%s",
        ros_domain_id,
        config.config_path,
        config.imu_topic,
        config.cam0_topic,
        config.cam1_topic,
    )
    return subprocess.Popen(
        ["bash", "-lc", shell_cmd],
        env=ov_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def start_openvins_stack(
    *,
    config: OpenVinsConfig,
    gz_partition: str,
    ros_domain_id: int,
) -> list[tuple[str, subprocess.Popen]]:
    """Start all OpenVINS-side state-estimation processes for one environment."""

    processes = start_openvins_camera_bridges(
        config=config,
        gz_partition=gz_partition,
        ros_domain_id=ros_domain_id,
    )
    processes.append(("openvins", start_openvins_node(config=config, ros_domain_id=ros_domain_id)))
    return processes
