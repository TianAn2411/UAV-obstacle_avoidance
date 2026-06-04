#!/usr/bin/env python3
"""
LiDAR sector alignment test.

Usage:
    source ~/drone_rl_env/bin/activate
    cd ~/PX4-Autopilot
    ROS_DOMAIN_ID=30 GZ_PARTITION=drone_rl_0 python3 -m obstacle_avoidance.test_lidar_alignment

Requires: Gazebo running with x500_depth_0 + lidar ros_gz_bridge active.
If bridge not running, script prints the exact command to start it.

Spawns a pillar 3m in front of drone origin, prints which sector detects it.
Expected: sector 17 or 18 (0-indexed) = LiDAR 0 degrees aligned with drone forward.
Sector layout: index 0 = -135deg (hard left), index 35 = +135deg (hard right).
"""

import os
import sys
import time

import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from obstacle_avoidance.utils.gz_transport_client import GzTransportClient
from obstacle_avoidance.utils.spawn_world import spawn_pillar

NUM_SECTORS = 36
MIN_RANGE = 0.1
MAX_RANGE = 30.0
WORLD = os.environ.get("GZ_WORLD", "default_0")
MODEL_NAME = os.environ.get("GZ_MODEL", "x500_depth_0")
# ros_gz_bridge maps the scoped GZ topic to /lidar/scan.
# For testing with scoped topic (no flat topic in SDF yet), run:
#   ros2 run ros_gz_bridge parameter_bridge \
#     '/world/default_0/model/x500_depth_0/link/link/sensor/lidar_2d_v2/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan' \
#     --ros-args -r '/world/default_0/model/x500_depth_0/link/link/sensor/lidar_2d_v2/scan:=/lidar/scan'
LIDAR_TOPIC = "/lidar/scan"
PILLAR_NAME = "test_lidar_align_pillar"
PILLAR_X = 3.0  # 3m ahead — drone spawns at origin facing +X (East/ENU)
PILLAR_Y = 0.0
SPIN_SECS = 2.0


class LidarTestNode(Node):
    def __init__(self):
        super().__init__("lidar_alignment_test")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.lidar_raw = np.ones(1080, dtype=np.float32) * MAX_RANGE
        self.scan_count = 0
        self.create_subscription(LaserScan, LIDAR_TOPIC, self._cb, qos)

    def _cb(self, msg):
        r = np.array(msg.ranges, dtype=np.float32)
        # posinf = beam exceeded max range -> clear space
        r[np.isposinf(r)] = MAX_RANGE
        # nan / neginf / <=0 = signal error -> treat as close obstacle (conservative)
        r[np.isnan(r) | np.isneginf(r) | (r <= 0.0)] = MIN_RANGE
        self.lidar_raw = np.clip(r, MIN_RANGE, MAX_RANGE)
        self.scan_count += 1


def compute_sectors(scan, n=NUM_SECTORS):
    sps = len(scan) // n  # samples per sector: 1080 // 36 = 30
    return np.array(
        [float(np.min(scan[i * sps:(i + 1) * sps])) for i in range(n)],
        dtype=np.float32,
    )


def print_sectors(sectors, label, baseline=None):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  {'Idx':>3}  {'Angle range':^20}  {'Min dist':>8}  {'Delta':>8}")
    print(f"  {'-' * 3}  {'-' * 20}  {'-' * 8}  {'-' * 8}")
    closest = int(np.argmin(sectors))
    for i, v in enumerate(sectors):
        a0 = -135.0 + i * 7.5
        a1 = a0 + 7.5
        delta = f"{v - baseline[i]:+.2f}" if baseline is not None else "     —"
        marker = "  <-- CLOSEST" if i == closest else ""
        print(f"  {i:>3}  {a0:>7.1f} -> {a1:>6.1f} deg  {v:>8.2f}m  {delta:>8}{marker}")
    print()


def spin_collect(node, executor, seconds):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        executor.spin_once(timeout_sec=0.05)


def main():
    if not os.environ.get("ROS_DOMAIN_ID"):
        print("ERROR: ROS_DOMAIN_ID not set.")
        print("Run: ROS_DOMAIN_ID=30 GZ_PARTITION=drone_rl_0 python3 -m obstacle_avoidance.test_lidar_alignment")
        sys.exit(1)

    rclpy.init()
    node = LidarTestNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    print(f"\n[TEST] Collecting baseline scan ({SPIN_SECS}s, world={WORLD})...")
    spin_collect(node, executor, SPIN_SECS)

    if node.scan_count == 0:
        print(f"ERROR: No messages on {LIDAR_TOPIC}. Is ros_gz_bridge running?")
        print("Start bridge (single quotes required to prevent shell glob on '['):")
        print(f"  ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID','30')} GZ_PARTITION={os.environ.get('GZ_PARTITION','drone_rl_0')} \\")
        print(f"  ros2 run ros_gz_bridge parameter_bridge \\")
        print(f"  '/world/{WORLD}/model/{MODEL_NAME}/link/link/sensor/lidar_2d_v2/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan' \\")
        print(f"  --ros-args -r '/world/{WORLD}/model/{MODEL_NAME}/link/link/sensor/lidar_2d_v2/scan:=/lidar/scan'")
        rclpy.shutdown()
        sys.exit(1)

    baseline = compute_sectors(node.lidar_raw)
    print_sectors(baseline, "BASELINE (no pillar)")

    # Spawn test pillar 3m ahead of drone origin
    gz_env = {**os.environ}
    print(f"[TEST] Spawning '{PILLAR_NAME}' at ({PILLAR_X}, {PILLAR_Y}) ...")
    spawn_pillar(PILLAR_NAME, PILLAR_X, PILLAR_Y, radius=0.3, height=3.0, world_name=WORLD, env=gz_env)
    time.sleep(0.5)

    print(f"[TEST] Collecting scan with pillar ({SPIN_SECS}s)...")
    spin_collect(node, executor, SPIN_SECS)

    with_pillar = compute_sectors(node.lidar_raw)
    print_sectors(with_pillar, f"WITH PILLAR at ({PILLAR_X}m ahead)", baseline=baseline)

    # Result
    closest = int(np.argmin(with_pillar))
    angle_lo = -135.0 + closest * 7.5
    angle_hi = angle_lo + 7.5
    print(f"[RESULT] Closest sector : {closest}")
    print(f"[RESULT] Angle range    : {angle_lo:.1f} -> {angle_hi:.1f} deg")
    print(f"[RESULT] Distance       : {with_pillar[closest]:.2f}m")

    if closest in (17, 18):
        print("[RESULT] PASS -- LiDAR 0 deg aligned with drone forward (+X / East)")
    else:
        offset = (closest - 17.5) * 7.5
        print(f"[RESULT] FAIL -- expected sector 17 or 18, got {closest}")
        print(f"[RESULT] Mounting offset from drone nose: {offset:+.1f} deg")

    # Cleanup
    gz_client = GzTransportClient(gz_partition=os.environ.get("GZ_PARTITION"))
    if gz_client.available():
        removed = gz_client.remove_model(WORLD, PILLAR_NAME)
        if removed:
            print(f"[TEST] Removed '{PILLAR_NAME}'")
        else:
            print(f"[TEST] Warning: remove_model returned False for '{PILLAR_NAME}'")
    else:
        print(f"[TEST] Warning: GzTransportClient unavailable. Remove manually:")
        print(f"  gz service -s /world/{WORLD}/remove --reqtype gz.msgs.Entity --req 'name: \"{PILLAR_NAME}\" type: 2'")

    rclpy.shutdown()
    print("[TEST] Done.")


if __name__ == "__main__":
    main()
