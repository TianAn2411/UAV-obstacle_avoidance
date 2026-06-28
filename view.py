#!/usr/bin/env python3
"""Check where depth cloud points land in world frame after FLU fix."""
import sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from px4_msgs.msg import VehicleOdometry

N_FRAMES = 3

class CloudViewer(Node):
    def __init__(self):
        super().__init__("cloud_viewer")
        self.drone_pos = None
        self.done = False
        self._frames = 0

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.create_subscription(PointCloud2, "/symbolic_extractor/cloud_colored",
                                 self._cloud_cb, 10)
        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry",
                                 self._odom_cb, px4_qos)

    def _odom_cb(self, msg):
        x_n, y_e, z_d = msg.position
        self.drone_pos = (float(y_e), float(x_n), float(-z_d))

    def _cloud_cb(self, msg):
        self._frames += 1
        if self._frames > N_FRAMES:
            self.done = True
            return

        pts = self._parse_cloud(msg)
        pos = self.drone_pos or (0.0, 0.0, 0.0)
        n = len(pts)
        if n == 0:
            print(f"[FRAME {self._frames}] 0 pts"); return

        # Relative to drone
        dx = pts[:, 0] - pos[0]  # East offset
        dy = pts[:, 1] - pos[1]  # North offset
        dz = pts[:, 2] - pos[2]  # Up offset

        horiz = np.sqrt(dx**2 + dy**2)

        print(f"\n[FRAME {self._frames}] {n} pts  drone=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
        print(f"  World X(East):  [{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]  mean={pts[:,0].mean():.2f}")
        print(f"  World Y(North): [{pts[:,1].min():.2f}, {pts[:,1].max():.2f}]  mean={pts[:,1].mean():.2f}")
        print(f"  World Z(Up):    [{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]  mean={pts[:,2].mean():.2f}")
        print(f"  Rel East  dX:   [{dx.min():.2f}, {dx.max():.2f}]")
        print(f"  Rel North dY:   [{dy.min():.2f}, {dy.max():.2f}]")
        print(f"  Rel Up    dZ:   [{dz.min():.2f}, {dz.max():.2f}]")
        print(f"  Horiz dist:     [{horiz.min():.2f}, {horiz.max():.2f}]  mean={horiz.mean():.2f}")

        # Histogram: which horizontal direction do most points go?
        angles = np.degrees(np.arctan2(dy, dx))  # CCW from East
        bins = ['E(±22)', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
        edges = np.linspace(-180, 180, 9)
        counts, _ = np.histogram(angles, bins=edges)
        print(f"  Azimuth histogram (E=0°,N=90°,W=180°/−180°):")
        for b, c in zip(bins, counts):
            bar = '#' * (c * 40 // max(counts, default=1))
            print(f"    {b:8s}: {c:4d} {bar}")

        # Points near drone (< 4m horiz) — likely real obstacles
        near = horiz < 4.0
        if near.sum():
            print(f"  Near pts (horiz<4m): {near.sum()}")
            for pt in pts[near][:10]:
                d = np.sqrt((pt[0]-pos[0])**2 + (pt[1]-pos[1])**2)
                print(f"    world({pt[0]:.2f},{pt[1]:.2f},{pt[2]:.2f})  horiz={d:.2f}m")

    def _parse_cloud(self, msg: PointCloud2):
        offsets = {f.name: f.offset for f in msg.fields if f.name in ('x','y','z')}
        if not all(k in offsets for k in ('x','y','z')):
            return np.zeros((0, 3))
        n = msg.width * msg.height
        ps = msg.point_step
        data = bytes(msg.data)
        x = np.frombuffer(data, dtype=np.float32)[offsets['x']//4::ps//4][:n]
        y = np.frombuffer(data, dtype=np.float32)[offsets['y']//4::ps//4][:n]
        z = np.frombuffer(data, dtype=np.float32)[offsets['z']//4::ps//4][:n]
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        return np.column_stack([x[valid], y[valid], z[valid]])


def main():
    rclpy.init()
    node = CloudViewer()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    print("Checking world-frame cloud distribution...", flush=True)
    t0 = time.time()
    while not node.done:
        executor.spin_once(timeout_sec=0.05)
        if time.time() - t0 > 15.0:
            print("TIMEOUT"); break
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
