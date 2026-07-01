#!/usr/bin/env python3
"""Benchmark symbolic_extractor pipeline latency (no ROS needed).

Run from obstacle_avoidance/ dir:
    python3 utils/bench_extractor.py
"""
import sys, time, os
import numpy as np

_repo = str(__import__("pathlib").Path(__file__).resolve().parents[2])
_oa   = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, _repo)
sys.path.insert(0, _oa)

from symbolic_extractor.configs import ExtractorConfig
from symbolic_extractor.filters import DepthProcessor, LidarProcessor, CloudCleaner
from symbolic_extractor.pipeline import HALOPipeline


# ── Mock objects (no ROS) ────────────────────────────────────────────────────

class MockDepthMsg:
    """Minimal sensor_msgs/Image stub (32FC1, sim resolution 640×480)."""
    encoding = "32FC1"
    height, width = 480, 640

    def __init__(self):
        arr = np.random.uniform(1.5, 8.0, (self.height, self.width)).astype(np.float32)
        # punch a fake obstacle region ~3m out
        arr[200:280, 280:360] = 3.0
        self.data = arr.tobytes()


class MockOdom:
    """Minimal px4_msgs/VehicleOdometry stub."""
    def __init__(self, altitude: float = 1.5):
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity [w,x,y,z]
        # NED: z = -altitude
        self.position = np.array([0.0, 0.0, -altitude], dtype=np.float32)


class MockTFManager:
    """Identity-transform TFManager — no ROS, no rclpy."""
    def __init__(self):
        self._fallback_pitch = 0.0
        self._fallback_yaw   = 0.0

    def invalidate_latest_cache(self):
        pass

    def set_fallback_odom(self, odom):
        pass  # identity quaternion → yaw=0, pitch=0

    def transform_cloud(self, cloud, src, dst, stamp=None):
        return cloud  # identity transform

    def get_yaw_from_odom(self, odom) -> float:
        return 0.0

    def get_pitch_at_stamp(self, stamp) -> float:
        return 0.0


# ── Setup ────────────────────────────────────────────────────────────────────

cfg      = ExtractorConfig()
tf_mock  = MockTFManager()
pipeline = HALOPipeline(tf_manager=tf_mock, config=cfg)

depth_msg = MockDepthMsg()
odom      = MockOdom()

print(f"Config: {cfg.arena_size_m}m arena, voxel={cfg.voxel_size}m, "
      f"downsample={cfg.downsample_factor}, BEV={cfg.tensor_size}px")

# ── Warmup (Numba JIT compile on first call) ─────────────────────────────────
print("Warming up (Numba JIT compile)...", flush=True)
for i in range(15):
    pipeline.process(depth_msg, None, odom)
    if i == 0:
        print(f"  After 1st call (JIT done): {pipeline.last_latency_ms:.1f} ms")
print(f"  After warmup: {pipeline.last_latency_ms:.1f} ms")
print(f"  Breakdown: {pipeline.last_breakdown}")

# ── Benchmark ────────────────────────────────────────────────────────────────
N = 200
times  = []
breakdowns = []

for _ in range(N):
    pipeline.process(depth_msg, None, odom)
    times.append(pipeline.last_latency_ms)
    breakdowns.append(pipeline.last_breakdown)

times = np.array(times)

# parse per-step breakdown from last sample
print(f"\nPipeline latency (N={N}, CPU, depth-only no lidar):")
print(f"  mean  : {times.mean():.2f} ms")
print(f"  median: {np.median(times):.2f} ms")
print(f"  p95   : {np.percentile(times, 95):.2f} ms")
print(f"  p99   : {np.percentile(times, 99):.2f} ms")
print(f"  max   : {times.max():.2f} ms")
print(f"\nLast breakdown: {breakdowns[-1]}")
print(f"\nMax throughput: {1000/times.mean():.0f} Hz")
print(f"At 30 Hz control: {times.mean():.2f} ms / 33.3 ms budget = "
      f"{times.mean()/33.3*100:.0f}% budget used")
