"""Frame conversions between ROS/OpenVINS and PX4 conventions."""

from __future__ import annotations

import math
from collections.abc import Sequence


def diag3(values, default: float) -> list[float]:
    if values is None or len(values) != 36:
        return [default, default, default]
    diag = [float(values[0]), float(values[7]), float(values[14])]
    if any(not math.isfinite(v) for v in diag):
        return [default, default, default]
    if max(abs(v) for v in diag) <= 0.0:
        return [default, default, default]
    return [max(0.0, v) for v in diag]


def orientation_diag3(values, default: float) -> list[float]:
    if values is None or len(values) != 36:
        return [default, default, default]
    diag = [float(values[21]), float(values[28]), float(values[35])]
    if any(not math.isfinite(v) for v in diag):
        return [default, default, default]
    if max(abs(v) for v in diag) <= 0.0:
        return [default, default, default]
    return [max(0.0, v) for v in diag]


def quat_normalize_wxyz(q: Sequence[float]) -> list[float]:
    if len(q) != 4:
        return [1.0, 0.0, 0.0, 0.0]
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm <= 1e-9 or not math.isfinite(norm):
        return [1.0, 0.0, 0.0, 0.0]
    return [float(v) / norm for v in q]


def quat_enu_flu_to_ned_frd(q_enu_flu: Sequence[float]) -> list[float]:
    """Convert Hamilton quaternion [w, x, y, z] from ROS ENU/FLU to PX4 NED/FRD."""

    w, x, y, z = quat_normalize_wxyz(q_enu_flu)
    s = 0.7071067811865476
    return quat_normalize_wxyz(
        [
            s * (w + z),
            s * (x + y),
            s * (x - y),
            s * (w - z),
        ]
    )


def enu_position_to_ned(position_enu: Sequence[float]) -> list[float]:
    x, y, z = [float(v) for v in position_enu[:3]]
    return [y, x, -z]


def enu_velocity_to_ned(velocity_enu: Sequence[float]) -> list[float]:
    vx, vy, vz = [float(v) for v in velocity_enu[:3]]
    return [vy, vx, -vz]


def frd_vector_to_flu(vector_frd: Sequence[float]) -> list[float]:
    x, y, z = [float(v) for v in vector_frd[:3]]
    return [x, -y, -z]


def covariance3(diag_value: float) -> list[float]:
    return [
        float(diag_value),
        0.0,
        0.0,
        0.0,
        float(diag_value),
        0.0,
        0.0,
        0.0,
        float(diag_value),
    ]
