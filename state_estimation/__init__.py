"""VIO/state-estimation stack for GPS-denied navigation."""

from .config import CameraBridgeSpec, OpenVinsConfig
from .runtime import (
    start_openvins_camera_bridges,
    start_openvins_node,
    start_openvins_stack,
)

__all__ = [
    "CameraBridgeSpec",
    "OpenVinsConfig",
    "OpenVinsPx4Bridge",
    "VioStatus",
    "start_openvins_camera_bridges",
    "start_openvins_node",
    "start_openvins_stack",
]


def __getattr__(name):
    if name in {"OpenVinsPx4Bridge", "VioStatus"}:
        from .openvins_px4_bridge import OpenVinsPx4Bridge, VioStatus

        return {"OpenVinsPx4Bridge": OpenVinsPx4Bridge, "VioStatus": VioStatus}[name]
    raise AttributeError(name)
