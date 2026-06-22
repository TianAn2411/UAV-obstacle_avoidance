
from __future__ import annotations

import contextlib
import fcntl
import logging
import math
import os
import re

try:
    import gz.transport13 as gz_transport
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.entity_pb2 import Entity
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    from gz.msgs10.empty_pb2 import Empty
    from gz.msgs10.scene_pb2 import Scene
except Exception:
    gz_transport = None
    Boolean = None
    Pose = None
    Pose_V = None
    Entity = None
    EntityFactory = None
    Empty = None
    Scene = None

try:
    from gz.msgs10.wind_pb2 import Wind as GzWind
except Exception:
    GzWind = None


@contextlib.contextmanager
def gz_partition_env(gz_partition):
    old_value = os.environ.get("GZ_PARTITION")

    try:
        if gz_partition:
            os.environ["GZ_PARTITION"] = str(gz_partition)
        else:
            os.environ.pop("GZ_PARTITION", None)
        yield
    finally:
        if old_value is None:
            os.environ.pop("GZ_PARTITION", None)
        else:
            os.environ["GZ_PARTITION"] = old_value


@contextlib.contextmanager
def gz_service_lock(path=None, gz_partition=None):
    if path is None:
        # Per-partition lock: each env gets its own lock file so
        # envs with separate Gazebo worlds never block each other.
        suffix = f"_{gz_partition}" if gz_partition else ""
        path = f"/tmp/drone_rl_gz_service{suffix}.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class GzTransportClient:
    def __init__(self, gz_partition=None, use_lock=True, logger=None):
        self.gz_partition = gz_partition or os.environ.get("GZ_PARTITION", "")
        self.use_lock = bool(use_lock)
        self.logger = logger or logging.getLogger(__name__)
        self.node = None

        if gz_transport is not None:
            try:
                self.node = gz_transport.Node()
            except Exception as exc:
                self.logger.warning(f"[GZ TRANSPORT] node init failed: {exc}")
                self.node = None

    def available(self) -> bool:
        return self.node is not None

    def _lock_context(self):
        if self.use_lock:
            return gz_service_lock(gz_partition=self.gz_partition)
        return contextlib.nullcontext()

    def _bool_response_ok(self, rep) -> bool:
        if rep is None:
            return False
        if isinstance(rep, bool):
            return rep
        if Boolean is not None and isinstance(rep, Boolean):
            return bool(getattr(rep, "data", False))
        if hasattr(rep, "data"):
            return bool(getattr(rep, "data"))
        return bool(rep)

    def _request(self, service, req, req_type, rep_type, timeout_ms):
        if not self.available():
            return False
        if req_type is None or rep_type is None:
            return False

        try:
            with gz_partition_env(self.gz_partition):
                with self._lock_context():
                    result = self.node.request(
                        str(service),
                        req,
                        req_type,
                        rep_type,
                        int(timeout_ms),
                    )
        except Exception as exc:
            self.logger.warning(
                f"[GZ TRANSPORT] request failed service={service} error={exc}"
            )
            return False

        ok = False
        rep = None

        if isinstance(result, tuple):
            if len(result) >= 2:
                first, second = result[0], result[1]
                if isinstance(first, bool):
                    ok = first
                    rep = second
                elif isinstance(second, bool):
                    ok = second
                    rep = first
                else:
                    rep = second
                    ok = bool(first)
            elif len(result) == 1:
                ok = bool(result[0])
        elif isinstance(result, bool):
            ok = result
        else:
            rep = result
            ok = self._bool_response_ok(rep)

        if rep_type is Boolean:
            if rep is None:
                return bool(ok)
            return bool(ok and self._bool_response_ok(rep))

        if not ok:
            return False
        return rep if rep is not None else True

    def make_pose_msg(self, name, x, y, z, yaw=0.0):
        if Pose is None:
            return None

        pose = Pose()
        pose.name = str(name)
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)

        qz = math.sin(float(yaw) * 0.5)
        qw = math.cos(float(yaw) * 0.5)
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)
        return pose

    def set_pose(self, world_name, name, x, y, z, yaw=0.0, timeout_ms=2000) -> bool:
        pose = self.make_pose_msg(name, x, y, z, yaw=yaw)
        if pose is None:
            return False

        result = self._request(
            service=f"/world/{world_name}/set_pose",
            req=pose,
            req_type=Pose,
            rep_type=Boolean,
            timeout_ms=timeout_ms,
        )
        return bool(result)

    def set_pose_vector(self, world_name, poses, timeout_ms=2500) -> bool:
        if Pose_V is None or Pose is None:
            return False

        pose_v = Pose_V()
        for item in poses:
            if isinstance(item, dict):
                name = item["name"]
                x, y, z = item["x"], item["y"], item["z"]
                yaw = item.get("yaw", 0.0)
            elif len(item) == 4:
                name, x, y, z = item
                yaw = 0.0
            elif len(item) == 5:
                name, x, y, z, yaw = item
            else:
                return False

            pose_msg = self.make_pose_msg(name, x, y, z, yaw=yaw)
            if pose_msg is None:
                return False
            pose_v.pose.add().CopyFrom(pose_msg)

        result = self._request(
            service=f"/world/{world_name}/set_pose_vector",
            req=pose_v,
            req_type=Pose_V,
            rep_type=Boolean,
            timeout_ms=timeout_ms,
        )
        return bool(result)

    def create_sdf(self, world_name, name, sdf, timeout_ms=3000) -> bool:
        if EntityFactory is None:
            return False

        req = EntityFactory()
        req.name = str(name)
        req.sdf = str(sdf)
        if hasattr(req, "allow_renaming"):
            req.allow_renaming = False

        result = self._request(
            service=f"/world/{world_name}/create",
            req=req,
            req_type=EntityFactory,
            rep_type=Boolean,
            timeout_ms=timeout_ms,
        )
        return bool(result)

    def remove_model(self, world_name, name, timeout_ms=2000) -> bool:
        if Entity is None:
            return False

        req = Entity()
        req.name = str(name)
        req.type = getattr(Entity, "MODEL", 2)

        result = self._request(
            service=f"/world/{world_name}/remove",
            req=req,
            req_type=Entity,
            rep_type=Boolean,
            timeout_ms=timeout_ms,
        )
        return bool(result)

    def scene_entity_names(self, world_name, timeout_ms=2000):
        if Empty is None or Scene is None:
            return None

        req = Empty()
        result = self._request(
            service=f"/world/{world_name}/scene/info",
            req=req,
            req_type=Empty,
            rep_type=Scene,
            timeout_ms=timeout_ms,
        )
        if not result or isinstance(result, bool):
            return None

        names = set()
        try:
            for model in getattr(result, "model", []):
                name = getattr(model, "name", "")
                if name:
                    names.add(str(name))
        except Exception:
            pass

        if names:
            return names

        try:
            text = str(result)
            return set(re.findall(r'name:\s*"([^"]+)"', text))
        except Exception:
            return None

    def subscribe_pose_v(self, world_name, callback) -> bool:
        if not self.available() or Pose_V is None:
            return False

        topic = f"/world/{world_name}/dynamic_pose/info"
        try:
            with gz_partition_env(self.gz_partition):
                result = self.node.subscribe(Pose_V, topic, callback)
        except Exception as exc:
            self.logger.warning(
                f"[GZ TRANSPORT] subscribe failed topic={topic} error={exc}"
            )
            return False

        if result is None:
            return True
        return bool(result)

    def set_wind(self, world_name: str, wx: float, wy: float, wz: float = 0.0) -> bool:
        """Publish horizontal wind velocity (ENU m/s) to Gazebo WindEffects plugin."""
        if not self.available():
            return False

        if GzWind is not None:
            try:
                topic = f"/world/{world_name}/wind"
                with gz_partition_env(self.gz_partition):
                    pub = self.node.advertise(topic, GzWind)
                    msg = GzWind()
                    msg.linear_velocity.x = float(wx)
                    msg.linear_velocity.y = float(wy)
                    msg.linear_velocity.z = float(wz)
                    ok = pub.publish(msg)
                return bool(ok)
            except Exception as exc:
                self.logger.debug(f"[GZ WIND] gz.msgs Wind publish failed: {exc}")

        # Fallback: subprocess gz topic publish
        try:
            import subprocess
            payload = f"linear_velocity: {{x: {wx:.4f}, y: {wy:.4f}, z: {wz:.4f}}}"
            env = dict(__import__("os").environ)
            if self.gz_partition:
                env["GZ_PARTITION"] = str(self.gz_partition)
            result = subprocess.run(
                ["gz", "topic", "-t", f"/world/{world_name}/wind",
                 "-m", "gz.msgs.Wind", "-p", payload],
                timeout=1.0, capture_output=True, env=env,
            )
            return result.returncode == 0
        except Exception as exc:
            self.logger.debug(f"[GZ WIND] subprocess fallback failed: {exc}")
            return False
