"""
PX4InstanceManager — manages exactly one PX4 SITL instance.

Ported from obstacle_avoidance_mission/scripts/train.py (L293–978).

Skipped (diagnostic / unreliable):
    _snapshot_pillars     L425–491
    _log_pillar_snapshot  L493–507
    _gz_entity_exists     L636–660
    wait_model_gone       L662–675
"""

import logging
import os
import subprocess
import time
from typing import Optional

from obstacle_avoidance.utils.process_utils import stop_bridge_process

try:
    from obstacle_avoidance.utils.gz_transport_client import GzTransportClient
except Exception:
    GzTransportClient = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class PX4InstanceManager:
    """
    Manages exactly one PX4 instance.

    Stable principle:
        subprocess.Popen([px4_bin, "-i", str(rank)], ...)

    Features:
        - Kill the correct instance by command line
        - Remove the correct Gazebo model best-effort
        - Restart with the correct partition/domain
        - hard_reset only returns True when set_pose probe succeeds
        - Initial startup uses PX4_GZ_RUN=1 to launch Gazebo flow as before
        - HARD_RESET_PX4_ONLY=1 uses PX4_GZ_RUN=0 (px4-only reconnect)
        - HARD_RESET_PX4_ONLY=0 or unset uses full PX4 + model respawn
    """

    def __init__(
        self,
        rank: int,
        ros_domain: int,
        partition: str,
        px4_bin: str,
        rootfs_dir: str,
        env_vars: dict,
        model_name: str,
        world: str = "default",
        start_pose: str = "0.0,0.0,0.0,0.0,0.0,0.0",
        startup_sleep: float = 8.0,
        bridge_processes: Optional[list] = None,
    ) -> None:
        self.rank = int(rank)
        self.ros_domain = int(ros_domain)
        self.partition = str(partition)
        self.gz_partition = self.partition

        self.px4_bin = px4_bin
        self.rootfs_dir = rootfs_dir
        self.env_vars = env_vars.copy()

        self.model_name = model_name
        self.world = world
        self.world_name = self.world

        self.start_pose = str(start_pose)
        self.startup_sleep = float(startup_sleep)
        self.proc: Optional[subprocess.Popen] = None
        self.bridge_processes: list = list(bridge_processes or [])
        self.gz_client = (
            GzTransportClient(gz_partition=self.partition, use_lock=True, logger=logger)
            if GzTransportClient is not None
            else None
        )
        # HARD_RESET_PX4_ONLY=1 means PX4-only reconnect.
        # HARD_RESET_PX4_ONLY=0 or unset means full PX4 + model respawn.
        self.hard_reset_px4_only: bool = os.environ.get("HARD_RESET_PX4_ONLY", "0") == "1"
        self.hard_reset_pillar_diagnostic: bool = (
            os.environ.get("HARD_RESET_PILLAR_DIAGNOSTIC", "0") == "1"
        )

    def start(self, gz_run: bool = True) -> subprocess.Popen:
        """
        Start PX4 instance.

        - Initial startup should use gz_run=True (PX4_GZ_RUN=1) so the normal
          Gazebo launch/spawn flow remains unchanged.
        - Hard-reset reconnect should use gz_run=False (PX4_GZ_RUN=0) to
          reconnect to the existing Gazebo world/model.
        """
        self.env_vars["ROS_DOMAIN_ID"] = str(self.ros_domain)
        self.env_vars["PX4_INSTANCE"] = str(self.rank)
        self.env_vars["GZ_PARTITION"] = self.partition

        self.env_vars["PX4_GZ_WORLD"] = self.world
        self.env_vars["PX4_GZ_RUN"] = "1" if gz_run else "0"

        if gz_run:
            self.env_vars["PX4_GZ_MODEL_POSE"] = self.start_pose
            self.env_vars["PX4_SIM_MODEL"] = "gz_x500_depth"
            self.env_vars.pop("PX4_GZ_MODEL_NAME", None)
            self.env_vars.pop("PX4_SYS_AUTOSTART", None)
        else:
            # Reconnect to the existing Gazebo model. px4-rc.simulator chooses
            # attach mode only when PX4_GZ_MODEL_NAME is set and PX4_SIM_MODEL is absent.
            self.env_vars["PX4_GZ_MODEL_NAME"] = self.model_name
            self.env_vars["PX4_SYS_AUTOSTART"] = "4002"
            self.env_vars.pop("PX4_SIM_MODEL", None)
            self.env_vars.pop("PX4_GZ_MODEL_POSE", None)

        start_mode = "initial_gazebo_launch" if gz_run else "hard_reset_px4_only"
        logger.info(
            "[PX4 START CONFIG] "
            f"env_id={self.rank} "
            f"mode={start_mode} "
            f"gz_partition={self.partition} "
            f"PX4_GZ_RUN={self.env_vars.get('PX4_GZ_RUN')} "
            f"PX4_GZ_WORLD={self.env_vars.get('PX4_GZ_WORLD')} "
            f"PX4_SIM_MODEL={self.env_vars.get('PX4_SIM_MODEL')} "
            f"PX4_GZ_MODEL_NAME={self.env_vars.get('PX4_GZ_MODEL_NAME')} "
            f"PX4_SYS_AUTOSTART={self.env_vars.get('PX4_SYS_AUTOSTART')} "
            f"PX4_GZ_MODEL_POSE={self.env_vars.get('PX4_GZ_MODEL_POSE')} "
        )

        logger.info(
            f"[PX4 MANAGER {self.rank}] ENV CHECK "
            f"ROS_DOMAIN_ID={self.env_vars.get('ROS_DOMAIN_ID')} "
            f"GZ_PARTITION={self.env_vars.get('GZ_PARTITION')} "
            f"PX4_INSTANCE={self.env_vars.get('PX4_INSTANCE')} "
            f"PX4_UXRCE_DDS_PORT={self.env_vars.get('PX4_UXRCE_DDS_PORT')} "
            f"UXRCE_DDS_PORT={self.env_vars.get('UXRCE_DDS_PORT')} "
            f"PX4_GZ_MODEL_POSE={self.env_vars.get('PX4_GZ_MODEL_POSE')} "
            f"PX4_SIM_MODEL={self.env_vars.get('PX4_SIM_MODEL')} "
            f"PX4_GZ_MODEL_NAME={self.env_vars.get('PX4_GZ_MODEL_NAME')} "
            f"PX4_SYS_AUTOSTART={self.env_vars.get('PX4_SYS_AUTOSTART')} "
            f"PX4_GZ_WORLD={self.env_vars.get('PX4_GZ_WORLD')}"
        )

        self.proc = subprocess.Popen(
            [self.px4_bin, "-i", str(self.rank)],
            cwd=self.rootfs_dir,
            env=self.env_vars,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Sleep in small chunks to allow KeyboardInterrupt
        sleep_remaining = self.startup_sleep
        sleep_chunk = 0.5
        while sleep_remaining > 0:
            time.sleep(min(sleep_chunk, sleep_remaining))
            sleep_remaining -= sleep_chunk
        return self.proc

    def kill(self) -> None:
        """
        Kill PX4 instance by command line.

        Does not use self.proc/process group because in WSL/PX4 SITL
        self.proc sometimes no longer points to the real process.
        """
        logger.debug(f"[PX4 MANAGER {self.rank}] kill PX4 instance by command line")

        patterns = [
            rf"px4.*-i {self.rank}",
            rf"px4.*-i\s+{self.rank}",
            rf"bin/px4.*-i {self.rank}",
            rf"px4_sitl.*-i {self.rank}",
        ]

        for pat in patterns:
            subprocess.run(
                ["pkill", "-TERM", "-f", pat],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        time.sleep(1.0)

        for pat in patterns:
            subprocess.run(
                ["pkill", "-KILL", "-f", pat],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        time.sleep(0.5)

        self.proc = None

        check = subprocess.run(
            ["pgrep", "-af", rf"px4.*-i {self.rank}"],
            check=False,
            capture_output=True,
            text=True,
        )

        leftover = (check.stdout or "").strip()

        if leftover:
            logger.warning(f"[PX4 MANAGER {self.rank}] WARNING PX4 still alive:\n{leftover}")
        else:
            logger.debug(f"[PX4 MANAGER {self.rank}] PX4 instance killed cleanly")

    def close(self) -> None:
        """Close PX4 and the auxiliary ros_gz bridge processes for this env."""
        try:
            self.kill()
        finally:
            for label, proc in reversed(self.bridge_processes):
                stop_bridge_process(proc)
            self.bridge_processes = []

    def _gz_env(self) -> dict:
        """Returns os.environ copy with GZ_PARTITION set to this instance's partition."""
        env = os.environ.copy()
        env["GZ_PARTITION"] = self.partition
        return env

    def remove_model(self) -> None:
        """
        Remove the correct model of this instance in the correct GZ_PARTITION.

        Best-effort only. Remove timeout is not fatal because Gazebo topic/entity
        may be stale.
        """
        if self.gz_client is not None and self.gz_client.available():
            try:
                ok = bool(
                    self.gz_client.remove_model(
                        self.world,
                        self.model_name,
                        timeout_ms=2000,
                    )
                )
                if ok:
                    logger.info(
                        f"[GZ TRANSPORT] remove_model ok world={self.world} model={self.model_name}"
                    )
                    time.sleep(1.0)
                    return
                logger.warning(
                    f"[GZ TRANSPORT] remove_model failed; fallback CLI world={self.world} model={self.model_name}"
                )
            except Exception as exc:
                logger.warning(f"[GZ TRANSPORT] remove_model failed; fallback CLI: {exc}")

        cmd = [
            "gz",
            "service",
            "-s",
            f"/world/{self.world}/remove",
            "--reqtype",
            "gz.msgs.Entity",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            f'name: "{self.model_name}" type: MODEL',
        ]

        logger.debug(f"[PX4 MANAGER {self.rank}] remove Gazebo model {self.model_name}")

        result = subprocess.run(
            cmd,
            env=self._gz_env(),
            check=False,
            capture_output=True,
            text=True,
        )

        if result.stdout.strip():
            logger.debug(f"[PX4 MANAGER {self.rank}] remove stdout: {result.stdout.strip()}")
        if result.stderr.strip():
            logger.debug(f"[PX4 MANAGER {self.rank}] remove stderr: {result.stderr.strip()}")

        time.sleep(1.0)

    def probe_model_set_pose(self, timeout: int = 3000) -> bool:
        """
        Authoritative readiness check via Gazebo set_pose service.

        Only trusts the Gazebo set_pose service. Does not use topic list to
        confirm model spawned because topics may be stale.
        """
        if self.gz_client is not None and self.gz_client.available():
            try:
                ok = bool(
                    self.gz_client.set_pose(
                        self.world,
                        self.model_name,
                        -8,
                        0,
                        0,
                        yaw=0.0,
                        timeout_ms=timeout,
                    )
                )
                if ok:
                    logger.info(
                        f"[GZ TRANSPORT] set_pose ok world={self.world} model={self.model_name}"
                    )
                    return True
                logger.warning(
                    f"[GZ TRANSPORT] set_pose failed; fallback CLI world={self.world} model={self.model_name}"
                )
            except Exception as exc:
                logger.warning(f"[GZ TRANSPORT] set_pose failed; fallback CLI: {exc}")

        req = (
            f'name: "{self.model_name}" '
            'position { x: -8 y: 0 z: 0 } '
            'orientation { x: 0 y: 0 z: 0 w: 1 }'
        )

        cmd = [
            "gz",
            "service",
            "-s",
            f"/world/{self.world}/set_pose",
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            str(timeout),
            "--req",
            req,
        ]

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=(timeout / 1000.0) + 2.0,
                env=self._gz_env(),
            )
        except Exception as exc:
            logger.error(f"[PX4 MANAGER {self.rank}] probe set_pose exception: {exc}")
            return False

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if stdout:
            logger.debug(f"[PX4 MANAGER {self.rank}] probe set_pose stdout: {stdout}")

        if stderr:
            logger.debug(f"[PX4 MANAGER {self.rank}] probe set_pose stderr: {stderr}")

        ok = result.returncode == 0 and "data: true" in stdout.lower()

        logger.debug(
            f"[PX4 MANAGER {self.rank}] probe model set_pose "
            f"model={self.model_name} "
            f"ok={ok}"
        )

        return ok

    def hard_reset(self, profile: bool = False):
        """
        Hard reset this instance.

        PX4-only reconnect flow (HARD_RESET_PX4_ONLY=1):
            kill PX4 -> keep model -> start PX4 with PX4_GZ_RUN=0 -> probe set_pose

        Full respawn flow (HARD_RESET_PX4_ONLY=0 or unset):
            kill PX4 -> remove model -> start PX4 with PX4_GZ_RUN=1 -> probe set_pose

        Source of truth is the set_pose probe, not gz topic list (stale).

        Args:
            profile: If True, collect per-phase timing data and return
                     (ok, profile_data) instead of just ok.
                     Default False — normal training behavior unchanged.
        """
        logger.info(f"[PX4 MANAGER {self.rank}] HARD RESET start")
        hard_reset_t0 = time.time()
        hard_reset_t0_mono = time.monotonic()
        px4_pid = getattr(self.proc, "pid", None)
        process_group = None
        if px4_pid is not None:
            try:
                process_group = os.getpgid(px4_pid)
            except Exception:
                process_group = None
        ports = {
            "PX4_UXRCE_DDS_PORT": self.env_vars.get("PX4_UXRCE_DDS_PORT"),
            "UXRCE_DDS_PORT": self.env_vars.get("UXRCE_DDS_PORT"),
        }
        profile_phases = [] if profile else None
        reset_mode = "px4_only_reconnect" if self.hard_reset_px4_only else "full_remove_and_respawn"
        remove_model_used = not self.hard_reset_px4_only
        gz_run_for_restart = 0 if self.hard_reset_px4_only else 1
        logger.info(
            "[HARD RESET ISOLATION] "
            f"mono={time.monotonic():.3f} "
            f"env_id={self.rank} "
            f"model_name={self.model_name} "
            f"world_name={self.world_name} "
            f"ros_domain_id={self.ros_domain} "
            f"px4_instance={self.rank} "
            f"px4_pid={px4_pid} "
            f"process_group={process_group} "
            f"mavlink_ports={ports} "
            f"reset_reason=manager_hard_reset "
            f"command_or_method=PX4InstanceManager.hard_reset "
            f"starts_at_wall_time={hard_reset_t0:.3f} "
            f"restarted_px4_only={self.hard_reset_px4_only} "
            f"restarted_gazebo={not self.hard_reset_px4_only} "
            "killed_process_group=False "
            "touched_other_env_model=False "
            "touched_other_env_ports=False"
        )

        logger.info(
            "[HARD RESET MODE] "
            f"env_id={self.rank} "
            f"mode={reset_mode} "
            f"remove_model_used={remove_model_used} "
            f"gz_run={gz_run_for_restart}"
        )

        # Outer loop: 3 restart attempts (range(3) → tries 0,1,2 → displayed as 1,2,3)
        for _restart_idx in range(3):
            restart_try = _restart_idx + 1
            logger.info(f"[PX4 MANAGER {self.rank}] restart try {restart_try}/3")

            phase_data = {"restart_try": restart_try} if profile else None

            # Kill PX4 first to release model lock
            t_phase = time.time() if profile else None
            self.kill()
            time.sleep(1.0)
            if profile:
                phase_data["kill_duration"] = time.time() - t_phase

            if not self.hard_reset_px4_only:
                # Remove model (best-effort). Gazebo service returns data:true if accepted.
                # Do NOT use wait_model_gone() because _gz_entity_exists() relies on
                # `gz topic -l` — topics still exist after model is deleted (stale).
                t_phase = time.time() if profile else None
                self.remove_model()
                time.sleep(2.0)
                if profile:
                    phase_data["remove_duration"] = time.time() - t_phase

            t_phase = time.time() if profile else None
            self.start(gz_run=not self.hard_reset_px4_only)
            if profile:
                phase_data["start_duration"] = time.time() - t_phase

            # Source of truth: set_pose probe confirms new model is controllable
            controllable = False
            probe_durations = [] if profile else None
            t_probe_total = time.time() if profile else None

            # Inner probe loop: up to 4 retries (range(4) → indices 0,1,2,3)
            # First restart attempt uses only 2 probes; subsequent attempts use all 4.
            max_probe_attempts = 2 if restart_try == 1 else 4

            for _probe_idx in range(4):
                if _probe_idx >= max_probe_attempts:
                    break
                probe_try = _probe_idx + 1
                time.sleep(1.0)

                t_probe = time.time() if profile else None
                controllable = self.probe_model_set_pose(timeout=3000)
                if profile:
                    probe_durations.append(time.time() - t_probe)

                logger.debug(
                    f"[PX4 MANAGER {self.rank}] set_pose probe "
                    f"{probe_try}/{max_probe_attempts} controllable={controllable}"
                )

                if controllable:
                    duration = time.time() - hard_reset_t0
                    logger.info(
                        f"[PX4 MANAGER {self.rank}] HARD RESET done "
                        f"duration={duration:.1f}s"
                    )
                    logger.info(
                        "[HARD RESET ISOLATION] "
                        f"mono={time.monotonic():.3f} "
                        f"env_id={self.rank} "
                        f"model_name={self.model_name} "
                        f"world_name={self.world_name} "
                        f"ros_domain_id={self.ros_domain} "
                        f"px4_instance={self.rank} "
                        f"px4_pid={getattr(self.proc, 'pid', None)} "
                        f"process_group={process_group} "
                        f"mavlink_ports={ports} "
                        f"reset_reason=manager_hard_reset "
                        f"command_or_method=PX4InstanceManager.hard_reset "
                        f"starts_at_wall_time={hard_reset_t0:.3f} "
                        f"ends_at_wall_time={time.time():.3f} "
                        f"duration={time.monotonic() - hard_reset_t0_mono:.3f} "
                        f"restarted_px4_only={self.hard_reset_px4_only} "
                        f"restarted_gazebo={not self.hard_reset_px4_only} "
                        "killed_process_group=False "
                        "touched_other_env_model=False "
                        "touched_other_env_ports=False"
                    )

                    if profile:
                        phase_data["probe_attempts"] = probe_try
                        phase_data["probe_durations"] = probe_durations
                        phase_data["probe_total"] = time.time() - t_probe_total
                        phase_data["controllable"] = True
                        profile_phases.append(phase_data)

                        profile_data = {
                            "success": True,
                            "total_duration": duration,
                            "restart_try_used": restart_try,
                            "phases": profile_phases,
                        }
                        return True, profile_data

                    return True

            if profile:
                phase_data["probe_attempts"] = max_probe_attempts
                phase_data["probe_durations"] = probe_durations
                phase_data["probe_total"] = time.time() - t_probe_total
                phase_data["controllable"] = False
                profile_phases.append(phase_data)

            logger.info(
                f"[PX4 MANAGER {self.rank}] model not controllable after restart. "
                "Retry restart."
            )

        duration = time.time() - hard_reset_t0
        logger.error(
            f"[PX4 MANAGER {self.rank}] HARD RESET failed: "
            f"model is still not controllable by set_pose. "
            f"duration={duration:.1f}s"
        )
        logger.info(
            "[HARD RESET ISOLATION] "
            f"mono={time.monotonic():.3f} "
            f"env_id={self.rank} "
            f"model_name={self.model_name} "
            f"world_name={self.world_name} "
            f"ros_domain_id={self.ros_domain} "
            f"px4_instance={self.rank} "
            f"px4_pid={getattr(self.proc, 'pid', None)} "
            f"process_group={process_group} "
            f"mavlink_ports={ports} "
            f"reset_reason=manager_hard_reset "
            f"command_or_method=PX4InstanceManager.hard_reset "
            f"starts_at_wall_time={hard_reset_t0:.3f} "
            f"ends_at_wall_time={time.time():.3f} "
            f"duration={time.monotonic() - hard_reset_t0_mono:.3f} "
            f"restarted_px4_only={self.hard_reset_px4_only} "
            f"restarted_gazebo={not self.hard_reset_px4_only} "
            "killed_process_group=False "
            "touched_other_env_model=False "
            "touched_other_env_ports=False"
        )

        if profile:
            profile_data = {
                "success": False,
                "total_duration": duration,
                "restart_try_used": 3,
                "phases": profile_phases,
            }
            return False, profile_data

        return False
