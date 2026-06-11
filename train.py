import dataclasses
import logging
import multiprocessing as mp
import os
import time
from datetime import datetime
from typing import Callable

import cv2
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from obstacle_avoidance.configs.env_config import EnvConfig
from obstacle_avoidance.configs.pillar_config import PillarConfig
from obstacle_avoidance.configs.reward_config import RewardConfig
from obstacle_avoidance.envs.drone_env import DroneObstacleEnv
from obstacle_avoidance.envs.policy import DepthStateExtractor
from obstacle_avoidance.envs.monitor import TrainingMonitor
from obstacle_avoidance.utils.bridge_factory import make_bridge, make_spawner
from obstacle_avoidance.utils.logger import setup_logger
from obstacle_avoidance.utils.checkpoint_utils import (
    find_latest_checkpoint,
    load_stage_start_step,
    save_stage_start_step,
)
from obstacle_avoidance.utils.process_utils import (
    start_gz_clock_bridge,
    start_gz_depth_bridge,
    start_gz_lidar_bridge,
    start_gz_pose_bridge,
    start_microxrce_agent,
)
from obstacle_avoidance.utils.px4_manager import PX4InstanceManager

logger = logging.getLogger("obstacle_avoidance")


def make_env(
    rank: int,
    num_pillars: int,
    curriculum_stage: int,
    run_id: str,
    total_envs: int,
    env_log_dir: str | None = None,
    stage_conf: dict | None = None,
) -> Callable:
    def _init():
        # 1. Thread limits — old L992-998
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        cv2.setNumThreads(0)

        ros_domain = 30 + rank
        uxrce_port = 8888 + rank
        partition = f"drone_rl_{rank}"

        # Stagger startup: avoid 2 Gazebo processes starting simultaneously.
        # Each Gazebo instance needs ~15-20s to fully start (load world, register
        # services). SubprocVecEnv starts all envs in parallel so we stagger
        # manually. — old L1006-1016
        STARTUP_STAGGER_S = 5  # seconds between each rank (Gazebo needs ~15-20s)
        if rank > 0 and total_envs > 1:
            wait_s = rank * STARTUP_STAGGER_S
            logger.info(
                f"[ENV {rank}/{total_envs}] [STAGGER] Waiting {wait_s}s "
                f"for previous rank to finish starting Gazebo..."
            )
            time.sleep(wait_s)

        headless = "1"

        # 2. Per-rank env var setup (os.environ direct) — old L1020-1022
        os.environ["ROS_DOMAIN_ID"] = str(ros_domain)
        os.environ["PX4_INSTANCE"] = str(rank)
        os.environ["GZ_PARTITION"] = partition

        px4_path = os.path.expanduser("~/antruong_drone_rl/PX4-Autopilot")
        px4_bin = os.path.join(px4_path, "build/px4_sitl_default/bin/px4")
        rootfs_base_dir = os.path.join(px4_path, "build/px4_sitl_default/rootfs")
        rootfs_dir = os.path.join(rootfs_base_dir, str(rank))
        os.makedirs(rootfs_dir, exist_ok=True)

        if not os.path.exists(px4_bin):
            raise RuntimeError(
                f"PX4 binary not found: {px4_bin}\n"
                "Run this first:\n"
                "  cd ~/PX4-Autopilot && make px4_sitl"
            )

        model_path = os.path.join(px4_path, "Tools/simulation/gz/models")
        world_path = os.path.join(px4_path, "Tools/simulation/gz/worlds")

        # Per-rank world isolation — old L1040-1060
        # IMPORTANT: must change <world name="default"> -> <world name="default_N">
        # inside the SDF because Gazebo creates services by the world name declared
        # in the file, not by the filename. Without this patch
        # /world/default_N/set_pose won't exist -> teleport timeout.
        world_name = f"default_{rank}"
        src_world = os.path.join(world_path, "default.sdf")
        dst_world = os.path.join(world_path, f"{world_name}.sdf")
        with open(src_world, "r", encoding="utf-8") as _f:
            _sdf_content = _f.read()
        _sdf_content = _sdf_content.replace(
            '<world name="default">', f'<world name="{world_name}">'
        )
        with open(dst_world, "w", encoding="utf-8") as _f:
            _f.write(_sdf_content)
        logger.info(
            f"[ENV {rank}] [WORLD ISOLATION] Written per-rank world: {dst_world} "
            f"(world name='{world_name}')"
        )

        # 3. Per-rank env var dict for subprocess — old L1062-1086
        env_vars = os.environ.copy()
        env_vars["ROS_DOMAIN_ID"] = str(ros_domain)
        env_vars["PX4_INSTANCE"] = str(rank)
        env_vars["GZ_PARTITION"] = partition
        env_vars["GZ_SIM_RESOURCE_PATH"] = f"{model_path}:{world_path}"

        env_vars["HEADLESS"] = headless
        env_vars["PX4_GZ_RUN"] = "1"

        env_vars["PX4_SIM_MODEL"] = "gz_x500_depth"
        env_vars["PX4_GZ_WORLD"] = world_name

        env_vars["PX4_UXRCE_DDS_PORT"] = str(uxrce_port)
        env_vars["UXRCE_DDS_PORT"] = str(uxrce_port)

        env_vars["PX4_GZ_MODEL_POSE"] = "0,0,0,0,0,0"

        logger.info(
            f"[ENV {rank}/{total_envs}] run_id={run_id}, stage={curriculum_stage} | "
            f"ROS_DOMAIN_ID={ros_domain}, "
            f"PX4_INSTANCE={rank}, "
            f"GZ_PARTITION={partition}, "
            f"UXRCE_PORT={uxrce_port}, "
            f"HEADLESS={headless}"
        )

        # 4. Launch MicroXRCE agent — old L1088
        xrce_proc = start_microxrce_agent(rank, ros_domain)

        model_name = f"x500_depth_{rank}"
        bridge_processes = []

        # 5. Launch clock + depth bridges (+ pose bridge) — old L1090-1130
        # Bridge Gazebo clock -> ROS /clock for sim-time nodes.
        bridge_processes.append(
            (
                "clock",
                start_gz_clock_bridge(
                    gz_partition=partition,
                    ros_domain_id=ros_domain,
                ),
            )
        )

        # Bridge Gazebo model pose -> ROS Pose topic.
        # If ros_gz bridge/type errors, bridge_factory still falls back to parser.
        bridge_processes.append(
            (
                "pose",
                start_gz_pose_bridge(
                    model_name=model_name,
                    gz_partition=partition,
                    ros_domain_id=ros_domain,
                ),
            )
        )

        # Bridge Gazebo depth camera -> ROS Image topic.
        # Without this bridge the agent cannot see obstacles.
        bridge_processes.append(
            (
                "depth",
                start_gz_depth_bridge(
                    model_name=model_name,
                    gz_partition=partition,
                    ros_domain_id=ros_domain,
                ),
            )
        )

        bridge_processes.append(
            (
                "lidar",
                start_gz_lidar_bridge(
                    model_name=model_name,
                    gz_partition=partition,
                    ros_domain_id=ros_domain,
                ),
            )
        )

        # 6. Create PX4InstanceManager, call .start() — old L1132-1146
        px4_manager = PX4InstanceManager(
            rank=rank,
            ros_domain=ros_domain,
            partition=partition,
            px4_bin=px4_bin,
            rootfs_dir=rootfs_dir,
            env_vars=env_vars,
            model_name=model_name,
            world=world_name,          # per-rank world
            start_pose="0,0,0,0,0,0",
            startup_sleep=18.0,  # enough time for Gazebo to load world + spawn model
            bridge_processes=bridge_processes,
        )

        px4_manager.start(gz_run=True)

        # 7. Build bridge via make_bridge(...), spawner via make_spawner(...) — old L1148-1162
        px4_ns = "" if rank == 0 else f"/px4_{rank}"

        bridge = make_bridge(
            gazebo_port=1134 + rank,
            world=world_name,          # per-rank world
            model_name=model_name,
            px4_ns=px4_ns,
            target_system=rank + 1,
            gz_partition=partition,
            xrce_proc=xrce_proc,
        )

        spawner = make_spawner(
            world=world_name,          # per-rank world
            gz_partition=partition,
        )

        # 8. Construct DroneObstacleEnv — old L1176-1197
        ecfg = EnvConfig()
        _ecfg_fields = {f.name for f in dataclasses.fields(EnvConfig)}
        for key, val in (stage_conf or {}).items():
            if key in _ecfg_fields:
                setattr(ecfg, key, val)
        rcfg = RewardConfig()
        pcfg = PillarConfig(num_pillars=num_pillars)

        # Log dir: env_logs/env_{rank}/stage{N}_{ts}/
        if env_log_dir:
            # env_log_dir = env_logs/stage{N}_{ts} — remap to env_logs/env_{rank}/stage{N}_{ts}
            base = os.path.dirname(env_log_dir)          # env_logs/
            ts_folder = os.path.basename(env_log_dir)    # stage{N}_{ts}
            rank_log_dir = os.path.join(base, f"env_{rank}", ts_folder)
            os.makedirs(rank_log_dir, exist_ok=True)
        else:
            rank_log_dir = None
        env = DroneObstacleEnv(bridge, spawner, ecfg, rcfg, pcfg, env_id=rank, log_dir=rank_log_dir)

        # 9. Wrap with Monitor — old L1199
        env = Monitor(env)
        return env

    return _init


def run_training(stage: int = 1) -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Load ppo_config.yaml
    config_path = os.path.join(project_root, "configs", "ppo_config.yaml")
    with open(config_path) as f:
        ppo_cfg = yaml.safe_load(f)

    # Find curriculum entry for this stage (1-based)
    conf = None
    for entry in ppo_cfg["curriculum"]:
        if entry["stage"] == stage:
            conf = entry
            break
    if conf is None:
        raise ValueError(
            f"[run_training] No curriculum entry found for stage={stage} in {config_path}"
        )

    n_envs = ppo_cfg["n_envs"]
    check_freq = ppo_cfg.get("checkpoint_freq", 10000)
    model_prefix = f"ppo_drone_stage{stage}"

    ckpt_dir = os.path.join(project_root, "ckpts", f"stage{stage}")
    log_dir = os.path.join(project_root, "runs")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_log_base = os.path.join(log_dir, "env_logs")  # env_logs/env_{rank}/stage{N}_{ts}/
    env_log_dir = os.path.join(env_log_base, f"stage{stage}_{run_ts}")
    setup_logger(
        "obstacle_avoidance",
        info_log_file=os.path.join(log_dir, f"train_stage{stage}_{run_ts}.log"),
    )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"STAGE {stage}: num_pillars={conf['num_pillars']} min_steps={conf['min_steps']}")
    logger.info(f"{'=' * 60}\n")

    # --- Path definitions ---
    interrupted_model_path = os.path.join(project_root, f"{model_prefix}_interrupted.zip")
    final_model_path = os.path.join(project_root, f"{model_prefix}_final.zip")
    prev_final_model_path = os.path.join(
        project_root,
        f"ppo_drone_stage{stage - 1}_final.zip",
    )
    interrupted_save_path = os.path.join(project_root, f"{model_prefix}_interrupted")

    latest_ckpt = find_latest_checkpoint(ckpt_dir, f"stage{stage}")

    # --- Resume priority chain ---
    # 1. interrupted
    # 2. latest_ckpt
    # 3. final
    # 4. prev_final (stage-transfer)
    # 5. fresh
    resume_source = None
    if os.path.exists(interrupted_model_path):
        resume_source = interrupted_model_path
    elif latest_ckpt is not None:
        resume_source = latest_ckpt
    elif os.path.exists(final_model_path):
        resume_source = final_model_path
    elif stage > 1 and os.path.exists(prev_final_model_path):
        resume_source = prev_final_model_path

    # --- Probe resume timesteps ---
    resume_timesteps = 0
    if resume_source is not None:
        try:
            probe = PPO.load(
                resume_source,
                n_steps=ppo_cfg["n_steps"],
                batch_size=ppo_cfg["batch_size"],
            )
            resume_timesteps = int(getattr(probe, "num_timesteps", 0))
            del probe
        except Exception as e:
            logger.warning(
                f"[RESUME] failed to read num_timesteps from {resume_source}: {e}"
            )

    # --- Stage start step (for monitor / curriculum accounting) ---
    stage_start_step = load_stage_start_step(ckpt_dir, stage)
    if stage > 1 and stage_start_step == 0:
        # No metadata yet: try to infer from previous stage's final model
        if os.path.exists(prev_final_model_path):
            try:
                prev_probe = PPO.load(prev_final_model_path)
                step = int(getattr(prev_probe, "num_timesteps", 0))
                del prev_probe
                save_stage_start_step(
                    ckpt_dir, stage, step, source_path=prev_final_model_path
                )
                stage_start_step = step
                logger.info(
                    f"[STAGE START STEP] inferred from prev_final={prev_final_model_path} "
                    f"stage_start_step={stage_start_step}"
                )
            except Exception as e:
                logger.warning(
                    f"[STAGE START STEP] failed to infer from {prev_final_model_path}: {e}. "
                    f"fallback=0"
                )
        else:
            logger.warning(
                f"[STAGE START STEP] metadata missing for stage={stage} and no prev_final found. "
                f"fallback=0"
            )

    logger.info(
        "[CURRICULUM RESUME] "
        f"resume_source={resume_source} "
        f"resume_timesteps={resume_timesteps} "
        f"n_envs={n_envs} "
        f"stage_start_step={stage_start_step}"
    )

    # --- Build vectorised env ---
    env = SubprocVecEnv([
        make_env(i, conf["num_pillars"], stage, f"stage{stage}", n_envs, env_log_dir, stage_conf=conf)
        for i in range(n_envs)
    ])

    # --- Policy kwargs ---
    policy_kwargs = dict(
        features_extractor_class=DepthStateExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    # --- Load or create model (matches resume priority chain order) ---
    def _log_model_load(source: str, label: str) -> None:
        logger.info("=" * 60)
        logger.info(f"[MODEL LOAD] {label}")
        logger.info(f"   {source}")
        logger.info("=" * 60)

    if os.path.exists(interrupted_model_path):
        _log_model_load(interrupted_model_path, f"Resume stage {stage} from INTERRUPTED model")
        model = PPO.load(
            interrupted_model_path,
            env=env,
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            device="cuda",
        )
        logger.info(f"[MODEL LOAD] loaded_num_timesteps={int(getattr(model, 'num_timesteps', -1))}")

    elif latest_ckpt is not None:
        _log_model_load(latest_ckpt, f"Resume stage {stage} from latest checkpoint")
        model = PPO.load(
            latest_ckpt,
            env=env,
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            device="cuda",
        )
        logger.info(f"[MODEL LOAD] loaded_num_timesteps={int(getattr(model, 'num_timesteps', -1))}")

    elif os.path.exists(final_model_path):
        _log_model_load(final_model_path, f"Resume stage {stage} from final model")
        model = PPO.load(
            final_model_path,
            env=env,
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            device="cuda",
        )
        logger.info(f"[MODEL LOAD] loaded_num_timesteps={int(getattr(model, 'num_timesteps', -1))}")

    elif stage > 1 and os.path.exists(prev_final_model_path):
        _log_model_load(prev_final_model_path, f"Stage-transfer: stage {stage - 1} -> {stage} (lr=1e-4)")
        model = PPO.load(
            prev_final_model_path,
            env=env,
            learning_rate=1e-4,
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            device="cuda",
        )
        logger.info(f"[MODEL LOAD] loaded_num_timesteps={int(getattr(model, 'num_timesteps', -1))}")

    else:
        logger.info("=" * 60)
        logger.info(f"[MODEL LOAD] Fresh PPO model — stage {stage}")
        logger.info("=" * 60)
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda",
        )

    # --- CNN freeze (stage 0/1: no pillars, train state MLP only) ---
    freeze_cnn = bool(conf.get("freeze_cnn", False))
    extractor = model.policy.features_extractor
    for name, param in extractor.named_parameters():
        if name.startswith("cnn"):  # covers cnn.* and cnn_fc.*
            param.requires_grad = not freeze_cnn
    logger.info(f"[CNN] freeze_cnn={freeze_cnn} — "
                + ("cnn+cnn_fc frozen" if freeze_cnn else "all params trainable"))

    # --- Callbacks ---
    ckpt_callback = CheckpointCallback(
        save_freq=max(check_freq // n_envs, 1),
        save_path=ckpt_dir,
        name_prefix=f"stage{stage}",
    )
    monitor_callback = TrainingMonitor(
        check_freq=check_freq,
        csv_freq=check_freq,
        stage=stage,
        stage_steps=int(conf["min_steps"]),
        n_envs=n_envs,
        num_pillars=int(conf["num_pillars"]),
    )

    # --- Crash handler ---
    def save_interrupted_model(reason: str) -> None:
        try:
            model.save(interrupted_save_path)
            logger.info(
                f"[TRAIN SAVE] interrupted reason={reason} path={interrupted_save_path}.zip"
            )
        except BaseException as save_exc:
            logger.error(
                f"[TRAIN SAVE FAILED] interrupted reason={reason} "
                f"path={interrupted_save_path}.zip "
                f"error={type(save_exc).__name__}: {save_exc}"
            )

    logger.info(
        f"[TRAIN START] stage={stage} n_envs={n_envs} "
        f"n_steps={model.n_steps} batch_size={model.batch_size} "
        f"min_steps={conf['min_steps']} ckpt_dir={ckpt_dir}"
    )

    try:
        model.tensorboard_log = log_dir
        
        model.learn(
            total_timesteps=conf["min_steps"],
            callback=[ckpt_callback, monitor_callback],
            reset_num_timesteps=False,
        )
        model.save(os.path.join(project_root, f"{model_prefix}_final"))
        logger.info(
            f"[TRAIN DONE] saved final model: "
            f"{os.path.join(project_root, model_prefix + '_final')}.zip"
        )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            save_interrupted_model("keyboard_interrupt")
        elif isinstance(exc, SystemExit):
            save_interrupted_model("system_exit")
            raise
        else:
            save_interrupted_model(type(exc).__name__.lower())
            raise
    finally:
        try:
            env.close()
        except BaseException as close_exc:
            logger.warning(f"[ENV CLOSE] {type(close_exc).__name__}: {close_exc}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, help="Curriculum stage (1-based)")
    args = parser.parse_args()
    run_training(stage=args.stage)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
