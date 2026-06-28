import dataclasses
import logging
import multiprocessing as mp
import os
import time
from datetime import datetime
from typing import Callable

import cv2
import torch
import torch.nn as nn
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

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
    start_microxrce_agent_single,
    stop_bridge_process,
)
from obstacle_avoidance.utils.px4_manager import PX4InstanceManager

logger = logging.getLogger("obstacle_avoidance")


def _get_env_cores(rank: int, machine_cfg: dict) -> list:
    main_cores = str(machine_cfg.get("main_process_cores", "0-3"))
    cores_per_env = int(machine_cfg.get("cores_per_env", 6))
    if "-" in main_cores:
        _, hi = main_cores.split("-")
        env_start = int(hi) + 1
    else:
        env_start = int(main_cores) + 1
    start = env_start + rank * cores_per_env
    return list(range(start, start + cores_per_env))



def make_env(
    rank: int,
    num_pillars: int,
    curriculum_stage: int,
    run_id: str,
    total_envs: int,
    env_log_dir: str | None = None,
    stage_conf: dict | None = None,
    machine_cfg: dict | None = None,
    start_step: int = 0,
) -> Callable:
    def _init():
        #1. Thread limits — old L992-998
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        cv2.setNumThreads(0)

        ros_domain = 30 + rank
        partition = f"drone_rl_{rank}"

        # Stagger startup: each rank gets its own Gazebo instance.
        # Sequential stagger prevents simultaneous Gazebo launches (~15-20s each).
        STARTUP_STAGGER_S = 5
        if rank > 0 and total_envs > 1:
            wait_s = rank * STARTUP_STAGGER_S
            logger.info(
                f"[ENV {rank}/{total_envs}] [STAGGER] Waiting {wait_s}s "
                f"for previous rank to finish starting Gazebo..."
            )
            time.sleep(wait_s)

        headless = "1"

        # 2. Per-rank env var setup (os.environ direct)
        os.environ["ROS_DOMAIN_ID"] = str(ros_domain)
        os.environ["PX4_INSTANCE"] = str(rank)
        os.environ["GZ_PARTITION"] = partition

        px4_path = os.path.expanduser("~/PX4-Autopilot")
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

        # Per-rank world isolation — patch <world name="default"> → <world name="default_N">
        # inside SDF so Gazebo service names (e.g. /world/default_N/set_pose) are unique.
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

        # 3. Per-rank env var dict for subprocess
        env_vars = os.environ.copy()
        env_vars["ROS_DOMAIN_ID"] = str(ros_domain)
        env_vars["PX4_INSTANCE"] = str(rank)
        env_vars["GZ_PARTITION"] = partition
        env_vars["GZ_SIM_RESOURCE_PATH"] = f"{model_path}:{world_path}"

        env_vars["HEADLESS"] = headless
        env_vars["PX4_GZ_RUN"] = "1"

        env_vars["PX4_SIM_MODEL"] = "gz_x500_depth"
        env_vars["PX4_GZ_WORLD"] = world_name

        # PX4_UXRCE_DDS_PORT NOT set — all instances connect to port 8888 (shared agent).
        # PX4 rcS distinguishes clients via UXRCE_DDS_KEY = px4_instance+1 (from -i N).

        env_vars["PX4_GZ_MODEL_POSE"] = "0,0,0,0,0,0"

        logger.info(
            f"[ENV {rank}/{total_envs}] run_id={run_id}, stage={curriculum_stage} | "
            f"ROS_DOMAIN_ID={ros_domain}, "
            f"PX4_INSTANCE={rank}, "
            f"GZ_PARTITION={partition}, "
            f"UXRCE_PORT=8888(shared), "
            f"HEADLESS={headless}"
        )

        model_name = f"x500_depth_{rank}"
        bridge_processes = []

        # 5. Launch clock + depth bridges (+ pose bridge)
        bridge_processes.append(
            (
                "clock",
                start_gz_clock_bridge(
                    gz_partition=partition,
                    ros_domain_id=ros_domain,
                ),
            )
        )

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

        # 6. Create PX4InstanceManager, call .start()
        px4_manager = PX4InstanceManager(
            rank=rank,
            ros_domain=ros_domain,
            partition=partition,
            px4_bin=px4_bin,
            rootfs_dir=rootfs_dir,
            env_vars=env_vars,
            model_name=model_name,
            world=world_name,
            start_pose="0,0,0,0,0,0",
            startup_sleep=18.0,
            bridge_processes=bridge_processes,
        )

        px4_manager.start(gz_run=True)

        # --- Core pinning (Tier 2): pin PX4 + bridges to dedicated cores ---
        # NOTE: do NOT pin the Python worker process itself — it contains timing-sensitive
        # keepalive (5 Hz) and ROS spin threads that get starved when sharing cores with PX4.
        if machine_cfg and machine_cfg.get("pin_processes", False):
            cores = _get_env_cores(rank, machine_cfg)
            cores_set = set(cores)
            if px4_manager.proc is not None:
                try:
                    os.sched_setaffinity(px4_manager.proc.pid, cores_set)
                except OSError as e:
                    logger.warning(f"[ENV {rank}] sched_setaffinity px4: {e}")
            for bp_name, bp_proc in bridge_processes:
                if bp_proc is not None:
                    try:
                        os.sched_setaffinity(bp_proc.pid, cores_set)
                    except OSError as e:
                        logger.warning(f"[ENV {rank}] sched_setaffinity {bp_name}: {e}")
            logger.info(f"[ENV {rank}] [AFFINITY] pinned PX4+bridges → cores {min(cores)}-{max(cores)}")

        # 7. Build bridge via make_bridge(...), spawner via make_spawner(...) — old L1148-1162
        px4_ns = "" if rank == 0 else f"/px4_{rank}"

        # 7a. EnvConfig — needed by bridge for use_symbolic_extractor flag
        ecfg = EnvConfig()
        _ecfg_fields = {f.name for f in dataclasses.fields(EnvConfig)}
        for key, val in (stage_conf or {}).items():
            if key in _ecfg_fields:
                setattr(ecfg, key, val)

        logger.info(f"[RANK {rank}] use_symbolic_extractor={ecfg.use_symbolic_extractor}")

        bridge = make_bridge(
            gazebo_port=1134 + rank,
            world=world_name,
            model_name=model_name,
            px4_ns=px4_ns,
            target_system=rank + 1,
            gz_partition=partition,
            xrce_proc=None,  # shared agent managed by run_training()
            env_config=ecfg,
        )

        spawner = make_spawner(
            world=world_name,
            gz_partition=partition,
        )

        # 8. Construct DroneObstacleEnv — old L1176-1197
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
        env = DroneObstacleEnv(bridge, spawner, ecfg, rcfg, pcfg, env_id=rank, log_dir=rank_log_dir, start_step=start_step)

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
    vecnorm_path = os.path.join(project_root, f"{model_prefix}_vecnormalize.pkl")
    vecnorm_interrupted_path = os.path.join(project_root, f"{model_prefix}_vecnormalize_interrupted.pkl")

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
    elif stage > 0 and os.path.exists(prev_final_model_path):
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
    if stage > 0 and stage_start_step == 0:
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

    # --- Machine config (per-machine CPU tuning) ---
    machine_config_path = os.path.join(project_root, "configs", "machine_config.yaml")
    machine_cfg: dict = {}
    if os.path.exists(machine_config_path):
        with open(machine_config_path) as _mcf:
            machine_cfg = yaml.safe_load(_mcf) or {}
        logger.info(f"[MACHINE CFG] {machine_cfg}")

    # --- Global seed ---
    _seed = int(ppo_cfg.get("seed", 42))
    import random as _random
    _random.seed(_seed)
    import numpy as _np
    _np.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    logger.info(f"[SEED] global seed={_seed}")

    # Tier 3: PyTorch thread limits for main PPO process
    if machine_cfg.get("torch_num_threads"):
        torch.set_num_threads(int(machine_cfg["torch_num_threads"]))
    if machine_cfg.get("torch_interop_threads"):
        try:
            torch.set_num_interop_threads(int(machine_cfg["torch_interop_threads"]))
        except RuntimeError:
            pass  # already set — benign
    if machine_cfg.get("pin_processes", False) and machine_cfg.get("main_process_cores"):
        main_cores = str(machine_cfg["main_process_cores"])
        try:
            if "-" in main_cores:
                lo, hi = map(int, main_cores.split("-"))
                os.sched_setaffinity(0, set(range(lo, hi + 1)))
            else:
                os.sched_setaffinity(0, {int(main_cores)})
            logger.info(f"[MAIN AFFINITY] pinned main process → cores {main_cores}")
        except OSError as e:
            logger.warning(f"[MAIN AFFINITY] sched_setaffinity failed: {e}")

    # --- Single shared MicroXRCEAgent for ALL PX4 instances ---
    # One agent on port 8888 handles N instances via unique UXRCE_DDS_KEY (px4_instance+1).
    # Each PX4 still has its own ROS_DOMAIN_ID + namespace — full topic isolation preserved.
    xrce_agent = start_microxrce_agent_single(port=8888)
    logger.info(f"[UXRCE] shared agent started port=8888 pid={xrce_agent.pid}")

    # --- Build vectorised env ---
    raw_env = SubprocVecEnv([
        make_env(i, conf["num_pillars"], stage, f"stage{stage}", n_envs, env_log_dir, stage_conf=conf, machine_cfg=machine_cfg, start_step=resume_timesteps)
        for i in range(n_envs)
    ])
    # VecNormalize load path mirrors model resume priority — ret_rms must match the loaded model.
    # CheckpointCallback(save_vecnormalize=True) saves to {ckpt_dir}/{prefix}_{steps}_steps_vecnormalize.pkl.
    _vn_load_path: str | None = None
    if os.path.exists(interrupted_model_path):
        if os.path.exists(vecnorm_interrupted_path):
            _vn_load_path = vecnorm_interrupted_path
    elif latest_ckpt is not None:
        _ckpt_base = os.path.basename(latest_ckpt)           # "stage0_44626_steps.zip"
        _ckpt_dir_p = os.path.dirname(latest_ckpt)
        _parts = _ckpt_base.split("_", 1)                    # ["stage0", "44626_steps.zip"]
        _vn_name = f"{_parts[0]}_vecnormalize_{_parts[1].replace('.zip', '.pkl')}"
        _ckpt_vn = os.path.join(_ckpt_dir_p, _vn_name)      # "stage0_vecnormalize_44626_steps.pkl"
        if os.path.exists(_ckpt_vn):
            _vn_load_path = _ckpt_vn
    elif os.path.exists(final_model_path):
        if os.path.exists(vecnorm_path):
            _vn_load_path = vecnorm_path
    elif stage > 0 and os.path.exists(prev_final_model_path):
        # Stage transfer: load prev stage ret_rms — base reward scale (time/altitude/progress)
        # is identical across stages; only pillar rewards are added. Starting from calibrated
        # ret_rms avoids cold-start scaling error for first few thousand steps.
        prev_vecnorm_path = os.path.join(project_root, f"ppo_drone_stage{stage - 1}_vecnormalize.pkl")
        if os.path.exists(prev_vecnorm_path):
            _vn_load_path = prev_vecnorm_path

    # --- Terminal banner: resume sources ---
    _ckpt_label = resume_source if resume_source else "FRESH (no checkpoint)"
    _vn_label   = _vn_load_path  if _vn_load_path  else "FRESH (no pkl)"
    print("\n" + "=" * 70)
    print(f"  [STAGE {stage}] CHECKPOINT  : {_ckpt_label}")
    print(f"  [STAGE {stage}] VECNORMALIZE: {_vn_label}")
    print("=" * 70 + "\n")

    if _vn_load_path is not None:
        env = VecNormalize.load(_vn_load_path, raw_env)
        logger.info(f"[VECNORM] loaded stats from {_vn_load_path}")
    else:
        env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=ppo_cfg.get("gamma", 0.99))
        logger.info("[VECNORM] fresh stats (no prior pkl found)")

    # --- Policy kwargs ---
    policy_kwargs = dict(
        features_extractor_class=DepthStateExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=nn.SiLU,
    )

    # --- Load or create model (matches resume priority chain order) ---
    def _log_model_load(source: str, label: str) -> None:
        logger.info("=" * 60)
        logger.info(f"[MODEL LOAD] {label}")
        logger.info(f"   {source}")
        logger.info("=" * 60)

    def _log_ppo_params(model: PPO) -> None:
        logger.info(
            f"[PPO PARAMS] lr={model.learning_rate} | n_steps={model.n_steps} | "
            f"batch_size={model.batch_size} | gamma={model.gamma} | "
            f"gae_lambda={model.gae_lambda} | clip_range={model.clip_range} | "
            f"ent_coef={model.ent_coef} | vf_coef={model.vf_coef} | "
            f"max_grad_norm={model.max_grad_norm}"
        )

    _ppo_override_kwargs = dict(
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg.get("n_epochs", 10),
        learning_rate=ppo_cfg["learning_rate"],
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.0),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        seed=_seed,
        device="cuda",
    )

    if os.path.exists(interrupted_model_path):
        _log_model_load(interrupted_model_path, f"Resume stage {stage} from INTERRUPTED model")
        model = PPO.load(interrupted_model_path, env=env, **_ppo_override_kwargs)
        logger.info(f"[MODEL LOAD] loaded_num_timesteps={int(getattr(model, 'num_timesteps', -1))}")
        _log_ppo_params(model)

    elif latest_ckpt is not None:
        _log_model_load(latest_ckpt, f"Resume stage {stage} from latest checkpoint")
        model = PPO.load(latest_ckpt, env=env, **_ppo_override_kwargs)
        logger.info(f"[MODEL LOAD] loaded_num_timesteps={int(getattr(model, 'num_timesteps', -1))}")
        _log_ppo_params(model)

    elif os.path.exists(final_model_path):
        _log_model_load(final_model_path, f"Resume stage {stage} from final model")
        model = PPO.load(final_model_path, env=env, **_ppo_override_kwargs)
        logger.info(f"[MODEL LOAD] loaded_num_timesteps={int(getattr(model, 'num_timesteps', -1))}")
        _log_ppo_params(model)

    elif stage > 0 and os.path.exists(prev_final_model_path):
        _log_model_load(prev_final_model_path, f"Stage-transfer: stage {stage - 1} -> {stage} (lr=1e-4)")
        model = PPO.load(prev_final_model_path, env=env, **_ppo_override_kwargs)
        logger.info(f"[MODEL LOAD] loaded_num_timesteps={int(getattr(model, 'num_timesteps', -1))}")
        _log_ppo_params(model)

    else:
        logger.info("=" * 60)
        logger.info(f"[MODEL LOAD] Fresh PPO model — stage {stage}")
        logger.info("=" * 60)
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            **_ppo_override_kwargs,
        )
        _log_ppo_params(model)

    # --- CNN freeze (stage 0/1: no pillars, train state MLP only) ---
    freeze_cnn = bool(conf.get("freeze_cnn", False))
    extractor = model.policy.features_extractor
    _state_fc0 = extractor.state_fc[0]
    logger.info(
        f"[MODEL] state_fc[0]: Linear({_state_fc0.in_features}, {_state_fc0.out_features}) "
        f"— state_dim={'✓ 31' if _state_fc0.in_features == 31 else f'⚠ {_state_fc0.in_features} (expected 31)'}"
    )
    for name, param in extractor.named_parameters():
        if name.startswith("cnn"):  # covers cnn.* and cnn_fc.*
            param.requires_grad = not freeze_cnn
    logger.info(f"[CNN] freeze_cnn={freeze_cnn} — "
                + ("cnn+cnn_fc frozen" if freeze_cnn else "all params trainable"))

    # --- Per-group LR: CNN cold-start transition ---
    # Fires when the previous stage froze CNN but this stage unfreezes it.
    # Scales read from ppo_config.yaml (cnn_coldstart_*_lr_scale).
    _prev_conf = next((e for e in ppo_cfg["curriculum"] if e["stage"] == stage - 1), None)
    _prev_freeze_cnn = bool(_prev_conf.get("freeze_cnn", False)) if _prev_conf else False
    if not freeze_cnn and _prev_freeze_cnn:
        _base_lr = float(ppo_cfg["learning_rate"])
        _s_cnn    = float(ppo_cfg.get("cnn_coldstart_cnn_lr_scale",    1.0))
        _s_state  = float(ppo_cfg.get("cnn_coldstart_state_lr_scale",  0.1))
        _s_fusion = float(ppo_cfg.get("cnn_coldstart_fusion_lr_scale", 0.33))
        _ext = model.policy.features_extractor
        _pg_lrs = [
            (_ext.cnn.parameters(),                    _base_lr * _s_cnn),
            (_ext.cnn_fc.parameters(),                 _base_lr * _s_cnn),
            (_ext.state_fc.parameters(),               _base_lr * _s_state),
            (_ext.fusion_fc.parameters(),              _base_lr * _s_fusion),
            (_ext.fusion_norm.parameters(),            _base_lr * _s_fusion),
            (model.policy.mlp_extractor.parameters(),  _base_lr * _s_fusion),
            (model.policy.action_net.parameters(),     _base_lr * _s_fusion),
            (model.policy.value_net.parameters(),      _base_lr * _s_fusion),
        ]
        model.policy.optimizer = torch.optim.Adam(
            [{"params": p, "lr": lr, "initial_lr": lr} for p, lr in _pg_lrs],
            eps=1e-5,
        )
        # SB3 calls _update_learning_rate() each rollout and flattens all param groups
        # to the same LR — restore per-group values afterwards.
        import types as _types
        _orig_update_lr = type(model)._update_learning_rate
        def _patched_update_lr(self_m, optimizers):
            _orig_update_lr(self_m, optimizers)
            for pg in self_m.policy.optimizer.param_groups:
                if "initial_lr" in pg:
                    pg["lr"] = pg["initial_lr"]
        model._update_learning_rate = _types.MethodType(_patched_update_lr, model)
        logger.info(
            f"[CNN COLD-START] per-group LR override: "
            f"cnn/cnn_fc={_base_lr * _s_cnn:.2e} (×{_s_cnn}) | "
            f"state_fc={_base_lr * _s_state:.2e} (×{_s_state}) | "
            f"fusion/mlp/heads={_base_lr * _s_fusion:.2e} (×{_s_fusion})"
        )

    # --- Callbacks ---
    ckpt_callback = CheckpointCallback(
        save_freq=max(check_freq // n_envs, 1),
        save_path=ckpt_dir,
        name_prefix=f"stage{stage}",
        save_vecnormalize=True,
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
            env.save(vecnorm_interrupted_path)
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

        steps_done_in_stage = resume_timesteps - stage_start_step
        remaining_steps = max(int(conf["min_steps"]) - steps_done_in_stage, 0)
        logger.info(
            f"[TRAIN STEPS] stage_start={stage_start_step} resume={resume_timesteps} "
            f"done_in_stage={steps_done_in_stage} min_steps={conf['min_steps']} remaining={remaining_steps}"
        )

        model.learn(
            total_timesteps=remaining_steps,
            callback=[ckpt_callback, monitor_callback],
            reset_num_timesteps=False,
        )
        model.save(os.path.join(project_root, f"{model_prefix}_final"))
        env.save(vecnorm_path)
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
        try:
            stop_bridge_process(xrce_agent)
            logger.info("[UXRCE] shared agent stopped")
        except BaseException as agent_exc:
            logger.warning(f"[UXRCE CLOSE] {type(agent_exc).__name__}: {agent_exc}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, help="Curriculum stage (1-based)")
    args = parser.parse_args()
    run_training(stage=args.stage)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
