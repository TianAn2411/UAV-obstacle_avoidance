
import sys
import os
import csv
from collections import deque
from obstacle_avoidance.utils.logger import setup_logger, PROJECT_ROOT

logger = setup_logger("MONITOR", log_file=os.path.join(PROJECT_ROOT, "runs", "train_main.log"))

import numpy as np
import time
from stable_baselines3.common.callbacks import BaseCallback

class TrainingMonitor(BaseCallback):
    def __init__(
        self,
        check_freq: int = 5000,
        csv_freq: int = 10000,
        verbose: int = 1,
        stage: int = 1,
        stage_steps: int = 0,
        stage_start_step: int = 0,
        n_envs: int = 1,
        num_pillars: int = 0,
        is_symbolic: bool = False,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.csv_freq = int(csv_freq)
        self.stage = int(stage)
        self.stage_steps = int(stage_steps)
        self.stage_start_step = int(stage_start_step)
        self.n_envs = int(n_envs)
        self.num_pillars = str(num_pillars)
        self.is_symbolic = bool(is_symbolic)
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_episodes = 0
        self.start_time = None
        self.episode_reward_sum = 0.0
        self.episode_step_sum = 0
        self.ep_reward_buffer = deque(maxlen=100)
        self.ep_len_buffer = deque(maxlen=100)
        self.window_episode_count = 0
        self.done_reason_counts = {
            "goal_xy": 0,
            "goal_3d": 0,
            "success": 0,
            "collision": 0,
            "out_of_fence": 0,
            "fell_to_ground": 0,
            "ground": 0,
        }
        self.total_done_count = 0
        self._ep_comp_window: dict[str, float] = {}
        self._ep_comp_window_count: int = 0
        # SB3 only auto-logs train/std for policies with a log_std attribute
        # (Gaussian) -- BetaDistribution has no such attribute (parameterized
        # by alpha/beta concentration instead), so Track B gets no std log by
        # default. Accumulated here per-step (policy.action_dist.distribution
        # already reflects the batch just sampled by the time _on_step fires,
        # no extra forward pass needed) and reduced to a mean at rollout end,
        # same cadence/shape as SB3's own train/std.
        self._beta_alpha_accum: list[float] = []
        self._beta_beta_accum: list[float] = []
        # Split by track (raw vs symbolic) — same runs/{raws,symbolics}/ layout
        # as ckpt_dir/interrupt_dir in train.py, so a stage's CSV history never
        # mixes data from the two (different reward/action-space semantics).
        self.progress_csv_path = os.path.join(
            PROJECT_ROOT,
            "runs",
            "symbolics" if self.is_symbolic else "raws",
            f"training_progress_stage{self.stage}.csv",
        )
        self.progress_csv_fields = [
            "wall_time",
            "stage",
            "train_step",
            "progress_pct",
            "episodes",
            "num_pillars",
            "ep_rew_mean",
            "ep_len_mean",
            "success_rate",
            "collision_rate",
            "out_fence_rate",
            "ground_rate",
            "approx_kl",
            "clip_fraction",
            "explained_variance",
        ]

    def _on_training_start(self):
        # Biểu ngữ BẮT ĐẦU HUẤN LUYỆN
        self.start_time = time.time()
        os.makedirs(os.path.dirname(self.progress_csv_path), exist_ok=True)
        if not os.path.exists(self.progress_csv_path):
            with open(self.progress_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.progress_csv_fields)
                writer.writeheader()
        logger.info(f"\n{'='*60}")
        logger.info("BẮT ĐẦU HUẤN LUYỆN")
        logger.info(f"{'='*60}")
        logger.info(f"Số bước thu thập mỗi lượt (n_steps): {self.model.n_steps}")
        logger.info(f"Batch size: {self.model.batch_size}")
        logger.info(f"Learning rate: {self.model.learning_rate:.6f}")
        logger.info(f"{'='*60}\n")

    def _on_step(self):
        # Keep per-env curriculum step exactly synced with SB3 global num_timesteps.
        try:
            self.training_env.set_attr("global_train_step", int(self.num_timesteps))
        except Exception:
            pass

        if self.is_symbolic:
            dist = getattr(getattr(self.model.policy, "action_dist", None), "distribution", None)
            if dist is not None and hasattr(dist, "concentration1"):
                try:
                    self._beta_alpha_accum.append(float(dist.concentration1.detach().mean().item()))
                    self._beta_beta_accum.append(float(dist.concentration0.detach().mean().item()))
                except Exception:
                    pass

        infos = self.locals.get("infos") or []
        for info in infos:
            if not isinstance(info, dict):
                continue
            ep_info = info.get("episode")
            if isinstance(ep_info, dict):
                try:
                    self.ep_reward_buffer.append(float(ep_info.get("r", 0.0)))
                    self.ep_len_buffer.append(float(ep_info.get("l", 0.0)))
                    self.window_episode_count += 1
                except Exception:
                    pass

        dones = self.locals.get("dones")
        if infos is not None and dones is not None:
            for done_flag, info in zip(dones, infos):
                if not done_flag or not isinstance(info, dict):
                    continue
                self.total_done_count += 1
                reason = str(info.get("done_reason", ""))
                if reason in self.done_reason_counts:
                    self.done_reason_counts[reason] += 1
                ep_comps = info.get("ep_reward_components")
                if isinstance(ep_comps, dict):
                    for k, v in ep_comps.items():
                        self._ep_comp_window[k] = self._ep_comp_window.get(k, 0.0) + float(v)
                    self._ep_comp_window_count += 1
                    self._log_ep_reward(
                        reason, ep_comps,
                        steps=int(info.get("step_count", 0)),
                        dist_xy=float(info.get("dist_xy", 0.0)),
                        dist_start=float(info.get("dist_start", 0.0)),
                        pos=info.get("pos"),
                        start=info.get("start"),
                        dfa_q=int(info.get("dfa_q", 0)),
                        dfa_N=int(info.get("dfa_N", 1)),
                    )

        # 1. Log tiến độ định kỳ [📊]
        if self.n_calls % self.check_freq == 0:
            self._log_progress(write_csv=False)

        if self.csv_freq > 0 and self.n_calls % self.csv_freq == 0:
            self._log_progress(write_csv=True)

        # 3. Kiểm tra và log khi kết thúc Episode
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            # Chỉ log nếu đây là episode mới
            if self.total_episodes < self.model._episode_num:
                self.total_episodes = self.model._episode_num
                self.episode_reward_sum += ep_info['r']
                self.episode_step_sum += ep_info['l']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])

        return True

    def _on_rollout_end(self) -> None:
        if not self.is_symbolic or not self._beta_alpha_accum:
            return
        alpha_mean = float(np.mean(self._beta_alpha_accum))
        beta_mean = float(np.mean(self._beta_beta_accum))
        s = alpha_mean + beta_mean
        # Beta(a,b) variance = a*b / (s^2 * (s+1)) -- same interpretive role
        # as Gaussian's std: shrinks as the policy grows more confident/peaked.
        beta_var = (alpha_mean * beta_mean) / (s * s * (s + 1.0))
        self.model.logger.record("train/beta_alpha_mean", alpha_mean)
        self.model.logger.record("train/beta_beta_mean", beta_mean)
        self.model.logger.record("train/beta_std_mean", float(np.sqrt(max(0.0, beta_var))))
        self._beta_alpha_accum.clear()
        self._beta_beta_accum.clear()

    def _log_progress(self, write_csv: bool = False):
        elapsed = max(1e-6, time.time() - self.start_time)
        timesteps_done = int(self.num_timesteps)
        fps = int(timesteps_done / elapsed)

        name_to_val = getattr(getattr(self.model, "logger", None), "name_to_value", {}) or {}
        if len(self.ep_reward_buffer) > 0:
            ep_rew_mean = float(np.mean(self.ep_reward_buffer))
        elif hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            ep_infos = list(self.model.ep_info_buffer)
            rewards = [float(ep["r"]) for ep in ep_infos if "r" in ep]
            ep_rew_mean = float(np.mean(rewards)) if rewards else 0.0
        else:
            ep_rew_mean = float(name_to_val.get("rollout/ep_rew_mean", np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0.0))

        if len(self.ep_len_buffer) > 0:
            ep_len_mean = float(np.mean(self.ep_len_buffer))
        elif hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            ep_infos = list(self.model.ep_info_buffer)
            lengths = [float(ep["l"]) for ep in ep_infos if "l" in ep]
            ep_len_mean = float(np.mean(lengths)) if lengths else 0.0
        else:
            ep_len_mean = float(name_to_val.get("rollout/ep_len_mean", np.mean(self.episode_lengths[-50:]) if self.episode_lengths else 0.0))

        approx_kl = name_to_val.get("train/approx_kl", None)
        clip_fraction = name_to_val.get("train/clip_fraction", None)
        explained_variance = name_to_val.get("train/explained_variance", None)
        std = name_to_val.get("train/std", None)

        total_done = max(1, self.total_done_count)
        success_count = (
            self.done_reason_counts["goal_xy"]
            + self.done_reason_counts["goal_3d"]
            + self.done_reason_counts["success"]
        )
        collision_count = self.done_reason_counts["collision"]
        out_fence_count = self.done_reason_counts["out_of_fence"]
        ground_count = self.done_reason_counts["fell_to_ground"] + self.done_reason_counts["ground"]

        line1 = (
            f"[TRAIN] stage={self.stage} "
            f"step={timesteps_done}/{self.stage_steps if self.stage_steps > 0 else '?'} "
            f"fps={fps} envs={self.n_envs} pillars={self.num_pillars}"
        )
        line2 = (
            f"ep_rew={ep_rew_mean:.2f} ep_len={ep_len_mean:.1f} "
            f"success={(100.0 * success_count / total_done):.1f}% "
            f"collision={(100.0 * collision_count / total_done):.1f}% "
            f"out_fence={(100.0 * out_fence_count / total_done):.1f}% "
            f"ground={(100.0 * ground_count / total_done):.1f}%"
        )

        train_parts = []
        if approx_kl is not None:
            train_parts.append(f"KL={float(approx_kl):.4f}")
        if clip_fraction is not None:
            train_parts.append(f"clip={float(clip_fraction):.3f}")
        if explained_variance is not None:
            train_parts.append(f"EV={float(explained_variance):.3f}")
        if std is not None:
            train_parts.append(f"std={float(std):.4f}")
        line3 = " ".join(train_parts)

        logger.info(line1)
        logger.info(line2)
        if self._ep_comp_window_count > 0:
            logger.info(self._format_reward_breakdown())
        if line3:
            logger.info(line3)
        if write_csv:
            self._append_progress_csv(
                timesteps_done=timesteps_done,
                episodes=int(self.window_episode_count),
                ep_rew_mean=ep_rew_mean,
                ep_len_mean=ep_len_mean,
                success_rate=(100.0 * success_count / total_done),
                collision_rate=(100.0 * collision_count / total_done),
                out_fence_rate=(100.0 * out_fence_count / total_done),
                ground_rate=(100.0 * ground_count / total_done),
                approx_kl=approx_kl,
                clip_fraction=clip_fraction,
                explained_variance=explained_variance,
            )

    def _append_progress_csv(
        self,
        timesteps_done: int,
        episodes: int,
        ep_rew_mean: float,
        ep_len_mean: float,
        success_rate: float,
        collision_rate: float,
        out_fence_rate: float,
        ground_rate: float,
        approx_kl,
        clip_fraction,
        explained_variance,
    ):
        progress_pct = 0.0
        if self.stage_steps > 0:
            steps_in_stage = float(timesteps_done - self.stage_start_step)
            progress_pct = min(100.0, 100.0 * steps_in_stage / float(self.stage_steps))

        row = {
            "wall_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "stage": self.stage,
            "train_step": int(timesteps_done),
            "progress_pct": round(progress_pct, 2),
            "episodes": int(episodes),
            "num_pillars": self.num_pillars,
            "ep_rew_mean": f"{float(ep_rew_mean):.2f}",
            "ep_len_mean": f"{float(ep_len_mean):.1f}",
            "success_rate": round(float(success_rate), 4),
            "collision_rate": round(float(collision_rate), 4),
            "out_fence_rate": round(float(out_fence_rate), 4),
            "ground_rate": round(float(ground_rate), 4),
            "approx_kl": "" if approx_kl is None else round(float(approx_kl), 6),
            "clip_fraction": "" if clip_fraction is None else round(float(clip_fraction), 6),
            "explained_variance": "" if explained_variance is None else round(float(explained_variance), 6),
        }

        with open(self.progress_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.progress_csv_fields)
            writer.writerow(row)

        logger.info(
            "[TRAIN MONITOR CSV] "
            f"step={int(timesteps_done)} "
            f"episodes={int(episodes)} "
            f"ep_rew_mean={float(ep_rew_mean):.2f} "
            f"ep_len_mean={float(ep_len_mean):.1f}"
        )
        self.window_episode_count = 0
        self._ep_comp_window.clear()
        self._ep_comp_window_count = 0

    def _log_ep_reward(
        self, reason: str, ep_comps: dict,
        steps: int = 0, dist_xy: float = 0.0, dist_start: float = 0.0,
        pos=None, start=None, dfa_q: int = 0, dfa_N: int = 1,
    ) -> None:
        s = ep_comps
        pbrs     = s.get("pbrs", 0.0)
        rm       = s.get("rm_bonus", 0.0)
        terminal = s.get("terminal", 0.0)
        total    = s.get("total", 0.0)
        r_base   = total - pbrs - rm
        pos_str = "(?,?,?)"
        if pos is not None:
            try:
                p = pos.reshape(-1)
                pos_str = f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})"
            except Exception:
                pass
        start_str = "(?,?)"
        if start is not None:
            try:
                s2 = start.reshape(-1)
                start_str = f"({s2[0]:.1f},{s2[1]:.1f})"
            except Exception:
                pass
        base_fields = [
            "ground", "altitude", "smooth", "speed_penalty", "yaw_align",
            "yaw_rate_penalty", "fall_penalty", "lateral", "near_fence",
            "time", "velocity_goal", "progress",
        ]
        detail = [
            f"{f}={s.get(f, 0.0):.2f}"
            for f in base_fields
            if abs(s.get(f, 0.0)) >= 0.05
        ]
        line = (
            f"[EP] reason={reason} steps={steps} dist={dist_xy:.2f}/{dist_start:.2f} "
            f"dfa={dfa_q}/{dfa_N} start={start_str} pos={pos_str} | "
            f"total={total:.1f} r_base={r_base:.1f} pbrs={pbrs:.2f} rm={rm:.2f} terminal={terminal:.1f}"
        )
        if detail:
            line += " | " + " ".join(detail)
        logger.info(line)

    def _format_reward_breakdown(self) -> str:
        n = max(1, self._ep_comp_window_count)
        s = self._ep_comp_window
        pbrs     = s.get("pbrs", 0.0) / n
        rm       = s.get("rm_bonus", 0.0) / n
        terminal = s.get("terminal", 0.0) / n
        total    = s.get("total", 0.0) / n
        r_base   = total - pbrs - rm
        base_fields = [
            "ground", "altitude", "smooth", "speed_penalty", "yaw_align",
            "yaw_rate_penalty", "fall_penalty", "lateral", "near_fence",
            "time", "velocity_goal", "progress",
        ]
        detail = [
            f"{f}={s.get(f, 0.0) / n:.2f}"
            for f in base_fields
            if abs(s.get(f, 0.0) / n) >= 0.01
        ]
        line = (
            f"[REWARD] r_base={r_base:.2f} pbrs={pbrs:.2f} "
            f"rm={rm:.2f} terminal={terminal:.2f} total={total:.2f}"
        )
        if detail:
            line += " | " + " ".join(detail)
        return line

    def _on_training_end(self):
        # Biểu ngữ HOÀN TẤT HUẤN LUYỆN
        elapsed = time.time() - self.start_time
        logger.info(f"\n{'='*60}")
        logger.info("HOÀN TẤT HUẤN LUYỆN")
        logger.info(f"{'='*60}")
        logger.info(f"Thời gian: {elapsed/60:.1f} phút")
        if self.episode_rewards:
            logger.info(f"Thành công: {sum(1 for r in self.episode_rewards if r > 50)}/{self.total_episodes}")
        logger.info(f"{'='*60}\n")
