import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ─────────────────────────────────────────────
#  Residual MLP block (2 lớp, skip connection)
# ─────────────────────────────────────────────
class ResidualMLP(nn.Module):
    """
    2-layer Residual MLP với SiLU activation.
    Input và output cùng chiều `dim` để skip connection hoạt động.

        out = x + Linear(SiLU(Linear(SiLU(x))))
    """

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ─────────────────────────────────────────────
#  Feature Extractor chính
# ─────────────────────────────────────────────
class DepthStateExtractor(BaseFeaturesExtractor):
    """
    Kiến trúc xử lý observation dạng Dict gồm:
        - "depth" : (3, H, W)  — ảnh depth / stereo từ camera
        - "state" : (22,)      — state vector đã normalize

    State vector layout (22 chiều):
        [0:3]   vel FLU norm          (vx, vy, vz body)
        [3:6]   ang_vel FLU norm      (roll_rate, pitch_rate, yaw_rate)
        [6]     altitude norm
        [7:10]  goal body-FLU norm    (gx, gy, gz)
        [10:14] quaternion            (qw, qx, qy, qz)  — unit norm, [-1, 1]
        [14:18] last action           (vx, vy, vz, yaw_rate)
        [18:22] fence dist body-FLU   (forward, back, left, right)

    Flow:
        depth → CNN → Linear+SiLU(3136→256) ──────────────┐
                                                          cat(288) → fusion_fc → ResidualMLP → features
        state(22) → Linear+SiLU(22→32) ───────────────────┘
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        depth_channels = observation_space["depth"].shape[0]
        n_state        = observation_space["state"].shape[0]

        assert depth_channels == 3, (
            f"Expects depth channels=3, got {depth_channels}"
        )
        assert n_state == 22, (
            f"Expects state dim=22, got {n_state}"
        )

        # ── Nhánh CNN (depth) ──────────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=8, stride=4), nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.SiLU(),
            nn.Flatten(),
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(3136, 256),
            nn.SiLU(),
        )

        # ── Nhánh State ────────────────────────────────────────────────────
        # 22 chiều đã normalize đồng nhất → nâng lên 32 để cân bằng với CNN
        # Giữ SiLU vì cần học tương tác phi tuyến giữa các nhóm
        # (vel × goal, ang_vel × quaternion, v.v.)
        self.state_fc = nn.Sequential(
            nn.Linear(22, 32),
            nn.SiLU(),
        )

        # ── Fusion → Residual MLP ──────────────────────────────────────────
        # cat([cnn(256), state(32)]) = 288 → features_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 32, features_dim),
            nn.SiLU(),
        )
        self.residual_mlp = ResidualMLP(features_dim)

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, obs: dict) -> torch.Tensor:
        d = self.cnn_fc(self.cnn(obs["depth"]))              # (B, 256)
        s = self.state_fc(obs["state"])                      # (B, 32)
        fused = self.fusion_fc(torch.cat([d, s], dim=1))     # (B, 256)
        return self.residual_mlp(fused)                      # (B, 256)