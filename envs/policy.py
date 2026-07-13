import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    import sys
    import os
    _srk_path = os.path.join(os.path.dirname(__file__), "..", "SRK-PPO_Deployment_Package", "RL_Deployment_Package")
    if os.path.exists(_srk_path):
        sys.path.insert(0, _srk_path)
    from srk_layer import SRKEncoder
    _SRK_AVAILABLE = True
except ImportError:
    _SRK_AVAILABLE = False
    SRKEncoder = None


# ─────────────────────────────────────────────
#  Feature Extractor chính
# ─────────────────────────────────────────────
class DepthStateExtractor(BaseFeaturesExtractor):
    """
    Kiến trúc xử lý observation dạng Dict gồm:
        - "depth" : (3, H, W)  — ảnh depth / stereo từ camera
        - "state" : (22,)      — state vector đã normalize

    State vector layout (31 chiều):
        [0:3]   vel FLU norm          (vx, vy, vz body)
        [3:6]   ang_vel FLU norm      (roll_rate, pitch_rate, yaw_rate)
        [6]     altitude norm
        [7:10]  goal body-FLU norm    (gx, gy, gz)
        [10:14] orientation           (sin_yaw, cos_yaw, pitch/45°, roll/45°)
        [14:18] last action A_t       (vx, vy, vz, yaw_rate) in [-1,1]
        [18:22] delta_A1 = A_t - A_{t-1}   kinematic velocity, clip/2 → [-1,1]
        [22:26] delta_A2 = ΔA1_t - ΔA1_{t-1}  kinematic accel,  clip/2 → [-1,1]
        [26:30] fence dist body-FLU   (forward, back, left, right)
        [30]    DFA progress          q/N ∈ [0,1]

    Flow:
        depth → CNN → Linear+SiLU(5184→256) ────────────────────┐
                                                                cat(320) → fusion_fc → LayerNorm → features
        state(31) → Linear+SiLU(31→64) → Linear+SiLU(64→64) ───┘
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        depth_channels = observation_space["depth"].shape[0]
        n_state        = observation_space["state"].shape[0]

        assert depth_channels == 3, (
            f"Expects depth channels=3, got {depth_channels}"
        )
        assert n_state == 31, (
            f"Expects state dim=31, got {n_state}"
        )

        # ── Nhánh CNN (depth) ──────────────────────────────────────────────
        # k=4,s=2 throughout: 84→41→19→9, flatten=5184
        # Better near-field resolution vs Atari-style k=8,s=4 (4px→8px in 0-3m zone)
        self.cnn = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=4, stride=2), nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.SiLU(),
            nn.Flatten(),
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(5184, 256),
            nn.SiLU(),
        )

        # ── Nhánh State ────────────────────────────────────────────────────
        # n_state→64→64: dùng n_state thay hardcode để không phải sửa lại khi dim thay đổi
        self.state_fc = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

        # ── Fusion → LayerNorm ─────────────────────────────────────────────
        # cat([cnn(256), state(64)]) = 320 → features_dim
        # LayerNorm thay ResidualMLP: normalize gradient scale, ít params hơn
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 64, features_dim),
            nn.SiLU(),
        )
        self.fusion_norm = nn.LayerNorm(features_dim)

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, obs: dict) -> torch.Tensor:
        d = self.cnn_fc(self.cnn(obs["depth"]))              # (B, 256)
        s = self.state_fc(obs["state"])                      # (B, 64)
        fused = self.fusion_fc(torch.cat([d, s], dim=1))     # (B, 256)
        return self.fusion_norm(fused)                       # (B, 256)


# ─────────────────────────────────────────────
#  Hybrid SRK Feature Extractor (Stage 3+)
# ─────────────────────────────────────────────
class HybridSRKExtractor(BaseFeaturesExtractor):
    """
    Hybrid CNN + SRK feature extractor for stage 3+ when use_srk=true.

    Architecture:
        depth(3,84,84) → CNN → cnn_fc(5184→256) → 256
        state(31) → SRKEncoder(history=15) → 128
        concat(256+128=384) → fusion_fc(384→256) → LayerNorm → 256

    CNN branch unchanged from DepthStateExtractor (spatial perception).
    SRK branch replaces state MLP (temporal memory via reservoir).
    """

    def __init__(self, observation_space, features_dim: int = 256):
        if not _SRK_AVAILABLE:
            raise ImportError(
                "SRKEncoder not available — check SRK-PPO_Deployment_Package/ exists "
                "and srk_layer.py is accessible"
            )

        super().__init__(observation_space, features_dim)

        depth_channels = observation_space["depth"].shape[0]
        n_state = observation_space["state"].shape[0]

        assert depth_channels == 3, f"Expects depth channels=3, got {depth_channels}"
        assert n_state == 31, f"Expects state dim=31, got {n_state}"

        # ── CNN branch (unchanged from DepthStateExtractor) ───────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2), nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.SiLU(),
            nn.Flatten(),
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(5184, 256),
            nn.SiLU(),
        )

        # ── SRK branch (replaces state MLP) ────────────────────────────────
        # SRKEncoder: 15-step history buffer + reservoir + RFF + NGRC
        # Output dim 128 (half of CNN's 256) — temporal memory is smaller
        # feature space than spatial perception
        self.srk_encoder = SRKEncoder(
            input_dim=n_state,
            hidden_dim=128,
            output_dim=128,
            history_len=15,
            reservoir_size=256,
            spectral_radius=0.9,
            input_scaling=0.1,
            leak_rate=0.3,
        )

        # ── Fusion → LayerNorm ─────────────────────────────────────────────
        # cat([cnn(256), srk(128)]) = 384 → features_dim(256)
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 128, features_dim),
            nn.SiLU(),
        )
        self.fusion_norm = nn.LayerNorm(features_dim)

    def forward(self, obs: dict) -> torch.Tensor:
        d = self.cnn_fc(self.cnn(obs["depth"]))        # (B, 256)
        s = self.srk_encoder(obs["state"])             # (B, 128)
        fused = self.fusion_fc(torch.cat([d, s], dim=1))  # (B, 256)
        return self.fusion_norm(fused)

    def reset(self):
        """Reset SRK reservoir state at episode boundary — called by env reset."""
        self.srk_encoder.reset()
