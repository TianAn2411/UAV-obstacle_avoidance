import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DepthStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # Lấy số chiều của phần vector trạng thái (x, y, z, v...)
        n_state = observation_space["state"].shape[0]
        depth_channels = observation_space["depth"].shape[0]
        assert depth_channels == 3, f"DepthStateExtractor expects depth channels=3, got {depth_channels}"
        assert n_state > 0, f"DepthStateExtractor expects state dim > 0, got {n_state}"

        # Nhánh CNN: 
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),

        )

        # Lớp kết nối cho nhánh ảnh
        self.cnn_fc = nn.Sequential(nn.Linear(3136, 256), nn.ReLU())

        # Lớp kết nối cho nhánh vector trạng thái
        self.state_fc = nn.Sequential(nn.Linear(n_state, 64), nn.ReLU())

        # Lớp tổng hợp (Fusion Layer): Kết hợp thông tin từ cả 2 nhánh
        self.head = nn.Sequential(
            nn.Linear(256 + 64, features_dim), nn.ReLU()
        )

    def forward(self, obs):
        # obs là một Dictionary chứa "depth" và "state"
        d = self.cnn_fc(self.cnn(obs["depth"]))
        s = self.state_fc(obs["state"])

        # Ghép (concatenate) hai vector thông tin lại với nhau
        return self.head(torch.cat([d, s], dim=1))
