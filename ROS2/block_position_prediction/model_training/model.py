"""Hybrid tactile CNN model for block pose prediction."""

from __future__ import annotations

import torch
from torch import nn


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int | tuple[int, int] = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class TactilePoseNet(nn.Module):
    """Small CNN with a physical-feature side branch."""

    def __init__(self, physics_dim: int = 10) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            ConvGNAct(4, 24),
            ResidualBlock(24),
            ConvGNAct(24, 32, stride=(1, 2)),
            ResidualBlock(32),
            ConvGNAct(32, 48, stride=2),
            ResidualBlock(48),
        )
        self.physics_mlp = nn.Sequential(
            nn.Linear(physics_dim, 32),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(48 * 4 * 4 + 32, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
        )
        self.position_head = nn.Linear(64, 2)
        self.yaw_head = nn.Linear(64, 2)
        self.presence_head = nn.Linear(64, 1)
        self.presence_available = True

    def forward(self, maps: torch.Tensor, physics: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.cnn(maps).flatten(start_dim=1)
        physics_features = self.physics_mlp(physics)
        fused = self.fusion(torch.cat([features, physics_features], dim=1))
        return {
            "position": self.position_head(fused),
            "yaw_vector": self.yaw_head(fused),
            "presence_logit": self.presence_head(fused).squeeze(-1),
        }


def load_tactile_pose_state(model: TactilePoseNet, state_dict: dict) -> tuple[list[str], list[str]]:
    """Load new or legacy checkpoints.

    Legacy checkpoints do not contain the presence head. Position and yaw
    weights are still valid, while confidence must be treated as unavailable.
    """

    result = model.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    model.presence_available = not any(key.startswith("presence_head.") for key in missing)
    return missing, unexpected

