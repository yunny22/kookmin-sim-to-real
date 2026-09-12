from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class ResNet18Encoder(nn.Module):
    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        from torchvision import models

        base = models.resnet18(weights=None)
        if int(input_channels) != 3:
            base.conv1 = nn.Conv2d(
                int(input_channels),
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        self.features = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        self.feature_dim = int(base.fc.in_features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image)


class Lidar1DEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )
        self.feature_dim = 64 * 4

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        return self.features(lidar)


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ResNet18LidarActor(nn.Module):
    """BC-compatible actor whose output is normalized to the physical limit."""

    def __init__(
        self,
        *,
        bc_max_steer_command: float = 100.0,
        action_max_steer_command: float = 42.0,
    ) -> None:
        super().__init__()
        self.image_encoder = ResNet18Encoder()
        self.lidar_encoder = Lidar1DEncoder()
        self.head = RegressionHead(
            self.image_encoder.feature_dim + self.lidar_encoder.feature_dim
        )
        self.output_scale = float(bc_max_steer_command) / float(
            action_max_steer_command
        )

    def forward(self, image: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        lidar_features = self.lidar_encoder(lidar)
        bc_normalized = self.head(
            torch.cat([image_features, lidar_features], dim=1)
        )
        return torch.clamp(bc_normalized * self.output_scale, -1.0, 1.0)

    def disable_dropout(self) -> None:
        self.head.net[2] = nn.Identity()


class CriticEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
        )
        self.lidar = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=3, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.output_dim = 64 * 3 * 5 + 32 * 8

    def forward(self, image: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.image(image), self.lidar(lidar)], dim=1)


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = CriticEncoder()
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim + 1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        lidar: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        state = self.encoder(image, lidar)
        return self.head(torch.cat([state, action], dim=1))


class TwinCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q1 = QNetwork()
        self.q2 = QNetwork()

    def forward(self, image, lidar, action):
        return (
            self.q1(image, lidar, action),
            self.q2(image, lidar, action),
        )


class ResidualActor(nn.Module):
    """Small correction policy for online S-curve and recovery refinement."""

    def __init__(self, max_residual_norm: float = 0.25) -> None:
        super().__init__()
        self.encoder = CriticEncoder()
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)
        self.max_residual_norm = float(max_residual_norm)

    def forward(self, image, lidar):
        return self.head(self.encoder(image, lidar)) * self.max_residual_norm


def checkpoint_payload(path: str | Path, device: str | torch.device = "cpu") -> dict:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a dictionary: {path}")
    return payload


def load_bc_actor(
    checkpoint_path: str | Path,
    *,
    action_max_steer_command: float = 42.0,
    device: str | torch.device = "cpu",
) -> tuple[ResNet18LidarActor, dict[str, Any]]:
    payload = checkpoint_payload(checkpoint_path, device)
    model_type = str(payload.get("model_type", ""))
    if model_type != "resnet18_lidar":
        raise ValueError(
            f"expected resnet18_lidar BC checkpoint, got {model_type!r}"
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("BC checkpoint has no state_dict")
    actor = ResNet18LidarActor(
        bc_max_steer_command=float(payload.get("max_steer_deg", 100.0)),
        action_max_steer_command=action_max_steer_command,
    )
    actor.load_state_dict(state_dict, strict=True)
    actor.disable_dropout()
    actor.to(device)
    return actor, payload


def load_rl_actor(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ResNet18LidarActor, dict[str, Any]]:
    payload = checkpoint_payload(checkpoint_path, device)
    config = payload.get("actor_config", {})
    actor = ResNet18LidarActor(
        bc_max_steer_command=float(config.get("bc_max_steer_command", 100.0)),
        action_max_steer_command=float(
            config.get("action_max_steer_command", 42.0)
        ),
    )
    actor.disable_dropout()
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.to(device)
    actor.eval()
    return actor, payload
