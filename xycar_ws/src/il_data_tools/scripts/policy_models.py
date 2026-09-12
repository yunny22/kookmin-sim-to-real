#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


IMAGE_MODEL_TYPES = {
    "pilotnet",
    "mobilenet_v3_small",
    "resnet18",
    "vit_tiny",
}
PHASE_MODEL_TYPES = {
    "pilotnet_phase",
    "mobilenet_v3_small_phase",
    "resnet18_phase",
    "vit_tiny_phase",
}
LIDAR_MODEL_TYPES = {
    "resnet18_lidar",
}
SUPPORTED_MODEL_TYPES = tuple(
    sorted(IMAGE_MODEL_TYPES | PHASE_MODEL_TYPES | LIDAR_MODEL_TYPES)
)


class PilotNetPolicy(nn.Module):
    """Small NVIDIA PilotNet-inspired steering regression model."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = PilotNetEncoder()
        self.head = RegressionHead(self.encoder.feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(image))


class PilotNetEncoder(nn.Module):
    # Preserve coarse left/right spatial layout for steering regression.
    feature_dim = 64 * 2 * 4

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class MobileNetV3SmallPolicy(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        encoder = MobileNetV3SmallEncoder(pretrained=pretrained)
        self.encoder = encoder
        self.head = RegressionHead(encoder.feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(image))


class MobileNetV3SmallEncoder(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for mobilenet_v3_small. "
                "Install torchvision in the training environment."
            ) from exc

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)
        self.features = nn.Sequential(base.features, base.avgpool, nn.Flatten())
        self.feature_dim = base.classifier[0].in_features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image)


class ResNet18Policy(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        encoder = ResNet18Encoder(pretrained=pretrained)
        self.encoder = encoder
        self.head = RegressionHead(encoder.feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(image))


class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for resnet18. "
                "Install torchvision in the training environment."
            ) from exc

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        self.feature_dim = base.fc.in_features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image)


class Lidar1DEncoder(nn.Module):
    """Encode normalized LaserScan ranges and validity mask [B, 2, N]."""

    feature_dim = 64

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


class ResNet18LidarPolicy(nn.Module):
    """Mid-fusion steering policy using a camera image and synchronized 2D LiDAR."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.image_encoder = ResNet18Encoder(pretrained=pretrained)
        self.lidar_encoder = Lidar1DEncoder()
        self.head = RegressionHead(
            self.image_encoder.feature_dim + self.lidar_encoder.feature_dim
        )

    def forward(self, image: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        lidar_features = self.lidar_encoder(lidar)
        return self.head(torch.cat([image_features, lidar_features], dim=1))


class ViTTinyPolicy(nn.Module):
    """Experimental ViT-Tiny steering model."""

    def __init__(
        self,
        input_width: int = 160,
        input_height: int = 90,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        encoder = ViTTinyEncoder(input_width, input_height, pretrained=pretrained)
        self.encoder = encoder
        self.head = RegressionHead(encoder.feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(image))


class ViTTinyEncoder(nn.Module):
    def __init__(
        self,
        input_width: int,
        input_height: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "vit_tiny is experimental and requires optional dependency timm. "
                "Install timm only when you intentionally test ViT-Tiny."
            ) from exc

        self.model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )
        self.feature_dim = self.model.num_features
        self.input_size = (input_height, input_width)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # timm ViT expects 224x224 by default; resize inside the model wrapper so
        # the rest of the pipeline can keep using 160x90 images.
        image = nn.functional.interpolate(
            image,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        return self.model(image)


class PhaseConditionedPolicy(nn.Module):
    """Wrap an image encoder and condition steering on overtake phase [B, 1]."""

    def __init__(self, image_encoder: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.head = RegressionHead(feature_dim + 1)

    def forward(self, image: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(image)
        if phase.dim() == 1:
            phase = phase.unsqueeze(1)
        phase = phase.to(device=features.device, dtype=features.dtype)
        return self.head(torch.cat([features, phase], dim=1))


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


def create_policy_model(
    model_type: str,
    input_width: int = 160,
    input_height: int = 90,
    use_phase: bool = False,
    pretrained: bool = False,
) -> nn.Module:
    normalized = normalize_model_type(model_type)
    if normalized == "resnet18_lidar":
        if use_phase:
            raise ValueError("resnet18_lidar does not accept overtake phase input")
        return ResNet18LidarPolicy(pretrained=pretrained)
    phase_from_name = normalized.endswith("_phase")
    base_type = normalized[: -len("_phase")] if phase_from_name else normalized
    phase_enabled = use_phase or phase_from_name

    if phase_enabled:
        encoder, feature_dim = create_image_encoder(
            base_type,
            input_width=input_width,
            input_height=input_height,
            pretrained=pretrained,
        )
        return PhaseConditionedPolicy(encoder, feature_dim)

    if base_type == "pilotnet":
        return PilotNetPolicy()
    if base_type == "mobilenet_v3_small":
        return MobileNetV3SmallPolicy(pretrained=pretrained)
    if base_type == "resnet18":
        return ResNet18Policy(pretrained=pretrained)
    if base_type == "vit_tiny":
        return ViTTinyPolicy(input_width, input_height, pretrained=pretrained)
    raise ValueError(f"unsupported model_type: {model_type}")


def create_image_encoder(
    base_type: str,
    input_width: int,
    input_height: int,
    pretrained: bool,
) -> Tuple[nn.Module, int]:
    if base_type == "pilotnet":
        encoder = PilotNetEncoder()
        return encoder, encoder.feature_dim
    if base_type == "mobilenet_v3_small":
        encoder = MobileNetV3SmallEncoder(pretrained=pretrained)
        return encoder, encoder.feature_dim
    if base_type == "resnet18":
        encoder = ResNet18Encoder(pretrained=pretrained)
        return encoder, encoder.feature_dim
    if base_type == "vit_tiny":
        encoder = ViTTinyEncoder(input_width, input_height, pretrained=pretrained)
        return encoder, encoder.feature_dim
    raise ValueError(f"unsupported base model_type: {base_type}")


def normalize_model_type(model_type: str) -> str:
    value = str(model_type).strip().lower().replace("-", "_")
    if value not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "model_type must be one of "
            + ", ".join(SUPPORTED_MODEL_TYPES)
            + f"; got {model_type!r}"
        )
    return value


def model_uses_phase(model_type: str, use_phase: bool = False) -> bool:
    return bool(use_phase or normalize_model_type(model_type).endswith("_phase"))


def model_uses_lidar(model_type: str) -> bool:
    return normalize_model_type(model_type) in LIDAR_MODEL_TYPES
