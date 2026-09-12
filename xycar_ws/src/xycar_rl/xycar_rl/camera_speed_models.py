from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from xycar_rl.models import ResNet18Encoder, checkpoint_payload


DEFAULT_MIN_SPEED_COMMAND = 4.0
# This is the actor's normalization range, not a deployment speed cap.
# The runtime can use the full learned output when deployment_speed_cap <= 0.
DEFAULT_MAX_SPEED_COMMAND = 25.0


def normalize_speed_command(
    speed_command: float,
    min_speed_command: float = DEFAULT_MIN_SPEED_COMMAND,
    max_speed_command: float = DEFAULT_MAX_SPEED_COMMAND,
) -> float:
    low = float(min_speed_command)
    high = float(max_speed_command)
    if high <= low:
        raise ValueError("max_speed_command must be greater than min_speed_command")
    return max(-1.0, min(1.0, 2.0 * (float(speed_command) - low) / (high - low) - 1.0))


def denormalize_speed_command(
    speed_norm: float,
    min_speed_command: float = DEFAULT_MIN_SPEED_COMMAND,
    max_speed_command: float = DEFAULT_MAX_SPEED_COMMAND,
) -> float:
    low = float(min_speed_command)
    high = float(max_speed_command)
    if high <= low:
        raise ValueError("max_speed_command must be greater than min_speed_command")
    normalized = max(-1.0, min(1.0, float(speed_norm)))
    return low + 0.5 * (normalized + 1.0) * (high - low)


def speed_head_range_transform(
    *,
    old_min_speed_command: float,
    old_max_speed_command: float,
    new_min_speed_command: float,
    new_max_speed_command: float,
    preserve_speed_command: float,
) -> tuple[float, float]:
    """Return a logit affine transform preserving speed and local sensitivity."""
    old_low = float(old_min_speed_command)
    old_high = float(old_max_speed_command)
    new_low = float(new_min_speed_command)
    new_high = float(new_max_speed_command)
    if old_high <= old_low or new_high <= new_low:
        raise ValueError("speed command ranges must be increasing")
    preserve = float(preserve_speed_command)
    if not old_low < preserve < old_high:
        raise ValueError("preserved speed must be inside the old command range")
    if not new_low < preserve < new_high:
        raise ValueError("preserved speed must be inside the new command range")

    old_action = normalize_speed_command(preserve, old_low, old_high)
    new_action = normalize_speed_command(preserve, new_low, new_high)
    old_logit = math.atanh(max(-0.999999, min(0.999999, old_action)))
    new_logit = math.atanh(max(-0.999999, min(0.999999, new_action)))
    old_sensitivity = (old_high - old_low) * (1.0 - old_action * old_action)
    new_sensitivity = (new_high - new_low) * (1.0 - new_action * new_action)
    scale = old_sensitivity / max(1.0e-9, new_sensitivity)
    offset = new_logit - scale * old_logit
    return float(scale), float(offset)


def retarget_actor_speed_range(
    actor: nn.Module,
    *,
    old_min_speed_command: float,
    old_max_speed_command: float,
    new_min_speed_command: float,
    new_max_speed_command: float,
    preserve_speed_command: float,
) -> tuple[float, float]:
    """Expand an actor's speed range without an initial physical-speed jump."""
    head = getattr(actor, "head", None)
    if not isinstance(head, nn.Sequential) or len(head) < 2:
        raise TypeError("camera-speed actor has no supported output head")
    output_layer = head[-2]
    if not isinstance(output_layer, nn.Linear) or output_layer.out_features != 2:
        raise TypeError("camera-speed actor output layer must be Linear(..., 2)")
    scale, offset = speed_head_range_transform(
        old_min_speed_command=old_min_speed_command,
        old_max_speed_command=old_max_speed_command,
        new_min_speed_command=new_min_speed_command,
        new_max_speed_command=new_max_speed_command,
        preserve_speed_command=preserve_speed_command,
    )
    with torch.no_grad():
        output_layer.weight[1].mul_(scale)
        output_layer.bias[1].mul_(scale).add_(offset)
    return scale, offset


class CameraSpeedActor(nn.Module):
    """Camera-only actor returning normalized steering and speed actions."""

    def __init__(self, temporal_frames: int = 1) -> None:
        super().__init__()
        self.temporal_frames = int(temporal_frames)
        if self.temporal_frames not in {1, 2}:
            raise ValueError("temporal_frames must be 1 or 2")
        self.image_encoder = ResNet18Encoder(input_channels=3 * self.temporal_frames)
        self.head = nn.Sequential(
            nn.Linear(self.image_encoder.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.image_encoder(image))

    def disable_dropout(self) -> None:
        self.head[2] = nn.Identity()


class CompactCameraSpeedActor(nn.Module):
    """Small canonical-mask actor intended for limited sim DAgger datasets."""

    def __init__(self, temporal_frames: int = 1) -> None:
        super().__init__()
        self.temporal_frames = int(temporal_frames)
        if self.temporal_frames not in {1, 2}:
            raise ValueError("temporal_frames must be 1 or 2")
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3 * self.temporal_frames, 24, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(96 * 3 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.image_encoder(image))

    def disable_dropout(self) -> None:
        self.head[2] = nn.Identity()


class RangeExpandedCameraSpeedActor(nn.Module):
    """Preserve source speeds exactly while learning into a wider range."""

    def __init__(
        self,
        base_actor: nn.Module,
        *,
        source_min_speed_command: float,
        source_max_speed_command: float,
        target_min_speed_command: float,
        target_max_speed_command: float,
    ) -> None:
        super().__init__()
        self.temporal_frames = int(getattr(base_actor, "temporal_frames", 1))
        self.image_encoder = base_actor.image_encoder
        self.head = base_actor.head
        self.source_min_speed_command = float(source_min_speed_command)
        self.source_max_speed_command = float(source_max_speed_command)
        self.target_min_speed_command = float(target_min_speed_command)
        self.target_max_speed_command = float(target_max_speed_command)
        if self.source_max_speed_command <= self.source_min_speed_command:
            raise ValueError("source speed range must be increasing")
        if self.target_max_speed_command <= self.target_min_speed_command:
            raise ValueError("target speed range must be increasing")
        self.speed_extension = nn.Linear(int(self.head[0].in_features), 1)
        nn.init.zeros_(self.speed_extension.weight)
        nn.init.zeros_(self.speed_extension.bias)
        self._base_policy_frozen = False
        self._encoder_frozen = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(image)
        base_action = self.head(features)
        source_speed = self.source_min_speed_command + 0.5 * (
            base_action[:, 1] + 1.0
        ) * (
            self.source_max_speed_command - self.source_min_speed_command
        )
        target_speed_action = (
            2.0
            * (source_speed - self.target_min_speed_command)
            / (
                self.target_max_speed_command
                - self.target_min_speed_command
            )
            - 1.0
        )
        base_speed_logit = torch.atanh(
            target_speed_action.clamp(-0.999999, 0.999999)
        )
        expanded_speed_action = torch.tanh(
            base_speed_logit
            + self.speed_extension(features).squeeze(1)
        )
        return torch.stack(
            [base_action[:, 0], expanded_speed_action],
            dim=1,
        )

    def disable_dropout(self) -> None:
        self.head[2] = nn.Identity()

    def freeze_base_policy(self) -> None:
        """Keep perception and steering fixed while training speed extension."""
        self._base_policy_frozen = True
        self._encoder_frozen = True
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)
        for parameter in self.speed_extension.parameters():
            parameter.requires_grad_(True)
        self.image_encoder.eval()
        self.head.eval()

    def freeze_encoder(self) -> None:
        """Keep visual features fixed while fine-tuning the control heads."""
        self._encoder_frozen = True
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad_(False)
        self.image_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_frozen:
            self.image_encoder.eval()
        if self._base_policy_frozen:
            self.head.eval()
            self.speed_extension.train(mode)
        return self

    @property
    def speed_range_expansion(self) -> dict[str, float]:
        return {
            "source_min_speed_command": self.source_min_speed_command,
            "source_max_speed_command": self.source_max_speed_command,
            "target_min_speed_command": self.target_min_speed_command,
            "target_max_speed_command": self.target_max_speed_command,
        }


def expand_actor_speed_range(
    actor: nn.Module,
    *,
    source_min_speed_command: float,
    source_max_speed_command: float,
    target_min_speed_command: float,
    target_max_speed_command: float,
) -> RangeExpandedCameraSpeedActor:
    return RangeExpandedCameraSpeedActor(
        actor,
        source_min_speed_command=source_min_speed_command,
        source_max_speed_command=source_max_speed_command,
        target_min_speed_command=target_min_speed_command,
        target_max_speed_command=target_max_speed_command,
    )


def initialize_temporal_actor(
    source: CameraSpeedActor,
    temporal_frames: int = 2,
) -> CameraSpeedActor:
    """Expand a single-frame actor while preserving its current-frame behavior."""
    if source.temporal_frames != 1:
        raise ValueError("source actor must use one temporal frame")
    target = CameraSpeedActor(temporal_frames=temporal_frames)
    source_state = source.state_dict()
    target_state = target.state_dict()
    first_conv_key = "image_encoder.features.0.weight"
    for key, value in source_state.items():
        if key != first_conv_key:
            target_state[key] = value.detach().clone()
    source_conv = source_state[first_conv_key]
    expanded_conv = torch.zeros_like(target_state[first_conv_key])
    expanded_conv[:, -3:, :, :] = source_conv
    target_state[first_conv_key] = expanded_conv
    target.load_state_dict(target_state, strict=True)
    return target


class CameraOnlyCriticEncoder(nn.Module):
    def __init__(self, temporal_frames: int = 1) -> None:
        super().__init__()
        self.temporal_frames = int(temporal_frames)
        if self.temporal_frames not in {1, 2}:
            raise ValueError("temporal_frames must be 1 or 2")
        self.image = nn.Sequential(
            nn.Conv2d(
                3 * self.temporal_frames,
                24,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
        )
        self.output_dim = 64 * 3 * 5

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.image(image)


class CameraSpeedQNetwork(nn.Module):
    def __init__(self, temporal_frames: int = 1) -> None:
        super().__init__()
        self.encoder = CameraOnlyCriticEncoder(temporal_frames=temporal_frames)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim + 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, image: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(image), action], dim=1))


class CameraSpeedTwinCritic(nn.Module):
    def __init__(self, temporal_frames: int = 1) -> None:
        super().__init__()
        self.q1 = CameraSpeedQNetwork(temporal_frames=temporal_frames)
        self.q2 = CameraSpeedQNetwork(temporal_frames=temporal_frames)

    def forward(self, image: torch.Tensor, action: torch.Tensor):
        return self.q1(image, action), self.q2(image, action)


def load_camera_speed_actor(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict]:
    payload = checkpoint_payload(checkpoint_path, device)
    model_type = payload.get("model_type")
    expanded = str(model_type).endswith("_range_expanded")
    base_model_type = (
        str(model_type).removesuffix("_range_expanded")
        if expanded
        else str(model_type)
    )
    if base_model_type not in {
        "camera_speed_resnet18",
        "camera_speed_temporal_resnet18",
        "camera_speed_compact",
        "camera_speed_temporal_compact",
    }:
        raise ValueError(
            "expected a camera-speed ResNet18 checkpoint, got "
            f"{model_type!r}"
        )
    temporal_frames = int(
        payload.get(
            "temporal_frames",
            2 if "temporal" in str(model_type) else 1,
        )
    )
    actor = (
        CompactCameraSpeedActor(temporal_frames=temporal_frames)
        if "compact" in base_model_type
        else CameraSpeedActor(temporal_frames=temporal_frames)
    )
    if expanded:
        expansion = payload.get("speed_range_expansion")
        if not isinstance(expansion, dict):
            raise ValueError(
                "range-expanded camera-speed checkpoint has no expansion metadata"
            )
        actor = expand_actor_speed_range(actor, **expansion)
    actor.disable_dropout()
    state_dict = payload.get("actor_state_dict", payload.get("state_dict"))
    if not isinstance(state_dict, dict):
        raise ValueError("camera-speed checkpoint has no actor state dict")
    actor.load_state_dict(state_dict, strict=True)
    actor.to(device).eval()
    return actor, payload
