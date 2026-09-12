from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from xycar_rl.models import (
    ResidualActor,
    checkpoint_payload,
    load_bc_actor,
    load_rl_actor,
)
from xycar_rl.camera_speed_models import load_camera_speed_actor


class SteeringPolicy:
    def __init__(self, actor, device: torch.device, residual=None) -> None:
        self.actor = actor.to(device).eval()
        self.residual = None if residual is None else residual.to(device).eval()
        self.device = device

    @torch.no_grad()
    def action_components(self, observation: dict) -> tuple[float, float, float]:
        image = torch.as_tensor(
            observation["image"], device=self.device
        ).float().unsqueeze(0)
        lidar = torch.as_tensor(
            observation["lidar"], device=self.device
        ).float().unsqueeze(0)
        base = self.actor(image, lidar)
        residual = (
            torch.zeros_like(base)
            if self.residual is None
            else self.residual(image, lidar)
        )
        final = (base + residual).clamp(-1.0, 1.0)
        return float(final.item()), float(base.item()), float(residual.item())

    def __call__(self, observation: dict) -> float:
        return self.action_components(observation)[0]


class CameraSpeedPolicy:
    def __init__(self, actor, device: torch.device) -> None:
        self.actor = actor.to(device).eval()
        self.device = device
        self.temporal_frames = int(getattr(actor, "temporal_frames", 1))
        self.previous_image: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_image = None

    @torch.no_grad()
    def __call__(self, observation: dict) -> np.ndarray:
        current_image = np.asarray(observation["image"], dtype=np.float32)
        if self.temporal_frames == 2:
            previous_image = (
                current_image
                if self.previous_image is None
                else self.previous_image
            )
            model_image = np.concatenate(
                [previous_image, current_image], axis=0
            )
        else:
            model_image = current_image
        image = torch.as_tensor(
            model_image, device=self.device
        ).float().unsqueeze(0)
        action = self.actor(image).squeeze(0).clamp(-1.0, 1.0)
        self.previous_image = current_image.copy()
        return action.detach().cpu().numpy().astype(np.float32, copy=False)


def load_camera_speed_policy(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CameraSpeedPolicy, dict]:
    device = torch.device(device)
    actor, payload = load_camera_speed_actor(checkpoint_path, device=device)
    return CameraSpeedPolicy(actor, device), payload


class ScriptedActorAdapter(nn.Module):
    def __init__(self, model: torch.jit.ScriptModule, output_scale: float) -> None:
        super().__init__()
        self.model = model
        self.output_scale = float(output_scale)

    def forward(self, image, lidar):
        return torch.clamp(self.model(image, lidar) * self.output_scale, -1.0, 1.0)


def load_steering_policy(
    kind: str,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    residual_base_checkpoint: str | Path | None = None,
) -> SteeringPolicy:
    device = torch.device(device)
    kind = str(kind).strip().lower()
    if kind in {"bc_scripted", "scripted"}:
        scripted = torch.jit.load(str(checkpoint_path), map_location=device)
        scale = 100.0 / 42.0 if kind == "bc_scripted" else 1.0
        return SteeringPolicy(ScriptedActorAdapter(scripted, scale), device)
    if kind == "bc":
        actor, _ = load_bc_actor(checkpoint_path, device=device)
        return SteeringPolicy(actor, device)
    if kind == "td3_bc":
        actor, _ = load_rl_actor(checkpoint_path, device=device)
        return SteeringPolicy(actor, device)
    if kind != "residual":
        raise ValueError(
            "policy kind must be bc, bc_scripted, td3_bc, scripted, or residual"
        )

    payload = checkpoint_payload(checkpoint_path, device)
    base_path = residual_base_checkpoint or payload.get("base_checkpoint")
    if not base_path:
        raise ValueError("residual policy requires a base checkpoint")
    base_path = Path(base_path).expanduser()
    if not base_path.is_absolute():
        base_path = Path(checkpoint_path).expanduser().resolve().parent / base_path
    base_path = base_path.resolve()
    base_kind = str(payload.get("base_kind", "td3_bc"))
    if base_kind == "bc":
        base_actor, _ = load_bc_actor(base_path, device=device)
    elif base_kind == "td3_bc":
        base_actor, _ = load_rl_actor(base_path, device=device)
    elif base_kind in {"bc_scripted", "scripted"}:
        base_actor = load_steering_policy(
            base_kind, base_path, device=device
        ).actor
    else:
        raise ValueError(f"unsupported residual base kind: {base_kind}")
    residual = ResidualActor(float(payload.get("max_residual_norm", 0.25)))
    residual.load_state_dict(payload["residual_state_dict"], strict=True)
    return SteeringPolicy(base_actor, device, residual)
