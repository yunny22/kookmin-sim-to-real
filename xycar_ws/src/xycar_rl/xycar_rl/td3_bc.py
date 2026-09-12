from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from xycar_rl.models import ResNet18LidarActor, TwinCritic
from xycar_rl.camera_speed_models import CameraSpeedActor, CameraSpeedTwinCritic


@dataclass(frozen=True)
class TD3BCConfig:
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    bc_alpha: float = 2.5
    actor_lr: float = 1.0e-5
    critic_lr: float = 3.0e-4
    steering_bc_weight: float = 1.0
    speed_bc_weight: float = 1.0


def camera_speed_bc_loss(
    predicted_action: torch.Tensor,
    action: torch.Tensor,
    *,
    steering_weight: float = 1.0,
    speed_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    total_weight = float(steering_weight) + float(speed_weight)
    if steering_weight < 0.0 or speed_weight < 0.0 or total_weight <= 0.0:
        raise ValueError("camera-speed BC weights must be nonnegative and nonzero")
    per_sample = (
        float(steering_weight)
        * (predicted_action[:, 0] - action[:, 0]).square()
        + float(speed_weight)
        * (predicted_action[:, 1] - action[:, 1]).square()
    ) / total_weight
    if sample_weight is None:
        return per_sample.mean()
    weights = sample_weight.reshape(-1).to(per_sample)
    weight_sum = weights.sum()
    if float(weight_sum.detach().cpu()) <= 0.0:
        return per_sample.sum() * 0.0
    return (per_sample * weights).sum() / weight_sum


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)


class TD3BCAgent:
    def __init__(
        self,
        actor: ResNet18LidarActor,
        *,
        device: str | torch.device = "cpu",
        config: TD3BCConfig = TD3BCConfig(),
    ) -> None:
        self.device = torch.device(device)
        self.config = config
        self.actor = actor.to(self.device)
        self.actor_target = deepcopy(actor).to(self.device).eval()
        self.critic = TwinCritic().to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device).eval()
        actor_parameters = [
            parameter
            for parameter in self.actor.parameters()
            if parameter.requires_grad
        ]
        if not actor_parameters:
            raise ValueError("actor has no trainable parameters")
        self.actor_optimizer = torch.optim.Adam(
            actor_parameters, lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )
        self.update_count = 0

    def _device_batch(self, batch: dict[str, torch.Tensor]):
        return {key: value.to(self.device) for key, value in batch.items()}

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        batch = self._device_batch(batch)
        image = batch["image"]
        lidar = batch["lidar"]
        action = batch["action"]
        reward = batch["reward"]
        next_image = batch["next_image"]
        next_lidar = batch["next_lidar"]
        done = batch["done"]
        cfg = self.config

        with torch.no_grad():
            noise = torch.randn_like(action) * cfg.policy_noise
            noise = noise.clamp(-cfg.noise_clip, cfg.noise_clip)
            next_action = (
                self.actor_target(next_image, next_lidar) + noise
            ).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(
                next_image, next_lidar, next_action
            )
            target_q = reward + (1.0 - done) * cfg.gamma * torch.minimum(
                target_q1, target_q2
            )

        current_q1, current_q2 = self.critic(image, lidar, action)
        critic_loss = nn.functional.mse_loss(
            current_q1, target_q
        ) + nn.functional.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_count += 1
        actor_loss_value = float("nan")
        bc_loss_value = float("nan")
        q_scale_value = float("nan")
        if self.update_count % cfg.policy_delay == 0:
            predicted_action = self.actor(image, lidar)
            q_value = self.critic.q1(image, lidar, predicted_action)
            q_scale = cfg.bc_alpha / q_value.abs().mean().detach().clamp_min(1.0e-6)
            bc_loss = nn.functional.mse_loss(predicted_action, action)
            actor_loss = -q_scale * q_value.mean() + bc_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            soft_update(self.actor_target, self.actor, cfg.tau)
            soft_update(self.critic_target, self.critic, cfg.tau)
            actor_loss_value = float(actor_loss.detach().cpu())
            bc_loss_value = float(bc_loss.detach().cpu())
            q_scale_value = float(q_scale.detach().cpu())

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
            "bc_loss": bc_loss_value,
            "q_scale": q_scale_value,
        }


class CameraSpeedTD3BCAgent:
    """TD3+BC agent whose policy and critic consume camera images only."""

    def __init__(
        self,
        actor: nn.Module,
        *,
        device: str | torch.device = "cpu",
        config: TD3BCConfig = TD3BCConfig(),
    ) -> None:
        self.device = torch.device(device)
        self.config = config
        self.actor = actor.to(self.device)
        self.actor_target = deepcopy(actor).to(self.device).eval()
        self.temporal_frames = int(getattr(actor, "temporal_frames", 1))
        self.critic = CameraSpeedTwinCritic(
            temporal_frames=self.temporal_frames
        ).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device).eval()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )
        self.update_count = 0

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        batch = {key: value.to(self.device) for key, value in batch.items()}
        image = batch["image"]
        action = batch["action"]
        bc_action = batch.get("bc_action", action)
        bc_weight = batch.get("bc_weight")
        reward = batch["reward"]
        next_image = batch["next_image"]
        done = batch["done"]
        cfg = self.config

        with torch.no_grad():
            noise = (torch.randn_like(action) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip
            )
            next_action = (self.actor_target(next_image) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_image, next_action)
            target_q = reward + (1.0 - done) * cfg.gamma * torch.minimum(
                target_q1, target_q2
            )

        current_q1, current_q2 = self.critic(image, action)
        critic_loss = nn.functional.mse_loss(
            current_q1, target_q
        ) + nn.functional.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_count += 1
        actor_loss_value = float("nan")
        bc_loss_value = float("nan")
        q_scale_value = float("nan")
        if self.update_count % cfg.policy_delay == 0:
            predicted_action = self.actor(image)
            q_value = self.critic.q1(image, predicted_action)
            q_scale = cfg.bc_alpha / q_value.abs().mean().detach().clamp_min(1.0e-6)
            bc_loss = camera_speed_bc_loss(
                predicted_action,
                bc_action,
                steering_weight=cfg.steering_bc_weight,
                speed_weight=cfg.speed_bc_weight,
                sample_weight=bc_weight,
            )
            actor_loss = -q_scale * q_value.mean() + bc_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            soft_update(self.actor_target, self.actor, cfg.tau)
            soft_update(self.critic_target, self.critic, cfg.tau)
            actor_loss_value = float(actor_loss.detach().cpu())
            bc_loss_value = float(bc_loss.detach().cpu())
            q_scale_value = float(q_scale.detach().cpu())

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
            "bc_loss": bc_loss_value,
            "q_scale": q_scale_value,
        }
