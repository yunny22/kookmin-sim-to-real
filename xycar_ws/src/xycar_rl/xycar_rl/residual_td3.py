from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from xycar_rl.models import ResidualActor, ResNet18LidarActor, TwinCritic
from xycar_rl.td3_bc import soft_update


@dataclass(frozen=True)
class ResidualTD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.08
    noise_clip: float = 0.15
    policy_delay: int = 2
    actor_lr: float = 1.0e-4
    critic_lr: float = 3.0e-4
    residual_l2: float = 0.2


class ResidualTD3Agent:
    def __init__(
        self,
        base_actor: ResNet18LidarActor,
        *,
        max_residual_norm: float = 0.25,
        device: str | torch.device = "cpu",
        config: ResidualTD3Config = ResidualTD3Config(),
    ) -> None:
        self.device = torch.device(device)
        self.config = config
        self.base_actor = base_actor.to(self.device).eval()
        for parameter in self.base_actor.parameters():
            parameter.requires_grad_(False)
        self.residual_actor = ResidualActor(max_residual_norm).to(self.device)
        self.residual_target = deepcopy(self.residual_actor).to(self.device).eval()
        self.critic = TwinCritic().to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device).eval()
        self.actor_optimizer = torch.optim.Adam(
            self.residual_actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )
        self.update_count = 0

    @torch.no_grad()
    def action(self, observation: dict, noise_std: float = 0.0) -> tuple[float, float, float]:
        image = torch.as_tensor(
            observation["image"], device=self.device
        ).float().unsqueeze(0)
        lidar = torch.as_tensor(
            observation["lidar"], device=self.device
        ).float().unsqueeze(0)
        base = self.base_actor(image, lidar)
        residual = self.residual_actor(image, lidar)
        if noise_std > 0.0:
            residual = residual + torch.randn_like(residual) * noise_std
        final = (base + residual).clamp(-1.0, 1.0)
        return float(final.item()), float(base.item()), float(residual.item())

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.config
        image, lidar = batch["image"], batch["lidar"]
        action, reward = batch["action"], batch["reward"]
        next_image, next_lidar = batch["next_image"], batch["next_lidar"]
        done = batch["done"]
        with torch.no_grad():
            base_next = self.base_actor(next_image, next_lidar)
            residual_next = self.residual_target(next_image, next_lidar)
            noise = (torch.randn_like(action) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip
            )
            next_action = (base_next + residual_next + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(
                next_image, next_lidar, next_action
            )
            target_q = reward + (1.0 - done) * cfg.gamma * torch.minimum(
                target_q1, target_q2
            )

        q1, q2 = self.critic(image, lidar, action)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(
            q2, target_q
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.update_count += 1

        actor_loss_value = float("nan")
        residual_rms = float("nan")
        if self.update_count % cfg.policy_delay == 0:
            with torch.no_grad():
                base = self.base_actor(image, lidar)
            residual = self.residual_actor(image, lidar)
            final_action = (base + residual).clamp(-1.0, 1.0)
            q_value = self.critic.q1(image, lidar, final_action)
            actor_loss = -q_value.mean() + cfg.residual_l2 * residual.square().mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            soft_update(self.residual_target, self.residual_actor, cfg.tau)
            soft_update(self.critic_target, self.critic, cfg.tau)
            actor_loss_value = float(actor_loss.detach().cpu())
            residual_rms = float(residual.square().mean().sqrt().detach().cpu())
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
            "residual_rms": residual_rms,
        }
