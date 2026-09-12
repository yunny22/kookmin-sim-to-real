from __future__ import annotations

from collections import deque
import random

import numpy as np
import torch


class ReplayBuffer:
    """Memory-bounded replay buffer with compressed image and LiDAR storage."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.items = deque(maxlen=max(1, int(capacity)))

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _compress_observation(observation: dict[str, np.ndarray]):
        image = np.clip(observation["image"] * 255.0, 0.0, 255.0).astype(
            np.uint8
        )
        lidar = observation["lidar"].astype(np.float16)
        return image, lidar

    def add(
        self,
        observation: dict[str, np.ndarray],
        action: float,
        reward: float,
        next_observation: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        image, lidar = self._compress_observation(observation)
        next_image, next_lidar = self._compress_observation(next_observation)
        self.items.append(
            (
                image,
                lidar,
                float(action),
                float(reward),
                next_image,
                next_lidar,
                float(done),
            )
        )

    def sample(self, batch_size: int, device: torch.device):
        sampled = random.sample(self.items, min(int(batch_size), len(self.items)))
        image, lidar, action, reward, next_image, next_lidar, done = zip(*sampled)
        return {
            "image": torch.from_numpy(np.stack(image)).to(device).float() / 255.0,
            "lidar": torch.from_numpy(np.stack(lidar)).to(device).float(),
            "action": torch.tensor(action, device=device).float().unsqueeze(1),
            "reward": torch.tensor(reward, device=device).float().unsqueeze(1),
            "next_image": torch.from_numpy(np.stack(next_image)).to(device).float()
            / 255.0,
            "next_lidar": torch.from_numpy(np.stack(next_lidar)).to(device).float(),
            "done": torch.tensor(done, device=device).float().unsqueeze(1),
        }
