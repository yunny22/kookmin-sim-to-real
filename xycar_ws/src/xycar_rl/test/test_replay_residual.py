import unittest

import numpy as np
import torch

from xycar_rl.replay_buffer import ReplayBuffer


class ReplayBufferTest(unittest.TestCase):
    def test_compresses_and_samples_observations(self):
        observation = {
            "image": np.random.default_rng(1).random((3, 12, 16), dtype=np.float32),
            "lidar": np.ones((2, 20), dtype=np.float32),
            "aux": np.zeros(2, dtype=np.float32),
        }
        replay = ReplayBuffer(capacity=2)
        replay.add(observation, 0.2, 1.0, observation, False)
        replay.add(observation, -0.1, -1.0, observation, True)
        batch = replay.sample(2, torch.device("cpu"))
        self.assertEqual(tuple(batch["image"].shape), (2, 3, 12, 16))
        self.assertEqual(tuple(batch["lidar"].shape), (2, 2, 20))
        self.assertEqual(tuple(batch["action"].shape), (2, 1))
        self.assertTrue(torch.all(batch["image"] >= 0.0))
        self.assertTrue(torch.all(batch["image"] <= 1.0))


if __name__ == "__main__":
    unittest.main()
