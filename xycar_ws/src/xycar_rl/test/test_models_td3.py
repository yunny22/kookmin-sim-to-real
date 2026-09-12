from pathlib import Path
import unittest

import torch

from xycar_rl.models import ResidualActor, load_bc_actor
from xycar_rl.policy_loader import load_steering_policy
from xycar_rl.td3_bc import TD3BCAgent, TD3BCConfig


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BC_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "il_policies"
    / "drive_canonical_real_reference_200k_20260715"
    / "drive_resnet18_lidar_best.pth"
)
BC_SCRIPTED = (
    PROJECT_ROOT
    / "xycar_ws"
    / "src"
    / "il_data_tools"
    / "models"
    / "drive_canonical_policy_scripted.pt"
)


class ModelTD3Test(unittest.TestCase):
    def test_residual_actor_starts_as_exact_zero_correction(self):
        actor = ResidualActor(max_residual_norm=0.25).eval()
        image = torch.randn(2, 3, 90, 160)
        lidar = torch.randn(2, 2, 360)
        with torch.no_grad():
            output = actor(image, lidar)
        torch.testing.assert_close(output, torch.zeros_like(output))

    @classmethod
    def setUpClass(cls):
        cls.actor, cls.payload = load_bc_actor(BC_CHECKPOINT)

    def test_bc_checkpoint_loads_strictly_and_scales_to_physical_action(self):
        self.assertEqual(self.payload["model_type"], "resnet18_lidar")
        self.assertAlmostEqual(self.actor.output_scale, 100.0 / 42.0)
        self.actor.eval()
        with torch.no_grad():
            output = self.actor(
                torch.zeros(1, 3, 90, 160),
                torch.zeros(1, 2, 360),
            )
        self.assertEqual(tuple(output.shape), (1, 1))
        self.assertLessEqual(float(output.abs().max()), 1.0)

    def test_td3_bc_update_produces_finite_losses(self):
        actor, _ = load_bc_actor(BC_CHECKPOINT)
        agent = TD3BCAgent(
            actor,
            config=TD3BCConfig(policy_delay=1),
        )
        batch_size = 2
        batch = {
            "image": torch.rand(batch_size, 3, 32, 32),
            "lidar": torch.rand(batch_size, 2, 64),
            "action": torch.zeros(batch_size, 1),
            "reward": torch.ones(batch_size, 1) * 0.1,
            "next_image": torch.rand(batch_size, 3, 32, 32),
            "next_lidar": torch.rand(batch_size, 2, 64),
            "done": torch.zeros(batch_size, 1),
        }
        metrics = agent.update(batch)
        for value in metrics.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))

    def test_packaged_scripted_bc_matches_checkpoint_action_units(self):
        policy = load_steering_policy("bc_scripted", BC_SCRIPTED)
        rng = torch.Generator().manual_seed(3)
        image = torch.rand(1, 3, 90, 160, generator=rng)
        lidar = torch.rand(1, 2, 360, generator=rng)
        self.actor.eval()
        with torch.no_grad():
            expected = self.actor(image, lidar)
            actual = policy.actor(image, lidar)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
