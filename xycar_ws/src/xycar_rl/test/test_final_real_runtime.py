import unittest
from pathlib import Path

from xycar_rl.camera_speed_models import load_camera_speed_actor
from xycar_rl.policy_runtime_node import apply_optional_speed_cap


class FinalRealRuntimeTest(unittest.TestCase):
    def test_zero_cap_preserves_learned_speed(self):
        self.assertEqual(apply_optional_speed_cap(19.5, 0.0), 19.5)
        self.assertEqual(apply_optional_speed_cap(19.5, -1.0), 19.5)
        self.assertEqual(apply_optional_speed_cap(19.5, 17.0), 17.0)

    def test_packaged_final_checkpoint_contract(self):
        checkpoint = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "straight_speed25_recovery_v3_20260805"
            / "camera_speed_td3_bc_best.pth"
        )
        actor, payload = load_camera_speed_actor(checkpoint, device="cpu")

        self.assertEqual(payload["epoch"], 29)
        self.assertEqual(payload["temporal_frames"], 2)
        self.assertEqual(payload["min_speed_command"], 4.0)
        self.assertEqual(payload["max_speed_command"], 25.0)
        self.assertEqual(actor.temporal_frames, 2)


if __name__ == "__main__":
    unittest.main()
