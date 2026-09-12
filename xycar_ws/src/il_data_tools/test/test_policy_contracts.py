import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PACKAGE_ROOT))


class PolicyContractTests(unittest.TestCase):
    def test_pilotnet_preserves_spatial_features(self):
        try:
            import torch
            from policy_models import PilotNetEncoder
        except ImportError:
            self.skipTest("torch is not installed")
        encoder = PilotNetEncoder()
        output = encoder(torch.zeros(2, 3, 90, 160))
        self.assertEqual(tuple(output.shape), (2, 64 * 2 * 4))

    def test_augmentation_does_not_shift_image(self):
        source = (SCRIPTS / "policy_dataset.py").read_text(encoding="utf-8")
        self.assertNotIn("random_shift_crop", source)
        self.assertIn("steer_norm = -steer_norm", source)

    def test_phase_benchmark_uses_batch_column_shape(self):
        source = (SCRIPTS / "benchmark_policy_model.py").read_text(encoding="utf-8")
        self.assertIn("torch.zeros(1, 1", source)

    def test_reference_camera_crop_preserves_16_by_9(self):
        try:
            import numpy as np
            from image_preprocessing import crop_to_target_aspect
        except ImportError:
            self.skipTest("opencv/numpy are not installed")
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = crop_to_target_aspect(image, 160, 90)
        self.assertEqual(tuple(roi.shape), (360, 640, 3))
        # Encode row index to verify the documented y=80:440 crop.
        image[:, :, 0] = np.arange(480, dtype=np.uint16)[:, None] % 256
        roi = crop_to_target_aspect(image, 160, 90)
        self.assertEqual(int(roi[0, 0, 0]), 80)
        self.assertEqual(int(roi[-1, 0, 0]), 439 % 256)

    def test_model_input_debug_conversion_restores_bgr_pixels(self):
        try:
            import numpy as np
            from il_data_tools.runtime_preprocessing import model_input_to_bgr
        except ImportError:
            self.skipTest("numpy is not installed")
        image = np.zeros((3, 2, 3), dtype=np.float32)
        image[0, :, :] = 1.0
        bgr = model_input_to_bgr(image)
        self.assertEqual(tuple(bgr.shape), (2, 3, 3))
        self.assertEqual(bgr[0, 0].tolist(), [0, 0, 255])

        with self.assertRaises(ValueError):
            model_input_to_bgr(np.zeros((2, 2, 3), dtype=np.float32))

    def test_training_wrapper_forwards_initial_checkpoint(self):
        from il_data_tools.train_from_raw_dataset import (
            PROFILE_CONFIG,
            build_parser,
            make_train_command,
        )

        args = build_parser().parse_args(
            [
                "--profile",
                "drive",
                "--init-checkpoint",
                "~/models/sim_drive_best.pth",
            ]
        )
        command = make_train_command(
            args,
            PROFILE_CONFIG["drive"],
            "resnet18_lidar",
            Path("/tmp/processed"),
            Path("/tmp/model"),
        )
        option_index = command.index("--init-checkpoint")
        self.assertEqual(
            command[option_index + 1],
            str(Path("~/models/sim_drive_best.pth").expanduser().resolve()),
        )

    def test_canonical_boundary_dropout_hides_only_selected_white_side(self):
        try:
            import numpy as np
            from policy_dataset import random_canonical_boundary_dropout
        except ImportError:
            self.skipTest("training image dependencies are not installed")

        image = np.full((144, 256, 3), 36, dtype=np.uint8)
        image[:, 48:53] = (255, 255, 255)
        image[:, 203:208] = (255, 255, 255)
        image[20:120, 126:131] = (0, 220, 255)
        with patch("policy_dataset.random.random", return_value=0.0), patch(
            "policy_dataset.random.choice", return_value="left"
        ):
            output = random_canonical_boundary_dropout(image, 0.30)

        self.assertTrue(np.all(output[:, 48:53] == 36))
        self.assertTrue(np.all(output[:, 203:208] == 255))
        self.assertTrue(np.all(output[20:120, 126:131] == (0, 220, 255)))

    def test_training_wrapper_forwards_canonical_lane_dropout(self):
        from il_data_tools.train_from_raw_dataset import (
            PROFILE_CONFIG,
            build_parser,
            make_train_command,
        )

        args = build_parser().parse_args(
            [
                "--profile",
                "drive",
                "--canonical-input",
                "--lane-dropout-probability",
                "0.4",
            ]
        )
        command = make_train_command(
            args,
            PROFILE_CONFIG["drive"],
            "resnet18_lidar",
            Path("/tmp/processed"),
            Path("/tmp/model"),
        )
        self.assertIn("--canonical-input", command)
        option_index = command.index("--lane-dropout-probability")
        self.assertEqual(command[option_index + 1], "0.4")

    def test_real_launch_exposes_steering_calibration(self):
        source = (PACKAGE_ROOT / "launch" / "real_policy_inference.launch.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"steering_output_sign"', source)
        self.assertIn('"max_steer_scale"', source)
        self.assertIn('"steering_temporal_alpha"', source)
        self.assertIn('"sync_tolerance_sec"', source)
        self.assertIn('"sensor_timeout_sec"', source)

    def test_real_canonical_launch_exposes_camera_transport(self):
        source = (
            PACKAGE_ROOT / "launch" / "real_canonical_policy_drive.launch.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"enable_rectify"', source)
        self.assertIn('"use_compressed_image"', source)
        self.assertIn('"sync_tolerance_sec"', source)
        self.assertIn('"perception_launch_file"', source)
        self.assertIn(
            'default_value="real_canonical_perception.launch.py"', source
        )
        self.assertIn("competition and temporary tracks", source)

    def test_policy_republishes_stop_while_waiting_for_sensors(self):
        source = (
            PACKAGE_ROOT / "il_data_tools" / "policy_inference_node.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self.last_stop_wall", source)
        self.assertIn("now_wall - self.last_stop_wall >= 0.5", source)

    def test_offline_eval_supports_canonical_preprocessing(self):
        source = (SCRIPTS / "eval_policy.py").read_text(encoding="utf-8")
        self.assertIn('"--canonical-input"', source)
        self.assertIn("canonical_input=args.canonical_input", source)


if __name__ == "__main__":
    unittest.main()
