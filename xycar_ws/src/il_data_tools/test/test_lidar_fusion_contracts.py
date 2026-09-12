from pathlib import Path
from collections import Counter
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))


class LidarFusionContractTests(unittest.TestCase):
    def test_builder_rejects_missing_or_unsynchronized_scan(self):
        from il_data_tools.dataset_builder_common import BuildConfig, parse_row

        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            (session / "front.jpg").touch()
            (session / "scan.npz").touch()
            config = BuildConfig(
                dataset_roots=[],
                session_dirs=[session],
                output_dir=session / "out",
                include_labels=["general_drive"],
                excluded_labels=[],
                max_steer_deg=100.0,
                min_abs_speed=0.0,
                keep_stopped=True,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=1,
                require_scan=True,
                max_scan_time_offset_ms=50.0,
            )

            def report():
                return {
                    "filtered": Counter(),
                    "invalid_rows": [],
                    "missing_images": [],
                    "missing_scans": [],
                }

            base = {
                "timestamp_ns": "1000000000",
                "front_image_path": "front.jpg",
                "scan_npz_path": "scan.npz",
                "motor_angle": "0",
                "motor_speed": "5",
                "mission_label": "general_drive",
                "session_id": "drive_01",
            }
            missing_timestamp = parse_row(base, session, config, 2, report())
            self.assertIsNone(missing_timestamp)

            unsynced = dict(base, scan_timestamp_ns="1100000000", scan_time_offset_ms="100")
            self.assertIsNone(parse_row(unsynced, session, config, 2, report()))

            synced = dict(base, scan_timestamp_ns="1030000000", scan_time_offset_ms="30")
            parsed = parse_row(synced, session, config, 2, report())
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed.scan_time_offset_ms, 30.0)

    def test_drive_and_cone_builders_require_scan(self):
        from build_drive_dataset import build_parser as drive_parser
        from build_cone_dataset import build_parser as cone_parser

        drive = drive_parser().parse_args(
            ["--dataset-root", "raw", "--output-dir", "out", "--max-steer-deg", "100"]
        )
        cone = cone_parser().parse_args(
            ["--dataset-root", "raw", "--output-dir", "out", "--max-steer-deg", "100"]
        )
        self.assertTrue(drive.require_scan)
        self.assertTrue(cone.require_scan)
        self.assertEqual(drive.max_scan_time_offset_ms, 50.0)

    def test_lidar_model_contract(self):
        try:
            import torch
            from policy_models import create_policy_model, model_uses_lidar
        except ImportError:
            self.skipTest("torch/torchvision is not installed")

        self.assertTrue(model_uses_lidar("resnet18_lidar"))
        model = create_policy_model("resnet18_lidar")
        output = model(torch.zeros(2, 3, 90, 160), torch.zeros(2, 2, 360))
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_recorder_launches_enforce_strict_sync(self):
        for name in ["record_drive_dataset.launch.py", "record_cone_dataset.launch.py"]:
            source = (ROOT / "launch" / name).read_text(encoding="utf-8")
            self.assertIn('declare("require_scan", "true"', source)
            self.assertIn('declare("sync_tolerance_sec", "0.05"', source)


if __name__ == "__main__":
    unittest.main()
