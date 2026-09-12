import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import yaml

from il_data_tools.domain_randomization import generate_randomized_assets


ROOT = Path(__file__).resolve().parents[4]
SOURCE_WORLD = ROOT / "worlds" / "kookmin_xycar_track_final.sdf"
SOURCE_BRIDGE = (
    ROOT / "xycar_ws" / "src" / "xycar_gazebo_bridge" / "config" / "xycar_gazebo_bridge.yaml"
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_poses(path: Path, prefix: str):
    world = ET.parse(path).getroot().find("world")
    return {
        model.get("name"): model.findtext("pose")
        for model in world.findall("model")
        if model.get("name", "").startswith(prefix)
    }


class DomainRandomizationTests(unittest.TestCase):
    def generate(self, directory: Path, seed=2026, preset="mixed"):
        return generate_randomized_assets(
            str(SOURCE_WORLD),
            str(directory / "world.sdf"),
            str(SOURCE_BRIDGE),
            str(directory / "bridge.yaml"),
            str(directory / "manifest.json"),
            seed,
            preset,
        )

    def test_same_seed_is_reproducible_and_source_is_unchanged(self):
        source_hash = file_hash(SOURCE_WORLD)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            manifest_a = self.generate(Path(first))
            manifest_b = self.generate(Path(second))
            self.assertEqual(
                Path(first, "world.sdf").read_bytes(),
                Path(second, "world.sdf").read_bytes(),
            )
            self.assertEqual(
                Path(first, "bridge.yaml").read_bytes(),
                Path(second, "bridge.yaml").read_bytes(),
            )
            for key in ("visual", "sensors", "dynamics"):
                self.assertEqual(manifest_a[key], manifest_b[key])
        self.assertEqual(file_hash(SOURCE_WORLD), source_hash)

    def test_randomization_preserves_all_lane_model_poses(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "world.sdf")
            self.generate(Path(directory), seed=77)
            self.assertEqual(
                model_poses(SOURCE_WORLD, "white_line_cad_"),
                model_poses(output, "white_line_cad_"),
            )
            self.assertEqual(
                model_poses(SOURCE_WORLD, "yellow_centerline_dash_"),
                model_poses(output, "yellow_centerline_dash_"),
            )

    def test_mixed_manifest_and_bridge_values_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.generate(root, seed=99)
            saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            bridge = yaml.safe_load((root / "bridge.yaml").read_text(encoding="utf-8"))
            self.assertEqual(saved["seed"], 99)
            self.assertEqual(saved["preset"], "mixed")
            self.assertLessEqual(abs(manifest["sensors"]["camera_pose_offsets"]["z_m"]), 0.010)
            self.assertLessEqual(manifest["sensors"]["lidar_noise_stddev_m"], 0.025)
            self.assertGreaterEqual(manifest["dynamics"]["speed_gain_scale"], 0.955)
            self.assertLessEqual(manifest["dynamics"]["speed_gain_scale"], 1.045)
            params = bridge["xycar_motor_bridge"]["ros__parameters"]
            self.assertAlmostEqual(
                params["speed_gain"],
                manifest["dynamics"]["speed_gain_mps_per_cmd"],
            )

    def test_unknown_preset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.generate(Path(directory), preset="impossible")


if __name__ == "__main__":
    unittest.main()
