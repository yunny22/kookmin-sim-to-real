from pathlib import Path
import csv
import tempfile
import unittest
import json

import numpy as np

from xycar_rl.transition_io import TransitionWriter, TRANSITION_COLUMNS


class TransitionWriterTest(unittest.TestCase):
    def test_camera_only_observation_does_not_write_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            with TransitionWriter(directory) as writer:
                image_path, scan_path = writer.save_observation(10, image, None)
            self.assertTrue((Path(directory) / image_path).is_file())
            self.assertEqual(scan_path, "")
            self.assertEqual(list((Path(directory) / "scan").iterdir()), [])

    def test_writes_deduplicated_observations_and_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            image = np.full((16, 24, 3), 36, dtype=np.uint8)
            scan = {
                "ranges": np.asarray([1.0, 2.0], dtype=np.float32),
                "range_min": np.float32(0.1),
                "range_max": np.float32(12.0),
            }
            with TransitionWriter(directory) as writer:
                first = writer.save_observation(10, image, scan)
                self.assertEqual(first, writer.save_observation(10, image, scan))
                second = writer.save_observation(20, image, scan)
                writer.append(
                    {
                        "episode_id": 0,
                        "step_id": 0,
                        "state_timestamp_ns": 10,
                        "next_timestamp_ns": 20,
                        "state_image_path": first[0],
                        "state_scan_path": first[1],
                        "next_image_path": second[0],
                        "next_scan_path": second[1],
                        "action_norm": 0.1,
                        "reward": 1.0,
                        "terminated": 0,
                        "truncated": 0,
                    }
                )
            with (Path(directory) / "transitions.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(list(rows[0]), TRANSITION_COLUMNS)
            self.assertEqual(len(list((Path(directory) / "images").iterdir())), 2)
            metadata = json.loads(
                (Path(directory) / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["schema_version"], 5)
            self.assertIn("expert_action_norm", TRANSITION_COLUMNS)
            self.assertIn("expert_speed_command", TRANSITION_COLUMNS)
            with self.assertRaises(FileExistsError):
                TransitionWriter(directory)


if __name__ == "__main__":
    unittest.main()
