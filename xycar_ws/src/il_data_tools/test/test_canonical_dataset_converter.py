import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from il_data_tools.canonical_artifacts import canonical_masks
from il_data_tools.canonical_dataset_converter import convert_session


class CanonicalDatasetConverterTests(unittest.TestCase):
    def test_conversion_preserves_source_session_group_and_scan_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            image_dir = source / "images" / "front"
            scan_dir = source / "scan"
            image_dir.mkdir(parents=True)
            scan_dir.mkdir()
            rows = []
            for index in range(6):
                image = np.full((144, 256, 3), 36, dtype=np.uint8)
                image[:, 45:50] = (255, 255, 255)
                image[:, 205:210] = (255, 255, 255)
                image[20:120, 126:131] = (0, 220, 255)
                image_path = image_dir / f"{index}.png"
                scan_path = scan_dir / f"{index}.npz"
                cv2.imwrite(str(image_path), image)
                np.savez(scan_path, ranges=np.ones(10, dtype=np.float32))
                rows.append(
                    {
                        "timestamp_ns": str(index),
                        "image_timestamp_ns": str(index),
                        "front_image_path": str(image_path.relative_to(source)),
                        "scan_npz_path": str(scan_path.relative_to(source)),
                        "motor_angle": "4.0",
                        "motor_speed": "3.0",
                        "mission_label": "general_drive",
                        "session_id": "original_drive",
                    }
                )
            blank_path = image_dir / "blank.png"
            cv2.imwrite(
                str(blank_path), np.full((144, 256, 3), 36, dtype=np.uint8)
            )
            rows.append(
                {
                    "timestamp_ns": "99",
                    "image_timestamp_ns": "99",
                    "front_image_path": str(blank_path.relative_to(source)),
                    "scan_npz_path": str((scan_dir / "0.npz").relative_to(source)),
                    "motor_angle": "4.0",
                    "motor_speed": "3.0",
                    "mission_label": "general_drive",
                    "session_id": "original_drive",
                }
            )
            with (source / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            converted = convert_session(
                source,
                root / "derived",
                {
                    "white_both_probability": 0.0,
                    "white_one_probability": 0.0,
                    "white_none_probability": 1.0,
                    "yellow_visible_probability": 0.0,
                },
                seed=7,
                variant="v2",
            )
            with (converted / "samples.csv").open(newline="", encoding="utf-8") as handle:
                output_rows = list(csv.DictReader(handle))
            self.assertEqual(len(output_rows), len(rows) - 1)
            metadata = json.loads((converted / "metadata.json").read_text())
            self.assertEqual(metadata["skipped_blank_canonical"], 1)
            self.assertEqual({row["session_id"] for row in output_rows}, {"original_drive"})
            self.assertTrue(all(Path(row["scan_npz_path"]).is_absolute() for row in output_rows))
            for row in output_rows:
                image = cv2.imread(str(converted / row["front_image_path"]))
                white, yellow = canonical_masks(image)
                self.assertFalse(np.any(white))
                self.assertTrue(np.any(yellow))


if __name__ == "__main__":
    unittest.main()
