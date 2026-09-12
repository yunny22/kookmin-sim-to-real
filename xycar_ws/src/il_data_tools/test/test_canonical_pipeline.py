import csv
import json
from pathlib import Path
import tempfile
import unittest

from il_data_tools.canonical_pipeline import validate_sessions


class CanonicalPipelineTests(unittest.TestCase):
    def test_session_validation_counts_recovery_and_checks_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "sim_canonical_mixed_s1_01"
            image_dir = session / "front"
            scan_dir = session / "scan"
            image_dir.mkdir(parents=True)
            scan_dir.mkdir()
            rows = []
            for index in range(10):
                image = image_dir / f"{index}.png"
                scan = scan_dir / f"{index}.npz"
                image.touch()
                scan.touch()
                rows.append(
                    {
                        "front_image_path": str(image.relative_to(session)),
                        "scan_npz_path": str(scan.relative_to(session)),
                        "mission_label": "recovery" if index < 2 else "general_drive",
                        "motor_speed": "3.0",
                    }
                )
            with (session / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            (session / "metadata.json").write_text(
                json.dumps(
                    {
                        "parameters": {"bad_data_preroll_sec": 10.0},
                        "discarded_bad_data_preroll": 4,
                    }
                ),
                encoding="utf-8",
            )

            report = validate_sessions([session], 10)
            self.assertEqual(report["total_rows"], 10)
            self.assertEqual(report["label_counts"]["recovery"], 2)
            self.assertAlmostEqual(report["recovery_ratio"], 0.2)
            self.assertEqual(report["discarded_bad_data_preroll"], 4)

    def test_session_validation_rejects_too_little_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            image = session / "frame.png"
            scan = session / "scan.npz"
            image.touch()
            scan.touch()
            with (session / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "front_image_path",
                        "scan_npz_path",
                        "mission_label",
                        "motor_speed",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "front_image_path": image.name,
                        "scan_npz_path": scan.name,
                        "mission_label": "general_drive",
                        "motor_speed": "3.0",
                    }
                )
            (session / "metadata.json").write_text(
                json.dumps({"parameters": {"bad_data_preroll_sec": 10.0}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "recovery data ratio"):
                validate_sessions([session], 1)

    def test_session_validation_requires_augmented_topic_and_event_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "sim_canonical_artifacts_s1_01"
            image_dir = session / "images" / "front"
            scan_dir = session / "scan"
            image_dir.mkdir(parents=True)
            scan_dir.mkdir()
            event_log = root / "events.jsonl"
            event_log.write_text('{"kind":"center_jump"}\n', encoding="utf-8")
            rows = []
            for index in range(10):
                image = image_dir / f"{index}.png"
                scan = scan_dir / f"{index}.npz"
                image.touch()
                scan.touch()
                rows.append(
                    {
                        "front_image_path": str(image.relative_to(session)),
                        "scan_npz_path": str(scan.relative_to(session)),
                        "mission_label": "recovery" if index < 2 else "general_drive",
                        "motor_speed": "3.0",
                    }
                )
            with (session / "samples.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            (session / "metadata.json").write_text(
                json.dumps(
                    {
                        "parameters": {
                            "bad_data_preroll_sec": 10.0,
                            "camera_front_topic": (
                                "/perception/canonical_road_image_augmented"
                            ),
                        },
                        "run_manifest": {
                            "canonical_artifacts": {
                                "enabled": True,
                                "mode": "real_visibility",
                                "event_log": str(event_log),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = validate_sessions(
                [session], 10, require_canonical_artifacts=True
            )
            self.assertEqual(report["canonical_artifact_events"], 1)


if __name__ == "__main__":
    unittest.main()
