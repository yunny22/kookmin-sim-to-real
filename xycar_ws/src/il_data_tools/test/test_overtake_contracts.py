import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from il_data_tools.dataset_builder_common import Sample
from build_overtake_dataset import assign_overtake_phase, validate_speed_consistency


def sample(timestamp, label, speed=8.0, session="overtake_01"):
    return Sample(
        image_path="dummy.jpg",
        steer_norm=0.0,
        angle_deg=0.0,
        speed=speed,
        mission_label=label,
        session_id=session,
        timestamp_ns=timestamp,
    )


class OvertakeContractTests(unittest.TestCase):
    def test_explicit_single_maneuver_phase(self):
        rows = [
            sample(0, "overtake_start"),
            sample(1_000_000_000, "vehicle_overtake"),
            sample(2_000_000_000, "overtake_end"),
        ]
        report = assign_overtake_phase(rows, 4.0, require_explicit_boundaries=True)
        self.assertTrue(report["overtake_01"]["explicit_boundaries_valid"])
        self.assertFalse(report["overtake_01"]["used_fallback"])
        self.assertEqual([row.phase for row in rows], [0.0, 0.5, 1.0])

    def test_multiple_maneuvers_rejected_in_strict_mode(self):
        rows = [
            sample(0, "overtake_start"),
            sample(1, "overtake_end"),
            sample(2, "overtake_start"),
            sample(3, "overtake_end"),
        ]
        with self.assertRaises(ValueError):
            assign_overtake_phase(rows, 4.0, require_explicit_boundaries=True)

    def test_speed_std_contract(self):
        rows = [sample(0, "vehicle_overtake", 8.0), sample(1, "vehicle_overtake", 8.1)]
        report = validate_speed_consistency(rows, 7.0, 9.0, 0.2)
        self.assertLess(report["overtake_01"]["std"], 0.2)
        with self.assertRaises(ValueError):
            validate_speed_consistency(rows, 7.0, 9.0, 0.01)


if __name__ == "__main__":
    unittest.main()
