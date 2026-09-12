import hashlib
from pathlib import Path
import unittest

import yaml


class RealCameraCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.package_root = Path(__file__).resolve().parents[1]
        self.calibration_path = (
            self.package_root
            / "config"
            / "wide_camera_fisheye_1280x1024_20260708.yaml"
        )

    def test_measured_fisheye_profile_is_immutable(self):
        digest = hashlib.sha256(self.calibration_path.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "0bc9b224e8105b7c9da2f280097d5f6ea75395482f0c504d288745cbd5a25dfe",
        )

    def test_measured_intrinsics_and_distortion(self):
        with self.calibration_path.open(encoding="utf-8") as stream:
            calibration = yaml.safe_load(stream)

        self.assertEqual(calibration["image_width"], 1280)
        self.assertEqual(calibration["image_height"], 1024)
        self.assertEqual(calibration["distortion_model"], "equidistant")
        self.assertEqual(
            calibration["camera_matrix"]["data"],
            [
                725.9760698863265,
                0.0,
                685.4283617571,
                0.0,
                725.9655676685525,
                497.80435337486205,
                0.0,
                0.0,
                1.0,
            ],
        )
        self.assertEqual(
            calibration["distortion_coefficients"]["data"],
            [
                -0.023278262724104614,
                0.0040440926898205444,
                -0.006251740223863257,
                0.00246793717714305,
            ],
        )
        self.assertAlmostEqual(
            calibration["calibration_info"]["rms_reprojection_error_px"],
            0.45765194028955725,
        )


if __name__ == "__main__":
    unittest.main()
