import unittest

import numpy as np

from xycar_rl.canonical_preview import CanonicalPreviewSteering


def yellow_image(near_x: int, middle_x: int, far_x: int) -> np.ndarray:
    image = np.zeros((3, 90, 160), dtype=np.float32)
    for low, high, x_value in ((62, 90, near_x), (38, 62, middle_x), (13, 38, far_x)):
        image[0, low:high, x_value - 1 : x_value + 2] = 1.0
        image[1, low:high, x_value - 1 : x_value + 2] = 1.0
    return image


class CanonicalPreviewSteeringTest(unittest.TestCase):
    def test_centered_straight_commands_zero(self):
        controller = CanonicalPreviewSteering()
        estimate = controller.update(yellow_image(80, 80, 80))
        self.assertAlmostEqual(estimate.steering_norm, 0.0, places=2)
        self.assertAlmostEqual(estimate.curve_hint, 0.0)

    def test_far_line_turns_before_near_line_moves(self):
        controller = CanonicalPreviewSteering()
        estimate = controller.update(yellow_image(80, 90, 112))
        self.assertGreater(estimate.steering_norm, 0.15)
        self.assertGreater(estimate.curve_hint, 0.4)

    def test_dash_gap_holds_last_preview_briefly(self):
        controller = CanonicalPreviewSteering(hold_frames=2)
        first = controller.update(yellow_image(80, 90, 112))
        blank = np.zeros((3, 90, 160), dtype=np.float32)
        held = controller.update(blank)
        self.assertGreater(held.steering_norm, 0.0)
        self.assertLess(held.steering_norm, first.steering_norm)
        controller.update(blank)
        expired = controller.update(blank)
        self.assertEqual(expired.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
