import unittest

import numpy as np

from il_data_tools.canonical_artifacts import (
    CanonicalArtifactAugmenter,
    CanonicalVisibilityAugmenter,
    apply_center_jump,
    apply_line_dropout,
    apply_visibility_profile,
    apply_white_bend,
    canonical_masks,
)


def canonical_test_image() -> np.ndarray:
    image = np.full((144, 256, 3), 36, dtype=np.uint8)
    image[10:140, 48:53] = (255, 255, 255)
    image[10:140, 203:208] = (255, 255, 255)
    image[20:125, 126:131] = (0, 220, 255)
    return image


class CanonicalArtifactTests(unittest.TestCase):
    def test_center_jump_moves_only_yellow_line(self):
        image = canonical_test_image()
        output = apply_center_jump(image, 12.0)
        white_before, yellow_before = canonical_masks(image)
        white_after, yellow_after = canonical_masks(output)

        self.assertTrue(np.array_equal(white_before, white_after))
        self.assertFalse(np.any(yellow_after[:, 126:131]))
        self.assertEqual(
            int(np.count_nonzero(yellow_before)),
            int(np.count_nonzero(yellow_after)),
        )
        self.assertTrue(np.any(yellow_after[:, 138:143]))

    def test_white_bend_preserves_other_boundary_and_center(self):
        image = canonical_test_image()
        output = apply_white_bend(image, "left", 40.0, exponent=2.0)
        white_after, yellow_after = canonical_masks(output)

        self.assertTrue(np.all(output[20:125, 126:131] == (0, 220, 255)))
        self.assertTrue(np.all(white_after[10:140, 203:208]))
        self.assertFalse(np.any(white_after[130:140, 48:53]))
        self.assertTrue(np.any(white_after[130:140, 82:94]))
        self.assertTrue(np.any(yellow_after))

    def test_partial_dropout_does_not_remove_other_classes(self):
        image = canonical_test_image()
        output = apply_line_dropout(
            image,
            "white",
            side="right",
            start_row_ratio=0.5,
            end_row_ratio=1.0,
        )
        white_after, yellow_after = canonical_masks(output)

        self.assertTrue(np.any(white_after[:60, 203:208]))
        self.assertFalse(np.any(white_after[80:, 203:208]))
        self.assertTrue(np.all(white_after[10:140, 48:53]))
        self.assertTrue(np.any(yellow_after))

    def test_explicit_event_is_temporally_bounded(self):
        image = canonical_test_image()
        augmenter = CanonicalArtifactAugmenter(
            seed=7,
            event_start_probability=0.0,
        )
        augmenter.start_event("center_jump", 2, offset_px=10.0)

        first, first_event, first_started = augmenter.process(image)
        second, second_event, second_started = augmenter.process(image)
        third, third_event, third_started = augmenter.process(image)

        self.assertIsNotNone(first_event)
        self.assertIsNotNone(second_event)
        self.assertIsNone(third_event)
        self.assertFalse(first_started)
        self.assertFalse(second_started)
        self.assertFalse(third_started)
        self.assertFalse(np.array_equal(first, image))
        self.assertFalse(np.array_equal(second, image))
        self.assertTrue(np.array_equal(third, image))

    def test_real_visibility_never_moves_the_retained_components(self):
        image = canonical_test_image()
        output, forced = apply_visibility_profile(
            image,
            white_state="one",
            keep_white_side="left",
            yellow_visible=True,
        )
        white, yellow = canonical_masks(output)

        self.assertFalse(forced)
        self.assertTrue(np.all(white[10:140, 48:53]))
        self.assertFalse(np.any(white[:, 203:208]))
        self.assertTrue(np.all(yellow[20:125, 126:131]))

    def test_real_visibility_prevents_synthetic_blank_inputs(self):
        image = canonical_test_image()
        output, forced = apply_visibility_profile(
            image,
            white_state="none",
            keep_white_side="right",
            yellow_visible=False,
            prevent_blank=True,
        )
        white, yellow = canonical_masks(output)

        self.assertTrue(forced)
        self.assertFalse(np.any(white))
        self.assertTrue(np.any(yellow))

    def test_visibility_state_is_temporally_held(self):
        image = canonical_test_image()
        augmenter = CanonicalVisibilityAugmenter(
            seed=9,
            white_duration_frames=(4, 4),
            yellow_duration_frames=(4, 4),
        )
        states = []
        for _ in range(4):
            _, event, _ = augmenter.process(image)
            states.append(event.parameters)
        self.assertTrue(all(state == states[0] for state in states))


if __name__ == "__main__":
    unittest.main()
