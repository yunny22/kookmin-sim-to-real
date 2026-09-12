import unittest

import cv2
import numpy as np

from xycar_perception.canonical_road import (
    make_canonical_road_image_from_masks,
)
from xycar_perception.yolo_lane_segmenter import (
    YoloLaneSegmenter,
    merge_lane_instance_masks,
)


class YoloCanonicalTest(unittest.TestCase):
    def test_segmenter_forwards_lightweight_mask_mode(self):
        class FakeModel:
            def __init__(self):
                self.kwargs = None

            def predict(self, **kwargs):
                self.kwargs = kwargs
                return [object()]

        segmenter = YoloLaneSegmenter.__new__(YoloLaneSegmenter)
        segmenter.model = FakeModel()
        segmenter.image_size = 256
        segmenter.confidence = 0.25
        segmenter.iou = 0.5
        segmenter.max_detections = 30
        segmenter.device = "cpu"
        segmenter.retina_masks = False

        result = segmenter._predict(np.zeros((20, 30, 3), dtype=np.uint8))

        self.assertIsNotNone(result)
        self.assertFalse(segmenter.model.kwargs["retina_masks"])
        self.assertEqual(segmenter.model.kwargs["imgsz"], 256)

    def test_instance_masks_merge_by_lane_class(self):
        masks = np.zeros((3, 20, 30), dtype=np.float32)
        masks[0, 2:18, 3:6] = 1.0
        masks[1, 4:16, 14:17] = 1.0
        masks[2, 2:18, 24:27] = 1.0

        white, yellow = merge_lane_instance_masks(
            masks,
            np.array([0, 1, 0], dtype=np.float32),
            (40, 60),
        )

        self.assertEqual(white.shape, (40, 60))
        self.assertGreater(int(np.count_nonzero(white)), 0)
        self.assertGreater(int(np.count_nonzero(yellow)), 0)
        self.assertEqual(int(np.count_nonzero(white & yellow)), 0)

    def test_binary_masks_produce_fixed_canonical_contract(self):
        white = np.zeros((220, 640), dtype=np.uint8)
        yellow = np.zeros_like(white)
        cv2.line(white, (130, 0), (180, 219), 255, 8)
        cv2.line(white, (510, 0), (460, 219), 255, 8)
        for row in range(20, 210, 45):
            cv2.line(yellow, (320, row), (320, row + 20), 255, 7)

        canonical, canonical_white, canonical_yellow = (
            make_canonical_road_image_from_masks(
                white,
                yellow,
                lateral_m_per_px=1.4 / 640.0,
                forward_m_per_px=1.5 / 220.0,
                lateral_range_m=1.4,
                forward_range_m=1.5,
                output_width=256,
                output_height=144,
                line_width_px=5,
                bottom_ignore_m=0.0,
            )
        )

        colors = {tuple(pixel) for pixel in canonical.reshape(-1, 3)}
        self.assertTrue(
            colors.issubset(
                {(36, 36, 36), (255, 255, 255), (0, 220, 255)}
            )
        )
        self.assertGreater(int(np.count_nonzero(canonical_white)), 0)
        self.assertGreater(int(np.count_nonzero(canonical_yellow)), 0)
        self.assertEqual(
            int(np.count_nonzero(canonical_white & canonical_yellow)), 0
        )

    def test_yolo_yellow_can_bypass_color_geometry_filter(self):
        white = np.zeros((220, 640), dtype=np.uint8)
        yellow = np.zeros_like(white)
        cv2.line(yellow, (250, 90), (390, 90), 255, 6)

        _, _, filtered = make_canonical_road_image_from_masks(
            white,
            yellow,
            lateral_m_per_px=1.4 / 640.0,
            forward_m_per_px=1.5 / 220.0,
            geometry_filter_enabled=True,
            min_line_verticality=0.30,
            min_component_area_px=4,
            bottom_ignore_m=0.0,
        )
        _, _, preserved = make_canonical_road_image_from_masks(
            white,
            yellow,
            lateral_m_per_px=1.4 / 640.0,
            forward_m_per_px=1.5 / 220.0,
            geometry_filter_enabled=True,
            yellow_geometry_filter_enabled=False,
            min_line_verticality=0.30,
            min_component_area_px=4,
            bottom_ignore_m=0.0,
        )

        self.assertEqual(int(np.count_nonzero(filtered)), 0)
        self.assertGreater(int(np.count_nonzero(preserved)), 0)

    def test_yolo_white_can_bypass_thickness_and_geometry_filters(self):
        white = np.zeros((220, 640), dtype=np.uint8)
        yellow = np.zeros_like(white)
        cv2.line(white, (500, 0), (520, 219), 255, 34)

        _, filtered, _ = make_canonical_road_image_from_masks(
            white,
            yellow,
            lateral_m_per_px=1.4 / 640.0,
            forward_m_per_px=1.5 / 220.0,
            white_max_component_thickness_px=28.0,
            geometry_filter_enabled=True,
            min_component_area_px=12,
            bottom_ignore_m=0.0,
        )
        _, preserved, _ = make_canonical_road_image_from_masks(
            white,
            yellow,
            lateral_m_per_px=1.4 / 640.0,
            forward_m_per_px=1.5 / 220.0,
            white_max_component_thickness_px=28.0,
            geometry_filter_enabled=True,
            min_component_area_px=12,
            preserve_white_mask=True,
            bottom_ignore_m=0.0,
        )

        self.assertEqual(int(np.count_nonzero(filtered)), 0)
        self.assertGreater(int(np.count_nonzero(preserved)), 0)


if __name__ == "__main__":
    unittest.main()
