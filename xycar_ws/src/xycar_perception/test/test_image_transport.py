import unittest

import cv2
import numpy as np

from xycar_perception.camera_perception_node import (
    decode_compressed_image,
    scale_camera_matrix,
)


class ImageTransportTest(unittest.TestCase):
    def test_jpeg_is_decoded_as_bgr_image(self):
        source = np.zeros((24, 32, 3), dtype=np.uint8)
        source[:, :, 1] = 180
        encoded_ok, encoded = cv2.imencode(".jpg", source)
        self.assertTrue(encoded_ok)

        decoded = decode_compressed_image(encoded.tobytes())

        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, source.shape)
        self.assertGreater(float(decoded[:, :, 1].mean()), 150.0)

    def test_empty_payload_is_rejected(self):
        self.assertIsNone(decode_compressed_image(b""))

    def test_invalid_payload_is_rejected(self):
        self.assertIsNone(decode_compressed_image(b"not-an-image"))

    def test_camera_matrix_scales_to_runtime_resolution(self):
        matrix = np.array(
            [[800.0, 0.0, 640.0], [0.0, 820.0, 512.0], [0.0, 0.0, 1.0]]
        )
        scaled = scale_camera_matrix(matrix, (1280, 1024), (640, 512))
        np.testing.assert_allclose(
            scaled,
            np.array(
                [[400.0, 0.0, 320.0], [0.0, 410.0, 256.0], [0.0, 0.0, 1.0]]
            ),
        )

    def test_bev_outside_source_is_filled_with_neutral_gray(self):
        from xycar_perception.camera_perception_node import CameraPerceptionNode

        node = CameraPerceptionNode.__new__(CameraPerceptionNode)
        node.projection_mode = "bev_homography"
        node.enable_rectify = False
        node.src_tl_x_ratio = 0.35
        node.src_tr_x_ratio = 0.65
        node.src_bl_x_ratio = 0.15
        node.src_br_x_ratio = 0.85
        node.src_top_y_ratio = 0.40
        node.src_bottom_y_ratio = 0.80
        node.bev_width = 80
        node.bev_height = 40
        node.dst_left_ratio = 0.20
        node.dst_right_ratio = 0.80
        node.dst_top_y_ratio = 0.0
        node.dst_bottom_y_ratio = 0.75
        node.bev_border_gray = 70
        node.bev_valid_erode_px = 2
        node.M = None
        node.M_inv = None
        node.homography_input_shape = None
        node.homography_output_shape = None

        source = np.full((60, 100, 3), 110, dtype=np.uint8)
        bev = node.prepare_projection_image(source)

        self.assertEqual(tuple(bev.shape), (40, 80, 3))
        self.assertGreaterEqual(int(bev.min()), 70)
        self.assertEqual(tuple(node.current_bev_valid_mask.shape), (40, 80))
        invalid = np.argwhere(node.current_bev_valid_mask == 0)
        self.assertGreater(len(invalid), 0)
        neutral_border = np.all(bev == 70, axis=2)
        self.assertGreater(int(np.count_nonzero(neutral_border)), 0)
        self.assertTrue(
            np.all(node.current_bev_valid_mask[neutral_border] == 0)
        )
        self.assertEqual(
            int(np.count_nonzero(node.current_bev_valid_mask[32:, :])),
            0,
        )

    def test_homography_supports_independent_corner_rows(self):
        from xycar_perception.camera_perception_node import CameraPerceptionNode

        node = CameraPerceptionNode.__new__(CameraPerceptionNode)
        node.src_tl_x_ratio = 0.30
        node.src_tr_x_ratio = 0.70
        node.src_br_x_ratio = 0.90
        node.src_bl_x_ratio = 0.10
        node.src_top_y_ratio = 0.40
        node.src_bottom_y_ratio = 0.80
        node.src_tl_y_ratio = 0.39
        node.src_tr_y_ratio = 0.41
        node.src_br_y_ratio = 0.82
        node.src_bl_y_ratio = 0.78
        node.bev_width = 80
        node.bev_height = 40
        node.dst_left_ratio = 0.20
        node.dst_right_ratio = 0.80
        node.dst_top_y_ratio = 0.0
        node.dst_bottom_y_ratio = 0.75

        node.build_homography(100, 60)

        source = np.float32(
            [[[30.0, 23.4], [70.0, 24.6], [90.0, 49.2], [10.0, 46.8]]]
        )
        transformed = cv2.perspectiveTransform(source, node.M)
        expected = np.float32(
            [[[16.0, 0.0], [64.0, 0.0], [64.0, 30.0], [16.0, 30.0]]]
        )
        np.testing.assert_allclose(transformed, expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
