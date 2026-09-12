import unittest

from kaiev26_msgs.msg import RoadSegment
from std_msgs.msg import Header

from xycar_perception.camera_perception_node import CameraPerceptionNode, make_point


class CenterlineFitTest(unittest.TestCase):
    def make_node(self):
        node = CameraPerceptionNode.__new__(CameraPerceptionNode)
        node.min_centerline_points = 3
        node.centerline_polyfit_enabled = True
        node.centerline_polyfit_degree = 2
        node.centerline_sample_spacing_m = 0.05
        node.centerline_temporal_alpha = 0.45
        node.centerline_temporal_timeout_sec = 0.5
        node.previous_centerline_points = []
        node.previous_centerline_wall_time = 0.0
        return node

    def make_segment(self, y_value):
        segment = RoadSegment()
        segment.points = [
            make_point(x_value, y_value) for x_value in (0.2, 0.4, 0.6, 0.8)
        ]
        segment.confidence = 0.9
        return segment

    def test_quadratic_centerline_is_resampled(self):
        node = self.make_node()
        points = [
            make_point(x_value, 0.2 * x_value * x_value)
            for x_value in (0.2, 0.4, 0.6, 0.8)
        ]

        result = node.fit_and_smooth_centerline(points)

        self.assertGreater(len(result), len(points))
        for point in result:
            self.assertAlmostEqual(point.y, 0.2 * point.x * point.x, places=5)

    def test_fresh_previous_path_is_temporally_blended(self):
        node = self.make_node()
        first = [make_point(x_value, 0.0) for x_value in (0.2, 0.4, 0.6)]
        second = [make_point(x_value, 0.1) for x_value in (0.2, 0.4, 0.6)]

        node.fit_and_smooth_centerline(first)
        result = node.fit_and_smooth_centerline(second)

        for point in result:
            self.assertAlmostEqual(point.y, 0.045, places=5)

    def test_yellow_and_one_visible_boundary_produce_lane_midline(self):
        node = self.make_node()
        node.centerline_mode = "lane_midline"
        node.use_yellow_as_centerline = False
        node.lane_width_m = 0.8
        node.source_name = "test"
        node.detection_id = 1

        yellow = self.make_segment(0.0)
        visible_boundary = self.make_segment(-0.4)
        result = node.build_centerline(Header(), None, visible_boundary, yellow)

        self.assertGreaterEqual(len(result.points), node.min_centerline_points)
        for point in result.points:
            self.assertAlmostEqual(point.y, -0.2, places=5)
        self.assertGreater(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
