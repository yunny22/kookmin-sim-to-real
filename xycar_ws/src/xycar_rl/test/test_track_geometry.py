from pathlib import Path
import math
import unittest

import numpy as np

from xycar_rl.track_geometry import TrackReference, wrap_angle
from xycar_rl.privileged_track_expert import command_for_curvature


PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORLD = PROJECT_ROOT / "worlds" / "kookmin_xycar_track_final.sdf"


class TrackGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = TrackReference.from_sdf(WORLD)

    def test_loads_closed_cad_reference(self):
        self.assertGreater(self.track.points.shape[0], 800)
        self.assertGreater(self.track.length_m, 20.0)
        self.assertLess(self.track.length_m, 40.0)

    def test_projection_has_small_error_on_reference(self):
        for index in range(0, len(self.track.points), 97):
            point = self.track.points[index]
            next_point = self.track.points[(index + 1) % len(self.track.points)]
            yaw = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
            projection = self.track.project(point[0], point[1], yaw)
            self.assertLess(abs(projection.cross_track_error_m), 1.0e-6)
            self.assertLess(abs(projection.heading_error_rad), 0.08)

    def test_pose_and_projection_are_consistent(self):
        pose = self.track.pose_at(
            0.37,
            lateral_error_m=0.12,
            yaw_error_rad=0.08,
        )
        projection = self.track.project(pose.x, pose.y, pose.yaw)
        self.assertAlmostEqual(projection.progress_fraction, 0.37, delta=0.01)
        self.assertAlmostEqual(projection.cross_track_error_m, 0.12, delta=0.01)
        self.assertAlmostEqual(projection.heading_error_rad, 0.08, delta=0.03)

    def test_progress_delta_wraps_at_finish_line(self):
        previous = 0.99 * self.track.length_m
        current = 0.01 * self.track.length_m
        self.assertAlmostEqual(
            self.track.progress_delta(previous, current),
            0.02 * self.track.length_m,
            places=5,
        )

    def test_progress_sampling_wraps_and_curvature_is_finite(self):
        first, yaw_first = self.track.sample_at_progress(0.37)
        wrapped, yaw_wrapped = self.track.sample_at_progress(
            self.track.length_m + 0.37
        )
        np.testing.assert_allclose(first, wrapped, atol=1.0e-9)
        self.assertAlmostEqual(wrap_angle(yaw_first - yaw_wrapped), 0.0)
        self.assertTrue(math.isfinite(self.track.curvature_at(0.37)))
        self.assertTrue(
            math.isfinite(
                self.track.max_abs_curvature_ahead(
                    self.track.length_m - 0.10,
                )
            )
        )

    def test_curvature_to_command_uses_measured_sign(self):
        self.assertLess(command_for_curvature(1.0), 0.0)
        self.assertGreater(command_for_curvature(-1.0), 0.0)
        self.assertAlmostEqual(command_for_curvature(0.0), 0.0)

    def test_default_reference_uses_competition_direction(self):
        forward = TrackReference.from_sdf(WORLD, target_right_offset_m=0.0,
                                          reverse_direction=False)
        competition = TrackReference.from_sdf(WORLD, target_right_offset_m=0.0,
                                              reverse_direction=True)
        for fraction in (0.17, 0.43, 0.81):
            forward_pose = forward.pose_at((1.0 - fraction) % 1.0)
            competition_pose = competition.pose_at(fraction)
            self.assertLess(
                np.hypot(
                    competition_pose.x - forward_pose.x,
                    competition_pose.y - forward_pose.y,
                ),
                0.05,
            )
            yaw_difference = abs(
                wrap_angle(competition_pose.yaw - forward_pose.yaw)
            )
            self.assertAlmostEqual(yaw_difference, np.pi, delta=0.12)

    def test_lane_offset_is_to_the_right_in_both_directions(self):
        for reverse_direction in (False, True):
            center = TrackReference.from_sdf(
                WORLD,
                target_right_offset_m=0.0,
                reverse_direction=reverse_direction,
            )
            right_lane = TrackReference.from_sdf(
                WORLD,
                target_right_offset_m=0.20,
                reverse_direction=reverse_direction,
            )
            for index in range(0, len(center.points), 113):
                previous = center.points[(index - 1) % len(center.points)]
                following = center.points[(index + 1) % len(center.points)]
                tangent = following - previous
                tangent /= np.linalg.norm(tangent)
                right_normal = np.asarray([tangent[1], -tangent[0]])
                offset = right_lane.points[index] - center.points[index]
                self.assertAlmostEqual(
                    float(np.dot(offset, right_normal)),
                    0.20,
                    delta=0.015,
                )

    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(3.0 * math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle(-3.0 * math.pi), -math.pi)


if __name__ == "__main__":
    unittest.main()
