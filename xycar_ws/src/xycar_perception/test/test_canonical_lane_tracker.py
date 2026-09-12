import unittest

import cv2
import numpy as np

from xycar_perception.canonical_lane_tracker import (
    CanonicalLaneTracker,
    CurveCandidate,
    _component_candidates,
    _continuation_chain,
    _is_curve_continuation,
    _prune_curve_components,
)


class CanonicalLaneTrackerTest(unittest.TestCase):
    def setUp(self):
        self.tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            expected_half_lane_width_m=0.412,
            lane_width_tolerance_m=0.15,
            min_white_yellow_offset_m=0.18,
            confirmation_frames=3,
            coast_sec=0.30,
            search_sec=1.00,
            transverse_clutter_row_fraction=0.12,
            transverse_clutter_min_rows=6,
            persistent_prediction_enabled=True,
            line_width_px=5,
        )

    @staticmethod
    def masks(
        *,
        left=True,
        right=True,
        reflection=False,
        yellow=True,
    ):
        white = np.zeros((144, 256), dtype=np.uint8)
        yellow_mask = np.zeros_like(white)
        if left:
            cv2.line(white, (53, 8), (53, 136), 255, 5)
        if right:
            cv2.line(white, (203, 8), (203, 136), 255, 5)
        if reflection:
            cv2.line(white, (160, 10), (160, 134), 255, 5)
        if yellow:
            for row in range(10, 130, 32):
                cv2.line(
                    yellow_mask,
                    (128, row),
                    (128, min(row + 19, 136)),
                    255,
                    5,
                )
        return white, yellow_mask

    def confirm(self, white, yellow):
        result = None
        for index in range(3):
            result = self.tracker.update(
                white,
                yellow,
                timestamp_sec=index / 30.0,
            )
        return result

    def test_rejects_white_reflection_inside_lane_corridor(self):
        white, yellow = self.masks(reflection=True)
        result = self.confirm(white, yellow)

        self.assertEqual(result.statuses["left_white"], "confirmed")
        self.assertEqual(result.statuses["right_white"], "confirmed")
        self.assertGreater(np.count_nonzero(result.white_mask[:, 48:59]), 0)
        self.assertGreater(np.count_nonzero(result.white_mask[:, 198:209]), 0)
        self.assertEqual(np.count_nonzero(result.white_mask[:, 156:165]), 0)

    def test_coasts_then_infers_missing_boundary(self):
        white, yellow = self.masks()
        self.confirm(white, yellow)

        missing_white, yellow = self.masks(right=False)
        coast = self.tracker.update(missing_white, yellow, timestamp_sec=0.15)
        self.assertEqual(coast.statuses["right_white"], "coasting")
        self.assertGreater(np.count_nonzero(coast.white_mask[:, 198:209]), 0)

        inferred = self.tracker.update(
            missing_white,
            yellow,
            timestamp_sec=0.50,
        )
        self.assertEqual(
            inferred.statuses["right_white"], "width_predicted"
        )
        self.assertGreater(
            np.count_nonzero(inferred.white_mask[:, 198:209]),
            0,
        )

    def test_persistent_interior_reflection_cannot_replace_outer_track(self):
        white, yellow = self.masks()
        self.confirm(white, yellow)

        reflection_only = np.zeros_like(white)
        cv2.line(reflection_only, (175, 10), (175, 134), 255, 5)
        result = None
        for index in range(1, 8):
            result = self.tracker.update(
                reflection_only,
                yellow,
                timestamp_sec=0.10 + index * 0.10,
            )

        self.assertNotEqual(result.statuses["right_white"], "confirmed")
        self.assertEqual(np.count_nonzero(result.white_mask[:, 171:180]), 0)

    def test_requires_persistent_candidate_before_initial_lock(self):
        white, yellow = self.masks()
        first = self.tracker.update(white, yellow, timestamp_sec=0.0)
        second = self.tracker.update(white, yellow, timestamp_sec=0.03)
        third = self.tracker.update(white, yellow, timestamp_sec=0.06)

        self.assertEqual(first.statuses["yellow"], "acquiring")
        self.assertEqual(second.statuses["left_white"], "acquiring")
        self.assertEqual(third.statuses["yellow"], "confirmed")
        self.assertEqual(third.statuses["left_white"], "confirmed")

    def test_inferred_curve_stays_parallel_to_observed_boundary(self):
        white = np.zeros((144, 256), dtype=np.uint8)
        yellow = np.zeros_like(white)
        rows = np.arange(8, 137, dtype=np.int32)
        left_x = np.rint(48.0 + 0.0025 * (rows - 72) ** 2).astype(
            np.int32
        )
        right_x = left_x + 150
        left_points = np.column_stack((left_x, rows))
        right_points = np.column_stack((right_x, rows))
        cv2.polylines(white, [left_points], False, 255, 5)
        cv2.polylines(white, [right_points], False, 255, 5)
        for row in range(10, 130, 32):
            cv2.line(yellow, (128, row), (128, row + 19), 255, 5)
        self.confirm(white, yellow)

        missing_right = np.zeros_like(white)
        cv2.polylines(missing_right, [left_points], False, 255, 5)
        inferred = self.tracker.update(
            missing_right,
            yellow,
            timestamp_sec=0.50,
        )

        self.assertEqual(
            inferred.statuses["right_white"], "width_predicted"
        )
        separations = []
        for row in range(20, 125, 8):
            xs = np.flatnonzero(inferred.white_mask[row])
            left = xs[xs < 128]
            right = xs[xs >= 128]
            if left.size and right.size:
                separations.append(float(np.median(right) - np.median(left)))
        self.assertGreater(len(separations), 8)
        self.assertAlmostEqual(float(np.median(separations)), 150.0, delta=2.0)
        self.assertLess(float(np.std(separations)), 2.0)

    def test_width_prediction_persists_and_ignores_interior_clutter(self):
        cluttered, yellow = self.masks(right=False)
        cv2.line(cluttered, (175, 8), (175, 136), 255, 5)
        initial = self.confirm(cluttered, yellow)
        self.assertEqual(
            initial.statuses["right_white"],
            "width_predicted",
        )

        result = None
        for index in range(1, 301):
            result = self.tracker.update(
                cluttered,
                yellow,
                timestamp_sec=0.10 + index / 30.0,
            )
            self.assertEqual(
                result.statuses["right_white"],
                "width_predicted",
            )

        self.assertGreater(
            np.count_nonzero(result.white_mask[:, 198:209]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.white_mask[:, 171:180]),
            0,
        )

    def test_can_start_with_only_center_and_one_white_boundary(self):
        one_side, yellow = self.masks(right=False)
        first = self.tracker.update(one_side, yellow, timestamp_sec=0.0)
        second = self.tracker.update(one_side, yellow, timestamp_sec=0.03)
        third = self.tracker.update(one_side, yellow, timestamp_sec=0.06)

        self.assertEqual(first.statuses["right_white"], "lost")
        self.assertEqual(second.statuses["right_white"], "lost")
        self.assertEqual(
            third.statuses["right_white"],
            "width_predicted",
        )
        self.assertGreater(
            np.count_nonzero(third.white_mask[:, 198:209]),
            0,
        )

    def test_predicts_left_boundary_from_center_and_right_boundary(self):
        one_side, yellow = self.masks(left=False)
        result = self.confirm(one_side, yellow)

        self.assertEqual(
            result.statuses["left_white"],
            "width_predicted",
        )
        self.assertEqual(result.statuses["right_white"], "confirmed")
        self.assertGreater(
            np.count_nonzero(result.white_mask[:, 48:59]),
            0,
        )

    def test_real_boundary_replaces_width_prediction_in_one_frame(self):
        one_side, yellow = self.masks(right=False)
        predicted = self.confirm(one_side, yellow)
        self.assertEqual(
            predicted.statuses["right_white"],
            "width_predicted",
        )

        both_sides, yellow = self.masks()
        replaced = self.tracker.update(
            both_sides,
            yellow,
            timestamp_sec=0.10,
        )
        self.assertEqual(replaced.statuses["right_white"], "confirmed")

    def test_recovered_boundary_blends_back_without_a_lateral_jump(self):
        one_side, yellow = self.masks(right=False)
        self.confirm(one_side, yellow)

        recovered, yellow = self.masks(right=False)
        cv2.line(recovered, (215, 8), (215, 136), 255, 5)
        result = self.tracker.update(
            recovered,
            yellow,
            timestamp_sec=0.10,
        )

        self.assertEqual(result.statuses["right_white"], "confirmed")
        self.assertGreater(
            np.count_nonzero(result.white_mask[:, 207:214]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.white_mask[:, 218:224]),
            0,
        )

    def test_continuation_chain_rejects_a_component_beyond_30cm(self):
        mask = np.zeros((144, 256), dtype=np.uint8)
        cv2.line(mask, (128, 85), (128, 136), 255, 5)
        cv2.line(mask, (128, 5), (128, 45), 255, 5)
        candidates = _component_candidates(mask, min_span_px=6)
        seed = max(candidates, key=lambda candidate: candidate.row_max)
        combined = _continuation_chain(
            candidates,
            seed,
            shape=mask.shape,
            max_row_gap_px=0.30 / (1.5 / 144.0),
            lateral_gate_px=12.0,
        )

        self.assertGreaterEqual(combined.row_min, 80)
        self.assertEqual(np.count_nonzero(combined.mask[:50]), 0)

    def test_continuation_chain_accepts_the_next_30cm_segment(self):
        mask = np.zeros((144, 256), dtype=np.uint8)
        cv2.line(mask, (128, 85), (128, 136), 255, 5)
        cv2.line(mask, (128, 54), (128, 70), 255, 5)
        candidates = _component_candidates(mask, min_span_px=6)
        seed = max(candidates, key=lambda candidate: candidate.row_max)
        combined = _continuation_chain(
            candidates,
            seed,
            shape=mask.shape,
            max_row_gap_px=0.30 / (1.5 / 144.0),
            lateral_gate_px=12.0,
        )

        self.assertLessEqual(combined.row_min, 56)
        self.assertGreater(np.count_nonzero(combined.mask[50:75]), 0)

    def test_tracker_connects_visible_dashes_across_a_38cm_gap(self):
        mask = np.zeros((144, 256), dtype=np.uint8)
        cv2.line(mask, (128, 100), (128, 136), 255, 5)
        cv2.line(mask, (128, 50), (128, 64), 255, 5)
        candidates = _component_candidates(mask, min_span_px=6)
        seed = max(candidates, key=lambda candidate: candidate.row_max)

        combined = _continuation_chain(
            candidates,
            seed,
            shape=mask.shape,
            max_row_gap_px=self.tracker.max_continuation_gap_px,
            lateral_gate_px=12.0,
        )

        self.assertAlmostEqual(
            self.tracker.max_continuation_gap_px * (1.5 / 144.0),
            0.38,
            places=6,
        )
        self.assertLessEqual(combined.row_min, 52)
        self.assertGreater(np.count_nonzero(combined.mask[48:68]), 0)

    def test_yellow_fit_gate_removes_off_curve_chained_component(self):
        mask = np.zeros((144, 256), dtype=np.uint8)
        cv2.line(mask, (128, 96), (128, 136), 255, 5)
        cv2.line(mask, (128, 48), (128, 70), 255, 5)
        cv2.line(mask, (170, 20), (170, 42), 255, 5)
        candidate = CurveCandidate(
            coefficients=np.array([0.0, 0.0, 128.0]),
            mask=mask,
            row_min=20,
            row_max=136,
            area=int(np.count_nonzero(mask)),
            span=117,
        )

        pruned = _prune_curve_components(candidate, max_residual_px=15.0)

        self.assertGreater(np.count_nonzero(pruned.mask[:, 123:134]), 0)
        self.assertEqual(np.count_nonzero(pruned.mask[:, 165:176]), 0)

    def test_rejects_short_white_center_impostor_without_yellow(self):
        tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            expected_half_lane_width_m=0.49,
            lane_width_tolerance_m=0.14,
            confirmation_frames=1,
            coast_sec=0.0,
            search_sec=0.0,
            width_prediction_enabled=False,
            persistent_prediction_enabled=False,
        )
        white = np.zeros((144, 256), dtype=np.uint8)
        cv2.line(white, (45, 8), (55, 136), 255, 5)
        cv2.line(white, (135, 75), (135, 105), 255, 5)

        result = tracker.update(
            white, np.zeros_like(white), timestamp_sec=0.0
        )

        self.assertGreater(np.count_nonzero(result.white_mask[:, 40:61]), 0)
        self.assertEqual(np.count_nonzero(result.white_mask[:, 130:141]), 0)

    def test_stale_yellow_does_not_bypass_white_pair_check(self):
        tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            expected_half_lane_width_m=0.49,
            lane_width_tolerance_m=0.14,
            confirmation_frames=1,
            coast_sec=0.0,
            search_sec=0.0,
            smoothing_alpha=1.0,
            width_prediction_enabled=False,
            persistent_prediction_enabled=False,
        )
        initial_white = np.zeros((144, 256), dtype=np.uint8)
        initial_yellow = np.zeros_like(initial_white)
        cv2.line(initial_white, (35, 8), (35, 136), 255, 5)
        cv2.line(initial_white, (215, 8), (215, 136), 255, 5)
        cv2.line(initial_yellow, (125, 30), (125, 90), 255, 5)
        tracker.update(initial_white, initial_yellow, timestamp_sec=0.0)

        next_white = np.zeros_like(initial_white)
        cv2.line(next_white, (45, 8), (55, 136), 255, 5)
        cv2.line(next_white, (135, 75), (135, 105), 255, 5)
        result = tracker.update(
            next_white,
            np.zeros_like(initial_yellow),
            timestamp_sec=1.0 / 30.0,
        )

        self.assertGreater(np.count_nonzero(result.white_mask[:, 40:61]), 0)
        self.assertEqual(np.count_nonzero(result.white_mask[:, 130:141]), 0)

    def test_short_curved_yellow_uses_only_overlapping_white_rows(self):
        tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            expected_half_lane_width_m=0.49,
            lane_width_tolerance_m=0.14,
            confirmation_frames=1,
            coast_sec=0.0,
            search_sec=0.0,
            smoothing_alpha=1.0,
            width_prediction_enabled=False,
            persistent_prediction_enabled=False,
        )
        white = np.zeros((144, 256), dtype=np.uint8)
        yellow = np.zeros_like(white)
        white_rows = np.arange(33, 113, dtype=np.float64)
        white_x = np.polyval(
            np.array([0.0141894624, -4.01056043, 330.596988]),
            white_rows,
        )
        yellow_rows = np.arange(80, 105, dtype=np.float64)
        yellow_x = np.polyval(
            np.array([0.0238294314, -6.17038462, 558.023946]),
            yellow_rows,
        )
        cv2.polylines(
            white,
            [
                np.column_stack((white_x, white_rows))
                .round()
                .astype(np.int32)
            ],
            False,
            255,
            5,
        )
        cv2.polylines(
            yellow,
            [
                np.column_stack((yellow_x, yellow_rows))
                .round()
                .astype(np.int32)
            ],
            False,
            255,
            5,
        )

        result = tracker.update(white, yellow, timestamp_sec=0.0)

        self.assertGreater(np.count_nonzero(result.white_mask), 0)
        self.assertGreater(np.count_nonzero(result.yellow_mask), 0)

    def test_short_yellow_is_not_rejected_by_nonoverlapping_other_boundary(self):
        tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            expected_half_lane_width_m=0.49,
            lane_width_tolerance_m=0.14,
            confirmation_frames=1,
            coast_sec=0.0,
            search_sec=0.0,
            smoothing_alpha=1.0,
            width_prediction_enabled=False,
            persistent_prediction_enabled=False,
        )
        white = np.zeros((144, 256), dtype=np.uint8)
        yellow = np.zeros_like(white)
        cv2.line(white, (38, 12), (38, 58), 255, 5)
        cv2.line(white, (218, 82), (218, 136), 255, 5)
        cv2.line(yellow, (128, 22), (128, 46), 255, 5)

        result = tracker.update(white, yellow, timestamp_sec=0.0)

        self.assertEqual(result.statuses["yellow"], "confirmed")
        self.assertGreater(np.count_nonzero(result.yellow_mask), 0)

    def test_yolo_center_yellow_can_survive_when_white_is_temporarily_missing(self):
        tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            confirmation_frames=1,
            coast_sec=0.0,
            search_sec=0.0,
            width_prediction_enabled=False,
            persistent_prediction_enabled=False,
            allow_unpaired_yellow=True,
        )
        white = np.zeros((144, 256), dtype=np.uint8)
        yellow = np.zeros_like(white)
        cv2.line(yellow, (128, 32), (142, 58), 255, 5)

        result = tracker.update(white, yellow, timestamp_sec=0.0)

        self.assertEqual(result.statuses["yellow"], "confirmed")
        self.assertGreater(np.count_nonzero(result.yellow_mask), 0)

        outside = np.zeros_like(yellow)
        cv2.line(outside, (245, 32), (245, 70), 255, 5)
        rejected = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            confirmation_frames=1,
            coast_sec=0.0,
            search_sec=0.0,
            width_prediction_enabled=False,
            persistent_prediction_enabled=False,
            allow_unpaired_yellow=True,
        ).update(white, outside, timestamp_sec=0.0)
        self.assertEqual(rejected.statuses["yellow"], "lost")

    def test_confirmed_white_is_rendered_as_one_continuous_curve(self):
        tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            confirmation_frames=1,
            coast_sec=0.0,
            search_sec=0.0,
            smoothing_alpha=1.0,
            width_prediction_enabled=False,
            persistent_prediction_enabled=False,
        )
        white = np.zeros((144, 256), dtype=np.uint8)
        yellow = np.zeros_like(white)
        rows = np.arange(8, 137, dtype=np.int32)
        xs = np.rint(50.0 + 5.0 * np.sin(rows / 12.0)).astype(np.int32)
        cv2.polylines(
            white,
            [np.column_stack((xs, rows))],
            False,
            255,
            5,
        )

        result = tracker.update(white, yellow, timestamp_sec=0.0)

        self.assertEqual(result.statuses["left_white"], "confirmed")
        self.assertFalse(np.array_equal(result.white_mask, white))
        occupied_rows = np.any(result.white_mask > 0, axis=1)
        self.assertTrue(np.all(occupied_rows[8:137]))

    def test_rejects_yellow_outside_near_field_right_boundary(self):
        tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            expected_half_lane_width_m=0.49,
            lane_width_tolerance_m=0.14,
            confirmation_frames=1,
            coast_sec=0.0,
            search_sec=0.0,
            smoothing_alpha=1.0,
            width_prediction_enabled=False,
            persistent_prediction_enabled=False,
        )
        white = np.zeros((144, 256), dtype=np.uint8)
        yellow = np.zeros_like(white)
        white_rows = np.arange(20, 119, dtype=np.float64)
        white_x = np.polyval(
            np.polyfit(
                np.array([20.0, 40.0, 118.0]),
                np.array([30.0, 76.0, 178.0]),
                2,
            ),
            white_rows,
        )
        cv2.polylines(
            white,
            [
                np.column_stack((white_x, white_rows))
                .round()
                .astype(np.int32)
            ],
            False,
            255,
            5,
        )
        cv2.line(yellow, (118, 14), (123, 40), 255, 5)
        # A short reflection sits outside the false yellow candidate. The
        # longer physical right boundary must remain the topology reference.
        cv2.line(white, (222, 77), (222, 93), 255, 5)

        result = tracker.update(white, yellow, timestamp_sec=0.0)

        self.assertEqual(result.statuses["yellow"], "lost")
        self.assertEqual(np.count_nonzero(result.yellow_mask), 0)
        self.assertEqual(result.statuses["left_white"], "lost")
        self.assertEqual(result.statuses["right_white"], "confirmed")
        self.assertGreater(np.count_nonzero(result.white_mask), 0)

    def test_observation_only_replaces_distant_curve_in_one_frame(self):
        tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            confirmation_frames=1,
            coast_sec=0.0,
            search_sec=0.0,
            smoothing_alpha=1.0,
            width_prediction_enabled=False,
            persistent_prediction_enabled=False,
        )
        first = np.zeros((144, 256), dtype=np.uint8)
        shifted = np.zeros_like(first)
        empty = np.zeros_like(first)
        cv2.line(first, (35, 8), (35, 136), 255, 5)
        cv2.line(shifted, (80, 8), (80, 136), 255, 5)
        tracker.update(first, empty, timestamp_sec=0.0)

        result = tracker.update(
            shifted, empty, timestamp_sec=1.0 / 30.0
        )

        self.assertGreater(np.count_nonzero(result.white_mask[:, 75:86]), 0)
        self.assertEqual(np.count_nonzero(result.white_mask[:, 30:41]), 0)

    def test_curvature_adaptive_gate_connects_a_bending_dash(self):
        shape = (144, 256)
        first = CurveCandidate(
            coefficients=np.array([0.006, -0.3, 120.0]),
            mask=np.zeros(shape, dtype=np.uint8),
            row_min=80,
            row_max=130,
            area=100,
            span=51,
        )
        second = CurveCandidate(
            coefficients=np.array([0.010, -0.62, 135.6]),
            mask=np.zeros(shape, dtype=np.uint8),
            row_min=40,
            row_max=60,
            area=60,
            span=21,
        )

        self.assertFalse(
            _is_curve_continuation(
                first,
                second,
                max_row_gap_px=40.0,
                lateral_gate_px=12.0,
            )
        )
        self.assertTrue(
            _is_curve_continuation(
                first,
                second,
                max_row_gap_px=40.0,
                lateral_gate_px=12.0,
                curvature_gate_gain=1.0,
                curvature_max_extra_px=18.0,
            )
        )

    def test_checkerboard_startline_holds_previous_lane_tracks(self):
        white, yellow = self.masks()
        self.confirm(white, yellow)

        checkerboard = np.zeros_like(white)
        cell_width = 16
        for row_index, top in enumerate(range(45, 81, 12)):
            for column_index, left in enumerate(range(32, 224, cell_width)):
                if (row_index + column_index) % 2 == 0:
                    cv2.rectangle(
                        checkerboard,
                        (left, top),
                        (left + cell_width - 1, top + 11),
                        255,
                        -1,
                    )
        held = self.tracker.update(
            checkerboard,
            yellow,
            timestamp_sec=2.0,
        )

        self.assertEqual(held.statuses["yellow"], "startline_hold")
        self.assertEqual(held.statuses["left_white"], "startline_hold")
        self.assertEqual(held.statuses["right_white"], "startline_hold")
        self.assertGreater(np.count_nonzero(held.white_mask[:, 48:59]), 0)
        self.assertGreater(np.count_nonzero(held.white_mask[:, 198:209]), 0)
        self.assertEqual(np.count_nonzero(held.white_mask[:, 90:166]), 0)

    def test_complete_camera_dropout_keeps_last_road_indefinitely(self):
        white, yellow = self.masks()
        self.confirm(white, yellow)
        empty = np.zeros_like(white)

        for timestamp in (0.50, 2.0, 10.0, 60.0):
            predicted = self.tracker.update(empty, empty, timestamp)
            self.assertEqual(
                predicted.statuses["yellow"],
                "persistent_predicted",
            )
            self.assertEqual(
                predicted.statuses["left_white"],
                "persistent_predicted",
            )
            self.assertEqual(
                predicted.statuses["right_white"],
                "persistent_predicted",
            )
            self.assertGreater(np.count_nonzero(predicted.yellow_mask), 0)
            self.assertGreater(
                np.count_nonzero(predicted.white_mask[:, 48:59]),
                0,
            )
            self.assertGreater(
                np.count_nonzero(predicted.white_mask[:, 198:209]),
                0,
            )

    def test_observation_replaces_persistent_road_in_one_frame(self):
        white, yellow = self.masks()
        self.confirm(white, yellow)
        empty = np.zeros_like(white)
        predicted = self.tracker.update(empty, empty, timestamp_sec=10.0)
        self.assertEqual(
            predicted.statuses["left_white"],
            "persistent_predicted",
        )

        shifted_white = np.zeros_like(white)
        shifted_yellow = np.zeros_like(yellow)
        cv2.line(shifted_white, (58, 8), (58, 136), 255, 5)
        cv2.line(shifted_white, (208, 8), (208, 136), 255, 5)
        for row in range(10, 130, 32):
            cv2.line(
                shifted_yellow,
                (133, row),
                (133, min(row + 19, 136)),
                255,
                5,
            )
        observed = self.tracker.update(
            shifted_white,
            shifted_yellow,
            timestamp_sec=10.03,
        )

        self.assertEqual(observed.statuses["yellow"], "confirmed")
        self.assertEqual(observed.statuses["left_white"], "confirmed")
        self.assertEqual(observed.statuses["right_white"], "confirmed")
        self.assertGreater(
            np.count_nonzero(observed.white_mask[:, 54:63]),
            0,
        )

    def test_one_boundary_seed_survives_complete_dropout(self):
        one_side, yellow = self.masks(right=False)
        seeded = self.confirm(one_side, yellow)
        self.assertEqual(
            seeded.statuses["right_white"],
            "width_predicted",
        )

        empty = np.zeros_like(one_side)
        predicted = self.tracker.update(empty, empty, timestamp_sec=30.0)
        self.assertEqual(
            predicted.statuses["left_white"],
            "persistent_predicted",
        )
        self.assertEqual(
            predicted.statuses["right_white"],
            "width_predicted",
        )
        self.assertGreater(
            np.count_nonzero(predicted.white_mask[:, 48:59]),
            0,
        )
        self.assertGreater(
            np.count_nonzero(predicted.white_mask[:, 198:209]),
            0,
        )

    def test_empty_startup_does_not_invent_a_road(self):
        empty = np.zeros((144, 256), dtype=np.uint8)
        result = self.tracker.update(empty, empty, timestamp_sec=0.0)

        self.assertEqual(result.statuses["yellow"], "lost")
        self.assertEqual(np.count_nonzero(result.white_mask), 0)
        self.assertEqual(np.count_nonzero(result.yellow_mask), 0)

    def test_rejects_yellow_wall_outside_right_white_boundary(self):
        white, yellow = self.masks()
        self.confirm(white, yellow)

        wall = np.zeros_like(yellow)
        cv2.line(wall, (245, 8), (245, 136), 255, 5)
        result = self.tracker.update(white, wall, timestamp_sec=0.15)

        self.assertNotEqual(result.statuses["yellow"], "confirmed")
        self.assertEqual(np.count_nonzero(result.yellow_mask[:, 240:251]), 0)
        self.assertGreater(np.count_nonzero(result.yellow_mask[:, 123:134]), 0)

    def test_rejects_outside_yellow_wall_during_initial_acquisition(self):
        white, _ = self.masks()
        wall = np.zeros_like(white)
        cv2.line(wall, (245, 8), (245, 136), 255, 5)
        result = self.tracker.update(white, wall, timestamp_sec=0.0)

        self.assertEqual(result.statuses["yellow"], "lost")
        self.assertEqual(np.count_nonzero(result.yellow_mask), 0)

    def test_recovers_washed_out_center_dash_before_one_frame_lock(self):
        tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            expected_half_lane_width_m=0.412,
            lane_width_tolerance_m=0.18,
            min_white_yellow_offset_m=0.18,
            confirmation_frames=1,
            persistent_prediction_enabled=True,
            line_width_px=5,
        )
        white = np.zeros((144, 256), dtype=np.uint8)
        false_yellow = np.zeros_like(white)
        cv2.line(white, (15, 8), (15, 136), 255, 5)
        cv2.line(white, (165, 8), (165, 136), 255, 5)
        # The real yellow tape can lose saturation under ceiling lights and
        # arrive in the white mask as a short dash.
        cv2.line(white, (90, 42), (90, 70), 255, 5)
        cv2.line(false_yellow, (245, 8), (245, 40), 255, 5)

        result = tracker.update(white, false_yellow, timestamp_sec=0.0)

        self.assertEqual(result.statuses["yellow"], "confirmed")
        self.assertEqual(result.statuses["left_white"], "confirmed")
        self.assertEqual(result.statuses["right_white"], "confirmed")
        self.assertGreater(
            np.count_nonzero(result.yellow_mask[:, 86:95]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.yellow_mask[:, 240:251]),
            0,
        )
        self.assertGreater(np.count_nonzero(result.white_mask[:, 11:20]), 0)
        self.assertGreater(np.count_nonzero(result.white_mask[:, 161:170]), 0)

    def test_reclassifies_washed_out_center_dash_on_every_frame(self):
        tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            expected_half_lane_width_m=0.412,
            lane_width_tolerance_m=0.18,
            min_white_yellow_offset_m=0.18,
            confirmation_frames=1,
            coast_sec=0.0,
            search_sec=0.0,
            smoothing_alpha=1.0,
            width_prediction_enabled=False,
            persistent_prediction_enabled=False,
            line_width_px=5,
        )
        white = np.zeros((144, 256), dtype=np.uint8)
        yellow = np.zeros_like(white)
        cv2.line(white, (53, 8), (53, 136), 255, 5)
        cv2.line(white, (203, 8), (203, 136), 255, 5)
        cv2.line(yellow, (128, 10), (128, 29), 255, 5)
        tracker.update(white, yellow, timestamp_sec=0.0)

        washed_white = white.copy()
        cv2.line(washed_white, (128, 43), (128, 62), 255, 5)
        result = tracker.update(
            washed_white,
            yellow,
            timestamp_sec=1.0 / 30.0,
        )

        self.assertGreater(
            np.count_nonzero(result.yellow_mask[41:65, 124:133]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.white_mask[41:65, 124:133]),
            0,
        )

    def test_observation_only_mode_never_outputs_a_missing_line(self):
        tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            expected_half_lane_width_m=0.412,
            lane_width_tolerance_m=0.18,
            min_white_yellow_offset_m=0.18,
            confirmation_frames=1,
            coast_sec=0.0,
            search_sec=0.0,
            smoothing_alpha=1.0,
            width_prediction_enabled=False,
            transverse_clutter_row_fraction=0.12,
            transverse_clutter_min_rows=6,
            persistent_prediction_enabled=False,
            line_width_px=5,
        )
        white, yellow = self.masks()
        visible = tracker.update(white, yellow, timestamp_sec=0.0)
        self.assertEqual(visible.statuses["left_white"], "confirmed")
        self.assertEqual(visible.statuses["right_white"], "confirmed")

        right_missing, yellow = self.masks(right=False)
        missing = tracker.update(
            right_missing,
            yellow,
            timestamp_sec=1.0 / 30.0,
        )
        self.assertGreater(
            np.count_nonzero(missing.white_mask[:, 48:59]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(missing.white_mask[:, 198:209]),
            0,
        )
        self.assertNotIn(
            missing.statuses["right_white"],
            {"coasting", "width_predicted", "persistent_predicted"},
        )

        empty = np.zeros_like(white)
        blank = tracker.update(empty, empty, timestamp_sec=2.0 / 30.0)
        self.assertEqual(np.count_nonzero(blank.white_mask), 0)
        self.assertEqual(np.count_nonzero(blank.yellow_mask), 0)

        startline = np.zeros_like(white)
        cv2.rectangle(startline, (0, 58), (255, 66), 255, -1)
        no_hold = tracker.update(
            startline,
            empty,
            timestamp_sec=3.0 / 30.0,
        )
        self.assertEqual(np.count_nonzero(no_hold.white_mask), 0)
        self.assertEqual(np.count_nonzero(no_hold.yellow_mask), 0)

        reacquired = tracker.update(
            white,
            yellow,
            timestamp_sec=4.0 / 30.0,
        )
        self.assertEqual(reacquired.statuses["left_white"], "confirmed")
        self.assertEqual(reacquired.statuses["right_white"], "confirmed")
        self.assertEqual(reacquired.statuses["yellow"], "confirmed")

    def test_observation_only_mode_does_not_hold_a_rejected_candidate(self):
        tracker = CanonicalLaneTracker(
            width=256,
            height=144,
            lateral_range_m=1.4,
            forward_range_m=1.5,
            confirmation_frames=1,
            coast_sec=0.0,
            search_sec=0.0,
            smoothing_alpha=1.0,
            width_prediction_enabled=False,
            persistent_prediction_enabled=False,
        )
        white, yellow = self.masks()
        visible = tracker.update(white, yellow, timestamp_sec=0.0)
        self.assertGreater(np.count_nonzero(visible.white_mask[:, 48:59]), 0)

        displaced_white = np.zeros_like(white)
        cv2.line(displaced_white, (103, 8), (103, 136), 255, 5)
        rejected = tracker.update(
            displaced_white,
            np.zeros_like(yellow),
            timestamp_sec=1.0 / 30.0,
        )

        self.assertEqual(np.count_nonzero(rejected.white_mask[:, 48:59]), 0)

    def test_resets_when_bag_timestamp_moves_backwards(self):
        white, yellow = self.masks()
        self.confirm(white, yellow)
        reset = self.tracker.update(white, yellow, timestamp_sec=-1.0)
        self.assertEqual(reset.statuses["yellow"], "acquiring")


if __name__ == "__main__":
    unittest.main()
