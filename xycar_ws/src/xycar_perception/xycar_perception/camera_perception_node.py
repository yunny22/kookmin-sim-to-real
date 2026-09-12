from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from kaiev26_msgs.msg import (
    Centerline,
    PerceptionObjectArray,
    RoadSegment,
    RoadSegmentArray,
    TrafficLightObservationArray,
)
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from visualization_msgs.msg import Marker, MarkerArray

from xycar_perception.canonical_lane_tracker import CanonicalLaneTracker
from xycar_perception.canonical_road import (
    CanonicalRoadStages,
    make_canonical_road_image,
    make_canonical_road_image_from_masks,
)
from xycar_perception.yolo_lane_segmenter import YoloLaneSegmenter


@dataclass
class SegmentCandidate:
    segment_type: int
    points: list[Point]
    confidence: float


def make_point(x: float, y: float, z: float = 0.0) -> Point:
    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


def moving_average(values: list[float], window: int = 3) -> list[float]:
    if len(values) < window or window <= 1:
        return values
    half = window // 2
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        smoothed.append(float(np.mean(values[start:end])))
    return smoothed


def decode_compressed_image(data: bytes | bytearray | memoryview) -> np.ndarray | None:
    encoded = np.frombuffer(data, dtype=np.uint8)
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def scale_camera_matrix(
    matrix: np.ndarray,
    calibration_size: tuple[int, int] | None,
    image_size: tuple[int, int],
) -> np.ndarray:
    scaled = np.asarray(matrix, dtype=np.float64).reshape(3, 3).copy()
    if calibration_size is None or calibration_size == image_size:
        return scaled
    calibration_width, calibration_height = calibration_size
    image_width, image_height = image_size
    if calibration_width <= 0 or calibration_height <= 0:
        return scaled
    scale_x = image_width / float(calibration_width)
    scale_y = image_height / float(calibration_height)
    scaled[0, 0] *= scale_x
    scaled[0, 1] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 0] *= scale_y
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


class CameraPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("xycar_camera_perception")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("use_compressed_image", False)
        self.declare_parameter("lane_segmentation_backend", "color")
        self.declare_parameter("yolo_model_path", "")
        self.declare_parameter("yolo_device", "cpu")
        self.declare_parameter("yolo_confidence", 0.25)
        self.declare_parameter("yolo_iou", 0.50)
        self.declare_parameter("yolo_image_size", 640)
        self.declare_parameter("yolo_max_detections", 30)
        self.declare_parameter("yolo_white_class_id", 0)
        self.declare_parameter("yolo_yellow_class_id", 1)
        self.declare_parameter("yolo_cpu_threads", 0)
        self.declare_parameter("yolo_retina_masks", True)
        self.declare_parameter(
            "yolo_debug_image_topic", "/perception/yolo_debug_image"
        )
        self.declare_parameter("road_segments_topic", "/perception/road_segments")
        self.declare_parameter("centerline_topic", "/perception/centerline")
        self.declare_parameter("objects_topic", "/perception/objects")
        self.declare_parameter("traffic_lights_topic", "/perception/traffic_lights")
        self.declare_parameter(
            "rectified_image_topic", "/perception/rectified_camera_image"
        )
        self.declare_parameter("debug_image_topic", "/perception/debug_image")
        self.declare_parameter("debug_markers_topic", "/perception/debug_markers")
        self.declare_parameter(
            "canonical_road_topic", "/perception/canonical_road_image"
        )
        self.declare_parameter(
            "canonical_white_mask_topic", "/perception/canonical_white_mask"
        )
        self.declare_parameter(
            "canonical_yellow_mask_topic", "/perception/canonical_yellow_mask"
        )
        self.declare_parameter(
            "canonical_tracking_debug_topic",
            "/perception/canonical_tracking_debug",
        )
        self.declare_parameter(
            "canonical_pregeometry_white_mask_topic",
            "/perception/canonical_pregeometry_white_mask",
        )
        self.declare_parameter(
            "canonical_pregeometry_yellow_mask_topic",
            "/perception/canonical_pregeometry_yellow_mask",
        )
        self.declare_parameter(
            "canonical_pretrack_white_mask_topic",
            "/perception/canonical_pretrack_white_mask",
        )
        self.declare_parameter(
            "canonical_pretrack_yellow_mask_topic",
            "/perception/canonical_pretrack_yellow_mask",
        )
        self.declare_parameter(
            "canonical_valid_mask_topic",
            "/perception/canonical_valid_mask",
        )
        self.declare_parameter(
            "canonical_metric_debug_topic",
            "/perception/canonical_metric_debug",
        )
        self.declare_parameter("base_frame_id", "base_footprint")
        self.declare_parameter("source_name", "xycar_camera_perception")
        self.declare_parameter("publish_rate_limit_hz", 15.0)
        self.declare_parameter("publish_empty_optional_topics", True)
        self.declare_parameter("roi_top_row", 0)
        self.declare_parameter("roi_bottom_row", 219)
        self.declare_parameter("row_step_px", 6)
        self.declare_parameter("image_center_x_px", -1.0)
        self.declare_parameter("projection_mode", "bev_homography")
        self.declare_parameter("horizon_row_px", 236.0)
        self.declare_parameter("vanishing_point_x_px", -1.0)
        self.declare_parameter("ipm_x_scale_m_px", 67.0)
        self.declare_parameter("ipm_y_scale_m_px", 0.18)
        self.declare_parameter("horizon_min_denom_px", 8.0)
        self.declare_parameter("min_projected_x_m", 0.20)
        self.declare_parameter("max_projected_x_m", 5.00)
        self.declare_parameter("calib_yaml", "")
        self.declare_parameter("enable_rectify", True)
        self.declare_parameter("rect_balance", 0.3)
        self.declare_parameter("src_tl_x_ratio", 0.39)
        self.declare_parameter("src_tr_x_ratio", 0.67)
        self.declare_parameter("src_br_x_ratio", 1.10)
        self.declare_parameter("src_bl_x_ratio", -0.10)
        self.declare_parameter("src_top_y_ratio", 0.48)
        self.declare_parameter("src_bottom_y_ratio", 0.72)
        self.declare_parameter("src_tl_y_ratio", -1.0)
        self.declare_parameter("src_tr_y_ratio", -1.0)
        self.declare_parameter("src_br_y_ratio", -1.0)
        self.declare_parameter("src_bl_y_ratio", -1.0)
        self.declare_parameter("bev_width", 640)
        self.declare_parameter("bev_height", 220)
        self.declare_parameter("dst_left_ratio", 0.10)
        self.declare_parameter("dst_right_ratio", 0.90)
        self.declare_parameter("dst_top_y_ratio", 0.0)
        self.declare_parameter("dst_bottom_y_ratio", 1.0)
        self.declare_parameter("lateral_m_per_px", 0.0022)
        self.declare_parameter("forward_m_per_px", 0.010)
        self.declare_parameter("bev_x_offset_m", 0.0)
        self.declare_parameter("bev_bottom_ignore_px", 28)
        self.declare_parameter("bev_border_gray", 70)
        self.declare_parameter("bev_valid_erode_px", 8)
        self.declare_parameter("bev_valid_lateral_margin_px", 0)
        self.declare_parameter("near_x_m", 0.20)
        self.declare_parameter("far_x_m", 2.20)
        self.declare_parameter("near_m_per_px", 0.0022)
        self.declare_parameter("far_m_per_px", 0.0065)
        self.declare_parameter("lane_width_m", 0.80)
        self.declare_parameter("min_cluster_width_px", 3)
        self.declare_parameter("max_cluster_gap_px", 8)
        self.declare_parameter("min_segment_points", 4)
        self.declare_parameter("min_centerline_points", 3)
        self.declare_parameter("white_s_max", 80)
        self.declare_parameter("white_v_min", 145)
        self.declare_parameter("yellow_h_min", 15)
        self.declare_parameter("yellow_h_max", 42)
        self.declare_parameter("yellow_s_min", 60)
        self.declare_parameter("yellow_v_min", 110)
        self.declare_parameter("morphology_kernel_px", 3)
        self.declare_parameter("centerline_mode", "lane_midline")
        self.declare_parameter("use_yellow_as_centerline", True)
        self.declare_parameter("centerline_polyfit_enabled", True)
        self.declare_parameter("centerline_polyfit_degree", 2)
        self.declare_parameter("centerline_sample_spacing_m", 0.05)
        self.declare_parameter("centerline_temporal_alpha", 0.45)
        self.declare_parameter("centerline_temporal_timeout_sec", 0.50)
        self.declare_parameter("canonical_width", 256)
        self.declare_parameter("canonical_height", 144)
        self.declare_parameter("canonical_lateral_range_m", 1.4)
        self.declare_parameter("canonical_forward_range_m", 1.5)
        self.declare_parameter("canonical_background_gray", 36)
        self.declare_parameter("canonical_line_width_px", 5)
        self.declare_parameter("canonical_white_s_max", 120)
        self.declare_parameter("canonical_white_v_min", 145)
        self.declare_parameter("canonical_white_v_floor", 70)
        self.declare_parameter("canonical_white_relative_delta", 9.0)
        self.declare_parameter("canonical_white_min_component_median_v", 0.0)
        self.declare_parameter("canonical_min_component_area_px", 8)
        self.declare_parameter("canonical_white_max_component_thickness_px", 0.0)
        self.declare_parameter("canonical_yellow_max_component_thickness_px", 0.0)
        self.declare_parameter("canonical_geometry_filter_enabled", False)
        self.declare_parameter("canonical_white_min_line_span_px", 14)
        self.declare_parameter("canonical_yellow_min_line_span_px", 7)
        self.declare_parameter("canonical_min_line_elongation", 1.8)
        self.declare_parameter("canonical_min_line_verticality", 0.30)
        self.declare_parameter("canonical_white_max_components_per_side", 1)
        self.declare_parameter("canonical_yellow_max_components", 5)
        self.declare_parameter("canonical_white_clutter_component_limit", 5)
        self.declare_parameter("canonical_white_max_mask_fraction", 0.0)
        self.declare_parameter("canonical_yellow_max_mask_fraction", 0.0)
        self.declare_parameter("canonical_max_line_fit_rmse_px", 0.0)
        self.declare_parameter(
            "canonical_geometry_redraw_fitted_lines", True
        )
        self.declare_parameter(
            "canonical_yolo_yellow_geometry_filter_enabled", True
        )
        self.declare_parameter("canonical_yolo_preserve_white_mask", False)
        self.declare_parameter("canonical_top_ignore_m", 0.0)
        self.declare_parameter("canonical_bottom_ignore_m", 0.08)
        self.declare_parameter("canonical_tracking_enabled", False)
        self.declare_parameter(
            "canonical_expected_half_lane_width_m", 0.412
        )
        self.declare_parameter("canonical_lane_width_tolerance_m", 0.18)
        self.declare_parameter(
            "canonical_min_white_yellow_offset_m", 0.18
        )
        self.declare_parameter("canonical_tracking_base_gate_m", 0.08)
        self.declare_parameter("canonical_tracking_max_gate_m", 0.25)
        self.declare_parameter("canonical_tracking_continuation_m", 0.38)
        self.declare_parameter("canonical_yellow_fit_gate_m", 0.0)
        self.declare_parameter("canonical_tracking_curvature_gate_gain", 1.0)
        self.declare_parameter(
            "canonical_tracking_curvature_max_extra_m", 0.10
        )
        self.declare_parameter("canonical_tracking_confirmation_frames", 1)
        self.declare_parameter("canonical_tracking_coast_sec", 0.0)
        self.declare_parameter("canonical_tracking_search_sec", 0.0)
        self.declare_parameter("canonical_tracking_smoothing_alpha", 1.0)
        self.declare_parameter("canonical_width_prediction_enabled", False)
        self.declare_parameter(
            "canonical_width_prediction_replacement_frames", 1
        )
        self.declare_parameter(
            "canonical_transverse_clutter_row_fraction", 0.0
        )
        self.declare_parameter("canonical_transverse_clutter_min_rows", 6)
        self.declare_parameter(
            "canonical_persistent_prediction_enabled", False
        )
        self.declare_parameter(
            "canonical_yolo_allow_unpaired_yellow", False
        )

        self.bridge = CvBridge()
        self.use_compressed_image = bool(
            self.get_parameter("use_compressed_image").value
        )
        self.lane_segmentation_backend = str(
            self.get_parameter("lane_segmentation_backend").value
        ).strip().lower()
        if self.lane_segmentation_backend not in ("color", "yolo"):
            raise ValueError(
                "lane_segmentation_backend must be either color or yolo"
            )
        self.base_frame_id = str(self.get_parameter("base_frame_id").value)
        self.source_name = str(self.get_parameter("source_name").value)
        self.publish_empty_optional_topics = bool(
            self.get_parameter("publish_empty_optional_topics").value
        )
        self.roi_top_row = int(self.get_parameter("roi_top_row").value)
        self.roi_bottom_row = int(self.get_parameter("roi_bottom_row").value)
        self.row_step_px = max(1, int(self.get_parameter("row_step_px").value))
        self.image_center_x_px = float(self.get_parameter("image_center_x_px").value)
        self.projection_mode = str(self.get_parameter("projection_mode").value)
        self.horizon_row_px = float(self.get_parameter("horizon_row_px").value)
        self.vanishing_point_x_px = float(self.get_parameter("vanishing_point_x_px").value)
        self.ipm_x_scale_m_px = float(self.get_parameter("ipm_x_scale_m_px").value)
        self.ipm_y_scale_m_px = float(self.get_parameter("ipm_y_scale_m_px").value)
        self.horizon_min_denom_px = float(self.get_parameter("horizon_min_denom_px").value)
        self.min_projected_x_m = float(self.get_parameter("min_projected_x_m").value)
        self.max_projected_x_m = float(self.get_parameter("max_projected_x_m").value)
        self.calib_yaml = str(self.get_parameter("calib_yaml").value)
        self.enable_rectify = bool(self.get_parameter("enable_rectify").value)
        self.rect_balance = float(self.get_parameter("rect_balance").value)
        self.src_tl_x_ratio = float(self.get_parameter("src_tl_x_ratio").value)
        self.src_tr_x_ratio = float(self.get_parameter("src_tr_x_ratio").value)
        self.src_br_x_ratio = float(self.get_parameter("src_br_x_ratio").value)
        self.src_bl_x_ratio = float(self.get_parameter("src_bl_x_ratio").value)
        self.src_top_y_ratio = float(self.get_parameter("src_top_y_ratio").value)
        self.src_bottom_y_ratio = float(self.get_parameter("src_bottom_y_ratio").value)
        self.src_tl_y_ratio = float(self.get_parameter("src_tl_y_ratio").value)
        self.src_tr_y_ratio = float(self.get_parameter("src_tr_y_ratio").value)
        self.src_br_y_ratio = float(self.get_parameter("src_br_y_ratio").value)
        self.src_bl_y_ratio = float(self.get_parameter("src_bl_y_ratio").value)
        self.bev_width = int(self.get_parameter("bev_width").value)
        self.bev_height = int(self.get_parameter("bev_height").value)
        self.dst_left_ratio = float(self.get_parameter("dst_left_ratio").value)
        self.dst_right_ratio = float(self.get_parameter("dst_right_ratio").value)
        self.dst_top_y_ratio = float(
            self.get_parameter("dst_top_y_ratio").value
        )
        self.dst_bottom_y_ratio = float(
            self.get_parameter("dst_bottom_y_ratio").value
        )
        self.lateral_m_per_px = float(self.get_parameter("lateral_m_per_px").value)
        self.forward_m_per_px = float(self.get_parameter("forward_m_per_px").value)
        self.bev_x_offset_m = float(self.get_parameter("bev_x_offset_m").value)
        self.bev_bottom_ignore_px = max(
            0, int(self.get_parameter("bev_bottom_ignore_px").value)
        )
        self.bev_border_gray = int(
            np.clip(self.get_parameter("bev_border_gray").value, 0, 255)
        )
        self.bev_valid_erode_px = max(
            0, int(self.get_parameter("bev_valid_erode_px").value)
        )
        self.bev_valid_lateral_margin_px = max(
            0, int(self.get_parameter("bev_valid_lateral_margin_px").value)
        )
        self.near_x_m = float(self.get_parameter("near_x_m").value)
        self.far_x_m = float(self.get_parameter("far_x_m").value)
        self.near_m_per_px = float(self.get_parameter("near_m_per_px").value)
        self.far_m_per_px = float(self.get_parameter("far_m_per_px").value)
        self.lane_width_m = float(self.get_parameter("lane_width_m").value)
        self.min_cluster_width_px = int(self.get_parameter("min_cluster_width_px").value)
        self.max_cluster_gap_px = int(self.get_parameter("max_cluster_gap_px").value)
        self.min_segment_points = int(self.get_parameter("min_segment_points").value)
        self.min_centerline_points = int(self.get_parameter("min_centerline_points").value)
        self.white_s_max = int(self.get_parameter("white_s_max").value)
        self.white_v_min = int(self.get_parameter("white_v_min").value)
        self.yellow_h_min = int(self.get_parameter("yellow_h_min").value)
        self.yellow_h_max = int(self.get_parameter("yellow_h_max").value)
        self.yellow_s_min = int(self.get_parameter("yellow_s_min").value)
        self.yellow_v_min = int(self.get_parameter("yellow_v_min").value)
        self.morphology_kernel_px = int(self.get_parameter("morphology_kernel_px").value)
        self.centerline_mode = str(self.get_parameter("centerline_mode").value)
        self.use_yellow_as_centerline = bool(
            self.get_parameter("use_yellow_as_centerline").value
        )
        self.centerline_polyfit_enabled = bool(
            self.get_parameter("centerline_polyfit_enabled").value
        )
        self.centerline_polyfit_degree = max(
            1, int(self.get_parameter("centerline_polyfit_degree").value)
        )
        self.centerline_sample_spacing_m = max(
            0.01, float(self.get_parameter("centerline_sample_spacing_m").value)
        )
        self.centerline_temporal_alpha = float(
            np.clip(self.get_parameter("centerline_temporal_alpha").value, 0.0, 1.0)
        )
        self.centerline_temporal_timeout_sec = max(
            0.0, float(self.get_parameter("centerline_temporal_timeout_sec").value)
        )
        self.canonical_width = int(self.get_parameter("canonical_width").value)
        self.canonical_height = int(self.get_parameter("canonical_height").value)
        self.canonical_lateral_range_m = float(
            self.get_parameter("canonical_lateral_range_m").value
        )
        self.canonical_forward_range_m = float(
            self.get_parameter("canonical_forward_range_m").value
        )
        self.canonical_background_gray = int(
            self.get_parameter("canonical_background_gray").value
        )
        self.canonical_line_width_px = int(
            self.get_parameter("canonical_line_width_px").value
        )
        self.canonical_white_s_max = int(
            self.get_parameter("canonical_white_s_max").value
        )
        self.canonical_white_v_min = int(
            self.get_parameter("canonical_white_v_min").value
        )
        self.canonical_white_v_floor = int(
            self.get_parameter("canonical_white_v_floor").value
        )
        self.canonical_white_relative_delta = float(
            self.get_parameter("canonical_white_relative_delta").value
        )
        self.canonical_white_min_component_median_v = float(
            self.get_parameter(
                "canonical_white_min_component_median_v"
            ).value
        )
        self.canonical_min_component_area_px = int(
            self.get_parameter("canonical_min_component_area_px").value
        )
        self.canonical_white_max_component_thickness_px = float(
            self.get_parameter("canonical_white_max_component_thickness_px").value
        )
        self.canonical_yellow_max_component_thickness_px = float(
            self.get_parameter("canonical_yellow_max_component_thickness_px").value
        )
        self.canonical_geometry_filter_enabled = bool(
            self.get_parameter("canonical_geometry_filter_enabled").value
        )
        self.canonical_white_min_line_span_px = int(
            self.get_parameter("canonical_white_min_line_span_px").value
        )
        self.canonical_yellow_min_line_span_px = int(
            self.get_parameter("canonical_yellow_min_line_span_px").value
        )
        self.canonical_min_line_elongation = float(
            self.get_parameter("canonical_min_line_elongation").value
        )
        self.canonical_min_line_verticality = float(
            self.get_parameter("canonical_min_line_verticality").value
        )
        self.canonical_white_max_components_per_side = int(
            self.get_parameter("canonical_white_max_components_per_side").value
        )
        self.canonical_yellow_max_components = int(
            self.get_parameter("canonical_yellow_max_components").value
        )
        self.canonical_white_clutter_component_limit = int(
            self.get_parameter("canonical_white_clutter_component_limit").value
        )
        self.canonical_white_max_mask_fraction = float(
            self.get_parameter("canonical_white_max_mask_fraction").value
        )
        self.canonical_yellow_max_mask_fraction = float(
            self.get_parameter("canonical_yellow_max_mask_fraction").value
        )
        self.canonical_max_line_fit_rmse_px = float(
            self.get_parameter("canonical_max_line_fit_rmse_px").value
        )
        self.canonical_geometry_redraw_fitted_lines = bool(
            self.get_parameter(
                "canonical_geometry_redraw_fitted_lines"
            ).value
        )
        self.canonical_yolo_yellow_geometry_filter_enabled = bool(
            self.get_parameter(
                "canonical_yolo_yellow_geometry_filter_enabled"
            ).value
        )
        self.canonical_yolo_preserve_white_mask = bool(
            self.get_parameter("canonical_yolo_preserve_white_mask").value
        )
        self.canonical_top_ignore_m = float(
            self.get_parameter("canonical_top_ignore_m").value
        )
        self.canonical_bottom_ignore_m = float(
            self.get_parameter("canonical_bottom_ignore_m").value
        )
        self.canonical_tracking_enabled = bool(
            self.get_parameter("canonical_tracking_enabled").value
        )
        self.canonical_expected_half_lane_width_m = float(
            self.get_parameter(
                "canonical_expected_half_lane_width_m"
            ).value
        )
        self.canonical_lane_width_tolerance_m = float(
            self.get_parameter("canonical_lane_width_tolerance_m").value
        )
        self.canonical_min_white_yellow_offset_m = float(
            self.get_parameter(
                "canonical_min_white_yellow_offset_m"
            ).value
        )
        self.canonical_tracking_base_gate_m = float(
            self.get_parameter("canonical_tracking_base_gate_m").value
        )
        self.canonical_tracking_max_gate_m = float(
            self.get_parameter("canonical_tracking_max_gate_m").value
        )
        self.canonical_tracking_continuation_m = float(
            self.get_parameter("canonical_tracking_continuation_m").value
        )
        self.canonical_yellow_fit_gate_m = float(
            self.get_parameter("canonical_yellow_fit_gate_m").value
        )
        self.canonical_tracking_curvature_gate_gain = float(
            self.get_parameter(
                "canonical_tracking_curvature_gate_gain"
            ).value
        )
        self.canonical_tracking_curvature_max_extra_m = float(
            self.get_parameter(
                "canonical_tracking_curvature_max_extra_m"
            ).value
        )
        self.canonical_tracking_confirmation_frames = int(
            self.get_parameter(
                "canonical_tracking_confirmation_frames"
            ).value
        )
        self.canonical_tracking_coast_sec = float(
            self.get_parameter("canonical_tracking_coast_sec").value
        )
        self.canonical_tracking_search_sec = float(
            self.get_parameter("canonical_tracking_search_sec").value
        )
        self.canonical_tracking_smoothing_alpha = float(
            self.get_parameter(
                "canonical_tracking_smoothing_alpha"
            ).value
        )
        self.canonical_width_prediction_enabled = bool(
            self.get_parameter("canonical_width_prediction_enabled").value
        )
        self.canonical_width_prediction_replacement_frames = int(
            self.get_parameter(
                "canonical_width_prediction_replacement_frames"
            ).value
        )
        self.canonical_transverse_clutter_row_fraction = float(
            self.get_parameter(
                "canonical_transverse_clutter_row_fraction"
            ).value
        )
        self.canonical_transverse_clutter_min_rows = int(
            self.get_parameter(
                "canonical_transverse_clutter_min_rows"
            ).value
        )
        self.canonical_persistent_prediction_enabled = bool(
            self.get_parameter(
                "canonical_persistent_prediction_enabled"
            ).value
        )
        self.canonical_yolo_allow_unpaired_yellow = bool(
            self.get_parameter(
                "canonical_yolo_allow_unpaired_yellow"
            ).value
        )

        rate_limit_hz = float(self.get_parameter("publish_rate_limit_hz").value)
        self.min_publish_period = 0.0 if rate_limit_hz <= 0.0 else 1.0 / rate_limit_hz
        self.last_publish_wall_time = 0.0
        self.processed_image_count = 0
        self.detection_id = 1
        self.K: np.ndarray | None = None
        self.D: np.ndarray | None = None
        self.calib_size: tuple[int, int] | None = None
        self.distortion_model = "fisheye"
        self.K_rect: np.ndarray | None = None
        self.rect_map1: np.ndarray | None = None
        self.rect_map2: np.ndarray | None = None
        self.rect_size: tuple[int, int] | None = None
        self.M: np.ndarray | None = None
        self.M_inv: np.ndarray | None = None
        self.homography_source_polygon: np.ndarray | None = None
        self.homography_input_shape: tuple[int, int] | None = None
        self.homography_output_shape: tuple[int, int] | None = None
        self.current_projection_height = self.bev_height
        self.current_bev_valid_mask: np.ndarray | None = None
        self.current_rectified_image: np.ndarray | None = None
        self.current_yolo_bev_white_mask: np.ndarray | None = None
        self.current_yolo_bev_yellow_mask: np.ndarray | None = None
        self.current_yolo_debug_image: np.ndarray | None = None
        self.current_canonical_tracking_debug: np.ndarray | None = None
        self.current_canonical_pregeometry_white: np.ndarray | None = None
        self.current_canonical_pregeometry_yellow: np.ndarray | None = None
        self.current_canonical_pretrack_white: np.ndarray | None = None
        self.current_canonical_pretrack_yellow: np.ndarray | None = None
        self.current_canonical_valid_mask: np.ndarray | None = None
        self.current_canonical_metric_debug: np.ndarray | None = None
        self.previous_centerline_points: list[Point] = []
        self.previous_centerline_wall_time = 0.0
        self.canonical_lane_tracker = (
            CanonicalLaneTracker(
                width=self.canonical_width,
                height=self.canonical_height,
                lateral_range_m=self.canonical_lateral_range_m,
                forward_range_m=self.canonical_forward_range_m,
                expected_half_lane_width_m=(
                    self.canonical_expected_half_lane_width_m
                ),
                lane_width_tolerance_m=(
                    self.canonical_lane_width_tolerance_m
                ),
                min_white_yellow_offset_m=(
                    self.canonical_min_white_yellow_offset_m
                ),
                base_search_gate_m=self.canonical_tracking_base_gate_m,
                max_search_gate_m=self.canonical_tracking_max_gate_m,
                continuation_distance_m=(
                    self.canonical_tracking_continuation_m
                ),
                yellow_fit_gate_m=self.canonical_yellow_fit_gate_m,
                curvature_gate_gain=(
                    self.canonical_tracking_curvature_gate_gain
                ),
                curvature_max_extra_m=(
                    self.canonical_tracking_curvature_max_extra_m
                ),
                confirmation_frames=(
                    self.canonical_tracking_confirmation_frames
                ),
                coast_sec=self.canonical_tracking_coast_sec,
                search_sec=self.canonical_tracking_search_sec,
                smoothing_alpha=self.canonical_tracking_smoothing_alpha,
                width_prediction_enabled=(
                    self.canonical_width_prediction_enabled
                ),
                width_prediction_replacement_frames=(
                    self.canonical_width_prediction_replacement_frames
                ),
                transverse_clutter_row_fraction=(
                    self.canonical_transverse_clutter_row_fraction
                ),
                transverse_clutter_min_rows=(
                    self.canonical_transverse_clutter_min_rows
                ),
                persistent_prediction_enabled=(
                    self.canonical_persistent_prediction_enabled
                ),
                allow_unpaired_yellow=(
                    self.lane_segmentation_backend == "yolo"
                    and self.canonical_yolo_allow_unpaired_yellow
                ),
                line_width_px=self.canonical_line_width_px,
                background_gray=self.canonical_background_gray,
            )
            if self.canonical_tracking_enabled
            else None
        )
        self.load_calib_yaml()
        self.yolo_lane_segmenter = None
        if self.lane_segmentation_backend == "yolo":
            self.yolo_lane_segmenter = YoloLaneSegmenter(
                str(self.get_parameter("yolo_model_path").value),
                device=str(self.get_parameter("yolo_device").value),
                confidence=float(
                    self.get_parameter("yolo_confidence").value
                ),
                iou=float(self.get_parameter("yolo_iou").value),
                image_size=int(
                    self.get_parameter("yolo_image_size").value
                ),
                max_detections=int(
                    self.get_parameter("yolo_max_detections").value
                ),
                white_class_id=int(
                    self.get_parameter("yolo_white_class_id").value
                ),
                yellow_class_id=int(
                    self.get_parameter("yolo_yellow_class_id").value
                ),
                cpu_threads=int(
                    self.get_parameter("yolo_cpu_threads").value
                ),
                retina_masks=bool(
                    self.get_parameter("yolo_retina_masks").value
                ),
            )
            self.get_logger().info(
                "YOLO lane segmentation loaded: "
                f"{self.yolo_lane_segmenter.model_path}, "
                f"classes={self.yolo_lane_segmenter.class_names}, "
                f"device={self.yolo_lane_segmenter.device}, "
                f"image_size={self.yolo_lane_segmenter.image_size}, "
                f"cpu_threads={self.yolo_lane_segmenter.cpu_threads}"
            )

        self.road_segments_pub = self.create_publisher(
            RoadSegmentArray,
            str(self.get_parameter("road_segments_topic").value),
            10,
        )
        self.centerline_pub = self.create_publisher(
            Centerline,
            str(self.get_parameter("centerline_topic").value),
            10,
        )
        self.objects_pub = self.create_publisher(
            PerceptionObjectArray,
            str(self.get_parameter("objects_topic").value),
            10,
        )
        self.traffic_lights_pub = self.create_publisher(
            TrafficLightObservationArray,
            str(self.get_parameter("traffic_lights_topic").value),
            10,
        )
        self.rectified_image_pub = self.create_publisher(
            Image,
            str(self.get_parameter("rectified_image_topic").value),
            10,
        )
        self.yolo_debug_image_pub = self.create_publisher(
            Image,
            str(self.get_parameter("yolo_debug_image_topic").value),
            10,
        )
        self.debug_image_pub = self.create_publisher(
            Image,
            str(self.get_parameter("debug_image_topic").value),
            10,
        )
        self.debug_markers_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("debug_markers_topic").value),
            10,
        )
        self.canonical_road_pub = self.create_publisher(
            Image,
            str(self.get_parameter("canonical_road_topic").value),
            10,
        )
        self.canonical_white_mask_pub = self.create_publisher(
            Image,
            str(self.get_parameter("canonical_white_mask_topic").value),
            10,
        )
        self.canonical_yellow_mask_pub = self.create_publisher(
            Image,
            str(self.get_parameter("canonical_yellow_mask_topic").value),
            10,
        )
        self.canonical_tracking_debug_pub = self.create_publisher(
            Image,
            str(
                self.get_parameter(
                    "canonical_tracking_debug_topic"
                ).value
            ),
            10,
        )
        self.canonical_pregeometry_white_pub = self.create_publisher(
            Image,
            str(
                self.get_parameter(
                    "canonical_pregeometry_white_mask_topic"
                ).value
            ),
            10,
        )
        self.canonical_pregeometry_yellow_pub = self.create_publisher(
            Image,
            str(
                self.get_parameter(
                    "canonical_pregeometry_yellow_mask_topic"
                ).value
            ),
            10,
        )
        self.canonical_pretrack_white_pub = self.create_publisher(
            Image,
            str(
                self.get_parameter(
                    "canonical_pretrack_white_mask_topic"
                ).value
            ),
            10,
        )
        self.canonical_pretrack_yellow_pub = self.create_publisher(
            Image,
            str(
                self.get_parameter(
                    "canonical_pretrack_yellow_mask_topic"
                ).value
            ),
            10,
        )
        self.canonical_valid_mask_pub = self.create_publisher(
            Image,
            str(self.get_parameter("canonical_valid_mask_topic").value),
            10,
        )
        self.canonical_metric_debug_pub = self.create_publisher(
            Image,
            str(self.get_parameter("canonical_metric_debug_topic").value),
            10,
        )
        image_topic = str(self.get_parameter("image_topic").value)
        camera_qos = QoSProfile(
            history=qos_profile_sensor_data.history,
            depth=1,
            reliability=qos_profile_sensor_data.reliability,
            durability=qos_profile_sensor_data.durability,
        )
        if self.use_compressed_image:
            self.image_sub = self.create_subscription(
                CompressedImage,
                image_topic,
                self.on_compressed_image,
                camera_qos,
            )
            transport = "sensor_msgs/CompressedImage"
        else:
            self.image_sub = self.create_subscription(
                Image,
                image_topic,
                self.on_image,
                camera_qos,
            )
            transport = "sensor_msgs/Image"
        self.get_logger().info(
            f"camera perception ready: {image_topic} ({transport}) -> "
            f"/perception/road_segments, /perception/centerline, "
            f"projection_mode={self.projection_mode}, "
            f"lane_backend={self.lane_segmentation_backend}"
        )

    def next_detection_id(self) -> int:
        value = self.detection_id
        self.detection_id += 1
        return value

    def on_image(self, msg: Image) -> None:
        if not self.should_process_image():
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"failed to convert image: {exc}", throttle_duration_sec=2.0)
            return
        self.process_image(image, msg.header)

    def on_compressed_image(self, msg: CompressedImage) -> None:
        if not self.should_process_image():
            return
        image = decode_compressed_image(msg.data)
        if image is None:
            self.get_logger().warn(
                "failed to decode compressed image",
                throttle_duration_sec=2.0,
            )
            return
        self.process_image(image, msg.header)

    def should_process_image(self) -> bool:
        now = time.monotonic()
        if now - self.last_publish_wall_time < self.min_publish_period:
            return False
        self.last_publish_wall_time = now
        return True

    def process_image(self, image: np.ndarray, input_header) -> None:
        process_started = time.monotonic()
        header = RoadSegmentArray().header
        header.stamp = (
            input_header.stamp
            if input_header.stamp.sec or input_header.stamp.nanosec
            else self.get_clock().now().to_msg()
        )
        header.frame_id = self.base_frame_id

        (
            road_segments,
            centerline,
            debug,
            canonical,
            canonical_white,
            canonical_yellow,
        ) = self.detect_lanes(image, header)
        self.road_segments_pub.publish(road_segments)
        self.centerline_pub.publish(centerline)

        canonical_msg = self.bridge.cv2_to_imgmsg(canonical, encoding="bgr8")
        canonical_msg.header = header
        self.canonical_road_pub.publish(canonical_msg)
        white_msg = self.bridge.cv2_to_imgmsg(canonical_white, encoding="mono8")
        white_msg.header = header
        self.canonical_white_mask_pub.publish(white_msg)
        yellow_msg = self.bridge.cv2_to_imgmsg(canonical_yellow, encoding="mono8")
        yellow_msg.header = header
        self.canonical_yellow_mask_pub.publish(yellow_msg)

        stage_masks = (
            (
                self.canonical_pregeometry_white_pub,
                self.current_canonical_pregeometry_white,
            ),
            (
                self.canonical_pregeometry_yellow_pub,
                self.current_canonical_pregeometry_yellow,
            ),
            (
                self.canonical_pretrack_white_pub,
                self.current_canonical_pretrack_white,
            ),
            (
                self.canonical_pretrack_yellow_pub,
                self.current_canonical_pretrack_yellow,
            ),
            (
                self.canonical_valid_mask_pub,
                self.current_canonical_valid_mask,
            ),
        )
        for publisher, stage_mask in stage_masks:
            if publisher.get_subscription_count() <= 0 or stage_mask is None:
                continue
            stage_msg = self.bridge.cv2_to_imgmsg(stage_mask, encoding="mono8")
            stage_msg.header = header
            publisher.publish(stage_msg)

        if (
            self.canonical_metric_debug_pub.get_subscription_count() > 0
            and self.current_canonical_metric_debug is not None
        ):
            metric_debug_msg = self.bridge.cv2_to_imgmsg(
                self.current_canonical_metric_debug,
                encoding="bgr8",
            )
            metric_debug_msg.header = header
            self.canonical_metric_debug_pub.publish(metric_debug_msg)

        if (
            self.canonical_tracking_debug_pub.get_subscription_count() > 0
            and self.current_canonical_tracking_debug is not None
        ):
            tracking_msg = self.bridge.cv2_to_imgmsg(
                self.current_canonical_tracking_debug,
                encoding="bgr8",
            )
            tracking_msg.header = header
            self.canonical_tracking_debug_pub.publish(tracking_msg)

        if (
            self.rectified_image_pub.get_subscription_count() > 0
            and self.current_rectified_image is not None
        ):
            rectified_msg = self.bridge.cv2_to_imgmsg(
                self.current_rectified_image, encoding="bgr8"
            )
            rectified_msg.header = header
            self.rectified_image_pub.publish(rectified_msg)

        if (
            self.yolo_debug_image_pub.get_subscription_count() > 0
            and self.current_yolo_debug_image is not None
        ):
            yolo_debug_msg = self.bridge.cv2_to_imgmsg(
                self.current_yolo_debug_image, encoding="bgr8"
            )
            yolo_debug_msg.header = header
            self.yolo_debug_image_pub.publish(yolo_debug_msg)

        if self.publish_empty_optional_topics:
            objects = PerceptionObjectArray()
            objects.header = header
            self.objects_pub.publish(objects)

            traffic_lights = TrafficLightObservationArray()
            traffic_lights.header = header
            self.traffic_lights_pub.publish(traffic_lights)

        if self.debug_image_pub.get_subscription_count() > 0:
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = header
            self.debug_image_pub.publish(debug_msg)

        if self.debug_markers_pub.get_subscription_count() > 0:
            self.debug_markers_pub.publish(self.build_markers(header, road_segments, centerline))
        self.processed_image_count += 1
        if self.processed_image_count == 1:
            elapsed_ms = (time.monotonic() - process_started) * 1000.0
            self.get_logger().info(
                f"first camera frame processed in {elapsed_ms:.1f} ms"
            )

    def detect_lanes(
        self, image: np.ndarray, header
    ) -> tuple[
        RoadSegmentArray,
        Centerline,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        image = self.prepare_projection_image(image)
        height, width = image.shape[:2]
        self.current_projection_height = height
        center_x = self.image_center_x_px if self.image_center_x_px >= 0.0 else width * 0.5
        roi_top = max(0, min(height - 1, self.roi_top_row))
        roi_bottom = max(roi_top + 1, min(height - 1, self.roi_bottom_row))
        roi_bottom = max(roi_top + 1, roi_bottom - self.bev_bottom_ignore_px)

        if self.lane_segmentation_backend == "yolo":
            if (
                self.current_yolo_bev_white_mask is None
                or self.current_yolo_bev_yellow_mask is None
            ):
                raise RuntimeError("YOLO BEV masks were not generated")
            white_mask = self.current_yolo_bev_white_mask.copy()
            yellow_mask = self.current_yolo_bev_yellow_mask.copy()
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(
                hsv,
                np.array([0, 0, self.white_v_min], dtype=np.uint8),
                np.array([180, self.white_s_max, 255], dtype=np.uint8),
            )
            yellow_mask = cv2.inRange(
                hsv,
                np.array(
                    [
                        self.yellow_h_min,
                        self.yellow_s_min,
                        self.yellow_v_min,
                    ],
                    dtype=np.uint8,
                ),
                np.array([self.yellow_h_max, 255, 255], dtype=np.uint8),
            )
        valid_mask = self.current_bev_valid_mask
        if valid_mask is not None and valid_mask.shape == white_mask.shape:
            white_mask = cv2.bitwise_and(white_mask, valid_mask)
            yellow_mask = cv2.bitwise_and(yellow_mask, valid_mask)
        white_mask[:roi_top, :] = 0
        white_mask[roi_bottom + 1:, :] = 0
        yellow_mask[:roi_top, :] = 0
        yellow_mask[roi_bottom + 1:, :] = 0

        preserve_yolo_white = (
            self.lane_segmentation_backend == "yolo"
            and self.canonical_yolo_preserve_white_mask
        )
        if self.morphology_kernel_px > 1:
            kernel = np.ones((self.morphology_kernel_px, self.morphology_kernel_px), np.uint8)
            if not preserve_yolo_white:
                white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
                white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

        left_points: list[Point] = []
        right_points: list[Point] = []
        yellow_points: list[Point] = []
        debug = image.copy()
        if self.projection_mode == "bev_homography":
            debug = self.draw_bev_guides(debug)
        cv2.rectangle(debug, (0, roi_top), (width - 1, roi_bottom), (80, 80, 80), 1)

        for row in range(roi_bottom, roi_top - 1, -self.row_step_px):
            white_clusters = self.row_clusters(white_mask[row])
            yellow_clusters = self.row_clusters(yellow_mask[row])

            left_cluster = self.closest_left_cluster(white_clusters, center_x)
            right_cluster = self.closest_right_cluster(white_clusters, center_x)
            yellow_cluster = self.closest_center_cluster(yellow_clusters, center_x)

            if left_cluster is not None:
                point = self.pixel_to_vehicle(left_cluster, row, center_x, roi_top, roi_bottom)
                if point is not None:
                    left_points.append(point)
                    cv2.circle(debug, (int(left_cluster), row), 3, (255, 255, 255), -1)
            if right_cluster is not None:
                point = self.pixel_to_vehicle(right_cluster, row, center_x, roi_top, roi_bottom)
                if point is not None:
                    right_points.append(point)
                    cv2.circle(debug, (int(right_cluster), row), 3, (255, 255, 255), -1)
            if yellow_cluster is not None:
                point = self.pixel_to_vehicle(yellow_cluster, row, center_x, roi_top, roi_bottom)
                if point is not None:
                    yellow_points.append(point)
                    cv2.circle(debug, (int(yellow_cluster), row), 3, (0, 220, 255), -1)

        left_points = self.smooth_points(left_points)
        right_points = self.smooth_points(right_points)
        yellow_points = self.smooth_points(yellow_points)

        road_segments = RoadSegmentArray()
        road_segments.header = header
        left_segment = self.make_segment(RoadSegment.TYPE_WHSOL, left_points)
        right_segment = self.make_segment(RoadSegment.TYPE_WHSOL, right_points)
        yellow_segment = self.make_segment(RoadSegment.TYPE_YEDOT, yellow_points)
        for segment in (left_segment, right_segment, yellow_segment):
            if segment is not None:
                road_segments.segments.append(segment)

        centerline = self.build_centerline(header, left_segment, right_segment, yellow_segment)
        self.draw_centerline_on_debug(debug, centerline, center_x, roi_top, roi_bottom)
        canonical_common = dict(
            valid_mask=valid_mask,
            lateral_m_per_px=self.lateral_m_per_px,
            forward_m_per_px=self.forward_m_per_px,
            lateral_range_m=self.canonical_lateral_range_m,
            forward_range_m=self.canonical_forward_range_m,
            output_width=self.canonical_width,
            output_height=self.canonical_height,
            background_gray=self.canonical_background_gray,
            line_width_px=self.canonical_line_width_px,
            min_component_area_px=self.canonical_min_component_area_px,
            white_max_component_thickness_px=(
                self.canonical_white_max_component_thickness_px
            ),
            yellow_max_component_thickness_px=(
                self.canonical_yellow_max_component_thickness_px
            ),
            geometry_filter_enabled=self.canonical_geometry_filter_enabled,
            white_min_line_span_px=self.canonical_white_min_line_span_px,
            yellow_min_line_span_px=self.canonical_yellow_min_line_span_px,
            min_line_elongation=self.canonical_min_line_elongation,
            min_line_verticality=self.canonical_min_line_verticality,
            white_max_components_per_side=(
                self.canonical_white_max_components_per_side
            ),
            yellow_max_components=self.canonical_yellow_max_components,
            white_clutter_component_limit=(
                self.canonical_white_clutter_component_limit
            ),
            white_max_mask_fraction=self.canonical_white_max_mask_fraction,
            yellow_max_mask_fraction=self.canonical_yellow_max_mask_fraction,
            max_line_fit_rmse_px=self.canonical_max_line_fit_rmse_px,
            geometry_redraw_fitted_lines=(
                self.canonical_geometry_redraw_fitted_lines
            ),
            yellow_geometry_filter_enabled=(
                self.lane_segmentation_backend != "yolo"
                or self.canonical_yolo_yellow_geometry_filter_enabled
            ),
            top_ignore_m=self.canonical_top_ignore_m,
            bottom_ignore_m=self.canonical_bottom_ignore_m,
            return_stages=True,
        )
        if self.lane_segmentation_backend == "yolo":
            canonical_stages = make_canonical_road_image_from_masks(
                white_mask,
                yellow_mask,
                preserve_white_mask=preserve_yolo_white,
                **canonical_common,
            )
        else:
            canonical_stages = make_canonical_road_image(
                image,
                white_s_max=self.canonical_white_s_max,
                white_v_min=self.canonical_white_v_min,
                white_v_floor=self.canonical_white_v_floor,
                white_relative_delta=(
                    self.canonical_white_relative_delta
                ),
                yellow_h_min=self.yellow_h_min,
                yellow_h_max=self.yellow_h_max,
                yellow_s_min=self.yellow_s_min,
                yellow_v_min=self.yellow_v_min,
                white_min_component_median_v=(
                    self.canonical_white_min_component_median_v
                ),
                **canonical_common,
            )
        if not isinstance(canonical_stages, CanonicalRoadStages):
            raise RuntimeError("canonical stage output was not requested")
        canonical = canonical_stages.road_image
        canonical_white = canonical_stages.white_mask
        canonical_yellow = canonical_stages.yellow_mask
        self.current_canonical_pregeometry_white = (
            canonical_stages.pre_geometry_white_mask
        )
        self.current_canonical_pregeometry_yellow = (
            canonical_stages.pre_geometry_yellow_mask
        )
        self.current_canonical_pretrack_white = canonical_white.copy()
        self.current_canonical_pretrack_yellow = canonical_yellow.copy()
        self.current_canonical_valid_mask = canonical_stages.valid_mask
        self.current_canonical_metric_debug = self.make_canonical_metric_debug(
            canonical_white,
            canonical_yellow,
            canonical_stages.valid_mask,
        )
        if self.canonical_lane_tracker is not None:
            timestamp_sec = (
                float(header.stamp.sec)
                + float(header.stamp.nanosec) * 1.0e-9
            )
            if timestamp_sec <= 0.0:
                timestamp_sec = time.monotonic()
            tracked = self.canonical_lane_tracker.update(
                canonical_white,
                canonical_yellow,
                timestamp_sec,
            )
            self.current_canonical_tracking_debug = tracked.debug_image
            if not preserve_yolo_white:
                canonical = tracked.road_image
                canonical_white = tracked.white_mask
                canonical_yellow = tracked.yellow_mask
        else:
            self.current_canonical_tracking_debug = None
        return (
            road_segments,
            centerline,
            debug,
            canonical,
            canonical_white,
            canonical_yellow,
        )

    def make_canonical_metric_debug(
        self,
        white_mask: np.ndarray,
        yellow_mask: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        debug = np.full(
            (self.canonical_height, self.canonical_width, 3),
            self.canonical_background_gray,
            dtype=np.uint8,
        )
        debug[valid_mask == 0] = (30, 18, 18)
        debug[white_mask > 0] = (255, 255, 255)
        debug[yellow_mask > 0] = (0, 220, 255)

        meters_per_row = (
            self.canonical_forward_range_m / max(1, self.canonical_height)
        )
        for distance_m in (0.5, 1.0, 1.5):
            row = int(
                round(
                    self.canonical_height
                    - distance_m / meters_per_row
                )
            )
            row = int(np.clip(row, 0, self.canonical_height - 1))
            cv2.line(
                debug,
                (0, row),
                (self.canonical_width - 1, row),
                (80, 150, 80),
                1,
                cv2.LINE_8,
            )
            text_row = min(self.canonical_height - 3, max(11, row + 11))
            cv2.putText(
                debug,
                f"{distance_m:.1f}m",
                (3, text_row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (100, 230, 100),
                1,
                cv2.LINE_AA,
            )
        return debug

    def row_clusters(self, row_mask: np.ndarray) -> list[float]:
        xs = np.flatnonzero(row_mask)
        if xs.size == 0:
            return []
        clusters: list[np.ndarray] = []
        start = 0
        for index in range(1, xs.size):
            if xs[index] - xs[index - 1] > self.max_cluster_gap_px:
                clusters.append(xs[start:index])
                start = index
        clusters.append(xs[start:])

        centers = []
        for cluster in clusters:
            if cluster.size < self.min_cluster_width_px:
                continue
            if int(cluster[-1] - cluster[0] + 1) < self.min_cluster_width_px:
                continue
            centers.append(float(np.mean(cluster)))
        return centers

    def closest_left_cluster(self, clusters: list[float], center_x: float) -> float | None:
        left = [cluster for cluster in clusters if cluster < center_x - 4.0]
        return max(left) if left else None

    def closest_right_cluster(self, clusters: list[float], center_x: float) -> float | None:
        right = [cluster for cluster in clusters if cluster > center_x + 4.0]
        return min(right) if right else None

    def closest_center_cluster(self, clusters: list[float], center_x: float) -> float | None:
        if not clusters:
            return None
        return min(clusters, key=lambda value: abs(value - center_x))

    def load_calib_yaml(self) -> None:
        if not self.calib_yaml:
            return
        try:
            import yaml

            with open(self.calib_yaml, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
            if "camera_matrix" in data:
                self.K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
            elif "K" in data:
                self.K = np.array(data["K"], dtype=np.float64).reshape(3, 3)
            else:
                raise RuntimeError("camera_matrix or K is missing")

            if "distortion_coefficients" in data:
                self.D = np.array(data["distortion_coefficients"]["data"], dtype=np.float64)
            elif "D" in data:
                self.D = np.array(data["D"], dtype=np.float64)
            else:
                self.D = np.zeros(4, dtype=np.float64)

            self.distortion_model = str(data.get("distortion_model", "fisheye")).lower()
            calibration_width = int(data.get("image_width") or 0)
            calibration_height = int(data.get("image_height") or 0)
            self.calib_size = (
                (calibration_width, calibration_height)
                if calibration_width > 0 and calibration_height > 0
                else None
            )
            self.get_logger().info(f"loaded camera calibration yaml: {self.calib_yaml}")
        except Exception as exc:
            self.get_logger().warn(f"failed to load calib_yaml: {exc}")
            self.K = None
            self.D = None
            self.calib_size = None

    def build_rectify_map(self, width: int, height: int) -> bool:
        if self.K is None or self.D is None:
            return False

        size = (width, height)
        camera_matrix = scale_camera_matrix(self.K, self.calib_size, size)
        R = np.eye(3, dtype=np.float64)
        try:
            if "fisheye" in self.distortion_model or "equidistant" in self.distortion_model:
                d4 = np.zeros((4, 1), dtype=np.float64)
                count = min(4, len(self.D))
                d4[:count, 0] = self.D[:count]
                self.K_rect = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    camera_matrix,
                    d4,
                    size,
                    R,
                    balance=self.rect_balance,
                    new_size=size,
                )
                self.rect_map1, self.rect_map2 = cv2.fisheye.initUndistortRectifyMap(
                    camera_matrix,
                    d4,
                    R,
                    self.K_rect,
                    size,
                    cv2.CV_32FC1,
                )
            else:
                self.K_rect, _ = cv2.getOptimalNewCameraMatrix(
                    camera_matrix,
                    self.D,
                    size,
                    alpha=self.rect_balance,
                    newImgSize=size,
                )
                self.rect_map1, self.rect_map2 = cv2.initUndistortRectifyMap(
                    camera_matrix,
                    self.D,
                    R,
                    self.K_rect,
                    size,
                    cv2.CV_32FC1,
                )
            self.rect_size = size
            self.get_logger().info(f"rectify map built: {size}, balance={self.rect_balance}")
            return True
        except cv2.error as exc:
            self.get_logger().warn(f"rectify map failed: {exc}")
            return False

    def rectify_image(self, image: np.ndarray) -> np.ndarray:
        if not self.enable_rectify:
            return image
        height, width = image.shape[:2]
        if self.rect_map1 is None or self.rect_map2 is None or self.rect_size != (width, height):
            if not self.build_rectify_map(width, height):
                return image
        return cv2.remap(
            image,
            self.rect_map1,
            self.rect_map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def build_homography(self, width: int, height: int) -> None:
        tl_x = self.src_tl_x_ratio * width
        tr_x = self.src_tr_x_ratio * width
        br_x = self.src_br_x_ratio * width
        bl_x = self.src_bl_x_ratio * width
        top_y = self.src_top_y_ratio * height
        bottom_y = self.src_bottom_y_ratio * height
        tl_y_ratio = getattr(self, "src_tl_y_ratio", -1.0)
        tr_y_ratio = getattr(self, "src_tr_y_ratio", -1.0)
        br_y_ratio = getattr(self, "src_br_y_ratio", -1.0)
        bl_y_ratio = getattr(self, "src_bl_y_ratio", -1.0)
        tl_y = (tl_y_ratio * height) if tl_y_ratio >= 0.0 else top_y
        tr_y = (tr_y_ratio * height) if tr_y_ratio >= 0.0 else top_y
        br_y = (br_y_ratio * height) if br_y_ratio >= 0.0 else bottom_y
        bl_y = (bl_y_ratio * height) if bl_y_ratio >= 0.0 else bottom_y
        dst_l = self.dst_left_ratio * self.bev_width
        dst_r = self.dst_right_ratio * self.bev_width
        dst_top = self.dst_top_y_ratio * self.bev_height
        dst_bottom = self.dst_bottom_y_ratio * self.bev_height

        src = np.float32(
            [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]]
        )
        dst = np.float32(
            [
                [dst_l, dst_top],
                [dst_r, dst_top],
                [dst_r, dst_bottom],
                [dst_l, dst_bottom],
            ]
        )
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
        self.homography_source_polygon = np.rint(src).astype(np.int32)
        self.homography_input_shape = (width, height)
        self.homography_output_shape = (self.bev_width, self.bev_height)

    def prepare_projection_image(self, image: np.ndarray) -> np.ndarray:
        if self.projection_mode != "bev_homography":
            self.current_rectified_image = image
            self.current_bev_valid_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            yolo_segmenter = getattr(self, "yolo_lane_segmenter", None)
            if yolo_segmenter is not None:
                (
                    self.current_yolo_bev_white_mask,
                    self.current_yolo_bev_yellow_mask,
                    self.current_yolo_debug_image,
                ) = yolo_segmenter.predict(
                    image,
                    render_debug=(
                        self.yolo_debug_image_pub.get_subscription_count() > 0
                    ),
                )
            return image

        rectified = self.rectify_image(image)
        self.current_rectified_image = rectified
        height, width = rectified.shape[:2]
        output_shape = (self.bev_width, self.bev_height)
        if (
            self.M is None
            or self.homography_input_shape != (width, height)
            or self.homography_output_shape != output_shape
        ):
            self.build_homography(width, height)
        source_valid = np.full(image.shape[:2], 255, dtype=np.uint8)
        if self.enable_rectify and self.rect_map1 is not None:
            source_valid = cv2.remap(
                source_valid,
                self.rect_map1,
                self.rect_map2,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        source_roi = np.zeros_like(source_valid)
        if self.homography_source_polygon is not None:
            cv2.fillConvexPoly(
                source_roi,
                self.homography_source_polygon,
                255,
                lineType=cv2.LINE_8,
            )
            source_valid = cv2.bitwise_and(source_valid, source_roi)
        valid_mask = cv2.warpPerspective(
            source_valid,
            self.M,
            output_shape,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        lateral_margin = max(
            0, int(getattr(self, "bev_valid_lateral_margin_px", 0))
        )
        if lateral_margin > 0:
            valid_mask = cv2.dilate(
                valid_mask,
                np.ones((1, lateral_margin * 2 + 1), dtype=np.uint8),
            )
        if self.bev_valid_erode_px > 0:
            size = self.bev_valid_erode_px * 2 + 1
            valid_mask = cv2.erode(valid_mask, np.ones((size, size), dtype=np.uint8))
        self.current_bev_valid_mask = valid_mask

        yolo_segmenter = getattr(self, "yolo_lane_segmenter", None)
        if yolo_segmenter is not None:
            source_white, source_yellow, yolo_debug = (
                yolo_segmenter.predict(
                    rectified,
                    render_debug=(
                        self.yolo_debug_image_pub.get_subscription_count() > 0
                    ),
                )
            )
            self.current_yolo_bev_white_mask = cv2.warpPerspective(
                source_white,
                self.M,
                output_shape,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            self.current_yolo_bev_yellow_mask = cv2.warpPerspective(
                source_yellow,
                self.M,
                output_shape,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            self.current_yolo_bev_white_mask = cv2.bitwise_and(
                self.current_yolo_bev_white_mask, valid_mask
            )
            self.current_yolo_bev_yellow_mask = cv2.bitwise_and(
                self.current_yolo_bev_yellow_mask, valid_mask
            )
            self.current_yolo_debug_image = yolo_debug

        border = (self.bev_border_gray,) * 3
        return cv2.warpPerspective(
            rectified,
            self.M,
            output_shape,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )

    def draw_bev_guides(self, image: np.ndarray) -> np.ndarray:
        out = image.copy()
        height, width = out.shape[:2]
        if self.lateral_m_per_px <= 0.0:
            return out
        center_x = width * 0.5

        def col_from_y_left(y_left_m: float) -> int:
            return int(round(center_x - y_left_m / self.lateral_m_per_px))

        for y_left, label, color in [
            (0.40, "left +0.40m", (255, 0, 0)),
            (0.00, "center 0.00m", (0, 255, 255)),
            (-0.40, "right -0.40m", (0, 0, 255)),
        ]:
            col = col_from_y_left(y_left)
            if 0 <= col < width:
                cv2.line(out, (col, 0), (col, height - 1), color, 1)
                cv2.putText(out, label, (col + 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if self.forward_m_per_px > 0.0:
            forward_range_m = height * self.forward_m_per_px
            distance_m = 0.5
            while distance_m <= forward_range_m + 1.0e-6:
                row = int(
                    round(
                        height
                        - (distance_m - self.bev_x_offset_m)
                        / self.forward_m_per_px
                    )
                )
                if 0 <= row < height:
                    cv2.line(
                        out,
                        (0, row),
                        (width - 1, row),
                        (70, 70, 70),
                        1,
                    )
                    cv2.putText(
                        out,
                        f"{distance_m:.1f}m",
                        (4, max(14, row - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (80, 210, 80),
                        1,
                        cv2.LINE_AA,
                    )
                distance_m += 0.5
        return out

    def pixel_to_vehicle(
        self,
        pixel_x: float,
        pixel_y: int,
        center_x: float,
        roi_top: int,
        roi_bottom: int,
    ) -> Point | None:
        if self.projection_mode == "bev_homography":
            return self.pixel_to_vehicle_bev(pixel_x, pixel_y, center_x)
        if self.projection_mode == "vanishing_point":
            return self.pixel_to_vehicle_vanishing_point(pixel_x, pixel_y, center_x)

        span = max(1, roi_bottom - roi_top)
        t = (roi_bottom - pixel_y) / float(span)
        t = max(0.0, min(1.0, t))
        x_m = self.near_x_m + t * (self.far_x_m - self.near_x_m)
        meters_per_pixel = self.near_m_per_px + t * (self.far_m_per_px - self.near_m_per_px)
        y_m = (center_x - pixel_x) * meters_per_pixel
        return make_point(x_m, y_m, 0.0)

    def pixel_to_vehicle_vanishing_point(
        self,
        pixel_x: float,
        pixel_y: int,
        center_x: float,
    ) -> Point | None:
        vanishing_x = self.vanishing_point_x_px
        if vanishing_x < 0.0:
            vanishing_x = center_x

        denom = float(pixel_y) - self.horizon_row_px
        if denom < self.horizon_min_denom_px:
            return None
        if self.ipm_x_scale_m_px <= 0.0 or self.ipm_y_scale_m_px <= 0.0:
            return None

        x_m = self.ipm_x_scale_m_px / denom
        if x_m < self.min_projected_x_m or x_m > self.max_projected_x_m:
            return None

        y_m = (vanishing_x - float(pixel_x)) * self.ipm_y_scale_m_px / denom
        return make_point(x_m, y_m, 0.0)

    def pixel_to_vehicle_bev(
        self,
        pixel_x: float,
        pixel_y: int,
        center_x: float,
    ) -> Point | None:
        if self.lateral_m_per_px <= 0.0 or self.forward_m_per_px <= 0.0:
            return None
        image_height = max(1, self.current_projection_height)
        x_m = self.bev_x_offset_m + (image_height - 1 - float(pixel_y)) * self.forward_m_per_px
        y_m = (center_x - float(pixel_x)) * self.lateral_m_per_px
        return make_point(x_m, y_m, 0.0)

    def smooth_points(self, points: list[Point]) -> list[Point]:
        if len(points) < 3:
            return sorted(points, key=lambda point: point.x)
        points = sorted(points, key=lambda point: point.x)
        ys = moving_average([point.y for point in points], window=3)
        return [make_point(point.x, y, 0.0) for point, y in zip(points, ys)]

    def make_segment(self, segment_type: int, points: list[Point]) -> RoadSegment | None:
        if len(points) < self.min_segment_points:
            return None
        segment = RoadSegment()
        segment.detection_id = self.next_detection_id()
        segment.track_id = 0
        segment.type = segment_type
        segment.points = points
        expected_count = max(1, int((self.roi_bottom_row - self.roi_top_row) / self.row_step_px))
        segment.confidence = float(min(1.0, 0.30 + 0.70 * len(points) / expected_count))
        segment.source = self.source_name
        return segment

    def build_centerline(
        self,
        header,
        left: RoadSegment | None,
        right: RoadSegment | None,
        yellow: RoadSegment | None,
    ) -> Centerline:
        centerline = Centerline()
        centerline.header = header
        centerline.track_id = 0
        centerline.source = self.source_name

        if self.centerline_mode == "lane_midline" and yellow is not None:
            centerline.points = self.best_yellow_boundary_midline(yellow, [left, right])
            if len(centerline.points) >= self.min_centerline_points:
                confidence_candidates = [
                    segment.confidence for segment in (left, right, yellow) if segment is not None
                ]
                centerline.confidence = (
                    float(min(confidence_candidates)) if confidence_candidates else 0.0
                )
        elif (
            self.use_yellow_as_centerline
            and yellow is not None
            and len(yellow.points) >= self.min_centerline_points
        ):
            centerline.points = list(yellow.points)
            centerline.confidence = float(yellow.confidence)

        if not centerline.points and left is not None and right is not None:
            centerline.points = self.midpoint_centerline_points(left.points, right.points)
            centerline.confidence = float(min(left.confidence, right.confidence))
        elif not centerline.points and left is not None:
            centerline.points = [
                make_point(point.x, point.y - self.lane_width_m * 0.5, 0.0)
                for point in left.points
            ]
            centerline.confidence = float(left.confidence * 0.70)
        elif not centerline.points and right is not None:
            centerline.points = [
                make_point(point.x, point.y + self.lane_width_m * 0.5, 0.0)
                for point in right.points
            ]
            centerline.confidence = float(right.confidence * 0.70)
        elif not centerline.points:
            centerline.confidence = 0.0

        centerline.points = self.fit_and_smooth_centerline(centerline.points)
        if len(centerline.points) < self.min_centerline_points:
            centerline.points = []
            centerline.confidence = 0.0
            return centerline

        centerline.detection_id = self.next_detection_id()
        return centerline

    def fit_and_smooth_centerline(self, points: list[Point]) -> list[Point]:
        if len(points) < self.min_centerline_points:
            return points

        source_points = points if self.centerline_polyfit_enabled else self.smooth_points(points)
        current = sorted(source_points, key=lambda point: point.x)
        if self.centerline_polyfit_enabled and len(current) >= 2:
            x_values = np.asarray([point.x for point in current], dtype=np.float64)
            y_values = np.asarray([point.y for point in current], dtype=np.float64)
            unique_x, unique_indices = np.unique(x_values, return_index=True)
            unique_y = y_values[unique_indices]
            if unique_x.size >= 2:
                degree = min(self.centerline_polyfit_degree, int(unique_x.size - 1))
                try:
                    coefficients = np.polyfit(unique_x, unique_y, degree)
                    span = float(unique_x[-1] - unique_x[0])
                    sample_count = max(
                        self.min_centerline_points,
                        min(
                            40,
                            int(math.ceil(span / self.centerline_sample_spacing_m)) + 1,
                        ),
                    )
                    sample_x = np.linspace(unique_x[0], unique_x[-1], sample_count)
                    sample_y = np.polyval(coefficients, sample_x)
                    current = [
                        make_point(x_value, y_value, 0.0)
                        for x_value, y_value in zip(sample_x, sample_y)
                        if np.isfinite(x_value) and np.isfinite(y_value)
                    ]
                except (TypeError, ValueError, np.linalg.LinAlgError):
                    pass

        now = time.monotonic()
        previous_is_fresh = bool(self.previous_centerline_points) and (
            now - self.previous_centerline_wall_time
            <= self.centerline_temporal_timeout_sec
        )
        if previous_is_fresh:
            alpha = self.centerline_temporal_alpha
            blended = []
            for point in current:
                previous_y = self.y_at_x(self.previous_centerline_points, point.x)
                y_value = (
                    point.y
                    if previous_y is None
                    else alpha * point.y + (1.0 - alpha) * previous_y
                )
                blended.append(make_point(point.x, y_value, 0.0))
            current = blended

        self.previous_centerline_points = [
            make_point(point.x, point.y, point.z) for point in current
        ]
        self.previous_centerline_wall_time = now
        return current

    def best_yellow_boundary_midline(
        self,
        yellow: RoadSegment,
        boundaries: list[RoadSegment | None],
    ) -> list[Point]:
        best_points: list[Point] = []
        best_score = -1.0
        for boundary in boundaries:
            if boundary is None:
                continue
            candidate = self.midpoint_centerline_points(yellow.points, boundary.points)
            if len(candidate) < self.min_centerline_points:
                continue
            score = self.midline_score(yellow.points, boundary.points, candidate)
            if score > best_score:
                best_points = candidate
                best_score = score
        return best_points

    def midline_score(
        self,
        yellow_points: list[Point],
        boundary_points: list[Point],
        midline_points: list[Point],
    ) -> float:
        width_errors = []
        for point in midline_points:
            yellow_y = self.y_at_x(yellow_points, point.x)
            boundary_y = self.y_at_x(boundary_points, point.x)
            if yellow_y is None or boundary_y is None:
                continue
            width_errors.append(abs(abs(yellow_y - boundary_y) - self.lane_width_m))
        mean_error = sum(width_errors) / max(1, len(width_errors))
        return float(len(midline_points) - 3.0 * mean_error)

    def midpoint_centerline_points(self, left: list[Point], right: list[Point]) -> list[Point]:
        left_bounds = self.x_bounds(left)
        right_bounds = self.x_bounds(right)
        if left_bounds is None or right_bounds is None:
            return []
        start_x = max(left_bounds[0], right_bounds[0])
        end_x = min(left_bounds[1], right_bounds[1])
        if end_x <= start_x:
            return []
        sample_count = max(3, min(24, int(math.ceil((end_x - start_x) / 0.15)) + 1))
        points = []
        for index in range(sample_count):
            ratio = index / float(sample_count - 1)
            target_x = start_x + (end_x - start_x) * ratio
            left_y = self.y_at_x(left, target_x)
            right_y = self.y_at_x(right, target_x)
            if left_y is None or right_y is None:
                continue
            points.append(make_point(target_x, (left_y + right_y) * 0.5, 0.0))
        return points

    def x_bounds(self, points: list[Point]) -> tuple[float, float] | None:
        if len(points) < 2:
            return None
        return min(point.x for point in points), max(point.x for point in points)

    def y_at_x(self, points: list[Point], target_x: float) -> float | None:
        if not points:
            return None
        points = sorted(points, key=lambda point: point.x)
        if target_x <= points[0].x:
            return points[0].y
        if target_x >= points[-1].x:
            return points[-1].y
        for start, end in zip(points, points[1:]):
            if start.x <= target_x <= end.x:
                dx = end.x - start.x
                if abs(dx) < 1.0e-6:
                    return start.y
                ratio = (target_x - start.x) / dx
                return start.y + (end.y - start.y) * ratio
        return None

    def draw_centerline_on_debug(
        self,
        image: np.ndarray,
        centerline: Centerline,
        center_x: float,
        roi_top: int,
        roi_bottom: int,
    ) -> None:
        for point in centerline.points:
            pixel = self.vehicle_to_pixel(point, center_x, roi_top, roi_bottom)
            if pixel is not None:
                cv2.circle(image, pixel, 3, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"centerline pts={len(centerline.points)} conf={centerline.confidence:.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    def vehicle_to_pixel(
        self,
        point: Point,
        center_x: float,
        roi_top: int,
        roi_bottom: int,
    ) -> tuple[int, int] | None:
        if self.projection_mode == "bev_homography":
            return self.vehicle_to_pixel_bev(point, center_x)
        if self.projection_mode == "vanishing_point":
            return self.vehicle_to_pixel_vanishing_point(point, center_x)

        if self.far_x_m <= self.near_x_m:
            return None
        t = (point.x - self.near_x_m) / (self.far_x_m - self.near_x_m)
        if t < 0.0 or t > 1.0:
            return None
        meters_per_pixel = self.near_m_per_px + t * (self.far_m_per_px - self.near_m_per_px)
        if meters_per_pixel <= 1.0e-6:
            return None
        pixel_y = int(round(roi_bottom - t * (roi_bottom - roi_top)))
        pixel_x = int(round(center_x - point.y / meters_per_pixel))
        return pixel_x, pixel_y

    def vehicle_to_pixel_vanishing_point(
        self,
        point: Point,
        center_x: float,
    ) -> tuple[int, int] | None:
        if point.x <= 1.0e-6 or self.ipm_x_scale_m_px <= 0.0 or self.ipm_y_scale_m_px <= 0.0:
            return None
        vanishing_x = self.vanishing_point_x_px
        if vanishing_x < 0.0:
            vanishing_x = center_x
        denom = self.ipm_x_scale_m_px / float(point.x)
        if denom < self.horizon_min_denom_px:
            return None
        pixel_y = int(round(self.horizon_row_px + denom))
        pixel_x = int(round(vanishing_x - float(point.y) * denom / self.ipm_y_scale_m_px))
        return pixel_x, pixel_y

    def vehicle_to_pixel_bev(self, point: Point, center_x: float) -> tuple[int, int] | None:
        if self.lateral_m_per_px <= 0.0 or self.forward_m_per_px <= 0.0:
            return None
        image_height = max(1, self.current_projection_height)
        pixel_y = int(
            round(
                (image_height - 1)
                - (point.x - self.bev_x_offset_m) / self.forward_m_per_px
            )
        )
        pixel_x = int(round(center_x - point.y / self.lateral_m_per_px))
        return pixel_x, pixel_y

    def build_markers(
        self,
        header,
        road_segments: RoadSegmentArray,
        centerline: Centerline,
    ) -> MarkerArray:
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.header = header
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        marker_id = 1
        for segment in road_segments.segments:
            marker = self.line_marker(header, marker_id, "road_segments", segment.points)
            if segment.type in {RoadSegment.TYPE_YESOL, RoadSegment.TYPE_YEDOT}:
                marker.color.r = 1.0
                marker.color.g = 0.75
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
            markers.markers.append(marker)
            marker_id += 1

        marker = self.line_marker(header, marker_id, "centerline", centerline.points)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.scale.x = 0.035
        markers.markers.append(marker)
        return markers

    def line_marker(self, header, marker_id: int, namespace: str, points: list[Point]) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.025
        marker.color.a = 1.0
        marker.points = points
        return marker


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
