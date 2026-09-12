from __future__ import annotations

import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray, String
import torch

from il_data_tools.runtime_preprocessing import (
    model_input_to_bgr,
    preprocess_bgr_image,
    preprocess_lidar_ranges,
)
from xycar_rl.policy_loader import load_steering_policy
from xycar_rl.policy_loader import load_camera_speed_policy
from xycar_rl.camera_speed_models import denormalize_speed_command
from xycar_rl.canonical_preview import CanonicalPreviewSteering
from xycar_rl.steering_stabilizer import (
    AdaptiveSteeringStabilizer,
    SteeringStabilizerConfig,
)


def message_stamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def front_obstacle_distance(
    scan: LaserScan,
    half_angle_rad: float,
) -> float:
    ranges = np.asarray(scan.ranges, dtype=np.float32)
    if ranges.size == 0:
        return float("inf")
    angles = float(scan.angle_min) + np.arange(ranges.size) * float(
        scan.angle_increment
    )
    wrapped = (angles + math.pi) % (2.0 * math.pi) - math.pi
    valid = (
        np.isfinite(ranges)
        & (ranges >= float(scan.range_min))
        & (ranges <= float(scan.range_max))
        & (np.abs(wrapped) <= float(half_angle_rad))
    )
    return float(np.min(ranges[valid])) if np.any(valid) else float("inf")


def apply_optional_speed_cap(
    learned_speed_command: float,
    deployment_speed_cap: float,
) -> float:
    """Apply a positive deployment cap; zero or negative means uncapped."""
    learned = float(learned_speed_command)
    cap = float(deployment_speed_cap)
    return min(learned, cap) if cap > 0.0 else learned


def inference_frame_due(
    now_sec: float,
    last_sec: float,
    maximum_rate_hz: float,
    slack_sec: float,
) -> bool:
    """Return true when a frame should run; a non-positive rate is source-driven."""
    if maximum_rate_hz <= 0.0 or last_sec <= 0.0:
        return True
    minimum_interval = max(
        0.0,
        1.0 / maximum_rate_hz - max(0.0, slack_sec),
    )
    return now_sec - last_sec >= minimum_interval


def temporal_pair_requires_reset(
    current_stamp_ns: int,
    previous_stamp_ns: int,
    maximum_gap_sec: float,
) -> bool:
    if current_stamp_ns <= 0 or previous_stamp_ns <= 0:
        return False
    gap_sec = (current_stamp_ns - previous_stamp_ns) / 1.0e9
    return gap_sec <= 0.0 or (
        maximum_gap_sec > 0.0 and gap_sec > maximum_gap_sec
    )


def latest_sensor_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )


class RLPolicyRuntimeNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_policy_inference")
        self._declare_parameters()
        device_name = str(self.get_parameter("device").value)
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        policy_cpu_threads = max(
            1, int(self.get_parameter("policy_cpu_threads").value)
        )
        policy_opencv_threads = max(
            1, int(self.get_parameter("policy_opencv_threads").value)
        )
        if self.device.type == "cpu":
            torch.set_num_threads(policy_cpu_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        cv2.setNumThreads(policy_opencv_threads)
        checkpoint = Path(
            str(self.get_parameter("checkpoint_path").value)
        ).expanduser().resolve()
        residual_base_value = str(
            self.get_parameter("residual_base_checkpoint").value
        ).strip()
        residual_base = (
            Path(residual_base_value).expanduser().resolve()
            if residual_base_value
            else None
        )
        self.policy_kind = str(self.get_parameter("policy_kind").value).strip().lower()
        self.camera_speed_policy = self.policy_kind in {
            "camera_speed_bc",
            "camera_speed_td3_bc",
        }
        if self.camera_speed_policy:
            self.policy, payload = load_camera_speed_policy(
                checkpoint, device=self.device
            )
            self.min_speed_command = float(
                payload.get(
                    "min_speed_command",
                    self.get_parameter("min_speed_command").value,
                )
            )
            self.max_speed_command = float(
                payload.get(
                    "max_speed_command",
                    self.get_parameter("max_speed_command").value,
                )
            )
        else:
            self.policy = load_steering_policy(
                self.policy_kind,
                checkpoint,
                device=self.device,
                residual_base_checkpoint=residual_base,
            )
            self.min_speed_command = float(
                self.get_parameter("min_speed_command").value
            )
            self.max_speed_command = float(
                self.get_parameter("max_speed_command").value
            )
        self.bridge = CvBridge()
        self.drive_enabled = bool(self.get_parameter("drive_enabled").value)
        self.speed_command = float(self.get_parameter("speed_command").value)
        self.deployment_speed_cap = float(
            self.get_parameter("deployment_speed_cap").value
        )
        self.speed_alpha = float(
            np.clip(self.get_parameter("speed_temporal_alpha").value, 0.0, 1.0)
        )
        self.max_steering_command = abs(
            float(self.get_parameter("max_steering_command").value)
        )
        self.steering_gain = float(self.get_parameter("steering_gain").value)
        self.steering_output_sign = float(
            self.get_parameter("steering_output_sign").value
        )
        self.steering_alpha = float(
            np.clip(self.get_parameter("steering_temporal_alpha").value, 0.0, 1.0)
        )
        self.adaptive_steering_enabled = bool(
            self.get_parameter("adaptive_steering_enabled").value
        )
        self.steering_stabilizer = AdaptiveSteeringStabilizer(
            SteeringStabilizerConfig(
                straight_alpha=float(
                    self.get_parameter("straight_steering_temporal_alpha").value
                ),
                curve_alpha=self.steering_alpha,
                straight_threshold=float(
                    self.get_parameter("steering_straight_threshold").value
                ),
                curve_threshold=float(
                    self.get_parameter("steering_curve_threshold").value
                ),
                straight_rate_limit=float(
                    self.get_parameter("straight_steering_rate_limit").value
                ),
                curve_rate_limit=float(
                    self.get_parameter("curve_steering_rate_limit").value
                ),
                deadband=float(self.get_parameter("steering_deadband").value),
                turn_in_anticipation_gain=float(
                    self.get_parameter("turn_in_anticipation_gain").value
                ),
                turn_in_anticipation_threshold=float(
                    self.get_parameter("turn_in_anticipation_threshold").value
                ),
            )
        )
        self.preview_steering_enabled = bool(
            self.get_parameter("preview_steering_enabled").value
        )
        self.preview_steering_blend = float(
            np.clip(self.get_parameter("preview_steering_blend").value, 0.0, 1.0)
        )
        self.preview_steering = CanonicalPreviewSteering()
        self.sync_tolerance_ns = int(
            float(self.get_parameter("sync_tolerance_sec").value) * 1.0e9
        )
        self.sensor_timeout_sec = float(
            self.get_parameter("sensor_timeout_sec").value
        )
        self.max_inference_rate_hz = float(
            self.get_parameter("max_inference_rate_hz").value
        )
        self.inference_rate_slack_sec = max(
            0.0, float(self.get_parameter("inference_rate_slack_sec").value)
        )
        self.max_image_age_sec = max(
            0.0, float(self.get_parameter("max_image_age_sec").value)
        )
        self.max_temporal_frame_gap_sec = max(
            0.0,
            float(self.get_parameter("max_temporal_frame_gap_sec").value),
        )
        self.stop_distance_m = float(
            self.get_parameter("lidar_stop_distance_m").value
        )
        self.front_half_angle_rad = math.radians(
            float(self.get_parameter("lidar_front_half_angle_deg").value)
        )
        self.lidar_safety_enabled = bool(
            self.get_parameter("lidar_safety_enabled").value
        )
        self.input_width = int(self.get_parameter("input_width").value)
        self.input_height = int(self.get_parameter("input_height").value)
        self.lidar_points = int(self.get_parameter("lidar_points").value)
        self.latest_scan: tuple[int, np.ndarray, float] | None = None
        self.latest_scan_wall_sec = 0.0
        self.last_inference_wall_sec = 0.0
        self.last_processed_image_stamp_ns = 0
        self.rate_limited_frame_count = 0
        self.stale_frame_count = 0
        self.last_steering_command = 0.0
        self.last_speed_command = self.min_speed_command
        self.last_valid_output_wall_sec = 0.0
        self.lock = threading.Lock()

        motor_topic = str(self.get_parameter("motor_topic").value)
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.motor_pub = self.create_publisher(
            Float32MultiArray,
            motor_topic,
            command_qos,
        )
        self.shadow_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("shadow_topic").value),
            command_qos,
        )
        self.debug_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("debug_topic").value),
            command_qos,
        )
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            command_qos,
        )
        self.debug_image_pub = self.create_publisher(
            Image, str(self.get_parameter("debug_image_topic").value), 2
        )
        if self.lidar_safety_enabled or not self.camera_speed_policy:
            self.create_subscription(
                LaserScan,
                str(self.get_parameter("scan_topic").value),
                self._on_scan,
                latest_sensor_qos(),
            )
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._on_image,
            latest_sensor_qos(),
        )
        self.create_timer(0.05, self._safety_timer)
        mode = "DRIVE" if self.drive_enabled else "SHADOW"
        self.get_logger().info(
            f"RL policy ready in {mode} mode: {checkpoint}, "
            f"device={self.device}, threads=torch:{policy_cpu_threads},"
            f"opencv:{policy_opencv_threads}"
        )

    def _declare_parameters(self) -> None:
        default_bc = (
            Path.home()
            / "xycar_kookmin_gazebo_track"
            / "models"
            / "il_policies"
            / "drive_canonical_real_reference_200k_20260715"
            / "drive_resnet18_lidar_best.pth"
        )
        self.declare_parameter("policy_kind", "bc")
        self.declare_parameter("checkpoint_path", str(default_bc))
        self.declare_parameter("residual_base_checkpoint", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("policy_cpu_threads", 4)
        self.declare_parameter("policy_opencv_threads", 1)
        self.declare_parameter("image_topic", "/perception/canonical_road_image")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("motor_topic", "/xycar_motor")
        self.declare_parameter("shadow_topic", "/rl/policy_motor_shadow")
        self.declare_parameter("debug_topic", "/rl/policy_debug")
        self.declare_parameter("status_topic", "/rl/policy_status")
        self.declare_parameter("debug_image_topic", "/rl/policy_input_image")
        self.declare_parameter("drive_enabled", False)
        self.declare_parameter("speed_command", 3.0)
        self.declare_parameter("min_speed_command", 4.0)
        self.declare_parameter("max_speed_command", 25.0)
        self.declare_parameter("deployment_speed_cap", 0.0)
        self.declare_parameter("speed_temporal_alpha", 0.35)
        self.declare_parameter("max_steering_command", 42.0)
        self.declare_parameter("steering_gain", 1.0)
        self.declare_parameter("steering_output_sign", 1.0)
        self.declare_parameter("adaptive_steering_enabled", True)
        self.declare_parameter("steering_temporal_alpha", 0.90)
        self.declare_parameter("straight_steering_temporal_alpha", 0.20)
        self.declare_parameter("steering_straight_threshold", 0.08)
        self.declare_parameter("steering_curve_threshold", 0.35)
        self.declare_parameter("straight_steering_rate_limit", 0.05)
        self.declare_parameter("curve_steering_rate_limit", 0.38)
        self.declare_parameter("steering_deadband", 0.02)
        self.declare_parameter("turn_in_anticipation_gain", 0.35)
        self.declare_parameter("turn_in_anticipation_threshold", 0.04)
        self.declare_parameter("preview_steering_enabled", False)
        self.declare_parameter("preview_steering_blend", 0.35)
        self.declare_parameter("sync_tolerance_sec", 0.08)
        self.declare_parameter("sensor_timeout_sec", 0.50)
        self.declare_parameter("max_inference_rate_hz", 15.0)
        self.declare_parameter("inference_rate_slack_sec", 0.04)
        self.declare_parameter("max_image_age_sec", 0.30)
        self.declare_parameter("max_temporal_frame_gap_sec", 0.25)
        self.declare_parameter("lidar_safety_enabled", False)
        self.declare_parameter("lidar_stop_distance_m", 0.25)
        self.declare_parameter("lidar_front_half_angle_deg", 20.0)
        self.declare_parameter("input_width", 160)
        self.declare_parameter("input_height", 90)
        self.declare_parameter("lidar_points", 360)

    def _on_scan(self, msg: LaserScan) -> None:
        try:
            lidar = preprocess_lidar_ranges(
                msg.ranges,
                msg.range_min,
                msg.range_max,
                self.lidar_points,
            )
            front_min = front_obstacle_distance(msg, self.front_half_angle_rad)
        except ValueError:
            return
        with self.lock:
            self.latest_scan = (message_stamp_ns(msg), lidar, front_min)
            self.latest_scan_wall_sec = time.monotonic()

    def _on_image(self, msg: Image) -> None:
        now_wall = time.monotonic()
        if not inference_frame_due(
            now_wall,
            self.last_inference_wall_sec,
            self.max_inference_rate_hz,
            self.inference_rate_slack_sec,
        ):
            self.rate_limited_frame_count += 1
            return
        with self.lock:
            scan = self.latest_scan
        if not self.camera_speed_policy and scan is None:
            return
        image_stamp = message_stamp_ns(msg)
        now_ros_ns = self.get_clock().now().nanoseconds
        image_age_sec = (
            (now_ros_ns - image_stamp) / 1.0e9
            if image_stamp > 0 and now_ros_ns >= image_stamp
            else -1.0
        )
        if (
            self.max_image_age_sec > 0.0
            and 0.0 <= image_age_sec < 60.0
            and image_age_sec > self.max_image_age_sec
        ):
            self.stale_frame_count += 1
            return
        if self.camera_speed_policy and temporal_pair_requires_reset(
            image_stamp,
            self.last_processed_image_stamp_ns,
            self.max_temporal_frame_gap_sec,
        ):
            self.policy.reset()
        if scan is None:
            scan_stamp = 0
            lidar = None
            front_min = float("inf")
        else:
            scan_stamp, lidar, front_min = scan
            if not self.camera_speed_policy and image_stamp > 0 and scan_stamp > 0:
                if abs(image_stamp - scan_stamp) > self.sync_tolerance_ns:
                    return
        started = time.perf_counter()
        image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image = preprocess_bgr_image(
            image_bgr, self.input_width, self.input_height
        )
        if self.camera_speed_policy:
            action = self.policy({"image": image})
            final_norm = float(action[0])
            base_norm = final_norm
            residual_norm = 0.0
            learned_speed = denormalize_speed_command(
                float(action[1]),
                self.min_speed_command,
                self.max_speed_command,
            )
            target_speed = apply_optional_speed_cap(
                learned_speed,
                self.deployment_speed_cap,
            )
        else:
            final_norm, base_norm, residual_norm = self.policy.action_components(
                {"image": image, "lidar": lidar}
            )
            target_speed = self.speed_command
        preview_value = 0.0
        preview_confidence = 0.0
        preview_curve_hint = 0.0
        if self.camera_speed_policy and self.preview_steering_enabled:
            preview = self.preview_steering.update(image)
            preview_value = preview.steering_norm
            preview_confidence = preview.confidence
            preview_curve_hint = preview.curve_hint * preview.confidence
            blend = self.preview_steering_blend * preview.confidence
            final_norm = (1.0 - blend) * final_norm + blend * preview_value
        raw_norm = float(
            np.clip(
                final_norm * self.steering_gain * self.steering_output_sign,
                -1.0,
                1.0,
            )
        )
        if self.adaptive_steering_enabled:
            steering_norm = self.steering_stabilizer.update(
                raw_norm, curve_hint=preview_curve_hint
            )
            angle = steering_norm * self.max_steering_command
        else:
            raw_angle = raw_norm * self.max_steering_command
            angle = (
                self.steering_alpha * raw_angle
                + (1.0 - self.steering_alpha) * self.last_steering_command
            )
        obstacle_stop = self.lidar_safety_enabled and front_min < self.stop_distance_m
        smoothed_speed = (
            self.speed_alpha * target_speed
            + (1.0 - self.speed_alpha) * self.last_speed_command
        )
        speed = 0.0 if obstacle_stop else smoothed_speed
        self.last_steering_command = angle
        self.last_speed_command = speed
        self.last_inference_wall_sec = now_wall
        self.last_processed_image_stamp_ns = image_stamp
        self.last_valid_output_wall_sec = now_wall
        inference_ms = (time.perf_counter() - started) * 1000.0
        sensor_age_ms = 0.0
        publish_ros_ns = self.get_clock().now().nanoseconds
        if image_stamp > 0 and publish_ros_ns >= image_stamp:
            delta_ns = publish_ros_ns - image_stamp
            sensor_age_ms = delta_ns / 1.0e6 if delta_ns < 60_000_000_000 else -1.0
        self._publish_command(angle, speed)
        debug = Float32MultiArray()
        debug.data = [
            base_norm,
            residual_norm,
            final_norm,
            angle,
            speed,
            front_min,
            inference_ms,
            sensor_age_ms,
            preview_value,
            preview_confidence,
            preview_curve_hint,
            float(self.rate_limited_frame_count),
            float(self.stale_frame_count),
        ]
        self.debug_pub.publish(debug)
        status = String()
        status.data = "obstacle_stop" if obstacle_stop else "running"
        self.status_pub.publish(status)
        if self.debug_image_pub.get_subscription_count() > 0:
            debug_image = self.bridge.cv2_to_imgmsg(
                model_input_to_bgr(image), encoding="bgr8"
            )
            debug_image.header = msg.header
            self.debug_image_pub.publish(debug_image)

    def _publish_command(self, angle: float, speed: float) -> None:
        message = Float32MultiArray()
        message.data = [float(angle), float(speed)]
        self.shadow_pub.publish(message)
        if self.drive_enabled:
            self.motor_pub.publish(message)

    def _safety_timer(self) -> None:
        if not self.drive_enabled:
            return
        now = time.monotonic()
        scan_stale = (
            not self.camera_speed_policy
            and now - self.latest_scan_wall_sec > self.sensor_timeout_sec
        )
        sensor_stale = (
            scan_stale
            or now - self.last_valid_output_wall_sec > self.sensor_timeout_sec
        )
        if sensor_stale:
            if self.camera_speed_policy:
                self.policy.reset()
            self.last_processed_image_stamp_ns = 0
            self.steering_stabilizer.reset()
            self.preview_steering.reset()
            self.last_steering_command = 0.0
            self.last_speed_command = self.min_speed_command
            self._publish_command(0.0, 0.0)
            status = String()
            status.data = "sensor_timeout_stop"
            self.status_pub.publish(status)

    def destroy_node(self):
        if self.drive_enabled:
            self._publish_command(0.0, 0.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RLPolicyRuntimeNode()
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
            rclpy.shutdown()
