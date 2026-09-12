from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray

from il_data_tools.record_schema import stamp_to_ns
from il_data_tools.runtime_preprocessing import (
    model_input_to_bgr,
    preprocess_bgr_image,
    preprocess_lidar_ranges,
)
from il_data_tools.sync_buffer import TimedBuffer


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


class PolicyInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("il_policy_inference")
        self._declare_parameters()

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.motor_topic = str(self.get_parameter("motor_topic").value)
        self.shadow_topic = str(self.get_parameter("shadow_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.drive_enabled = bool(self.get_parameter("drive_enabled").value)
        self.input_width = int(self.get_parameter("input_width").value)
        self.input_height = int(self.get_parameter("input_height").value)
        self.lidar_points = int(self.get_parameter("lidar_points").value)
        self.max_steer_scale = float(self.get_parameter("max_steer_scale").value)
        configured_sign = float(self.get_parameter("steering_output_sign").value)
        if configured_sign not in (-1.0, 1.0):
            raise ValueError("steering_output_sign must be exactly -1.0 or 1.0")
        self.steering_output_sign = configured_sign
        self.angle_min = float(self.get_parameter("angle_command_min").value)
        self.angle_max = float(self.get_parameter("angle_command_max").value)
        self.speed_command = float(self.get_parameter("speed_command").value)
        configured_min_speed = float(self.get_parameter("min_speed_command").value)
        if self.speed_command >= 0.0:
            self.min_speed_command = clamp(
                configured_min_speed,
                0.0,
                self.speed_command,
            )
        else:
            self.min_speed_command = configured_min_speed
        self.slow_down_angle = float(self.get_parameter("slow_down_angle_cmd").value)
        self.max_drive_angle = float(self.get_parameter("max_abs_angle_for_drive").value)
        self.steering_alpha = clamp(
            float(self.get_parameter("steering_temporal_alpha").value),
            0.0,
            1.0,
        )
        self.sync_tolerance_ns = int(
            max(0.001, float(self.get_parameter("sync_tolerance_sec").value)) * 1.0e9
        )
        self.timeout_sec = max(0.1, float(self.get_parameter("sensor_timeout_sec").value))
        max_rate_hz = max(0.1, float(self.get_parameter("max_inference_rate_hz").value))
        self.min_inference_period = 1.0 / max_rate_hz

        model_path = Path(str(self.get_parameter("model_path").value)).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"TorchScript model not found: {model_path}")

        import torch

        requested_device = str(self.get_parameter("device").value).lower()
        if requested_device == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA unavailable; falling back to CPU")
            requested_device = "cpu"
        self.torch = torch
        self.device = torch.device(requested_device)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        self.bridge = CvBridge()
        self.scan_buffer = TimedBuffer(maxlen=100)

        self.shadow_pub = self.create_publisher(Float32MultiArray, self.shadow_topic, 10)
        self.debug_pub = self.create_publisher(Float32MultiArray, self.debug_topic, 10)
        self.debug_image_pub = self.create_publisher(
            Image,
            self.debug_image_topic,
            10,
        )
        self.motor_pub = (
            self.create_publisher(Float32MultiArray, self.motor_topic, 10)
            if self.drive_enabled
            else None
        )
        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(0.10, self._watchdog)

        self.last_inference_wall = 0.0
        self.last_valid_wall = 0.0
        self.last_stop_wall = 0.0
        self.last_angle = 0.0
        self.stop_sent = False
        self.inference_count = 0
        self.missing_scan_count = 0
        self.pending_image: tuple[int, Image] | None = None
        mode = "DRIVE" if self.drive_enabled else "SHADOW"
        self.get_logger().info(
            f"IL policy ready: {mode}, model={model_path}, device={self.device}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("model_path", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("motor_topic", "/xycar_motor")
        self.declare_parameter("shadow_topic", "/il/policy_motor_shadow")
        self.declare_parameter("debug_topic", "/il/policy_debug")
        self.declare_parameter("debug_image_topic", "/il/policy_input_image")
        self.declare_parameter("drive_enabled", False)
        self.declare_parameter("input_width", 160)
        self.declare_parameter("input_height", 90)
        self.declare_parameter("lidar_points", 360)
        self.declare_parameter("max_steer_scale", 100.0)
        self.declare_parameter("steering_output_sign", 1.0)
        self.declare_parameter("angle_command_min", -42.0)
        self.declare_parameter("angle_command_max", 42.0)
        self.declare_parameter("speed_command", 4.0)
        self.declare_parameter("min_speed_command", 3.0)
        self.declare_parameter("slow_down_angle_cmd", 18.0)
        self.declare_parameter("max_abs_angle_for_drive", 43.0)
        self.declare_parameter("steering_temporal_alpha", 0.55)
        self.declare_parameter("sync_tolerance_sec", 0.05)
        self.declare_parameter("sensor_timeout_sec", 0.50)
        self.declare_parameter("max_inference_rate_hz", 15.0)

    def _scan_callback(self, msg: LaserScan) -> None:
        fallback_ns = self.get_clock().now().nanoseconds
        scan_stamp_ns = stamp_to_ns(msg, fallback_ns)
        self.scan_buffer.add(scan_stamp_ns, msg)
        self._try_pending_image(scan_stamp_ns)

    def _image_callback(self, msg: Image) -> None:
        now_wall = time.monotonic()
        if now_wall - self.last_inference_wall < self.min_inference_period:
            return
        image_stamp_ns = stamp_to_ns(msg, self.get_clock().now().nanoseconds)
        scan_item = self.scan_buffer.nearest(image_stamp_ns, self.sync_tolerance_ns)
        if scan_item is None:
            if self.pending_image is not None:
                self.missing_scan_count += 1
            self.pending_image = (image_stamp_ns, msg)
            return
        self.pending_image = None
        self._run_inference(msg, image_stamp_ns, scan_item, now_wall)

    def _try_pending_image(self, latest_scan_stamp_ns: int) -> None:
        if self.pending_image is None:
            return
        image_stamp_ns, image_msg = self.pending_image
        scan_item = self.scan_buffer.nearest(image_stamp_ns, self.sync_tolerance_ns)
        if scan_item is None:
            if latest_scan_stamp_ns > image_stamp_ns + self.sync_tolerance_ns:
                self.pending_image = None
                self.missing_scan_count += 1
            return
        self.pending_image = None
        self._run_inference(image_msg, image_stamp_ns, scan_item, time.monotonic())

    def _run_inference(self, msg, image_stamp_ns, scan_item, now_wall) -> None:
        started = time.perf_counter()
        image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image_np = preprocess_bgr_image(
            image_bgr,
            self.input_width,
            self.input_height,
        )
        debug_image = self.bridge.cv2_to_imgmsg(
            model_input_to_bgr(image_np),
            encoding="bgr8",
        )
        debug_image.header = msg.header
        self.debug_image_pub.publish(debug_image)
        scan = scan_item.msg
        lidar_np = preprocess_lidar_ranges(
            scan.ranges,
            scan.range_min,
            scan.range_max,
            self.lidar_points,
        )
        image_tensor = self.torch.from_numpy(image_np).unsqueeze(0).to(self.device)
        lidar_tensor = self.torch.from_numpy(lidar_np).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            steer_norm = float(self.model(image_tensor, lidar_tensor).reshape(-1)[0].item())

        raw_angle = clamp(
            steer_norm * self.max_steer_scale * self.steering_output_sign,
            self.angle_min,
            self.angle_max,
        )
        angle = (
            self.steering_alpha * raw_angle
            + (1.0 - self.steering_alpha) * self.last_angle
        )
        speed = self._speed_for_angle(angle)
        inference_ms = (time.perf_counter() - started) * 1000.0
        scan_offset_ms = abs(scan_item.stamp_ns - image_stamp_ns) / 1.0e6

        self.last_angle = angle
        self.last_inference_wall = now_wall
        self.last_valid_wall = now_wall
        self.stop_sent = False
        self.inference_count += 1
        self._publish_command(angle, speed)

        debug = Float32MultiArray()
        debug.data = [
            steer_norm,
            raw_angle,
            angle,
            speed,
            scan_offset_ms,
            inference_ms,
            float(self.inference_count),
            float(self.missing_scan_count),
            float(image_bgr.shape[1]),
            float(image_bgr.shape[0]),
            float(image_np.mean()),
        ]
        self.debug_pub.publish(debug)

    def _speed_for_angle(self, angle: float) -> float:
        magnitude = abs(angle)
        if magnitude >= self.max_drive_angle:
            return 0.0
        if magnitude <= self.slow_down_angle:
            return self.speed_command
        ratio = (magnitude - self.slow_down_angle) / max(
            1.0e-6,
            self.max_drive_angle - self.slow_down_angle,
        )
        return self.speed_command + ratio * (
            self.min_speed_command - self.speed_command
        )

    def _publish_command(self, angle: float, speed: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(angle), float(speed)]
        self.shadow_pub.publish(msg)
        if self.motor_pub is not None:
            self.motor_pub.publish(msg)

    def _watchdog(self) -> None:
        now_wall = time.monotonic()
        if self.last_valid_wall <= 0.0:
            if now_wall - self.last_stop_wall >= 0.5:
                self.last_angle = 0.0
                self._publish_command(0.0, 0.0)
                self.last_stop_wall = now_wall
            return
        if now_wall - self.last_valid_wall <= self.timeout_sec:
            return
        if self.stop_sent and now_wall - self.last_stop_wall < 0.5:
            return
        self.last_angle = 0.0
        self._publish_command(0.0, 0.0)
        self.last_stop_wall = now_wall
        if self.stop_sent:
            return
        self.stop_sent = True
        self.get_logger().warn("policy sensor timeout: stop command published")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node._publish_command(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
