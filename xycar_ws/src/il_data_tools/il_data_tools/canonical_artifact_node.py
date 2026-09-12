from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TextIO

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from il_data_tools.canonical_artifacts import (
    CanonicalArtifactAugmenter,
    CanonicalVisibilityAugmenter,
)


class CanonicalArtifactNode(Node):
    def __init__(self) -> None:
        super().__init__("il_canonical_artifact_augmenter")
        self._declare_parameters()
        self.bridge = CvBridge()
        self.mode = str(self.get_parameter("mode").value)
        seed = int(self.get_parameter("seed").value)
        background_gray = int(self.get_parameter("background_gray").value)
        if self.mode == "real_visibility":
            self.augmenter = CanonicalVisibilityAugmenter(
                seed=seed,
                white_probabilities=(
                    float(self.get_parameter("white_both_probability").value),
                    float(self.get_parameter("white_one_probability").value),
                    float(self.get_parameter("white_none_probability").value),
                ),
                yellow_visible_probability=float(
                    self.get_parameter("yellow_visible_probability").value
                ),
                white_duration_frames=(
                    int(self.get_parameter("white_min_duration_frames").value),
                    int(self.get_parameter("white_max_duration_frames").value),
                ),
                yellow_duration_frames=(
                    int(self.get_parameter("yellow_min_duration_frames").value),
                    int(self.get_parameter("yellow_max_duration_frames").value),
                ),
                background_gray=background_gray,
                prevent_blank=bool(self.get_parameter("prevent_blank").value),
            )
        elif self.mode == "legacy":
            self.augmenter = CanonicalArtifactAugmenter(
                seed=seed,
                event_start_probability=float(
                    self.get_parameter("event_start_probability").value
                ),
                min_duration_frames=int(
                    self.get_parameter("min_duration_frames").value
                ),
                max_duration_frames=int(
                    self.get_parameter("max_duration_frames").value
                ),
                center_jump_min_px=float(
                    self.get_parameter("center_jump_min_px").value
                ),
                center_jump_max_px=float(
                    self.get_parameter("center_jump_max_px").value
                ),
                white_bend_min_px=float(
                    self.get_parameter("white_bend_min_px").value
                ),
                white_bend_max_px=float(
                    self.get_parameter("white_bend_max_px").value
                ),
                background_gray=background_gray,
            )
        else:
            raise ValueError("mode must be real_visibility or legacy")
        output_topic = str(self.get_parameter("output_topic").value)
        self.publisher = self.create_publisher(Image, output_topic, 10)
        self.state_publisher = self.create_publisher(
            String,
            str(self.get_parameter("state_topic").value),
            10,
        )
        self.event_log: Optional[TextIO] = None
        event_log_path = str(self.get_parameter("event_log_path").value).strip()
        if event_log_path:
            path = Path(event_log_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.event_log = path.open("a", encoding="utf-8", buffering=1)
        self.create_subscription(
            Image,
            str(self.get_parameter("input_topic").value),
            self._image_cb,
            qos_profile_sensor_data,
        )
        self.frame_count = 0
        self.artifact_frame_count = 0
        self.event_count = 0
        self.get_logger().info(
            f"canonical augmentation ready: mode={self.mode} "
            f"{self.get_parameter('input_topic').value} -> {output_topic}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("input_topic", "/perception/canonical_road_image")
        self.declare_parameter(
            "output_topic", "/perception/canonical_road_image_augmented"
        )
        self.declare_parameter("state_topic", "/il/canonical_artifact_state")
        self.declare_parameter("event_log_path", "")
        self.declare_parameter("seed", 20260715)
        self.declare_parameter("mode", "real_visibility")
        self.declare_parameter("white_both_probability", 0.462)
        self.declare_parameter("white_one_probability", 0.512)
        self.declare_parameter("white_none_probability", 0.026)
        self.declare_parameter("yellow_visible_probability", 0.571)
        self.declare_parameter("white_min_duration_frames", 3)
        self.declare_parameter("white_max_duration_frames", 40)
        self.declare_parameter("yellow_min_duration_frames", 3)
        self.declare_parameter("yellow_max_duration_frames", 30)
        self.declare_parameter("prevent_blank", True)
        self.declare_parameter("event_start_probability", 0.018)
        self.declare_parameter("min_duration_frames", 3)
        self.declare_parameter("max_duration_frames", 9)
        self.declare_parameter("center_jump_min_px", 7.0)
        self.declare_parameter("center_jump_max_px", 22.0)
        self.declare_parameter("white_bend_min_px", 18.0)
        self.declare_parameter("white_bend_max_px", 58.0)
        self.declare_parameter("background_gray", 36)

    def _image_cb(self, message: Image) -> None:
        image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        output, event, started = self.augmenter.process(image)
        self.frame_count += 1
        if event is not None:
            self.artifact_frame_count += 1
        output_message = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
        output_message.header = message.header
        self.publisher.publish(output_message)

        state = {
            "frame": self.frame_count,
            "artifact_frames": self.artifact_frame_count,
            "event_count": self.event_count + int(started),
            "kind": event.kind if event is not None else "none",
            "remaining_frames": event.remaining_frames if event is not None else 0,
            "parameters": event.parameters if event is not None else {},
            "timestamp_ns": int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec),
        }
        if started:
            self.event_count += 1
            state["event_count"] = self.event_count
            if self.event_log is not None:
                self.event_log.write(json.dumps(state, sort_keys=True) + "\n")
        state_message = String()
        state_message.data = json.dumps(state, sort_keys=True)
        self.state_publisher.publish(state_message)

    def destroy_node(self):
        if self.event_log is not None:
            self.event_log.close()
            self.event_log = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CanonicalArtifactNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        # ROS 2 Humble can race a SensorData subscription teardown against
        # SIGINT after the recorder reaches its sample limit.
        if "Unable to convert call argument to Python object" not in str(exc):
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
