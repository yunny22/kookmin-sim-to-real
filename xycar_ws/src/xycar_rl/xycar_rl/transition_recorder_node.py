from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, Float32MultiArray
from tf2_msgs.msg import TFMessage

from xycar_rl.gazebo_env import quaternion_to_yaw, select_model_transform
from xycar_rl.reward import calculate_reward
from xycar_rl.termination import EpisodeTermination, TerminationConfig
from xycar_rl.track_geometry import TrackProjection, TrackReference
from xycar_rl.transition_io import TransitionWriter


def stamp_ns(message, fallback_ns: int) -> int:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return int(fallback_ns)
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else int(fallback_ns)


@dataclass
class RecorderSnapshot:
    timestamp_ns: int
    image_bgr: np.ndarray
    scan_payload: dict | None
    x: float
    y: float
    yaw: float
    speed_mps: float
    projection: TrackProjection
    angle_command: float
    speed_command: float


class TransitionRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_transition_recorder")
        self._declare_parameters()
        world_sdf = Path(
            str(self.get_parameter("world_sdf").value)
        ).expanduser().resolve()
        output_dir = Path(
            str(self.get_parameter("output_dir").value)
        ).expanduser().resolve()
        self.max_transitions = max(0, int(self.get_parameter("max_transitions").value))
        self.max_rate_hz = max(0.1, float(self.get_parameter("max_rate_hz").value))
        self.rate_period_tolerance_ratio = max(
            0.90,
            min(
                1.0,
                float(
                    self.get_parameter(
                        "rate_period_tolerance_ratio"
                    ).value
                ),
            ),
        )
        self.min_period_ns = int(
            self.rate_period_tolerance_ratio
            * 1.0e9
            / self.max_rate_hz
        )
        self.max_steering_command = abs(
            float(self.get_parameter("max_steering_command").value)
        )
        self.require_reset_after_terminal = bool(
            self.get_parameter("require_reset_after_terminal").value
        )
        self.auto_stop_on_terminal = bool(
            self.get_parameter("auto_stop_on_terminal").value
        )
        self.require_action_trace = bool(
            self.get_parameter("require_action_trace").value
        )
        self.require_expert_action_trace = bool(
            self.get_parameter("require_expert_action_trace").value
        )
        self.require_lidar = bool(self.get_parameter("require_lidar").value)
        self.track = TrackReference.from_sdf(
            world_sdf,
            target_right_offset_m=float(
                self.get_parameter("target_right_offset_m").value
            ),
        )
        self.termination = EpisodeTermination(
            TerminationConfig(
                off_track_threshold_m=float(
                    self.get_parameter("off_track_threshold_m").value
                ),
                stuck_timeout_sec=float(
                    self.get_parameter("stuck_timeout_sec").value
                ),
                max_episode_sec=float(
                    self.get_parameter("max_episode_sec").value
                ),
                minimum_lap_fraction=float(
                    self.get_parameter("minimum_lap_fraction").value
                ),
            )
        )
        self.writer = TransitionWriter(
            output_dir,
            metadata={
                "world_sdf": str(world_sdf),
                "track_length_m": self.track.length_m,
                "target_right_offset_m": float(
                    self.get_parameter("target_right_offset_m").value
                ),
                "max_steering_command": self.max_steering_command,
                "observation_contract": (
                    "canonical_bgr_plus_raw_lidar"
                    if self.require_lidar
                    else "canonical_bgr_camera_only"
                ),
                "action_alignment": (
                    "state_scan_timestamp_join_exact_or_motor_fallback"
                    if self.require_lidar
                    else "state_image_timestamp_join_exact_or_motor_fallback"
                ),
                "expert_action_alignment": "optional_state_timestamp_join",
                "require_action_trace": self.require_action_trace,
                "require_expert_action_trace": self.require_expert_action_trace,
                "sensor_sync_tolerance_sec": float(
                    self.get_parameter("sensor_sync_tolerance_sec").value
                ),
                "max_rate_hz": self.max_rate_hz,
                "rate_period_tolerance_ratio": (
                    self.rate_period_tolerance_ratio
                ),
            },
            allow_overwrite=bool(self.get_parameter("allow_overwrite").value),
        )
        self.bridge = CvBridge()
        self.latest_image: tuple[int, np.ndarray] | None = None
        self.latest_scan: LaserScan | None = None
        self.latest_odom: Odometry | None = None
        self.latest_world_pose: tuple[float, float, float] | None = None
        self.latest_angle_command = 0.0
        self.latest_speed_command = 0.0
        self.previous: RecorderSnapshot | None = None
        self.previous_action_norm = 0.0
        self.steering_history: deque[float] = deque(maxlen=12)
        self.last_saved_timestamp_ns = 0
        self.collision = False
        self.waiting_for_reset = False
        self.episode_id = 0
        self.step_id = 0
        self.elapsed_sec = 0.0
        self.cumulative_progress_m = 0.0
        self.finished = False
        self.actions_by_state_timestamp: dict[int, tuple[float, float]] = {}
        self.expert_actions_by_state_timestamp: dict[
            int, tuple[float, float]
        ] = {}
        self.pending_transitions: deque[
            tuple[RecorderSnapshot, RecorderSnapshot]
        ] = deque()
        self.exact_action_count = 0
        self.expert_action_count = 0
        self.fallback_action_count = 0
        self.skipped_untraced_count = 0

        image_topic = str(self.get_parameter("image_topic").value)
        scan_topic = str(self.get_parameter("scan_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        world_pose_topic = str(self.get_parameter("world_pose_topic").value)
        contact_topic = str(self.get_parameter("contact_topic").value)
        motor_topic = str(self.get_parameter("motor_topic").value)
        reset_topic = str(self.get_parameter("episode_reset_topic").value)
        action_trace_topic = str(self.get_parameter("action_trace_topic").value)
        expert_action_trace_topic = str(
            self.get_parameter("expert_action_trace_topic").value
        )
        self.create_subscription(
            Image, image_topic, self._on_image, qos_profile_sensor_data
        )
        if self.require_lidar:
            self.create_subscription(
                LaserScan, scan_topic, self._on_scan, qos_profile_sensor_data
            )
        self.create_subscription(
            Odometry, odom_topic, self._on_odom, qos_profile_sensor_data
        )
        self.create_subscription(
            TFMessage,
            world_pose_topic,
            self._on_world_pose,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Contacts, contact_topic, self._on_contact, qos_profile_sensor_data
        )
        self.create_subscription(Float32MultiArray, motor_topic, self._on_motor, 10)
        self.create_subscription(Bool, reset_topic, self._on_reset, 10)
        self.create_subscription(
            TwistStamped, action_trace_topic, self._on_action_trace, 10
        )
        self.create_subscription(
            TwistStamped,
            expert_action_trace_topic,
            self._on_expert_action_trace,
            10,
        )
        self.stop_pub = self.create_publisher(Float32MultiArray, motor_topic, 10)
        self.get_logger().info(
            f"RL transition recorder ready: {output_dir}, track={self.track.length_m:.2f}m"
        )

    def _declare_parameters(self) -> None:
        home = Path.home()
        root = home / "xycar_kookmin_gazebo_track"
        default_session = time.strftime("rl_transitions_%Y%m%d_%H%M%S")
        self.declare_parameter(
            "world_sdf", str(root / "worlds" / "kookmin_xycar_track_final.sdf")
        )
        self.declare_parameter(
            "output_dir", str(root / "datasets" / "rl" / default_session)
        )
        self.declare_parameter("image_topic", "/perception/canonical_road_image")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("model_name", "xycar_ackermann")
        self.declare_parameter("odom_topic", "/model/xycar_ackermann/odometry")
        self.declare_parameter(
            "world_pose_topic", "/world/kookmin_xycar_track/dynamic_pose/info"
        )
        self.declare_parameter("contact_topic", "/xycar_contact")
        self.declare_parameter("motor_topic", "/xycar_motor")
        self.declare_parameter("episode_reset_topic", "/rl/episode_reset")
        self.declare_parameter("action_trace_topic", "/rl/action_applied")
        self.declare_parameter("expert_action_trace_topic", "/rl/action_expert")
        self.declare_parameter("target_right_offset_m", 0.0)
        self.declare_parameter("off_track_threshold_m", 0.38)
        self.declare_parameter("stuck_timeout_sec", 2.0)
        self.declare_parameter("max_episode_sec", 120.0)
        self.declare_parameter("minimum_lap_fraction", 0.98)
        self.declare_parameter("max_steering_command", 42.0)
        self.declare_parameter("sensor_sync_tolerance_sec", 0.25)
        self.declare_parameter("max_rate_hz", 10.0)
        self.declare_parameter("rate_period_tolerance_ratio", 0.99)
        self.declare_parameter("max_transitions", 0)
        self.declare_parameter("allow_overwrite", False)
        self.declare_parameter("require_action_trace", False)
        self.declare_parameter("require_expert_action_trace", False)
        self.declare_parameter("require_lidar", True)
        self.declare_parameter("require_reset_after_terminal", True)
        self.declare_parameter("auto_stop_on_terminal", True)

    def _on_image(self, msg: Image) -> None:
        timestamp_ns = stamp_ns(msg, self.get_clock().now().nanoseconds)
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_image = (timestamp_ns, image)
        self._process_latest_pair()

    def _on_odom(self, msg: Odometry) -> None:
        self.latest_odom = msg
        self._process_latest_pair()

    def _on_world_pose(self, msg: TFMessage) -> None:
        transform = select_model_transform(
            msg.transforms,
            str(self.get_parameter("model_name").value),
        )
        if transform is None:
            return
        transform = transform.transform
        rotation = transform.rotation
        self.latest_world_pose = (
            float(transform.translation.x),
            float(transform.translation.y),
            quaternion_to_yaw(
                rotation.x, rotation.y, rotation.z, rotation.w
            ),
        )
        self._process_latest_pair()

    def _on_contact(self, msg: Contacts) -> None:
        if msg.contacts:
            self.collision = True

    def _on_motor(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            return
        self.latest_angle_command = float(msg.data[0])
        self.latest_speed_command = float(msg.data[1])

    def _on_action_trace(self, msg: TwistStamped) -> None:
        self._store_action_trace(msg, self.actions_by_state_timestamp)
        self._drain_pending_transitions()

    def _on_expert_action_trace(self, msg: TwistStamped) -> None:
        self._store_action_trace(msg, self.expert_actions_by_state_timestamp)
        self._drain_pending_transitions()

    def _store_action_trace(
        self,
        msg: TwistStamped,
        destination: dict[int, tuple[float, float]],
    ) -> None:
        timestamp_ns = stamp_ns(msg, 0)
        if timestamp_ns <= 0:
            return
        destination[timestamp_ns] = (
            float(msg.twist.angular.z),
            float(msg.twist.linear.x),
        )
        if len(destination) > 64:
            oldest = sorted(destination)[:-32]
            for key in oldest:
                destination.pop(key, None)

    def _on_reset(self, msg: Bool) -> None:
        if not msg.data:
            return
        self.episode_id += 1
        self.step_id = 0
        self.previous = None
        self.previous_action_norm = 0.0
        self.steering_history.clear()
        self.elapsed_sec = 0.0
        self.cumulative_progress_m = 0.0
        self.collision = False
        self.waiting_for_reset = False
        self.latest_image = None
        self.latest_scan = None
        self.latest_odom = None
        self.latest_world_pose = None
        self.actions_by_state_timestamp.clear()
        self.expert_actions_by_state_timestamp.clear()
        self.pending_transitions.clear()
        self.termination.reset()

    def _scan_payload(self, msg: LaserScan) -> dict:
        return {
            "ranges": np.asarray(msg.ranges, dtype=np.float32),
            "intensities": np.asarray(msg.intensities, dtype=np.float32),
            "angle_min": np.float32(msg.angle_min),
            "angle_max": np.float32(msg.angle_max),
            "angle_increment": np.float32(msg.angle_increment),
            "range_min": np.float32(msg.range_min),
            "range_max": np.float32(msg.range_max),
            "frame_id": np.asarray(msg.header.frame_id),
        }

    def _snapshot(
        self,
        timestamp_ns: int,
        scan_payload: dict | None,
    ) -> RecorderSnapshot | None:
        if self.latest_image is None or self.latest_odom is None:
            return None
        image_timestamp_ns = self.latest_image[0]
        tolerance_ns = int(
            max(
                0.0,
                float(self.get_parameter("sensor_sync_tolerance_sec").value),
            )
            * 1.0e9
        )
        if (
            tolerance_ns > 0
            and image_timestamp_ns > 0
            and timestamp_ns > 0
            and self.require_lidar
            and abs(image_timestamp_ns - timestamp_ns) > tolerance_ns
        ):
            return None
        if timestamp_ns - self.last_saved_timestamp_ns < self.min_period_ns:
            return None
        speed = self.latest_odom.twist.twist.linear.x
        if self.latest_world_pose is None:
            pose = self.latest_odom.pose.pose
            x = float(pose.position.x)
            y = float(pose.position.y)
            yaw = quaternion_to_yaw(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
        else:
            x, y, yaw = self.latest_world_pose
        hint = None if self.previous is None else self.previous.projection.segment_index
        projection = self.track.project(
            x,
            y,
            yaw,
            hint_segment_index=hint,
        )
        return RecorderSnapshot(
            timestamp_ns=timestamp_ns,
            image_bgr=self.latest_image[1].copy(),
            scan_payload=scan_payload,
            x=float(x),
            y=float(y),
            yaw=float(yaw),
            speed_mps=float(speed),
            projection=projection,
            angle_command=self.latest_angle_command,
            speed_command=self.latest_speed_command,
        )

    def _on_scan(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        self._process_latest_pair()

    def _process_latest_pair(self) -> None:
        if self.waiting_for_reset or self.finished:
            return
        if self.require_lidar:
            if self.latest_scan is None:
                return
            timestamp_ns = stamp_ns(
                self.latest_scan, self.get_clock().now().nanoseconds
            )
            scan_payload = self._scan_payload(self.latest_scan)
        else:
            if self.latest_image is None:
                return
            timestamp_ns = self.latest_image[0]
            scan_payload = None
        current = self._snapshot(timestamp_ns, scan_payload)
        if current is None:
            return
        self.last_saved_timestamp_ns = current.timestamp_ns
        if self.previous is None:
            self.previous = current
            return

        previous = self.previous
        self.previous = current
        if self.require_action_trace:
            self.pending_transitions.append((previous, current))
            self._drain_pending_transitions()
            return

        traced_action = self.actions_by_state_timestamp.pop(
            previous.timestamp_ns, None
        )
        if traced_action is None:
            angle_command = current.angle_command
            speed_command = current.speed_command
            action_source = "latest_motor_fallback"
            self.fallback_action_count += 1
        else:
            angle_command, speed_command = traced_action
            action_source = "state_timestamp_trace"
            self.exact_action_count += 1
        self._record_transition(
            previous,
            current,
            angle_command,
            speed_command,
            action_source,
            self.expert_actions_by_state_timestamp.pop(
                previous.timestamp_ns, None
            ),
        )

    def _drain_pending_transitions(self) -> None:
        """Record exact timestamp joins and discard pre-policy warm-up scans."""
        while self.pending_transitions and not self.finished:
            previous, current = self.pending_transitions[0]
            traced_action = self.actions_by_state_timestamp.get(
                previous.timestamp_ns
            )
            expert_action = self.expert_actions_by_state_timestamp.get(
                previous.timestamp_ns
            )
            if traced_action is not None and (
                expert_action is not None
                or not self.require_expert_action_trace
            ):
                self.pending_transitions.popleft()
                self.actions_by_state_timestamp.pop(previous.timestamp_ns, None)
                self.expert_actions_by_state_timestamp.pop(
                    previous.timestamp_ns, None
                )
                self.exact_action_count += 1
                self._record_transition(
                    previous,
                    current,
                    traced_action[0],
                    traced_action[1],
                    "state_timestamp_trace",
                    expert_action,
                )
                continue

            newer_action_exists = any(
                timestamp_ns > previous.timestamp_ns
                for timestamp_ns in self.actions_by_state_timestamp
            )
            if self.require_expert_action_trace:
                newer_action_exists = newer_action_exists and any(
                    timestamp_ns > previous.timestamp_ns
                    for timestamp_ns in self.expert_actions_by_state_timestamp
                )
            queue_overflow = len(self.pending_transitions) > 32
            if not newer_action_exists and not queue_overflow:
                break

            self.pending_transitions.popleft()
            self.skipped_untraced_count += 1
            if self.skipped_untraced_count <= 3:
                self.get_logger().warning(
                    "skipping scan without an exact action trace: "
                    f"state={previous.timestamp_ns}"
                )

    def _record_transition(
        self,
        previous: RecorderSnapshot,
        current: RecorderSnapshot,
        angle_command: float,
        speed_command: float,
        action_source: str,
        expert_action: tuple[float, float] | None = None,
    ) -> None:
        if self.finished or self.waiting_for_reset:
            return

        dt_sec = max(
            0.001,
            (current.timestamp_ns - previous.timestamp_ns) / 1.0e9,
        )
        self.elapsed_sec += dt_sec
        progress_delta = self.track.progress_delta(
            previous.projection.progress_m,
            current.projection.progress_m,
        )
        plausible_progress_m = max(
            0.25,
            1.5 * abs(float(current.speed_mps)) * dt_sec,
        )
        progress_delta = float(
            np.clip(
                progress_delta,
                -plausible_progress_m,
                plausible_progress_m,
            )
        )
        self.cumulative_progress_m = max(
            0.0, self.cumulative_progress_m + progress_delta
        )
        action_norm = float(
            np.clip(
                angle_command / self.max_steering_command,
                -1.0,
                1.0,
            )
        )
        if expert_action is None:
            expert_angle_command = ""
            expert_speed_command = ""
            expert_action_norm = ""
            expert_action_source = ""
        else:
            self.expert_action_count += 1
            expert_angle_command = float(expert_action[0])
            expert_speed_command = float(expert_action[1])
            expert_action_norm = float(
                np.clip(
                    expert_angle_command / self.max_steering_command,
                    -1.0,
                    1.0,
                )
            )
            expert_action_source = "state_timestamp_expert_trace"
        terminal = self.termination.update(
            dt_sec=dt_sec,
            elapsed_sec=self.elapsed_sec,
            cross_track_error_m=current.projection.cross_track_error_m,
            linear_speed_mps=current.speed_mps,
            speed_command=speed_command,
            collision=self.collision,
            cumulative_forward_progress_m=self.cumulative_progress_m,
            track_length_m=self.track.length_m,
        )
        self.steering_history.append(action_norm)
        track_curvature = self.track.curvature_at(
            current.projection.progress_m,
            sample_distance_m=0.30,
        )
        preview_curvature = self.track.max_abs_curvature_ahead(
            current.projection.progress_m,
            preview_distance_m=1.5,
        )
        reward = calculate_reward(
            projection=current.projection,
            progress_delta_m=progress_delta,
            steering_norm=action_norm,
            previous_steering_norm=self.previous_action_norm,
            linear_speed_mps=current.speed_mps,
            track_curvature=track_curvature,
            preview_curvature=preview_curvature,
            steering_history=tuple(self.steering_history),
            collision=terminal.collision,
            off_track=terminal.off_track,
            stuck=terminal.stuck,
            lap_complete=terminal.lap_complete,
            dt_sec=dt_sec,
        )
        state_image, state_scan = self.writer.save_observation(
            previous.timestamp_ns,
            previous.image_bgr,
            previous.scan_payload,
        )
        next_image, next_scan = self.writer.save_observation(
            current.timestamp_ns,
            current.image_bgr,
            current.scan_payload,
        )
        self.writer.append(
            {
                "episode_id": self.episode_id,
                "step_id": self.step_id,
                "state_timestamp_ns": previous.timestamp_ns,
                "next_timestamp_ns": current.timestamp_ns,
                "state_image_path": state_image,
                "state_scan_path": state_scan,
                "next_image_path": next_image,
                "next_scan_path": next_scan,
                "action_norm": action_norm,
                "angle_command": angle_command,
                "speed_command": speed_command,
                "action_source": action_source,
                "expert_action_norm": expert_action_norm,
                "expert_angle_command": expert_angle_command,
                "expert_speed_command": expert_speed_command,
                "expert_action_source": expert_action_source,
                "reward": reward.total,
                "reward_progress": reward.progress,
                "reward_cross_track": reward.cross_track,
                "reward_heading": reward.heading,
                "reward_steering_rate": reward.steering_rate,
                "reward_safe_speed": reward.safe_speed,
                "reward_unsafe_speed": reward.unsafe_speed,
                "reward_time_efficiency": reward.time_efficiency,
                "reward_lap_time": reward.lap_time,
                "reward_lane_margin": reward.lane_margin,
                "reward_large_oscillation": reward.large_oscillation,
                "reward_terminal": reward.terminal,
                "terminated": int(terminal.terminated),
                "truncated": int(terminal.truncated),
                "termination_reason": terminal.reason,
                "x": previous.x,
                "y": previous.y,
                "yaw": previous.yaw,
                "next_x": current.x,
                "next_y": current.y,
                "next_yaw": current.yaw,
                "speed_mps": current.speed_mps,
                "cross_track_error_m": current.projection.cross_track_error_m,
                "heading_error_rad": current.projection.heading_error_rad,
                "progress_m": current.projection.progress_m,
                "progress_delta_m": progress_delta,
                "cumulative_progress_m": self.cumulative_progress_m,
                "collision": int(self.collision),
            }
        )
        self.step_id += 1
        self.previous_action_norm = action_norm

        done = terminal.terminated or terminal.truncated
        if done:
            self.get_logger().info(
                f"episode {self.episode_id} ended: {terminal.reason}, "
                f"progress={self.cumulative_progress_m:.2f}m"
            )
            if self.auto_stop_on_terminal:
                stop = Float32MultiArray()
                stop.data = [0.0, 0.0]
                self.stop_pub.publish(stop)
            if self.require_reset_after_terminal:
                self.waiting_for_reset = True
                self.pending_transitions.clear()
            else:
                reset = Bool()
                reset.data = True
                self._on_reset(reset)
        if self.max_transitions and self.writer.count >= self.max_transitions:
            self.get_logger().info("maximum transition count reached")
            self.writer.close()
            self.finished = True

    def destroy_node(self):
        self.writer.metadata["exact_action_count"] = self.exact_action_count
        self.writer.metadata["fallback_action_count"] = self.fallback_action_count
        self.writer.metadata["expert_action_count"] = self.expert_action_count
        self.writer.metadata["skipped_untraced_count"] = (
            self.skipped_untraced_count
        )
        self.writer.metadata["pending_transition_count"] = len(
            self.pending_transitions
        )
        self.writer.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TransitionRecorderNode()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
