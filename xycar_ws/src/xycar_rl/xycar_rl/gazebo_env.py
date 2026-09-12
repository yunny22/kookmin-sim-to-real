from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import math
import subprocess
import threading
import time
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Contacts
from ros_gz_interfaces.srv import ControlWorld
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, Float32MultiArray
from tf2_msgs.msg import TFMessage

from il_data_tools.runtime_preprocessing import (
    preprocess_bgr_image,
    preprocess_lidar_ranges,
)
from xycar_rl.reward import RewardWeights, calculate_reward
from xycar_rl.termination import EpisodeTermination, TerminationConfig
from xycar_rl.track_geometry import TrackProjection, TrackReference
from xycar_rl.camera_speed_models import (
    DEFAULT_MAX_SPEED_COMMAND,
    DEFAULT_MIN_SPEED_COMMAND,
    denormalize_speed_command,
    normalize_speed_command,
)


@dataclass(frozen=True)
class OdomSample:
    x: float
    y: float
    yaw: float
    linear_speed_mps: float
    angular_speed_rps: float


@dataclass(frozen=True)
class SensorSnapshot:
    sequence: int
    timestamp_ns: int
    image: np.ndarray
    lidar: np.ndarray
    odom: OdomSample
    collision: bool


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def select_model_transform(transforms, model_name: str):
    """Select the vehicle transform from Gazebo's Pose_V message.

    Gazebo publishes every dynamic entity on the same TFMessage. Relying on
    ``transforms[0]`` can silently project the track or a scenery object,
    which corrupts progress and off-track rewards.
    """
    short_name = str(model_name).split("/")[-1]
    for transform in transforms:
        frame_names = (
            str(getattr(transform, "child_frame_id", "")),
            str(getattr(transform, "header", None).frame_id)
            if getattr(transform, "header", None) is not None
            else "",
        )
        if any(short_name in frame_name for frame_name in frame_names):
            return transform
    # The current ros_gz Pose_V bridge strips frame names. In this world the
    # first transform is the only dynamic model pose (the remaining entries
    # are its links), so preserve that bridge contract explicitly. If names
    # are present but none match, prefer odometry over guessing another model.
    if transforms and all(
        not str(getattr(transform, "child_frame_id", ""))
        and not (
            getattr(transform, "header", None) is not None
            and str(transform.header.frame_id)
        )
        for transform in transforms
    ):
        return transforms[0]
    return None


class _GazeboEnvNode(Node):
    def __init__(
        self,
        *,
        world_name: str,
        model_name: str,
        image_topic: str,
        scan_topic: str,
        odom_topic: str,
        world_pose_topic: str,
        contact_topic: str,
        motor_topic: str,
        episode_reset_topic: str,
        action_trace_topic: str,
        expert_action_trace_topic: str,
        input_width: int,
        input_height: int,
        lidar_points: int,
        sync_tolerance_sec: float,
        require_lidar: bool,
    ) -> None:
        super().__init__("xycar_gym_environment")
        self.bridge = CvBridge()
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.lidar_points = int(lidar_points)
        self.require_lidar = bool(require_lidar)
        self.model_name = model_name
        self.world_name = world_name
        self._condition = threading.Condition()
        self._latest_image: tuple[int, np.ndarray] | None = None
        self._latest_scan: tuple[int, np.ndarray] | None = None
        self.sync_tolerance_ns = int(
            max(0.0, float(sync_tolerance_sec)) * 1.0e9
        )
        self._latest_odom: OdomSample | None = None
        self._latest_world_pose: tuple[float, float, float] | None = None
        self._latest_snapshot: SensorSnapshot | None = None
        self._last_snapshot_sensor_timestamp_ns = -1
        self._sequence = 0
        self._collision = False

        self.motor_pub = self.create_publisher(Float32MultiArray, motor_topic, 10)
        self.episode_reset_pub = self.create_publisher(
            Bool, episode_reset_topic, 10
        )
        self.action_trace_pub = self.create_publisher(
            TwistStamped, action_trace_topic, 10
        )
        self.expert_action_trace_pub = self.create_publisher(
            TwistStamped, expert_action_trace_topic, 10
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
            TFMessage, world_pose_topic, self._on_world_pose, qos_profile_sensor_data
        )
        self.create_subscription(
            Contacts, contact_topic, self._on_contact, qos_profile_sensor_data
        )
        self.control_client = self.create_client(
            ControlWorld, f"/world/{world_name}/control"
        )

    def _on_image(self, msg: Image) -> None:
        image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image = preprocess_bgr_image(
            image_bgr, self.input_width, self.input_height
        )
        with self._condition:
            stamp = msg.header.stamp
            timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            self._latest_image = (timestamp_ns, image)
            self._try_publish_snapshot_locked()

    def _on_odom(self, msg: Odometry) -> None:
        twist = msg.twist.twist
        with self._condition:
            self._latest_odom = OdomSample(
                x=float(msg.pose.pose.position.x),
                y=float(msg.pose.pose.position.y),
                yaw=quaternion_to_yaw(
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ),
                linear_speed_mps=float(twist.linear.x),
                angular_speed_rps=float(twist.angular.z),
            )
            self._try_publish_snapshot_locked()

    def _on_world_pose(self, msg: TFMessage) -> None:
        transform = select_model_transform(msg.transforms, self.model_name)
        if transform is None:
            return
        transform = transform.transform
        rotation = transform.rotation
        yaw = quaternion_to_yaw(
            rotation.x, rotation.y, rotation.z, rotation.w
        )
        with self._condition:
            self._latest_world_pose = (
                float(transform.translation.x),
                float(transform.translation.y),
                float(yaw),
            )

    def _on_scan(self, msg: LaserScan) -> None:
        lidar = preprocess_lidar_ranges(
            msg.ranges,
            msg.range_min,
            msg.range_max,
            self.lidar_points,
        )
        with self._condition:
            scan_stamp = msg.header.stamp
            scan_timestamp_ns = (
                int(scan_stamp.sec) * 1_000_000_000
                + int(scan_stamp.nanosec)
            )
            self._latest_scan = (scan_timestamp_ns, lidar)
            self._try_publish_snapshot_locked()

    def _try_publish_snapshot_locked(self) -> None:
        """Publish the newest camera state, optionally synchronized to LiDAR."""
        if self._latest_image is None or self._latest_odom is None:
            return
        image_timestamp_ns, image = self._latest_image
        if self.require_lidar:
            if self._latest_scan is None:
                return
            sensor_timestamp_ns, lidar = self._latest_scan
            if (
                self.sync_tolerance_ns > 0
                and image_timestamp_ns > 0
                and sensor_timestamp_ns > 0
                and abs(image_timestamp_ns - sensor_timestamp_ns)
                > self.sync_tolerance_ns
            ):
                return
        else:
            sensor_timestamp_ns = image_timestamp_ns
            lidar = np.zeros((2, self.lidar_points), dtype=np.float32)
        if sensor_timestamp_ns <= self._last_snapshot_sensor_timestamp_ns:
            return
        odom = self._latest_odom
        world_x, world_y, world_yaw = (
            self._latest_world_pose
            if self._latest_world_pose is not None
            else (odom.x, odom.y, odom.yaw)
        )
        self._sequence += 1
        self._latest_snapshot = SensorSnapshot(
            sequence=self._sequence,
            timestamp_ns=sensor_timestamp_ns,
            image=image.copy(),
            lidar=lidar.copy(),
            odom=OdomSample(
                x=world_x,
                y=world_y,
                yaw=world_yaw,
                linear_speed_mps=odom.linear_speed_mps,
                angular_speed_rps=odom.angular_speed_rps,
            ),
            collision=self._collision,
        )
        self._last_snapshot_sensor_timestamp_ns = sensor_timestamp_ns
        self._condition.notify_all()

    def _on_contact(self, msg: Contacts) -> None:
        if msg.contacts:
            with self._condition:
                self._collision = True

    def clear_episode_state(self) -> None:
        with self._condition:
            self._collision = False
            self._latest_snapshot = None
            self._latest_image = None
            self._latest_scan = None
            self._latest_odom = None
            self._latest_world_pose = None

    def latest_sequence(self) -> int:
        with self._condition:
            return self._sequence

    def wait_snapshot(
        self,
        after_sequence: int,
        timeout_sec: float,
    ) -> SensorSnapshot | None:
        deadline = time.monotonic() + float(timeout_sec)
        with self._condition:
            while (
                self._latest_snapshot is None
                or self._latest_snapshot.sequence <= after_sequence
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)
            return self._latest_snapshot

    def publish_motor(self, angle_command: float, speed_command: float) -> None:
        message = Float32MultiArray()
        message.data = [float(angle_command), float(speed_command)]
        self.motor_pub.publish(message)

    def publish_action_trace(
        self,
        state_timestamp_ns: int,
        angle_command: float,
        speed_command: float,
        *,
        expert: bool = False,
    ) -> None:
        message = TwistStamped()
        message.header.stamp.sec = int(state_timestamp_ns // 1_000_000_000)
        message.header.stamp.nanosec = int(state_timestamp_ns % 1_000_000_000)
        message.header.frame_id = "rl_state"
        message.twist.angular.z = float(angle_command)
        message.twist.linear.x = float(speed_command)
        publisher = self.expert_action_trace_pub if expert else self.action_trace_pub
        publisher.publish(message)

    def publish_episode_reset(self) -> None:
        message = Bool()
        message.data = True
        self.episode_reset_pub.publish(message)

    @staticmethod
    def _wait_future(future, timeout_sec: float):
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError("Gazebo service request timed out")
            time.sleep(0.002)
        exception = future.exception()
        if exception is not None:
            raise RuntimeError("Gazebo service request failed") from exception
        return future.result()

    def wait_for_services(self, timeout_sec: float = 10.0) -> None:
        if not self.control_client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError("Gazebo world control service is unavailable")

    def control(
        self,
        *,
        pause: bool = True,
        multi_step: int = 0,
        reset_models: bool = False,
        timeout_sec: float = 5.0,
    ) -> None:
        request = ControlWorld.Request()
        request.world_control.pause = bool(pause)
        request.world_control.multi_step = max(0, int(multi_step))
        request.world_control.reset.model_only = bool(reset_models)
        result = self._wait_future(
            self.control_client.call_async(request), timeout_sec
        )
        if not result.success:
            raise RuntimeError("Gazebo rejected world control request")

    def set_model_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        *,
        z: float = 0.05,
        max_attempts: int = 3,
    ) -> None:
        request = (
            f'name: "{self.model_name}", '
            f"position: {{x: {float(x):.12g}, y: {float(y):.12g}, "
            f"z: {float(z):.12g}}}, "
            "orientation: {"
            f"z: {math.sin(float(yaw) * 0.5):.12g}, "
            f"w: {math.cos(float(yaw) * 0.5):.12g}"
            "}"
        )
        detail = "unknown Gazebo service error"
        for attempt in range(1, max(1, int(max_attempts)) + 1):
            try:
                completed = subprocess.run(
                    [
                        "gz",
                        "service",
                        "-s",
                        f"/world/{self.world_name}/set_pose",
                        "--reqtype",
                        "gz.msgs.Pose",
                        "--reptype",
                        "gz.msgs.Boolean",
                        "--timeout",
                        "5000",
                        "--req",
                        request,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=6.0,
                )
                if (
                    completed.returncode == 0
                    and "true" in completed.stdout.lower()
                ):
                    return
                detail = (completed.stderr or completed.stdout).strip()
            except subprocess.TimeoutExpired as error:
                detail = str(error)
            if attempt < max_attempts:
                time.sleep(0.25 * attempt)
        raise RuntimeError(
            f"Gazebo rejected model pose request after {max_attempts} attempts: "
            f"{detail}"
        )


class GazeboXycarEnv(gym.Env):
    """Synchronous configurable-rate Gymnasium environment for Gazebo Sim."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        world_sdf: str | Path,
        world_name: str = "kookmin_xycar_track",
        model_name: str = "xycar_ackermann",
        image_topic: str = "/perception/canonical_road_image",
        scan_topic: str = "/scan",
        odom_topic: str = "/model/xycar_ackermann/odometry",
        world_pose_topic: str = "/world/kookmin_xycar_track/dynamic_pose/info",
        contact_topic: str = "/xycar_contact",
        motor_topic: str = "/xycar_motor",
        episode_reset_topic: str = "/rl/episode_reset",
        action_trace_topic: str = "/rl/action_applied",
        expert_action_trace_topic: str = "/rl/action_expert",
        input_width: int = 160,
        input_height: int = 90,
        lidar_points: int = 360,
        control_rate_hz: float = 10.0,
        physics_step_size_sec: float = 0.001,
        speed_command: float = 4.0,
        variable_speed: bool = False,
        require_lidar: bool | None = None,
        min_speed_command: float = DEFAULT_MIN_SPEED_COMMAND,
        max_speed_command: float = DEFAULT_MAX_SPEED_COMMAND,
        max_steering_command: float = 42.0,
        # Real-car calibration moved the old +10 cm right target 10 cm left.
        target_right_offset_m: float = 0.0,
        reset_lateral_error_m: float = 0.22,
        reset_yaw_error_rad: float = math.radians(14.0),
        observation_timeout_sec: float = 2.0,
        # Canonical perception is CPU-bound and can publish a stamped frame
        # a few source frames late in Gazebo; keep a bounded 250ms join gate.
        sensor_sync_tolerance_sec: float = 0.25,
        command_delivery_wait_sec: float = 0.02,
        termination_config: TerminationConfig = TerminationConfig(),
        reward_weights: RewardWeights = RewardWeights(),
    ) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.lidar_points = int(lidar_points)
        self.control_dt_sec = 1.0 / float(control_rate_hz)
        self.physics_steps = max(
            1,
            int(round(self.control_dt_sec / float(physics_step_size_sec))),
        )
        self.speed_command = float(speed_command)
        self.variable_speed = bool(variable_speed)
        self.require_lidar = (
            not self.variable_speed if require_lidar is None else bool(require_lidar)
        )
        self.min_speed_command = float(min_speed_command)
        self.max_speed_command = float(max_speed_command)
        if self.max_speed_command <= self.min_speed_command:
            raise ValueError("max_speed_command must be greater than min_speed_command")
        self.current_speed_command = float(speed_command)
        self.max_steering_command = abs(float(max_steering_command))
        self.reset_lateral_error_m = abs(float(reset_lateral_error_m))
        self.reset_yaw_error_rad = abs(float(reset_yaw_error_rad))
        self.observation_timeout_sec = float(observation_timeout_sec)
        self.command_delivery_wait_sec = max(
            0.0, float(command_delivery_wait_sec)
        )
        self.reward_weights = reward_weights
        self.track = TrackReference.from_sdf(
            world_sdf, target_right_offset_m=target_right_offset_m
        )
        self.termination = EpisodeTermination(termination_config)

        action_dimensions = 2 if self.variable_speed else 1
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(action_dimensions,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    0.0,
                    1.0,
                    shape=(3, self.input_height, self.input_width),
                    dtype=np.float32,
                ),
                "lidar": spaces.Box(
                    0.0,
                    1.0,
                    shape=(2, self.lidar_points),
                    dtype=np.float32,
                ),
                "aux": spaces.Box(
                    low=np.asarray([-1.0, -1.0], dtype=np.float32),
                    high=np.asarray([1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self.node = _GazeboEnvNode(
            world_name=world_name,
            model_name=model_name,
            image_topic=image_topic,
            scan_topic=scan_topic,
            odom_topic=odom_topic,
            world_pose_topic=world_pose_topic,
            contact_topic=contact_topic,
            motor_topic=motor_topic,
            episode_reset_topic=episode_reset_topic,
            action_trace_topic=action_trace_topic,
            expert_action_trace_topic=expert_action_trace_topic,
            input_width=self.input_width,
            input_height=self.input_height,
            lidar_points=self.lidar_points,
            sync_tolerance_sec=sensor_sync_tolerance_sec,
            require_lidar=self.require_lidar,
        )
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(
            target=self.executor.spin,
            name="xycar-gym-ros-executor",
            daemon=True,
        )
        self.spin_thread.start()
        self.node.wait_for_services()
        self.node.control(pause=True)

        self.previous_projection: TrackProjection | None = None
        self.previous_steering_norm = 0.0
        self.steering_history: deque[float] = deque(maxlen=12)
        self.cumulative_progress_m = 0.0
        self.elapsed_sec = 0.0
        self.last_snapshot: SensorSnapshot | None = None

    def _observation(
        self, snapshot: SensorSnapshot, steering_norm: float
    ) -> dict[str, np.ndarray]:
        speed_norm = (
            normalize_speed_command(
                self.current_speed_command,
                self.min_speed_command,
                self.max_speed_command,
            )
            if self.variable_speed
            else np.clip(self.current_speed_command / 100.0, -1.0, 1.0)
        )
        return {
            "image": snapshot.image.astype(np.float32, copy=False),
            "lidar": snapshot.lidar.astype(np.float32, copy=False),
            "aux": np.asarray(
                [float(steering_norm), float(speed_norm)], dtype=np.float32
            ),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        options = options or {}
        progress_fraction = float(
            options.get("progress_fraction", self.np_random.uniform(0.0, 1.0))
        )
        lateral_error = float(
            options.get(
                "lateral_error_m",
                self.np_random.uniform(
                    -self.reset_lateral_error_m,
                    self.reset_lateral_error_m,
                ),
            )
        )
        yaw_error = float(
            options.get(
                "yaw_error_rad",
                self.np_random.uniform(
                    -self.reset_yaw_error_rad, self.reset_yaw_error_rad
                ),
            )
        )
        pose = self.track.pose_at(
            progress_fraction,
            lateral_error_m=lateral_error,
            yaw_error_rad=yaw_error,
        )

        self.node.publish_motor(0.0, 0.0)
        self.node.control(pause=True, reset_models=True)
        self.node.set_model_pose(pose.x, pose.y, pose.yaw)
        self.node.clear_episode_state()
        self.node.publish_episode_reset()
        before = self.node.latest_sequence()
        self.node.publish_motor(0.0, 0.0)
        # ROS callbacks run on wall time while Gazebo is paused. Give the motor
        # bridge time to clear delayed commands before advancing simulation.
        time.sleep(0.08)
        self.node.control(pause=True, multi_step=max(self.physics_steps, 400))
        snapshot = self.node.wait_snapshot(before, self.observation_timeout_sec)
        if snapshot is None:
            required = "camera/odometry"
            if self.require_lidar:
                required = "camera/LiDAR/odometry"
            raise TimeoutError(f"no synchronized {required} after reset")

        projection = self.track.project(
            snapshot.odom.x, snapshot.odom.y, snapshot.odom.yaw
        )
        self.termination.reset()
        self.previous_projection = projection
        self.previous_steering_norm = 0.0
        self.steering_history.clear()
        self.current_speed_command = self.speed_command
        self.cumulative_progress_m = 0.0
        self.elapsed_sec = 0.0
        self.last_snapshot = snapshot
        info = self._info(snapshot, projection, "reset")
        return self._observation(snapshot, 0.0), info

    def step(self, action, *, trace_action=None):
        action_values = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_values.size < (2 if self.variable_speed else 1):
            raise ValueError("environment action has too few values")
        steering_norm = float(np.clip(action_values[0], -1.0, 1.0))
        if self.variable_speed:
            speed_norm = float(np.clip(action_values[1], -1.0, 1.0))
            speed_command = denormalize_speed_command(
                speed_norm,
                self.min_speed_command,
                self.max_speed_command,
            )
        else:
            speed_command = self.speed_command
        self.current_speed_command = float(speed_command)
        before = self.node.latest_sequence()
        angle_command = steering_norm * self.max_steering_command
        if self.last_snapshot is None:
            raise RuntimeError("environment has no state for action tracing")
        self.node.publish_action_trace(
            self.last_snapshot.timestamp_ns,
            angle_command,
            speed_command,
        )
        if trace_action is not None:
            expert_values = np.asarray(trace_action, dtype=np.float32).reshape(-1)
            expert_steering_norm = float(
                np.clip(expert_values[0], -1.0, 1.0)
            )
            if self.variable_speed and expert_values.size >= 2:
                expert_speed_command = denormalize_speed_command(
                    float(np.clip(expert_values[1], -1.0, 1.0)),
                    self.min_speed_command,
                    self.max_speed_command,
                )
            else:
                expert_speed_command = speed_command
            self.node.publish_action_trace(
                self.last_snapshot.timestamp_ns,
                expert_steering_norm * self.max_steering_command,
                expert_speed_command,
                expert=True,
            )
        self.node.publish_motor(angle_command, speed_command)
        if self.command_delivery_wait_sec > 0.0:
            time.sleep(self.command_delivery_wait_sec)
        self.node.control(pause=True, multi_step=self.physics_steps)
        snapshot = self.node.wait_snapshot(before, self.observation_timeout_sec)
        observation_timed_out = snapshot is None
        if snapshot is None:
            if self.last_snapshot is None or self.previous_projection is None:
                raise RuntimeError("environment has no valid observation")
            snapshot = self.last_snapshot
        step_dt_sec = self.control_dt_sec
        if (
            not observation_timed_out
            and self.last_snapshot is not None
            and snapshot.timestamp_ns > self.last_snapshot.timestamp_ns
        ):
            step_dt_sec = (
                snapshot.timestamp_ns - self.last_snapshot.timestamp_ns
            ) * 1.0e-9

        projection = self.track.project(
            snapshot.odom.x,
            snapshot.odom.y,
            snapshot.odom.yaw,
            hint_segment_index=(
                None
                if self.previous_projection is None
                else self.previous_projection.segment_index
            ),
        )
        progress_delta = 0.0
        if self.previous_projection is not None and not observation_timed_out:
            progress_delta = self.track.progress_delta(
                self.previous_projection.progress_m, projection.progress_m
            )
            plausible_progress_m = max(
                0.25,
                1.5
                * abs(float(snapshot.odom.linear_speed_mps))
                * step_dt_sec,
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
        self.elapsed_sec += step_dt_sec
        terminal = self.termination.update(
            dt_sec=step_dt_sec,
            elapsed_sec=self.elapsed_sec,
            cross_track_error_m=projection.cross_track_error_m,
            linear_speed_mps=snapshot.odom.linear_speed_mps,
            speed_command=speed_command,
            collision=snapshot.collision,
            cumulative_forward_progress_m=self.cumulative_progress_m,
            track_length_m=self.track.length_m,
            observation_timed_out=observation_timed_out,
        )
        self.steering_history.append(steering_norm)
        track_curvature = self.track.curvature_at(
            projection.progress_m,
            sample_distance_m=0.30,
        )
        preview_curvature = self.track.max_abs_curvature_ahead(
            projection.progress_m,
            preview_distance_m=1.5,
        )
        reward = calculate_reward(
            projection=projection,
            progress_delta_m=progress_delta,
            steering_norm=steering_norm,
            previous_steering_norm=self.previous_steering_norm,
            linear_speed_mps=snapshot.odom.linear_speed_mps,
            track_curvature=track_curvature,
            preview_curvature=preview_curvature,
            steering_history=tuple(self.steering_history),
            collision=terminal.collision,
            off_track=terminal.off_track,
            stuck=terminal.stuck,
            lap_complete=terminal.lap_complete,
            dt_sec=step_dt_sec,
            weights=self.reward_weights,
        )
        self.previous_projection = projection
        self.previous_steering_norm = steering_norm
        self.last_snapshot = snapshot
        info = self._info(snapshot, projection, terminal.reason)
        info["reward_terms"] = reward.__dict__
        info["progress_delta_m"] = progress_delta
        info["step_dt_sec"] = step_dt_sec
        info["track_curvature"] = track_curvature
        info["preview_curvature"] = preview_curvature
        return (
            self._observation(snapshot, steering_norm),
            reward.total,
            terminal.terminated,
            terminal.truncated,
            info,
        )

    def _info(
        self,
        snapshot: SensorSnapshot,
        projection: TrackProjection,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "reason": reason,
            "sequence": snapshot.sequence,
            "x": snapshot.odom.x,
            "y": snapshot.odom.y,
            "yaw": snapshot.odom.yaw,
            "speed_mps": snapshot.odom.linear_speed_mps,
            "speed_command": self.current_speed_command,
            "cross_track_error_m": projection.cross_track_error_m,
            "heading_error_rad": projection.heading_error_rad,
            "progress_m": projection.progress_m,
            "progress_fraction": projection.progress_fraction,
            "cumulative_progress_m": self.cumulative_progress_m,
            "elapsed_sec": self.elapsed_sec,
            "collision": snapshot.collision,
        }

    def close(self) -> None:
        try:
            self.node.publish_motor(0.0, 0.0)
        except Exception:
            pass
        self.executor.shutdown(timeout_sec=5.0)
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=5.0)
        self.node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
