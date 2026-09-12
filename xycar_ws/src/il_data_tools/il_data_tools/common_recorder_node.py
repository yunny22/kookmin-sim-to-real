#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import Float32MultiArray, String

try:
    from xycar_msgs.msg import XycarMotor
except ImportError:  # pragma: no cover
    XycarMotor = None

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover
    cv2 = None
    CvBridge = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from il_data_tools.record_schema import (
    atomic_write_json,
    extract_motor_command,
    image_suffix,
    make_session_dir,
    open_samples_csv,
    parse_allowed_labels,
    relative_to_session,
    ros_time_to_ns,
    stamp_to_ns,
    string_from_msg,
)
from il_data_tools.canonical_artifacts import canonical_lane_observation_present
from il_data_tools.sync_buffer import TimedBuffer
from il_data_tools.recorder_state import (
    DiskSpaceGuard,
    LabelLatch,
    wait_for_thread_shutdown,
)


def rate_limit_allows(
    stamp_ns: int,
    last_stamp_ns: Optional[int],
    max_rate_hz: float,
) -> bool:
    """Accept near-periodic sensor stamps without halving a nominal rate."""
    if max_rate_hz <= 0.0 or last_stamp_ns is None:
        return True
    period_ns = int(1e9 / max_rate_hz)
    tolerance_ns = min(10_000_000, int(period_ns * 0.10))
    return stamp_ns - last_stamp_ns >= period_ns - tolerance_ns


class BadDataPreroll:
    """Delay samples so a later stop can invalidate the preceding time window."""

    def __init__(self, window_ns: int) -> None:
        self.window_ns = max(0, int(window_ns))
        self.samples = deque()

    def push(self, sample: Any) -> list[Any]:
        if self.window_ns <= 0:
            return [sample]
        ready = self._pop_before(sample.timestamp_ns - self.window_ns)
        self.samples.append(sample)
        return ready

    def stop(self, stop_stamp_ns: int) -> tuple[list[Any], int]:
        if self.window_ns <= 0:
            return [], 0
        safe = self._pop_before(stop_stamp_ns - self.window_ns)
        discarded = len(self.samples)
        self.samples.clear()
        return safe, discarded

    def discard_all(self) -> int:
        discarded = len(self.samples)
        self.samples.clear()
        return discarded

    def _pop_before(self, cutoff_ns: int) -> list[Any]:
        ready = []
        while self.samples and self.samples[0].timestamp_ns < cutoff_ns:
            ready.append(self.samples.popleft())
        return ready

    def __len__(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class PendingSample:
    """A fully synchronized sample waiting for background disk I/O."""

    timestamp_ns: int
    scan_timestamp_ns: Optional[int]
    scan_time_offset_ms: Optional[float]
    front_msg: Image
    scan_msg: Optional[LaserScan]
    imu_msg: Optional[Imu]
    odom_msg: Optional[Odometry]
    motor_angle: Any
    motor_speed: Any
    mission_label: str


class ILCommonRecorder(Node):
    """Subscribe-only dataset recorder for imitation-learning samples."""

    def __init__(self) -> None:
        super().__init__("il_common_recorder")
        self._declare_parameters()
        self.params = self._read_parameters()
        self.run_manifest = self._load_run_manifest(
            self.params["run_manifest_path"]
        )
        if self.params["require_scan"] and not self.params["save_scan_npz"]:
            raise ValueError("require_scan=true requires save_scan_npz=true")
        if self.params["approximate_sync_tolerance_sec"] <= 0.0:
            raise ValueError("approximate_sync_tolerance_sec must be positive")
        self.motor_msg_type, self.resolved_motor_msg_type_name = self._resolve_motor_msg_type()
        self.allowed_labels = parse_allowed_labels(
            self.params["allowed_labels"], self.params["dataset_profile"]
        )
        self.session_id, self.session_dir = make_session_dir(
            self.params["output_root"],
            self.params["dataset_profile"],
            self.params["session_name"],
            auto_increment=self.params["session_auto_increment"],
        )
        self.csv_handle, self.csv_writer = open_samples_csv(
            self.session_dir / "samples.csv"
        )
        self.imu_handle = None
        self.imu_writer = None
        self.odom_handle = None
        self.odom_writer = None
        self._open_optional_debug_writers()
        self.bridge = CvBridge() if CvBridge is not None else None

        self.front_buffer = TimedBuffer()
        self.scan_buffer = TimedBuffer()
        self.imu_buffer = TimedBuffer()
        self.odom_buffer = TimedBuffer()
        self.motor_buffer = TimedBuffer()
        # Mission label is state, not a timestamp-synchronized sensor.  The last
        # received value remains active until another label arrives.
        self.label_latch = LabelLatch(self.params["default_mission_label"])
        self.sync_pending_images = deque()
        self.preroll_lock = threading.Lock()
        self.bad_data_preroll = BadDataPreroll(
            int(self.params["bad_data_preroll_sec"] * 1e9)
        )

        self.recording_enabled = self.params["enable_recording_on_start"]
        self.started_at = datetime.now()
        self.enqueued_sample_count = 0
        self.sample_count = 0
        self.image_count = 0
        self.scan_count = 0
        self.skipped_missing_motor = 0
        self.skipped_missing_scan = 0
        self.skipped_unsynced_scan = 0
        self.skipped_label_filter = 0
        self.skipped_speed_filter = 0
        self.skipped_blank_canonical = 0
        self.skipped_rate_limit = 0
        self.skipped_disk_limit = 0
        self.dropped_queue_full = 0
        self.dropped_sync_queue_full = 0
        self.discarded_bad_data_preroll = 0
        self.discarded_uncommitted_preroll = 0
        self.writer_errors = []
        self.stop_reason = ""
        self.auto_exit_requested = False
        self.auto_exit_triggered = False
        self.writer_shutdown_wait_sec = 0.0
        self._closed = False
        self.last_rate_accepted_ns: Optional[int] = None
        self.last_zero_speed_ns: Optional[int] = None
        self.last_free_disk_gb: Optional[float] = None
        self.last_debug_sec = 0.0
        self.warnings = []
        self.disk_guard = DiskSpaceGuard(
            min_free_gb=self.params["min_free_disk_gb"],
            period_sec=self.params["disk_check_period_sec"],
            enabled=self.params["stop_on_low_disk"],
        )

        self.pending_queue: queue.Queue[PendingSample] = queue.Queue(
            maxsize=self.params["writer_queue_size"]
        )
        self.writer_stop = threading.Event()
        self.accepting_samples = True
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name="il-data-writer",
            daemon=True,
        )
        self.writer_thread.start()

        self._write_static_files()
        self._create_subscriptions()
        self.create_timer(self.params["debug_print_period_sec"], self._debug_tick)
        self.create_timer(0.01, self._drain_sync_pending_images)
        self.create_timer(0.10, self._auto_exit_tick)
        self.get_logger().info(f"IL dataset session: {self.session_dir}")
        self.get_logger().warn(
            "Subscribe-only recorder: this node never publishes /xycar_motor."
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "output_root": "~/xycar_ws/datasets/il",
            "session_name": "session",
            "session_auto_increment": True,
            "dataset_profile": "drive",
            "allowed_labels": "",
            "camera_front_topic": "/image_raw",
            "scan_topic": "/scan",
            "imu_topic": "/imu",
            "odom_topic": "/odom",
            "motor_topic": "/xycar_motor",
            "motor_msg_type": "float32_multi_array",
            "mission_label_topic": "/il/mission_label",
            "default_mission_label": "idle",
            "run_manifest_path": "",
            "save_front_image": True,
            "save_scan_npz": False,
            "require_scan": False,
            "save_imu": False,
            "save_odom": False,
            "image_format": "jpg",
            "jpeg_quality": 90,
            "max_save_rate_hz": 10.0,
            "min_abs_speed_to_save": 0.0,
            "max_abs_speed_to_save": 0.0,
            "save_when_stopped": False,
            "require_motor_command": True,
            "approximate_sync_tolerance_sec": 0.10,
            "sync_wait_sec": 0.10,
            "flush_every_n_samples": 20,
            "max_session_duration_sec": 0.0,
            "max_samples": 0,
            "exit_on_limit_reached": False,
            "max_disk_usage_gb": 0.0,
            "writer_queue_size": 128,
            "writer_shutdown_timeout_sec": 15.0,
            "min_free_disk_gb": 10.0,
            "disk_check_period_sec": 5.0,
            "stop_on_low_disk": True,
            "enable_recording_on_start": True,
            "exclude_bad_data": True,
            "exclude_idle": True,
            "exclude_zero_speed": False,
            "exclude_blank_canonical": False,
            "canonical_min_lane_pixels": 15,
            "canonical_min_lane_rows": 6,
            "bad_data_preroll_sec": 0.0,
            "debug_print_period_sec": 5.0,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)

    def _read_parameters(self) -> Dict[str, Any]:
        return {
            "output_root": self._get_str("output_root"),
            "session_name": self._get_str("session_name"),
            "session_auto_increment": self._get_bool("session_auto_increment"),
            "dataset_profile": self._get_str("dataset_profile"),
            "allowed_labels": self.get_parameter("allowed_labels").value,
            "camera_front_topic": self._get_str("camera_front_topic"),
            "scan_topic": self._get_str("scan_topic"),
            "imu_topic": self._get_str("imu_topic"),
            "odom_topic": self._get_str("odom_topic"),
            "motor_topic": self._get_str("motor_topic"),
            "motor_msg_type": self._get_str("motor_msg_type"),
            "mission_label_topic": self._get_str("mission_label_topic"),
            "default_mission_label": self._get_str("default_mission_label"),
            "run_manifest_path": self._get_str("run_manifest_path"),
            "save_front_image": self._get_bool("save_front_image"),
            "save_scan_npz": self._get_bool("save_scan_npz"),
            "require_scan": self._get_bool("require_scan"),
            "save_imu": self._get_bool("save_imu"),
            "save_odom": self._get_bool("save_odom"),
            "image_format": image_suffix(self._get_str("image_format")),
            "jpeg_quality": self._get_int("jpeg_quality"),
            "max_save_rate_hz": self._get_float("max_save_rate_hz"),
            "min_abs_speed_to_save": self._get_float("min_abs_speed_to_save"),
            "max_abs_speed_to_save": self._get_float("max_abs_speed_to_save"),
            "save_when_stopped": self._get_bool("save_when_stopped"),
            "require_motor_command": self._get_bool("require_motor_command"),
            "approximate_sync_tolerance_sec": self._get_float(
                "approximate_sync_tolerance_sec"
            ),
            "sync_wait_sec": max(0.0, self._get_float("sync_wait_sec")),
            "flush_every_n_samples": self._get_int("flush_every_n_samples"),
            "max_session_duration_sec": self._get_float("max_session_duration_sec"),
            "max_samples": max(0, self._get_int("max_samples")),
            "exit_on_limit_reached": self._get_bool("exit_on_limit_reached"),
            "max_disk_usage_gb": self._get_float("max_disk_usage_gb"),
            "writer_queue_size": max(1, self._get_int("writer_queue_size")),
            "writer_shutdown_timeout_sec": max(
                1.0, self._get_float("writer_shutdown_timeout_sec")
            ),
            "min_free_disk_gb": max(0.0, self._get_float("min_free_disk_gb")),
            "disk_check_period_sec": max(
                0.5, self._get_float("disk_check_period_sec")
            ),
            "stop_on_low_disk": self._get_bool("stop_on_low_disk"),
            "enable_recording_on_start": self._get_bool("enable_recording_on_start"),
            "exclude_bad_data": self._get_bool("exclude_bad_data"),
            "exclude_idle": self._get_bool("exclude_idle"),
            "exclude_zero_speed": self._get_bool("exclude_zero_speed"),
            "exclude_blank_canonical": self._get_bool(
                "exclude_blank_canonical"
            ),
            "canonical_min_lane_pixels": max(
                1, self._get_int("canonical_min_lane_pixels")
            ),
            "canonical_min_lane_rows": max(
                1, self._get_int("canonical_min_lane_rows")
            ),
            "bad_data_preroll_sec": max(
                0.0, self._get_float("bad_data_preroll_sec")
            ),
            "debug_print_period_sec": max(self._get_float("debug_print_period_sec"), 1.0),
        }

    def _get_str(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _get_bool(self, name: str) -> bool:
        value = self.get_parameter(name).value
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "on"}

    def _get_int(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _get_float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _load_run_manifest(self, manifest_path: str) -> Dict[str, Any]:
        if not manifest_path.strip():
            return {}
        path = Path(manifest_path).expanduser().resolve()
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f"could not read run manifest {path}: {exc}")
            return {"path": str(path), "load_error": str(exc)}
        if not isinstance(data, dict):
            return {"path": str(path), "load_error": "manifest is not an object"}
        return data

    def _resolve_motor_msg_type(self):
        value = self.params["motor_msg_type"].strip().lower()
        xycar_values = {"xycar", "xycar_msgs/msg/xycarmotor"}
        float_array_values = {"float32_multi_array", "std_msgs/msg/float32multiarray"}

        if value == "auto":
            if XycarMotor is not None:
                self.get_logger().info(
                    "motor_msg_type=auto resolved to xycar_msgs/msg/XycarMotor. "
                    "If /xycar_motor is std_msgs/msg/Float32MultiArray, use "
                    "motor_msg_type:=float32_multi_array."
                )
                return XycarMotor, "xycar_msgs/msg/XycarMotor"
            self.get_logger().warn(
                "xycar_msgs is not installed; motor_msg_type=auto resolved to "
                "std_msgs/msg/Float32MultiArray. Check `ros2 topic info /xycar_motor -v` "
                "and set motor_msg_type explicitly if needed."
            )
            return Float32MultiArray, "std_msgs/msg/Float32MultiArray"

        if value in xycar_values:
            if XycarMotor is None:
                raise RuntimeError(
                    "motor_msg_type:=xycar requested, but xycar_msgs is not installed. "
                    "Install xycar_msgs, or use motor_msg_type:=float32_multi_array "
                    "when /xycar_motor is std_msgs/msg/Float32MultiArray."
                )
            self.get_logger().info("motor_msg_type resolved to xycar_msgs/msg/XycarMotor")
            return XycarMotor, "xycar_msgs/msg/XycarMotor"

        if value in float_array_values:
            self.get_logger().warn(
                "motor_msg_type resolved to std_msgs/msg/Float32MultiArray. "
                "Use motor_msg_type:=xycar only when /xycar_motor is published as "
                "xycar_msgs/msg/XycarMotor."
            )
            return Float32MultiArray, "std_msgs/msg/Float32MultiArray"

        raise ValueError(
            "motor_msg_type must be one of: auto, xycar, "
            "xycar_msgs/msg/XycarMotor, float32_multi_array, "
            "std_msgs/msg/Float32MultiArray"
        )

    def _write_static_files(self) -> None:
        self._write_metadata(final=False)

    def _open_optional_debug_writers(self) -> None:
        debug_dir = self.session_dir / "debug"
        if self.params["save_imu"]:
            self.imu_handle = (debug_dir / "imu.csv").open(
                "w", newline="", encoding="utf-8"
            )
            self.imu_writer = csv.DictWriter(
                self.imu_handle,
                fieldnames=[
                    "sample_timestamp_ns",
                    "imu_timestamp_ns",
                    "frame_id",
                    "orientation_x",
                    "orientation_y",
                    "orientation_z",
                    "orientation_w",
                    "angular_velocity_x",
                    "angular_velocity_y",
                    "angular_velocity_z",
                    "linear_acceleration_x",
                    "linear_acceleration_y",
                    "linear_acceleration_z",
                ],
            )
            self.imu_writer.writeheader()
        if self.params["save_odom"]:
            self.odom_handle = (debug_dir / "odom.csv").open(
                "w", newline="", encoding="utf-8"
            )
            self.odom_writer = csv.DictWriter(
                self.odom_handle,
                fieldnames=[
                    "sample_timestamp_ns",
                    "odom_timestamp_ns",
                    "frame_id",
                    "child_frame_id",
                    "position_x",
                    "position_y",
                    "position_z",
                    "orientation_x",
                    "orientation_y",
                    "orientation_z",
                    "orientation_w",
                    "linear_x",
                    "linear_y",
                    "linear_z",
                    "angular_x",
                    "angular_y",
                    "angular_z",
                ],
            )
            self.odom_writer.writeheader()

    def _create_subscriptions(self) -> None:
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            Image,
            self.params["camera_front_topic"],
            self._front_image_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan,
            self.params["scan_topic"],
            lambda msg: self.scan_buffer.add(stamp_to_ns(msg), msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            self.params["imu_topic"],
            lambda msg: self.imu_buffer.add(stamp_to_ns(msg), msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            self.params["odom_topic"],
            lambda msg: self.odom_buffer.add(stamp_to_ns(msg), msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            self.motor_msg_type,
            self.params["motor_topic"],
            self._motor_cb,
            reliable_qos,
        )
        self.create_subscription(
            String,
            self.params["mission_label_topic"],
            self._mission_label_cb,
            reliable_qos,
        )
        self.get_logger().info(
            "QoS: camera/scan/imu/odom=SensorDataQoS(BEST_EFFORT), "
            "motor/mission_label=RELIABLE(depth=10)"
        )

    def _front_image_cb(self, msg: Image) -> None:
        stamp_ns = stamp_to_ns(msg, ros_time_to_ns(self.get_clock().now()))
        self.front_buffer.add(stamp_ns, msg)
        if not self.recording_enabled:
            return
        max_pending = max(32, self.params["writer_queue_size"] * 2)
        if len(self.sync_pending_images) >= max_pending:
            self.sync_pending_images.popleft()
            self.dropped_sync_queue_full += 1
            self._warn_once("sensor synchronization queue is full; oldest image dropped")
        self.sync_pending_images.append((time.monotonic(), stamp_ns, msg))

    def _drain_sync_pending_images(self) -> None:
        """Delay matching so a slightly later scan can still pair with the image."""
        if not self.recording_enabled:
            self.sync_pending_images.clear()
            return
        now = time.monotonic()
        wait_sec = self.params["sync_wait_sec"]
        while self.sync_pending_images:
            received_at, stamp_ns, msg = self.sync_pending_images[0]
            if now - received_at < wait_sec:
                break
            self.sync_pending_images.popleft()
            self._try_record_sample(stamp_ns, msg)

    def _motor_cb(self, msg: Any) -> None:
        stamp_ns = stamp_to_ns(msg, ros_time_to_ns(self.get_clock().now()))
        self.motor_buffer.add(stamp_ns, msg)
        _, speed, _ = extract_motor_command(msg)
        if speed is None or abs(float(speed)) > 1.0e-6:
            return
        self.last_zero_speed_ns = stamp_ns
        with self.preroll_lock:
            safe, discarded = self.bad_data_preroll.stop(stamp_ns)
        self.discarded_bad_data_preroll += discarded
        self._enqueue_pending_samples(safe)

    def _mission_label_cb(self, msg: String) -> None:
        now_ns = ros_time_to_ns(self.get_clock().now())
        label = string_from_msg(msg).strip()
        if not label:
            return
        self.label_latch.update(label, now_ns)

    def _try_record_sample(self, stamp_ns: int, front_msg: Image) -> None:
        if self._sample_limit_reached():
            self._request_sample_limit_stop()
            return
        if not self._within_duration_limit():
            self.recording_enabled = False
            self._warn_once("max_session_duration_sec reached; recording stopped")
            return
        if not self._within_rate_limit(stamp_ns):
            self.skipped_rate_limit += 1
            return
        if self.params["exclude_blank_canonical"]:
            if self.bridge is None:
                raise RuntimeError(
                    "cv_bridge is required when exclude_blank_canonical=true"
                )
            try:
                canonical_image = self.bridge.imgmsg_to_cv2(
                    front_msg, desired_encoding="bgr8"
                )
            except Exception as exc:
                self.skipped_blank_canonical += 1
                self._warn_once(f"canonical image decode failed: {exc}")
                return
            if not canonical_lane_observation_present(
                canonical_image,
                min_lane_pixels=self.params["canonical_min_lane_pixels"],
                min_lane_rows=self.params["canonical_min_lane_rows"],
            ):
                self.skipped_blank_canonical += 1
                return
        if not self._disk_space_available():
            self.skipped_disk_limit += 1
            self.recording_enabled = False
            self.accepting_samples = False
            self.stop_reason = "low_disk_space"
            self._warn_once("low disk space; recording stopped safely")
            return

        tolerance_ns = int(self.params["approximate_sync_tolerance_sec"] * 1e9)
        motor_item = self.motor_buffer.nearest(stamp_ns, tolerance_ns)
        if motor_item is None:
            self.skipped_missing_motor += 1
            if self.params["require_motor_command"]:
                return
            motor_angle, motor_speed = "", ""
        else:
            motor_angle, motor_speed, reason = extract_motor_command(motor_item.msg)
            if motor_angle is None or motor_speed is None:
                self.skipped_missing_motor += 1
                self._warn_once(
                    f"invalid motor command on {self.params['motor_topic']}: {reason}"
                )
                if self.params["require_motor_command"]:
                    return
                motor_angle, motor_speed = "", ""

        mission_label = self.label_latch.active
        if not self._label_allowed(mission_label):
            self.skipped_label_filter += 1
            return

        if motor_item is not None and not self._speed_allowed(float(motor_speed)):
            self.skipped_speed_filter += 1
            return

        if self._inside_last_zero_speed_preroll(stamp_ns):
            self.discarded_bad_data_preroll += 1
            return

        if not self.accepting_samples:
            return
        scan_item = self.scan_buffer.nearest(stamp_ns, tolerance_ns)
        if self.params["require_scan"] and scan_item is None:
            if len(self.scan_buffer) == 0:
                self.skipped_missing_scan += 1
            else:
                self.skipped_unsynced_scan += 1
            return
        scan_timestamp_ns = scan_item.stamp_ns if scan_item is not None else None
        scan_time_offset_ms = (
            abs(scan_timestamp_ns - stamp_ns) / 1_000_000.0
            if scan_timestamp_ns is not None
            else None
        )
        imu_item = self.imu_buffer.nearest(stamp_ns, tolerance_ns)
        odom_item = self.odom_buffer.nearest(stamp_ns, tolerance_ns)
        pending = PendingSample(
            timestamp_ns=stamp_ns,
            scan_timestamp_ns=scan_timestamp_ns,
            scan_time_offset_ms=scan_time_offset_ms,
            front_msg=front_msg,
            scan_msg=scan_item.msg if scan_item else None,
            imu_msg=imu_item.msg if imu_item else None,
            odom_msg=odom_item.msg if odom_item else None,
            motor_angle=motor_angle,
            motor_speed=motor_speed,
            mission_label=mission_label,
        )
        self.last_rate_accepted_ns = stamp_ns
        with self.preroll_lock:
            ready = self.bad_data_preroll.push(pending)
        self._enqueue_pending_samples(ready)

    def _inside_last_zero_speed_preroll(self, stamp_ns: int) -> bool:
        if self.last_zero_speed_ns is None:
            return False
        window_ns = int(self.params["bad_data_preroll_sec"] * 1e9)
        return self.last_zero_speed_ns - window_ns <= stamp_ns <= self.last_zero_speed_ns

    def _enqueue_pending_samples(self, samples) -> None:
        for pending in samples:
            if not self.accepting_samples or self._sample_limit_reached():
                return
            try:
                self.pending_queue.put_nowait(pending)
                self.enqueued_sample_count += 1
                if self._sample_limit_reached():
                    self._request_sample_limit_stop()
                    return
            except queue.Full:
                self.dropped_queue_full += 1
                self._warn_once("writer queue is full; newest samples are being dropped")
                return

    def _writer_loop(self) -> None:
        """Serialize JPEG/NPZ/CSV writes away from ROS callbacks."""
        while not self.writer_stop.is_set() or not self.pending_queue.empty():
            try:
                pending = self.pending_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._write_pending_sample(pending)
            except Exception as exc:  # keep metadata even after a disk/codec failure
                message = f"{type(exc).__name__}: {exc}"
                self.writer_errors.append(message)
                self.recording_enabled = False
                self.accepting_samples = False
                self.stop_reason = "writer_error"
                self.writer_stop.set()
                while True:
                    try:
                        self.pending_queue.get_nowait()
                        self.pending_queue.task_done()
                    except queue.Empty:
                        break
            finally:
                self.pending_queue.task_done()

    def _write_pending_sample(self, pending: PendingSample) -> None:
        front_path = self._save_front_image(pending.timestamp_ns, pending.front_msg)
        scan_path = None
        if self.params["save_scan_npz"] and pending.scan_msg is not None:
            scan_path = self._save_scan(pending.timestamp_ns, pending.scan_msg)
        if self.imu_writer is not None and pending.imu_msg is not None:
            self.imu_writer.writerow(self._imu_row(pending.timestamp_ns, pending.imu_msg))
        if self.odom_writer is not None and pending.odom_msg is not None:
            self.odom_writer.writerow(self._odom_row(pending.timestamp_ns, pending.odom_msg))
        self.csv_writer.writerow(
            {
                "timestamp_ns": pending.timestamp_ns,
                "image_timestamp_ns": pending.timestamp_ns,
                "scan_timestamp_ns": pending.scan_timestamp_ns or "",
                "scan_time_offset_ms": (
                    "" if pending.scan_time_offset_ms is None else pending.scan_time_offset_ms
                ),
                "front_image_path": relative_to_session(self.session_dir, front_path),
                "scan_npz_path": relative_to_session(self.session_dir, scan_path),
                "motor_angle": pending.motor_angle,
                "motor_speed": pending.motor_speed,
                "mission_label": pending.mission_label,
                "dataset_profile": self.params["dataset_profile"],
                "session_id": self.session_id,
            }
        )
        self.sample_count += 1
        if self.sample_count % max(1, self.params["flush_every_n_samples"]) == 0:
            self.csv_handle.flush()
            self._flush_debug_handles()

    def _label_allowed(self, label: str) -> bool:
        if self.params["exclude_bad_data"] and label == "bad_data":
            return False
        if self.params["exclude_idle"] and label == "idle":
            return False
        if self.allowed_labels and label not in self.allowed_labels:
            return False
        return True

    def _speed_allowed(self, speed: float) -> bool:
        threshold = self.params["min_abs_speed_to_save"]
        maximum = self.params["max_abs_speed_to_save"]
        if maximum > 0.0 and abs(speed) > maximum:
            return False
        if self.params["exclude_zero_speed"] and abs(speed) <= max(threshold, 1e-6):
            return False
        if abs(speed) >= threshold:
            return True
        return self.params["save_when_stopped"]

    def _save_front_image(self, stamp_ns: int, front_msg: Image) -> Optional[Path]:
        if self.params["save_front_image"]:
            return self._save_image("front", stamp_ns, front_msg)
        return None

    def _save_image(self, camera_name: str, stamp_ns: int, msg: Image) -> Path:
        if self.bridge is None or cv2 is None:
            raise RuntimeError("cv_bridge and OpenCV are required to save images")
        suffix = self.params["image_format"]
        out_path = self.session_dir / "images" / camera_name / f"{stamp_ns}.{suffix}"
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if suffix == "jpg":
            options = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.params["jpeg_quality"])]
        else:
            options = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        ok = cv2.imwrite(str(out_path), image, options)
        if not ok:
            raise RuntimeError(f"failed to write image: {out_path}")
        self.image_count += 1
        return out_path

    def _save_nearest_scan(self, stamp_ns: int, tolerance_ns: int) -> Optional[Path]:
        if not self.params["save_scan_npz"]:
            return None
        item = self.scan_buffer.nearest(stamp_ns, tolerance_ns)
        if item is None:
            return None
        return self._save_scan(stamp_ns, item.msg)

    def _save_scan(self, stamp_ns: int, msg: LaserScan) -> Optional[Path]:
        if np is None:
            self._warn_once("numpy is unavailable; scan npz was not saved")
            return None
        out_path = self.session_dir / "scan" / f"{stamp_ns}.npz"
        np.savez_compressed(
            out_path,
            ranges=np.asarray(msg.ranges, dtype=np.float32),
            intensities=np.asarray(msg.intensities, dtype=np.float32),
            angle_min=np.float32(msg.angle_min),
            angle_max=np.float32(msg.angle_max),
            angle_increment=np.float32(msg.angle_increment),
            time_increment=np.float32(msg.time_increment),
            scan_time=np.float32(msg.scan_time),
            range_min=np.float32(msg.range_min),
            range_max=np.float32(msg.range_max),
            frame_id=np.asarray(getattr(msg.header, "frame_id", ""), dtype=str),
        )
        self.scan_count += 1
        return out_path

    def _write_optional_debug_sample(self, stamp_ns: int, tolerance_ns: int) -> None:
        if self.imu_writer is not None:
            item = self.imu_buffer.nearest(stamp_ns, tolerance_ns)
            if item is not None:
                self.imu_writer.writerow(self._imu_row(stamp_ns, item.msg))
        if self.odom_writer is not None:
            item = self.odom_buffer.nearest(stamp_ns, tolerance_ns)
            if item is not None:
                self.odom_writer.writerow(self._odom_row(stamp_ns, item.msg))

    def _imu_row(self, stamp_ns: int, msg: Imu) -> Dict[str, Any]:
        return {
            "sample_timestamp_ns": stamp_ns,
            "imu_timestamp_ns": stamp_to_ns(msg),
            "frame_id": getattr(msg.header, "frame_id", ""),
            "orientation_x": msg.orientation.x,
            "orientation_y": msg.orientation.y,
            "orientation_z": msg.orientation.z,
            "orientation_w": msg.orientation.w,
            "angular_velocity_x": msg.angular_velocity.x,
            "angular_velocity_y": msg.angular_velocity.y,
            "angular_velocity_z": msg.angular_velocity.z,
            "linear_acceleration_x": msg.linear_acceleration.x,
            "linear_acceleration_y": msg.linear_acceleration.y,
            "linear_acceleration_z": msg.linear_acceleration.z,
        }

    def _odom_row(self, stamp_ns: int, msg: Odometry) -> Dict[str, Any]:
        pose = msg.pose.pose
        twist = msg.twist.twist
        return {
            "sample_timestamp_ns": stamp_ns,
            "odom_timestamp_ns": stamp_to_ns(msg),
            "frame_id": getattr(msg.header, "frame_id", ""),
            "child_frame_id": getattr(msg, "child_frame_id", ""),
            "position_x": pose.position.x,
            "position_y": pose.position.y,
            "position_z": pose.position.z,
            "orientation_x": pose.orientation.x,
            "orientation_y": pose.orientation.y,
            "orientation_z": pose.orientation.z,
            "orientation_w": pose.orientation.w,
            "linear_x": twist.linear.x,
            "linear_y": twist.linear.y,
            "linear_z": twist.linear.z,
            "angular_x": twist.angular.x,
            "angular_y": twist.angular.y,
            "angular_z": twist.angular.z,
        }

    def _within_rate_limit(self, stamp_ns: int) -> bool:
        return rate_limit_allows(
            stamp_ns,
            self.last_rate_accepted_ns,
            self.params["max_save_rate_hz"],
        )

    def _sample_limit_reached(self) -> bool:
        limit = self.params["max_samples"]
        return limit > 0 and self.enqueued_sample_count >= limit

    def _request_sample_limit_stop(self) -> None:
        if self.stop_reason == "max_samples":
            return
        self.recording_enabled = False
        self.accepting_samples = False
        self.stop_reason = "max_samples"
        self.auto_exit_requested = self.params["exit_on_limit_reached"]
        self.writer_stop.set()
        with self.preroll_lock:
            self.discarded_uncommitted_preroll += self.bad_data_preroll.discard_all()
        self.get_logger().info(
            "sample limit reached: %d; draining writer queue"
            % self.params["max_samples"]
        )

    def _auto_exit_tick(self) -> None:
        if not self.auto_exit_requested or self.auto_exit_triggered:
            return
        if self.writer_thread.is_alive() or not self.pending_queue.empty():
            return
        self.auto_exit_triggered = True
        self.get_logger().info(
            "sample limit saved; shutting down recorder cleanly"
        )
        threading.Thread(
            target=self._shutdown_ros_context,
            name="il-recorder-shutdown",
            daemon=True,
        ).start()

    @staticmethod
    def _shutdown_ros_context() -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def _within_duration_limit(self) -> bool:
        limit = self.params["max_session_duration_sec"]
        if limit <= 0.0:
            return True
        return (datetime.now() - self.started_at).total_seconds() <= limit

    def _disk_space_available(self) -> bool:
        try:
            available = self.disk_guard.available(self.session_dir)
            self.last_free_disk_gb = self.disk_guard.last_free_gb
        except OSError as exc:
            self._warn_once(f"disk free-space check failed: {exc}")
            return True
        return available

    def _debug_tick(self) -> None:
        now_ns = ros_time_to_ns(self.get_clock().now())
        missing = []
        if len(self.front_buffer) == 0:
            missing.append("front_image")
        if len(self.motor_buffer) == 0:
            missing.append("motor")
        if not self.label_latch.ever_received:
            missing.append("mission_label")
        if self.params["save_scan_npz"] and len(self.scan_buffer) == 0:
            missing.append("scan")
        if missing:
            self._warn_once(f"waiting for: {', '.join(missing)}")

        motor_publishers = self.get_publishers_info_by_topic(self.params["motor_topic"])
        if len(motor_publishers) > 1:
            self._warn_once(
                f"{self.params['motor_topic']} has {len(motor_publishers)} publishers"
            )
        self.get_logger().info(
            "samples=%d images=%d scans=%d queue=%d dropped_queue=%d "
            "skipped(motor=%d,scan=%d,label=%d,speed=%d,blank=%d,rate=%d) "
            "buffers(front=%d,motor=%d,scan=%d) now_ns=%d"
            % (
                self.sample_count,
                self.image_count,
                self.scan_count,
                self.pending_queue.qsize(),
                self.dropped_queue_full,
                self.skipped_missing_motor,
                self.skipped_missing_scan,
                self.skipped_label_filter,
                self.skipped_speed_filter,
                self.skipped_blank_canonical,
                self.skipped_rate_limit,
                len(self.front_buffer),
                len(self.motor_buffer),
                len(self.scan_buffer),
                now_ns,
            )
        )
        self._write_metadata(final=False)

    def _warn_once(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            self.get_logger().warn(message)

    def _write_metadata(self, final: bool) -> None:
        data = {
            "package": "il_data_tools",
            "node": "il_common_recorder",
            "session_id": self.session_id,
            "dataset_profile": self.params["dataset_profile"],
            "session_dir": str(self.session_dir),
            "started_at": self.started_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "finished": bool(final),
            "allowed_labels": self.allowed_labels,
            "parameters": self.params,
            "run_manifest": self.run_manifest,
            "resolved_motor_msg_type": self.resolved_motor_msg_type_name,
            "sample_count": self.sample_count,
            "enqueued_sample_count": self.enqueued_sample_count,
            "image_count": self.image_count,
            "scan_count": self.scan_count,
            "skipped_missing_motor": self.skipped_missing_motor,
            "skipped_missing_scan": self.skipped_missing_scan,
            "skipped_unsynced_scan": self.skipped_unsynced_scan,
            "skipped_label_filter": self.skipped_label_filter,
            "skipped_speed_filter": self.skipped_speed_filter,
            "skipped_blank_canonical": self.skipped_blank_canonical,
            "skipped_rate_limit": self.skipped_rate_limit,
            "skipped_disk_limit": self.skipped_disk_limit,
            "queue_size": self.pending_queue.qsize(),
            "dropped_queue_full": self.dropped_queue_full,
            "dropped_sync_queue_full": self.dropped_sync_queue_full,
            "bad_data_preroll_buffer_size": len(self.bad_data_preroll),
            "discarded_bad_data_preroll": self.discarded_bad_data_preroll,
            "discarded_uncommitted_preroll": self.discarded_uncommitted_preroll,
            "writer_errors": list(self.writer_errors),
            "writer_shutdown_wait_sec": self.writer_shutdown_wait_sec,
            "stop_reason": self.stop_reason,
            "last_free_disk_gb": self.last_free_disk_gb,
            "label_ever_received": self.label_latch.ever_received,
            "final_active_label": self.label_latch.active,
            "last_label_timestamp_ns": self.label_latch.last_timestamp_ns,
            "label_change_count": self.label_latch.change_count,
            "warnings": self.warnings,
            "safety": {
                "publishes_xycar_motor": False,
                "note": "This recorder only subscribes to /xycar_motor.",
            },
        }
        atomic_write_json(self.session_dir / "metadata.json", data)

    def close(self) -> None:
        if self._closed:
            return
        self.recording_enabled = False
        self.accepting_samples = False
        with self.preroll_lock:
            self.discarded_uncommitted_preroll += self.bad_data_preroll.discard_all()
        self.writer_stop.set()
        warning_after = self.params["writer_shutdown_timeout_sec"]
        self.writer_shutdown_wait_sec = wait_for_thread_shutdown(
            self.writer_thread,
            warning_after_sec=warning_after,
            on_warning=lambda elapsed: self._warn_once(
                "writer shutdown is taking longer than %.1fs; waiting for a safe drain "
                "before closing CSV files" % elapsed
            ),
        )
        # At this point the writer is guaranteed not to touch these handles again.
        self.csv_handle.flush()
        self.csv_handle.close()
        self._flush_debug_handles()
        for handle in [self.imu_handle, self.odom_handle]:
            if handle is not None:
                handle.close()
        self._closed = True
        self._write_metadata(final=True)

    def _flush_debug_handles(self) -> None:
        for handle in [self.imu_handle, self.odom_handle]:
            if handle is not None:
                handle.flush()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ILCommonRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
