from __future__ import annotations

import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, String


# Clockwise yellow centerline samples recovered from the final CAD-derived map.
YELLOW_ROUTE: List[Tuple[float, float]] = [
    (-3.525, 2.400),
    (-2.928, 2.455),
    (-2.305, 2.465),
    (-1.705, 2.465),
    (-1.105, 2.465),
    (-0.505, 2.465),
    (0.105, 2.465),
    (0.705, 2.465),
    (1.305, 2.465),
    (1.900, 2.465),
    (2.395, 2.445),
    (2.965, 2.190),
    (3.195, 1.650),
    (3.155, 1.065),
    (2.810, 0.605),
    (2.365, 0.215),
    (2.240, -0.360),
    (2.335, -0.940),
    (2.715, -1.375),
    (3.140, -1.790),
    (3.265, -2.360),
    (3.115, -2.930),
    (2.755, -3.256),
    (2.315, -3.325),
    (1.900, -3.315),
    (1.305, -3.315),
    (0.705, -3.315),
    (0.105, -3.315),
    (-0.505, -3.315),
    (-1.105, -3.315),
    (-1.705, -3.315),
    (-2.305, -3.315),
    (-3.115, -3.305),
    (-3.630, -3.240),
    (-4.140, -2.945),
    (-4.465, -2.450),
    (-4.565, -1.865),
    (-4.575, -1.270),
    (-4.575, -0.670),
    (-4.575, -0.070),
    (-4.575, 0.530),
    (-4.555, 1.135),
    (-4.410, 1.710),
    (-4.035, 2.165),
]


def _segment_lengths(points: Sequence[Tuple[float, float]]) -> List[float]:
    return [
        math.hypot(
            points[(index + 1) % len(points)][0] - point[0],
            points[(index + 1) % len(points)][1] - point[1],
        )
        for index, point in enumerate(points)
    ]


def sample_recovery_pose(
    rng: random.Random,
    lateral_offsets_m: Sequence[float],
    yaw_offsets_deg: Sequence[float],
    lane_offset_from_yellow_m: float = 0.05,
) -> Dict[str, float]:
    lengths = _segment_lengths(YELLOW_ROUTE)
    segment_index = rng.choices(range(len(YELLOW_ROUTE)), weights=lengths, k=1)[0]
    p0 = YELLOW_ROUTE[segment_index]
    p1 = YELLOW_ROUTE[(segment_index + 1) % len(YELLOW_ROUTE)]
    ratio = rng.uniform(0.15, 0.85)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = max(1.0e-9, math.hypot(dx, dy))
    tx = dx / length
    ty = dy / length
    # For the clockwise route, this is the right-hand lane normal.
    nx = ty
    ny = -tx
    yellow_x = p0[0] + ratio * dx
    yellow_y = p0[1] + ratio * dy
    lateral_offset = float(rng.choice(list(lateral_offsets_m)))
    yaw_offset_deg = float(rng.choice(list(yaw_offsets_deg)))
    total_right_offset = lane_offset_from_yellow_m + lateral_offset
    nominal_yaw = math.atan2(ty, tx)
    return {
        "segment_index": float(segment_index),
        "yellow_x": yellow_x,
        "yellow_y": yellow_y,
        "x": yellow_x + nx * total_right_offset,
        "y": yellow_y + ny * total_right_offset,
        "z": 0.05,
        "nominal_yaw_rad": nominal_yaw,
        "yaw_rad": nominal_yaw + math.radians(yaw_offset_deg),
        "lateral_offset_m": lateral_offset,
        "yaw_offset_deg": yaw_offset_deg,
    }


def stopped_recovery_requires_retry(
    has_seen_motion: bool,
    phase: str,
    stopped_duration_sec: float,
    hold_sec: float,
) -> bool:
    return (
        has_seen_motion
        and phase in {"general_drive", "recovery"}
        and stopped_duration_sec >= hold_sec
    )


class RecoveryScenarioManager(Node):
    def __init__(self) -> None:
        super().__init__("il_recovery_scenario_manager")
        self._declare_parameters()
        self.world_name = str(self.get_parameter("world_name").value)
        self.model_name = str(self.get_parameter("model_name").value)
        self.seed = int(self.get_parameter("seed").value)
        self.preset = str(self.get_parameter("preset").value)
        self.warmup_sec = float(self.get_parameter("warmup_sec").value)
        self.interval_sec = float(self.get_parameter("scenario_interval_sec").value)
        self.settle_sec = float(self.get_parameter("settle_sec").value)
        self.recovery_hold_sec = float(self.get_parameter("recovery_hold_sec").value)
        self.stop_retry_hold_sec = float(
            self.get_parameter("stop_retry_hold_sec").value
        )
        self.retry_delay_sec = float(self.get_parameter("retry_delay_sec").value)
        self.lane_offset = float(
            self.get_parameter("lane_offset_from_yellow_m").value
        )
        self.lateral_offsets = [
            float(value) for value in self.get_parameter("lateral_offsets_m").value
        ]
        self.yaw_offsets = [
            float(value) for value in self.get_parameter("yaw_offsets_deg").value
        ]
        self.event_log_path = Path(
            str(self.get_parameter("event_log_path").value)
        ).expanduser()
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.label_pub = self.create_publisher(
            String,
            str(self.get_parameter("mission_label_topic").value),
            qos,
        )
        self.state_pub = self.create_publisher(
            String,
            str(self.get_parameter("scenario_state_topic").value),
            qos,
        )
        motor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("motor_topic").value),
            self._motor_cb,
            motor_qos,
        )
        self.rng = random.Random(self.seed + 100_003)
        self.phase = "general_drive"
        self.current_state: Dict[str, object] = {
            "scenario_id": 0,
            "phase": self.phase,
            "seed": self.seed,
            "preset": self.preset,
        }
        now = time.monotonic()
        self.next_intervention = now + max(1.0, self.warmup_sec)
        self.phase_deadline = 0.0
        self.scenario_id = 0
        self.has_seen_motion = False
        self.stopped_since: float | None = None
        self.create_timer(0.2, self._tick)
        self.get_logger().info(
            "recovery scenarios ready: "
            f"seed={self.seed}, interval={self.interval_sec:.1f}s, "
            f"recovery={self.recovery_hold_sec:.1f}s"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("world_name", "kookmin_xycar_track")
        self.declare_parameter("model_name", "xycar_ackermann")
        self.declare_parameter("seed", 2026)
        self.declare_parameter("preset", "mixed")
        self.declare_parameter("mission_label_topic", "/il/mission_label")
        self.declare_parameter("scenario_state_topic", "/il/scenario_state")
        self.declare_parameter("motor_topic", "/xycar_motor")
        self.declare_parameter("event_log_path", "/tmp/xycar_scenario_events.jsonl")
        self.declare_parameter("warmup_sec", 12.0)
        self.declare_parameter("scenario_interval_sec", 24.0)
        self.declare_parameter("settle_sec", 0.8)
        self.declare_parameter("recovery_hold_sec", 9.0)
        self.declare_parameter("stop_retry_hold_sec", 1.0)
        self.declare_parameter("retry_delay_sec", 1.0)
        self.declare_parameter("lane_offset_from_yellow_m", 0.05)
        self.declare_parameter(
            "lateral_offsets_m",
            [-0.22, -0.17, -0.12, -0.08, 0.08, 0.12, 0.17, 0.22],
        )
        self.declare_parameter(
            "yaw_offsets_deg",
            [-14.0, -10.0, -7.0, -4.0, 4.0, 7.0, 10.0, 14.0],
        )

    def _tick(self) -> None:
        now = time.monotonic()
        if self.phase == "failed_wait":
            if now >= self.phase_deadline:
                self._start_intervention(now)
            self._publish_state()
            return
        if self.phase == "bad_data" and now >= self.phase_deadline:
            self.phase = "recovery"
            self.phase_deadline = now + self.recovery_hold_sec
            self.current_state["phase"] = self.phase
            self._append_event("recovery_started")
        elif self.phase == "recovery" and now >= self.phase_deadline:
            self.phase = "general_drive"
            self.current_state["phase"] = self.phase
            self._append_event("recovery_finished")
        if self.phase == "general_drive" and now >= self.next_intervention:
            self._start_intervention(now)
        self._publish_state()

    def _motor_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            return
        now = time.monotonic()
        speed = float(msg.data[1])
        if abs(speed) > 1.0e-6:
            self.has_seen_motion = True
            self.stopped_since = None
            return
        if not self.has_seen_motion or self.phase not in {"general_drive", "recovery"}:
            return
        if self.stopped_since is None:
            self.stopped_since = now
            return
        stopped_duration = now - self.stopped_since
        if not stopped_recovery_requires_retry(
            self.has_seen_motion,
            self.phase,
            stopped_duration,
            self.stop_retry_hold_sec,
        ):
            return
        self.phase = "failed_wait"
        self.phase_deadline = now + self.retry_delay_sec
        self.current_state.update(
            {
                "phase": self.phase,
                "stop_duration_sec": stopped_duration,
            }
        )
        self.stopped_since = None
        self._append_event("recovery_stop_detected")
        self._publish_state()
        self.get_logger().warn(
            "vehicle remained stopped during recovery; scheduling a new pose"
        )

    def _start_intervention(self, now: float) -> None:
        self.stopped_since = None
        pose = sample_recovery_pose(
            self.rng,
            self.lateral_offsets,
            self.yaw_offsets,
            self.lane_offset,
        )
        self.scenario_id += 1
        self.phase = "bad_data"
        self.current_state = {
            "scenario_id": self.scenario_id,
            "phase": self.phase,
            "seed": self.seed,
            "preset": self.preset,
            **pose,
        }
        self._publish_state()
        if not self._set_model_pose(pose):
            self.phase = "general_drive"
            self.current_state["phase"] = "set_pose_failed"
            self.next_intervention = now + 5.0
            self._append_event("set_pose_failed")
            return
        self.phase_deadline = now + self.settle_sec
        self.next_intervention = now + max(
            self.interval_sec,
            self.settle_sec + self.recovery_hold_sec + 2.0,
        )
        self._append_event("teleported")
        self.get_logger().info(
            f"scenario={self.scenario_id} lateral={pose['lateral_offset_m']:+.2f}m "
            f"yaw={pose['yaw_offset_deg']:+.1f}deg "
            f"pose=({pose['x']:+.2f}, {pose['y']:+.2f})"
        )

    def _set_model_pose(self, pose: Dict[str, float]) -> bool:
        yaw = pose["yaw_rad"]
        request = (
            f'name: "{self.model_name}", '
            f'position: {{x: {pose["x"]:.9f}, y: {pose["y"]:.9f}, z: {pose["z"]:.9f}}}, '
            f'orientation: {{z: {math.sin(yaw / 2.0):.9f}, w: {math.cos(yaw / 2.0):.9f}}}'
        )
        command = [
            "gz",
            "service",
            "-s",
            f"/world/{self.world_name}/set_pose",
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            request,
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.get_logger().error(f"set_pose failed: {exc}")
            return False
        if result.returncode != 0 or "true" not in result.stdout.lower():
            detail = (result.stderr or result.stdout).strip()
            self.get_logger().error(f"set_pose rejected: {detail}")
            return False
        return True

    def _publish_state(self) -> None:
        label = String()
        if self.phase in {"bad_data", "failed_wait"}:
            label.data = "bad_data"
        elif self.phase == "recovery":
            label.data = "recovery"
        else:
            label.data = "general_drive"
        self.label_pub.publish(label)
        state = String()
        state.data = json.dumps(self.current_state, sort_keys=True)
        self.state_pub.publish(state)

    def _append_event(self, event: str) -> None:
        row = {
            "event": event,
            "wall_time": time.time(),
            **self.current_state,
        }
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RecoveryScenarioManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
