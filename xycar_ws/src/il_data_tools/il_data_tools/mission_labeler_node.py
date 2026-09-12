#!/usr/bin/env python3

import select
import sys
import termios
import tty
from typing import Dict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


LABELS: Dict[str, str] = {
    "0": "idle",
    "1": "general_drive",
    "2": "lane_drive",
    "3": "hill_drive",
    "4": "shortcut",
    "5": "recovery",
    "6": "cone_drive",
    "7": "vehicle_overtake",
    "8": "overtake_start",
    "9": "overtake_end",
    "p": "parking",
    "b": "bad_data",
    "r": "red_light_wait",
    "h": "pedestrian_wait",
}


HELP_TEXT = """
Mission label keys:
  0 idle | 1 general_drive | 2 lane_drive | 3 hill_drive
  4 shortcut | 5 recovery | 6 cone_drive | 7 vehicle_overtake
  8 overtake_start | 9 overtake_end | p parking
  b bad_data | r red_light_wait | h pedestrian_wait | q quit
"""


class ILMissionLabeler(Node):
    def __init__(self) -> None:
        super().__init__("il_mission_labeler")
        self.declare_parameter("mission_label_topic", "/il/mission_label")
        self.declare_parameter("initial_label", "idle")
        self.current_label = str(self.get_parameter("initial_label").value)
        topic = str(self.get_parameter("mission_label_topic").value)
        self.publisher = self.create_publisher(String, topic, 10)
        self.old_settings = None
        self.terminal_ready = False
        self._configure_terminal()
        self.create_timer(0.2, self._tick)
        self.get_logger().info(HELP_TEXT)
        self.get_logger().info(f"publishing label={self.current_label} topic={topic}")

    def _configure_terminal(self) -> None:
        if not sys.stdin.isatty():
            self.get_logger().warn("stdin is not a TTY; publishing initial label only")
            return
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.terminal_ready = True

    def _tick(self) -> None:
        self._read_key_once()
        msg = String()
        msg.data = self.current_label
        self.publisher.publish(msg)

    def _read_key_once(self) -> None:
        if not self.terminal_ready:
            return
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return
        key = sys.stdin.read(1)
        if key == "q":
            self.get_logger().info("quit requested")
            rclpy.shutdown()
            return
        if key in LABELS:
            self.current_label = LABELS[key]
            self.get_logger().info(f"mission_label={self.current_label}")

    def restore_terminal(self) -> None:
        if self.terminal_ready and self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ILMissionLabeler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.restore_terminal()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
