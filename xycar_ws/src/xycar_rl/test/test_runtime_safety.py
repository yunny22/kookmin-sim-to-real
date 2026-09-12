import math
import unittest

from sensor_msgs.msg import LaserScan

from xycar_rl.policy_runtime_node import front_obstacle_distance


class RuntimeSafetyTest(unittest.TestCase):
    def test_front_distance_ignores_rear_obstacle(self):
        scan = LaserScan()
        scan.angle_min = -math.pi
        scan.angle_increment = math.pi / 2.0
        scan.range_min = 0.1
        scan.range_max = 12.0
        scan.ranges = [0.2, 3.0, 1.2, 4.0, 0.3]
        distance = front_obstacle_distance(scan, math.radians(25.0))
        self.assertAlmostEqual(distance, 1.2)


if __name__ == "__main__":
    unittest.main()
