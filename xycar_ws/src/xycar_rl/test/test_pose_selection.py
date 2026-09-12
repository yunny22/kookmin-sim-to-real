from types import SimpleNamespace
import unittest

from xycar_rl.gazebo_env import select_model_transform


def transform(child_frame_id="", frame_id=""):
    return SimpleNamespace(
        child_frame_id=child_frame_id,
        header=SimpleNamespace(frame_id=frame_id),
        transform=SimpleNamespace(),
    )


class PoseSelectionTest(unittest.TestCase):
    def test_named_vehicle_frame_wins_over_other_entities(self):
        scenery = transform(child_frame_id="room/table")
        vehicle = transform(child_frame_id="xycar_ackermann/chassis")
        self.assertIs(
            select_model_transform([scenery, vehicle], "xycar_ackermann"),
            vehicle,
        )

    def test_unlabeled_pose_vector_uses_bridge_first_transform(self):
        first = transform()
        second = transform()
        self.assertIs(
            select_model_transform([first, second], "xycar_ackermann"),
            first,
        )

    def test_named_message_without_vehicle_does_not_guess(self):
        self.assertIsNone(
            select_model_transform(
                [transform(child_frame_id="room/table")],
                "xycar_ackermann",
            )
        )


if __name__ == "__main__":
    unittest.main()
