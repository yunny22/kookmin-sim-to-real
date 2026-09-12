from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model_path = LaunchConfiguration("model_path")
    drive_enabled = LaunchConfiguration("drive_enabled")
    speed_command = LaunchConfiguration("speed_command")
    sensor_timeout_sec = LaunchConfiguration("sensor_timeout_sec")
    device = LaunchConfiguration("device")
    image_topic = LaunchConfiguration("image_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_path",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("il_data_tools"),
                        "models",
                        "drive_policy_scripted.pt",
                    ]
                ),
                description="TorchScript drive policy.",
            ),
            DeclareLaunchArgument(
                "drive_enabled",
                default_value="false",
                description="Publish to /xycar_motor only when true.",
            ),
            DeclareLaunchArgument(
                "speed_command",
                default_value="4.0",
                description="Straight speed command, matching the rule-based driver.",
            ),
            DeclareLaunchArgument(
                "sensor_timeout_sec",
                default_value="1.5",
                description="Simulation camera gap allowed before publishing stop.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cuda",
                description="Torch inference device.",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/image_raw",
                description="Raw or canonical image topic used by the trained model.",
            ),
            Node(
                package="il_data_tools",
                executable="il_policy_inference",
                name="il_policy_inference",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "model_path": model_path,
                        "device": device,
                        "image_topic": image_topic,
                        "drive_enabled": ParameterValue(drive_enabled, value_type=bool),
                        "speed_command": ParameterValue(speed_command, value_type=float),
                        "sensor_timeout_sec": ParameterValue(
                            sensor_timeout_sec, value_type=float
                        ),
                    }
                ],
            ),
        ]
    )
