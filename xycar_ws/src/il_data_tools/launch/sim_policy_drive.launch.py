from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    world = LaunchConfiguration("world")
    model_path = LaunchConfiguration("model_path")
    drive_enabled = LaunchConfiguration("drive_enabled")
    speed_command = LaunchConfiguration("speed_command")
    device = LaunchConfiguration("device")
    enable_rviz = LaunchConfiguration("enable_rviz")
    image_topic = LaunchConfiguration("image_topic")

    bridge_launch = PathJoinSubstitution(
        [FindPackageShare("xycar_gazebo_bridge"), "launch", "xycar_gazebo_rviz.launch.py"]
    )
    inference_launch = PathJoinSubstitution(
        [FindPackageShare("il_data_tools"), "launch", "policy_inference.launch.py"]
    )
    gazebo = ExecuteProcess(
        cmd=["gz", "sim", "-r", world],
        name="kookmin_gazebo",
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "project_root",
                default_value=PathJoinSubstitution(
                    [EnvironmentVariable("HOME"), "xycar_kookmin_gazebo_track"]
                ),
                description="Repository root containing the world and trained model.",
            ),
            DeclareLaunchArgument(
                "world",
                default_value=PathJoinSubstitution(
                    [project_root, "worlds", "kookmin_xycar_track_final.sdf"]
                ),
                description="Gazebo world used for closed-loop policy testing.",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("il_data_tools"),
                        "models",
                        "drive_canonical_policy_scripted.pt",
                    ]
                ),
                description="TorchScript camera and LiDAR steering policy.",
            ),
            DeclareLaunchArgument(
                "drive_enabled",
                default_value="true",
                description="Publish policy commands to the simulated vehicle.",
            ),
            DeclareLaunchArgument(
                "speed_command",
                default_value="4.0",
                description="Straight speed command, matching the rule-based simulation.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cuda",
                description="Torch inference device.",
            ),
            DeclareLaunchArgument(
                "enable_rviz",
                default_value="true",
                description="Show camera, LiDAR, and perception outputs in RViz.",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/perception/canonical_road_image",
                description="Raw or canonical image topic expected by the model.",
            ),
            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH",
                value=[
                    project_root,
                    ":",
                    EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
                ],
            ),
            gazebo,
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(bridge_launch),
                launch_arguments={"enable_rviz": enable_rviz}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(inference_launch),
                launch_arguments={
                    "model_path": model_path,
                    "drive_enabled": drive_enabled,
                    "speed_command": speed_command,
                    "device": device,
                    "image_topic": image_topic,
                }.items(),
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=gazebo,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(
                                reason="Gazebo exited; stopping policy test stack"
                            )
                        )
                    ],
                )
            ),
        ]
    )
