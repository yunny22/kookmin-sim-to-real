from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("xycar_rl")
    checkpoint = PathJoinSubstitution(
        [
            package_share,
            "models",
            "lap_time_speed_only_round02_20260723",
            "camera_speed_lap_time_speed_only_actor.pth",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("drive_enabled", default_value="false"),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument(
                "deployment_speed_cap",
                default_value="4.0",
                description="Raise only after the previous real-car gate passes.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [package_share, "launch", "real_shadow.launch.py"]
                    )
                ),
                launch_arguments={
                    "policy_kind": "camera_speed_td3_bc",
                    "checkpoint_path": checkpoint,
                    "drive_enabled": LaunchConfiguration("drive_enabled"),
                    "device": LaunchConfiguration("device"),
                    "image_topic": "/perception/canonical_road_image",
                    "deployment_speed_cap": LaunchConfiguration(
                        "deployment_speed_cap"
                    ),
                    "lidar_safety_enabled": "false",
                    "adaptive_steering_enabled": "false",
                    "steering_temporal_alpha": "1.0",
                    "speed_temporal_alpha": "1.0",
                    "max_inference_rate_hz": "0.0",
                    "max_image_age_sec": "0.30",
                    "max_temporal_frame_gap_sec": "0.25",
                    "preview_steering_enabled": "false",
                }.items(),
            ),
        ]
    )
