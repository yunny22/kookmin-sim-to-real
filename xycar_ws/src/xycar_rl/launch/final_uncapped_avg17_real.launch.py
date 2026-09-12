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
            "final_rule_td3_bc_uncapped_avg17_20260723",
            "camera_speed_td3_bc_best.pth",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("drive_enabled", default_value="false"),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument("policy_cpu_threads", default_value="4"),
            DeclareLaunchArgument(
                "checkpoint_path",
                default_value=checkpoint,
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [package_share, "launch", "real_shadow.launch.py"]
                    )
                ),
                launch_arguments={
                    "policy_kind": "camera_speed_td3_bc",
                    "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                    "drive_enabled": LaunchConfiguration("drive_enabled"),
                    "device": LaunchConfiguration("device"),
                    "policy_cpu_threads": LaunchConfiguration(
                        "policy_cpu_threads"
                    ),
                    "image_topic": "/perception/canonical_road_image",
                    "min_speed_command": "4.0",
                    "max_speed_command": "24.0",
                    "deployment_speed_cap": "0.0",
                    "lidar_safety_enabled": "false",
                    "adaptive_steering_enabled": "false",
                    "steering_temporal_alpha": "1.0",
                    "speed_temporal_alpha": "1.0",
                    # Canonical input is already limited to about 7 Hz. A
                    # second 7 Hz gate drops jittered frames and produced only
                    # 4.47 Hz on the 2026-07-24 real-car run.
                    "max_inference_rate_hz": "0.0",
                    "max_image_age_sec": "0.30",
                    "max_temporal_frame_gap_sec": "0.25",
                    "preview_steering_enabled": "false",
                }.items(),
            ),
        ]
    )
