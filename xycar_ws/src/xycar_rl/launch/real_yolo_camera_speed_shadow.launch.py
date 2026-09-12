from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    perception_launch = PathJoinSubstitution(
        [
            FindPackageShare("xycar_perception"),
            "launch",
            "real_yolo_canonical_asus.launch.py",
        ]
    )
    policy_launch = PathJoinSubstitution(
        [FindPackageShare("xycar_rl"), "launch", "real_shadow.launch.py"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "checkpoint_path",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("xycar_rl"),
                        "models",
                        "high_speed_td3_bc_dual_dagger_v6_20260717",
                        "camera_speed_td3_bc_epoch_051.pth",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "image_topic", default_value="/wide_camera/rect/image_raw"
            ),
            DeclareLaunchArgument("use_compressed_image", default_value="false"),
            DeclareLaunchArgument("enable_rectify", default_value="false"),
            DeclareLaunchArgument("yolo_confidence", default_value="0.25"),
            DeclareLaunchArgument("yolo_image_size", default_value="512"),
            DeclareLaunchArgument("yolo_cpu_threads", default_value="4"),
            DeclareLaunchArgument("lane_rate_hz", default_value="15.0"),
            DeclareLaunchArgument("drive_enabled", default_value="false"),
            DeclareLaunchArgument("deployment_speed_cap", default_value="0.0"),
            DeclareLaunchArgument("steering_output_sign", default_value="1.0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(perception_launch),
                launch_arguments={
                    "image_topic": LaunchConfiguration("image_topic"),
                    "use_compressed_image": LaunchConfiguration(
                        "use_compressed_image"
                    ),
                    "enable_rectify": LaunchConfiguration("enable_rectify"),
                    "yolo_confidence": LaunchConfiguration("yolo_confidence"),
                    "yolo_image_size": LaunchConfiguration("yolo_image_size"),
                    "yolo_cpu_threads": LaunchConfiguration("yolo_cpu_threads"),
                    "publish_rate_limit_hz": LaunchConfiguration("lane_rate_hz"),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(policy_launch),
                launch_arguments={
                    "policy_kind": "camera_speed_td3_bc",
                    "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                    "image_topic": "/perception/canonical_road_image",
                    "drive_enabled": LaunchConfiguration("drive_enabled"),
                    "deployment_speed_cap": LaunchConfiguration(
                        "deployment_speed_cap"
                    ),
                    "lidar_safety_enabled": "false",
                    "steering_output_sign": LaunchConfiguration(
                        "steering_output_sign"
                    ),
                    "device": "cpu",
                }.items(),
            ),
        ]
    )
