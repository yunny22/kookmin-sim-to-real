from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    yolo_launch = PathJoinSubstitution(
        [
            FindPackageShare("xycar_perception"),
            "launch",
            "real_yolo_canonical_asus.launch.py",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "image_topic",
                default_value="/wide_camera_mjpeg/image_raw/compressed",
                description="Unrectified MJPEG camera topic.",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("yolo_confidence", default_value="0.25"),
            DeclareLaunchArgument("yolo_image_size", default_value="512"),
            DeclareLaunchArgument("yolo_cpu_threads", default_value="4"),
            DeclareLaunchArgument("publish_rate_limit_hz", default_value="15.0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(yolo_launch),
                launch_arguments={
                    "image_topic": LaunchConfiguration("image_topic"),
                    "use_compressed_image": "true",
                    "enable_rectify": "true",
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "yolo_confidence": LaunchConfiguration("yolo_confidence"),
                    "yolo_image_size": LaunchConfiguration("yolo_image_size"),
                    "yolo_cpu_threads": LaunchConfiguration("yolo_cpu_threads"),
                    "publish_rate_limit_hz": LaunchConfiguration(
                        "publish_rate_limit_hz"
                    ),
                }.items(),
            ),
        ]
    )
