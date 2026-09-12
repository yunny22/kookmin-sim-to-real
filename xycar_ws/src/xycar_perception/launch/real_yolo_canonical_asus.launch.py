from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    real_launch = PathJoinSubstitution(
        [
            FindPackageShare("xycar_perception"),
            "launch",
            "real_canonical_perception.launch.py",
        ]
    )

    arguments = {
        "image_topic": LaunchConfiguration("image_topic"),
        "use_compressed_image": LaunchConfiguration("use_compressed_image"),
        "enable_rectify": LaunchConfiguration("enable_rectify"),
        "use_sim_time": LaunchConfiguration("use_sim_time"),
        "lane_segmentation_backend": "yolo",
        "yolo_model_path": LaunchConfiguration("yolo_model_path"),
        "yolo_device": "cpu",
        "yolo_confidence": LaunchConfiguration("yolo_confidence"),
        "yolo_image_size": LaunchConfiguration("yolo_image_size"),
        "yolo_max_detections": LaunchConfiguration(
            "yolo_max_detections"
        ),
        "yolo_cpu_threads": LaunchConfiguration("yolo_cpu_threads"),
        "yolo_retina_masks": LaunchConfiguration("yolo_retina_masks"),
        "publish_rate_limit_hz": LaunchConfiguration(
            "publish_rate_limit_hz"
        ),
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "image_topic",
                default_value="/wide_camera_mjpeg/image_raw/compressed",
                description=(
                    "Raw MJPEG used by both live-camera and rosbag canonical "
                    "processing. Override only for an explicitly rectified input."
                ),
            ),
            DeclareLaunchArgument("use_compressed_image", default_value="true"),
            DeclareLaunchArgument("enable_rectify", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "yolo_model_path",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("xycar_perception"),
                        "models",
                        "kookmin_lane_yolo11n_256.pt",
                    ]
                ),
            ),
            DeclareLaunchArgument("yolo_confidence", default_value="0.25"),
            DeclareLaunchArgument("yolo_image_size", default_value="256"),
            DeclareLaunchArgument(
                "yolo_max_detections",
                default_value="8",
                description="Maximum lane instances kept before mask merging.",
            ),
            DeclareLaunchArgument("yolo_cpu_threads", default_value="4"),
            DeclareLaunchArgument(
                "yolo_retina_masks",
                default_value="false",
                description=(
                    "Use model-resolution masks and resize once in the lane "
                    "wrapper to reduce CPU post-processing."
                ),
            ),
            DeclareLaunchArgument(
                "publish_rate_limit_hz",
                default_value="15.0",
                description=(
                    "Lane inference rate cap. Keep obstacle YOLO at 5 Hz or less "
                    "on the Ryzen 5 vehicle PC."
                ),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(real_launch),
                launch_arguments=arguments.items(),
            ),
        ]
    )
