from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")
    use_compressed_image = LaunchConfiguration("use_compressed_image")
    enable_rectify = LaunchConfiguration("enable_rectify")
    use_sim_time = LaunchConfiguration("use_sim_time")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    yolo_device = LaunchConfiguration("yolo_device")
    yolo_confidence = LaunchConfiguration("yolo_confidence")
    yolo_image_size = LaunchConfiguration("yolo_image_size")
    real_launch = PathJoinSubstitution(
        [
            FindPackageShare("xycar_perception"),
            "launch",
            "real_canonical_perception.launch.py",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "image_topic",
                default_value="/wide_camera_mjpeg/image_raw/compressed",
            ),
            DeclareLaunchArgument(
                "use_compressed_image", default_value="true"
            ),
            DeclareLaunchArgument("enable_rectify", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "yolo_model_path",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("xycar_perception"),
                        "models",
                        "kookmin_lane_yolo11n_512.pt",
                    ]
                ),
            ),
            DeclareLaunchArgument("yolo_device", default_value="0"),
            DeclareLaunchArgument("yolo_confidence", default_value="0.25"),
            DeclareLaunchArgument("yolo_image_size", default_value="512"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(real_launch),
                launch_arguments={
                    "image_topic": image_topic,
                    "use_compressed_image": use_compressed_image,
                    "enable_rectify": enable_rectify,
                    "use_sim_time": use_sim_time,
                    "lane_segmentation_backend": "yolo",
                    "yolo_model_path": yolo_model_path,
                    "yolo_device": yolo_device,
                    "yolo_confidence": yolo_confidence,
                    "yolo_image_size": yolo_image_size,
                }.items(),
            ),
        ]
    )
