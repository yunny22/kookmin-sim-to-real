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
    lane_segmentation_backend = LaunchConfiguration(
        "lane_segmentation_backend"
    )
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
                "lane_segmentation_backend", default_value="color"
            ),
            DeclareLaunchArgument("yolo_model_path", default_value=""),
            DeclareLaunchArgument("yolo_device", default_value="cpu"),
            DeclareLaunchArgument("yolo_confidence", default_value="0.25"),
            DeclareLaunchArgument("yolo_image_size", default_value="640"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(real_launch),
                launch_arguments={
                    "image_topic": image_topic,
                    "use_compressed_image": use_compressed_image,
                    "enable_rectify": enable_rectify,
                    "use_sim_time": use_sim_time,
                    "lane_segmentation_backend": (
                        lane_segmentation_backend
                    ),
                    "yolo_model_path": yolo_model_path,
                    "yolo_device": yolo_device,
                    "yolo_confidence": yolo_confidence,
                    "yolo_image_size": yolo_image_size,
                    # Compatibility alias for the shared real-track profile.
                    "canonical_expected_half_lane_width_m": "0.49",
                    "canonical_lane_width_tolerance_m": "0.14",
                    # Floor seams are dim (median V 98-130); physical white
                    # tape in this bag is normally above V 140.
                    "canonical_white_min_component_median_v": "140.0",
                    # Keep disconnected dashes on one fitted center curve and
                    # discard nearby yellow-tinted floor/background clutter.
                    "canonical_yellow_fit_gate_m": "0.08",
                }.items(),
            ),
        ]
    )
