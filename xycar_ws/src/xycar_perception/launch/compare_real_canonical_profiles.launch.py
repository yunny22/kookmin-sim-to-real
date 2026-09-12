from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def real_perception_parameters() -> tuple[dict, str]:
    package_share = Path(get_package_share_directory("xycar_perception"))
    config_path = package_share / "config" / "camera_perception_real.yaml"
    calibration_path = (
        package_share / "config" / "wide_camera_fisheye_1280x1024.yaml"
    )
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    parameters = config["xycar_camera_perception"]["ros__parameters"]
    return parameters, str(calibration_path)


def perception_node(name: str, prefix: str, temporary: bool) -> Node:
    image_topic = LaunchConfiguration("image_topic")
    use_sim_time = LaunchConfiguration("use_sim_time")
    parameters, calibration = real_perception_parameters()
    overrides = {
        "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
        "image_topic": image_topic,
        "use_compressed_image": True,
        "enable_rectify": True,
        "calib_yaml": calibration,
        "road_segments_topic": f"/{prefix}/road_segments",
        "centerline_topic": f"/{prefix}/centerline",
        "objects_topic": f"/{prefix}/objects",
        "traffic_lights_topic": f"/{prefix}/traffic_lights",
        "rectified_image_topic": f"/{prefix}/rectified_camera_image",
        "debug_image_topic": f"/{prefix}/perception_debug",
        "debug_markers_topic": f"/{prefix}/debug_markers",
        "canonical_road_topic": f"/{prefix}/canonical_road_image",
        "canonical_white_mask_topic": f"/{prefix}/canonical_white_mask",
        "canonical_yellow_mask_topic": f"/{prefix}/canonical_yellow_mask",
        "canonical_tracking_debug_topic": (
            f"/{prefix}/canonical_tracking_debug"
        ),
        "canonical_pregeometry_white_mask_topic": (
            f"/{prefix}/canonical_pregeometry_white_mask"
        ),
        "canonical_pregeometry_yellow_mask_topic": (
            f"/{prefix}/canonical_pregeometry_yellow_mask"
        ),
        "canonical_pretrack_white_mask_topic": (
            f"/{prefix}/canonical_pretrack_white_mask"
        ),
        "canonical_pretrack_yellow_mask_topic": (
            f"/{prefix}/canonical_pretrack_yellow_mask"
        ),
        "canonical_valid_mask_topic": f"/{prefix}/canonical_valid_mask",
        "canonical_metric_debug_topic": f"/{prefix}/canonical_metric_debug",
    }
    if temporary:
        overrides.update(
            {
                "canonical_expected_half_lane_width_m": 0.49,
                "canonical_lane_width_tolerance_m": 0.14,
                "canonical_white_min_component_median_v": 140.0,
                "canonical_yellow_fit_gate_m": 0.08,
            }
        )

    return Node(
        package="xycar_perception",
        executable="camera_perception_node",
        name=name,
        parameters=[parameters, overrides],
        output="screen",
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "image_topic",
                default_value="/wide_camera_mjpeg/image_raw/compressed",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            perception_node(
                "xycar_camera_perception_competition",
                "profile_compare/competition",
                temporary=False,
            ),
            perception_node(
                "xycar_camera_perception_temporary",
                "profile_compare/temporary",
                temporary=True,
            ),
        ]
    )
