from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    base_launch = PathJoinSubstitution(
        [FindPackageShare("il_data_tools"), "launch", "collect_sim_drive_dataset.launch.py"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "project_root",
                default_value=PathJoinSubstitution(
                    [EnvironmentVariable("HOME"), "xycar_kookmin_gazebo_track"]
                ),
            ),
            DeclareLaunchArgument("max_samples", default_value="50000"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={
                    "project_root": project_root,
                    "output_root": PathJoinSubstitution(
                        [project_root, "datasets", "il_canonical"]
                    ),
                    "session_name": "sim_canonical_drive",
                    "max_samples": LaunchConfiguration("max_samples"),
                    "camera_front_topic": "/perception/canonical_road_image",
                    "image_format": "png",
                }.items(),
            ),
        ]
    )
