from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    output_root = LaunchConfiguration("output_root")
    session_name = LaunchConfiguration("session_name")
    default_mission_label = LaunchConfiguration("default_mission_label")
    max_save_rate_hz = LaunchConfiguration("max_save_rate_hz")
    sync_tolerance_sec = LaunchConfiguration("sync_tolerance_sec")
    max_samples = LaunchConfiguration("max_samples")
    camera_front_topic = LaunchConfiguration("camera_front_topic")

    drive_launch = PathJoinSubstitution(
        [FindPackageShare("il_data_tools"), "launch", "record_drive_dataset.launch.py"]
    )
    default_output_root = PathJoinSubstitution(
        [
            EnvironmentVariable("HOME"),
            "xycar_kookmin_gazebo_track",
            "datasets",
            "il",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "output_root",
                default_value=default_output_root,
                description="Raw simulation dataset root.",
            ),
            DeclareLaunchArgument(
                "session_name",
                default_value="sim_drive",
                description="Auto-incremented session base name.",
            ),
            DeclareLaunchArgument(
                "default_mission_label",
                default_value="general_drive",
                description="Drive label stored when no mission label is published.",
            ),
            DeclareLaunchArgument(
                "max_save_rate_hz",
                default_value="10.0",
                description="Maximum synchronized samples saved per second.",
            ),
            DeclareLaunchArgument(
                "sync_tolerance_sec",
                default_value="0.05",
                description="Maximum image, scan, and motor timestamp difference.",
            ),
            DeclareLaunchArgument(
                "max_samples",
                default_value="50000",
                description="Stop and close the recorder after this many samples.",
            ),
            DeclareLaunchArgument(
                "camera_front_topic",
                default_value="/image_raw",
                description="Image representation stored as the training input.",
            ),
            GroupAction(
                [
                    # Headerless motor commands must use the same clock as Gazebo sensors.
                    SetParameter(name="use_sim_time", value=True),
                    SetParameter(name="max_samples", value=max_samples),
                    SetParameter(name="exit_on_limit_reached", value=True),
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(drive_launch),
                        launch_arguments={
                            "output_root": output_root,
                            "session_name": session_name,
                            "session_auto_increment": "true",
                            "camera_front_topic": camera_front_topic,
                            "scan_topic": "/scan",
                            "motor_topic": "/xycar_motor",
                            "motor_msg_type": "float32_multi_array",
                            "default_mission_label": default_mission_label,
                            "save_scan_npz": "true",
                            "require_scan": "true",
                            "sync_tolerance_sec": sync_tolerance_sec,
                            "sync_wait_sec": "0.10",
                            "max_save_rate_hz": max_save_rate_hz,
                            "min_free_disk_gb": "10.0",
                            "enable_recording_on_start": "true",
                        }.items(),
                    ),
                ]
            ),
        ]
    )
