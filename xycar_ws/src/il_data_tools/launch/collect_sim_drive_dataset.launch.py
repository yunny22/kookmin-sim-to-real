from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    world = LaunchConfiguration("world")
    output_root = LaunchConfiguration("output_root")
    session_name = LaunchConfiguration("session_name")
    max_samples = LaunchConfiguration("max_samples")
    camera_front_topic = LaunchConfiguration("camera_front_topic")
    image_format = LaunchConfiguration("image_format")

    bridge_launch = PathJoinSubstitution(
        [FindPackageShare("xycar_gazebo_bridge"), "launch", "xycar_gazebo_rviz.launch.py"]
    )
    rule_launch = PathJoinSubstitution(
        [FindPackageShare("xycar_rule_drive"), "launch", "lane_rule_driver.launch.py"]
    )

    gazebo = ExecuteProcess(
        cmd=["gz", "sim", "-r", world],
        name="kookmin_gazebo",
        output="screen",
    )
    recorder = Node(
        package="il_data_tools",
        executable="il_common_recorder",
        name="il_common_recorder_drive",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "output_root": output_root,
                "session_name": session_name,
                "session_auto_increment": True,
                "dataset_profile": "drive",
                "allowed_labels": "general_drive,lane_drive,hill_drive,shortcut,recovery",
                "camera_front_topic": camera_front_topic,
                "scan_topic": "/scan",
                "imu_topic": "/imu",
                "odom_topic": "/odom",
                "motor_topic": "/xycar_motor",
                "motor_msg_type": "float32_multi_array",
                "mission_label_topic": "/il/mission_label",
                "default_mission_label": "general_drive",
                "save_front_image": True,
                "save_scan_npz": True,
                "require_scan": True,
                "approximate_sync_tolerance_sec": 0.05,
                "sync_wait_sec": 0.10,
                "writer_queue_size": 128,
                "min_free_disk_gb": 10.0,
                "disk_check_period_sec": 5.0,
                "stop_on_low_disk": True,
                "image_format": image_format,
                "jpeg_quality": 90,
                "max_save_rate_hz": 10.0,
                "enable_recording_on_start": True,
                "exclude_bad_data": True,
                "exclude_idle": True,
                "max_samples": ParameterValue(max_samples, value_type=int),
                "exit_on_limit_reached": True,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "project_root",
                default_value=PathJoinSubstitution(
                    [EnvironmentVariable("HOME"), "xycar_kookmin_gazebo_track"]
                ),
                description="Repository root containing the final world.",
            ),
            DeclareLaunchArgument(
                "world",
                default_value=PathJoinSubstitution(
                    [project_root, "worlds", "kookmin_xycar_track_final.sdf"]
                ),
                description="Gazebo world to collect in.",
            ),
            DeclareLaunchArgument(
                "output_root",
                default_value=PathJoinSubstitution(
                    [project_root, "datasets", "il"]
                ),
                description="Raw imitation-learning dataset root.",
            ),
            DeclareLaunchArgument(
                "session_name",
                default_value="sim_drive",
                description="Auto-incremented session base name.",
            ),
            DeclareLaunchArgument(
                "max_samples",
                default_value="50000",
                description="Stop the recorder and the complete simulation at this count.",
            ),
            DeclareLaunchArgument(
                "camera_front_topic",
                default_value="/image_raw",
                description="Image representation stored as the training input.",
            ),
            DeclareLaunchArgument(
                "image_format",
                default_value="jpg",
                description="jpg for raw RGB; use png for canonical semantic images.",
            ),
            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH",
                value=[
                    project_root,
                    ":",
                    EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
                ],
            ),
            gazebo,
            IncludeLaunchDescription(PythonLaunchDescriptionSource(bridge_launch)),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(rule_launch)),
            recorder,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=recorder,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(
                                reason="IL recorder finished; stopping complete simulation"
                            )
                        )
                    ],
                )
            ),
        ]
    )
