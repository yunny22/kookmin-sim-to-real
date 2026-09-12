from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def declare(name, default, description):
    return DeclareLaunchArgument(name, default_value=str(default), description=description)


def generate_launch_description():
    args = [
        declare("output_root", "~/xycar_ws/datasets/il", "Dataset output root."),
        declare("session_name", "overtake", "Session base name. With auto increment, overtake becomes overtake_01, overtake_02, ..."),
        declare("session_auto_increment", "true", "Create the next numbered session for session_name."),
        declare("camera_front_topic", "/image_raw", "Front camera topic."),
        declare("scan_topic", "/scan", "LaserScan topic."),
        declare("imu_topic", "/imu", "IMU topic."),
        declare("odom_topic", "/odom", "Odometry topic."),
        declare("motor_topic", "/xycar_motor", "Xycar motor command topic to subscribe."),
        declare("motor_msg_type", "float32_multi_array", "auto, xycar, or float32_multi_array."),
        declare("mission_label_topic", "/il/mission_label", "Mission label topic."),
        declare("default_mission_label", "vehicle_overtake", "Label to use when no label topic is received."),
        declare("save_scan_npz", "true", "Overtake profile default: save scan npz."),
        declare("max_save_rate_hz", "10.0", "Maximum sample save rate."),
        declare("min_abs_speed_to_save", "0.0", "Optional minimum absolute overtake speed."),
        declare("max_abs_speed_to_save", "0.0", "Optional maximum absolute overtake speed; 0 disables."),
        declare("writer_queue_size", "128", "Bounded asynchronous disk-writer queue."),
        declare("min_free_disk_gb", "10.0", "Stop before free disk falls below this value."),
        declare("disk_check_period_sec", "5.0", "Free-space check period."),
        declare("stop_on_low_disk", "true", "Stop recording safely on low disk."),
        declare("image_format", "jpg", "jpg or png."),
        declare("jpeg_quality", "90", "JPEG quality."),
        declare("enable_recording_on_start", "true", "Start recording immediately."),
    ]
    node = Node(
        package="il_data_tools",
        executable="il_common_recorder",
        name="il_common_recorder_overtake",
        output="screen",
        parameters=[
            {
                "output_root": LaunchConfiguration("output_root"),
                "session_name": LaunchConfiguration("session_name"),
                "session_auto_increment": ParameterValue(LaunchConfiguration("session_auto_increment"), value_type=bool),
                "dataset_profile": "overtake",
                "allowed_labels": "vehicle_overtake,overtake_start,overtake_end,recovery",
                "camera_front_topic": LaunchConfiguration("camera_front_topic"),
                "scan_topic": LaunchConfiguration("scan_topic"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "odom_topic": LaunchConfiguration("odom_topic"),
                "motor_topic": LaunchConfiguration("motor_topic"),
                "motor_msg_type": LaunchConfiguration("motor_msg_type"),
                "mission_label_topic": LaunchConfiguration("mission_label_topic"),
                "default_mission_label": LaunchConfiguration("default_mission_label"),
                "save_front_image": True,
                "save_scan_npz": ParameterValue(LaunchConfiguration("save_scan_npz"), value_type=bool),
                "writer_queue_size": ParameterValue(LaunchConfiguration("writer_queue_size"), value_type=int),
                "min_free_disk_gb": ParameterValue(LaunchConfiguration("min_free_disk_gb"), value_type=float),
                "disk_check_period_sec": ParameterValue(LaunchConfiguration("disk_check_period_sec"), value_type=float),
                "stop_on_low_disk": ParameterValue(LaunchConfiguration("stop_on_low_disk"), value_type=bool),
                "image_format": LaunchConfiguration("image_format"),
                "jpeg_quality": ParameterValue(LaunchConfiguration("jpeg_quality"), value_type=int),
                "max_save_rate_hz": ParameterValue(LaunchConfiguration("max_save_rate_hz"), value_type=float),
                "min_abs_speed_to_save": ParameterValue(LaunchConfiguration("min_abs_speed_to_save"), value_type=float),
                "max_abs_speed_to_save": ParameterValue(LaunchConfiguration("max_abs_speed_to_save"), value_type=float),
                "enable_recording_on_start": ParameterValue(LaunchConfiguration("enable_recording_on_start"), value_type=bool),
                "exclude_bad_data": True,
                "exclude_idle": True,
            }
        ],
    )
    return LaunchDescription(args + [node])
