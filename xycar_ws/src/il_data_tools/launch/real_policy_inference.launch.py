from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model_path = LaunchConfiguration("model_path")
    image_topic = LaunchConfiguration("image_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    motor_topic = LaunchConfiguration("motor_topic")
    shadow_topic = LaunchConfiguration("shadow_topic")
    debug_topic = LaunchConfiguration("debug_topic")
    debug_image_topic = LaunchConfiguration("debug_image_topic")
    drive_enabled = LaunchConfiguration("drive_enabled")
    speed_command = LaunchConfiguration("speed_command")
    min_speed_command = LaunchConfiguration("min_speed_command")
    max_steer_scale = LaunchConfiguration("max_steer_scale")
    steering_output_sign = LaunchConfiguration("steering_output_sign")
    angle_command_min = LaunchConfiguration("angle_command_min")
    angle_command_max = LaunchConfiguration("angle_command_max")
    steering_temporal_alpha = LaunchConfiguration("steering_temporal_alpha")
    slow_down_angle_cmd = LaunchConfiguration("slow_down_angle_cmd")
    max_abs_angle_for_drive = LaunchConfiguration("max_abs_angle_for_drive")
    sync_tolerance_sec = LaunchConfiguration("sync_tolerance_sec")
    sensor_timeout_sec = LaunchConfiguration("sensor_timeout_sec")
    max_inference_rate_hz = LaunchConfiguration("max_inference_rate_hz")
    device = LaunchConfiguration("device")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_path",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("il_data_tools"),
                        "models",
                        "drive_policy_scripted.pt",
                    ]
                ),
                description="Packaged TorchScript drive policy.",
            ),
            DeclareLaunchArgument("image_topic", default_value="/image_raw"),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("motor_topic", default_value="/xycar_motor"),
            DeclareLaunchArgument(
                "shadow_topic",
                default_value="/il/policy_motor_shadow",
            ),
            DeclareLaunchArgument(
                "debug_topic",
                default_value="/il/policy_debug",
            ),
            DeclareLaunchArgument(
                "debug_image_topic",
                default_value="/il/policy_input_image",
                description="Exact cropped and resized image received by the policy.",
            ),
            DeclareLaunchArgument(
                "drive_enabled",
                default_value="false",
                description="Publish to the physical motor only when explicitly enabled.",
            ),
            DeclareLaunchArgument(
                "speed_command",
                default_value="3.0",
                description="Measured minimum launch command for the first drive test.",
            ),
            DeclareLaunchArgument(
                "min_speed_command",
                default_value="3.0",
                description="Keep curve speed at or above the measured launch threshold.",
            ),
            DeclareLaunchArgument(
                "max_steer_scale",
                default_value="100.0",
                description="Convert normalized model output back to Xycar angle command.",
            ),
            DeclareLaunchArgument(
                "steering_output_sign",
                default_value="1.0",
                description="Set to -1.0 only when shadow steering direction is reversed.",
            ),
            DeclareLaunchArgument("angle_command_min", default_value="-42.0"),
            DeclareLaunchArgument("angle_command_max", default_value="42.0"),
            DeclareLaunchArgument(
                "steering_temporal_alpha",
                default_value="0.55",
                description="New steering sample weight; higher values respond faster.",
            ),
            DeclareLaunchArgument(
                "slow_down_angle_cmd",
                default_value="18.0",
                description="Begin reducing speed above this absolute steering command.",
            ),
            DeclareLaunchArgument(
                "max_abs_angle_for_drive",
                default_value="43.0",
                description="Publish zero speed when steering reaches this magnitude.",
            ),
            DeclareLaunchArgument(
                "sync_tolerance_sec",
                default_value="0.05",
                description="Maximum camera-to-LiDAR timestamp difference.",
            ),
            DeclareLaunchArgument(
                "sensor_timeout_sec",
                default_value="0.50",
                description="Stop if no synchronized inference completes in this time.",
            ),
            DeclareLaunchArgument(
                "max_inference_rate_hz",
                default_value="15.0",
                description="Maximum policy inference frequency.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cpu",
                description="Use CPU on the AMD real-car mini PC unless CUDA is available.",
            ),
            Node(
                package="il_data_tools",
                executable="il_policy_inference",
                name="il_policy_inference",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": False,
                        "model_path": model_path,
                        "device": device,
                        "image_topic": image_topic,
                        "scan_topic": scan_topic,
                        "motor_topic": motor_topic,
                        "shadow_topic": shadow_topic,
                        "debug_topic": debug_topic,
                        "debug_image_topic": debug_image_topic,
                        "drive_enabled": ParameterValue(
                            drive_enabled, value_type=bool
                        ),
                        "speed_command": ParameterValue(
                            speed_command, value_type=float
                        ),
                        "min_speed_command": ParameterValue(
                            min_speed_command, value_type=float
                        ),
                        "max_steer_scale": ParameterValue(
                            max_steer_scale, value_type=float
                        ),
                        "steering_output_sign": ParameterValue(
                            steering_output_sign, value_type=float
                        ),
                        "angle_command_min": ParameterValue(
                            angle_command_min, value_type=float
                        ),
                        "angle_command_max": ParameterValue(
                            angle_command_max, value_type=float
                        ),
                        "steering_temporal_alpha": ParameterValue(
                            steering_temporal_alpha, value_type=float
                        ),
                        "slow_down_angle_cmd": ParameterValue(
                            slow_down_angle_cmd, value_type=float
                        ),
                        "max_abs_angle_for_drive": ParameterValue(
                            max_abs_angle_for_drive, value_type=float
                        ),
                        "sync_tolerance_sec": ParameterValue(
                            sync_tolerance_sec, value_type=float
                        ),
                        "sensor_timeout_sec": ParameterValue(
                            sensor_timeout_sec, value_type=float
                        ),
                        "max_inference_rate_hz": ParameterValue(
                            max_inference_rate_hz, value_type=float
                        ),
                    }
                ],
            ),
        ]
    )
