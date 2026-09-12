from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    policy_kind = LaunchConfiguration("policy_kind")
    drive_enabled = LaunchConfiguration("drive_enabled")
    speed_command = LaunchConfiguration("speed_command")
    min_speed_command = LaunchConfiguration("min_speed_command")
    max_speed_command = LaunchConfiguration("max_speed_command")
    deployment_speed_cap = LaunchConfiguration("deployment_speed_cap")
    speed_temporal_alpha = LaunchConfiguration("speed_temporal_alpha")
    max_inference_rate_hz = LaunchConfiguration("max_inference_rate_hz")
    device = LaunchConfiguration("device")
    return LaunchDescription(
        [
            DeclareLaunchArgument("policy_kind", default_value="bc_scripted"),
            DeclareLaunchArgument(
                "checkpoint_path",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("il_data_tools"),
                        "models",
                        "drive_canonical_policy_scripted.pt",
                    ]
                ),
            ),
            DeclareLaunchArgument("residual_base_checkpoint", default_value=""),
            DeclareLaunchArgument(
                "image_topic", default_value="/perception/canonical_road_image"
            ),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("drive_enabled", default_value="false"),
            DeclareLaunchArgument("speed_command", default_value="3.0"),
            DeclareLaunchArgument("min_speed_command", default_value="4.0"),
            DeclareLaunchArgument("max_speed_command", default_value="24.0"),
            DeclareLaunchArgument(
                "deployment_speed_cap",
                default_value="0.0",
                description=(
                    "Positive values clip learned speed; zero disables the "
                    "additional deployment cap."
                ),
            ),
            DeclareLaunchArgument("speed_temporal_alpha", default_value="0.35"),
            DeclareLaunchArgument(
                "max_inference_rate_hz",
                default_value="0.0",
                description=(
                    "Zero processes every canonical frame. Use a positive value "
                    "only when the source is faster than the policy can handle."
                ),
            ),
            DeclareLaunchArgument(
                "inference_rate_slack_sec", default_value="0.04"
            ),
            DeclareLaunchArgument("max_image_age_sec", default_value="0.30"),
            DeclareLaunchArgument(
                "max_temporal_frame_gap_sec", default_value="0.25"
            ),
            DeclareLaunchArgument("lidar_safety_enabled", default_value="false"),
            DeclareLaunchArgument("steering_gain", default_value="1.0"),
            DeclareLaunchArgument("steering_output_sign", default_value="1.0"),
            DeclareLaunchArgument("adaptive_steering_enabled", default_value="true"),
            DeclareLaunchArgument(
                "straight_steering_temporal_alpha", default_value="0.20"
            ),
            DeclareLaunchArgument("steering_temporal_alpha", default_value="0.90"),
            DeclareLaunchArgument("steering_straight_threshold", default_value="0.08"),
            DeclareLaunchArgument("steering_curve_threshold", default_value="0.35"),
            DeclareLaunchArgument("straight_steering_rate_limit", default_value="0.05"),
            DeclareLaunchArgument("curve_steering_rate_limit", default_value="0.38"),
            DeclareLaunchArgument("steering_deadband", default_value="0.02"),
            DeclareLaunchArgument("turn_in_anticipation_gain", default_value="0.35"),
            DeclareLaunchArgument(
                "turn_in_anticipation_threshold", default_value="0.04"
            ),
            DeclareLaunchArgument("preview_steering_enabled", default_value="false"),
            DeclareLaunchArgument("preview_steering_blend", default_value="0.35"),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument("policy_cpu_threads", default_value="4"),
            DeclareLaunchArgument(
                "policy_opencv_threads", default_value="1"
            ),
            Node(
                package="xycar_rl",
                executable="rl_policy_inference",
                name="rl_policy_inference",
                output="screen",
                additional_env={
                    "OMP_NUM_THREADS": LaunchConfiguration(
                        "policy_cpu_threads"
                    ),
                    "OMP_WAIT_POLICY": "PASSIVE",
                    "MKL_NUM_THREADS": LaunchConfiguration(
                        "policy_cpu_threads"
                    ),
                    "OPENBLAS_NUM_THREADS": "1",
                },
                parameters=[
                    {
                        "use_sim_time": False,
                        "policy_kind": policy_kind,
                        "checkpoint_path": checkpoint_path,
                        "residual_base_checkpoint": LaunchConfiguration(
                            "residual_base_checkpoint"
                        ),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "scan_topic": LaunchConfiguration("scan_topic"),
                        "drive_enabled": ParameterValue(
                            drive_enabled, value_type=bool
                        ),
                        "speed_command": ParameterValue(
                            speed_command, value_type=float
                        ),
                        "min_speed_command": ParameterValue(
                            min_speed_command, value_type=float
                        ),
                        "max_speed_command": ParameterValue(
                            max_speed_command, value_type=float
                        ),
                        "deployment_speed_cap": ParameterValue(
                            deployment_speed_cap, value_type=float
                        ),
                        "speed_temporal_alpha": ParameterValue(
                            speed_temporal_alpha, value_type=float
                        ),
                        "max_inference_rate_hz": ParameterValue(
                            max_inference_rate_hz, value_type=float
                        ),
                        "inference_rate_slack_sec": ParameterValue(
                            LaunchConfiguration("inference_rate_slack_sec"),
                            value_type=float,
                        ),
                        "max_image_age_sec": ParameterValue(
                            LaunchConfiguration("max_image_age_sec"),
                            value_type=float,
                        ),
                        "max_temporal_frame_gap_sec": ParameterValue(
                            LaunchConfiguration("max_temporal_frame_gap_sec"),
                            value_type=float,
                        ),
                        "lidar_safety_enabled": ParameterValue(
                            LaunchConfiguration("lidar_safety_enabled"),
                            value_type=bool,
                        ),
                        "steering_gain": ParameterValue(
                            LaunchConfiguration("steering_gain"), value_type=float
                        ),
                        "steering_output_sign": ParameterValue(
                            LaunchConfiguration("steering_output_sign"),
                            value_type=float,
                        ),
                        "adaptive_steering_enabled": ParameterValue(
                            LaunchConfiguration("adaptive_steering_enabled"),
                            value_type=bool,
                        ),
                        "straight_steering_temporal_alpha": ParameterValue(
                            LaunchConfiguration("straight_steering_temporal_alpha"),
                            value_type=float,
                        ),
                        "steering_temporal_alpha": ParameterValue(
                            LaunchConfiguration("steering_temporal_alpha"),
                            value_type=float,
                        ),
                        "steering_straight_threshold": ParameterValue(
                            LaunchConfiguration("steering_straight_threshold"),
                            value_type=float,
                        ),
                        "steering_curve_threshold": ParameterValue(
                            LaunchConfiguration("steering_curve_threshold"),
                            value_type=float,
                        ),
                        "straight_steering_rate_limit": ParameterValue(
                            LaunchConfiguration("straight_steering_rate_limit"),
                            value_type=float,
                        ),
                        "curve_steering_rate_limit": ParameterValue(
                            LaunchConfiguration("curve_steering_rate_limit"),
                            value_type=float,
                        ),
                        "steering_deadband": ParameterValue(
                            LaunchConfiguration("steering_deadband"),
                            value_type=float,
                        ),
                        "turn_in_anticipation_gain": ParameterValue(
                            LaunchConfiguration("turn_in_anticipation_gain"),
                            value_type=float,
                        ),
                        "turn_in_anticipation_threshold": ParameterValue(
                            LaunchConfiguration("turn_in_anticipation_threshold"),
                            value_type=float,
                        ),
                        "preview_steering_enabled": ParameterValue(
                            LaunchConfiguration("preview_steering_enabled"),
                            value_type=bool,
                        ),
                        "preview_steering_blend": ParameterValue(
                            LaunchConfiguration("preview_steering_blend"),
                            value_type=float,
                        ),
                        "device": device,
                        "policy_cpu_threads": ParameterValue(
                            LaunchConfiguration("policy_cpu_threads"),
                            value_type=int,
                        ),
                        "policy_opencv_threads": ParameterValue(
                            LaunchConfiguration("policy_opencv_threads"),
                            value_type=int,
                        ),
                    }
                ],
            ),
        ]
    )
