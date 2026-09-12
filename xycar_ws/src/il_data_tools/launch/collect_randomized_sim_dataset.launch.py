import json
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from il_data_tools.domain_randomization import generate_randomized_assets


def _prepare_environment(context):
    project_root = Path(LaunchConfiguration("project_root").perform(context)).resolve()
    output_dir = Path(LaunchConfiguration("generated_output_dir").perform(context)).resolve()
    seed = int(LaunchConfiguration("seed").perform(context))
    preset = LaunchConfiguration("preset").perform(context)
    source_world = LaunchConfiguration("source_world").perform(context)
    source_bridge = LaunchConfiguration("source_bridge_config").perform(context)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{preset}_seed_{seed}"
    generated_world = output_dir / f"{stem}.sdf"
    generated_bridge = output_dir / f"{stem}_bridge.yaml"
    manifest = output_dir / f"{stem}_manifest.json"
    event_log = output_dir / f"{stem}_scenario_events.jsonl"
    artifact_event_log = output_dir / f"{stem}_canonical_artifacts.jsonl"
    if event_log.exists():
        event_log.unlink()
    if artifact_event_log.exists():
        artifact_event_log.unlink()

    result = generate_randomized_assets(
        source_world=source_world,
        output_world=str(generated_world),
        source_bridge_config=source_bridge,
        output_bridge_config=str(generated_bridge),
        manifest_path=str(manifest),
        seed=seed,
        preset=preset,
    )
    result["scenario_event_log"] = str(event_log)
    artifact_enabled = LaunchConfiguration("canonical_artifacts_enabled").perform(
        context
    ).lower() in {"1", "true", "yes", "on"}
    result["canonical_artifacts"] = {
        "enabled": artifact_enabled,
        "mode": LaunchConfiguration("canonical_artifact_mode").perform(context),
        "event_log": str(artifact_event_log),
        "event_start_probability": float(
            LaunchConfiguration("artifact_event_start_probability").perform(context)
        ),
        "duration_frames": [
            int(LaunchConfiguration("artifact_min_duration_frames").perform(context)),
            int(LaunchConfiguration("artifact_max_duration_frames").perform(context)),
        ],
        "center_jump_px": [
            float(LaunchConfiguration("artifact_center_jump_min_px").perform(context)),
            float(LaunchConfiguration("artifact_center_jump_max_px").perform(context)),
        ],
        "white_bend_px": [
            float(LaunchConfiguration("artifact_white_bend_min_px").perform(context)),
            float(LaunchConfiguration("artifact_white_bend_max_px").perform(context)),
        ],
        "visibility_target": {
            "white_both_probability": float(
                LaunchConfiguration("visibility_white_both_probability").perform(context)
            ),
            "white_one_probability": float(
                LaunchConfiguration("visibility_white_one_probability").perform(context)
            ),
            "white_none_probability": float(
                LaunchConfiguration("visibility_white_none_probability").perform(context)
            ),
            "yellow_visible_probability": float(
                LaunchConfiguration("visibility_yellow_probability").perform(context)
            ),
            "prevent_blank": LaunchConfiguration(
                "visibility_prevent_blank"
            ).perform(context).lower() in {"1", "true", "yes", "on"},
        },
    }
    manifest.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    context.launch_configurations["generated_world"] = str(generated_world)
    context.launch_configurations["generated_bridge_config"] = str(generated_bridge)
    context.launch_configurations["run_manifest_path"] = str(manifest)
    context.launch_configurations["scenario_event_log"] = str(event_log)
    context.launch_configurations["artifact_event_log"] = str(artifact_event_log)
    print(
        "[domain-randomization] "
        f"preset={result['preset']} seed={result['seed']} world={generated_world}"
    )

    show_gui = LaunchConfiguration("show_gui").perform(context).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    actions = [
        ExecuteProcess(
            cmd=["gz", "sim", "-s", "-r", str(generated_world)],
            name="kookmin_randomized_gazebo_server",
            output="screen",
        )
    ]
    if show_gui:
        actions.append(
            TimerAction(
                period=3.0,
                actions=[
                    ExecuteProcess(
                        cmd=[
                            "env",
                            "-u",
                            "QT_PLUGIN_PATH",
                            "-u",
                            "QT_QPA_PLATFORM_PLUGIN_PATH",
                            "-u",
                            "QT_QPA_FONTDIR",
                            "gz",
                            "sim",
                            "-g",
                        ],
                        name="kookmin_randomized_gazebo_gui",
                        output="screen",
                    )
                ],
            )
        )
    return actions


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    output_root = LaunchConfiguration("output_root")
    session_name = LaunchConfiguration("session_name")
    max_samples = LaunchConfiguration("max_samples")
    seed = LaunchConfiguration("seed")
    preset = LaunchConfiguration("preset")
    show_gui = LaunchConfiguration("show_gui")
    camera_front_topic = LaunchConfiguration("camera_front_topic")
    image_format = LaunchConfiguration("image_format")
    max_save_rate_hz = LaunchConfiguration("max_save_rate_hz")

    bridge_launch = PathJoinSubstitution(
        [FindPackageShare("xycar_gazebo_bridge"), "launch", "xycar_gazebo_rviz.launch.py"]
    )
    rule_launch = PathJoinSubstitution(
        [FindPackageShare("xycar_rule_drive"), "launch", "lane_rule_driver.launch.py"]
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
                "motor_topic": "/xycar_motor",
                "motor_msg_type": "float32_multi_array",
                "mission_label_topic": "/il/mission_label",
                "default_mission_label": "general_drive",
                "run_manifest_path": LaunchConfiguration("run_manifest_path"),
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
                "max_save_rate_hz": ParameterValue(
                    max_save_rate_hz, value_type=float
                ),
                "enable_recording_on_start": True,
                "exclude_bad_data": True,
                "exclude_idle": True,
                "exclude_zero_speed": True,
                "exclude_blank_canonical": ParameterValue(
                    LaunchConfiguration("exclude_blank_canonical"), value_type=bool
                ),
                "canonical_min_lane_pixels": 15,
                "canonical_min_lane_rows": 6,
                "bad_data_preroll_sec": 10.0,
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
            ),
            DeclareLaunchArgument(
                "source_world",
                default_value=PathJoinSubstitution(
                    [project_root, "worlds", "kookmin_xycar_track_final.sdf"]
                ),
            ),
            DeclareLaunchArgument(
                "source_bridge_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("xycar_gazebo_bridge"),
                        "config",
                        "xycar_gazebo_bridge.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "generated_output_dir",
                default_value=PathJoinSubstitution(
                    [project_root, "generated", "domain_randomization"]
                ),
            ),
            DeclareLaunchArgument(
                "output_root",
                default_value=PathJoinSubstitution([project_root, "datasets", "il"]),
            ),
            DeclareLaunchArgument("session_name", default_value="sim_randomized"),
            DeclareLaunchArgument("max_samples", default_value="5000"),
            DeclareLaunchArgument(
                "camera_front_topic",
                default_value="/image_raw",
                description="Raw RGB or canonical BEV image stored by the recorder.",
            ),
            DeclareLaunchArgument(
                "image_format",
                default_value="jpg",
                description="Use png for canonical semantic images.",
            ),
            DeclareLaunchArgument("max_save_rate_hz", default_value="10.0"),
            DeclareLaunchArgument(
                "exclude_blank_canonical",
                default_value="false",
                description="Skip canonical frames that contain no usable lane pixels.",
            ),
            DeclareLaunchArgument(
                "canonical_artifacts_enabled",
                default_value="false",
                description=(
                    "Inject short real-camera-like canonical lane artifacts into "
                    "the recorder input while the expert uses the clean stream."
                ),
            ),
            DeclareLaunchArgument(
                "canonical_artifact_mode",
                default_value="real_visibility",
                description="real_visibility preserves geometry; legacy bends or moves lines.",
            ),
            DeclareLaunchArgument(
                "canonical_artifact_input_topic",
                default_value="/perception/canonical_road_image",
            ),
            DeclareLaunchArgument(
                "canonical_artifact_output_topic",
                default_value="/perception/canonical_road_image_augmented",
            ),
            DeclareLaunchArgument("artifact_event_log", default_value=""),
            DeclareLaunchArgument(
                "artifact_event_start_probability", default_value="0.018"
            ),
            DeclareLaunchArgument("artifact_min_duration_frames", default_value="3"),
            DeclareLaunchArgument("artifact_max_duration_frames", default_value="9"),
            DeclareLaunchArgument("artifact_center_jump_min_px", default_value="7.0"),
            DeclareLaunchArgument("artifact_center_jump_max_px", default_value="22.0"),
            DeclareLaunchArgument("artifact_white_bend_min_px", default_value="18.0"),
            DeclareLaunchArgument("artifact_white_bend_max_px", default_value="58.0"),
            DeclareLaunchArgument(
                "visibility_white_both_probability", default_value="0.462"
            ),
            DeclareLaunchArgument(
                "visibility_white_one_probability", default_value="0.512"
            ),
            DeclareLaunchArgument(
                "visibility_white_none_probability", default_value="0.026"
            ),
            DeclareLaunchArgument(
                "visibility_yellow_probability", default_value="0.571"
            ),
            DeclareLaunchArgument("visibility_white_min_frames", default_value="3"),
            DeclareLaunchArgument("visibility_white_max_frames", default_value="40"),
            DeclareLaunchArgument("visibility_yellow_min_frames", default_value="3"),
            DeclareLaunchArgument("visibility_yellow_max_frames", default_value="30"),
            DeclareLaunchArgument("visibility_prevent_blank", default_value="true"),
            DeclareLaunchArgument("seed", default_value="2026"),
            DeclareLaunchArgument(
                "preset",
                default_value="mixed",
                description="baseline, visual_light, visual_dark, sensor, dynamics, or mixed",
            ),
            DeclareLaunchArgument(
                "show_gui",
                default_value="true",
                description="Show Gazebo and RViz for inspection; false is faster for batches.",
            ),
            DeclareLaunchArgument("generated_world", default_value=""),
            DeclareLaunchArgument("generated_bridge_config", default_value=""),
            DeclareLaunchArgument("run_manifest_path", default_value=""),
            DeclareLaunchArgument("scenario_event_log", default_value=""),
            DeclareLaunchArgument("scenario_interval_sec", default_value="24.0"),
            DeclareLaunchArgument("scenario_warmup_sec", default_value="12.0"),
            DeclareLaunchArgument("scenario_settle_sec", default_value="0.8"),
            DeclareLaunchArgument("recovery_hold_sec", default_value="9.0"),
            DeclareLaunchArgument("stop_retry_hold_sec", default_value="1.0"),
            DeclareLaunchArgument("retry_delay_sec", default_value="1.0"),
            DeclareLaunchArgument(
                "lane_offset_from_yellow_m",
                default_value="0.05",
                description="Nominal rule path offset from the yellow centerline.",
            ),
            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH",
                value=[
                    project_root,
                    ":",
                    EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
                ],
            ),
            OpaqueFunction(function=_prepare_environment),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(bridge_launch),
                launch_arguments={
                    "params_file": LaunchConfiguration("generated_bridge_config"),
                    "enable_rviz": show_gui,
                }.items(),
            ),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(rule_launch)),
            Node(
                package="il_data_tools",
                executable="il_canonical_artifact_augmenter",
                name="il_canonical_artifact_augmenter",
                output="screen",
                condition=IfCondition(
                    LaunchConfiguration("canonical_artifacts_enabled")
                ),
                parameters=[
                    {
                        "use_sim_time": True,
                        "seed": ParameterValue(seed, value_type=int),
                        "mode": LaunchConfiguration("canonical_artifact_mode"),
                        "input_topic": LaunchConfiguration(
                            "canonical_artifact_input_topic"
                        ),
                        "output_topic": LaunchConfiguration(
                            "canonical_artifact_output_topic"
                        ),
                        "event_log_path": LaunchConfiguration("artifact_event_log"),
                        "event_start_probability": ParameterValue(
                            LaunchConfiguration("artifact_event_start_probability"),
                            value_type=float,
                        ),
                        "min_duration_frames": ParameterValue(
                            LaunchConfiguration("artifact_min_duration_frames"),
                            value_type=int,
                        ),
                        "max_duration_frames": ParameterValue(
                            LaunchConfiguration("artifact_max_duration_frames"),
                            value_type=int,
                        ),
                        "center_jump_min_px": ParameterValue(
                            LaunchConfiguration("artifact_center_jump_min_px"),
                            value_type=float,
                        ),
                        "center_jump_max_px": ParameterValue(
                            LaunchConfiguration("artifact_center_jump_max_px"),
                            value_type=float,
                        ),
                        "white_bend_min_px": ParameterValue(
                            LaunchConfiguration("artifact_white_bend_min_px"),
                            value_type=float,
                        ),
                        "white_bend_max_px": ParameterValue(
                            LaunchConfiguration("artifact_white_bend_max_px"),
                            value_type=float,
                        ),
                        "white_both_probability": ParameterValue(
                            LaunchConfiguration("visibility_white_both_probability"),
                            value_type=float,
                        ),
                        "white_one_probability": ParameterValue(
                            LaunchConfiguration("visibility_white_one_probability"),
                            value_type=float,
                        ),
                        "white_none_probability": ParameterValue(
                            LaunchConfiguration("visibility_white_none_probability"),
                            value_type=float,
                        ),
                        "yellow_visible_probability": ParameterValue(
                            LaunchConfiguration("visibility_yellow_probability"),
                            value_type=float,
                        ),
                        "white_min_duration_frames": ParameterValue(
                            LaunchConfiguration("visibility_white_min_frames"),
                            value_type=int,
                        ),
                        "white_max_duration_frames": ParameterValue(
                            LaunchConfiguration("visibility_white_max_frames"),
                            value_type=int,
                        ),
                        "yellow_min_duration_frames": ParameterValue(
                            LaunchConfiguration("visibility_yellow_min_frames"),
                            value_type=int,
                        ),
                        "yellow_max_duration_frames": ParameterValue(
                            LaunchConfiguration("visibility_yellow_max_frames"),
                            value_type=int,
                        ),
                        "prevent_blank": ParameterValue(
                            LaunchConfiguration("visibility_prevent_blank"),
                            value_type=bool,
                        ),
                    }
                ],
            ),
            Node(
                package="il_data_tools",
                executable="il_recovery_scenario_manager",
                name="il_recovery_scenario_manager",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "seed": ParameterValue(seed, value_type=int),
                        "preset": preset,
                        "event_log_path": LaunchConfiguration("scenario_event_log"),
                        "warmup_sec": ParameterValue(
                            LaunchConfiguration("scenario_warmup_sec"), value_type=float
                        ),
                        "settle_sec": ParameterValue(
                            LaunchConfiguration("scenario_settle_sec"), value_type=float
                        ),
                        "scenario_interval_sec": ParameterValue(
                            LaunchConfiguration("scenario_interval_sec"), value_type=float
                        ),
                        "recovery_hold_sec": ParameterValue(
                            LaunchConfiguration("recovery_hold_sec"), value_type=float
                        ),
                        "lane_offset_from_yellow_m": ParameterValue(
                            LaunchConfiguration("lane_offset_from_yellow_m"),
                            value_type=float,
                        ),
                        "stop_retry_hold_sec": ParameterValue(
                            LaunchConfiguration("stop_retry_hold_sec"),
                            value_type=float,
                        ),
                        "retry_delay_sec": ParameterValue(
                            LaunchConfiguration("retry_delay_sec"),
                            value_type=float,
                        ),
                    }
                ],
            ),
            recorder,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=recorder,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(
                                reason="randomized IL session reached its sample limit"
                            )
                        )
                    ],
                )
            ),
        ]
    )
