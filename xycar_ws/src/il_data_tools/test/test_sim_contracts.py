from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class SimulationContractTests(unittest.TestCase):
    def test_packaged_policy_is_the_validated_full_dataset_model(self):
        model = ROOT / "models" / "drive_policy_scripted.pt"
        self.assertGreater(model.stat().st_size, 40_000_000)
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "960d5dcb64d7cae42f23e1038d36c7d866c510f35bab2f5240f898283ff1fbfa",
        )

    def test_sim_recorder_uses_gazebo_clock_and_real_vehicle_topics(self):
        source = (ROOT / "launch" / "record_sim_drive_dataset.launch.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('SetParameter(name="use_sim_time", value=True)', source)
        self.assertIn('default_value="/image_raw"', source)
        self.assertIn('"camera_front_topic": camera_front_topic', source)
        self.assertIn('"scan_topic": "/scan"', source)
        self.assertIn('"motor_topic": "/xycar_motor"', source)
        self.assertIn('"motor_msg_type": "float32_multi_array"', source)
        self.assertIn('default_value="50000"', source)
        self.assertIn('SetParameter(name="max_samples", value=max_samples)', source)
        recorder_source = (ROOT / "il_data_tools" / "common_recorder_node.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('name="il-recorder-shutdown"', recorder_source)

    def test_full_collection_launch_stops_everything_with_the_recorder(self):
        source = (ROOT / "launch" / "collect_sim_drive_dataset.launch.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('default_value="50000"', source)
        self.assertIn("OnProcessExit(", source)
        self.assertIn("target_action=recorder", source)
        self.assertIn("event=Shutdown(", source)

    def test_canonical_collection_is_isolated_from_raw_dataset(self):
        source = (
            ROOT / "launch" / "collect_sim_canonical_drive_dataset.launch.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"datasets", "il_canonical"', source)
        self.assertIn('"camera_front_topic": "/perception/canonical_road_image"', source)
        self.assertIn('"session_name": "sim_canonical_drive"', source)
        self.assertIn('"image_format": "png"', source)

    def test_sim_policy_drive_starts_inference_without_rule_driver(self):
        source = (ROOT / "launch" / "sim_policy_drive.launch.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('default_value="true"', source)
        self.assertIn('default_value="4.0"', source)
        self.assertIn("policy_inference.launch.py", source)
        self.assertIn("drive_canonical_policy_scripted.pt", source)
        self.assertIn('/perception/canonical_road_image', source)
        self.assertNotIn("lane_rule_driver.launch.py", source)

    def test_packaged_canonical_policy_is_the_200k_validated_model(self):
        model = ROOT / "models" / "drive_canonical_policy_scripted.pt"
        self.assertGreater(model.stat().st_size, 40_000_000)
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "8da35fa8a56904f9970679da0af68d938f60848991af3abf9a37027d9195b58b",
        )

    def test_randomized_collection_keeps_manifest_and_recovery_labels(self):
        source = (
            ROOT / "launch" / "collect_randomized_sim_dataset.launch.py"
        ).read_text(encoding="utf-8")
        self.assertIn("generate_randomized_assets", source)
        self.assertIn('"run_manifest_path": LaunchConfiguration', source)
        self.assertIn('"exclude_bad_data": True', source)
        self.assertIn('"exclude_zero_speed": True', source)
        self.assertIn('"bad_data_preroll_sec": 10.0', source)
        self.assertIn('"stop_retry_hold_sec"', source)
        self.assertIn('"retry_delay_sec"', source)
        self.assertIn('executable="il_recovery_scenario_manager"', source)
        self.assertIn("target_action=recorder", source)
        self.assertIn('"camera_front_topic": camera_front_topic', source)
        self.assertIn('"image_format": image_format', source)
        self.assertIn('"lane_offset_from_yellow_m": ParameterValue', source)
        self.assertIn('["gz", "sim", "-s", "-r", str(generated_world)]', source)
        self.assertIn('"QT_QPA_PLATFORM_PLUGIN_PATH"', source)
        self.assertIn('"gz",', source)
        self.assertIn('"sim",', source)
        self.assertIn('"-g",', source)

    def test_randomized_batch_collector_can_store_canonical_recovery_data(self):
        source = (
            ROOT / "il_data_tools" / "randomized_batch_collector.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--canonical-input"', source)
        self.assertIn("/perception/canonical_road_image", source)
        self.assertIn("image_format:=png", source)
        self.assertIn("def cleanup_gazebo_children()", source)
        self.assertIn("finally:\n            cleanup_gazebo_children()", source)

    def test_real_artifacts_are_applied_only_to_recorded_canonical_input(self):
        launch_source = (
            ROOT / "launch" / "collect_randomized_sim_dataset.launch.py"
        ).read_text(encoding="utf-8")
        collector_source = (
            ROOT / "il_data_tools" / "randomized_batch_collector.py"
        ).read_text(encoding="utf-8")
        pipeline_source = (
            ROOT / "il_data_tools" / "canonical_pipeline.py"
        ).read_text(encoding="utf-8")

        self.assertIn("il_canonical_artifact_augmenter", launch_source)
        self.assertIn("/perception/canonical_road_image_augmented", collector_source)
        self.assertIn('"exclude_blank_canonical"', launch_source)
        self.assertIn('"exclude_blank_canonical:=true"', collector_source)
        self.assertIn('command.append("--canonical-artifacts")', pipeline_source)

    def test_canonical_pipeline_has_quality_gates_before_poweroff(self):
        source = (ROOT / "il_data_tools" / "canonical_pipeline.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("validate_sessions", source)
        self.assertIn("max_test_mae_command", source)
        self.assertLess(source.index("publish_model("), source.index("poweroff()"))

    def test_real_canonical_policy_launch_starts_in_shadow(self):
        source = (
            ROOT / "launch" / "real_canonical_policy_drive.launch.py"
        ).read_text(encoding="utf-8")
        self.assertIn("real_canonical_perception.launch.py", source)
        self.assertIn("real_policy_inference.launch.py", source)
        self.assertIn("/perception/canonical_road_image", source)
        self.assertIn('"drive_enabled",\n                default_value="false"', source)

    def test_recorder_does_not_shutdown_an_already_closed_context(self):
        source = (ROOT / "il_data_tools" / "common_recorder_node.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("if rclpy.ok():\n            rclpy.shutdown()", source)

    def test_runtime_preprocessing_matches_training_contract(self):
        scripts = ROOT / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            from image_preprocessing import preprocess_bgr_image as train_image
            from policy_dataset import load_lidar_tensor
            from il_data_tools.runtime_preprocessing import (
                preprocess_bgr_image as runtime_image,
                preprocess_lidar_ranges,
            )

            image = np.random.default_rng(42).integers(
                0,
                256,
                size=(1024, 1280, 3),
                dtype=np.uint8,
            )
            np.testing.assert_allclose(
                runtime_image(image, 160, 90),
                train_image(image, 160, 90),
                rtol=0.0,
                atol=0.0,
            )

            ranges = np.linspace(0.1, 12.0, 505, dtype=np.float32)
            ranges[20] = np.inf
            ranges[40] = np.nan
            with tempfile.NamedTemporaryFile(suffix=".npz") as handle:
                np.savez(
                    handle.name,
                    ranges=ranges,
                    range_min=np.float32(0.1),
                    range_max=np.float32(12.0),
                )
                training_lidar = load_lidar_tensor(handle.name, 360).numpy()
            np.testing.assert_allclose(
                preprocess_lidar_ranges(ranges, 0.1, 12.0, 360),
                training_lidar,
                rtol=0.0,
                atol=0.0,
            )
        finally:
            sys.path.remove(str(scripts))


if __name__ == "__main__":
    unittest.main()
