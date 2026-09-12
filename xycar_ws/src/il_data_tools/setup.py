from glob import glob
from os.path import isfile
from setuptools import find_packages, setup


package_name = "il_data_tools"
all_script_files = [path for path in glob("scripts/*") if isfile(path)]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/models", glob("models/*")),
        (f"share/{package_name}/scripts", all_script_files),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Team KAI",
    maintainer_email="maintainers@example.invalid",
    description="Imitation-learning collection, training, and inference for Xycar ROS2.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "il_common_recorder = il_data_tools.common_recorder_node:main",
            "il_policy_inference = il_data_tools.policy_inference_node:main",
            "il_recovery_scenario_manager = il_data_tools.recovery_scenario_manager:main",
            "il_canonical_artifact_augmenter = il_data_tools.canonical_artifact_node:main",
            "convert_canonical_visibility = il_data_tools.canonical_dataset_converter:main",
            "collect_randomized_batches = il_data_tools.randomized_batch_collector:main",
            "run_canonical_50k_pipeline = il_data_tools.canonical_pipeline:main",
            "run_canonical_pipeline = il_data_tools.canonical_pipeline:main",
            "train_from_raw_dataset.py = il_data_tools.train_from_raw_dataset:main",
        ],
    },
)
