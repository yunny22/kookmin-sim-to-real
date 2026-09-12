from glob import glob
from pathlib import Path
from setuptools import find_packages, setup


package_name = "xycar_rl"
model_data_files = [
    (
        f"share/{package_name}/models/{Path(directory).name}",
        glob(f"{directory}/*"),
    )
    for directory in glob("models/*")
    if Path(directory).is_dir()
]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ] + model_data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Team KAI",
    maintainer_email="maintainers@example.invalid",
    description="Reinforcement-learning environment, training, and deployment for Xycar.",
    license="UNLICENSED",
    entry_points={
        "console_scripts": [
            "rl_transition_recorder = xycar_rl.transition_recorder_node:main",
            "rl_policy_inference = xycar_rl.policy_runtime_node:main",
            "train_td3_bc = xycar_rl.train_td3_bc:main",
            "train_camera_speed_bc = xycar_rl.train_camera_speed_bc:main",
            "train_camera_speed_td3_bc = xycar_rl.train_camera_speed_td3_bc:main",
            "train_lap_time_online = xycar_rl.train_lap_time_online:main",
            "train_residual_online = xycar_rl.train_residual_online:main",
            "evaluate_closed_loop = xycar_rl.evaluate_closed_loop:main",
            "rollout_policy = xycar_rl.rollout_policy:main",
            "rollout_track_expert = xycar_rl.rollout_track_expert:main",
            "collect_parallel_policy = xycar_rl.parallel_policy_collector:main",
            "rollout_dagger = xycar_rl.rollout_dagger:main",
        ],
    },
)
