from glob import glob
import os

from setuptools import find_packages, setup


package_name = "xycar_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "models"),
            glob("models/*"),
        ),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="Team KAI",
    maintainer_email="maintainers@example.invalid",
    description="Camera-based Xycar perception publisher compatible with KAIEV perception topics.",
    license="UNLICENSED",
    entry_points={
        "console_scripts": [
            "benchmark_yolo_lane = xycar_perception.benchmark_yolo_lane:main",
            "camera_perception_node = xycar_perception.camera_perception_node:main",
        ],
    },
)
