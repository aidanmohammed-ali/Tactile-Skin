from glob import glob
from setuptools import find_packages, setup

package_name = "tactile_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={
        "tactile_vision": ["block3.pt", "cv_aruco_src/board_calibration.json"],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jay",
    maintainer_email="jay@todo.todo",
    description="Camera, ArUco calibration, and block detection nodes.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vision_node = tactile_vision.vision_node:main",
        ],
    },
)
