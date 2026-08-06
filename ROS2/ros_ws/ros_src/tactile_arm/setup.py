from setuptools import find_packages, setup

package_name = "tactile_arm"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jay",
    maintainer_email="jay@todo.todo",
    description="Dynamixel and simulated arm control for tactile block picking.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hardware_arm_node = tactile_arm.hardware_arm_node:main",
            "sim_arm_node = tactile_arm.sim_arm_node:main",
        ],
    },
)
