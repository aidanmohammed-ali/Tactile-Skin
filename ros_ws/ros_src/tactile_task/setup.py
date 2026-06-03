from setuptools import find_packages, setup

package_name = "tactile_task"

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
    description="Task-level pick and place topic coordinator.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pick_place_node = tactile_task.pick_place_node:main",
        ],
    },
)
