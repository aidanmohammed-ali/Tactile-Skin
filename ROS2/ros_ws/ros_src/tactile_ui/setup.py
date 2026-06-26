from setuptools import find_packages, setup

package_name = "tactile_ui"

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
    description="Unified Tkinter operator UI for vision and arm control.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "operator_ui_node = tactile_ui.operator_ui_node:main",
        ],
    },
)
