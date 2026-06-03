from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from pathlib import Path


def _launch_nodes(context):
    mode = LaunchConfiguration("mode").perform(context)
    start_ui = LaunchConfiguration("start_ui").perform(context).lower() in ("1", "true", "yes")
    config_path = str(Path(FindPackageShare("tactile_bringup").perform(context)) / "config" / "system.yaml")
    arm_executable = "hardware_arm_node" if mode == "hardware" else "sim_arm_node"

    nodes = [
        Node(
            package="tactile_vision",
            executable="vision_node",
            name="vision_node",
            output="screen",
            parameters=[config_path],
        ),
        Node(
            package="tactile_arm",
            executable=arm_executable,
            name=arm_executable,
            output="screen",
            parameters=[config_path],
        ),
        Node(
            package="tactile_task",
            executable="pick_place_node",
            name="pick_place_node",
            output="screen",
            parameters=[config_path],
        ),
    ]
    if start_ui:
        nodes.append(
            Node(
                package="tactile_ui",
                executable="operator_ui_node",
                name="operator_ui_node",
                output="screen",
                parameters=[config_path],
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="sim", description="sim or hardware"),
            DeclareLaunchArgument("start_ui", default_value="true", description="start the Tkinter operator UI"),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
