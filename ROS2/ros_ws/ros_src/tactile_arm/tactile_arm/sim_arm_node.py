from __future__ import annotations

from typing import Sequence

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .arm_node_base import ArmNodeBase


class SimBackend:
    def __init__(self) -> None:
        self.connected = True
        self.current_port = "sim"
        self.joints = [0.0, 0.0, 0.0, 0.0]
        self.joint_speeds = [100, 100, 100, 100]
        self.gripper_position = 1800

    def connect(self, port_name: str) -> None:
        self.current_port = port_name or "sim"
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def set_joint_angles(self, joints: Sequence[float]) -> None:
        self.joints = list(float(v) for v in joints[:4])

    def set_joint_speeds(self, speeds: Sequence[int]) -> None:
        self.joint_speeds = list(int(v) for v in speeds[:4])

    def read_joint_angles(self) -> list[float]:
        return self.joints.copy()

    def set_gripper(self, position: int) -> int:
        self.gripper_position = int(position)
        return self.gripper_position

    def disable_torque(self) -> None:
        pass


class SimArmNode(ArmNodeBase):
    def __init__(self) -> None:
        super().__init__("sim_arm_node", backend=SimBackend(), mode="sim")

    def _available_ports(self) -> list[str]:
        return ["sim"]


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SimArmNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
