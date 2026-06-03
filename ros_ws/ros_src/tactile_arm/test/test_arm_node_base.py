from pathlib import Path
import os
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class CountingBackend:
    def __init__(self):
        self.connected = True
        self.current_port = "test"
        self.read_count = 0

    def connect(self, port_name):
        self.current_port = port_name
        self.connected = True

    def disconnect(self):
        self.connected = False

    def set_joint_angles(self, _joints):
        pass

    def read_joint_angles(self):
        self.read_count += 1
        return [0.0, 0.0, 0.0, 0.0]

    def set_gripper(self, position):
        return int(position)

    def disable_torque(self):
        pass


def test_publish_state_uses_cached_pose_while_busy():
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("tactile_interfaces")
    from tactile_arm.arm_node_base import ArmNodeBase

    initialized_here = not rclpy.ok()
    if initialized_here:
        os.environ.setdefault("ROS_LOG_DIR", "/tmp")
        rclpy.init()
    backend = CountingBackend()
    node = ArmNodeBase("test_arm_node_base_busy_state", backend=backend, mode="test")
    try:
        node.busy = True
        node._publish_state()
        assert backend.read_count == 0

        node.busy = False
        node._publish_state()
        assert backend.read_count == 1
    finally:
        node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()
