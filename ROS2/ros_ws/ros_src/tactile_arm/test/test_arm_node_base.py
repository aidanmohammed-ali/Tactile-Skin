from pathlib import Path
import os
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class CountingBackend:
    def __init__(self):
        self.connected = True
        self.current_port = "test"
        self.read_count = 0
        self.set_joint_angles_calls = []
        self.torque_disabled = False
        self.joint_speeds = []

    def connect(self, port_name):
        self.current_port = port_name
        self.connected = True

    def disconnect(self):
        self.connected = False

    def set_joint_angles(self, joints):
        self.set_joint_angles_calls.append(list(joints))

    def set_joint_speeds(self, speeds):
        self.joint_speeds = list(speeds)

    def read_joint_angles(self):
        self.read_count += 1
        return [0.0, 0.0, 0.0, 0.0]

    def set_gripper(self, position):
        return int(position)

    def disable_torque(self):
        self.torque_disabled = True


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


def test_publish_state_can_skip_idle_joint_polling():
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("tactile_interfaces")
    from tactile_arm.arm_node_base import ArmNodeBase

    initialized_here = not rclpy.ok()
    if initialized_here:
        os.environ.setdefault("ROS_LOG_DIR", "/tmp")
        rclpy.init()
    backend = CountingBackend()
    node = ArmNodeBase("test_arm_node_base_no_poll", backend=backend, mode="test")
    try:
        node.poll_joint_state = False
        node._publish_state()
        assert backend.read_count == 0

        node._refresh_pose_from_backend(force=True)
        assert backend.read_count == 1
    finally:
        node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()


def test_busy_arm_rejects_overlapping_command():
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("tactile_interfaces")
    from tactile_arm.arm_node_base import ArmNodeBase
    from tactile_interfaces.msg import ArmCommand

    initialized_here = not rclpy.ok()
    if initialized_here:
        os.environ.setdefault("ROS_LOG_DIR", "/tmp")
        rclpy.init()
    backend = CountingBackend()
    node = ArmNodeBase("test_arm_node_base_busy_reject", backend=backend, mode="test")
    try:
        node.busy = True
        node.active_command_id = "active"
        command = ArmCommand()
        command.id = "rejected"
        command.type = ArmCommand.TYPE_GRIPPER
        command.gripper_position = 1800

        node._on_arm_command(command)
        state = node._state_message()

        assert state.active_command_id == "active"
        assert state.completed_command_id == "rejected"
        assert state.completed_success is False
        assert "busy" in state.last_error
    finally:
        node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()


def test_stop_command_interrupts_active_move():
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("tactile_interfaces")
    from tactile_arm.arm_node_base import ArmNodeBase
    from tactile_interfaces.msg import ArmCommand

    class SlowBackend(CountingBackend):
        def __init__(self):
            super().__init__()
        def set_joint_angles(self, _joints):
            super().set_joint_angles(_joints)
            time.sleep(0.01)

    initialized_here = not rclpy.ok()
    if initialized_here:
        os.environ.setdefault("ROS_LOG_DIR", "/tmp")
        rclpy.init()
    backend = SlowBackend()
    node = ArmNodeBase("test_arm_node_base_stop_interrupt", backend=backend, mode="test")
    try:
        node.kinematics.inverse = lambda _pose: [0.0, 0.0, 0.0, 0.0]
        move = ArmCommand()
        move.id = "move-1"
        move.type = ArmCommand.TYPE_MOVE
        move.target_pose.x = 0.35
        move.target_pose.y = 0.0
        move.target_pose.z = 0.08
        move.target_pose.angle_rad = 0.0
        move.duration_sec = 1.0
        node._on_arm_command(move)

        deadline = time.monotonic() + 1.0
        while not node.busy and time.monotonic() < deadline:
            time.sleep(0.01)

        stop = ArmCommand()
        stop.id = "stop-1"
        stop.type = ArmCommand.TYPE_STOP
        node._on_arm_command(stop)

        worker = node.worker_thread
        if worker is not None:
            worker.join(timeout=1.0)
        state = node._state_message()

        assert backend.torque_disabled is False
        assert len(backend.set_joint_angles_calls) >= 2
        assert state.busy is False
        assert state.completed_command_id == "move-1"
        assert state.completed_success is False
        assert "stopped" in state.last_error or "stop" in state.last_error
    finally:
        node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()


def test_idle_stop_command_holds_position_without_disabling_torque():
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("tactile_interfaces")
    from tactile_arm.arm_node_base import ArmNodeBase
    from tactile_interfaces.msg import ArmCommand

    initialized_here = not rclpy.ok()
    if initialized_here:
        os.environ.setdefault("ROS_LOG_DIR", "/tmp")
        rclpy.init()
    backend = CountingBackend()
    node = ArmNodeBase("test_arm_node_base_idle_stop_hold", backend=backend, mode="test")
    try:
        stop = ArmCommand()
        stop.id = "stop-idle"
        stop.type = ArmCommand.TYPE_STOP
        node._on_arm_command(stop)

        worker = node.worker_thread
        if worker is not None:
            worker.join(timeout=1.0)
        state = node._state_message()

        assert backend.torque_disabled is False
        assert backend.set_joint_angles_calls[-1] == [0.0, 0.0, 0.0, 0.0]
        assert state.busy is False
        assert state.completed_command_id == "stop-idle"
        assert state.completed_success is True
    finally:
        node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()
