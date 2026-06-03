from __future__ import annotations

import time
from typing import Any, Protocol, Sequence

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Empty, Int32

from tactile_interfaces.msg import ArmMove
from tactile_interfaces.msg import ArmPose as ArmPoseMsg
from tactile_interfaces.msg import ArmState
from tactile_interfaces.srv import ListArmPorts, SetArmConnection

from .kinematics import ArmKinematics, ArmPose, ease_quintic, interpolate_pose
from .ports import list_serial_ports, recommended_port


class ArmBackend(Protocol):
    connected: bool
    current_port: str

    def connect(self, port_name: str) -> None: ...

    def disconnect(self) -> None: ...

    def set_joint_angles(self, joints: Sequence[float]) -> None: ...

    def read_joint_angles(self) -> list[float]: ...

    def set_gripper(self, position: int) -> int: ...

    def disable_torque(self) -> None: ...


class ArmNodeBase(Node):
    def __init__(self, node_name: str, backend: ArmBackend, mode: str) -> None:
        super().__init__(node_name)
        self.callback_group = ReentrantCallbackGroup()
        self.backend = backend
        self.mode = mode
        self.kinematics = ArmKinematics()
        self.busy = False
        self.stop_requested = False
        self.last_error = ""
        self.gripper_open_position = int(self.declare_parameter("gripper_open_position", 1800).value)
        self.gripper_close_position = int(self.declare_parameter("gripper_close_position", 2400).value)
        state_rate_hz = float(self.declare_parameter("state_rate_hz", 5.0).value)
        self.interpolation_step_sec = float(self.declare_parameter("interpolation_step_sec", 0.10).value)
        self.default_move_duration_sec = float(self.declare_parameter("default_move_duration_sec", 1.2).value)
        self.current_pose = self.kinematics.forward([0.0, 0.0, 0.0, 0.0])
        self.current_joints = [0.0, 0.0, 0.0, 0.0]
        self.current_gripper_position = self.gripper_open_position

        self.state_pub = self.create_publisher(ArmState, "/arm/state", 10)
        self.move_sub = self.create_subscription(
            ArmMove,
            "/arm/cartesian_goal",
            self._on_cartesian_goal,
            10,
            callback_group=self.callback_group,
        )
        self.gripper_sub = self.create_subscription(
            Int32,
            "/arm/gripper_position",
            self._on_gripper_position,
            10,
            callback_group=self.callback_group,
        )
        self.emergency_sub = self.create_subscription(
            Empty,
            "/arm/emergency_stop",
            self._on_emergency_stop,
            10,
            callback_group=self.callback_group,
        )
        self.list_ports_service = self.create_service(
            ListArmPorts,
            "/arm/list_ports",
            self._list_ports,
            callback_group=self.callback_group,
        )
        self.connection_service = self.create_service(
            SetArmConnection,
            "/arm/set_connection",
            self._set_connection,
            callback_group=self.callback_group,
        )
        self.state_timer = self.create_timer(
            1.0 / max(0.1, state_rate_hz),
            self._publish_state,
            callback_group=self.callback_group,
        )

    def _list_ports(self, _request: ListArmPorts.Request, response: ListArmPorts.Response) -> Any:
        ports = self._available_ports()
        response.ports = ports
        response.recommended_port = recommended_port(ports)
        return response

    def _available_ports(self) -> list[str]:
        return list_serial_ports()

    def _set_connection(self, request: SetArmConnection.Request, response: SetArmConnection.Response) -> Any:
        try:
            if request.connect:
                self.backend.connect(request.port)
                response.message = "connected"
            else:
                self.stop_requested = True
                self.busy = False
                self.backend.disconnect()
                response.message = "disconnected"
            response.success = True
            response.current_port = self.backend.current_port
            self.last_error = ""
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            response.current_port = self.backend.current_port
            self.last_error = response.message
        self._refresh_pose_from_backend()
        self._publish_state()
        return response

    def _on_gripper_position(self, msg: Int32) -> None:
        try:
            self.current_gripper_position = self.backend.set_gripper(int(msg.data))
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)
        self._publish_state()

    def _on_emergency_stop(self, _msg: Empty) -> None:
        self.stop_requested = True
        try:
            self.backend.disable_torque()
            self.last_error = "emergency stop requested"
        except Exception as exc:
            self.last_error = f"emergency stop failed: {exc}"
        self.busy = False
        self._publish_state()

    def _on_cartesian_goal(self, msg: ArmMove) -> None:
        if not self.backend.connected:
            self.last_error = "arm is not connected"
            self._publish_state()
            return
        if self.busy:
            self.last_error = "arm is busy; cartesian goal ignored"
            self._publish_state()
            return

        self.busy = True
        self.stop_requested = False
        target = self._pose_from_msg(msg.target_pose)
        duration = max(0.2, float(msg.duration_sec or self.default_move_duration_sec))
        start_pose = self.current_pose
        step_sec = max(0.02, float(self.interpolation_step_sec))
        steps = max(2, int(duration / step_sec))
        started = time.monotonic()

        try:
            for index in range(steps + 1):
                if self.stop_requested:
                    raise RuntimeError("move stopped by emergency stop")
                t = ease_quintic(index / steps)
                pose = interpolate_pose(start_pose, target, t)
                joints = self.kinematics.inverse(pose)
                self.backend.set_joint_angles(joints)
                self.current_joints = list(joints)
                self.current_pose = pose
                sleep_until = started + duration * index / steps
                delay = sleep_until - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            self.busy = False
            self._publish_state()

    def _refresh_pose_from_backend(self) -> None:
        if self.busy:
            return
        try:
            joints = self.backend.read_joint_angles()
        except Exception:
            return
        if len(joints) >= 4:
            self.current_joints = list(joints[:4])
            self.current_pose = self.kinematics.forward(self.current_joints)

    def _publish_state(self) -> None:
        self._refresh_pose_from_backend()
        self.state_pub.publish(self._state_message())

    def _state_message(self) -> ArmState:
        msg = ArmState()
        msg.stamp = self.get_clock().now().to_msg()
        msg.connected = bool(self.backend.connected)
        msg.busy = bool(self.busy)
        msg.mode = self.mode
        msg.current_port = self.backend.current_port
        msg.current_pose = self._pose_msg(self.current_pose)
        msg.joint_positions = list(float(v) for v in self.current_joints)
        msg.gripper_position = int(self.current_gripper_position)
        msg.last_error = self.last_error
        return msg

    def destroy_node(self) -> bool:
        try:
            self.stop_requested = True
            self.busy = False
            self.backend.disconnect()
        except Exception as exc:
            self.get_logger().warn(f"Could not disconnect arm backend during shutdown: {exc}")
        return super().destroy_node()

    @staticmethod
    def _pose_from_msg(msg: ArmPoseMsg) -> ArmPose:
        return ArmPose(x=float(msg.x), y=float(msg.y), z=float(msg.z), angle_rad=float(msg.angle_rad))

    @staticmethod
    def _pose_msg(pose: ArmPose) -> ArmPoseMsg:
        msg = ArmPoseMsg()
        msg.x = float(pose.x)
        msg.y = float(pose.y)
        msg.z = float(pose.z)
        msg.angle_rad = float(pose.angle_rad)
        return msg
