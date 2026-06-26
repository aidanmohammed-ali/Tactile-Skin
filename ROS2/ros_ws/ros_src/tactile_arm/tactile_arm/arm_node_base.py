from __future__ import annotations

import threading
import time
from typing import Any, Protocol, Sequence

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from tactile_interfaces.msg import ArmCommand
from tactile_interfaces.msg import ArmPose as ArmPoseMsg
from tactile_interfaces.msg import ArmState
from tactile_interfaces.srv import ListArmPorts, SetArmConnection

from .kinematics import ArmKinematics, ArmPose, synchronized_joint_speeds
from .ports import list_serial_ports, recommended_port


class ArmBackend(Protocol):
    connected: bool
    current_port: str

    def connect(self, port_name: str) -> None: ...

    def disconnect(self) -> None: ...

    def set_joint_angles(self, joints: Sequence[float]) -> None: ...

    def set_joint_speeds(self, speeds: Sequence[int]) -> None: ...

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
        self.command_lock = threading.RLock()
        self.worker_thread: threading.Thread | None = None
        self.busy = False
        self.stop_requested = False
        self.last_error = ""
        self.active_command_id = ""
        self.completed_command_id = ""
        self.completed_success = True
        self.gripper_open_position = int(self.declare_parameter("gripper_open_position", 1800).value)
        self.gripper_close_position = int(self.declare_parameter("gripper_close_position", 2400).value)
        state_rate_hz = float(self.declare_parameter("state_rate_hz", 5.0).value)
        self.poll_joint_state = bool(self.declare_parameter("poll_joint_state", True).value)
        self.maximum_joint_speed = int(self.declare_parameter("maximum_joint_speed", 100).value)
        self.minimum_joint_speed = int(self.declare_parameter("minimum_joint_speed", 10).value)
        self.default_move_duration_sec = float(self.declare_parameter("default_move_duration_sec", 1.2).value)
        self.current_pose = self.kinematics.forward([0.0, 0.0, 0.0, 0.0])
        self.current_joints = [0.0, 0.0, 0.0, 0.0]
        self.current_gripper_position = self.gripper_open_position

        self.state_pub = self.create_publisher(ArmState, "/arm/state", 10)
        self.command_sub = self.create_subscription(
            ArmCommand,
            "/arm/command",
            self._on_arm_command,
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
                with self.command_lock:
                    self.stop_requested = True
                    self.busy = False
                    self.active_command_id = ""
                self.backend.disconnect()
                response.message = "disconnected"
            response.success = True
            response.current_port = self.backend.current_port
            with self.command_lock:
                self.last_error = ""
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            response.current_port = self.backend.current_port
            with self.command_lock:
                self.last_error = response.message
        self._refresh_pose_from_backend(force=True)
        self._publish_state()
        return response

    def _on_arm_command(self, msg: ArmCommand) -> None:
        command_id = str(msg.id).strip() or f"command-{time.monotonic_ns()}"
        command_type = str(msg.type).strip().lower()
        if command_type == ArmCommand.TYPE_STOP:
            self._request_stop(command_id)
            return

        if command_type not in (ArmCommand.TYPE_MOVE, ArmCommand.TYPE_GRIPPER):
            self._reject_command(command_id, f"unknown arm command type: {command_type}")
            return

        with self.command_lock:
            if not self.backend.connected:
                reject = "arm is not connected"
            elif self.busy:
                reject = "arm is busy; command ignored"
            else:
                reject = ""
                self.busy = True
                self.stop_requested = False
                self.active_command_id = command_id
                self.completed_command_id = ""
                self.completed_success = False
                self.last_error = ""
        if reject:
            self._reject_command(command_id, reject)
            return

        self._publish_state()
        worker = threading.Thread(
            target=self._run_command,
            args=(command_id, command_type, msg),
            daemon=True,
        )
        with self.command_lock:
            self.worker_thread = worker
        worker.start()

    def _request_stop(self, command_id: str) -> None:
        start_stop_worker = False
        with self.command_lock:
            self.stop_requested = True
            if self.busy:
                self.last_error = "stop requested"
            else:
                self.busy = True
                self.active_command_id = command_id
                self.completed_command_id = ""
                self.completed_success = False
                self.last_error = "stop requested"
                start_stop_worker = True
        self._publish_state()
        if start_stop_worker:
            worker = threading.Thread(target=self._run_stop_command, args=(command_id,), daemon=True)
            with self.command_lock:
                self.worker_thread = worker
            worker.start()

    def _reject_command(self, command_id: str, message: str) -> None:
        with self.command_lock:
            self.last_error = message
            self.completed_command_id = command_id
            self.completed_success = False
        self._publish_state()

    def _run_command(self, command_id: str, command_type: str, msg: ArmCommand) -> None:
        success = False
        message = ""
        try:
            if command_type == ArmCommand.TYPE_MOVE:
                self._execute_move(msg)
            elif command_type == ArmCommand.TYPE_GRIPPER:
                position = self.backend.set_gripper(int(msg.gripper_position))
                with self.command_lock:
                    self.current_gripper_position = int(position)
            success = True
        except Exception as exc:
            message = str(exc)
        self._finish_command(command_id, success, message)

    def _run_stop_command(self, command_id: str) -> None:
        success = False
        message = ""
        try:
            self._hold_current_position_after_stop()
            success = True
            message = "stop requested"
        except Exception as exc:
            message = f"stop failed: {exc}"
        self._finish_command(command_id, success, message)

    def _execute_move(self, msg: ArmCommand) -> None:
        target = self._pose_from_msg(msg.target_pose)
        duration = max(0.2, float(msg.duration_sec or self.default_move_duration_sec))
        start_joints = self.backend.read_joint_angles()
        if len(start_joints) < 4:
            raise RuntimeError("arm backend returned fewer than four joint angles")
        start_joints = list(start_joints[:4])
        with self.command_lock:
            self.current_joints = start_joints
            self.current_pose = self.kinematics.forward(start_joints)
        target_joints = self.kinematics.inverse(target)
        speeds = synchronized_joint_speeds(
            start_joints,
            target_joints,
            self.maximum_joint_speed,
            self.minimum_joint_speed,
        )
        self.backend.set_joint_speeds(speeds)
        self.backend.set_joint_angles(target_joints)
        self._sleep_until(time.monotonic() + duration)
        if self._stop_is_requested():
            self._hold_current_position_after_stop()
            raise RuntimeError("move stopped by stop command")
        with self.command_lock:
            self.current_joints = list(target_joints)
            self.current_pose = target

    def _finish_command(self, command_id: str, success: bool, message: str) -> None:
        with self.command_lock:
            self.busy = False
            self.stop_requested = False
            self.active_command_id = ""
            self.completed_command_id = command_id
            self.completed_success = bool(success)
            if success and message == "stop requested":
                self.last_error = message
            elif success:
                self.last_error = ""
            else:
                self.last_error = message
        self._publish_state()

    def _stop_is_requested(self) -> bool:
        with self.command_lock:
            return bool(self.stop_requested)

    def _hold_current_position_after_stop(self) -> None:
        try:
            hold_joints = self.backend.read_joint_angles()
            if len(hold_joints) < 4:
                raise RuntimeError("arm backend returned fewer than four joint angles")
            hold_joints = list(hold_joints[:4])
            self.backend.set_joint_angles(hold_joints)
            with self.command_lock:
                self.current_joints = hold_joints
                self.current_pose = self.kinematics.forward(hold_joints)
        except Exception as exc:
            raise RuntimeError(f"stop requested, but hold position failed: {exc}") from exc

    def _sleep_until(self, deadline: float) -> None:
        while True:
            if self._stop_is_requested():
                return
            delay = deadline - time.monotonic()
            if delay <= 0:
                return
            time.sleep(min(delay, 0.02))

    def _refresh_pose_from_backend(self, *, force: bool = False) -> None:
        with self.command_lock:
            if self.busy:
                return
            should_poll = force or self.poll_joint_state
        if not should_poll:
            return
        try:
            joints = self.backend.read_joint_angles()
        except Exception:
            return
        if len(joints) >= 4:
            with self.command_lock:
                if self.busy:
                    return
                self.current_joints = list(joints[:4])
                self.current_pose = self.kinematics.forward(self.current_joints)

    def _publish_state(self) -> None:
        self._refresh_pose_from_backend()
        self.state_pub.publish(self._state_message())

    def _state_message(self) -> ArmState:
        msg = ArmState()
        msg.stamp = self.get_clock().now().to_msg()
        with self.command_lock:
            msg.connected = bool(self.backend.connected)
            msg.busy = bool(self.busy)
            msg.mode = self.mode
            msg.current_port = self.backend.current_port
            msg.current_pose = self._pose_msg(self.current_pose)
            msg.joint_positions = list(float(v) for v in self.current_joints)
            msg.gripper_position = int(self.current_gripper_position)
            msg.last_error = self.last_error
            msg.active_command_id = self.active_command_id
            msg.completed_command_id = self.completed_command_id
            msg.completed_success = bool(self.completed_success)
        return msg

    def destroy_node(self) -> bool:
        with self.command_lock:
            self.stop_requested = True
        worker = self.worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.0)
        try:
            with self.command_lock:
                self.busy = False
                self.active_command_id = ""
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
