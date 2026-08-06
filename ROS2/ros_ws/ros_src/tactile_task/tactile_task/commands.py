from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from importlib import import_module
from typing import Any

from std_msgs.msg import Empty

from tactile_interfaces.msg import ArmCommand, ArmPose, ArmState, BlockDetection
from tactile_interfaces.srv import GetTactilePose


class CancelledError(RuntimeError):
    """Raised inside task scripts when the task runner requests cancellation."""


class TaskStateCache:
    """Thread-safe latest-value cache shared by ROS callbacks and task workers."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._arm_state: ArmState | None = None
        self._arm_version = 0
        self._detection: BlockDetection | None = None
        self._detection_version = 0

    def set_arm_state(self, msg: ArmState) -> None:
        with self._condition:
            self._arm_state = msg
            self._arm_version += 1
            self._condition.notify_all()

    def set_detection(self, msg: BlockDetection) -> None:
        with self._condition:
            self._detection = msg
            self._detection_version += 1
            self._condition.notify_all()

    def arm_state(self) -> ArmState | None:
        with self._condition:
            return self._arm_state

    def wait_for_arm(
        self,
        predicate: Callable[[ArmState, int], bool],
        timeout_sec: float,
    ) -> ArmState | None:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        with self._condition:
            while True:
                if self._arm_state is not None and predicate(self._arm_state, self._arm_version):
                    return self._arm_state
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def detection_version(self) -> int:
        with self._condition:
            return self._detection_version

    def wait_for_detection(
        self,
        predicate: Callable[[BlockDetection, int], bool],
        timeout_sec: float,
    ) -> BlockDetection | None:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        with self._condition:
            while True:
                if self._detection is not None and predicate(self._detection, self._detection_version):
                    return self._detection
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)


class TaskLogger:
    def __init__(
        self,
        state_callback: Callable[[str, str, bool], None] | None = None,
        ros_logger: Any | None = None,
    ) -> None:
        self.state_callback = state_callback
        self.ros_logger = ros_logger

    def stage(self, message: str) -> None:
        self.info(message)

    def info(self, message: str) -> None:
        text = str(message)
        if self.state_callback is not None:
            self.state_callback(text, text, True)
        if self.ros_logger is not None:
            self.ros_logger.info(text)

    def error(self, message: str) -> None:
        text = str(message)
        if self.state_callback is not None:
            self.state_callback("", text, False)
        if self.ros_logger is not None:
            self.ros_logger.error(text)


class Robot:
    def __init__(
        self,
        arm_command_pub: Any,
        detect_trigger_pub: Any,
        cache: TaskStateCache,
        clock: Any,
        logger: TaskLogger | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        tactile_pose_request: Callable[[float], GetTactilePose.Response] | None = None,
    ) -> None:
        self.arm_command_pub = arm_command_pub
        self.detect_trigger_pub = detect_trigger_pub
        self.cache = cache
        self.clock = clock
        self.logger = logger or TaskLogger()
        self.is_cancelled = is_cancelled or (lambda: False)
        self.tactile_pose_request = tactile_pose_request

    def wait_ready(self, timeout_sec: float = 5.0) -> ArmState:
        timeout = float(timeout_sec)
        state = self.cache.wait_for_arm(
            lambda arm_state, _version: (arm_state.connected and not arm_state.busy) or not arm_state.connected,
            timeout,
        )
        if state is None:
            raise RuntimeError(f"arm did not become ready within {timeout:.2f}s")
        if not state.connected:
            raise RuntimeError("arm is not connected")
        return state

    def move(
        self,
        x: float,
        y: float,
        z: float,
        angle_rad: float,
        duration: float,
        *,
        stage: str | None = None,
        timeout_sec: float | None = None,
        timeout_margin_sec: float = 2.0,
        settle_sec: float = 0.0,
        angle_retry_step_rad: float = 0.01,
        angle_retry_max_rad: float = -1.0,
    ) -> ArmState:
        self.check_cancelled()
        if stage:
            self.log(stage)
        target_angle = self._reachable_angle_or_original(
            x,
            y,
            z,
            angle_rad,
            angle_retry_step_rad,
            angle_retry_max_rad,
            stage or "move",
        )
        command = ArmCommand()
        command.id = self._new_command_id()
        command.type = ArmCommand.TYPE_MOVE
        command.target_pose = pose(x, y, z, target_angle)
        command.duration_sec = max(0.0, float(duration))
        timeout = (
            float(timeout_sec)
            if timeout_sec is not None
            else command.duration_sec + float(timeout_margin_sec)
        )
        state = self._send_and_wait(command, timeout, stage or "move")
        self._settle(settle_sec)
        return state

    def gripper(
        self,
        position: int,
        *,
        stage: str | None = None,
        timeout_sec: float = 2.0,
        settle_sec: float = 0.0,
    ) -> ArmState:
        self.check_cancelled()
        if stage:
            self.log(stage)
        command = ArmCommand()
        command.id = self._new_command_id()
        command.type = ArmCommand.TYPE_GRIPPER
        command.gripper_position = int(position)
        state = self._send_and_wait(command, float(timeout_sec), stage or "gripper")
        self._settle(settle_sec)
        return state

    def open_gripper(
        self,
        position: int,
        *,
        stage: str = "gripper_open",
        timeout_sec: float = 2.0,
        settle_sec: float = 0.0,
    ) -> ArmState:
        return self.gripper(position, stage=stage, timeout_sec=timeout_sec, settle_sec=settle_sec)

    def close_gripper(
        self,
        position: int,
        *,
        stage: str = "gripper_close",
        timeout_sec: float = 2.0,
        settle_sec: float = 0.0,
    ) -> ArmState:
        return self.gripper(position, stage=stage, timeout_sec=timeout_sec, settle_sec=settle_sec)

    def stop(self, *, wait: bool = True, timeout_sec: float = 2.0) -> None:
        command = ArmCommand()
        command.id = self._new_command_id()
        command.type = ArmCommand.TYPE_STOP
        self.arm_command_pub.publish(command)
        if not wait:
            return
        state = self.cache.wait_for_arm(lambda arm_state, _version: not arm_state.busy, timeout_sec)
        if state is None:
            raise RuntimeError(f"arm did not stop within {timeout_sec:.2f}s")

    def detect_block(self, timeout: float = 3.0) -> BlockDetection:
        self.check_cancelled()
        timeout_sec = float(timeout)
        deadline = time.monotonic() + max(0.0, timeout_sec)
        start_ns = self.clock.now().nanoseconds
        version_before = self.cache.detection_version()
        self.log("detecting")
        self.detect_trigger_pub.publish(Empty())
        while True:
            self.check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("vision detection timed out")
            detection = self.cache.wait_for_detection(
                lambda msg, version: version > version_before and _stamp_ns(msg) >= start_ns,
                min(0.1, remaining),
            )
            if detection is not None:
                return detection

    def detect_tactile_pose(
        self,
        timeout: float = 3.0,
    ) -> GetTactilePose.Response:
        self.check_cancelled()
        if self.tactile_pose_request is None:
            raise RuntimeError("tactile pose service client is not configured")
        self.log("detecting_tactile_pose")
        response = self.tactile_pose_request(float(timeout))
        self.check_cancelled()
        if not response.success:
            raise RuntimeError(response.message or "tactile pose prediction failed")
        return response

    def log(self, stage: str) -> None:
        self.logger.stage(str(stage))

    def check_cancelled(self) -> None:
        if self.is_cancelled():
            raise CancelledError("task cancelled")

    def _settle(self, duration_sec: float) -> None:
        deadline = time.monotonic() + max(0.0, float(duration_sec))
        while True:
            self.check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 1e-6:
                return
            time.sleep(min(remaining, 0.05))

    def _send_and_wait(self, command: ArmCommand, timeout_sec: float, stage: str) -> ArmState:
        self.wait_ready()
        self.arm_command_pub.publish(command)
        state = self.cache.wait_for_arm(
            lambda arm_state, _version: arm_state.completed_command_id == command.id or not arm_state.connected,
            max(0.0, float(timeout_sec)),
        )
        if state is None:
            raise RuntimeError(f"{stage}: arm command timed out after {timeout_sec:.2f}s")
        if not state.connected:
            raise RuntimeError(f"{stage}: arm is not connected")
        if state.completed_command_id != command.id:
            raise RuntimeError(f"{stage}: arm did not report completion for command {command.id}")
        if not state.completed_success:
            raise RuntimeError(f"{stage}: {state.last_error or 'arm command failed'}")
        self.check_cancelled()
        return state

    def _reachable_angle_or_original(
        self,
        x: float,
        y: float,
        z: float,
        angle_rad: float,
        step_rad: float,
        max_angle_rad: float,
        stage: str,
    ) -> float:
        kinematics = _load_arm_kinematics()
        if kinematics is None:
            return float(angle_rad)

        step = abs(float(step_rad))
        target_angle = float(angle_rad)
        max_angle = float(max_angle_rad)
        attempt = 0
        while target_angle <= max_angle + 1e-9:
            self.check_cancelled()
            try:
                kinematics.inverse(_arm_pose(x, y, z, target_angle))
                if attempt > 0:
                    self.logger.info(
                        f"{stage}: adjusted angle_rad from "
                        f"{float(angle_rad):.3f} to {target_angle:.3f}"
                    )
                return target_angle
            except ValueError:
                target_angle += step
                attempt += 1
        raise RuntimeError(
            f"{stage}: no reachable angle found from "
            f"{float(angle_rad):.3f} to {max_angle:.3f}"
        )

    @staticmethod
    def _new_command_id() -> str:
        return uuid.uuid4().hex


RobotCommands = Robot


def pose(x: float, y: float, z: float, angle_rad: float) -> ArmPose:
    msg = ArmPose()
    msg.x = float(x)
    msg.y = float(y)
    msg.z = float(z)
    msg.angle_rad = float(angle_rad)
    return msg


def _stamp_ns(msg: BlockDetection) -> int:
    return int(msg.stamp.sec) * 1_000_000_000 + int(msg.stamp.nanosec)


def _load_arm_kinematics() -> Any | None:
    try:
        module = import_module("tactile_arm.kinematics")
    except Exception:
        return None
    return module.ArmKinematics()


def _arm_pose(x: float, y: float, z: float, angle_rad: float) -> Any:
    module = import_module("tactile_arm.kinematics")
    return module.ArmPose(
        x=float(x),
        y=float(y),
        z=float(z),
        angle_rad=float(angle_rad),
    )
