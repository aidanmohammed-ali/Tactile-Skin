from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

from .kinematics import ArmKinematics, dxl_to_rad, rad_to_dxl


@dataclass(frozen=True)
class DxlAddresses:
    op_mode: int = 11
    max_pos: int = 48
    min_pos: int = 52
    torque_enable: int = 64
    profile_accel: int = 108
    profile_velocity: int = 112
    goal_pos: int = 116
    present_pos: int = 132


class DynamixelDriver:
    protocol_version = 2.0
    baudrate = 1_000_000
    min_dxl_position = 0
    max_dxl_position = 4095

    def __init__(
        self,
        motor_ids: Sequence[int] = (11, 12, 13, 14, 15),
        default_speed: int = 100,
        default_accel: int = 20,
    ) -> None:
        if len(motor_ids) != 5:
            raise ValueError("motor_ids must contain four joint ids and one gripper id")
        self.motor_ids = tuple(int(v) for v in motor_ids)
        self.default_speed = int(default_speed)
        self.default_accel = int(default_accel)
        self.pos_limits = [(self.min_dxl_position, self.max_dxl_position)] * 5
        self.address = DxlAddresses()
        self.kinematics = ArmKinematics()
        self.port = None
        self.packet = None
        self.current_port = ""
        self.connected = False
        self.last_sent_dxl = [2048, 2048, 2048, 2048]
        self.io_lock = threading.RLock()
        self.io_lock_timeout_sec = 2.0

    def connect(self, port_name: str) -> None:
        with self._locked_io("connect"):
            if self.connected or self.port is not None:
                self._disconnect_unlocked()
            if not port_name:
                raise RuntimeError("serial port is empty")
            if not os.path.exists(port_name):
                raise RuntimeError(f"serial device does not exist: {port_name}")
            if not os.access(port_name, os.R_OK | os.W_OK):
                raise RuntimeError(f"no read/write permission for {port_name}")
            try:
                from dynamixel_sdk import PacketHandler, PortHandler
            except ImportError as exc:
                raise RuntimeError(
                    "Python package 'dynamixel-sdk' is required. Install it with: pip install dynamixel-sdk"
                ) from exc
            port = PortHandler(port_name)
            packet = PacketHandler(self.protocol_version)
            if not port.openPort():
                raise RuntimeError(f"could not open serial port {port_name}")
            if not port.setBaudRate(self.baudrate):
                port.closePort()
                raise RuntimeError(f"could not set baudrate to {self.baudrate}")
            self.port = port
            self.packet = packet
            self.current_port = port_name
            self.connected = True
            try:
                self._init_motors_unlocked()
            except Exception:
                try:
                    self._disconnect_unlocked()
                except Exception:
                    pass
                raise

    def disconnect(self) -> None:
        with self._locked_io("disconnect"):
            self._disconnect_unlocked()

    def _disconnect_unlocked(self) -> None:
        torque_error: Exception | None = None
        close_error: Exception | None = None
        try:
            if self.connected and self.port is not None and self.packet is not None:
                for dxl_id in self.motor_ids:
                    try:
                        self._write1(dxl_id, self.address.torque_enable, 0)
                    except Exception as exc:
                        if torque_error is None:
                            torque_error = exc
        finally:
            if self.port is not None:
                try:
                    self.port.closePort()
                except Exception as exc:
                    close_error = exc
            self.port = None
            self.packet = None
            self.current_port = ""
            self.connected = False
        if close_error is not None:
            raise RuntimeError(f"could not close Dynamixel port cleanly: {close_error}") from close_error
        if torque_error is not None:
            raise RuntimeError(f"port closed, but torque disable failed: {torque_error}") from torque_error

    def init_motors(self) -> None:
        with self._locked_io("initialize motors"):
            self._init_motors_unlocked()

    def _init_motors_unlocked(self) -> None:
        self._require_connection()
        for idx, dxl_id in enumerate(self.motor_ids):
            self._write1(dxl_id, self.address.torque_enable, 0)
            self._write1(dxl_id, self.address.op_mode, 4 if idx == 4 else 3)
            if idx < 4:
                min_pos, max_pos = self.pos_limits[idx]
                self._write4(dxl_id, self.address.min_pos, min_pos)
                self._write4(dxl_id, self.address.max_pos, max_pos)
            self._write4(dxl_id, self.address.profile_velocity, self.default_speed)
            self._write4(dxl_id, self.address.profile_accel, self.default_accel)
            self._write1(dxl_id, self.address.torque_enable, 1)

    def set_joint_angles(self, joints: Sequence[float]) -> None:
        with self._locked_io("set joint angles"):
            if len(joints) < 4:
                raise ValueError("joints must contain at least four angles")
            dxl_pos = rad_to_dxl(self.kinematics.remove_offsets(joints[:4]))
            self.last_sent_dxl = [self._clamp_int(pos, *limit) for pos, limit in zip(dxl_pos, self.pos_limits[:4])]
            if not self.connected:
                return
            for dxl_id, pos in zip(self.motor_ids[:4], self.last_sent_dxl):
                self._write4(dxl_id, self.address.goal_pos, pos)

    def read_joint_angles(self) -> list[float]:
        with self._locked_io("read joint angles"):
            if not self.connected:
                return self.kinematics.add_offsets(dxl_to_rad(self.last_sent_dxl))
            dxl_positions = [
                self._to_signed32(self._read4(dxl_id, self.address.present_pos))
                for dxl_id in self.motor_ids[:4]
            ]
            return self.kinematics.add_offsets(dxl_to_rad(dxl_positions))

    def set_gripper(self, position: int) -> int:
        with self._locked_io("set gripper"):
            position = self._clamp_int(position, self.min_dxl_position, self.max_dxl_position)
            if self.connected:
                self._write4(self.motor_ids[4], self.address.goal_pos, position)
            return position

    def disable_torque(self) -> None:
        with self._locked_io("disable torque"):
            if not self.connected:
                return
            for dxl_id in self.motor_ids:
                self._write1(dxl_id, self.address.torque_enable, 0)

    @contextmanager
    def _locked_io(self, operation: str) -> Iterator[None]:
        acquired = self.io_lock.acquire(timeout=self.io_lock_timeout_sec)
        if not acquired:
            port = self.current_port or "the Dynamixel port"
            raise RuntimeError(
                f"Dynamixel serial bus is busy while trying to {operation} on {port}. "
                "This usually means another arm operation is still using the SDK port; "
                "if it persists, check that old control_node or a second hardware_arm_node is not running."
            )
        try:
            yield
        finally:
            self.io_lock.release()

    def _write1(self, dxl_id: int, address: int, value: int) -> None:
        self._check_result(*self.packet.write1ByteTxRx(self.port, dxl_id, address, int(value)))

    def _write4(self, dxl_id: int, address: int, value: int) -> None:
        self._check_result(*self.packet.write4ByteTxRx(self.port, dxl_id, address, int(value)))

    def _read4(self, dxl_id: int, address: int) -> int:
        value, result, error = self.packet.read4ByteTxRx(self.port, dxl_id, address)
        self._check_result(result, error)
        return value

    def _check_result(self, result: int, error: int) -> None:
        if result != 0:
            message = self.packet.getTxRxResult(result)
            if "Port is in use" in message:
                message = (
                    f"{message}. Dynamixel SDK reported a busy serial transaction; "
                    "check that only one arm node owns the port and that hardware access is serialized."
                )
            raise RuntimeError(message)
        if error:
            raise RuntimeError(self.packet.getRxPacketError(error))

    def _require_connection(self) -> None:
        if not self.connected:
            raise RuntimeError("Dynamixel bus is not connected")

    @staticmethod
    def _to_signed32(value: int) -> int:
        return value - 2**32 if value >= 2**31 else value

    @staticmethod
    def _clamp_int(value: float, low: int, high: int) -> int:
        return max(low, min(high, round(float(value))))
