from __future__ import annotations

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .arm_node_base import ArmNodeBase
from .dynamixel_driver import DynamixelDriver


class HardwareArmNode(ArmNodeBase):
    def __init__(self) -> None:
        motor_ids = self._default_motor_ids()
        backend = DynamixelDriver(motor_ids=motor_ids)
        super().__init__("hardware_arm_node", backend=backend, mode="hardware")
        default_speed = int(self.declare_parameter("default_speed", backend.default_speed).value)
        default_accel = int(self.declare_parameter("default_accel", backend.default_accel).value)
        io_retry_count = int(self.declare_parameter("io_retry_count", backend.io_retry_count).value)
        io_retry_delay_sec = float(self.declare_parameter("io_retry_delay_sec", backend.io_retry_delay_sec).value)
        io_lock_timeout_sec = float(self.declare_parameter("io_lock_timeout_sec", backend.io_lock_timeout_sec).value)
        goal_write_ack = bool(self.declare_parameter("goal_write_ack", backend.goal_write_ack).value)
        backend.default_speed = default_speed
        backend.default_accel = default_accel
        backend.io_retry_count = max(0, io_retry_count)
        backend.io_retry_delay_sec = max(0.0, io_retry_delay_sec)
        backend.io_lock_timeout_sec = max(0.1, io_lock_timeout_sec)
        backend.goal_write_ack = goal_write_ack
        backend.motor_ids = tuple(int(v) for v in self.declare_parameter("motor_ids", list(motor_ids)).value)
        initial_port = str(self.declare_parameter("device_name", "").value)
        auto_connect = bool(self.declare_parameter("auto_connect", False).value)
        if auto_connect and initial_port:
            try:
                backend.connect(initial_port)
                self.get_logger().info(f"Connected to Dynamixel bus on {initial_port}")
            except Exception as exc:
                self.last_error = str(exc)
                self.get_logger().error(self.last_error)

    @staticmethod
    def _default_motor_ids() -> tuple[int, int, int, int, int]:
        return (11, 12, 13, 14, 15)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HardwareArmNode()
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
