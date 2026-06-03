from __future__ import annotations

import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Int32, String

from tactile_interfaces.msg import ArmMove, ArmPose, ArmState, BlockDetection


class PickPlaceNode(Node):
    def __init__(self) -> None:
        super().__init__("pick_place_node")
        self.grid_center_x = float(self.declare_parameter("grid_center_x", 8.0).value)
        self.grid_center_y = float(self.declare_parameter("grid_center_y", 2.0).value)
        self.grid_pitch_m = float(self.declare_parameter("grid_pitch_m", 0.025).value)
        self.pick_z = float(self.declare_parameter("pick_z", 0.02).value)
        self.carry_z = float(self.declare_parameter("carry_z", 0.10).value)
        self.retreat_z = float(self.declare_parameter("retreat_z", 0.12).value)
        self.pick_angle_rad = float(self.declare_parameter("pick_angle_rad", -1.0).value)
        self.approach_duration_sec = float(self.declare_parameter("approach_duration_sec", 0.8).value)
        self.descend_duration_sec = float(self.declare_parameter("descend_duration_sec", 0.5).value)
        self.lift_duration_sec = float(self.declare_parameter("lift_duration_sec", 0.6).value)
        self.transfer_duration_sec = float(self.declare_parameter("transfer_duration_sec", 0.9).value)
        self.retreat_duration_sec = float(self.declare_parameter("retreat_duration_sec", 0.6).value)
        self.gripper_wait_sec = float(self.declare_parameter("gripper_wait_sec", 0.4).value)
        self.detection_timeout_sec = float(self.declare_parameter("detection_timeout_sec", 3.0).value)
        self.move_settle_sec = float(self.declare_parameter("move_settle_sec", 0.15).value)
        self.gripper_open_position = int(self.declare_parameter("gripper_open_position", 1800).value)
        self.gripper_close_position = int(self.declare_parameter("gripper_close_position", 2400).value)

        self.status_pub = self.create_publisher(String, "/task/status", 10)
        self.detect_trigger_pub = self.create_publisher(Empty, "/vision/detect_trigger", 10)
        self.move_pub = self.create_publisher(ArmMove, "/arm/cartesian_goal", 10)
        self.gripper_pub = self.create_publisher(Int32, "/arm/gripper_position", 10)
        self.create_subscription(ArmPose, "/task/pick_place_goal", self._on_pick_place_goal, 10)
        self.create_subscription(BlockDetection, "/vision/block_detection", self._on_detection, 10)
        self.create_subscription(ArmState, "/arm/state", self._on_arm_state, 10)

        self.latest_detection: BlockDetection | None = None
        self.latest_arm_state: ArmState | None = None
        self.lock = threading.Lock()
        self.task_running = False

    def _on_detection(self, msg: BlockDetection) -> None:
        with self.lock:
            self.latest_detection = msg

    def _on_arm_state(self, msg: ArmState) -> None:
        with self.lock:
            self.latest_arm_state = msg

    def _on_pick_place_goal(self, place_pose: ArmPose) -> None:
        with self.lock:
            if self.task_running:
                self._publish_status("PickPlace ignored: task is already running")
                return
            self.task_running = True
        thread = threading.Thread(target=self._run_pick_place, args=(place_pose,), daemon=True)
        thread.start()

    def _run_pick_place(self, place_pose: ArmPose) -> None:
        try:
            self._require_arm_ready()
            self._publish_status("detecting")
            detection = self._detect_now()
            if not detection.detected:
                raise RuntimeError(detection.message or "no block detected")
            if not detection.grid_position_valid:
                raise RuntimeError("block detected, but grid position is invalid")

            pick_x, pick_y = self._grid_to_robot(detection.grid_column, detection.grid_row)
            above_pick = self._pose(pick_x, pick_y, self.carry_z, self.pick_angle_rad)
            pick_pose = self._pose(pick_x, pick_y, self.pick_z, self.pick_angle_rad)
            lift_pose = self._pose(pick_x, pick_y, self.retreat_z, self.pick_angle_rad)
            above_place = self._pose(place_pose.x, place_pose.y, max(place_pose.z, self.carry_z), place_pose.angle_rad)

            self._move("move_above_pick", above_pick, self.approach_duration_sec)
            self._gripper("gripper_open", self.gripper_open_position)
            self._move("descend_to_pick", pick_pose, self.descend_duration_sec)
            self._gripper("gripper_close", self.gripper_close_position)
            self._move("lift", lift_pose, self.lift_duration_sec)
            self._move("move_above_place", above_place, self.transfer_duration_sec)
            self._move("descend_to_place", place_pose, self.descend_duration_sec)
            self._gripper("release", self.gripper_open_position)
            self._move("retreat", above_place, self.retreat_duration_sec)
            self._publish_status("pick-place complete")
        except Exception as exc:
            self._publish_status(f"pick-place failed: {exc}")
        finally:
            with self.lock:
                self.task_running = False

    def _detect_now(self) -> BlockDetection:
        start_ns = self.get_clock().now().nanoseconds
        self.detect_trigger_pub.publish(Empty())
        deadline = time.monotonic() + self.detection_timeout_sec
        while time.monotonic() < deadline:
            with self.lock:
                detection = self.latest_detection
            if detection is not None and self._stamp_ns(detection) >= start_ns:
                return detection
            time.sleep(0.05)
        raise RuntimeError("vision detection timed out")

    def _require_arm_ready(self) -> None:
        with self.lock:
            state = self.latest_arm_state
        if state is None:
            raise RuntimeError("arm state is unavailable")
        if not state.connected:
            raise RuntimeError("arm is not connected")
        if state.busy:
            raise RuntimeError("arm is busy")
        if state.last_error:
            self._publish_status(f"arm warning before PickPlace: {state.last_error}")

    def _move(self, stage: str, pose: ArmPose, duration_sec: float) -> None:
        self._publish_status(f"{stage} start")
        started = time.monotonic()
        msg = ArmMove()
        msg.target_pose = pose
        msg.duration_sec = float(duration_sec)
        self.move_pub.publish(msg)
        time.sleep(max(0.0, duration_sec) + self.move_settle_sec)
        self._raise_arm_error(stage)
        self._publish_status(f"{stage} done elapsed={time.monotonic() - started:.2f}s")

    def _gripper(self, stage: str, position: int) -> None:
        self._publish_status(stage)
        msg = Int32()
        msg.data = int(position)
        self.gripper_pub.publish(msg)
        time.sleep(self.gripper_wait_sec)
        self._raise_arm_error(stage)

    def _raise_arm_error(self, stage: str) -> None:
        with self.lock:
            state = self.latest_arm_state
        if state is not None and state.last_error and state.last_error != "emergency stop requested":
            raise RuntimeError(f"{stage}: {state.last_error}")

    def _grid_to_robot(self, col: float, row: float) -> tuple[float, float]:
        x = (float(row) - self.grid_center_y) * self.grid_pitch_m
        y = -(float(col) - self.grid_center_x) * self.grid_pitch_m
        return x, y

    def _publish_status(self, message: str) -> None:
        msg = String()
        msg.data = str(message)
        self.status_pub.publish(msg)
        self.get_logger().info(msg.data)

    @staticmethod
    def _stamp_ns(msg: BlockDetection) -> int:
        return int(msg.stamp.sec) * 1_000_000_000 + int(msg.stamp.nanosec)

    @staticmethod
    def _pose(x: float, y: float, z: float, angle_rad: float) -> ArmPose:
        pose = ArmPose()
        pose.x = float(x)
        pose.y = float(y)
        pose.z = float(z)
        pose.angle_rad = float(angle_rad)
        return pose


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PickPlaceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
