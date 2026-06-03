from __future__ import annotations

from typing import Any

import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from tactile_interfaces.action import MoveCartesian, PickPlace
from tactile_interfaces.msg import ArmPose, BlockDetection
from tactile_interfaces.srv import DetectBlock, EmergencyStop, SetGripper


class PickPlaceActionServer(Node):
    def __init__(self) -> None:
        super().__init__("pick_place_action_server")
        self.callback_group = ReentrantCallbackGroup()
        self.grid_center_x = float(self.declare_parameter("grid_center_x", 8.0).value)
        self.grid_center_y = float(self.declare_parameter("grid_center_y", 2.0).value)
        self.grid_pitch_m = float(self.declare_parameter("grid_pitch_m", 0.025).value)
        self.pick_z = float(self.declare_parameter("pick_z", 0.02).value)
        self.carry_z = float(self.declare_parameter("carry_z", 0.10).value)
        self.retreat_z = float(self.declare_parameter("retreat_z", 0.12).value)
        self.pick_angle_rad = float(self.declare_parameter("pick_angle_rad", -1.0).value)
        self.move_duration_sec = float(self.declare_parameter("move_duration_sec", 1.2).value)
        self.detect_client = self.create_client(
            DetectBlock,
            "/vision/detect_block",
            callback_group=self.callback_group,
        )
        self.gripper_client = self.create_client(
            SetGripper,
            "/arm/set_gripper",
            callback_group=self.callback_group,
        )
        self.emergency_client = self.create_client(
            EmergencyStop,
            "/arm/emergency_stop",
            callback_group=self.callback_group,
        )
        self.move_client = ActionClient(
            self,
            MoveCartesian,
            "/arm/move_cartesian",
            callback_group=self.callback_group,
        )
        self.server = ActionServer(
            self,
            PickPlace,
            "/task/pick_place",
            execute_callback=self._execute_pick_place,
            callback_group=self.callback_group,
        )

    async def _execute_pick_place(self, goal_handle: Any) -> PickPlace.Result:
        result = PickPlace.Result()
        try:
            self._feedback(goal_handle, "detecting", 0.05)
            detection = await self._detect(bool(goal_handle.request.recalibrate_before_pick))
            result.detection = detection
            if not detection.detected:
                result.success = False
                result.message = detection.message or "no block detected"
                goal_handle.abort()
                return result
            if not detection.grid_position_valid:
                result.success = False
                result.message = "block detected, but grid position is invalid"
                goal_handle.abort()
                return result

            pick_x, pick_y = self._grid_to_robot(detection.grid_column, detection.grid_row)
            above_pick = self._pose(pick_x, pick_y, self.carry_z, self.pick_angle_rad)
            pick_pose = self._pose(pick_x, pick_y, self.pick_z, self.pick_angle_rad)
            lift_pose = self._pose(pick_x, pick_y, self.retreat_z, self.pick_angle_rad)
            place_pose = goal_handle.request.place_pose
            above_place = self._pose(place_pose.x, place_pose.y, max(place_pose.z, self.carry_z), place_pose.angle_rad)

            await self._move(goal_handle, "move_above_pick", 0.15, above_pick)
            await self._gripper(SetGripper.Request.OPEN)
            self._feedback(goal_handle, "gripper_open", 0.25, detection)
            await self._move(goal_handle, "descend_to_pick", 0.35, pick_pose)
            await self._gripper(SetGripper.Request.CLOSE)
            self._feedback(goal_handle, "gripper_close", 0.50, detection)
            await self._move(goal_handle, "lift", 0.62, lift_pose)
            await self._move(goal_handle, "move_above_place", 0.76, above_place)
            await self._move(goal_handle, "descend_to_place", 0.88, place_pose)
            await self._gripper(SetGripper.Request.OPEN)
            self._feedback(goal_handle, "release", 0.94, detection)
            await self._move(goal_handle, "retreat", 0.98, above_place)

            result.success = True
            result.message = "pick-place complete"
            goal_handle.succeed()
            return result
        except Exception as exc:
            await self._best_effort_emergency_stop()
            result.success = False
            result.message = str(exc)
            goal_handle.abort()
            return result

    async def _detect(self, recalibrate: bool) -> BlockDetection:
        if not self.detect_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("vision detect service is unavailable")
        request = DetectBlock.Request()
        request.recalibrate = recalibrate
        request.include_annotated_frame = False
        response = await self.detect_client.call_async(request)
        if not response.success:
            detection = response.detection
            detection.message = response.message
            return detection
        return response.detection

    async def _move(self, goal_handle: Any, stage: str, progress: float, pose: ArmPose) -> None:
        if goal_handle.is_cancel_requested:
            raise RuntimeError("pick-place cancelled")
        if not self.move_client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("arm move action is unavailable")
        self._feedback(goal_handle, stage, progress)
        move_goal = MoveCartesian.Goal()
        move_goal.target_pose = pose
        move_goal.duration_sec = self.move_duration_sec
        goal_response = await self.move_client.send_goal_async(move_goal)
        if not goal_response.accepted:
            raise RuntimeError(f"arm rejected move stage: {stage}")
        result_response = await goal_response.get_result_async()
        move_result = result_response.result
        if not move_result.success:
            raise RuntimeError(f"{stage} failed: {move_result.message}")

    async def _gripper(self, command: int) -> None:
        if not self.gripper_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("arm gripper service is unavailable")
        request = SetGripper.Request()
        request.command = int(command)
        request.position = 0
        response = await self.gripper_client.call_async(request)
        if not response.success:
            raise RuntimeError(response.message)

    async def _best_effort_emergency_stop(self) -> None:
        if not self.emergency_client.wait_for_service(timeout_sec=0.1):
            return
        try:
            await self.emergency_client.call_async(EmergencyStop.Request())
        except Exception:
            pass

    def _feedback(
        self,
        goal_handle: Any,
        stage: str,
        progress: float,
        detection: BlockDetection | None = None,
    ) -> None:
        feedback = PickPlace.Feedback()
        feedback.stage = stage
        feedback.progress = float(progress)
        if detection is not None:
            feedback.detection = detection
        goal_handle.publish_feedback(feedback)

    def _grid_to_robot(self, col: float, row: float) -> tuple[float, float]:
        x = (float(row) - self.grid_center_y) * self.grid_pitch_m
        y = -(float(col) - self.grid_center_x) * self.grid_pitch_m
        return x, y

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
    node = PickPlaceActionServer()
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
