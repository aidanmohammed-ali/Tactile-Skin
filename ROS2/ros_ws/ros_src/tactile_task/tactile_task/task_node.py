from __future__ import annotations

import importlib
import json
import re
import threading
import time
from types import ModuleType
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

from tactile_interfaces.msg import ArmCommand, ArmState, BlockDetection, TaskCommand, TaskState
from tactile_interfaces.srv import GetTactilePose

from .commands import CancelledError, Robot, TaskLogger, TaskStateCache


_TASK_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class TaskNode(Node):
    def __init__(self) -> None:
        super().__init__("task_node")
        self.task_package = str(self.declare_parameter("task_package", "tactile_task.tasks").value)
        self.default_task = str(self.declare_parameter("default_task", "").value).strip()
        self.auto_start = bool(self.declare_parameter("auto_start", False).value)
        self.tactile_pose_service_name = str(
            self.declare_parameter(
                "tactile_pose_service",
                "/tactile_sensor/predict_pose",
            ).value
        )

        self.cache = TaskStateCache()
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.task_running = False
        self.current_task_id = ""
        self.current_task_name = ""
        self.current_robot: Robot | None = None

        self.state_pub = self.create_publisher(TaskState, "/task/state", 10)
        self.arm_command_pub = self.create_publisher(ArmCommand, "/arm/command", 10)
        self.detect_trigger_pub = self.create_publisher(Empty, "/vision/detect_trigger", 10)
        self.tactile_pose_client = self.create_client(
            GetTactilePose,
            self.tactile_pose_service_name,
        )
        self.create_subscription(TaskCommand, "/task/run", self._on_run_task, 10)
        self.create_subscription(Empty, "/task/cancel", self._on_cancel_task, 10)
        self.create_subscription(BlockDetection, "/vision/block_detection", self._on_detection, 10)
        self.create_subscription(ArmState, "/arm/state", self._on_arm_state, 10)

        if self.auto_start and self.default_task:
            self.create_timer(0.1, self._auto_start_once)
            self._auto_started = False
        else:
            self._auto_started = True

    def _auto_start_once(self) -> None:
        if self._auto_started:
            return
        self._auto_started = True
        msg = TaskCommand()
        msg.id = f"auto-{time.monotonic_ns()}"
        msg.task = self.default_task
        msg.args_json = "{}"
        self._on_run_task(msg)

    def _on_detection(self, msg: BlockDetection) -> None:
        self.cache.set_detection(msg)

    def _on_arm_state(self, msg: ArmState) -> None:
        self.cache.set_arm_state(msg)

    def _on_cancel_task(self, _msg: Empty) -> None:
        robot: Robot | None
        with self.lock:
            if not self.task_running:
                self._publish_state("", False, "", "", False, "no task is running")
                return
            self.cancel_event.set()
            robot = self.current_robot
            task_id = self.current_task_id
            task_name = self.current_task_name
        self._publish_state(task_id, True, task_name, "cancel_requested", False, "cancel requested")
        if robot is not None:
            try:
                robot.stop(wait=False)
            except Exception as exc:
                self.get_logger().warn(f"Could not send stop command during task cancel: {exc}")

    def _on_run_task(self, msg: TaskCommand) -> None:
        command_id = str(msg.id).strip() or f"task-{time.monotonic_ns()}"
        task_name = str(msg.task).strip()
        module: ModuleType
        args: dict[str, Any]
        try:
            module, args = self._load_task(task_name, msg.args_json)
        except Exception as exc:
            self._publish_state(command_id, False, task_name, "", False, str(exc))
            return

        with self.lock:
            if self.task_running:
                self._publish_state(command_id, False, task_name, "", False, "task already running")
                return
            self.task_running = True
            self.current_task_id = command_id
            self.current_task_name = task_name
            self.cancel_event.clear()

        thread = threading.Thread(
            target=self._run_task,
            args=(command_id, task_name, module, args),
            daemon=True,
        )
        thread.start()

    def _load_task(self, task_name: str, args_json: str) -> tuple[ModuleType, dict[str, Any]]:
        if not _TASK_NAME_RE.match(task_name) or task_name.startswith("_"):
            raise RuntimeError(f"invalid task name: {task_name!r}")
        try:
            parsed = json.loads(args_json or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"args_json is invalid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("args_json must decode to a JSON object")
        module = importlib.import_module(f"{self.task_package}.{task_name}")
        run = getattr(module, "run", None)
        if not callable(run):
            raise RuntimeError(f"task module {task_name!r} does not define run(robot, args)")
        return module, parsed

    def _run_task(self, task_id: str, task_name: str, module: ModuleType, args: dict[str, Any]) -> None:
        logger = TaskLogger(
            lambda stage, message, success: self._publish_state(task_id, True, task_name, stage, success, message),
            self.get_logger(),
        )
        robot = Robot(
            self.arm_command_pub,
            self.detect_trigger_pub,
            self.cache,
            self.get_clock(),
            logger,
            self.cancel_event.is_set,
            tactile_pose_request=self.call_tactile_pose_service,
        )
        with self.lock:
            self.current_robot = robot
        self._publish_state(task_id, True, task_name, "started", True, "started")
        success = False
        message = ""
        try:
            module.run(robot, args)
            robot.check_cancelled()
            success = True
            message = "complete"
        except CancelledError as exc:
            message = str(exc)
        except Exception as exc:
            message = str(exc)
        finally:
            with self.lock:
                self.task_running = False
                self.current_robot = None
                self.current_task_id = ""
                self.current_task_name = ""
            self._publish_state(task_id, False, task_name, "done", success, message)
            if success:
                self.get_logger().info(f"task {task_name} complete")
            else:
                self.get_logger().error(f"task {task_name} failed: {message}")

    def call_tactile_pose_service(
        self,
        timeout_sec: float = 3.0,
    ) -> GetTactilePose.Response:
        timeout = max(0.0, float(timeout_sec))
        if not self.tactile_pose_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(
                "tactile pose service unavailable: "
                f"{self.tactile_pose_service_name}"
            )

        event = threading.Event()
        result: dict[str, Any] = {}
        future = self.tactile_pose_client.call_async(GetTactilePose.Request())

        def done_callback(done_future) -> None:
            try:
                result["response"] = done_future.result()
            except Exception as exc:
                result["error"] = exc
            finally:
                event.set()

        future.add_done_callback(done_callback)
        if not event.wait(timeout):
            if hasattr(future, "cancel"):
                future.cancel()
            raise RuntimeError("tactile pose service call timed out")
        if "error" in result:
            raise RuntimeError(f"tactile pose service call failed: {result['error']}")
        response = result.get("response")
        if response is None:
            raise RuntimeError("tactile pose service returned no response")
        return response

    def _publish_state(
        self,
        task_id: str,
        running: bool,
        task: str,
        stage: str,
        success: bool,
        message: str,
    ) -> None:
        msg = TaskState()
        msg.id = str(task_id)
        msg.running = bool(running)
        msg.task = str(task)
        msg.stage = str(stage)
        msg.success = bool(success)
        msg.message = str(message)
        self.state_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TaskNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
