from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RecursiveRobot:
    def __init__(self, detections, tactile_poses=()):
        self.detections = list(detections)
        self.tactile_poses = list(tactile_poses)
        self.calls = []

    def wait_ready(self, timeout_sec=5.0):
        self.calls.append(("wait_ready", timeout_sec))

    def check_cancelled(self):
        pass

    def detect_block(self, timeout):
        self.calls.append(("detect_block", timeout))
        if not self.detections:
            from tactile_task.commands import CancelledError

            raise CancelledError("test complete")
        detection = self.detections.pop(0)
        if isinstance(detection, Exception):
            raise detection
        return detection

    def detect_tactile_pose(self, timeout):
        self.calls.append(("detect_tactile_pose", timeout))
        if not self.tactile_poses:
            raise AssertionError("unexpected tactile pose request")
        return self.tactile_poses.pop(0)

    def move(
        self,
        x,
        y,
        z,
        angle_rad,
        duration,
        *,
        stage=None,
        timeout_margin_sec=2.0,
        settle_sec=0.0,
    ):
        self.calls.append(
            (
                "move",
                stage,
                x,
                y,
                z,
                angle_rad,
                duration,
                timeout_margin_sec,
                settle_sec,
            )
        )

    def open_gripper(
        self,
        position,
        *,
        stage="gripper_open",
        timeout_sec=2.0,
        settle_sec=0.0,
    ):
        self.calls.append(("open_gripper", stage, position, timeout_sec, settle_sec))

    def close_gripper(
        self,
        position,
        *,
        stage="gripper_close",
        timeout_sec=2.0,
        settle_sec=0.0,
    ):
        self.calls.append(("close_gripper", stage, position, timeout_sec, settle_sec))

    def log(self, message):
        self.calls.append(("log", str(message)))


class CancelDuringWaitRobot(RecursiveRobot):
    def __init__(self):
        super().__init__([_detection(confidence=0.1)])
        self.cancelled = False

    def check_cancelled(self):
        if self.cancelled:
            from tactile_task.commands import CancelledError

            raise CancelledError("task cancelled")

    def move(self, *args, **kwargs):
        super().move(*args, **kwargs)
        self.cancelled = True


def test_recursive_low_confidence_moves_to_avoidance_once_and_retries():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.commands import CancelledError
    from tactile_task.tasks.recursive_tactile_pick_place import run

    robot = RecursiveRobot(
        [
            _detection(confidence=0.2),
            _detection(confidence=0.3),
        ]
    )

    with pytest.raises(CancelledError, match="test complete"):
        run(robot, _fast_args())

    no_block_moves = [
        call for call in robot.calls if call[0] == "move" and call[1] == "no_block_pose"
    ]
    assert len(no_block_moves) == 1
    assert (no_block_moves[0][2], no_block_moves[0][3], no_block_moves[0][4]) == (
        0.07,
        0.0,
        0.1,
    )
    assert [call[0] for call in robot.calls].count("detect_block") == 3


def test_recursive_block_inside_place_tolerance_is_skipped():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.commands import CancelledError
    from tactile_task.tasks.recursive_tactile_pick_place import run

    robot = RecursiveRobot(
        [_detection(grid_column=3.04, grid_row=6.0, confidence=0.9)]
    )

    with pytest.raises(CancelledError, match="test complete"):
        run(
            robot,
            _fast_args(
                {
                    "place": {"x": 0.1, "y": 0.124, "z": 0.037, "angle_rad": -1.45}
                }
            ),
        )

    assert all(call[0] != "close_gripper" for call in robot.calls)
    assert all(call[0] != "move" for call in robot.calls)
    assert any(
        call[0] == "log" and call[1].startswith("block_at_place")
        for call in robot.calls
    )


def test_recursive_block_outside_tolerance_runs_tactile_pick_place():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.commands import CancelledError
    from tactile_task.tasks.recursive_tactile_pick_place import run

    robot = RecursiveRobot(
        [_detection(grid_column=8.0, grid_row=2.0, confidence=0.9)],
        [_tactile_pose(detected=True)],
    )

    with pytest.raises(CancelledError, match="test complete"):
        run(robot, _fast_args())

    command_stages = [
        call[1]
        for call in robot.calls
        if call[0] in {"move", "open_gripper", "close_gripper"}
    ]
    assert command_stages[:9] == [
        "move_above_pick",
        "gripper_open",
        "descend_to_pick",
        "gripper_close",
        "lift",
        "move_above_place",
        "descend_to_place",
        "release",
        "retreat",
    ]


def test_recursive_tactile_miss_opens_and_moves_to_avoidance_before_retry():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.commands import CancelledError
    from tactile_task.tasks.recursive_tactile_pick_place import run

    robot = RecursiveRobot(
        [_detection(grid_column=8.0, grid_row=2.0, confidence=0.9)],
        [_tactile_pose(detected=False, message="no tactile block detected")],
    )

    with pytest.raises(CancelledError, match="test complete"):
        run(robot, _fast_args())

    retry_open = [
        call
        for call in robot.calls
        if call[0] == "open_gripper" and call[1] == "retry_open_after_tactile_miss"
    ]
    retry_moves = [
        call
        for call in robot.calls
        if call[0] == "move" and call[1] == "retry_no_block_pose"
    ]
    assert len(retry_open) == 1
    assert len(retry_moves) == 1


def test_recursive_cancel_during_loop_wait_exits_promptly():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.commands import CancelledError
    from tactile_task.tasks.recursive_tactile_pick_place import run

    robot = CancelDuringWaitRobot()

    with pytest.raises(CancelledError, match="task cancelled"):
        run(robot, _fast_args({"loop_period_sec": 1.0}))


def _fast_args(extra=None):
    args = {
        "loop_period_sec": 0.0,
        "detection_timeout_sec": 0.0,
        "tactile_pose_wait_sec": 0.0,
        "move_settle_sec": 0.0,
        "gripper_settle_sec": 0.0,
    }
    if extra:
        args.update(extra)
    return args


def _detection(
    *,
    detected=True,
    grid_valid=True,
    grid_column=8.0,
    grid_row=2.0,
    confidence=0.9,
    message="",
):
    from tactile_interfaces.msg import BlockDetection

    detection = BlockDetection()
    detection.detected = detected
    detection.grid_position_valid = grid_valid
    detection.grid_column = float(grid_column)
    detection.grid_row = float(grid_row)
    detection.confidence = float(confidence)
    detection.message = str(message)
    return detection


def _tactile_pose(*, detected=True, message="prediction ready"):
    return type(
        "TactilePose",
        (),
        {
            "detected": bool(detected),
            "x": 7.5,
            "y": 3.5,
            "message": str(message),
        },
    )()
