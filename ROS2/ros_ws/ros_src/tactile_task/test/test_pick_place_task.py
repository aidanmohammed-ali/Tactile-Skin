from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeRobot:
    def __init__(self, detection):
        self.detection = detection
        self.calls = []

    def wait_ready(self, timeout_sec=5.0):
        self.calls.append(("wait_ready", timeout_sec))

    def detect_block(self, timeout):
        self.calls.append(("detect_block", timeout))
        return self.detection

    def move(self, x, y, z, angle_rad, duration, *, stage=None, timeout_margin_sec=2.0, settle_sec=0.0):
        self.calls.append(("move", stage, x, y, z, angle_rad, duration, timeout_margin_sec, settle_sec))

    def open_gripper(self, position, *, stage="gripper_open", timeout_sec=2.0, settle_sec=0.0):
        self.calls.append(("open_gripper", stage, position, timeout_sec, settle_sec))

    def close_gripper(self, position, *, stage="gripper_close", timeout_sec=2.0, settle_sec=0.0):
        self.calls.append(("close_gripper", stage, position, timeout_sec, settle_sec))

    def log(self, stage):
        self.calls.append(("log", stage))


def test_pick_place_script_uses_robot_commands_for_sequence():
    pytest.importorskip("tactile_interfaces")
    from tactile_interfaces.msg import BlockDetection
    from tactile_task.tasks.pick_place import run

    detection = BlockDetection()
    detection.detected = True
    detection.grid_position_valid = True
    detection.grid_column = 8.0
    detection.grid_row = 2.0
    robot = FakeRobot(detection)

    run(robot, {"place": {"x": 0.12, "y": -0.05, "z": 0.03, "angle_rad": -1.2}})

    stages = [call[1] for call in robot.calls if call[0] in ("move", "open_gripper", "close_gripper")]
    assert stages == [
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
    assert robot.calls[0] == ("wait_ready", 5.0)
    assert robot.calls[-1] == ("log", "pick-place complete")


def test_pick_place_script_rejects_invalid_detection():
    pytest.importorskip("tactile_interfaces")
    from tactile_interfaces.msg import BlockDetection
    from tactile_task.tasks.pick_place import run

    detection = BlockDetection()
    detection.detected = False
    detection.message = "no block detected"

    with pytest.raises(RuntimeError, match="no block detected"):
        run(FakeRobot(detection), {})


def test_tactile_pick_height_uses_minimum_below_start_distance():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.tasks.tactile_pick_place import CONFIG, _pick_height

    assert _pick_height(0.06, 0.08, CONFIG) == pytest.approx(CONFIG["pick_z"])


def test_tactile_pick_height_scales_between_distance_limits():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.tasks.tactile_pick_place import CONFIG, _pick_height

    assert _pick_height(0.15, 0.0, CONFIG) == pytest.approx(CONFIG["pick_z"] + 0.005)


def test_tactile_pick_height_reaches_maximum_raise_at_far_distance():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.tasks.tactile_pick_place import CONFIG, _pick_height

    assert _pick_height(0.2, 0.0, CONFIG) == pytest.approx(CONFIG["pick_z"] + 0.01)
    assert _pick_height(0.3, 0.0, CONFIG) == pytest.approx(CONFIG["pick_z"] + 0.01)


def test_blind_tactile_pick_place_uses_configured_pick_position():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.tasks.blind_tactile_pick_place import run

    class BlindRobot:
        def __init__(self):
            self.calls = []

        def wait_ready(self, timeout_sec=5.0):
            self.calls.append(("wait_ready", timeout_sec))

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
            self.calls.append(("move", stage, x, y, z))

        def open_gripper(self, *args, **kwargs):
            pass

        def close_gripper(self, *args, **kwargs):
            pass

        def check_cancelled(self):
            pass

        def detect_tactile_pose(self, timeout):
            return type("TactilePose", (), {"detected": True, "x": 7.5, "y": 3.5})()

        def log(self, message):
            pass

    robot = BlindRobot()
    run(
        robot,
        {
            "pick": {
                "x": 0.14,
                "y": -0.03,
                "z": 0.025,
                "angle_rad": -1.4,
            },
            "tactile_pose_wait_sec": 0.0,
            "move_settle_sec": 0.0,
            "gripper_settle_sec": 0.0,
        },
    )

    pick_moves = [
        call
        for call in robot.calls
        if call[0] == "move"
        and call[1] in {"move_above_pick", "descend_to_pick", "lift"}
    ]
    assert [(call[2], call[3]) for call in pick_moves] == [
        (0.14, -0.03),
        (0.14, -0.03),
        (0.14, -0.03),
    ]


def test_blind_pick_stops_after_lifting_from_configured_position():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.tasks.blind_pick import run

    robot = FakeRobot(None)
    run(
        robot,
        {
            "pick": {
                "x": 0.14,
                "y": -0.03,
                "z": 0.025,
                "angle_rad": -1.4,
            },
            "move_settle_sec": 0.0,
            "gripper_settle_sec": 0.0,
        },
    )

    command_calls = [
        call
        for call in robot.calls
        if call[0] in ("move", "open_gripper", "close_gripper")
    ]
    assert [call[1] for call in command_calls] == [
        "move_above_pick",
        "gripper_open",
        "descend_to_pick",
        "gripper_close",
        "lift",
    ]
    assert [(call[2], call[3]) for call in command_calls if call[0] == "move"] == [
        (0.14, -0.03),
        (0.14, -0.03),
        (0.14, -0.03),
    ]
    assert robot.calls[-1] == ("log", "blind pick complete")


def test_tactile_pick_calls_tactile_service_after_lifting():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.tasks.tactile_pick import run

    class TactilePickRobot(FakeRobot):
        def check_cancelled(self):
            pass

        def detect_tactile_pose(self, timeout):
            self.calls.append(("detect_tactile_pose", timeout))
            return type(
                "TactilePose",
                (),
                {
                    "detected": True,
                    "x": 7.5,
                    "y": 3.5,
                    "angle_rad": 0.2,
                    "confidence": 0.9,
                    "fully_inside_sensor": True,
                    "message": "",
                },
            )()

    robot = TactilePickRobot(None)
    run(
        robot,
        {
            "tactile_pose_wait_sec": 0.0,
            "tactile_pose_timeout_sec": 4.0,
            "move_settle_sec": 0.0,
            "gripper_settle_sec": 0.0,
        },
    )

    lift_index = next(
        index
        for index, call in enumerate(robot.calls)
        if call[0] == "move" and call[1] == "lift"
    )
    tactile_index = robot.calls.index(("detect_tactile_pose", 4.0))
    assert tactile_index > lift_index
    assert robot.calls[-1] == ("log", "tactile pick complete")


def test_tactile_pick_fails_when_tactile_detection_is_empty():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.tasks.tactile_pick import run

    class EmptyTactileRobot(FakeRobot):
        def check_cancelled(self):
            pass

        def detect_tactile_pose(self, _timeout):
            return type(
                "TactilePose",
                (),
                {"detected": False, "message": "no contact"},
            )()

    with pytest.raises(RuntimeError, match="no contact"):
        run(
            EmptyTactileRobot(None),
            {
                "tactile_pose_wait_sec": 0.0,
                "move_settle_sec": 0.0,
                "gripper_settle_sec": 0.0,
            },
        )


def test_blind_pick_place_uses_fixed_configured_poses_without_detection():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.tasks.blind_pick_place import run

    robot = FakeRobot(None)
    run(
        robot,
        {
            "pick": {
                "x": 0.14,
                "y": -0.03,
                "z": 0.025,
                "angle_rad": -1.4,
            },
            "place": {
                "x": 0.1,
                "y": 0.08,
                "z": 0.03,
                "angle_rad": -1.3,
            },
            "move_settle_sec": 0.0,
            "gripper_settle_sec": 0.0,
        },
    )

    command_calls = [
        call
        for call in robot.calls
        if call[0] in ("move", "open_gripper", "close_gripper")
    ]
    assert [call[1] for call in command_calls] == [
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
    place_moves = [
        call
        for call in command_calls
        if call[0] == "move"
        and call[1] in {"move_above_place", "descend_to_place", "retreat"}
    ]
    assert [(call[2], call[3]) for call in place_moves] == [
        (0.1, 0.08),
        (0.1, 0.08),
        (0.1, 0.08),
    ]
    assert all(call[0] != "detect_block" for call in robot.calls)
    assert robot.calls[-1] == ("log", "blind pick-place complete")
