from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))

    def error(self, message):
        self.messages.append(str(message))

    def stage(self, message):
        self.messages.append(str(message))


def _arm_state(connected=True, busy=False, last_error="", completed_id="", completed_success=True):
    from tactile_interfaces.msg import ArmState

    state = ArmState()
    state.connected = bool(connected)
    state.busy = bool(busy)
    state.last_error = str(last_error)
    state.completed_command_id = str(completed_id)
    state.completed_success = bool(completed_success)
    return state


def test_robot_move_waits_for_matching_command_completion():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.commands import Robot, TaskStateCache

    cache = TaskStateCache()
    cache.set_arm_state(_arm_state(connected=True, busy=False))

    class ArmPublisher:
        def __init__(self):
            self.messages = []

        def publish(self, msg):
            self.messages.append(msg)

            def worker():
                cache.set_arm_state(_arm_state(connected=True, busy=True, completed_id="older"))
                time.sleep(0.02)
                cache.set_arm_state(_arm_state(connected=True, busy=False, completed_id=msg.id, completed_success=True))

            threading.Thread(target=worker, daemon=True).start()

    arm_pub = ArmPublisher()
    robot = Robot(arm_pub, ArmPublisher(), cache, _FakeClock(), FakeLogger())

    robot.move(0.1, 0.2, 0.3, -1.0, 0.1)

    assert len(arm_pub.messages) == 1
    command = arm_pub.messages[0]
    assert command.type == command.TYPE_MOVE
    assert command.target_pose.x == pytest.approx(0.1)


def test_robot_raises_failed_command_message():
    pytest.importorskip("tactile_interfaces")
    from tactile_task.commands import Robot, TaskStateCache

    cache = TaskStateCache()
    cache.set_arm_state(_arm_state(connected=True, busy=False))

    class ArmPublisher:
        def publish(self, msg):
            cache.set_arm_state(
                _arm_state(
                    connected=True,
                    busy=False,
                    last_error="pose is outside reachable workspace",
                    completed_id=msg.id,
                    completed_success=False,
                )
            )

    robot = Robot(ArmPublisher(), ArmPublisher(), cache, _FakeClock(), FakeLogger())

    with pytest.raises(RuntimeError, match="outside reachable workspace"):
        robot.move(9.0, 9.0, 9.0, 0.0, 0.1)


def test_robot_move_increases_unreachable_angle_before_sending(monkeypatch):
    pytest.importorskip("tactile_interfaces")
    from tactile_task.commands import Robot, TaskStateCache

    cache = TaskStateCache()
    cache.set_arm_state(_arm_state(connected=True, busy=False))

    class FakeKinematics:
        def inverse(self, pose):
            if pose.angle_rad < -1.0:
                raise ValueError("pose is outside reachable workspace")
            return [0.0, 0.0, 0.0, 0.0]

    class FakePose:
        def __init__(self, x, y, z, angle_rad):
            self.x = x
            self.y = y
            self.z = z
            self.angle_rad = angle_rad

    class ArmPublisher:
        def __init__(self):
            self.messages = []

        def publish(self, msg):
            self.messages.append(msg)
            cache.set_arm_state(
                _arm_state(
                    connected=True,
                    busy=False,
                    completed_id=msg.id,
                    completed_success=True,
                )
            )

    monkeypatch.setattr(
        "tactile_task.commands._load_arm_kinematics",
        lambda: FakeKinematics(),
    )
    monkeypatch.setattr("tactile_task.commands._arm_pose", FakePose)
    arm_pub = ArmPublisher()
    robot = Robot(arm_pub, _NoopPublisher(), cache, _FakeClock(), FakeLogger())

    robot.move(0.1, 0.2, 0.3, -1.57, 0.1)

    assert arm_pub.messages[0].target_pose.angle_rad == pytest.approx(-1.0)


def test_robot_move_stops_angle_retry_at_minus_one(monkeypatch):
    pytest.importorskip("tactile_interfaces")
    from tactile_task.commands import Robot, TaskStateCache

    cache = TaskStateCache()
    cache.set_arm_state(_arm_state(connected=True, busy=False))

    class FakeKinematics:
        def inverse(self, _pose):
            raise ValueError("pose is outside reachable workspace")

    class FakePose:
        def __init__(self, x, y, z, angle_rad):
            self.x = x
            self.y = y
            self.z = z
            self.angle_rad = angle_rad

    monkeypatch.setattr(
        "tactile_task.commands._load_arm_kinematics",
        lambda: FakeKinematics(),
    )
    monkeypatch.setattr("tactile_task.commands._arm_pose", FakePose)
    robot = Robot(_NoopPublisher(), _NoopPublisher(), cache, _FakeClock(), FakeLogger())

    with pytest.raises(RuntimeError, match="to -1.000"):
        robot.move(0.1, 0.2, 0.3, -1.57, 0.1)


def test_robot_detect_block_ignores_stale_detection_stamp():
    pytest.importorskip("tactile_interfaces")
    from tactile_interfaces.msg import BlockDetection
    from tactile_task.commands import Robot, TaskStateCache

    cache = TaskStateCache()
    cache.set_arm_state(_arm_state(connected=True, busy=False))

    class DetectPublisher:
        def publish(self, _msg):
            def worker():
                stale = BlockDetection()
                stale.stamp.sec = 0
                stale.detected = False
                cache.set_detection(stale)
                fresh = BlockDetection()
                fresh.stamp.sec = 2
                fresh.detected = True
                fresh.grid_position_valid = True
                cache.set_detection(fresh)

            threading.Thread(target=worker, daemon=True).start()

    robot = Robot(_NoopPublisher(), DetectPublisher(), cache, _FakeClock(), FakeLogger())

    detection = robot.detect_block(0.5)

    assert detection.detected is True
    assert detection.stamp.sec == 2


def test_robot_settles_after_successful_move_and_gripper(monkeypatch):
    pytest.importorskip("tactile_interfaces")
    from tactile_task.commands import Robot, TaskStateCache

    cache = TaskStateCache()
    cache.set_arm_state(_arm_state(connected=True, busy=False))
    sleeps = []
    now = [100.0]

    def fake_monotonic():
        return now[0]

    def fake_sleep(duration):
        sleeps.append(float(duration))
        now[0] += float(duration)

    monkeypatch.setattr("tactile_task.commands.time.monotonic", fake_monotonic)
    monkeypatch.setattr("tactile_task.commands.time.sleep", fake_sleep)

    class ArmPublisher:
        def publish(self, msg):
            cache.set_arm_state(_arm_state(connected=True, busy=False, completed_id=msg.id, completed_success=True))

    robot = Robot(ArmPublisher(), _NoopPublisher(), cache, _FakeClock(), FakeLogger())

    robot.move(0.1, 0.2, 0.3, -1.0, 0.1, settle_sec=0.15)
    robot.gripper(2400, settle_sec=0.4)

    assert len(sleeps) == 11
    assert sum(sleeps) == pytest.approx(0.55)


def test_robot_settle_responds_to_cancel(monkeypatch):
    pytest.importorskip("tactile_interfaces")
    from tactile_task.commands import CancelledError, Robot, TaskStateCache

    cache = TaskStateCache()
    cache.set_arm_state(_arm_state(connected=True, busy=False))
    now = [100.0]
    cancelled = [False]

    def fake_monotonic():
        return now[0]

    def fake_sleep(duration):
        cancelled[0] = True
        now[0] += float(duration)

    monkeypatch.setattr("tactile_task.commands.time.monotonic", fake_monotonic)
    monkeypatch.setattr("tactile_task.commands.time.sleep", fake_sleep)

    class ArmPublisher:
        def publish(self, msg):
            cache.set_arm_state(_arm_state(connected=True, busy=False, completed_id=msg.id, completed_success=True))

    robot = Robot(
        ArmPublisher(),
        _NoopPublisher(),
        cache,
        _FakeClock(),
        FakeLogger(),
        is_cancelled=lambda: cancelled[0],
    )

    with pytest.raises(CancelledError):
        robot.move(0.1, 0.2, 0.3, -1.0, 0.1, settle_sec=0.2)


class _FakeClock:
    class Now:
        nanoseconds = 1_000_000_000

    def now(self):
        return self.Now()


class _NoopPublisher:
    def publish(self, _msg):
        pass
