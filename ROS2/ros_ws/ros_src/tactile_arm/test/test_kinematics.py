from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tactile_arm.kinematics import ArmKinematics, ArmPose, synchronized_joint_speeds


def test_synchronized_joint_speeds_scale_with_joint_change():
    speeds = synchronized_joint_speeds(
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.25, 0.0],
        maximum_speed=100,
        minimum_speed=1,
    )

    assert speeds == [100, 50, 25, 1]


def test_synchronized_joint_speeds_respect_minimum_speed():
    speeds = synchronized_joint_speeds(
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.001, 0.0, 0.5],
        maximum_speed=80,
        minimum_speed=5,
    )

    assert speeds == [80, 5, 5, 40]


def test_inverse_forward_round_trip_for_reachable_pose():
    kinematics = ArmKinematics()
    pose = ArmPose(0.14, 0.04, 0.11, -0.8)

    joints = kinematics.inverse(pose)
    actual = kinematics.forward(joints)

    assert actual.x == pytest.approx(pose.x, abs=1e-6)
    assert actual.y == pytest.approx(pose.y, abs=1e-6)
    assert actual.z == pytest.approx(pose.z, abs=1e-6)
    assert actual.angle_rad == pytest.approx(pose.angle_rad, abs=1e-6)


def test_inverse_rejects_unreachable_pose():
    with pytest.raises(ValueError):
        ArmKinematics().inverse(ArmPose(1.0, 0.0, 1.0, 0.0))


@pytest.mark.parametrize("joint4_degrees", [-110.0, 90.0])
def test_inverse_accepts_joint4_limit_boundaries(joint4_degrees):
    kinematics = ArmKinematics()
    joints = [0.0, 0.5, -1.0, math.radians(joint4_degrees)]

    actual = kinematics.inverse(kinematics.forward(joints))

    assert actual[3] == pytest.approx(joints[3])


@pytest.mark.parametrize("joint4_degrees", [-111.0, 91.0])
def test_inverse_rejects_joint4_outside_limits(joint4_degrees):
    kinematics = ArmKinematics()
    joints = [0.0, 0.5, -1.0, math.radians(joint4_degrees)]

    with pytest.raises(ValueError, match="joint4 outside limits"):
        kinematics.inverse(kinematics.forward(joints))
