from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tactile_arm.kinematics import ArmKinematics, ArmPose


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
