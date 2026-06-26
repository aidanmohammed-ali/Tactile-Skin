from pathlib import Path
import sys

import pytest

pytest.importorskip("rclpy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from tactile_vision.qos import image_qos_profile


def test_image_qos_profile_defaults_to_latest_best_effort():
    qos = image_qos_profile()

    assert qos.depth == 1
    assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert qos.durability == DurabilityPolicy.VOLATILE


def test_image_qos_profile_can_use_reliable_when_configured():
    qos = image_qos_profile(depth=3, reliability="reliable")

    assert qos.depth == 3
    assert qos.reliability == ReliabilityPolicy.RELIABLE
