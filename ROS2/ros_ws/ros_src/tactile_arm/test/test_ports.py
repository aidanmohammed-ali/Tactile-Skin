from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tactile_arm.ports import _port_sort_key, recommended_port


def test_port_sort_prefers_usb_then_acm():
    ports = ["/dev/serial/by-id/device", "/dev/ttyACM0", "/dev/ttyUSB1"]

    assert sorted(ports, key=_port_sort_key) == ["/dev/ttyUSB1", "/dev/ttyACM0", "/dev/serial/by-id/device"]


def test_recommended_port_uses_first_available():
    assert recommended_port(["/dev/ttyUSB0", "/dev/ttyACM0"]) == "/dev/ttyUSB0"
    assert recommended_port([]) == ""
