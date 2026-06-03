from pathlib import Path
import sys
import threading
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tactile_arm.dynamixel_driver import DynamixelDriver


class FakePortHandler:
    last = None

    def __init__(self, port_name):
        self.port_name = port_name
        self.closed = False
        FakePortHandler.last = self

    def openPort(self):
        return True

    def setBaudRate(self, _baudrate):
        return True

    def closePort(self):
        self.closed = True


class FailingClosePortHandler(FakePortHandler):
    def closePort(self):
        self.closed = True
        raise RuntimeError("close failed")


class FakePacketHandler:
    def __init__(self, _protocol):
        pass

    def write1ByteTxRx(self, _port, _dxl_id, _address, _value):
        return (0, 0)

    def write4ByteTxRx(self, _port, _dxl_id, _address, _value):
        return (0, 0)

    def read4ByteTxRx(self, _port, _dxl_id, _address):
        return (2048, 0, 0)

    def getTxRxResult(self, result):
        return f"result {result}"

    def getRxPacketError(self, error):
        return f"error {error}"


def test_connect_failure_closes_opened_port_and_clears_state(tmp_path, monkeypatch):
    fake_sdk = types.SimpleNamespace(
        PacketHandler=FakePacketHandler,
        PortHandler=FakePortHandler,
    )
    monkeypatch.setitem(sys.modules, "dynamixel_sdk", fake_sdk)
    driver = DynamixelDriver()
    monkeypatch.setattr(driver, "_init_motors_unlocked", lambda: (_ for _ in ()).throw(RuntimeError("init failed")))
    port_path = tmp_path / "ttyUSB-test"
    port_path.write_text("")

    with pytest.raises(RuntimeError, match="init failed"):
        driver.connect(str(port_path))

    assert FakePortHandler.last.closed is True
    assert driver.port is None
    assert driver.packet is None
    assert driver.current_port == ""
    assert driver.connected is False


def test_disconnect_clears_state_even_if_close_raises():
    driver = DynamixelDriver()
    port = FailingClosePortHandler("/dev/ttyUSB-test")
    driver.port = port
    driver.packet = None
    driver.current_port = "/dev/ttyUSB-test"
    driver.connected = False

    with pytest.raises(RuntimeError, match="close failed"):
        driver.disconnect()

    assert port.closed is True
    assert driver.port is None
    assert driver.packet is None
    assert driver.current_port == ""
    assert driver.connected is False


def test_disconnect_closes_stale_port_even_when_marked_disconnected():
    driver = DynamixelDriver()
    port = FakePortHandler("/dev/ttyUSB-test")
    driver.port = port
    driver.packet = None
    driver.current_port = "/dev/ttyUSB-test"
    driver.connected = False

    driver.disconnect()

    assert port.closed is True
    assert driver.port is None
    assert driver.packet is None
    assert driver.current_port == ""
    assert driver.connected is False


def test_io_lock_reports_busy_instead_of_touching_sdk_from_two_threads():
    driver = DynamixelDriver()
    driver.io_lock_timeout_sec = 0.01
    errors = []

    def call_driver():
        try:
            driver.set_gripper(1800)
        except Exception as exc:
            errors.append(str(exc))

    driver.io_lock.acquire()
    try:
        worker = threading.Thread(target=call_driver)
        worker.start()
        worker.join(timeout=1.0)
    finally:
        driver.io_lock.release()

    assert not worker.is_alive()
    assert errors
    assert "Dynamixel serial bus is busy" in errors[0]
