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
        self.txrx_writes = []
        self.txonly_writes = []

    def write1ByteTxRx(self, _port, _dxl_id, _address, _value):
        return (0, 0)

    def write4ByteTxRx(self, _port, _dxl_id, _address, _value):
        self.txrx_writes.append((_dxl_id, _address, _value))
        return (0, 0)

    def write4ByteTxOnly(self, _port, _dxl_id, _address, _value):
        self.txonly_writes.append((_dxl_id, _address, _value))
        return 0

    def read4ByteTxRx(self, _port, _dxl_id, _address):
        return (2048, 0, 0)

    def getTxRxResult(self, result):
        return f"result {result}"

    def getRxPacketError(self, error):
        return f"error {error}"


class FailsOncePacketHandler(FakePacketHandler):
    def __init__(self, _protocol=None):
        self.write_count = 0

    def write4ByteTxRx(self, _port, _dxl_id, _address, _value):
        self.write_count += 1
        if self.write_count == 1:
            return (-3001, 0)
        return (0, 0)

    def getTxRxResult(self, result):
        if result == -3001:
            return "[TxRxResult] There is no status packet!"
        return super().getTxRxResult(result)


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


def test_txrx_retries_transient_no_status_packet():
    driver = DynamixelDriver(io_retry_count=1, io_retry_delay_sec=0.0)
    packet = FailsOncePacketHandler()
    driver.packet = packet
    driver.port = FakePortHandler("/dev/ttyUSB-test")

    driver._write4(11, driver.address.goal_pos, 2048, "test write")

    assert packet.write_count == 2


def test_goal_position_writes_require_ack_by_default():
    driver = DynamixelDriver()
    packet = FakePacketHandler(None)
    driver.packet = packet
    driver.port = FakePortHandler("/dev/ttyUSB-test")
    driver.connected = True

    driver.set_joint_angles([0.0, 0.0, 0.0, 0.0])

    assert packet.txonly_writes == []
    assert len(packet.txrx_writes) == 4
    assert {address for _dxl_id, address, _value in packet.txrx_writes} == {driver.address.goal_pos}


def test_goal_position_writes_can_use_tx_only():
    driver = DynamixelDriver(goal_write_ack=False)
    packet = FakePacketHandler(None)
    driver.packet = packet
    driver.port = FakePortHandler("/dev/ttyUSB-test")
    driver.connected = True

    driver.set_gripper(1800)

    assert packet.txonly_writes == [(15, driver.address.goal_pos, 1800)]
    assert packet.txrx_writes == []


def test_set_joint_speeds_writes_profile_velocity_for_each_joint():
    driver = DynamixelDriver()
    packet = FakePacketHandler(None)
    driver.packet = packet
    driver.port = FakePortHandler("/dev/ttyUSB-test")
    driver.connected = True

    driver.set_joint_speeds([100, 50, 25, 1])

    assert packet.txrx_writes == [
        (11, driver.address.profile_velocity, 100),
        (12, driver.address.profile_velocity, 50),
        (13, driver.address.profile_velocity, 25),
        (14, driver.address.profile_velocity, 1),
    ]
