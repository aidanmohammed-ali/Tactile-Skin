import time

import numpy as np

from block_position_prediction.data_collection_manual.tactile import (
    NUM_TAXELS,
    ThreadedTactileReader,
    available_tactile_ports,
    _is_tty_acm_port,
    canonicalize_tactile_values,
    draw_tactile_heatmap,
    top5_average_over_recent_frames,
)


def test_top5_average_over_recent_frames_matches_visualiser_t_mode():
    frames = [np.full(NUM_TAXELS, value, dtype=np.uint16) for value in range(10)]

    normalized, raw_average = top5_average_over_recent_frames(frames)

    assert normalized is not None
    assert raw_average is not None
    assert normalized.shape == (NUM_TAXELS,)
    assert raw_average.shape == (NUM_TAXELS,)
    assert float(raw_average[0]) == 7.0
    assert abs(float(normalized[0]) - (7.0 / 65535.0)) < 1e-8


def test_threaded_tactile_reader_simulator_produces_snapshot():
    reader = ThreadedTactileReader("SIMULATOR", target_hz=120.0)
    reader.start()
    try:
        deadline = time.time() + 1.0
        snapshot = reader.snapshot()
        while time.time() < deadline:
            snapshot = reader.snapshot()
            if snapshot.available:
                break
            time.sleep(0.02)
        assert snapshot.available
        assert snapshot.processed is not None
        assert len(snapshot.recent_raw_frames) <= 10
        assert snapshot.to_sensor_payload()["available"] is True
    finally:
        reader.stop()


def test_threaded_tactile_reader_calls_callback_for_each_new_frame():
    timestamps = []
    reader = ThreadedTactileReader(
        "SIMULATOR",
        target_hz=120.0,
        frame_callback=lambda snapshot: timestamps.append(snapshot.timestamp),
    )
    reader.start()
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline and len(timestamps) < 3:
            time.sleep(0.01)
        assert len(timestamps) >= 3
        assert timestamps == sorted(set(timestamps))
    finally:
        reader.stop()


def test_capture_new_raw_frames_waits_for_consecutive_future_frames():
    reader = ThreadedTactileReader("SIMULATOR", target_hz=120.0)
    reader.start()
    try:
        frames = reader.capture_new_raw_frames(5, timeout=1.0)

        assert len(frames) == 5
        assert all(frame.shape == (NUM_TAXELS,) for frame in frames)
    finally:
        reader.stop()


def test_capture_new_raw_frames_times_out_without_input():
    reader = ThreadedTactileReader("SIMULATOR")

    frames = reader.capture_new_raw_frames(5, timeout=0.01)

    assert frames == ()


def test_available_tactile_ports_always_includes_simulator():
    assert available_tactile_ports()[0] == "SIMULATOR"


def test_tactile_port_filter_only_accepts_tty_acm_names():
    assert _is_tty_acm_port("/dev/ttyACM0")
    assert _is_tty_acm_port("ttyACM1")
    assert not _is_tty_acm_port("/dev/ttyUSB0")
    assert not _is_tty_acm_port("COM7")


def test_canonicalize_tactile_values_flips_hardware_order_without_changing_source():
    values = np.zeros(NUM_TAXELS, dtype=np.float32)
    values[0] = 1.0

    canonical = canonicalize_tactile_values(values)

    assert float(values[0]) == 1.0
    assert canonical is not None
    assert float(canonical[15]) == 1.0
    assert float(canonical[0]) == 0.0


def test_tactile_heatmap_draws_values_in_given_order():
    values = np.zeros(NUM_TAXELS, dtype=np.float32)
    values[0] = 1.0

    heatmap = draw_tactile_heatmap(values, width=160)

    assert int(heatmap[45, 1, 2]) == 255
    assert int(heatmap[45, 151, 2]) == 0


def test_threaded_tactile_reader_reconnects_to_simulator():
    reader = ThreadedTactileReader("SIMULATOR", target_hz=120.0)
    reader.start()
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline and not reader.snapshot().available:
            time.sleep(0.02)
        assert reader.snapshot().available

        reader.reconnect("SIMULATOR")
        assert reader.port == "SIMULATOR"
        deadline = time.time() + 1.0
        while time.time() < deadline and not reader.snapshot().available:
            time.sleep(0.02)
        assert reader.snapshot().available
    finally:
        reader.stop()
