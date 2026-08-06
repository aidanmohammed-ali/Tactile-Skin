import time

import numpy as np

from block_position_prediction.data_collection.app import (
    DataCollectionApp,
    SourceEditor,
    VisionWorker,
    build_sample_label,
    load_camera_source,
    save_camera_source,
    sensor_payload_from_tactile,
)
from block_position_prediction.data_collection.aruco import PaperCalibration
from block_position_prediction.data_collection.camera import CameraFrame
from block_position_prediction.data_collection.detector import BlockDetection, DetectionResult
from block_position_prediction.data_collection.geometry import SheetConfig
from block_position_prediction.data_collection.tactile import NUM_TAXELS, TactileSnapshot


class SlowDetector:
    confidence = 0.1

    def detect(self, frame, frame_id=0, timestamp=None):
        time.sleep(0.05)
        return DetectionResult(frame_id=frame_id, timestamp=timestamp or time.time(), detections=(), elapsed_ms=50.0)


class FakeQuality:
    median_paper_error_mm = 0.1


class FakeCalibration:
    marker_ids = (0, 1, 2, 3)
    quality = FakeQuality()

    def __init__(self):
        self.saved_to = None

    def save_json(self, path):
        self.saved_to = path


class FakeCalibrator:
    def __init__(self):
        self.calibration = FakeCalibration()

    def calibrate(self, _image):
        return self.calibration


def test_vision_worker_replaces_pending_detection_tasks(tmp_path):
    worker = VisionWorker(SheetConfig(), tmp_path / "calibration.json", SlowDetector())
    worker.start()
    try:
        for frame_id in range(1, 8):
            worker.request_detection(CameraFrame(frame_id, np.zeros((8, 8, 3), dtype=np.uint8), time.time()))
        deadline = time.time() + 1.5
        snapshot = worker.snapshot()
        while time.time() < deadline:
            snapshot = worker.snapshot()
            if snapshot.detection is not None and snapshot.detection.frame_id == 7 and not snapshot.busy:
                break
            time.sleep(0.02)
        assert snapshot.detection is not None
        assert snapshot.detection.frame_id == 7
        assert snapshot.detection_frame is not None
        assert snapshot.detection_frame.frame_id == 7
    finally:
        worker.stop()


def test_camera_source_store_round_trips(tmp_path):
    source_store = tmp_path / "camera_source.txt"

    assert load_camera_source("0", source_store) == "0"
    saved = save_camera_source("http://example.test/video", source_store)

    assert saved == source_store
    assert load_camera_source("0", source_store) == "http://example.test/video"


def test_source_editor_handles_typing_commit_and_cancel():
    editor = SourceEditor("0")
    editor.begin()
    assert editor.handle_key(ord("1")) is None
    assert editor.text == "01"
    assert editor.handle_key(8) is None
    assert editor.text == "0"
    assert editor.handle_key(13) == "commit"
    assert editor.finish() == "0"

    editor.begin()
    editor.handle_key(ord("x"))
    assert editor.text == "0x"
    assert editor.handle_key(27) == "cancel"
    assert editor.text == "0"
    assert not editor.editing


def test_vision_worker_auto_calibration_updates_memory_without_saving(tmp_path):
    worker = VisionWorker(SheetConfig(), tmp_path / "calibration.json", SlowDetector())
    fake_calibrator = FakeCalibrator()
    worker.calibrator = fake_calibrator
    worker.start()
    try:
        assert worker.request_auto_calibration(CameraFrame(1, np.zeros((8, 8, 3), dtype=np.uint8), time.time()))
        deadline = time.time() + 1.0
        snapshot = worker.snapshot()
        while time.time() < deadline:
            snapshot = worker.snapshot()
            if snapshot.calibration is fake_calibrator.calibration and not snapshot.busy:
                break
            time.sleep(0.02)
        assert snapshot.calibration is fake_calibrator.calibration
        assert fake_calibrator.calibration.saved_to is None

        worker.request_calibration(CameraFrame(2, np.zeros((8, 8, 3), dtype=np.uint8), time.time()), save=True)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if fake_calibrator.calibration.saved_to == tmp_path / "calibration.json":
                break
            time.sleep(0.02)
        assert fake_calibrator.calibration.saved_to == tmp_path / "calibration.json"
    finally:
        worker.stop()


def test_build_sample_label_includes_tactile_payload():
    calibration = PaperCalibration(
        image_to_paper=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        paper_to_image=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        marker_ids=(0, 1, 2, 3),
        dictionary="DICT_4X4_50",
    )
    detection = DetectionResult(
        frame_id=3,
        timestamp=123.0,
        detections=(BlockDetection((1, 2, 5, 6), (3.0, 4.0), 0.9, 0, "block", (3.0, 4.0)),),
        elapsed_ms=2.0,
    )
    top5 = np.arange(NUM_TAXELS, dtype=np.float32) / float(NUM_TAXELS)
    tactile = TactileSnapshot(
        timestamp=456.0,
        port="SIMULATOR",
        hardware_online=False,
        status="simulation",
        error=None,
        tared=False,
        processed=np.zeros(NUM_TAXELS, dtype=np.float32),
        top5_normalized=top5,
        top5_raw_average=np.full(NUM_TAXELS, 100.0, dtype=np.float32),
        recent_raw_frames=(np.full(NUM_TAXELS, 100, dtype=np.uint16),),
    )

    label = build_sample_label(SheetConfig(), calibration, detection, tactile)

    assert label["schema_version"] == "tactile_pose_v1"
    assert label["input"]["frame"] == "taxel_center_v1"
    assert len(label["input"]["values"]) == NUM_TAXELS
    assert label["input"]["values"][0] == float(top5[15])
    assert label["target"]["pose"]["available"] is True
    assert label["target"]["pose"]["source"] == "bbox_homography"
    assert label["quality"]["tactile_available"] is True
    assert "calibration" not in label
    assert "sensor_data" not in label


def test_sensor_payload_without_tactile_is_unavailable():
    payload = sensor_payload_from_tactile(None)

    assert payload["available"] is False
    assert payload["top5_normalized"] is None
    assert payload["recent_raw_frames"] == []


def test_app_retains_and_clears_last_block_detection():
    app = DataCollectionApp.__new__(DataCollectionApp)
    app.last_block_detection = None
    detection = DetectionResult(
        frame_id=1,
        timestamp=1.0,
        detections=(BlockDetection((1, 2, 5, 6), (3.0, 4.0), 0.9, 0, "block", (3.0, 4.0)),),
        elapsed_ms=1.0,
    )
    missing = DetectionResult(frame_id=2, timestamp=2.0, detections=(), elapsed_ms=1.0)

    assert app._display_detection(detection) is detection
    assert app.last_block_detection is detection
    assert app._display_detection(missing) is detection
    assert app._using_retained_detection(missing, detection)
    app._clear_last_block_detection()
    assert app.last_block_detection is None


def test_build_sample_label_marks_retained_block_position():
    calibration = PaperCalibration(
        image_to_paper=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        paper_to_image=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        marker_ids=(0, 1, 2, 3),
        dictionary="DICT_4X4_50",
    )
    detection = DetectionResult(
        frame_id=9,
        timestamp=10.0,
        detections=(BlockDetection((1, 2, 5, 6), (3.0, 4.0), 0.9, 0, "block", (3.0, 4.0)),),
        elapsed_ms=1.0,
    )

    label = build_sample_label(SheetConfig(), calibration, detection, retained_block=True)

    assert label["quality"]["retained_block_position"] is True
