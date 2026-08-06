import argparse
import time

import numpy as np

from block_position_prediction.data_collection_manual.app import (
    DataCollectionApp,
    SourceEditor,
    VisionWorker,
    load_camera_source,
    save_camera_source,
    sensor_payload_from_tactile,
)
from block_position_prediction.data_collection_manual.aruco import PaperCalibration
from block_position_prediction.data_collection_manual.camera import CameraFrame
from block_position_prediction.data_collection_manual.geometry import SheetConfig
from block_position_prediction.data_collection_manual.labels import (
    CapturedSample,
    build_manual_tactile_pose_label,
    create_manual_annotation,
)
from block_position_prediction.data_collection_manual.tactile import NUM_TAXELS, TactileSnapshot
from block_position_prediction.data_collection_manual.writer import DatasetStore


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
        time.sleep(0.02)
        return self.calibration


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
    worker = VisionWorker(SheetConfig(), tmp_path / "calibration.json")
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


def test_sensor_payload_without_tactile_is_unavailable():
    payload = sensor_payload_from_tactile(None)

    assert payload["available"] is False
    assert payload["top5_normalized"] is None
    assert payload["recent_raw_frames"] == []


def test_app_browses_saved_samples_back_to_live(tmp_path):
    app = DataCollectionApp.__new__(DataCollectionApp)
    app.dataset = DatasetStore.create_new(tmp_path)
    app.dataset.save_new(np.zeros((8, 8, 3), dtype=np.uint8), {"sample_id": "000001", "image_path": "images/000001.jpg"})
    app.mode = "live"
    app.saved_index = None
    app.saved_annotation_override = None
    app.user_status = ""

    app._go_previous()
    assert app.mode == "saved"
    assert app.saved_index == 0

    app._go_next()
    assert app.mode == "live"
    assert app.saved_index is None


def test_app_saves_draft_and_returns_to_live(tmp_path):
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)
    app = DataCollectionApp.__new__(DataCollectionApp)
    app.dataset = DatasetStore.create_new(tmp_path)
    app.sheet_config = config
    app.args = argparse.Namespace(frame_width=1280, frame_height=720, tactile_baud=115200)
    app.connected_source_text = "0"
    app.mode = "draft"
    app.saved_index = None
    app.saved_annotation_override = None
    app._dragging_annotation = False
    app._saved_image_cache = None
    app.user_status = ""
    app.draft = CapturedSample(
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        frame_id=1,
        timestamp=123.0,
        tactile=_tactile(np.zeros(NUM_TAXELS, dtype=np.float32)),
        calibration=calibration,
    )
    app.draft_annotation = create_manual_annotation(config, calibration, center_px=(30.0, 16.0), direction_px=(40.0, 16.0))

    app._save_draft()

    assert app.mode == "live"
    assert len(app.dataset.labels) == 1
    assert app.dataset.labels[0]["quality"]["label_source"] == "manual_aruco"
    assert app.dataset.labels[0]["sample_id"] == "000001"


def test_app_keyboard_tuning_moves_and_rotates_draft_annotation(tmp_path):
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)
    app = DataCollectionApp.__new__(DataCollectionApp)
    app.dataset = DatasetStore.create_new(tmp_path)
    app.sheet_config = config
    app.mode = "draft"
    app.saved_index = None
    app.saved_annotation_override = None
    app.user_status = ""
    app.draft = CapturedSample(
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        frame_id=1,
        timestamp=123.0,
        tactile=None,
        calibration=calibration,
    )
    app.draft_annotation = create_manual_annotation(config, calibration, center_px=(30.0, 16.0), direction_px=(40.0, 16.0))

    app._nudge_active_annotation(0.1, 0.0)
    moved = app.draft_annotation
    assert moved is not None
    assert abs(moved.center_taxel[0] - 6.6) < 1e-8

    before_yaw = moved.yaw_mod90_rad
    app._rotate_active_annotation(0.1)
    rotated = app.draft_annotation
    assert rotated is not None
    assert abs(rotated.yaw_mod90_rad - before_yaw - 0.1) < 1e-8


def test_app_disables_saved_annotation_edit_without_sample_calibration(tmp_path):
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)
    annotation = create_manual_annotation(config, calibration, center_px=(30.0, 16.0), direction_px=(40.0, 16.0))
    label = build_manual_tactile_pose_label(config, calibration, annotation, _tactile(np.zeros(NUM_TAXELS)))
    label.pop("calibration")
    app = DataCollectionApp.__new__(DataCollectionApp)
    app.dataset = DatasetStore.create_new(tmp_path)
    app.dataset.save_new(np.zeros((20, 20, 3), dtype=np.uint8), label)
    app.sheet_config = config
    app.mode = "saved"
    app.saved_index = 0
    app.saved_annotation_override = None
    app.user_status = ""

    app._nudge_active_annotation(0.1, 0.0)

    assert app.saved_annotation_override is None
    assert app.user_status == "edit disabled: saved sample has no calibration snapshot"


def test_app_deletes_saved_sample_and_keeps_browsing(tmp_path):
    app = DataCollectionApp.__new__(DataCollectionApp)
    app.dataset = DatasetStore.create_new(tmp_path)
    app.dataset.save_new(np.zeros((8, 8, 3), dtype=np.uint8), {"frame_id": 1})
    app.dataset.save_new(np.zeros((8, 8, 3), dtype=np.uint8), {"frame_id": 2})
    app.mode = "saved"
    app.saved_index = 0
    app.saved_annotation_override = object()
    app._saved_image_cache = (0, np.zeros((8, 8, 3), dtype=np.uint8))
    app.user_status = ""

    app._delete_saved_sample()

    assert len(app.dataset.labels) == 1
    assert app.dataset.labels[0]["sample_id"] == "000002"
    assert app.mode == "saved"
    assert app.saved_index == 0
    assert app.saved_annotation_override is None
    assert app._saved_image_cache is None


def _calibration_at_sensor_origin(config: SheetConfig) -> PaperCalibration:
    ox, oy = config.sensor_origin_mm
    return PaperCalibration(
        image_to_paper=((1, 0, ox), (0, 1, oy), (0, 0, 1)),
        paper_to_image=((1, 0, -ox), (0, 1, -oy), (0, 0, 1)),
        marker_ids=(0, 1, 2, 3),
        dictionary="DICT_4X4_50",
    )


def _tactile(values):
    return TactileSnapshot(
        timestamp=1.0,
        port="SIMULATOR",
        hardware_online=False,
        status="simulation",
        error=None,
        tared=False,
        processed=None,
        top5_normalized=np.asarray(values, dtype=np.float32),
        top5_raw_average=np.zeros(NUM_TAXELS, dtype=np.float32),
        recent_raw_frames=(),
    )
