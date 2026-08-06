from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tactile_vision.detector import BlockDetection, BlockDetector


class FakeBox:
    def __init__(self, conf: float, xyxy: tuple[int, int, int, int], cls: int = 0):
        self.conf = [conf]
        self.cls = [cls]
        self.xyxy = [np.array(xyxy, dtype=float)]


class FakeBoxes:
    def __init__(self, boxes):
        self._boxes = boxes

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = FakeBoxes(boxes)
        self.names = {0: "cube"}


class FakeModel:
    def predict(self, source, **kwargs):
        assert kwargs["conf"] == 0.5
        assert source.shape == (20, 20, 3)
        return [
            FakeResult(
                [
                    FakeBox(0.4, (1, 1, 4, 4)),
                    FakeBox(0.9, (2, 2, 8, 8)),
                    FakeBox(0.7, (10, 10, 14, 14)),
                ]
            )
        ]


def test_detect_blocks_filters_and_sorts_by_confidence(tmp_path):
    detector = BlockDetector(weights_path=tmp_path / "unused.pt", calibration_path=tmp_path / "missing.json", confidence=0.5)
    detector._model = FakeModel()

    detections = detector.detect_blocks(np.zeros((20, 20, 3), dtype=np.uint8))

    assert [round(detection.confidence, 1) for detection in detections] == [0.9, 0.7]
    assert detector.detect_best_block(np.zeros((20, 20, 3), dtype=np.uint8)).confidence == 0.9


def test_render_frame_draws_multiple_detections(tmp_path):
    detector = BlockDetector(weights_path=tmp_path / "unused.pt", calibration_path=tmp_path / "missing.json", confidence=0.5)
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    detections = [
        BlockDetection((2, 2, 8, 8), (5.0, 5.0), 0.9, 0, "cube"),
        BlockDetection((12, 12, 18, 18), (15.0, 15.0), 0.7, 0, "cube"),
    ]

    rendered = detector.render_frame(frame, detections)

    assert rendered.shape == frame.shape
    assert int(rendered.sum()) > 0
