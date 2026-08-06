import numpy as np

from block_position_prediction.data_collection.detector import BlockDetector


class FakeBox:
    def __init__(self, conf, xyxy, cls=0):
        self.conf = [conf]
        self.cls = [cls]
        self.xyxy = [np.array(xyxy, dtype=float)]


class FakeBoxes:
    def __init__(self, boxes):
        self.boxes = boxes

    def __iter__(self):
        return iter(self.boxes)

    def __len__(self):
        return len(self.boxes)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = FakeBoxes(boxes)
        self.names = {0: "block"}


class FakeModel:
    def predict(self, source, **kwargs):
        assert kwargs["conf"] == 0.5
        assert source.shape == (20, 20, 3)
        return [FakeResult([FakeBox(0.4, (1, 1, 3, 3)), FakeBox(0.9, (2, 2, 8, 8)), FakeBox(0.7, (10, 10, 14, 14))])]


def test_detector_filters_sorts_and_uses_center_anchor():
    detector = BlockDetector(confidence=0.5, model=FakeModel())

    detections = detector.detect_blocks(np.zeros((20, 20, 3), dtype=np.uint8))

    assert [round(detection.confidence, 1) for detection in detections] == [0.9, 0.7]
    assert detections[0].anchor_px == (5.0, 5.0)
    assert detections[0].center_px == (5.0, 5.0)

