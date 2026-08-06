import json

import numpy as np

from block_position_prediction.data_collection.writer import DatasetWriter


def test_writer_saves_image_and_jsonl(tmp_path):
    writer = DatasetWriter(tmp_path)
    metadata_path = writer.write_metadata({"schema_version": "test"})
    writer.start()
    assert writer.enqueue(np.zeros((12, 12, 3), dtype=np.uint8), {"frame_id": 7})
    writer._queue.join()
    writer.stop()

    assert metadata_path.exists()
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["schema_version"] == "test"
    labels = writer.labels_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(labels) == 1
    payload = json.loads(labels[0])
    assert payload["frame_id"] == 7
    assert payload["sample_id"] == "000001"
    assert payload["image_path"] == "images/000001.jpg"
    assert (writer.session_dir / payload["image_path"]).exists()
