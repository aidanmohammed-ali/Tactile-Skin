import json

import numpy as np

from block_position_prediction.data_collection_manual.writer import DatasetStore, list_recent_runs


def test_dataset_store_saves_and_loads_new_sample(tmp_path):
    store = DatasetStore.create_new(tmp_path)
    store.write_metadata({"schema_version": "manual_collection_v1"})
    index = store.save_new(np.zeros((12, 12, 3), dtype=np.uint8), {"frame_id": 7})

    assert index == 0
    assert json.loads(store.metadata_path.read_text(encoding="utf-8"))["schema_version"] == "manual_collection_v1"
    labels = store.labels_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(labels) == 1
    payload = json.loads(labels[0])
    assert payload["frame_id"] == 7
    assert payload["sample_id"] == "000001"
    assert payload["image_path"] == "images/000001.jpg"
    assert (store.session_dir / payload["image_path"]).exists()

    resumed = DatasetStore.resume(store.session_dir)
    assert len(resumed.labels) == 1
    assert resumed.load_image(0).shape == (12, 12, 3)


def test_dataset_store_overwrites_label_without_changing_image_path(tmp_path):
    store = DatasetStore.create_new(tmp_path)
    store.save_new(np.zeros((12, 12, 3), dtype=np.uint8), {"frame_id": 1})
    old_path = store.labels[0]["image_path"]

    store.update_label(0, {"frame_id": 2, "sample_id": "wrong", "image_path": "wrong.jpg"})

    assert store.labels[0]["frame_id"] == 2
    assert store.labels[0]["sample_id"] == "000001"
    assert store.labels[0]["image_path"] == old_path


def test_dataset_store_deletes_label_without_deleting_image_by_default(tmp_path):
    store = DatasetStore.create_new(tmp_path)
    store.save_new(np.zeros((12, 12, 3), dtype=np.uint8), {"frame_id": 1})
    store.save_new(np.zeros((12, 12, 3), dtype=np.uint8), {"frame_id": 2})
    image_path = store.session_dir / store.labels[0]["image_path"]

    deleted = store.delete_sample(0)

    assert deleted["sample_id"] == "000001"
    assert len(store.labels) == 1
    assert store.labels[0]["sample_id"] == "000002"
    assert image_path.exists()


def test_list_recent_runs_orders_by_mtime(tmp_path):
    first = DatasetStore.create_new(tmp_path)
    second = DatasetStore.create_new(tmp_path)

    runs = list_recent_runs(tmp_path)

    assert runs[0] == second.session_dir
    assert first.session_dir in runs
