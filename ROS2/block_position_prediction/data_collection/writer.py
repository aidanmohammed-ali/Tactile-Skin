from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SampleWrite:
    image: Any
    label: Mapping[str, Any]


class DatasetWriter:
    """Background image + JSONL writer for captured samples."""

    def __init__(self, root: str | Path = "block_position_prediction/data_collection/runs") -> None:
        self.root = Path(root)
        self.session_dir = self.root / time.strftime("%Y%m%d_%H%M%S")
        self.images_dir = self.session_dir / "images"
        self.labels_path = self.session_dir / "labels.jsonl"
        self.metadata_path = self.session_dir / "metadata.json"
        self._queue: queue.Queue[SampleWrite | None] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._counter = 0
        self.last_error: str | None = None
        self.last_saved: str | None = None
        self._metadata: Mapping[str, Any] | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._writer_loop, name="data_collection_writer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=2.0)
        self._thread = None

    def enqueue(self, image: Any, label: Mapping[str, Any]) -> bool:
        if self._thread is None:
            self.start()
        try:
            self._queue.put_nowait(SampleWrite(image.copy(), dict(label)))
            return True
        except queue.Full:
            self.last_error = "writer queue is full"
            return False

    def write_metadata(self, metadata: Mapping[str, Any]) -> Path:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = dict(metadata)
        with self.metadata_path.open("w", encoding="utf-8") as file:
            json.dump(self._metadata, file, indent=2, sort_keys=True)
            file.write("\n")
        return self.metadata_path

    def _writer_loop(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                break
            try:
                self._write_sample(item.image, item.label)
            except Exception as exc:
                self.last_error = str(exc)
            finally:
                self._queue.task_done()

    def _write_sample(self, image: Any, label: Mapping[str, Any]) -> None:
        import cv2

        self._counter += 1
        sample_id = f"{self._counter:06d}"
        filename = f"{sample_id}.jpg"
        image_path = self.images_dir / filename
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"could not encode {image_path}")
        encoded.tofile(str(image_path))
        payload = dict(label)
        if not payload.get("sample_id"):
            payload["sample_id"] = sample_id
        payload["image_path"] = str(image_path.relative_to(self.session_dir))
        with self.labels_path.open("a", encoding="utf-8") as file:
            json.dump(payload, file, sort_keys=True)
            file.write("\n")
        self.last_saved = str(image_path)
        self.last_error = None
