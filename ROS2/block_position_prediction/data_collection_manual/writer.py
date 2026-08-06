from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DatasetSample:
    index: int
    sample_id: str
    image_path: str
    label: dict[str, Any]


class DatasetStore:
    """Synchronous dataset store that can append and overwrite labels."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        self.images_dir = self.session_dir / "images"
        self.labels_path = self.session_dir / "labels.jsonl"
        self.metadata_path = self.session_dir / "metadata.json"
        self.labels: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self.last_error: str | None = None
        self.last_saved: str | None = None
        self.load()

    @classmethod
    def create_new(cls, root: str | Path) -> "DatasetStore":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = root / stamp
        suffix = 1
        while session_dir.exists():
            session_dir = root / f"{stamp}_{suffix:02d}"
            suffix += 1
        session_dir.mkdir(parents=True, exist_ok=False)
        (session_dir / "images").mkdir(parents=True, exist_ok=True)
        return cls(session_dir)

    @classmethod
    def resume(cls, path: str | Path) -> "DatasetStore":
        candidate = Path(path)
        if candidate.name == "labels.jsonl":
            candidate = candidate.parent
        return cls(candidate)

    def load(self) -> None:
        self.labels = []
        if self.labels_path.exists():
            with self.labels_path.open("r", encoding="utf-8") as file:
                for line in file:
                    text = line.strip()
                    if text:
                        self.labels.append(json.loads(text))
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as file:
                self.metadata = json.load(file)
        else:
            self.metadata = {}

    def list_samples(self) -> tuple[DatasetSample, ...]:
        samples: list[DatasetSample] = []
        for index, label in enumerate(self.labels):
            samples.append(
                DatasetSample(
                    index=index,
                    sample_id=str(label.get("sample_id") or f"{index + 1:06d}"),
                    image_path=str(label.get("image_path") or ""),
                    label=dict(label),
                )
            )
        return tuple(samples)

    def write_metadata(self, metadata: Mapping[str, Any]) -> Path:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        tmp_path = self.metadata_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, indent=2, sort_keys=True)
            file.write("\n")
        tmp_path.replace(self.metadata_path)
        return self.metadata_path

    def save_new(self, image: Any, label: Mapping[str, Any]) -> int:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        sample_id = self._next_sample_id()
        image_path = self.images_dir / f"{sample_id}.jpg"
        self._write_image(image_path, image)

        payload = dict(label)
        payload["sample_id"] = sample_id
        payload["image_path"] = str(image_path.relative_to(self.session_dir))
        self.labels.append(payload)
        self._write_labels_atomic()
        self.last_saved = str(image_path)
        self.last_error = None
        return len(self.labels) - 1

    def update_label(self, index: int, label: Mapping[str, Any]) -> None:
        if index < 0 or index >= len(self.labels):
            raise IndexError(f"sample index out of range: {index}")
        old = self.labels[index]
        payload = dict(label)
        payload["sample_id"] = str(old.get("sample_id") or f"{index + 1:06d}")
        payload["image_path"] = str(old.get("image_path") or payload.get("image_path") or "")
        self.labels[index] = payload
        self._write_labels_atomic()
        self.last_saved = payload["image_path"]
        self.last_error = None

    def delete_sample(self, index: int, delete_image: bool = False) -> dict[str, Any]:
        if index < 0 or index >= len(self.labels):
            raise IndexError(f"sample index out of range: {index}")
        label = self.labels.pop(index)
        image_path_text = str(label.get("image_path") or "")
        if delete_image and image_path_text:
            try:
                (self.session_dir / image_path_text).unlink()
            except FileNotFoundError:
                pass
        self._write_labels_atomic()
        self.last_saved = f"deleted {label.get('sample_id') or index + 1}"
        self.last_error = None
        return dict(label)

    def load_image(self, index: int) -> Any:
        import cv2
        import numpy as np

        if index < 0 or index >= len(self.labels):
            raise IndexError(f"sample index out of range: {index}")
        image_path = self.session_dir / str(self.labels[index].get("image_path") or "")
        raw = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read image: {image_path}")
        return image

    def _next_sample_id(self) -> str:
        largest = 0
        for label in self.labels:
            text = str(label.get("sample_id") or "")
            if text.isdigit():
                largest = max(largest, int(text))
        return f"{largest + 1:06d}"

    def _write_labels_atomic(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.labels_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            for label in self.labels:
                json.dump(label, file, sort_keys=True)
                file.write("\n")
        tmp_path.replace(self.labels_path)

    def _write_image(self, image_path: Path, image: Any) -> None:
        import cv2

        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"could not encode {image_path}")
        encoded.tofile(str(image_path))


def list_recent_runs(root: str | Path, limit: int = 12) -> tuple[Path, ...]:
    root = Path(root)
    if not root.exists():
        return ()
    runs = [path for path in root.iterdir() if path.is_dir()]
    runs.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    return tuple(runs[: max(0, int(limit))])
