from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

Point2 = tuple[float, float]
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class CalibrationQuality:
    point_count: int = 0
    inlier_count: int = 0
    marker_point_count: int = 0
    hole_point_count: int = 0
    mean_grid_error: float | None = None
    median_grid_error: float | None = None
    max_grid_error: float | None = None
    mean_pixel_error: float | None = None
    median_pixel_error: float | None = None
    max_pixel_error: float | None = None


@dataclass(frozen=True)
class BoardCalibration:
    """Saved homography for fast runtime mapping."""

    image_to_board: Matrix3
    board_to_image: Matrix3
    rows: int = 12
    cols: int = 17
    aruco_dictionary: str = "DICT_4X4_50"
    marker_ids: tuple[int, ...] = ()
    quality: CalibrationQuality = field(default_factory=CalibrationQuality)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_to_board", _as_matrix3(self.image_to_board))
        object.__setattr__(self, "board_to_image", _as_matrix3(self.board_to_image))
        object.__setattr__(self, "marker_ids", tuple(int(value) for value in self.marker_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def pixel_to_board(self, u: float, v: float) -> Point2:
        return pixel_to_board(u, v, self)

    def board_to_pixel(self, col: float, row: float) -> Point2:
        return board_to_pixel(col, row, self)

    def save_json(self, path: str | Path) -> None:
        save_calibration(path, self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_to_board": [list(row) for row in self.image_to_board],
            "board_to_image": [list(row) for row in self.board_to_image],
            "rows": self.rows,
            "cols": self.cols,
            "aruco_dictionary": self.aruco_dictionary,
            "marker_ids": list(self.marker_ids),
            "quality": asdict(self.quality),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoardCalibration":
        quality_data = data.get("quality") or {}
        quality = quality_data if isinstance(quality_data, CalibrationQuality) else CalibrationQuality(**quality_data)
        return cls(
            image_to_board=data["image_to_board"],
            board_to_image=data["board_to_image"],
            rows=int(data.get("rows", 12)),
            cols=int(data.get("cols", 17)),
            aruco_dictionary=str(data.get("aruco_dictionary", "DICT_4X4_50")),
            marker_ids=tuple(data.get("marker_ids") or ()),
            quality=quality,
            metadata=data.get("metadata") or {},
        )


def load_calibration(path: str | Path) -> BoardCalibration:
    with Path(path).open("r", encoding="utf-8") as file:
        return BoardCalibration.from_dict(json.load(file))


def save_calibration(path: str | Path, calibration: BoardCalibration) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(calibration.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")


def pixel_to_board(u: float, v: float, calibration: BoardCalibration) -> Point2:
    return _apply_homography(calibration.image_to_board, float(u), float(v))


def board_to_pixel(col: float, row: float, calibration: BoardCalibration) -> Point2:
    return _apply_homography(calibration.board_to_image, float(col), float(row))


def _apply_homography(matrix: Matrix3, x: float, y: float) -> Point2:
    denom = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(denom) < 1e-12:
        raise ZeroDivisionError("homogeneous point has zero scale")
    return (
        (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denom,
        (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denom,
    )


def _as_matrix3(matrix: Sequence[Sequence[float]]) -> Matrix3:
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("homography must be a 3x3 matrix")
    return rows  # type: ignore[return-value]
