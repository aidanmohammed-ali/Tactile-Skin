from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

Point2 = tuple[float, float]


@dataclass(frozen=True)
class BoardGeometry:
    """Hole lattice in board coordinates.

    Board coordinates are continuous grid units. The top-left hole is `(0, 0)`;
    the bottom-right hole is `(cols - 1, rows - 1)`.
    """

    rows: int = 12
    cols: int = 17

    def __post_init__(self) -> None:
        if self.rows < 2 or self.cols < 2:
            raise ValueError("rows and cols must both be at least 2")

    @property
    def max_row(self) -> int:
        return self.rows - 1

    @property
    def max_col(self) -> int:
        return self.cols - 1


@dataclass(frozen=True)
class MarkerSpec:
    """One known marker and its four board-space corners.

    Corner order follows OpenCV ArUco order: top-left, top-right, bottom-right,
    bottom-left in the marker's canonical orientation.
    """

    marker_id: int
    board_corners: tuple[Point2, Point2, Point2, Point2]

    def __post_init__(self) -> None:
        if len(self.board_corners) != 4:
            raise ValueError("a marker must have exactly four board corners")


@dataclass(frozen=True)
class HoleRefineConfig:
    """Hole detection options used only during calibration/refinement."""

    enabled: bool = True
    min_area_px: float = 25.0
    max_area_px: float = 1500.0
    max_area_fraction: float = 0.002
    adaptive_block_size: int = 31
    adaptive_c: float = -5.0
    min_aspect_ratio: float = 0.45
    max_aspect_ratio: float = 2.2
    min_fill_ratio: float = 0.18
    assignment_max_grid_error: float = 0.35
    board_margin_grid: float = 0.55
    min_holes: int = 12

    def __post_init__(self) -> None:
        if self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size must be odd and at least 3")


@dataclass(frozen=True)
class ArucoBoardConfig:
    """Physical board and marker layout.

    Defaults use four `DICT_4X4_50` markers with IDs 0, 1, 2, 3 placed just
    inside the hole grid corners: top-left, top-right, bottom-right, bottom-left.
    """

    geometry: BoardGeometry = field(default_factory=BoardGeometry)
    hole_pitch_mm: float = 25.0
    aruco_dictionary: str = "DICT_4X4_50"
    marker_specs: tuple[MarkerSpec, ...] = ()
    marker_size_grid: float = 1.6
    marker_margin_grid: float = -1.40
    marker_ids: tuple[int, int, int, int] = (0, 1, 2, 3)
    refine_holes: HoleRefineConfig = field(default_factory=HoleRefineConfig)
    ransac_threshold_grid: float = 0.18

    @property
    def marker_size_mm(self) -> float:
        return self.marker_size_grid * self.hole_pitch_mm

    @property
    def marker_margin_mm(self) -> float:
        return self.marker_margin_grid * self.hole_pitch_mm

    @property
    def marker_inner_offset_mm(self) -> float:
        return -self.marker_margin_mm

    def resolved_marker_specs(self) -> tuple[MarkerSpec, ...]:
        if self.marker_specs:
            return self.marker_specs
        return default_marker_specs(
            self.geometry,
            marker_size_grid=self.marker_size_grid,
            marker_margin_grid=self.marker_margin_grid,
            marker_ids=self.marker_ids,
        )


def default_marker_specs(
    geometry: BoardGeometry,
    marker_size_grid: float = 1.6,
    marker_margin_grid: float = -1.40,
    marker_ids: Sequence[int] = (0, 1, 2, 3),
) -> tuple[MarkerSpec, ...]:
    """Default four-corner layout in board grid units."""

    if len(marker_ids) != 4:
        raise ValueError("default_marker_specs requires exactly four marker IDs")
    s = float(marker_size_grid)
    m = float(marker_margin_grid)
    if s <= 0:
        raise ValueError("marker_size_grid must be positive")
    left = -m - s
    right = geometry.max_col + m
    top = -m - s
    bottom = geometry.max_row + m
    return (
        MarkerSpec(
            int(marker_ids[0]),
            ((left, top), (left + s, top), (left + s, top + s), (left, top + s)),
        ),
        MarkerSpec(
            int(marker_ids[1]),
            ((right, top), (right + s, top), (right + s, top + s), (right, top + s)),
        ),
        MarkerSpec(
            int(marker_ids[2]),
            (
                (right, bottom),
                (right + s, bottom),
                (right + s, bottom + s),
                (right, bottom + s),
            ),
        ),
        MarkerSpec(
            int(marker_ids[3]),
            (
                (left, bottom),
                (left + s, bottom),
                (left + s, bottom + s),
                (left, bottom + s),
            ),
        ),
    )


def marker_specs_from_centers(
    marker_centers: Mapping[int, Point2],
    marker_size_grid: float,
) -> tuple[MarkerSpec, ...]:
    """Create marker specs when marker centers are measured in grid units."""

    half = float(marker_size_grid) / 2.0
    specs: list[MarkerSpec] = []
    for marker_id, (cx, cy) in marker_centers.items():
        specs.append(
            MarkerSpec(
                int(marker_id),
                (
                    (float(cx) - half, float(cy) - half),
                    (float(cx) + half, float(cy) - half),
                    (float(cx) + half, float(cy) + half),
                    (float(cx) - half, float(cy) + half),
                ),
            )
        )
    return tuple(sorted(specs, key=lambda item: item.marker_id))
