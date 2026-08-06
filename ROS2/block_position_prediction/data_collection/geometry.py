from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


Point2 = tuple[float, float]
SENSOR_COORD_FRAME = "taxel_center_v1"
BLOCK_SIDE_CM = 2.4
BLOCK_SIDE_TAXEL = 6.0


@dataclass(frozen=True)
class SensorArrayConfig:
    """Physical sensor and taxel geometry in millimeters."""

    width_mm: float = 68.0
    height_mm: float = 40.0
    cols: int = 16
    rows: int = 8
    pitch_mm: float = 4.0
    left_margin_mm: float = 4.0
    top_margin_mm: float = 4.0

    def __post_init__(self) -> None:
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ValueError("sensor width and height must be positive")
        if self.cols < 1 or self.rows < 1:
            raise ValueError("sensor cols and rows must be positive")
        if self.pitch_mm <= 0:
            raise ValueError("sensor pitch must be positive")

    @property
    def right_margin_mm(self) -> float:
        return self.width_mm - self.left_margin_mm - self.pitch_mm * (self.cols - 1)

    @property
    def bottom_margin_mm(self) -> float:
        return self.height_mm - self.top_margin_mm - self.pitch_mm * (self.rows - 1)

    def taxel_center_mm(self, col: float, row: float) -> Point2:
        return (
            self.left_margin_mm + float(col) * self.pitch_mm,
            self.top_margin_mm + float(row) * self.pitch_mm,
        )

    def sensor_mm_to_array(self, x_mm: float, y_mm: float) -> Point2:
        return (
            (float(x_mm) - self.left_margin_mm) / self.pitch_mm,
            (float(y_mm) - self.top_margin_mm) / self.pitch_mm,
        )

    def sensor_mm_to_taxel_center(self, x_mm: float, y_mm: float) -> Point2:
        return self.sensor_mm_to_array(x_mm, y_mm)

    def taxel_center_to_sensor_mm(self, x_taxel: float, y_taxel: float) -> Point2:
        return self.taxel_center_mm(x_taxel, y_taxel)

    def sensor_mm_to_normalized(self, x_mm: float, y_mm: float) -> Point2:
        return (float(x_mm) / self.width_mm, float(y_mm) / self.height_mm)

    def taxel_center_to_normalized(self, x_taxel: float, y_taxel: float) -> Point2:
        x_denominator = max(1.0, float(self.cols - 1))
        y_denominator = max(1.0, float(self.rows - 1))
        return (float(x_taxel) / x_denominator, float(y_taxel) / y_denominator)

    def taxel_center_to_cm_from_taxel0(self, x_taxel: float, y_taxel: float) -> Point2:
        scale = self.pitch_mm / 10.0
        return (float(x_taxel) * scale, float(y_taxel) * scale)

    @property
    def taxel_sensor_bounds(self) -> tuple[float, float, float, float]:
        return (
            -self.left_margin_mm / self.pitch_mm,
            -self.top_margin_mm / self.pitch_mm,
            (self.width_mm - self.left_margin_mm) / self.pitch_mm,
            (self.height_mm - self.top_margin_mm) / self.pitch_mm,
        )

    def contains_mm(self, x_mm: float, y_mm: float) -> bool:
        return 0.0 <= float(x_mm) <= self.width_mm and 0.0 <= float(y_mm) <= self.height_mm

    def contains_taxel_sensor(self, x_taxel: float, y_taxel: float) -> bool:
        left, top, right, bottom = self.taxel_sensor_bounds
        return left <= float(x_taxel) <= right and top <= float(y_taxel) <= bottom

    def footprint_fully_inside_sensor(self, corners_taxel: tuple[Point2, ...]) -> bool:
        return all(self.contains_taxel_sensor(x, y) for x, y in corners_taxel)

    def array_contains(self, col: float, row: float) -> bool:
        return 0.0 <= float(col) <= self.cols - 1 and 0.0 <= float(row) <= self.rows - 1


@dataclass(frozen=True)
class MarkerLayout:
    """Four ArUco markers around the reserved sensor rectangle."""

    marker_size_mm: float = 20.0
    quiet_zone_mm: float = 4.0
    gap_from_sensor_mm: float = 10.0
    ids: tuple[int, int, int, int] = (0, 1, 2, 3)
    dictionary: str = "DICT_4X4_50"

    def __post_init__(self) -> None:
        if self.marker_size_mm <= 0:
            raise ValueError("marker size must be positive")
        if self.quiet_zone_mm < 0:
            raise ValueError("quiet zone must be non-negative")
        if len(self.ids) != 4:
            raise ValueError("exactly four marker IDs are required")


@dataclass(frozen=True)
class SheetConfig:
    """A4 sheet geometry with a centered reserved tactile sensor rectangle."""

    paper_width_mm: float = 297.0
    paper_height_mm: float = 210.0
    sensor: SensorArrayConfig = field(default_factory=SensorArrayConfig)
    marker_layout: MarkerLayout = field(default_factory=MarkerLayout)

    def __post_init__(self) -> None:
        if self.paper_width_mm <= 0 or self.paper_height_mm <= 0:
            raise ValueError("paper width and height must be positive")
        if self.sensor.width_mm >= self.paper_width_mm or self.sensor.height_mm >= self.paper_height_mm:
            raise ValueError("sensor must fit inside the paper")

    @property
    def sensor_origin_mm(self) -> Point2:
        return (
            (self.paper_width_mm - self.sensor.width_mm) / 2.0,
            (self.paper_height_mm - self.sensor.height_mm) / 2.0,
        )

    @property
    def sensor_rect_mm(self) -> tuple[float, float, float, float]:
        x, y = self.sensor_origin_mm
        return (x, y, x + self.sensor.width_mm, y + self.sensor.height_mm)

    def paper_to_sensor_mm(self, x_mm: float, y_mm: float) -> Point2:
        sx, sy = self.sensor_origin_mm
        return (float(x_mm) - sx, float(y_mm) - sy)

    def sensor_to_paper_mm(self, x_mm: float, y_mm: float) -> Point2:
        sx, sy = self.sensor_origin_mm
        return (float(x_mm) + sx, float(y_mm) + sy)

    def marker_specs(self) -> tuple["MarkerSpec", ...]:
        x1, y1, x2, y2 = self.sensor_rect_mm
        size = self.marker_layout.marker_size_mm
        gap = self.marker_layout.gap_from_sensor_mm
        centers = (
            (x1 - gap - size / 2.0, y1 - gap - size / 2.0),
            (x2 + gap + size / 2.0, y1 - gap - size / 2.0),
            (x2 + gap + size / 2.0, y2 + gap + size / 2.0),
            (x1 - gap - size / 2.0, y2 + gap + size / 2.0),
        )
        specs: list[MarkerSpec] = []
        for marker_id, center in zip(self.marker_layout.ids, centers):
            specs.append(MarkerSpec.from_center(int(marker_id), center, size))
        return tuple(specs)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sensor_origin_mm"] = list(self.sensor_origin_mm)
        data["sensor_rect_mm"] = list(self.sensor_rect_mm)
        data["marker_specs"] = [spec.to_dict() for spec in self.marker_specs()]
        return data


@dataclass(frozen=True)
class MarkerSpec:
    """Marker corners in paper millimeter coordinates.

    Corner order follows OpenCV ArUco order: top-left, top-right, bottom-right,
    bottom-left in the marker's canonical orientation.
    """

    marker_id: int
    paper_corners_mm: tuple[Point2, Point2, Point2, Point2]

    @classmethod
    def from_center(cls, marker_id: int, center_mm: Point2, size_mm: float) -> "MarkerSpec":
        cx, cy = center_mm
        half = float(size_mm) / 2.0
        return cls(
            marker_id=int(marker_id),
            paper_corners_mm=(
                (cx - half, cy - half),
                (cx + half, cy - half),
                (cx + half, cy + half),
                (cx - half, cy + half),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker_id": self.marker_id,
            "paper_corners_mm": [list(point) for point in self.paper_corners_mm],
        }


@dataclass(frozen=True)
class PositionLabel:
    paper_mm: Point2
    sensor_mm: Point2
    sensor_normalized: Point2
    array_col_row: Point2
    inside_sensor: bool
    inside_array: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_mm": list(self.paper_mm),
            "sensor_mm": list(self.sensor_mm),
            "sensor_normalized": list(self.sensor_normalized),
            "array_col_row": list(self.array_col_row),
            "inside_sensor": self.inside_sensor,
            "inside_array": self.inside_array,
        }


def label_from_paper_position(config: SheetConfig, paper_x_mm: float, paper_y_mm: float) -> PositionLabel:
    sensor_x, sensor_y = config.paper_to_sensor_mm(paper_x_mm, paper_y_mm)
    col, row = config.sensor.sensor_mm_to_array(sensor_x, sensor_y)
    return PositionLabel(
        paper_mm=(float(paper_x_mm), float(paper_y_mm)),
        sensor_mm=(sensor_x, sensor_y),
        sensor_normalized=config.sensor.sensor_mm_to_normalized(sensor_x, sensor_y),
        array_col_row=(col, row),
        inside_sensor=config.sensor.contains_mm(sensor_x, sensor_y),
        inside_array=config.sensor.array_contains(col, row),
    )


def normalize_yaw_mod90(yaw_rad: float) -> float:
    half_pi = math.pi / 2.0
    return float(yaw_rad) % half_pi


def yaw_mod90_vector(yaw_rad: float) -> Point2:
    yaw = normalize_yaw_mod90(yaw_rad)
    return (math.cos(4.0 * yaw), math.sin(4.0 * yaw))


def fixed_square_footprint(
    center_taxel: Point2,
    yaw_rad: float = 0.0,
    side_taxel: float = BLOCK_SIDE_TAXEL,
) -> tuple[Point2, Point2, Point2, Point2]:
    cx, cy = center_taxel
    half = float(side_taxel) / 2.0
    cos_t = math.cos(float(yaw_rad))
    sin_t = math.sin(float(yaw_rad))
    corners = []
    for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half)):
        corners.append((cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t))
    return tuple(corners)  # type: ignore[return-value]
