from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Any

from tactile_interfaces.msg import BlockDetection

from tactile_task.commands import CancelledError, Robot


CONFIG = {
    "grid_center_x": 8.0,
    "grid_center_y": 2.0,
    "grid_pitch_m": 0.025,
    "pick_z": 0.017,
    "pick_z_raise_start_distance_m": 0.1,
    "pick_z_raise_end_distance_m": 0.2,
    "pick_z_max_raise_m": 0.01,
    "carry_z": 0.12,
    "retreat_z": 0.08,
    "pick_angle_rad": -1.45,
    "approach_duration_sec": 0.8,
    "descend_duration_sec": 0.5,
    "lift_duration_sec": 0.6,
    "transfer_duration_sec": 0.9,
    "retreat_duration_sec": 0.6,
    "detection_timeout_sec": 3.0,
    "min_detection_confidence": 0.6,
    "loop_period_sec": 1.0,
    "place_tolerance_m": 0.03,
    "tactile_pose_wait_sec": 2.0,
    "tactile_pose_timeout_sec": 3.0,
    "tactile_center_x": 7.5,
    "tactile_center_y": 3.5,
    "tactile_pitch_m": 0.004,
    "move_settle_sec": 0.15,
    "gripper_settle_sec": 1.0,
    "gripper_timeout_sec": 2.0,
    "command_timeout_margin_sec": 2.0,
    "ready_timeout_sec": 5.0,
    "gripper_open_position": 1400,
    "gripper_close_position": 2340,
    "no_block_move_duration_sec": 0.8,
    "no_block_pose": {
        "x": 0.07,
        "y": 0.0,
        "z": 0.1,
        "angle_rad": -1.0,
    },
    "place": {"x": 0.12, "y": 0.12, "z": 0.037,"angle_rad": -1.45,},
}


def run(robot: Robot, args: dict[str, Any]) -> None:
    cfg = _config(args)

    robot.wait_ready(cfg["ready_timeout_sec"])
    robot.log("recursive_tactile_pick_place started")
    at_avoidance_pose = False

    while True:
        robot.check_cancelled()
        loop_start = time.monotonic()
        detection = _detect_valid_block(robot, cfg)
        if detection is None:
            if not at_avoidance_pose:
                _move_to_no_block_pose(robot, cfg)
                at_avoidance_pose = True
            _wait_remaining_period(robot, cfg, loop_start)
            continue

        pick_x, pick_y = _grid_to_robot(detection.grid_column, detection.grid_row, cfg)
        distance_to_place = _place_distance(pick_x, pick_y, cfg)
        if distance_to_place <= float(cfg["place_tolerance_m"]):
            robot.log(
                "block_at_place "
                f"distance={distance_to_place:.4f}m "
                f"tolerance={float(cfg['place_tolerance_m']):.4f}m"
            )
            _wait_remaining_period(robot, cfg, loop_start)
            continue

        at_avoidance_pose = False
        grasped = _attempt_tactile_pick_place(robot, cfg, pick_x, pick_y)
        if grasped:
            robot.log("recursive pick-place attempt complete")
            _wait_remaining_period(robot, cfg, loop_start)
            continue

        robot.log("tactile miss; opening gripper and retrying from vision")
        _open(robot, cfg, "retry_open_after_tactile_miss")
        _move_to_no_block_pose(robot, cfg, stage="retry_no_block_pose")
        at_avoidance_pose = True
        _wait_remaining_period(robot, cfg, loop_start)


def _config(args: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(CONFIG)
    for key, value in args.items():
        if key in {"place", "no_block_pose"}:
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"recursive_tactile_pick_place args.{key} must be a JSON object"
                )
            cfg[key].update({name: float(raw) for name, raw in value.items()})
        elif key in cfg:
            cfg[key] = value
    cfg["place"] = {name: float(value) for name, value in cfg["place"].items()}
    cfg["no_block_pose"] = {
        name: float(value) for name, value in cfg["no_block_pose"].items()
    }
    cfg["loop_period_sec"] = max(0.0, float(cfg["loop_period_sec"]))
    cfg["min_detection_confidence"] = float(cfg["min_detection_confidence"])
    cfg["place_tolerance_m"] = max(0.0, float(cfg["place_tolerance_m"]))
    return cfg


def _detect_valid_block(robot: Robot, cfg: dict[str, Any]) -> BlockDetection | None:
    try:
        detection = robot.detect_block(cfg["detection_timeout_sec"])
    except CancelledError:
        raise
    except RuntimeError as exc:
        robot.log(f"vision retry: {exc}")
        return None

    if not detection.detected:
        robot.log(detection.message or "vision retry: no block detected")
        return None
    if not detection.grid_position_valid:
        robot.log("vision retry: block grid position is invalid")
        return None
    if float(detection.confidence) < float(cfg["min_detection_confidence"]):
        robot.log(
            "vision retry: "
            f"confidence={float(detection.confidence):.2f} "
            f"< {float(cfg['min_detection_confidence']):.2f}"
        )
        return None
    return detection


def _attempt_tactile_pick_place(
    robot: Robot,
    cfg: dict[str, Any],
    pick_x: float,
    pick_y: float,
) -> bool:
    pick_z = _pick_height(pick_x, pick_y, cfg)
    _move(
        robot,
        cfg,
        "move_above_pick",
        pick_x,
        pick_y,
        cfg["carry_z"],
        cfg["pick_angle_rad"],
        cfg["approach_duration_sec"],
    )
    _open(robot, cfg, "gripper_open")
    _move(
        robot,
        cfg,
        "descend_to_pick",
        pick_x,
        pick_y,
        pick_z,
        cfg["pick_angle_rad"],
        cfg["descend_duration_sec"],
    )
    _close(robot, cfg, "gripper_close")
    _move(
        robot,
        cfg,
        "lift",
        pick_x,
        pick_y,
        cfg["retreat_z"],
        cfg["pick_angle_rad"],
        cfg["lift_duration_sec"],
    )

    _wait(robot, cfg["tactile_pose_wait_sec"])
    tactile_pose = robot.detect_tactile_pose(cfg["tactile_pose_timeout_sec"])
    if not tactile_pose.detected:
        robot.log(tactile_pose.message or "no tactile block detected")
        return False

    place = cfg["place"]
    offset_x, offset_y = _tactile_offset_to_robot(tactile_pose, cfg)
    place_x, place_y = _corrected_place_position(
        place["x"],
        place["y"],
        offset_y,
    )
    robot.log(
        "tactile_offset "
        f"x={offset_x:.4f}m y={offset_y:.4f}m "
        f"place=({place_x:.4f}, {place_y:.4f})"
    )

    above_place_z = max(place["z"], cfg["carry_z"])
    _move(
        robot,
        cfg,
        "move_above_place",
        place_x,
        place_y,
        above_place_z,
        place["angle_rad"],
        cfg["transfer_duration_sec"],
    )
    _move(
        robot,
        cfg,
        "descend_to_place",
        place_x,
        place_y,
        place["z"],
        place["angle_rad"],
        cfg["descend_duration_sec"],
    )
    _open(robot, cfg, "release")
    _move(
        robot,
        cfg,
        "retreat",
        place_x,
        place_y,
        above_place_z,
        place["angle_rad"],
        cfg["retreat_duration_sec"],
    )
    return True


def _move_to_no_block_pose(
    robot: Robot,
    cfg: dict[str, Any],
    stage: str = "no_block_pose",
) -> None:
    pose = cfg["no_block_pose"]
    _move(
        robot,
        cfg,
        stage,
        pose["x"],
        pose["y"],
        pose["z"],
        pose["angle_rad"],
        cfg["no_block_move_duration_sec"],
    )


def _move(
    robot: Robot,
    cfg: dict[str, Any],
    stage: str,
    x: float,
    y: float,
    z: float,
    angle_rad: float,
    duration_sec: float,
) -> None:
    robot.move(
        x,
        y,
        z,
        angle_rad,
        duration_sec,
        stage=stage,
        timeout_margin_sec=cfg["command_timeout_margin_sec"],
        settle_sec=cfg["move_settle_sec"],
    )


def _open(robot: Robot, cfg: dict[str, Any], stage: str) -> None:
    robot.open_gripper(
        cfg["gripper_open_position"],
        stage=stage,
        timeout_sec=cfg["gripper_timeout_sec"],
        settle_sec=cfg["gripper_settle_sec"],
    )


def _close(robot: Robot, cfg: dict[str, Any], stage: str) -> None:
    robot.close_gripper(
        cfg["gripper_close_position"],
        stage=stage,
        timeout_sec=cfg["gripper_timeout_sec"],
        settle_sec=cfg["gripper_settle_sec"],
    )


def _wait(robot: Robot, duration_sec: float) -> None:
    deadline = time.monotonic() + max(0.0, float(duration_sec))
    while True:
        robot.check_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(remaining, 0.05))


def _wait_remaining_period(
    robot: Robot,
    cfg: dict[str, Any],
    loop_start: float,
) -> None:
    elapsed = time.monotonic() - loop_start
    _wait(robot, max(0.0, float(cfg["loop_period_sec"]) - elapsed))


def _grid_to_robot(col: float, row: float, cfg: dict[str, Any]) -> tuple[float, float]:
    x = (float(row) - cfg["grid_center_y"]) * cfg["grid_pitch_m"]
    y = -(float(col) - cfg["grid_center_x"]) * cfg["grid_pitch_m"]
    return x, y


def _place_distance(x: float, y: float, cfg: dict[str, Any]) -> float:
    place = cfg["place"]
    return math.hypot(float(x) - place["x"], float(y) - place["y"])


def _block_is_at_place(detection: BlockDetection, cfg: dict[str, Any]) -> bool:
    x, y = _grid_to_robot(detection.grid_column, detection.grid_row, cfg)
    return _place_distance(x, y, cfg) <= float(cfg["place_tolerance_m"])


def _pick_height(x: float, y: float, cfg: dict[str, Any]) -> float:
    minimum_height = float(cfg["pick_z"])
    extra_height = ((x**2 + y**2) ** 0.5 - 0.1) * 0.08
    return minimum_height + extra_height


def _tactile_offset_to_robot(pose: Any, cfg: dict[str, Any]) -> tuple[float, float]:
    offset_x = (float(pose.x) - cfg["tactile_center_x"]) * cfg["tactile_pitch_m"]
    offset_y = -(float(pose.y) - cfg["tactile_center_y"]) * cfg["tactile_pitch_m"]
    return offset_x, -offset_y


def _corrected_place_position(
    place_x: float,
    place_y: float,
    distance_offset: float,
) -> tuple[float, float]:
    x = float(place_x)
    y = float(place_y)
    radius = math.hypot(x, y)
    if radius <= 1e-9:
        return x, y + float(distance_offset)
    return (
        x + float(distance_offset) * x / radius,
        y + float(distance_offset) * y / radius,
    )
