from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Any

from tactile_task.commands import Robot


CONFIG = {
    "pick_z_raise_start_distance_m": 0.1,
    "pick_z_raise_end_distance_m": 0.2,
    "pick_z_max_raise_m": 0.01,
    "carry_z": 0.08,
    "retreat_z": 0.08,
    "approach_duration_sec": 0.8,
    "descend_duration_sec": 0.5,
    "lift_duration_sec": 0.6,
    "transfer_duration_sec": 0.9,
    "retreat_duration_sec": 0.6,
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
    "pick": {
        "x": 0.2,
        "y": 0.0,
        "z": 0.017,
        "angle_rad": -1.45,
    },
    "place": {
        "x": 0.122,
        "y": 0.122,
        "z": 0.037,
        "angle_rad": -1.45,
    },
}


def run(robot: Robot, args: dict[str, Any]) -> None:
    cfg = _config(args)

    robot.wait_ready(cfg["ready_timeout_sec"])

    pick = cfg["pick"]
    pick_x = pick["x"]
    pick_y = pick["y"]
    pick_z = _pick_height(pick_x, pick_y, pick["z"])
    _move(
        robot,
        cfg,
        "move_above_pick",
        pick_x,
        pick_y,
        cfg["carry_z"],
        pick["angle_rad"],
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
        pick["angle_rad"],
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
        pick["angle_rad"],
        cfg["lift_duration_sec"],
    )

    place = cfg["place"]
    _wait(robot, cfg["tactile_pose_wait_sec"])
    tactile_pose = robot.detect_tactile_pose(cfg["tactile_pose_timeout_sec"])
    _validate_tactile_pose(tactile_pose)

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
    robot.log("pick-place complete")


def _config(args: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(CONFIG)
    for key, value in args.items():
        if key in {"pick", "place"}:
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"blind_tactile_pick_place args.{key} must be a JSON object"
                )
            cfg[key].update({name: float(raw) for name, raw in value.items()})
        elif key in cfg:
            cfg[key] = value
    cfg["pick"] = {name: float(value) for name, value in cfg["pick"].items()}
    cfg["place"] = {name: float(value) for name, value in cfg["place"].items()}
    return cfg


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


def _pick_height(x: float, y: float, minimum_height: float) -> float:
    extra_height = ((x**2 + y**2) ** 0.5 - 0.1) * 0.08
    return float(minimum_height) + extra_height


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


def _validate_tactile_pose(pose: Any) -> None:
    if not pose.detected:
        raise RuntimeError(pose.message or "no tactile block detected")
