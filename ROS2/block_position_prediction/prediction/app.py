from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch

from block_position_prediction.data_collection_manual.geometry import (
    BLOCK_SIDE_TAXEL,
    SheetConfig,
    fixed_square_footprint,
)
from block_position_prediction.data_collection_manual.labels import tactile_values_for_training
from block_position_prediction.data_collection_manual.preview_render import (
    PreviewPose,
    TactilePreview,
    draw_tactile_preview,
)
from block_position_prediction.data_collection_manual.tactile import (
    ThreadedTactileReader,
    TactileSnapshot,
    available_tactile_ports,
)
from block_position_prediction.model_training.dataset import (
    NUM_TAXELS,
    physics_features,
    tactile_map_channels,
    vector_to_yaw_np,
)
from block_position_prediction.model_training.model import TactilePoseNet, load_tactile_pose_state


DEFAULT_CHECKPOINT = Path("block_position_prediction/model/tactile_pose_best.pt")


@dataclass(frozen=True)
class PosePrediction:
    position_taxel: tuple[float, float]
    yaw_mod90_rad: float
    yaw_vector_norm: float
    footprint_corners_taxel: tuple[tuple[float, float], ...]
    fully_inside_sensor: bool
    object_present: bool
    confidence: float
    confidence_available: bool
    confidence_source: str
    force_sum: float

    @property
    def yaw_mod90_deg(self) -> float:
        return float(np.degrees(self.yaw_mod90_rad))


class TactilePosePredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        *,
        confidence_threshold: float = 0.5,
        legacy_force_threshold: float = 1.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = _resolve_device(device)
        self.confidence_threshold = float(confidence_threshold)
        self.legacy_force_threshold = float(legacy_force_threshold)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        self.model = TactilePoseNet().to(self.device)
        load_tactile_pose_state(self.model, checkpoint["model_state"])
        self.model.eval()
        self.config = checkpoint.get("config") or {}
        self.presence_available = bool(getattr(self.model, "presence_available", True))

    @torch.no_grad()
    def predict(self, values: np.ndarray, sheet: SheetConfig) -> PosePrediction:
        values = np.asarray(values, dtype=np.float32).reshape(NUM_TAXELS)
        maps = torch.from_numpy(tactile_map_channels(values))[None].to(self.device)
        raw_features = physics_features(values)
        force_sum = float(raw_features[0])
        features = (raw_features - self.feature_mean) / self.feature_std
        physics = torch.from_numpy(features.astype(np.float32))[None].to(self.device)
        output = self.model(maps, physics)
        position = output["position"][0].detach().cpu().numpy().astype(float)
        yaw_vector = output["yaw_vector"][0].detach().cpu().numpy().astype(np.float32)
        yaw = vector_to_yaw_np(yaw_vector)
        if self.presence_available and "presence_logit" in output:
            confidence = float(torch.sigmoid(output["presence_logit"][0]).detach().cpu())
            confidence_available = True
            confidence_source = "model"
        else:
            confidence = 1.0 if force_sum >= self.legacy_force_threshold else 0.0
            confidence_available = False
            confidence_source = "legacy_force"
        object_present = confidence >= self.confidence_threshold
        corners = fixed_square_footprint(
            (float(position[0]), float(position[1])),
            yaw_rad=yaw,
            side_taxel=BLOCK_SIDE_TAXEL,
        )
        return PosePrediction(
            position_taxel=(float(position[0]), float(position[1])),
            yaw_mod90_rad=float(yaw),
            yaw_vector_norm=float(np.linalg.norm(yaw_vector)),
            footprint_corners_taxel=corners,
            fully_inside_sensor=sheet.sensor.footprint_fully_inside_sensor(corners),
            object_present=object_present,
            confidence=confidence,
            confidence_available=confidence_available,
            confidence_source=confidence_source,
            force_sum=force_sum,
        )


def main() -> None:
    args = build_parser().parse_args()
    app = PredictionApp(args)
    if args.once_output:
        app.save_one_frame(args.once_output)
    else:
        app.run()


class PredictionApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sheet = SheetConfig()
        self.predictor = TactilePosePredictor(
            args.checkpoint,
            args.device,
            confidence_threshold=args.confidence_threshold,
            legacy_force_threshold=args.legacy_force_threshold,
        )
        self.tactile = ThreadedTactileReader(args.tactile_port, args.tactile_baud)
        self.ports = _ports_with_current(args.tactile_port)
        self.port_index = max(0, self.ports.index(args.tactile_port) if args.tactile_port in self.ports else 0)
        self.status = "starting"

    def run(self) -> None:
        import cv2

        self.tactile.start()
        cv2.namedWindow(self.args.window, cv2.WINDOW_NORMAL)
        try:
            while True:
                image = self.render()
                cv2.imshow(self.args.window, image)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("p"), ord("P")):
                    self._cycle_port()
                elif key in (ord("r"), ord("R")):
                    self.tactile.reset_calibration()
                    self.status = "reset tactile processor"
                elif key in (ord("t"), ord("T")):
                    self.tactile.tare()
                    self.status = "tared tactile processor"
        finally:
            self.tactile.stop()
            cv2.destroyWindow(self.args.window)

    def save_one_frame(self, path: str | Path) -> Path:
        import cv2

        self.tactile.start()
        try:
            deadline = time.time() + max(0.1, float(self.args.once_timeout))
            while time.time() < deadline:
                if self.tactile.snapshot().available:
                    break
                time.sleep(0.02)
            image = self.render()
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out_path), image):
                raise RuntimeError(f"could not write {out_path}")
            return out_path
        finally:
            self.tactile.stop()

    def render(self) -> np.ndarray:
        import cv2

        snapshot = self.tactile.snapshot()
        values = tactile_values_for_training(snapshot)
        prediction = None
        error = snapshot.error
        if values is not None:
            try:
                prediction = self.predictor.predict(values, self.sheet)
                self.status = "prediction ready" if prediction.object_present else "no block detected"
            except Exception as exc:
                error = str(exc)
                self.status = "prediction failed"
        else:
            self.status = "waiting for tactile top5 input"

        preview = self._preview(values, prediction)
        heatmap = draw_tactile_preview(preview, config=self.sheet, width=int(self.args.heatmap_width))
        canvas_height = max(520, heatmap.shape[0])
        heatmap = _pad_to_height(heatmap, canvas_height)
        panel = self._status_panel(snapshot, values, prediction, error, canvas_height)
        return np.hstack([heatmap, panel])

    def _preview(self, values: np.ndarray | None, prediction: PosePrediction | None) -> TactilePreview:
        if prediction is None or not prediction.object_present:
            return TactilePreview(values=values, position_taxel=None, title="Live tactile input")
        pose = PreviewPose(
            available=True,
            source="model",
            yaw_mod90_rad=prediction.yaw_mod90_rad,
            footprint_side_taxel=BLOCK_SIDE_TAXEL,
            footprint_corners_taxel=prediction.footprint_corners_taxel,
            fully_inside_sensor=prediction.fully_inside_sensor,
        )
        return TactilePreview(
            values=values,
            position_taxel=prediction.position_taxel,
            pose=pose,
            title="Model prediction",
        )

    def _status_panel(
        self,
        snapshot: TactileSnapshot,
        values: np.ndarray | None,
        prediction: PosePrediction | None,
        error: str | None,
        height: int,
    ) -> np.ndarray:
        import cv2

        panel_w = 430
        panel = np.zeros((height, panel_w, 3), dtype=np.uint8)
        panel[:] = (18, 18, 18)
        lines = [
            ("Tactile Pose Prediction", 0.72, (0, 230, 255), 2),
            (f"checkpoint: {self.predictor.checkpoint_path.name}", 0.45, (230, 230, 230), 1),
            (f"device: {self.predictor.device}", 0.45, (230, 230, 230), 1),
            (f"port: {snapshot.port}", 0.50, (230, 230, 230), 1),
            (f"sensor: {snapshot.status}", 0.45, (185, 225, 185), 1),
            (f"input: 10-frame top5 normalized", 0.45, (185, 225, 185), 1),
            (f"confidence threshold: {self.predictor.confidence_threshold:.2f}", 0.43, (185, 225, 185), 1),
        ]
        y = 30
        for text, scale, color, thickness in lines:
            cv2.putText(panel, _clip(text, 52), (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            y += 28
        y += 8

        if prediction is None:
            cv2.putText(panel, "prediction: waiting", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)
            y += 30
        else:
            conf_label = "model" if prediction.confidence_available else "legacy force"
            cv2.putText(
                panel,
                f"confidence: {prediction.confidence:.3f} ({conf_label})",
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 28
            if not prediction.object_present:
                cv2.putText(
                    panel,
                    "prediction: no block detected",
                    (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (90, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
                y += 28
            if not prediction.object_present:
                pred_lines = [
                    f"force sum: {prediction.force_sum:.3f}",
                ]
            else:
                x, row = prediction.position_taxel
                inside = "inside" if prediction.fully_inside_sensor else "edge/outside"
                pred_lines = [
                    f"pos taxel: x={x:.2f}, y={row:.2f}",
                    f"yaw mod90: {prediction.yaw_mod90_deg:.1f} deg",
                    f"footprint: {inside}",
                    f"yaw vector norm: {prediction.yaw_vector_norm:.2f}",
                ]
            for text in pred_lines:
                cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
                y += 28

        if values is not None:
            grid = values.reshape(8, 16)
            stats = [
                f"force sum: {float(grid.sum()):.3f}",
                f"max taxel: {float(grid.max()):.3f}",
                f"active >0.05: {float((grid > 0.05).mean()) * 100.0:.1f}%",
            ]
            y += 8
            for text in stats:
                cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1, cv2.LINE_AA)
                y += 24

        if error:
            y += 8
            for text in _wrap(f"error: {error}", 48):
                cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 80, 255), 1, cv2.LINE_AA)
                y += 22

        help_lines = [
            "Q/Esc quit",
            "P cycle tactile port",
            "R reset processor",
            "T tare processor",
        ]
        y = max(y + 12, height - 112)
        for text in help_lines:
            cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)
            y += 24
        return panel

    def _cycle_port(self) -> None:
        self.ports = _ports_with_current(self.tactile.port)
        self.port_index = (self.port_index + 1) % max(1, len(self.ports))
        port = self.ports[self.port_index]
        self.tactile.reconnect(port, self.args.tactile_baud)
        self.status = f"connected tactile port: {port}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime tactile sensor block pose prediction UI.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tactile-port", default="SIMULATOR", help="Serial port, or SIMULATOR.")
    parser.add_argument("--tactile-baud", type=int, default=115200)
    parser.add_argument("--heatmap-width", type=int, default=640)
    parser.add_argument("--window", default="Tactile Pose Prediction")
    parser.add_argument("--once-output", default=None, help="Render one frame to an image path and exit.")
    parser.add_argument("--once-timeout", type=float, default=2.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--legacy-force-threshold",
        type=float,
        default=1.0,
        help="Fallback no-block threshold for legacy checkpoints without a confidence head.",
    )
    return parser


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ports_with_current(current: str) -> list[str]:
    ports = list(available_tactile_ports())
    if current not in ports:
        ports.insert(0, current)
    return ports or ["SIMULATOR"]


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _wrap(text: str, width: int) -> Iterable[str]:
    while len(text) > width:
        yield text[:width]
        text = text[width:]
    if text:
        yield text


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    padded = np.zeros((height, image.shape[1], image.shape[2]), dtype=image.dtype)
    padded[:] = (18, 18, 18)
    padded[: image.shape[0], :, :] = image
    return padded


if __name__ == "__main__":
    main()
