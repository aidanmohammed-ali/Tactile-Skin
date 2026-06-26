"""ROS 2 service node for tactile block pose prediction."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from tactile_interfaces.srv import GetTactilePose

from .tactile_filtering import (
    mean_tactile_frames,
    median_filter_3x3,
    subtract_tare_baseline,
)
from .tactile_visualization import (
    draw_tactile_heatmap,
    draw_tactile_prediction,
)


def _ensure_prediction_package_on_path() -> None:
    if importlib.util.find_spec("block_position_prediction") is not None:
        return

    candidates: list[Path] = []
    env_root = os.environ.get("BLOCK_POSITION_PREDICTION_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(Path.cwd().resolve().parents)
    candidates.append(Path.cwd().resolve())
    candidates.extend(Path(__file__).resolve().parents)

    for candidate in candidates:
        if (candidate / "block_position_prediction").is_dir():
            sys.path.insert(0, str(candidate))
            return


_ensure_prediction_package_on_path()
tactile_geometry = importlib.import_module(
    "block_position_prediction.data_collection_manual.geometry"
)
tactile_reader = importlib.import_module(
    "block_position_prediction.data_collection_manual.tactile"
)
prediction_app = importlib.import_module(
    "block_position_prediction.prediction.app"
)


def _resolve_checkpoint_path(checkpoint: str) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_file() or path.is_absolute():
        return path
    package_root = Path(prediction_app.__file__).resolve().parents[1]
    model_path = package_root / "model" / path.name
    if model_path.is_file():
        return model_path
    project_path = package_root / path.name
    if project_path.is_file():
        return project_path
    return path


class TactilePoseNode(Node):
    """Expose live tactile model predictions through a ROS service."""

    def __init__(self) -> None:
        super().__init__("tactile_pose_node")
        self.sheet = tactile_geometry.SheetConfig()

        checkpoint = str(
            self.declare_parameter(
                "checkpoint",
                str(prediction_app.DEFAULT_CHECKPOINT),
            ).value
        )
        checkpoint = str(_resolve_checkpoint_path(checkpoint))
        device = str(self.declare_parameter("device", "auto").value)
        tactile_port = str(
            self.declare_parameter("tactile_port", "SIMULATOR").value
        )
        tactile_baud = int(
            self.declare_parameter("tactile_baud", 115200).value
        )
        self.prediction_frame_count = 5
        self.prediction_capture_timeout = max(
            0.0,
            float(
                self.declare_parameter(
                    "prediction_capture_timeout",
                    1.0,
                ).value
            ),
        )
        confidence_threshold = float(
            self.declare_parameter("confidence_threshold", 0.5).value
        )
        legacy_force_threshold = float(
            self.declare_parameter("legacy_force_threshold", 1.0).value
        )
        service_name = str(
            self.declare_parameter(
                "service_name",
                "/tactile_sensor/predict_pose",
            ).value
        )
        stream_frames = bool(
            self.declare_parameter("stream_frames", False).value
        )
        stream_topic = str(
            self.declare_parameter(
                "stream_topic",
                "/tactile_sensor/frame",
            ).value
        )
        self.stream_frame_kind = str(
            self.declare_parameter("stream_frame_kind", "raw_latest").value
        )
        self.display_heatmap = bool(
            self.declare_parameter("display_heatmap", True).value
        )
        self.heatmap_window = str(
            self.declare_parameter(
                "heatmap_window",
                "Tactile Sensor Heatmap",
            ).value
        )
        self.heatmap_width = int(
            self.declare_parameter("heatmap_width", 640).value
        )
        self.heatmap_value_max = float(
            self.declare_parameter("heatmap_value_max", 0.0).value
        )
        self.heatmap_flip_x = bool(
            self.declare_parameter("heatmap_flip_x", True).value
        )
        self.heatmap_rate_hz = max(
            1.0,
            float(self.declare_parameter("heatmap_rate_hz", 30.0).value),
        )
        self.save_prediction_images = bool(
            self.declare_parameter("save_prediction_images", True).value
        )
        self.prediction_image_topic = str(
            self.declare_parameter(
                "prediction_image_topic",
                "/tactile_sensor/prediction_image",
            ).value
        )
        self.prediction_image_dir = Path(
            str(
                self.declare_parameter(
                    "prediction_image_dir",
                    "~/.ros/tactile_sensor/predictions",
                ).value
            )
        ).expanduser()
        self.prediction_image_width = int(
            self.declare_parameter(
                "prediction_image_width",
                self.heatmap_width,
            ).value
        )
        labels_file = str(
            self.declare_parameter(
                "prediction_labels_file",
                "labels.jsonl",
            ).value
        ).strip()
        self.prediction_labels_file = Path(
            labels_file or "labels.jsonl"
        ).name
        self._prediction_image_count = self._existing_prediction_count()
        self._prediction_log_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._prediction_frame_condition = threading.Condition()
        self._recent_prediction_frames: list[np.ndarray] = []
        self._prediction_frame_sequence = 0
        self._latest_filtered_frame = None
        self._tare_baseline = None
        self._latest_display_frame = None
        self._display_frame_seq = 0
        self._rendered_display_frame_seq = -1
        self._heatmap_window_created = False

        self.predictor = prediction_app.TactilePosePredictor(
            checkpoint,
            device,
            confidence_threshold=confidence_threshold,
            legacy_force_threshold=legacy_force_threshold,
        )
        self.frame_pub = None
        if stream_frames:
            self.frame_pub = self.create_publisher(
                Float32MultiArray,
                stream_topic,
                10,
            )
        self.prediction_image_pub = self.create_publisher(
            Image,
            self.prediction_image_topic,
            10,
        )
        self.tactile = tactile_reader.ThreadedTactileReader(
            tactile_port,
            tactile_baud,
            frame_callback=self._handle_tactile_frame,
        )
        self.tactile.start()
        self.heatmap_timer = None
        if self.display_heatmap:
            self.heatmap_timer = self.create_timer(
                1.0 / self.heatmap_rate_hz,
                self._render_heatmap,
            )
        self.service = self.create_service(
            GetTactilePose,
            service_name,
            self._predict_pose,
        )
        if stream_frames:
            self.get_logger().info(
                "Tactile frame streaming enabled on "
                f"{stream_topic} ({self.stream_frame_kind}, per frame)"
            )
        if self.display_heatmap:
            self.get_logger().info(
                "Tactile heatmap display enabled "
                f"({self.stream_frame_kind}, {self.heatmap_rate_hz:.1f} Hz)"
            )
        if self.save_prediction_images:
            self.get_logger().info(
                "Tactile prediction images will be saved to "
                f"{self.prediction_image_dir}"
            )
        self.get_logger().info(
            "Tactile prediction images will be published on "
            f"{self.prediction_image_topic}"
        )
        self.get_logger().info(
            "Tactile pose service ready on "
            f"{service_name} using {self.predictor.checkpoint_path}"
        )

    def _existing_prediction_count(self) -> int:
        labels_path = (
            self.prediction_image_dir / self.prediction_labels_file
        )
        if not labels_path.is_file():
            return 0
        largest = 0
        try:
            with labels_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    sample_id = str(json.loads(line).get("sample_id") or "")
                    if sample_id.isdigit():
                        largest = max(largest, int(sample_id))
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().warning(
                f"Could not resume prediction sample numbering: {exc}"
            )
        return largest

    def _predict_pose(
        self,
        _request: GetTactilePose.Request,
        response: GetTactilePose.Response,
    ) -> GetTactilePose.Response:
        filtered_frames = self._capture_new_prediction_frames(
            self.prediction_frame_count,
            self.prediction_capture_timeout,
        )
        snapshot = self.tactile.snapshot()
        if len(filtered_frames) != self.prediction_frame_count:
            response.success = False
            response.detected = False
            response.x = math.nan
            response.y = math.nan
            response.angle_rad = math.nan
            response.angle_deg = math.nan
            response.confidence = 0.0
            response.fully_inside_sensor = False
            response.message = (
                snapshot.error
                or (
                    "timed out waiting for "
                    f"{self.prediction_frame_count} new tactile frames"
                )
            )
            return response

        values = mean_tactile_frames(filtered_frames)
        values = tactile_reader.canonicalize_tactile_values(values)

        try:
            prediction = self.predictor.predict(values, self.sheet)
        except Exception as exc:
            response.success = False
            response.detected = False
            response.x = math.nan
            response.y = math.nan
            response.angle_rad = math.nan
            response.angle_deg = math.nan
            response.confidence = 0.0
            response.fully_inside_sensor = False
            response.message = f"prediction failed: {exc}"
            return response

        x, y = prediction.position_taxel
        response.success = True
        response.detected = bool(prediction.object_present)
        response.x = float(x)
        response.y = float(y)
        response.angle_rad = float(prediction.yaw_mod90_rad)
        response.angle_deg = float(prediction.yaw_mod90_deg)
        response.confidence = float(prediction.confidence)
        response.fully_inside_sensor = bool(prediction.fully_inside_sensor)
        if prediction.object_present:
            response.message = "prediction ready"
        else:
            response.message = "no block detected"
        self._save_prediction(values, prediction, snapshot)
        return response

    def _save_prediction(self, values, prediction, snapshot) -> None:
        try:
            image = draw_tactile_prediction(
                values,
                detected=bool(prediction.object_present),
                position_taxel=prediction.position_taxel,
                angle_deg=float(prediction.yaw_mod90_deg),
                confidence=float(prediction.confidence),
                fully_inside_sensor=bool(prediction.fully_inside_sensor),
                footprint_corners_taxel=(
                    prediction.footprint_corners_taxel
                ),
                width=self.prediction_image_width,
                value_max=1.0,
                flip_x=False,
            )
            self.prediction_image_pub.publish(self._image_msg_from_bgr(image))
            if not self.save_prediction_images:
                return
            import cv2

            with self._prediction_log_lock:
                self.prediction_image_dir.mkdir(parents=True, exist_ok=True)
                self._prediction_image_count += 1
                sample_id = f"{self._prediction_image_count:06d}"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                detection = (
                    "detected" if prediction.object_present else "no_block"
                )
                image_name = (
                    f"prediction_{timestamp}_{sample_id}_{detection}.png"
                )
                output_path = self.prediction_image_dir / image_name
                if not cv2.imwrite(str(output_path), image):
                    raise OSError(
                        f"cv2.imwrite returned false for {output_path}"
                    )
                record = self._prediction_record(
                    sample_id,
                    image_name,
                    values,
                    prediction,
                    snapshot,
                )
                labels_path = (
                    self.prediction_image_dir
                    / self.prediction_labels_file
                )
                with labels_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True))
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            self.get_logger().info(
                "Saved tactile prediction image and label: "
                f"{output_path}"
            )
        except Exception as exc:
            self.get_logger().warning(
                f"Failed to publish/save tactile prediction data: {exc}"
            )

    def _image_msg_from_bgr(self, image: np.ndarray) -> Image:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "tactile_sensor"
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(msg.width * 3)
        msg.data = image.tobytes()
        return msg

    def _prediction_record(
        self,
        sample_id,
        image_name,
        values,
        prediction,
        snapshot,
    ) -> dict:
        detected = bool(prediction.object_present)
        position = tuple(float(value) for value in prediction.position_taxel)
        corners = [
            [float(x), float(y)]
            for x, y in prediction.footprint_corners_taxel
        ]
        yaw_rad = float(prediction.yaw_mod90_rad)
        sensor = self.sheet.sensor
        return {
            "schema_version": "tactile_pose_prediction_v1",
            "sample_id": sample_id,
            "timestamp": float(snapshot.timestamp),
            "saved_at": datetime.now().astimezone().isoformat(),
            "image_path": image_name,
            "input": {
                "kind": (
                    "five_consecutive_top5_normalized_median3x3_mean"
                ),
                "tare_enabled": self._tare_enabled(),
                "frame": tactile_geometry.SENSOR_COORD_FRAME,
                "rows": int(sensor.rows),
                "cols": int(sensor.cols),
                "values": [
                    float(value) for value in values.reshape(-1)
                ],
            },
            "prediction": {
                "object_present": detected,
                "position_taxel": list(position),
                "yaw_mod90_rad": yaw_rad,
                "yaw_mod90_deg": float(prediction.yaw_mod90_deg),
                "yaw_vector_norm": float(prediction.yaw_vector_norm),
                "footprint_corners_taxel": corners,
                "fully_inside_sensor": bool(
                    prediction.fully_inside_sensor
                ),
                "confidence": float(prediction.confidence),
                "confidence_available": bool(
                    prediction.confidence_available
                ),
                "confidence_source": str(
                    prediction.confidence_source
                ),
                "force_sum": float(prediction.force_sum),
            },
        }

    def _handle_tactile_frame(self, snapshot) -> None:
        if snapshot.top5_normalized is not None:
            prediction_frame = median_filter_3x3(
                snapshot.top5_normalized
            )
            with self._prediction_frame_condition:
                self._latest_filtered_frame = prediction_frame.copy()
                if self._tare_baseline is not None:
                    prediction_frame = subtract_tare_baseline(
                        prediction_frame,
                        self._tare_baseline,
                    )
                self._recent_prediction_frames.append(prediction_frame)
                if (
                    len(self._recent_prediction_frames)
                    > self.prediction_frame_count
                ):
                    del self._recent_prediction_frames[0]
                self._prediction_frame_sequence += 1
                self._prediction_frame_condition.notify_all()

        frame = self._stream_frame_from_snapshot(snapshot)
        if frame is None:
            return
        if self.frame_pub is not None:
            msg = Float32MultiArray()
            msg.layout.dim = [
                MultiArrayDimension(label="rows", size=8, stride=128),
                MultiArrayDimension(label="cols", size=16, stride=16),
            ]
            msg.layout.data_offset = 0
            msg.data = frame.astype(float).reshape(-1).tolist()
            self.frame_pub.publish(msg)
        if self.display_heatmap:
            with self._frame_lock:
                self._latest_display_frame = frame.copy()
                self._display_frame_seq += 1

    def _capture_new_prediction_frames(
        self,
        count: int,
        timeout: float,
    ) -> tuple[np.ndarray, ...]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._prediction_frame_condition:
            target_sequence = self._prediction_frame_sequence + count
            while self._prediction_frame_sequence < target_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return ()
                self._prediction_frame_condition.wait(remaining)
            return tuple(
                frame.copy()
                for frame in self._recent_prediction_frames[-count:]
            )

    def _render_heatmap(self) -> None:
        if not self.display_heatmap:
            return
        if os.name != "nt" and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            self._disable_heatmap("no graphical display is available")
            return
        with self._frame_lock:
            if (
                self._latest_display_frame is None
                or self._display_frame_seq == self._rendered_display_frame_seq
            ):
                return
            frame = self._latest_display_frame.copy()
            frame_seq = self._display_frame_seq
        try:
            import cv2

            image = draw_tactile_heatmap(
                frame,
                width=self.heatmap_width,
                value_max=self.heatmap_value_max,
                flip_x=self.heatmap_flip_x,
                title=(
                    f"Live tactile input | {self.stream_frame_kind} | "
                    f"TARE {'ON' if self._tare_enabled() else 'OFF'} "
                    "(T: toggle)"
                ),
            )
            if not self._heatmap_window_created:
                cv2.namedWindow(self.heatmap_window, cv2.WINDOW_NORMAL)
                self._heatmap_window_created = True
            elif cv2.getWindowProperty(
                self.heatmap_window,
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                self._disable_heatmap("window closed by user")
                return
            cv2.imshow(self.heatmap_window, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                self._disable_heatmap("closed by user")
                return
            if key in (ord("t"), ord("T")):
                self._toggle_tare()
        except Exception as exc:
            self._disable_heatmap(f"display failed: {exc}")
            return
        self._rendered_display_frame_seq = frame_seq

    def _disable_heatmap(self, reason: str) -> None:
        if not self.display_heatmap:
            return
        self.display_heatmap = False
        self.get_logger().warning(f"Tactile heatmap disabled: {reason}")
        if self._heatmap_window_created:
            try:
                import cv2

                cv2.destroyWindow(self.heatmap_window)
                cv2.waitKey(1)
            except Exception:
                pass
            self._heatmap_window_created = False

    def _stream_frame_from_snapshot(self, snapshot):
        kind = self.stream_frame_kind
        if kind == "raw_latest":
            if not snapshot.recent_raw_frames:
                return None
            return snapshot.recent_raw_frames[-1]
        if kind == "raw_top5_average":
            if snapshot.top5_raw_average is None:
                return None
            frame = median_filter_3x3(snapshot.top5_raw_average)
            return self._apply_tare(frame, scale=65535.0)
        if kind == "top5_normalized":
            if snapshot.top5_normalized is None:
                return None
            frame = median_filter_3x3(snapshot.top5_normalized)
            return self._apply_tare(frame)
        if kind == "processed":
            return snapshot.processed
        self.get_logger().warning(
            f"Unknown stream_frame_kind={kind!r}; using raw_latest"
        )
        self.stream_frame_kind = "raw_latest"
        if not snapshot.recent_raw_frames:
            return None
        return snapshot.recent_raw_frames[-1]

    def _tare_enabled(self) -> bool:
        with self._prediction_frame_condition:
            return self._tare_baseline is not None

    def _apply_tare(self, frame, scale: float = 1.0):
        with self._prediction_frame_condition:
            baseline = (
                None
                if self._tare_baseline is None
                else self._tare_baseline.copy()
            )
        if baseline is None:
            return frame
        return subtract_tare_baseline(frame, baseline, scale=scale)

    def _toggle_tare(self) -> None:
        with self._prediction_frame_condition:
            if self._tare_baseline is None:
                if self._latest_filtered_frame is None:
                    self.get_logger().warning(
                        "Cannot enable tactile tare before a frame "
                        "is available"
                    )
                    return
                self._tare_baseline = self._latest_filtered_frame.copy()
                enabled = True
            else:
                self._tare_baseline = None
                enabled = False
            self._recent_prediction_frames.clear()
        state = "enabled" if enabled else "disabled"
        self.get_logger().info(f"Tactile tare {state}")

    def destroy_node(self) -> bool:
        self.tactile.stop()
        if self._heatmap_window_created:
            try:
                import cv2

                cv2.destroyWindow(self.heatmap_window)
                cv2.waitKey(1)
            except Exception:
                pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: TactilePoseNode | None = None
    try:
        node = TactilePoseNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
