from __future__ import annotations

import math
from typing import Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty, Float32, String

from tactile_interfaces.msg import BlockDetection as BlockDetectionMsg

from .camera import ThreadedCamera, parse_camera_source
from .camera_defaults import DEFAULT_CAMERA_SOURCE, load_camera_source, save_camera_source
from .detector import (
    BlockDetection as RawBlockDetection,
    BlockDetector,
    default_calibration_path,
    default_weights_path,
    make_board_config,
    parse_device,
)
from .image_messages import bgr_to_image_msg
from .qos import image_qos_profile


class VisionNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_node")
        self.frame_id = str(self.declare_parameter("frame_id", "vision_camera").value)
        configured_source = str(self.declare_parameter("source", DEFAULT_CAMERA_SOURCE).value)
        self.camera_source_store = str(self.declare_parameter("camera_source_store", "").value).strip() or None
        self.source = parse_camera_source(load_camera_source(configured_source, self.camera_source_store))
        self.frame_width = int(self.declare_parameter("frame_width", 1280).value)
        self.frame_height = int(self.declare_parameter("frame_height", 720).value)
        publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 10.0).value)
        self.detect_on_timer = bool(self.declare_parameter("detect_on_timer", False).value)
        self.capture_low_latency = bool(self.declare_parameter("capture_low_latency", True).value)
        image_qos = image_qos_profile(
            depth=int(self.declare_parameter("image_qos_depth", 1).value),
            reliability=str(self.declare_parameter("image_qos_reliability", "best_effort").value),
        )

        weights = str(self.declare_parameter("weights", default_weights_path()).value).strip()
        calibration = str(self.declare_parameter("calibration", default_calibration_path()).value).strip()
        image_size = int(self.declare_parameter("image_size", 640).value)
        board_config = make_board_config(
            hole_pitch_mm=float(self.declare_parameter("hole_pitch_mm", 25.0).value),
            dictionary=str(self.declare_parameter("dictionary", "DICT_4X4_50").value),
            marker_size_grid=float(self.declare_parameter("marker_size_grid", 1.6).value),
            marker_margin_grid=float(self.declare_parameter("marker_margin_grid", -1.4).value),
            refine_holes=bool(self.declare_parameter("refine_holes", True).value),
        )
        self.detector = BlockDetector(
            weights_path=weights or default_weights_path(),
            calibration_path=calibration or default_calibration_path(),
            confidence=float(self.declare_parameter("confidence", 0.1).value),
            image_size=image_size if image_size > 0 else None,
            anchor=str(self.declare_parameter("anchor", "center").value),
            device=parse_device(str(self.declare_parameter("device", "auto").value)),
            board_config=board_config,
        )
        self.camera = ThreadedCamera(
            self.source,
            frame_width=self.frame_width if self.frame_width > 0 else None,
            frame_height=self.frame_height if self.frame_height > 0 else None,
            capture_low_latency=self.capture_low_latency,
        )
        self.annotated_pub = self.create_publisher(Image, "/vision/annotated_image", image_qos)
        self.detection_pub = self.create_publisher(BlockDetectionMsg, "/vision/block_detection", 10)
        self.status_pub = self.create_publisher(String, "/vision/status", 10)
        self.create_subscription(String, "/vision/camera_source", self._set_camera_source, 10)
        self.create_subscription(Float32, "/vision/confidence_threshold", self._set_confidence_threshold, 10)
        self.create_subscription(Bool, "/vision/calibrate_trigger", self._calibrate_board, 10)
        self.create_subscription(Empty, "/vision/detect_trigger", self._detect_block, 10)
        self.timer = self.create_timer(1.0 / max(0.1, publish_rate_hz), self._publish_latest)
        self.last_detection_msg = self._empty_detection("not run")
        self.last_annotated_frame: Any | None = None
        self.last_raw_detections: list[RawBlockDetection] = []
        self._start_camera()

    def _start_camera(self) -> None:
        try:
            self.camera.start()
            self.get_logger().info(f"Vision camera opened: {self.camera.status.source}")
        except Exception as exc:
            self.get_logger().error(f"Could not open camera source {self.source!r}: {exc}")

    def _set_camera_source(self, msg: String) -> None:
        source = parse_camera_source(msg.data)
        previous_source = self.source
        try:
            self.camera.reconnect(
                source,
                frame_width=self.frame_width if self.frame_width > 0 else None,
                frame_height=self.frame_height if self.frame_height > 0 else None,
                capture_low_latency=self.capture_low_latency,
            )
        except Exception as exc:
            try:
                self.camera.reconnect(
                    previous_source,
                    frame_width=self.frame_width if self.frame_width > 0 else None,
                    frame_height=self.frame_height if self.frame_height > 0 else None,
                    capture_low_latency=self.capture_low_latency,
                )
            except Exception:
                pass
            self._publish_status(f"camera source failed: {exc}")
            return
        self.source = source
        try:
            path = save_camera_source(str(source), self.camera_source_store)
        except Exception as exc:
            self._publish_status(f"camera source updated, but could not save default: {exc}")
            return
        self._publish_status(f"camera source updated and saved: {self.source} -> {path}")

    def _set_confidence_threshold(self, msg: Float32) -> None:
        threshold = max(0.0, min(1.0, float(msg.data)))
        self.detector.confidence = threshold
        self._publish_status(f"confidence threshold set to {threshold:.2f}")
        self._detect_current_frame()

    def _calibrate_board(self, msg: Bool) -> None:
        frame = self.camera.latest_frame()
        if frame is None:
            self._publish_status("calibration failed: no camera frame available")
            return
        try:
            calibration = self.detector.calibrate_from_frame(frame, save=bool(msg.data))
        except Exception as exc:
            self._publish_status(f"calibration failed: {exc}")
            return
        median = calibration.quality.median_grid_error
        median_text = "nan" if median is None else f"{float(median):.4f}"
        self._publish_status(
            f"calibrated markers={len(calibration.marker_ids)} "
            f"holes={calibration.quality.hole_point_count} median_error={median_text}"
        )

    def _detect_block(self, _msg: Empty) -> None:
        self._detect_current_frame()

    def _detect_current_frame(self) -> None:
        frame = self.camera.latest_frame()
        if frame is None:
            detection = self._empty_detection("no camera frame available")
            self.last_detection_msg = detection
            self.last_raw_detections = []
            self.detection_pub.publish(detection)
            self._publish_status(detection.message)
            return
        detection_msg, annotated, raw_detections = self._run_detection(frame)
        self.last_detection_msg = detection_msg
        self.last_annotated_frame = annotated
        self.last_raw_detections = raw_detections
        stamp = self.get_clock().now().to_msg()
        self.detection_pub.publish(detection_msg)
        self.annotated_pub.publish(bgr_to_image_msg(annotated, stamp, self.frame_id))
        self._publish_status(detection_msg.message)

    def _publish_latest(self) -> None:
        frame = self.camera.latest_frame()
        if frame is None:
            msg = self._empty_detection(self.camera.status.message)
            self.last_detection_msg = msg
            self.detection_pub.publish(msg)
            return
        if self.detect_on_timer:
            detection_msg, annotated, raw_detections = self._run_detection(frame)
            self.last_raw_detections = raw_detections
        else:
            detection_msg = self.last_detection_msg
            annotated = self.detector.render_frame(frame, self.last_raw_detections)
        self.last_detection_msg = detection_msg
        self.last_annotated_frame = annotated
        stamp = self.get_clock().now().to_msg()
        self.detection_pub.publish(detection_msg)
        self.annotated_pub.publish(bgr_to_image_msg(annotated, stamp, self.frame_id))

    def _run_detection(self, frame: Any) -> tuple[BlockDetectionMsg, Any, list[RawBlockDetection]]:
        try:
            detections = self.detector.detect_blocks(frame)
        except Exception as exc:
            msg = self._empty_detection(f"detection failed: {exc}")
            return msg, self.detector.render_frame(frame, None), []
        annotated = self.detector.render_frame(frame, detections)
        detection = detections[0] if detections else None
        if detection is None:
            return self._empty_detection(f"no block detected conf>={self.detector.confidence:.2f}"), annotated, []
        msg = BlockDetectionMsg()
        msg.stamp = self.get_clock().now().to_msg()
        msg.detected = True
        msg.pixel_x = float(detection.center[0])
        msg.pixel_y = float(detection.center[1])
        msg.confidence = float(detection.confidence)
        msg.class_name = detection.class_name
        board = self.detector.board_position(detection)
        if board is None:
            msg.grid_position_valid = False
            msg.grid_column = math.nan
            msg.grid_row = math.nan
            msg.message = (
                f"{len(detections)} block(s) detected conf>={self.detector.confidence:.2f}, "
                "but board is not calibrated"
            )
        else:
            msg.grid_position_valid = True
            msg.grid_column = float(board[0])
            msg.grid_row = float(board[1])
            msg.message = f"{len(detections)} block(s) detected conf>={self.detector.confidence:.2f}"
        return msg, annotated, detections

    def _empty_detection(self, message: str) -> BlockDetectionMsg:
        msg = BlockDetectionMsg()
        msg.stamp = self.get_clock().now().to_msg()
        msg.detected = False
        msg.grid_position_valid = False
        msg.grid_column = math.nan
        msg.grid_row = math.nan
        msg.pixel_x = math.nan
        msg.pixel_y = math.nan
        msg.confidence = 0.0
        msg.class_name = ""
        msg.message = str(message)
        return msg

    def _publish_status(self, message: str) -> None:
        msg = String()
        msg.data = str(message)
        self.status_pub.publish(msg)

    def destroy_node(self) -> bool:
        self.camera.stop()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: VisionNode | None = None
    try:
        node = VisionNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
