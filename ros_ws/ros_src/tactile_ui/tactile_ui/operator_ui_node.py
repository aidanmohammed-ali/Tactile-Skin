from __future__ import annotations

import math
import time
import tkinter as tk
from tkinter import scrolledtext, ttk

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty, Float32, Int32, String

from tactile_interfaces.msg import ArmMove, ArmPose, ArmState, BlockDetection
from tactile_interfaces.srv import (
    ListArmPorts,
    SetArmConnection,
)
from tactile_vision.camera_defaults import DEFAULT_CAMERA_SOURCE, load_camera_source

from .image_display import bgr_to_letterboxed_photo, image_msg_to_bgr, should_render_frame


class OperatorUiNode(Node):
    def __init__(self) -> None:
        super().__init__("operator_ui_node")
        self.latest_frame = None
        self.latest_frame_seq = 0
        self.latest_arm_state: ArmState | None = None
        self.latest_detection: BlockDetection | None = None
        self.status_messages: list[str] = []
        self.gripper_open_position = int(self.declare_parameter("gripper_open_position", 1800).value)
        self.gripper_close_position = int(self.declare_parameter("gripper_close_position", 2400).value)
        self.default_confidence = float(self.declare_parameter("confidence", 0.1).value)
        configured_source = str(self.declare_parameter("camera_source", DEFAULT_CAMERA_SOURCE).value)
        camera_source_store = str(self.declare_parameter("camera_source_store", "").value).strip() or None
        self.default_camera_source = load_camera_source(configured_source, camera_source_store)
        self.create_subscription(Image, "/vision/annotated_image", self._image_callback, 10)
        self.create_subscription(ArmState, "/arm/state", self._arm_state_callback, 10)
        self.create_subscription(BlockDetection, "/vision/block_detection", self._detection_callback, 10)
        self.create_subscription(String, "/task/status", self._task_status_callback, 10)
        self.create_subscription(String, "/vision/status", self._vision_status_callback, 10)
        self.list_ports_client = self.create_client(ListArmPorts, "/arm/list_ports")
        self.connection_client = self.create_client(SetArmConnection, "/arm/set_connection")
        self.camera_source_pub = self.create_publisher(String, "/vision/camera_source", 10)
        self.confidence_pub = self.create_publisher(Float32, "/vision/confidence_threshold", 10)
        self.calibrate_pub = self.create_publisher(Bool, "/vision/calibrate_trigger", 10)
        self.detect_pub = self.create_publisher(Empty, "/vision/detect_trigger", 10)
        self.move_pub = self.create_publisher(ArmMove, "/arm/cartesian_goal", 10)
        self.gripper_pub = self.create_publisher(Int32, "/arm/gripper_position", 10)
        self.emergency_pub = self.create_publisher(Empty, "/arm/emergency_stop", 10)
        self.pick_place_pub = self.create_publisher(ArmPose, "/task/pick_place_goal", 10)

    def _image_callback(self, msg: Image) -> None:
        try:
            self.latest_frame = image_msg_to_bgr(msg)
            self.latest_frame_seq += 1
        except Exception as exc:
            self.get_logger().warn(f"Could not decode image: {exc}")

    def _arm_state_callback(self, msg: ArmState) -> None:
        self.latest_arm_state = msg

    def _detection_callback(self, msg: BlockDetection) -> None:
        self.latest_detection = msg

    def _task_status_callback(self, msg: String) -> None:
        self.status_messages.append(f"Task: {msg.data}")

    def _vision_status_callback(self, msg: String) -> None:
        self.status_messages.append(f"Vision: {msg.data}")


class OperatorUi:
    def __init__(self, node: OperatorUiNode) -> None:
        self.node = node
        self.root = tk.Tk()
        self.root.title("Tactile Block Pick Operator")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)
        self.video_photo = None
        self.rendered_frame_seq = -1
        self.last_render_time = 0.0
        self.render_interval_sec = 1.0 / 12.0
        self.video_dirty = True
        self.video_canvas_size = (0, 0)
        self.last_logged_arm_error = ""

        self.port_var = tk.StringVar()
        self.camera_var = tk.StringVar(value=self.node.default_camera_source)
        self.confidence_var = tk.StringVar(value=f"{self.node.default_confidence:.2f}")
        self.status_var = tk.StringVar(value="Ready")
        self.arm_state_var = tk.StringVar(value="Arm: unknown")
        self.detection_var = tk.StringVar(value="Block: unknown")
        self.place_x_var = tk.StringVar(value="0.12")
        self.place_y_var = tk.StringVar(value="0.00")
        self.place_z_var = tk.StringVar(value="0.08")
        self.place_angle_var = tk.StringVar(value="-1.0")
        self.move_x_var = tk.StringVar(value="0.10")
        self.move_y_var = tk.StringVar(value="0.00")
        self.move_z_var = tk.StringVar(value="0.10")
        self.move_angle_var = tk.StringVar(value="-1.0")

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=(10, 8))
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        top.columnconfigure(9, weight=1)

        ttk.Label(top, text="Port").grid(row=0, column=0, padx=(0, 4))
        self.port_combo = ttk.Combobox(top, textvariable=self.port_var, width=28, values=[])
        self.port_combo.grid(row=0, column=1, padx=(0, 4))
        ttk.Button(top, text="Refresh", command=self.refresh_ports).grid(row=0, column=2, padx=2)
        ttk.Button(top, text="Connect", command=self.connect_arm).grid(row=0, column=3, padx=2)
        ttk.Button(top, text="Disconnect", command=self.disconnect_arm).grid(row=0, column=4, padx=(2, 16))

        ttk.Label(top, text="Camera").grid(row=0, column=5, padx=(0, 4))
        ttk.Entry(top, textvariable=self.camera_var, width=38).grid(row=0, column=6, padx=(0, 4))
        ttk.Button(top, text="Apply", command=self.apply_camera_source).grid(row=0, column=7, padx=2)
        ttk.Button(top, text="Calibrate", command=self.calibrate_board).grid(row=0, column=8, padx=2)
        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=9, sticky="e")

        video_frame = ttk.Frame(self.root, padding=(10, 0, 8, 10))
        video_frame.grid(row=1, column=0, sticky="nsew")
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        self.video_canvas = tk.Canvas(video_frame, bg="#121212", highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        self.video_canvas.bind("<Configure>", self._on_video_resize)

        panel = ttk.Frame(self.root, padding=(8, 0, 10, 10))
        panel.grid(row=1, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(6, weight=1)

        ttk.Label(panel, textvariable=self.arm_state_var).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(panel, textvariable=self.detection_var).grid(row=1, column=0, sticky="ew", pady=(0, 10))

        manual = ttk.LabelFrame(panel, text="Manual Move", padding=10)
        manual.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self._pose_inputs(manual, self.move_x_var, self.move_y_var, self.move_z_var, self.move_angle_var)
        ttk.Button(manual, text="Move", command=self.send_manual_move).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        gripper = ttk.LabelFrame(panel, text="Gripper", padding=10)
        gripper.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        gripper.columnconfigure((0, 1), weight=1)
        ttk.Button(gripper, text="Open", command=lambda: self.set_gripper(self.node.gripper_open_position)).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(gripper, text="Close", command=lambda: self.set_gripper(self.node.gripper_close_position)).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        pick = ttk.LabelFrame(panel, text="Pick Place", padding=10)
        pick.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(
            pick,
            text="Final placement pose in meters/radians; pick pose comes from CV detection.",
            wraplength=260,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self._pose_inputs(
            pick,
            self.place_x_var,
            self.place_y_var,
            self.place_z_var,
            self.place_angle_var,
            row_offset=1,
            label_prefix="Place",
        )
        ttk.Button(pick, text="Run Pick-Place", command=self.run_pick_place).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        safety = ttk.Frame(panel)
        safety.grid(row=5, column=0, sticky="ew")
        ttk.Label(safety, text="Conf").grid(row=0, column=0, sticky="w", padx=(0, 4), pady=(0, 6))
        ttk.Entry(safety, textvariable=self.confidence_var, width=8).grid(row=0, column=1, sticky="ew", padx=(0, 4), pady=(0, 6))
        ttk.Button(safety, text="Apply", command=self.apply_confidence_threshold).grid(row=0, column=2, sticky="ew", pady=(0, 6))
        ttk.Button(safety, text="Detect Now", command=self.detect_once).grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 4))
        ttk.Button(safety, text="Emergency Stop", command=self.emergency_stop).grid(row=1, column=2, sticky="ew")
        safety.columnconfigure(1, weight=1)
        safety.columnconfigure(2, weight=1)

        log_frame = ttk.LabelFrame(panel, text="Log", padding=6)
        log_frame.grid(row=6, column=0, sticky="nsew", pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def _pose_inputs(
        self,
        parent: ttk.Frame,
        x_var,
        y_var,
        z_var,
        angle_var,
        row_offset: int = 0,
        label_prefix: str = "",
    ) -> None:
        prefix = f"{label_prefix} " if label_prefix else ""
        labels = ((f"{prefix}X", x_var), (f"{prefix}Y", y_var), (f"{prefix}Z", z_var), (f"{prefix}Angle", angle_var))
        for row, (label, var) in enumerate(labels):
            ttk.Label(parent, text=label).grid(row=row + row_offset, column=0, sticky="w", pady=2)
            ttk.Entry(parent, textvariable=var, width=14).grid(row=row + row_offset, column=1, sticky="ew", pady=2)
        parent.columnconfigure(1, weight=1)

    def refresh_ports(self) -> None:
        if not self.node.list_ports_client.wait_for_service(timeout_sec=0.2):
            self.set_status("Port service unavailable")
            return
        future = self.node.list_ports_client.call_async(ListArmPorts.Request())
        future.add_done_callback(self._on_ports)
        self.set_status("Refreshing ports...")

    def _on_ports(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.set_status(f"Port refresh failed: {exc}")
            return
        self.port_combo["values"] = list(response.ports)
        if response.recommended_port:
            self.port_var.set(response.recommended_port)
        self.set_status(f"Found {len(response.ports)} port(s): {', '.join(response.ports) if response.ports else 'none'}")

    def connect_arm(self) -> None:
        self._set_connection(True)

    def disconnect_arm(self) -> None:
        self._set_connection(False)

    def _set_connection(self, connect: bool) -> None:
        if not self.node.connection_client.wait_for_service(timeout_sec=0.2):
            self.set_status("Arm connection service unavailable")
            return
        request = SetArmConnection.Request()
        request.connect = connect
        request.port = self.port_var.get().strip()
        future = self.node.connection_client.call_async(request)
        future.add_done_callback(lambda fut: self._set_status_from_response(fut, "Arm"))

    def apply_camera_source(self) -> None:
        msg = String()
        msg.data = self.camera_var.get().strip()
        self.node.camera_source_pub.publish(msg)
        self.set_status("Camera source sent")

    def calibrate_board(self) -> None:
        msg = Bool()
        msg.data = True
        self.node.calibrate_pub.publish(msg)
        self.set_status("Calibration requested")

    def apply_confidence_threshold(self) -> None:
        try:
            threshold = float(self.confidence_var.get())
        except ValueError:
            self.set_status("Confidence must be a number between 0 and 1")
            return
        if not 0.0 <= threshold <= 1.0:
            self.set_status("Confidence must be between 0 and 1")
            return
        msg = Float32()
        msg.data = float(threshold)
        self.node.confidence_pub.publish(msg)
        self.set_status(f"Confidence threshold sent: {threshold:.2f}")

    def detect_once(self) -> None:
        self.node.detect_pub.publish(Empty())
        self.set_status("Detect Now requested")

    def set_gripper(self, position: int) -> None:
        msg = Int32()
        msg.data = int(position)
        self.node.gripper_pub.publish(msg)
        self.set_status(f"Gripper position sent: {msg.data}")

    def emergency_stop(self) -> None:
        self.node.emergency_pub.publish(Empty())
        self.set_status("Emergency stop sent")

    def send_manual_move(self) -> None:
        pose = self._read_pose(self.move_x_var, self.move_y_var, self.move_z_var, self.move_angle_var)
        if pose is None:
            return
        msg = ArmMove()
        msg.target_pose = pose
        msg.duration_sec = 1.2
        self.node.move_pub.publish(msg)
        self.set_status("Move goal sent")

    def run_pick_place(self) -> None:
        pose = self._read_pose(self.place_x_var, self.place_y_var, self.place_z_var, self.place_angle_var)
        if pose is None:
            return
        self.node.pick_place_pub.publish(pose)
        self.set_status("PickPlace goal sent")

    def _set_status_from_response(self, future, prefix: str) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.set_status(f"{prefix} failed: {exc}")
            return
        success = getattr(response, "success", True)
        message = getattr(response, "message", "")
        self.set_status(f"{prefix}: {'OK' if success else 'Failed'} {message}")

    def _read_pose(self, x_var, y_var, z_var, angle_var) -> ArmPose | None:
        try:
            values = [float(var.get()) for var in (x_var, y_var, z_var, angle_var)]
        except ValueError:
            self.set_status("Pose inputs must be numbers")
            return None
        pose = ArmPose()
        pose.x, pose.y, pose.z, pose.angle_rad = values
        return pose

    def run(self) -> None:
        self.refresh_ports()
        self._tick()
        self.root.mainloop()

    def _tick(self) -> None:
        if rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.0)
        self._drain_status_messages()
        self._update_status_labels()
        self.render_video()
        self.root.after(30, self._tick)

    def _drain_status_messages(self) -> None:
        messages = self.node.status_messages
        if not messages:
            return
        self.node.status_messages = []
        for message in messages:
            self.set_status(message)

    def set_status(self, message: str, log: bool = True) -> None:
        text = str(message)
        self.status_var.set(text if len(text) <= 90 else text[:87] + "...")
        if log:
            self.append_log(text)

    def append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _update_status_labels(self) -> None:
        state = self.node.latest_arm_state
        if state is None:
            self.arm_state_var.set("Arm: unknown")
        else:
            pose = state.current_pose
            self.arm_state_var.set(
                f"Arm: {state.mode} {'connected' if state.connected else 'disconnected'} "
                f"busy={int(state.busy)} port={state.current_port or '-'} "
                f"pose=({pose.x:.3f},{pose.y:.3f},{pose.z:.3f})"
            )
            if state.last_error and state.last_error != self.last_logged_arm_error:
                self.append_log(f"Arm error: {state.last_error}")
                self.last_logged_arm_error = state.last_error
            elif not state.last_error:
                self.last_logged_arm_error = ""
        detection = self.node.latest_detection
        if detection is None:
            self.detection_var.set("Block: unknown")
        elif detection.detected and detection.grid_position_valid:
            self.detection_var.set(
                f"Block: grid=({detection.grid_column:.2f},{detection.grid_row:.2f}) "
                f"conf={detection.confidence:.2f}"
            )
        elif detection.detected:
            self.detection_var.set(f"Block: pixel=({detection.pixel_x:.0f},{detection.pixel_y:.0f})")
        else:
            self.detection_var.set(f"Block: {detection.message}")

    def render_video(self) -> None:
        frame = self.node.latest_frame
        if frame is None:
            return
        now = time.monotonic()
        frame_seq = self.node.latest_frame_seq
        width = max(1, self.video_canvas.winfo_width())
        height = max(1, self.video_canvas.winfo_height())
        canvas_size = (width, height)
        if not should_render_frame(
            frame_seq=frame_seq,
            rendered_frame_seq=self.rendered_frame_seq,
            canvas_size=canvas_size,
            rendered_canvas_size=self.video_canvas_size,
            video_dirty=self.video_dirty,
            now=now,
            last_render_time=self.last_render_time,
            min_interval_sec=self.render_interval_sec,
        ):
            return
        try:
            self.video_photo = bgr_to_letterboxed_photo(frame, width, height)
        except Exception as exc:
            self.set_status(f"Video render failed: {exc}")
            return
        self.video_canvas.delete("all")
        self.video_canvas.create_image(0, 0, anchor="nw", image=self.video_photo)
        self.rendered_frame_seq = frame_seq
        self.video_canvas_size = canvas_size
        self.video_dirty = False
        self.last_render_time = now

    def _on_video_resize(self, event) -> None:
        self.video_dirty = True
        self.video_canvas_size = (int(event.width), int(event.height))

    def _on_close(self) -> None:
        self.root.quit()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OperatorUiNode()
    ui = OperatorUi(node)
    try:
        ui.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
