"""
ROS2-to-HTTP bridge server for Rokae real robot control.

This file keeps the same HTTP API used by FrankaEnv so existing JAX training
code can run against Rokae hardware with minimal changes.
"""

from __future__ import annotations

import atexit
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from absl import app, flags
from flask import Flask, jsonify, request
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from robot_interfaces.msg import (
    EndControlStruct,
    EndPosStruct,
    EndVelStruct,
    ForceSensorStruct,
    JointPosStruct,
)

try:
    from robot_interfaces.msg import GripperControlStruct, GripperStatusStruct

    HAS_GRIPPER_MSG = True
except Exception:
    HAS_GRIPPER_MSG = False


FLAGS = flags.FLAGS

flags.DEFINE_string("flask_url", "127.0.0.1", "HTTP server host.")
flags.DEFINE_integer("flask_port", 5000, "HTTP server port.")

flags.DEFINE_string("joint_positions_topic", "/joint_positions", "ROS2 joint topic.")
flags.DEFINE_string("end_pos_topic", "/EndPos_Topic", "ROS2 end-effector pose topic.")
flags.DEFINE_string("end_vel_topic", "/end_velocity", "ROS2 end-effector velocity topic.")
flags.DEFINE_string("end_control_topic", "/EndControl_Topic", "ROS2 end-effector control topic.")
flags.DEFINE_string("force_sensor_topic", "/ForceSensor_Topic", "ROS2 force sensor topic.")

flags.DEFINE_integer("control_mode", 3, "Normal control mode sent to EndControlStruct.")
flags.DEFINE_integer("reset_control_mode", 4, "Reset control mode sent to EndControlStruct.")
flags.DEFINE_float("wait_for_state_s", 3.0, "Wait time for first joint/end state.")

flags.DEFINE_list(
    "home_pose",
    ["0.632", "0.0123", "0.14", "-3.1415926", "0.0", "0.0419"],
    "Home pose [x,y,z,rx,ry,rz] in meters/radians for /jointreset.",
)
flags.DEFINE_float("reset_hold_s", 0.2, "Hold time after sending reset mode command.")

flags.DEFINE_bool("enable_gripper_ros2", False, "Enable ROS2 gripper pub/sub.")
flags.DEFINE_string("gripper_control_topic", "/GripperControl_Topic", "ROS2 gripper control topic.")
flags.DEFINE_string("gripper_status_topic", "/GripperStatus_Topic", "ROS2 gripper status topic.")
flags.DEFINE_integer("gripper_open_command", 1000, "Gripper open position command.")
flags.DEFINE_integer("gripper_close_command", 0, "Gripper close position command.")
flags.DEFINE_integer("gripper_min_position", 0, "Gripper raw min position for normalization.")
flags.DEFINE_integer("gripper_max_position", 1000, "Gripper raw max position for normalization.")
flags.DEFINE_integer("gripper_force_command", 300, "Gripper force command.")
flags.DEFINE_integer("gripper_vel_command", 300, "Gripper velocity command.")


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_pose_list(values: list[str], size: int, default: float = 0.0) -> np.ndarray:
    out = [_safe_float(v, default=default) for v in values]
    if len(out) < size:
        out.extend([default] * (size - len(out)))
    return np.asarray(out[:size], dtype=np.float64)


class RokaeRos2Bridge(Node):
    def __init__(self):
        super().__init__("rokae_http_bridge")

        self._lock = threading.Lock()
        self._pose_euler = np.zeros(6, dtype=np.float64)
        self._pose_quat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._vel = np.zeros(6, dtype=np.float64)
        self._force = np.zeros(3, dtype=np.float64)
        self._torque = np.zeros(3, dtype=np.float64)
        self._q = np.zeros(6, dtype=np.float64)
        self._dq = np.zeros(6, dtype=np.float64)
        self._jacobian = np.zeros((6, 6), dtype=np.float64)
        self._gripper_pos = 0.0

        self._latest_joint_t: Optional[float] = None
        self._latest_ee_t: Optional[float] = None
        self._latest_q: Optional[np.ndarray] = None
        self._latest_ee: Optional[np.ndarray] = None
        self._has_end_vel = False

        self._got_joint = False
        self._got_ee = False

        self.pub_end = self.create_publisher(
            EndControlStruct, FLAGS.end_control_topic, 10
        )
        self.sub_joint = self.create_subscription(
            JointPosStruct, FLAGS.joint_positions_topic, self._on_joint, 10
        )
        self.sub_ee = self.create_subscription(
            EndPosStruct, FLAGS.end_pos_topic, self._on_ee, 10
        )
        self.sub_ee_vel = self.create_subscription(
            EndVelStruct, FLAGS.end_vel_topic, self._on_ee_vel, 10
        )
        self.sub_force = self.create_subscription(
            ForceSensorStruct, FLAGS.force_sensor_topic, self._on_force, 10
        )

        self.gripper_pub = None
        self.gripper_sub = None
        if FLAGS.enable_gripper_ros2 and HAS_GRIPPER_MSG:
            self.gripper_pub = self.create_publisher(
                GripperControlStruct, FLAGS.gripper_control_topic, 10
            )
            self.gripper_sub = self.create_subscription(
                GripperStatusStruct, FLAGS.gripper_status_topic, self._on_gripper_status, 10
            )
        elif FLAGS.enable_gripper_ros2 and not HAS_GRIPPER_MSG:
            self.get_logger().warn(
                "enable_gripper_ros2=True but Gripper message types are unavailable; "
                "falling back to internal-only gripper state."
            )

    def has_minimum_state(self) -> bool:
        with self._lock:
            return self._got_joint and self._got_ee

    def _now(self) -> float:
        return time.time()

    def _on_joint(self, msg: JointPosStruct) -> None:
        q = np.array(
            [
                msg.joint1_pos.data,
                msg.joint2_pos.data,
                msg.joint3_pos.data,
                msg.joint4_pos.data,
                msg.joint5_pos.data,
                msg.joint6_pos.data,
            ],
            dtype=np.float64,
        )
        t = self._now()
        with self._lock:
            if self._latest_joint_t is not None and self._latest_q is not None:
                dt = max(1e-6, t - self._latest_joint_t)
                self._dq = (q - self._latest_q) / dt
            self._latest_joint_t = t
            self._latest_q = q.copy()
            self._q = q
            if self._jacobian.shape[1] != q.size:
                self._jacobian = np.zeros((6, q.size), dtype=np.float64)
            self._got_joint = True

    def _on_ee(self, msg: EndPosStruct) -> None:
        euler = np.array(
            [
                msg.x_pos.data,
                msg.y_pos.data,
                msg.z_pos.data,
                msg.rx_pos.data,
                msg.ry_pos.data,
                msg.rz_pos.data,
            ],
            dtype=np.float64,
        )
        quat = R.from_euler("xyz", euler[3:]).as_quat()
        pose = np.concatenate([euler[:3], quat], axis=0)

        t = self._now()
        with self._lock:
            if (not self._has_end_vel) and self._latest_ee_t is not None and self._latest_ee is not None:
                dt = max(1e-6, t - self._latest_ee_t)
                self._vel = (euler - self._latest_ee) / dt
            self._latest_ee_t = t
            self._latest_ee = euler.copy()
            self._pose_euler = euler
            self._pose_quat = pose
            self._got_ee = True

    def _on_ee_vel(self, msg: EndVelStruct) -> None:
        vel = np.array(
            [
                msg.x_vel.data,
                msg.y_vel.data,
                msg.z_vel.data,
                msg.rx_vel.data,
                msg.ry_vel.data,
                msg.rz_vel.data,
            ],
            dtype=np.float64,
        )
        with self._lock:
            self._vel = vel
            self._has_end_vel = True

    def _on_force(self, msg: ForceSensorStruct) -> None:
        force = np.array([msg.fx.data, msg.fy.data, msg.fz.data], dtype=np.float64)
        torque = np.array([msg.mx.data, msg.my.data, msg.mz.data], dtype=np.float64)
        with self._lock:
            self._force = force
            self._torque = torque

    def _on_gripper_status(self, msg) -> None:
        mn = FLAGS.gripper_min_position
        mx = FLAGS.gripper_max_position
        raw = float(msg.gripper_real_pos.data)
        ratio = 0.0 if mx <= mn else (raw - mn) / (mx - mn)
        with self._lock:
            self._gripper_pos = float(np.clip(ratio, 0.0, 1.0))

    def _publish_gripper_command(self, target: int, reset: int = 0, initial: int = 0) -> None:
        if self.gripper_pub is None:
            return
        msg = GripperControlStruct()
        msg.gripper_initial.data = int(initial)
        msg.gripper_force_control.data = int(FLAGS.gripper_force_command)
        msg.gripper_pos_control.data = int(target)
        msg.gripper_vel_control.data = int(FLAGS.gripper_vel_command)
        msg.gripper_reset.data = int(reset)
        self.gripper_pub.publish(msg)

    def activate_gripper(self) -> None:
        self._publish_gripper_command(FLAGS.gripper_open_command, initial=1)

    def reset_gripper(self) -> None:
        self._publish_gripper_command(FLAGS.gripper_open_command, reset=1)
        self._publish_gripper_command(FLAGS.gripper_open_command, initial=1)

    def open_gripper(self) -> None:
        self._publish_gripper_command(FLAGS.gripper_open_command)
        with self._lock:
            self._gripper_pos = 0.0

    def close_gripper(self, slow: bool = False) -> None:
        _ = slow  # kept for API compatibility
        self._publish_gripper_command(FLAGS.gripper_close_command)
        with self._lock:
            self._gripper_pos = 1.0

    def move_gripper(self, pos_255: int) -> None:
        pos_255 = int(np.clip(pos_255, 0, 255))
        t = FLAGS.gripper_open_command + (
            (FLAGS.gripper_close_command - FLAGS.gripper_open_command) * (pos_255 / 255.0)
        )
        self._publish_gripper_command(int(t))
        with self._lock:
            self._gripper_pos = float(pos_255 / 255.0)

    def publish_target(self, target_euler: np.ndarray, control_mode: int) -> None:
        msg = EndControlStruct()
        msg.x_ctrl.data = float(target_euler[0])
        msg.y_ctrl.data = float(target_euler[1])
        msg.z_ctrl.data = float(target_euler[2])
        msg.rx_ctrl.data = float(target_euler[3])
        msg.ry_ctrl.data = float(target_euler[4])
        msg.rz_ctrl.data = float(target_euler[5])
        msg.control_mode = int(control_mode)
        self.pub_end.publish(msg)

    def current_pose_euler(self) -> np.ndarray:
        with self._lock:
            return self._pose_euler.copy()

    def get_state_dict(self) -> dict:
        with self._lock:
            return {
                "pose": self._pose_quat.tolist(),
                "vel": self._vel.tolist(),
                "force": self._force.tolist(),
                "torque": self._torque.tolist(),
                "q": self._q.tolist(),
                "dq": self._dq.tolist(),
                "jacobian": self._jacobian.tolist(),
                "gripper_pos": float(self._gripper_pos),
            }


def main(_):
    if not rclpy.ok():
        rclpy.init(args=None)

    bridge = RokaeRos2Bridge()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(bridge)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def _shutdown():
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            bridge.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    atexit.register(_shutdown)

    wait_until = time.time() + max(0.0, FLAGS.wait_for_state_s)
    while time.time() < wait_until:
        if bridge.has_minimum_state():
            break
        time.sleep(0.02)

    webapp = Flask(__name__)

    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        return "No-op on Rokae ROS2 bridge"

    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        return "No-op on Rokae ROS2 bridge"

    @webapp.route("/set_load", methods=["POST"])
    def set_load():
        return "No-op on Rokae ROS2 bridge"

    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        return jsonify({"pose": bridge.current_pose_euler().tolist()})

    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": bridge.get_state_dict()["pose"]})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": bridge.get_state_dict()["vel"]})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": bridge.get_state_dict()["force"]})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": bridge.get_state_dict()["torque"]})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": bridge.get_state_dict()["q"]})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": bridge.get_state_dict()["dq"]})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": bridge.get_state_dict()["jacobian"]})

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": bridge.get_state_dict()["gripper_pos"]})

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        target = _parse_pose_list(FLAGS.home_pose, 6, default=0.0)
        bridge.publish_target(target, FLAGS.reset_control_mode)
        time.sleep(max(0.0, FLAGS.reset_hold_s))
        bridge.publish_target(target, FLAGS.control_mode)
        return "Reset Joint"

    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        bridge.activate_gripper()
        return "Activated"

    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        bridge.reset_gripper()
        return "Reset"

    @webapp.route("/open_gripper", methods=["POST"])
    def open_gripper():
        bridge.open_gripper()
        return "Opened"

    @webapp.route("/close_gripper", methods=["POST"])
    def close_gripper():
        bridge.close_gripper(slow=False)
        return "Closed"

    @webapp.route("/close_gripper_slow", methods=["POST"])
    def close_gripper_slow():
        bridge.close_gripper(slow=True)
        return "Closed"

    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        data = request.json or {}
        bridge.move_gripper(int(data.get("gripper_pos", 0)))
        return "Moved Gripper"

    @webapp.route("/clearerr", methods=["POST"])
    def clear_err():
        return "Clear"

    @webapp.route("/pose", methods=["POST"])
    def pose():
        data = request.json or {}
        arr = np.asarray(data.get("arr", []), dtype=np.float64).reshape(-1)
        if arr.size == 7:
            target = np.concatenate(
                [arr[:3], R.from_quat(arr[3:]).as_euler("xyz")], axis=0
            )
        elif arr.size == 6:
            target = arr.copy()
        else:
            return jsonify({"error": "Expected pose length 6 or 7"}), 400
        control_mode = int(data.get("control_mode", FLAGS.control_mode))
        bridge.publish_target(target, control_mode)
        return "Moved"

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        return jsonify(bridge.get_state_dict())

    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        return "Updated compliance parameters"

    webapp.run(host=FLAGS.flask_url, port=FLAGS.flask_port)


if __name__ == "__main__":
    app.run(main)
