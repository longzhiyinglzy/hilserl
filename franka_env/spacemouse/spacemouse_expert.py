from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np

try:
    from evdev import InputDevice, ecodes, list_devices
except ImportError:  # pragma: no cover - depends on the robot workstation setup.
    InputDevice = None
    ecodes = None
    list_devices = None

try:
    from franka_env.spacemouse import pyspacemouse
except Exception:  # pragma: no cover - evdev is the preferred backend.
    pyspacemouse = None


DEADZONE = 30.0
MAX_VALUE = 350.0
ROT_SCALE = 0.5


def map_spacemouse_to_action(
    percentage: float,
    deadzone: float = DEADZONE,
    motion: float = 1.0,
) -> float:
    if abs(percentage) <= deadzone:
        return 0.0
    if percentage > 0:
        normalized = (percentage - deadzone) / (100.0 - deadzone)
        return -normalized * motion
    normalized = (abs(percentage) - deadzone) / (100.0 - deadzone)
    return normalized * motion


def raw_to_action(
    raw: np.ndarray,
    deadzone: float = DEADZONE,
    max_value: float = MAX_VALUE,
    rot_scale: float = ROT_SCALE,
) -> np.ndarray:
    """
    Convert Piper-style SpaceMouse raw input [x, y, z, rx, ry, rz] to a
    normalized SERL action [x, y, z, rx, ry, rz].
    """
    percentages = (np.asarray(raw, dtype=np.float64) / max_value) * 100.0

    action_x = map_spacemouse_to_action(percentages[1], deadzone)
    action_y = map_spacemouse_to_action(percentages[0], deadzone)
    action_z = map_spacemouse_to_action(percentages[2], deadzone)

    action_ry = map_spacemouse_to_action(percentages[3], deadzone) * rot_scale
    action_rx = map_spacemouse_to_action(percentages[4], deadzone) * rot_scale
    action_rz = map_spacemouse_to_action(percentages[5], deadzone) * rot_scale

    return np.clip(
        np.array(
            [action_x, action_y, action_z, action_rx, action_ry, action_rz],
            dtype=np.float64,
        ),
        -1.0,
        1.0,
    )


class SpaceMouseExpert:
    """
    Piper-style SpaceMouse reader for SERL intervention.

    The preferred backend reads Linux evdev events directly. Relative-motion
    devices report per-frame deltas, so each call starts from zero and only the
    newly received events contribute to the returned action.
    """

    def __init__(
        self,
        device_path: Optional[str] = None,
        grab_device: bool = True,
        deadzone: float = DEADZONE,
        max_value: float = MAX_VALUE,
        rot_scale: float = ROT_SCALE,
        read_hz: float = 10.0,
    ):
        self.deadzone = float(deadzone)
        self.max_value = float(max_value)
        self.rot_scale = float(rot_scale)
        self.grab_device = bool(grab_device)
        self.read_hz = float(read_hz)

        self.device = None
        self._use_pyspacemouse = False
        self.axis = [0, 0, 0, 0, 0, 0]
        self.buttons = [0, 0]
        self.latest_raw = np.zeros(6, dtype=np.float64)
        self._latest_action = np.zeros(6, dtype=np.float64)
        self._latest_buttons = [0, 0]
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        if InputDevice is not None:
            self._connect_evdev(device_path)
        elif pyspacemouse is not None:
            pyspacemouse.open()
            self._use_pyspacemouse = True
        else:
            raise RuntimeError("No SpaceMouse backend available. Install evdev or pyspacemouse.")

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _connect_evdev(self, device_path: Optional[str]) -> None:
        if device_path is not None:
            self.device = InputDevice(device_path)
        else:
            for path in list_devices():
                try:
                    dev = InputDevice(path)
                except Exception:
                    continue
                name = getattr(dev, "name", "")
                if "3Dconnexion" in name or "SpaceMouse" in name:
                    self.device = dev
                    break

            if self.device is None:
                for i in range(40):
                    try:
                        dev = InputDevice(f"/dev/input/event{i}")
                    except Exception:
                        continue
                    name = getattr(dev, "name", "")
                    if "3Dconnexion" in name or "SpaceMouse" in name:
                        self.device = dev
                        break

        if self.device is None:
            raise RuntimeError("SpaceMouse device not found in /dev/input/event*.")

        if self.grab_device:
            try:
                self.device.grab()
            except Exception:
                pass

    def _read_evdev(self) -> Tuple[np.ndarray, list[int]]:
        frame_axis = [0, 0, 0, 0, 0, 0]
        rel_to_idx = {
            ecodes.REL_X: 0,
            ecodes.REL_Y: 1,
            ecodes.REL_Z: 2,
            ecodes.REL_RX: 3,
            ecodes.REL_RY: 4,
            ecodes.REL_RZ: 5,
        }
        abs_to_idx = {
            ecodes.ABS_X: 0,
            ecodes.ABS_Y: 1,
            ecodes.ABS_Z: 2,
            ecodes.ABS_RX: 3,
            ecodes.ABS_RY: 4,
            ecodes.ABS_RZ: 5,
        }
        got_rel_event = False

        try:
            while True:
                event = self.device.read_one()
                if event is None:
                    break

                if event.type == ecodes.EV_REL and event.code in rel_to_idx:
                    frame_axis[rel_to_idx[event.code]] += event.value
                    got_rel_event = True
                elif event.type == ecodes.EV_ABS and event.code in abs_to_idx:
                    self.axis[abs_to_idx[event.code]] = event.value
                elif event.type == ecodes.EV_KEY:
                    if event.code in (256, getattr(ecodes, "BTN_0", 256)):
                        self.buttons[0] = event.value
                    elif event.code in (257, getattr(ecodes, "BTN_1", 257)):
                        self.buttons[1] = event.value
        except Exception:
            pass

        raw = frame_axis if got_rel_event else self.axis
        return np.array(raw, dtype=np.float64), self.buttons.copy()

    def _read_pyspacemouse(self) -> Tuple[np.ndarray, list[int]]:
        state = pyspacemouse.read_all()
        raw = np.zeros(6, dtype=np.float64)
        buttons = [0, 0]

        if len(state) > 0:
            raw = np.array(
                [
                    state[0].x,
                    state[0].y,
                    state[0].z,
                    state[0].roll,
                    state[0].pitch,
                    state[0].yaw,
                ],
                dtype=np.float64,
            ) * self.max_value
            buttons = list(state[0].buttons[:2])

        return raw, buttons

    def get_action(self) -> Tuple[np.ndarray, list[int]]:
        with self._lock:
            return self._latest_action.copy(), self._latest_buttons.copy()

    def _read_loop(self) -> None:
        period = 1.0 / self.read_hz if self.read_hz > 0 else 0.0
        while self._running:
            try:
                if self._use_pyspacemouse:
                    raw, buttons = self._read_pyspacemouse()
                else:
                    raw, buttons = self._read_evdev()
                action = raw_to_action(raw, self.deadzone, self.max_value, self.rot_scale)
                with self._lock:
                    self.latest_raw = raw.copy()
                    self._latest_action = action.copy()
                    self._latest_buttons = list(buttons)
            except Exception:
                with self._lock:
                    self.latest_raw = np.zeros(6, dtype=np.float64)
                    self._latest_action = np.zeros(6, dtype=np.float64)
                    self._latest_buttons = [0, 0]
            if period > 0.0:
                time.sleep(period)

    def close(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self.device is not None:
            try:
                self.device.ungrab()
            except Exception:
                pass
            self.device = None
        if self._use_pyspacemouse and pyspacemouse is not None:
            try:
                pyspacemouse.close()
            except Exception:
                pass
        self._use_pyspacemouse = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
