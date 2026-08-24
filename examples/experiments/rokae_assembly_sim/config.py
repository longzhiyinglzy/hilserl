# examples/experiments/rokae_assembly_sim/config.py

import os
import time
import numpy as np
import gymnasium as gym

from franka_env.envs.wrappers import MultiCameraBinaryRewardClassifierWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from examples.experiments.config import DefaultTrainingConfig
from franka_env.envs.franka_env import DefaultEnvConfig

from rokae_sim.rokae_sim.envs.rokae_assembly_gym_env import RokaexMateAssemblyGymEnv


class EnvConfig(DefaultEnvConfig):
    """
    仿真环境配置：
    - 保留 REALSENSE_CAMERAS / IMAGE_CROP 的键名，减少 pipeline 改动
    - 用 CAMERA_NAME_MAP 把逻辑相机名映射到 Mujoco camera name（scene.xml 中定义）
    """
    REALSENSE_CAMERAS = {
        "wrist_1": {"dim": (128, 128)},
        "wrist_2": {"dim": (128, 128)},
        # 如需第三路：打开并同步 TrainConfig.image_keys/classifier_keys
        # "front": {"dim": (128, 128)},
    }

    CAMERA_NAME_MAP = {
        "wrist_1": "power_cam_front",
        "wrist_2": "power_cam_back",
        # "front": "front",
    }

    IMAGE_CROP = {
        "wrist_1": lambda img: img,
        "wrist_2": lambda img: img,
        # "front": lambda img: img,
    }

    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 500  # 10s / 0.02 = 500


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]

    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"

    # 你的 env 目前 action 是 6 维（无夹爪），建议用 fixed-gripper
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, render_mode="human"):
        cfg = EnvConfig()

        if fake_env:
            render_mode = "rgb_array"

        env = RokaexMateAssemblyGymEnv(
            render_mode=render_mode,
            image_obs=True,
            config=cfg,
            control_dt=0.02,
            physics_dt=0.001,
            time_limit=10.0,
        )

        # ✅ teleop 风格干预：方向键/Shift/H/Space（按住即接管）
        # if (not fake_env) and (render_mode == "human"):
        #     env = TeleopKeyboardIntervention(env, action_length=0.3)
        if (not fake_env) and (render_mode == "human"):
            env = TeleopLikeIntervention(env, STEP=0.0005, REPEAT_HZ=10.0)

        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # ✅ 可选：reward classifier（延迟导入 jax，避免无谓的 CUDA 探测/版本问题）
        if classifier:
            import jax
            import jax.numpy as jnp
            from serl_launcher.networks.reward_classifier import load_classifier_func

            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier_fn(obs)) > 0.85)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        return env



class TeleopLikeIntervention(gym.Wrapper):
    """
    完全按 teleop.py 的逻辑：
      - Left  : y_neg
      - Right : y_pos
      - Up    : x_pos
      - Down  : x_neg
      - LShift: z_neg
      - RShift: z_pos

    - Step 大小：STEP（米）
    - 长按重复：REPEAT_HZ（Hz）
    - Space：切换 intervened（干预开/关）
    - H：request_reset（外层收到 info['request_reset'] 后 reset）
    """
    def __init__(self, env, STEP=0.0005, REPEAT_HZ=10.0):
        super().__init__(env)
        from pynput import keyboard

        self.STEP = float(STEP)
        self.HOLD_REPEAT_DT = 1.0 / float(REPEAT_HZ)

        self.intervened = False
        self._request_reset = False

        self.key_states = {
            "x_pos": False, "x_neg": False,
            "y_pos": False, "y_neg": False,
            "z_pos": False, "z_neg": False,
        }

        # teleop 的节流
        self._prev_pressed = False
        self._last_apply_t = time.time()

        def on_press(key):
            try:
                # ✅ 完全照你 teleop.py
                if key == keyboard.Key.left:
                    self.key_states["y_neg"] = True
                elif key == keyboard.Key.right:
                    self.key_states["y_pos"] = True
                elif key == keyboard.Key.up:
                    self.key_states["x_pos"] = True
                elif key == keyboard.Key.down:
                    self.key_states["x_neg"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["z_neg"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["z_pos"] = True

                elif key == keyboard.Key.space:
                    self.intervened = not self.intervened
                    try:
                        self.env.intervened = self.intervened
                    except Exception:
                        pass
                    print(f"[teleop] intervened={self.intervened}")

                elif hasattr(key, "char") and key.char in ("h", "H"):
                    self._request_reset = True
                    print("[teleop] request_reset=True")
            except Exception:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.left:
                    self.key_states["y_neg"] = False
                elif key == keyboard.Key.right:
                    self.key_states["y_pos"] = False
                elif key == keyboard.Key.up:
                    self.key_states["x_pos"] = False
                elif key == keyboard.Key.down:
                    self.key_states["x_neg"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["z_neg"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["z_pos"] = False
            except Exception:
                pass

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def _any_pressed(self):
        return any(self.key_states.values())

    def _get_step_sign(self):
        sx = (1 if self.key_states["x_pos"] else 0) + (-1 if self.key_states["x_neg"] else 0)
        sy = (1 if self.key_states["y_pos"] else 0) + (-1 if self.key_states["y_neg"] else 0)
        sz = (1 if self.key_states["z_pos"] else 0) + (-1 if self.key_states["z_neg"] else 0)
        return sx, sy, sz

    def _apply_target_pos_step_if_needed(self):
        """按 teleop 的 10Hz repeat 规则更新 unwrapped._target_pos"""
        if not self.intervened:
            return

        pressed = self._any_pressed()
        now = time.time()

        need_apply = False
        if pressed and not self._prev_pressed:
            need_apply = True
        elif pressed and (now - self._last_apply_t) >= self.HOLD_REPEAT_DT:
            need_apply = True

        if need_apply:
            sx, sy, sz = self._get_step_sign()
            step_vec = np.array([sx, sy, sz], dtype=np.float64) * self.STEP

            if np.any(step_vec):
                u = self.env.unwrapped

                # 这些字段在你的 Mujoco env 里一般存在：_target_pos, _mocap_id, _data
                if not hasattr(u, "_target_pos"):
                    raise RuntimeError("env.unwrapped 没有 _target_pos，无法用 teleop 方式直接改目标位置")

                u._target_pos = np.asarray(u._target_pos, dtype=np.float64) + step_vec

                # 如果环境用 mocap target，就同步 mocap_pos（和你 env.step 里一致）
                if hasattr(u, "_mocap_id") and hasattr(u, "_data"):
                    u._data.mocap_pos[u._mocap_id] = u._target_pos

            self._last_apply_t = now

        self._prev_pressed = pressed

    def step(self, action):
        # 干预时：先按 teleop 更新 target_pos，然后给 env 传 0 action（避免 action 再改一遍 target_pos）
        if self.intervened:
            self._apply_target_pos_step_if_needed()
            action = np.zeros(self.action_space.shape, dtype=np.float32)

        obs, rew, term, trunc, info = self.env.step(action)
        info = dict(info)
        info["intervened"] = bool(self.intervened)

        if self._request_reset:
            info["request_reset"] = True
            self._request_reset = False

        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        self._request_reset = False
        self._prev_pressed = False
        self._last_apply_t = time.time()
        return self.env.reset(**kwargs)

    def close(self):
        try:
            self._listener.stop()
        except Exception:
            pass
        return self.env.close()

