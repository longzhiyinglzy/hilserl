import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig, FrankaEnv
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig


class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    EXPECTED_JOINT_DOF = 6

    # Replace camera serials with your local setup.
    REALSENSE_CAMERAS = {
        "side": {
            "serial_number": "409122274027",
            "dim": (1280, 720),
            "exposure": 30000,
        },
        "wrist1": {
            "serial_number": "352122271950",
            "dim": (1280, 720),
            "exposure": 30000,
        },
        "wrist2": {
            "serial_number": "352122272208",
            "dim": (1280, 720),
            "exposure": 30000,
        },
    }
    # IMAGE_CROP = {
    #     "side": lambda img: img,
    #     "wrist1": lambda img: img,
    #     "wrist2": lambda img: img,
    # }

    # IMAGE_CROP = {
    # "side": lambda img: img[249:544, 373:704],
    # "wrist1": lambda img: img[18:354, 614:932],
    # "wrist2": lambda img: img[337:663, 308:865],
    # }

    IMAGE_CROP = {
        "side": lambda img: img[194:644, 361:743],
        "wrist1": lambda img: img[96:451, 638:884],
        "wrist2": lambda img: img[338:683, 315:850],
    }




    # Cartesian reset pose requested by user (mm/deg -> m/rad):
    # 565, -274, 345, -180, 0, 0
    # 412.54, -47.86, 299.84, -180, 0.05, -20.86
    TARGET_POSE = np.array([0.41254, -0.04786, 0.29984, -np.pi, 0.00087, -0.36408])
    RESET_POSE = TARGET_POSE.copy()
    REWARD_THRESHOLD = np.array([0.004, 0.004, 0.004, 0.08, 0.08, 0.08])
    USE_POSE_REWARD = False

    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.002
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = np.array([0.0002, 0.01, 1.0])

    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.4, 0.4, 0.6, 0.0, 0.0, 0.30])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.4, 0.4, 0.6, 0.0, 0.0, 0.30])

    # Rokae bridge treats these routes as no-op by default, kept for API compatibility.
    COMPLIANCE_PARAM = {}
    PRECISION_PARAM = {}
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 300
    JOINT_RESET_PERIOD = 0
    RESET_MOVE_TIMEOUT_S = 5.0
    RESET_CONTROL_MODE = 4


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["side", "wrist1", "wrist2"]
    classifier_keys = ["side", "wrist1", "wrist2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def _classifier_ckpt_path(self):
        local_path = os.path.abspath("classifier_ckpt/")
        if os.path.exists(local_path):
            return local_path

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        repo_path = os.path.join(repo_root, "classifier_ckpt")
        if os.path.exists(repo_path):
            return repo_path

        return local_path

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = FrankaEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env, action_indices=[0, 1, 2, 5])
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier:
            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=self._classifier_ckpt_path(),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                prob = sigmoid(classifier_fn(obs))
                return int(float(jax.device_get(jnp.squeeze(prob))) > 0.7)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
