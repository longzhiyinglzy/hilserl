from rokae_sim.rokae_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

# 用 gymnasium 的 register（不要用 gym）
from gymnasium.envs.registration import register

# 你的装配任务（state-only）
register(
    id="RokaeAssembly-v0",
    entry_point="rokae_sim.envs:RokaexMateAssemblyGymEnv",
    max_episode_steps=500,  # 10s / 0.02 = 500 步（按你的 time_limit/control_dt 调）
    kwargs={
        "render_mode": "rgb_array",
        "image_obs": False,
    },
)

# 你的装配任务（vision）
register(
    id="RokaeAssemblyVision-v0",
    entry_point="rokae_sim.envs:RokaexMateAssemblyGymEnv",
    max_episode_steps=500,
    kwargs={
        "render_mode": "rgb_array",
        "image_obs": True,
    },
)
