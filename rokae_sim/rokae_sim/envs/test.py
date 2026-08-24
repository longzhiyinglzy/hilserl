import numpy as np
from rokae_sim.rokae_sim.envs.rokae_assembly_gym_env import RokaexMateAssemblyGymEnv

env = RokaexMateAssemblyGymEnv(render_mode="rgb_array", image_obs=True)
obs, info = env.reset()

print("obs keys:", obs.keys())
print("state keys:", obs["state"].keys())
if "images" in obs:
    for k, v in obs["images"].items():
        print("image", k, v.shape, v.dtype)

for i in range(200):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    if i % 20 == 0:
        print(i, "r=", r, "term=", term, "trunc=", trunc, "info=", info)

env.close()
print("step test done")
