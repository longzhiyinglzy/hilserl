import importlib
import sys

# 导入内层真正的包，触发 rokae_sim/rokae_sim/__init__.py 里的 register()
importlib.import_module("rokae_sim.rokae_sim")

# 把 rokae_sim.envs 映射到内层 rokae_sim.rokae_sim.envs，保证 entry_point="rokae_sim.envs:..." 能找到
sys.modules[__name__ + ".envs"] = importlib.import_module("rokae_sim.rokae_sim.envs")
