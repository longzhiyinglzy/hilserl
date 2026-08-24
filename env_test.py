from examples.experiments.rokae_assembly_sim.config import TrainConfig
import numpy as np

# ✅ 用 human 方便你看 + 试干预
env = TrainConfig().get_environment(fake_env=False, classifier=False, render_mode="human")

obs, info = env.reset()
print("reset info:", info)

step = 0
while True:
    # ✅ policy action：建议先用 0，避免随机动作导致“乱动”
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    obs, reward, done, truncated, info = env.step(action)

    # ✅ H 请求 reset（由 wrapper 往 info 里塞 request_reset）
    if info.get("request_reset", False):
        obs, info = env.reset()
        print("reset info:", info)
        continue

    # 可选：降低打印频率
    if step % 20 == 0:
        succeed = info.get("succeed", info.get("success", False))
        intervened = info.get("intervened", False)
        print(f"step={step} reward={reward:.3f} intervened={intervened} succeed={succeed}")

    step += 1

    if done or truncated:
        obs, info = env.reset()
        print("reset info:", info)
