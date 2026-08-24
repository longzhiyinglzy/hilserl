# rokae_assembly_gym_env.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, List

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from rokae_sim.rokae_sim.controllers import opspace
from rokae_sim.rokae_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "scene.xml"

# 精密装配：建议更小的步长（位置 m，姿态 rad，夹爪增量比例）
_DEFAULT_ACTION_SCALE = np.asarray([0.0005, 0.01, 1.0], dtype=np.float64)

# 按你的工位调整
_CARTESIAN_BOUNDS = np.asarray([[0.20, -0.25, 0.00], [0.75, 0.25, 0.60]], dtype=np.float64)

# 你的 xml 里已有这些名字（scene.xml / xMateSR3_with_gripper.xml）
_DEFAULT_MOCAP_BODY = "target"
_DEFAULT_TCP_SITE = "tcp"
_DEFAULT_ASSEMBLY_BODY = "assembly_target"

# 你 teleop.py 里给的 home
_DEFAULT_HOME_Q = np.array(
    [0.0, 0.978361765, -0.924396185, 0.0, -1.235309140, 5.23598776e-05],
    dtype=np.float64,
)

# 默认使用 scene.xml 里定义的相机（你有 front / power_cam_front / power_cam_back）
_DEFAULT_CAMERA_NAMES = ("front", "power_cam_front", "power_cam_back")


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # (w, x, y, z)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def axis_angle_to_quat(a: np.ndarray) -> np.ndarray:
    ang = float(np.linalg.norm(a))
    if ang < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = a / ang
    s = np.sin(ang / 2.0)
    return np.array([np.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    d = abs(float(np.dot(q1, q2)))
    d = float(np.clip(d, -1.0, 1.0))
    return 2.0 * float(np.arccos(d))


def mat_to_quat(mat3x3: np.ndarray) -> np.ndarray:
    """MuJoCo: mju_mat2Quat 输入是 9 个数的 row-major."""
    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, np.asarray(mat3x3, dtype=np.float64).reshape(9))
    return q


class RokaexMateAssemblyGymEnv(MujocoGymEnv):
    """
    面向 HIL-SERL / SERL 的 Gymnasium 环境骨架：
    - action: 6DoF (dx,dy,dz, dRx,dRy,dRz) 归一化到 [-1,1]
      (如果后面你加入夹爪关节和 actuator，可扩展到 7 维)
    - obs: dict(state=..., images=...)，结构尽量贴 hil-serl-sim
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = _DEFAULT_ACTION_SCALE,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.001,  # 你 XML option timestep=0.001
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(height=128, width=128, camera_id=-1, mode="rgb_array"),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
        camera_names: Optional[Sequence[str]] = None,
        # 如果你要完全对齐 hil-serl 的 config.REALSENSE_CAMERAS，可传 config 并提供映射
        config: Any = None,
        # 成功判据（装配更严格）
        pos_tol: float = 0.001,  # 1mm
        ori_tol_deg: float = 3.0,
    ):
        self._action_scale = np.asarray(action_scale, dtype=np.float64)
        self.render_mode = render_mode
        self.image_obs = bool(image_obs)

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self._pos_tol = float(pos_tol)
        self._ori_tol = float(ori_tol_deg) * np.pi / 180.0

        # ---------- 1) mocap body ----------
        self._mocap_body_name = _DEFAULT_MOCAP_BODY
        mocap_body = self._model.body(self._mocap_body_name)
        if mocap_body.mocapid < 0:
            raise RuntimeError(f"Body '{self._mocap_body_name}' 不是 mocap body（mocapid<0）。")
        self._mocap_id = int(mocap_body.mocapid)

        # ---------- 2) tcp site ----------
        self._tcp_site_name = _DEFAULT_TCP_SITE
        self._tcp_site_id = int(self._model.site(self._tcp_site_name).id)

        # ---------- 3) assembly target body ----------
        self._assembly_body_name = _DEFAULT_ASSEMBLY_BODY
        self._assembly_body_id = int(self._model.body(self._assembly_body_name).id)

        # ---------- 4) arm actuators/joints ----------
        self._arm_ctrl_ids, self._arm_joint_ids, self._gripper_ctrl_ids = self._infer_actuators_and_joints()
        # opspace.py 里 dof_ids 要能索引 qpos/qvel/Jacobian/M 的列（对 1DoF hinge，qpos 和 dof 索引一致）
        self._arm_dof_ids = np.array([int(self._model.jnt_dofadr[j]) for j in self._arm_joint_ids], dtype=np.int32)
        self._arm_qpos_adr = np.array([int(self._model.jnt_qposadr[j]) for j in self._arm_joint_ids], dtype=np.int32)

        # home posture：长度必须等于 arm joint 数
        if len(_DEFAULT_HOME_Q) != len(self._arm_joint_ids):
            raise RuntimeError(
                f"HOME_Q 维度不匹配：HOME_Q={len(_DEFAULT_HOME_Q)} vs arm_joints={len(self._arm_joint_ids)}"
            )
        self._home_q = _DEFAULT_HOME_Q.copy()

        # 控制目标（pos/quat）缓存
        self._target_pos = np.zeros(3, dtype=np.float64)
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # ---------- 5) cameras / image keys ----------
        cams = list(camera_names) if camera_names is not None else list(_DEFAULT_CAMERA_NAMES)
        existing = {self._model.camera(i).name for i in range(self._model.ncam)}
        cams = [c for c in cams if c in existing]
        if len(cams) == 0:
            cams = [self._model.camera(0).name] if self._model.ncam > 0 else []

        # hil-serl-sim 通常用 config.REALSENSE_CAMERAS 作为 images 的 key（见其 pick_cube_sim 做法）
        if config is not None and hasattr(config, "REALSENSE_CAMERAS"):
            self._image_keys: List[str] = list(config.REALSENSE_CAMERAS)
            # 可选：提供 key->mujoco_camera_name 映射；否则假设同名
            cam_map = getattr(config, "CAMERA_NAME_MAP", {})
            self._camera_for_key = {k: cam_map.get(k, k) for k in self._image_keys}
        else:
            self._image_keys = cams
            self._camera_for_key = {k: k for k in self._image_keys}

        # ---------- 6) spaces ----------
        state_space: Dict[str, spaces.Space] = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),  # xyz + quat(wxyz)
            "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),   # v(3) + w(3)
            "gripper_pose": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            "tcp_force": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "tcp_torque": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            # task extras（精密装配建议保留）
            "assembly_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "assembly_quat": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "rel_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        }

        obs_space: Dict[str, spaces.Space] = {"state": spaces.Dict(state_space)}

        if self.image_obs:
            obs_space["images"] = spaces.Dict(
                {
                    k: spaces.Box(
                        0, 255, shape=(render_spec.height, render_spec.width, 3), dtype=np.uint8
                    )
                    for k in self._image_keys
                }
            )

        self.observation_space = spaces.Dict(obs_space)

        # 动作：dx dy dz dRx dRy dRz (+grasp 可选)
        self._has_gripper = len(self._gripper_ctrl_ids) > 0
        act_dim = 6 + (1 if self._has_gripper else 0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        # ---------- 7) renderers ----------
        # rgb_array：用 mujoco.Renderer（稳定、无需 gymnasium MujocoRenderer 的 width/height 约束）
        self._rgb_renderer = mujoco.Renderer(self._model, height=render_spec.height, width=render_spec.width)
        # human：懒加载 viewer（避免 __init__ 里就 render 导致你遇到的 assert）
        self._human_viewer = None

        # bookkeeping
        self._step_count = 0

    # ------------------ Gym API ------------------

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            # MujocoGymEnv 里用的是 RandomState；这里跟着用即可
            self._random = np.random.RandomState(seed)

        mujoco.mj_resetData(self._model, self._data)

        # arm to home
        self._data.qpos[self._arm_qpos_adr] = self._home_q
        self._data.qvel[self._arm_dof_ids] = 0.0
        mujoco.mj_forward(self._model, self._data)

        # 初始化 target 到当前 tcp（防止第一步跳变）
        tcp_pos, tcp_quat = self._get_site_pose(self._tcp_site_id)
        self._target_pos = tcp_pos.copy()
        self._target_quat = tcp_quat.copy()

        self._data.mocap_pos[self._mocap_id] = self._target_pos
        self._data.mocap_quat[self._mocap_id] = self._target_quat

        self._step_count = 0
        obs = self._compute_observation()
        return obs, {"succeed": False}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        a = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)

        if self._has_gripper:
            dx, dy, dz, drx, dry, drz, grasp = a
        else:
            dx, dy, dz, drx, dry, drz = a
            grasp = None

        # target pos
        dpos = np.array([dx, dy, dz], dtype=np.float64) * self._action_scale[0]
        self._target_pos = np.clip(self._target_pos + dpos, _CARTESIAN_BOUNDS[0], _CARTESIAN_BOUNDS[1])

        # target quat (axis-angle)
        drot = np.array([drx, dry, drz], dtype=np.float64) * self._action_scale[1]
        dq = axis_angle_to_quat(drot)
        self._target_quat = quat_mul(dq, self._target_quat)
        self._target_quat /= (np.linalg.norm(self._target_quat) + 1e-12)

        # write mocap (用于可视化/调试；控制实际靠 opspace 追踪 target_pose)
        self._data.mocap_pos[self._mocap_id] = self._target_pos
        self._data.mocap_quat[self._mocap_id] = self._target_quat

        # gripper (如果你后面在 xml 里加入夹爪 joint+actuator，这段就会生效)
        if grasp is not None:
            for cid in self._gripper_ctrl_ids:
                lo, hi = self._get_ctrlrange(cid)
                cur = float(self._data.ctrl[cid])
                nxt = np.clip(cur + grasp * self._action_scale[2] * (hi - lo), lo, hi)
                self._data.ctrl[cid] = nxt

        # control loop
        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._tcp_site_id,
                dof_ids=self._arm_dof_ids,
                pos=self._target_pos,
                ori=self._target_quat,
                joint=self._home_q,
                gravity_comp=True,
                pos_gains=(200.0, 200.0, 200.0),
                ori_gains=(200.0, 200.0, 200.0),
                damping_ratio=1.0,
            )
            self._data.ctrl[self._arm_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1

        obs = self._compute_observation()
        succeed = self._compute_success()

        # 默认给一个稀疏 reward（hil-serl 的 classifier wrapper 一般会覆盖 reward）
        reward = 1.0 if succeed else 0.0
        terminated = bool(succeed)
        truncated = bool(self.time_limit_exceeded())

        info = {"succeed": succeed}

        # human 渲染：不要在 __init__ 里 render（你之前的报错就源于这里）
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            # 懒加载 viewer
            if self._human_viewer is None:
                self._human_viewer = mujoco.viewer.launch_passive(self._model, self._data)
            self._human_viewer.sync()
            return None

        # rgb_array: 返回 dict(key->image)，方便直接塞进 obs["images"]
        return {k: self._render_camera(self._camera_for_key[k]) for k in self._image_keys}

    def close(self) -> None:
        if self._human_viewer is not None:
            try:
                self._human_viewer.close()
            except Exception:
                pass
            self._human_viewer = None

        if getattr(self, "_rgb_renderer", None) is not None:
            self._rgb_renderer.close()
            self._rgb_renderer = None

        super().close()

    # ------------------ observation / reward ------------------

    def _compute_observation(self) -> Dict[str, Any]:
        tcp_pos, tcp_quat = self._get_site_pose(self._tcp_site_id)
        v_lin, v_ang = self._get_site_vel(self._tcp_site_id)

        assembly_pos = self._data.xpos[self._assembly_body_id].copy()
        assembly_quat = self._data.xquat[self._assembly_body_id].copy()
        rel_pos = (assembly_pos - tcp_pos).copy()

        # 没有夹爪 DOF 的情况下先置 0；后面你加了夹爪 actuator 再替换成真实值
        gripper_pose = np.array([0.0], dtype=np.float32)

        state = {
            "tcp_pose": np.concatenate([tcp_pos, tcp_quat], axis=0).astype(np.float32),
            "tcp_vel": np.concatenate([v_lin, v_ang], axis=0).astype(np.float32),
            "gripper_pose": gripper_pose,
            "tcp_force": np.zeros(3, dtype=np.float32),
            "tcp_torque": np.zeros(3, dtype=np.float32),
            "assembly_pos": assembly_pos.astype(np.float32),
            "assembly_quat": assembly_quat.astype(np.float32),
            "rel_pos": rel_pos.astype(np.float32),
        }

        obs: Dict[str, Any] = {"state": state}

        if self.image_obs:
            obs["images"] = {k: self._render_camera(self._camera_for_key[k]) for k in self._image_keys}

        return obs

    def _compute_success(self) -> bool:
        tcp_pos, tcp_quat = self._get_site_pose(self._tcp_site_id)
        assembly_pos = self._data.xpos[self._assembly_body_id]
        assembly_quat = self._data.xquat[self._assembly_body_id]

        pos_err = float(np.linalg.norm(assembly_pos - tcp_pos))
        ori_err = float(quat_angle(tcp_quat, assembly_quat))
        return (pos_err < self._pos_tol) and (ori_err < self._ori_tol)

    # ------------------ mujoco helpers ------------------

    def _get_site_pose(self, site_id: int) -> Tuple[np.ndarray, np.ndarray]:
        pos = self._data.site_xpos[site_id].copy()
        xmat = self._data.site_xmat[site_id].reshape(3, 3).copy()
        quat = mat_to_quat(xmat)  # wxyz
        # 统一符号（可选）
        if quat[0] < 0:
            quat *= -1.0
        return pos, quat

    def _get_site_vel(self, site_id: int) -> Tuple[np.ndarray, np.ndarray]:
        # mj_objectVelocity 输出 [ang(3), lin(3)]
        v = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self._model, self._data, mujoco.mjtObj.mjOBJ_SITE, site_id, v, 0)
        w = v[:3].copy()
        lin = v[3:].copy()
        return lin, w

    def _render_camera(self, camera_name: str) -> np.ndarray:
        self._rgb_renderer.update_scene(self._data, camera=camera_name)
        img = self._rgb_renderer.render()
        return img  # uint8 HWC

    # ------------------ actuator inference ------------------

    def _infer_actuators_and_joints(self):
        arm_ctrl: List[int] = []
        gripper_ctrl: List[int] = []
        arm_joint_ids: set[int] = set()

        for i in range(self._model.nu):
            name = (self._model.actuator(i).name or "").lower()
            if any(k in name for k in ["gripper", "finger", "jaw"]):
                gripper_ctrl.append(i)
            else:
                arm_ctrl.append(i)
                j = int(self._model.actuator_trnid[i][0])
                if j >= 0:
                    arm_joint_ids.add(j)

        if len(arm_joint_ids) == 0:
            self._debug_print_model_names()
            raise RuntimeError("无法从 actuator 推断机械臂 joint。请检查 actuator 是否是 joint-driven。")

        # 关节按 dof 顺序
        arm_joint_ids_sorted = sorted(list(arm_joint_ids), key=lambda j: int(self._model.jnt_dofadr[j]))

        # actuator 也按其对应关节 dof 顺序
        def act_key(aid: int) -> int:
            j = int(self._model.actuator_trnid[aid][0])
            return int(self._model.jnt_dofadr[j]) if j >= 0 else 1_000_000

        arm_ctrl_sorted = sorted(arm_ctrl, key=act_key)

        return (
            np.asarray(arm_ctrl_sorted, dtype=np.int32),
            np.asarray(arm_joint_ids_sorted, dtype=np.int32),
            np.asarray(gripper_ctrl, dtype=np.int32),
        )

    def _get_ctrlrange(self, actuator_id: int) -> Tuple[float, float]:
        if int(self._model.actuator_ctrllimited[actuator_id]) != 0:
            lo, hi = self._model.actuator_ctrlrange[actuator_id]
            return float(lo), float(hi)
        return -1.0, 1.0

    def _debug_print_model_names(self):
        print("=== Debug model names ===")
        print("Bodies:", [self._model.body(i).name for i in range(self._model.nbody)])
        print("Joints:", [self._model.joint(i).name for i in range(self._model.njnt)])
        print("Actuators:", [self._model.actuator(i).name for i in range(self._model.nu)])
        print("Sites:", [self._model.site(i).name for i in range(self._model.nsite)])
        print("Sensors:", [self._model.sensor(i).name for i in range(self._model.nsensor)])
        print("Cameras:", [self._model.camera(i).name for i in range(self._model.ncam)])
        print("nmocap:", self._model.nmocap)
