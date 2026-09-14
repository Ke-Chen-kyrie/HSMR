"""坐标变换链 + 关节朝向工具.

移植自原工程 deploy/joint_service/tf_chain.py + main.py 的关节朝向逻辑:
  - TransformChain: optical → HEAD → BASE/world (TF 优先 + HEAD 兜底矩阵)
  - joint_pos_ori / joint_orientation: 由 person 数据 (joints_optical_m + joints_ori)
    计算某关节在目标系的 3D 位置 + 朝向 (统一约定 X=前 / Y=左 / Z=上)

纯 numpy, 无其他依赖.
"""
import numpy as np


def quat_to_rotmat(x, y, z, w):
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def inv_homog(M):
    R, t = M[:3, :3], M[:3, 3]
    Rinv = R.T
    out = np.eye(4)
    out[:3, :3] = Rinv
    out[:3, 3] = -Rinv @ t
    return out


def transform_points(pts, R, t):
    return (pts @ R.T) + t


def _rot_to_quat_wxyz(M):
    """3x3 旋转矩阵 → 四元数 (w,x,y,z)."""
    M = np.asarray(M, dtype=float)
    tr = M[0, 0] + M[1, 1] + M[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(0.25 * s), (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
    else:
        i = int(np.argmax([M[0, 0], M[1, 1], M[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            q = [(M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            q = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            q = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s]
    q = np.array(q) / np.linalg.norm(q)
    return q.tolist()


def _mat_to_euler_deg(M):
    """旋转矩阵 → ZYX 欧拉角(度) 的 roll/pitch/yaw."""
    M = np.asarray(M, dtype=float)
    sy = np.sqrt(max(M[0, 0] ** 2 + M[1, 0] ** 2, 1e-12))
    pitch = np.arctan2(-M[2, 0], sy)
    if abs(pitch) > np.pi / 2 - 1e-6:
        roll = np.arctan2(-M[1, 2], M[1, 1])
        yaw = 0.0
    else:
        roll = np.arctan2(M[2, 1], M[2, 2])
        yaw = np.arctan2(M[1, 0], M[0, 0])
    return {"roll_deg": round(float(np.degrees(roll)), 3),
            "pitch_deg": round(float(np.degrees(pitch)), 3),
            "yaw_deg": round(float(np.degrees(yaw)), 3)}


# 关节朝向统一 remap (与原工程 main.py / tests/test_capture_save.py 一致, 值不变):
#   R_unified = R_orig @ _JOINT_REMAP.T  → 统一约定 X=前/Y=左/Z=上 (new_x=old_x, new_y=-old_z, new_z=old_y)
_JOINT_REMAP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float)

# 24 SKEL 解剖关节名 (thirdparty/SKEL/skel/kin_skel.py)
SKEL_JOINTS = [
    "pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
    "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l",
    "lumbar_body", "thorax", "head",
    "scapula_r", "humerus_r", "ulna_r", "radius_r", "hand_r",
    "scapula_l", "humerus_l", "ulna_l", "radius_l", "hand_l",
]

# 24 关节 SKEL 骨架连接 (索引与 SKEL_JOINTS 一致)
SKEL_BONES = [
    [0, 11], [11, 12], [12, 13],
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
    [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
    [12, 14], [14, 15], [15, 16], [16, 17], [17, 18],
    [12, 19], [19, 20], [20, 21], [21, 22], [22, 23],
]


class TransformChain:
    """管理 optical→HEAD→BASE/world 的 (R, t)."""

    def __init__(self, cfg):
        # TF 优先, 仅 HEAD 有兜底矩阵 (verified, 防止 /tf_static 未到时 HEAD 报错)
        self.optical_frame = cfg["frames"]["optical"]
        self.head_frame = cfg["frames"]["head"]
        self._tf_cache = {}      # (parent, child) -> (R, t)
        self._head_fallback = None
        fb = cfg.get("head_to_optical_fallback")
        if fb is not None:
            self._head_fallback = inv_homog(np.asarray(fb, dtype=float))  # optical→HEAD

    # ── TF 缓存更新 (从 /tf_static 和 /tf 消息) ──
    def update_tf(self, msgs):
        for m in msgs:
            if m is None or not hasattr(m, "transforms"):
                continue
            for t in m.transforms:
                hdr = t.header
                q = t.transform.rotation
                tr = t.transform.translation
                R = quat_to_rotmat(q.x, q.y, q.z, q.w)
                self._tf_cache[(hdr.frame_id, t.child_frame_id)] = (R, np.array([tr.x, tr.y, tr.z]))

    # ── 取 optical→target 的 (R, t) ──
    BASE_CANDIDATES = ["base_link", "BASE", "base_footprint", "robot_base_link", "base"]
    WORLD_CANDIDATES = ["world", "map", "odom", "world_enu", "World"]

    def _find_frame_in_tf(self, candidates):
        """在 TF 缓存里找第一个存在的目标帧名 (返回帧名或 None)."""
        graph_frames = set()
        for (a, b) in self._tf_cache:
            graph_frames.add(a)
            graph_frames.add(b)
        for c in candidates:
            if c in graph_frames:
                return c
        return None

    def get_opt_to_target(self, target):
        """返回 (R_opt_target, t_opt_target). 不可达返回 None.

        TF 优先 (从 /tf + /tf_static topic BFS 链式连乘);
        HEAD 有 verified 兜底矩阵 (/tf_static 未到也能用);
        BASE/world 纯 TF, 找不到返回 None (前端报错, 不用矩阵瞎算).
        """
        if target == self.optical_frame:
            return np.eye(3), np.zeros(3)
        actual = target
        if target == "BASE":
            actual = self._find_frame_in_tf(self.BASE_CANDIDATES)
        elif target == "world":
            actual = self._find_frame_in_tf(self.WORLD_CANDIDATES)
        for tgt in {actual, target}:
            if tgt is None:
                continue
            r = self._tf_bfs(tgt)
            if r is not None:
                return r
        # 仅 HEAD 兜底 (verified 矩阵)
        if target == self.head_frame and self._head_fallback is not None:
            return self._head_fallback[:3, :3], self._head_fallback[:3, 3]
        return None

    def tf_frames(self):
        """TF 缓存里的所有帧 (调试用)."""
        frames = set()
        for (a, b) in self._tf_cache:
            frames.add(a)
            frames.add(b)
        return sorted(frames)

    def transform_source(self, target):
        """返回该 target 的变换来源: identity / tf_chained / head_matrix / None."""
        if target == self.optical_frame:
            return "identity"
        actual = target
        if target == "BASE":
            actual = self._find_frame_in_tf(self.BASE_CANDIDATES)
        elif target == "world":
            actual = self._find_frame_in_tf(self.WORLD_CANDIDATES)
        if actual is not None and self._tf_bfs(actual) is not None:
            return "tf_chained"
        if self._tf_bfs(target) is not None:
            return "tf_chained"
        if target == self.head_frame and self._head_fallback is not None:
            return "head_matrix"
        return None

    def available_frames(self):
        return ["BASE", "world", self.head_frame, self.optical_frame]

    # ── TF BFS (复刻 realtime_depth_3d.py) ──
    def _tf_bfs(self, target):
        graph = {}
        for (a, b) in self._tf_cache:
            graph.setdefault(a, []).append(b)
            graph.setdefault(b, []).append(a)

        def edge_tf(a, b):
            """返回 a→b 的变换 (沿路径正向).
            缓存存 (parent,child) → (R,t) 表示 p_parent = R@p_child + t, 即 child→parent 映射.
            因此:
              - (a,b) 在缓存: 给的是 b→a, 需求逆得 a→b: (R.T, -R.T@t)
              - (b,a) 在缓存: 给的就是 a→b, 直接可用: (R, t)
            """
            if (a, b) in self._tf_cache:
                R, t = self._tf_cache[(a, b)]
                return R.T, -R.T @ t                    # 反向 b→a 求逆得 a→b
            if (b, a) in self._tf_cache:
                return self._tf_cache[(b, a)]          # 正向 a→b 直接可用
            return None

        from collections import deque
        q = deque([(self.optical_frame, [])])
        seen = {self.optical_frame}
        while q:
            node, path = q.popleft()
            for nxt in graph.get(node, []):
                if nxt in seen:
                    continue
                np_ = path + [(node, nxt)]
                if nxt == target:
                    R_acc, t_acc = np.eye(3), np.zeros(3)
                    for (a, b) in np_:
                        etf = edge_tf(a, b)
                        if etf is None:
                            return None
                        R, p = etf
                        t_acc = R @ t_acc + p
                        R_acc = R @ R_acc
                    return R_acc, t_acc
                seen.add(nxt)
                q.append((nxt, np_))
        return None

    # ── 坐标变换 ──
    def to_target(self, points_opt, target):
        """points_opt: (N,3) optical 系坐标 → target 系."""
        r = self.get_opt_to_target(target)
        if r is None:
            return None
        R, t = r
        return transform_points(np.asarray(points_opt, dtype=float), R, t)

    def rot_to_target(self, R_joint_opt, target):
        """R_joint_opt: (3,3) 关节在 optical 系的世界朝向 → target 系."""
        r = self.get_opt_to_target(target)
        if r is None:
            return None
        R, _ = r
        return R @ np.asarray(R_joint_opt, dtype=float)


# ── 关节 3D 位置/朝向 (移植自 main.py) ──
def joint_pos_ori(person, joint_index, chain, target):
    """由 person 数据算某关节在 target 系的 (position_m, R_target, depth_valid).

    R_target = R_ot @ (R_joint @ _JOINT_REMAP.T): SKEL 原始朝向经统一 remap 后转到目标系.
    """
    R_ot_t = chain.get_opt_to_target(target)
    if R_ot_t is None:
        return None, None, False
    R_ot, t_ot = R_ot_t
    pos_opt = np.array(person["joints_optical_m"][joint_index], dtype=float)
    depth_valid = bool(person["depth_valid"][joint_index])
    if np.isnan(pos_opt).any():
        depth_valid = False
    pos = (R_ot @ pos_opt + t_ot).tolist() if depth_valid else None
    R_joint = np.asarray(person["joints_ori"][joint_index], dtype=float)
    R_target = (R_ot @ (R_joint @ _JOINT_REMAP.T)) if depth_valid else None
    return pos, R_target, depth_valid


def joint_orientation(R_target, is_whole=False):
    """由 3×3 矩阵生成 orientation dict (统一约定: X=前, Y=左, Z=上)."""
    fwd = (R_target @ np.array([1.0, 0.0, 0.0])).tolist()
    o = {
        "quaternion_wxyz": _rot_to_quat_wxyz(R_target),
        "euler_deg": _mat_to_euler_deg(R_target),
        "body_forward": fwd,
        "body_left": (R_target @ np.array([0.0, 1.0, 0.0])).tolist(),
        "body_up": (R_target @ np.array([0.0, 0.0, 1.0])).tolist(),
        "reference": "after unified remap (X=前,Y=左,Z=上), expressed in target frame",
    }
    if is_whole:
        # BASE 系水平朝向: 前向 = X 轴在水平面投影的角度 (X=前, Y=左)
        o["facing_yaw_deg"] = round(float(np.degrees(np.arctan2(fwd[1], fwd[0]))), 2)
    return o
