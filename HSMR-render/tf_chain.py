"""坐标变换链: optical → HEAD → BASE/world.

主路径: 配置里的固定矩阵 (用户提供 HEAD→optical、HEAD→BASE/world)。
辅路径: 若机器人 TF 里发布了 base/world, 用 TF BFS 查找 (复刻 realtime_depth_3d.py)。
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


class TransformChain:
    """管理 optical→HEAD→BASE/world 的 (R, t)."""

    def __init__(self, cfg):
        # TF 优先, 仅 HEAD 有兜底矩阵 (verified, 防止 /tf_static 未到时 HEAD 报错)
        frames = cfg.get("frames") or {}
        self.optical_frame = frames.get(
            "optical", "realsense_head_color_optical_frame"
        )
        self.head_frame = frames.get("head", "HEAD")
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
