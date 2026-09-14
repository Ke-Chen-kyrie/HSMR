"""把关节位置 + 旋转矩阵发布为 MarkerArray (zenoh → Foxglove).

移植自原工程 deploy/joint_service/ros_pub.py:
链路: zenoh-ros2-sdk 的 ROS2Publisher 连 zenoh router (默认 tcp/127.0.0.1:7447),
容器内 foxglove_bridge (RMW_IMPLEMENTATION=rmw_zenoh_cpp) 会以 ROS2 话题发现并转发到 Foxglove.

约定 (与 tf._JOINT_REMAP 统一 remap 一致): X=前, Y=左, Z=上.
Marker 内容:
  - 每关节 SPHERE (选中亮黄大球, 未选淡蓝小球)
  - 每关节 3 根 ARROW 朝向轴: 红=X前, 绿=Y左, 蓝=Z上 (即 rotation_matrix 的 3 列)
  - 骨骼 LINE_LIST (灰色)

发布失败/初始化失败只打印日志, 静默降级. zenoh_ros2_sdk 缺包时发布自动禁用.
"""
import threading
import time

import numpy as np

from hsmr_sdk.tf import SKEL_JOINTS, SKEL_BONES

# 颜色 (RGBA 0~1)
_SEL_COLOR = (1.0, 0.85, 0.2)        # 选中关节 亮黄
_UNSEL_COLOR = (0.45, 0.75, 1.0)     # 未选关节 淡蓝
_BONE_COLOR = (0.72, 0.72, 0.78)     # 骨骼 灰
_AXIS_COLORS = [                     # 朝向轴: X=前红, Y=左绿, Z=上蓝
    (1.0, 0.25, 0.25),
    (0.25, 1.0, 0.25),
    (0.25, 0.4, 1.0),
]

_SPHERE_R_SEL = 0.09
_SPHERE_R_UNSEL = 0.05
_AXIS_LEN = 0.12                     # 朝向轴长度 (米)
_AXIS_W = 0.035
_BONE_W = 0.02

_RETRY_INTERVAL_S = 30.0             # 初始化失败后多久重试


def _rot_to_quat_wxyz(M):
    """3x3 旋转矩阵 → 四元数 (w,x,y,z). 独立实现避免循环依赖."""
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


class RosPublisher:
    """把关节位置/旋转发布为 visualization_msgs/msg/MarkerArray (帧=frame)."""

    def __init__(self, cfg_ros=None):
        cfg_ros = cfg_ros or {}
        self.enabled = bool(cfg_ros.get("enabled", True))
        self.topic = cfg_ros.get("topic", "/hsmr/joints/markers")
        self.router_ip = cfg_ros.get("router_ip", "127.0.0.1")
        self.router_port = int(cfg_ros.get("router_port", 7447))
        self.publish_all = bool(cfg_ros.get("publish_all", True))
        # 非空 = 只发布这些关节 (关节名或索引), 忽略 publish_all; 空 = 按 publish_all
        self.publish_joints = list(cfg_ros.get("publish_joints") or [])
        # 标注图发布 (把骨盆轴/骨架画到彩色图后作为 CompressedImage 发出去)
        self.annotated_enabled = bool(cfg_ros.get("annotated_enabled", False))
        self.annotated_topic = cfg_ros.get("annotated_topic", "/hsmr/joints/annotated")
        self._pub = None          # zenoh_ros2_sdk.ROS2Publisher
        self._T = None            # store.types (rosbags 类)
        self._img_pub = None      # 标注图发布器 (CompressedImage)
        self._img_T = None
        self._lock = threading.Lock()
        self._dead = False
        self._dead_reason = None
        self._last_fail = 0.0

    def prewarm(self):
        """后台线程预初始化 (首个发布不卡几秒). 失败静默, 稍后重试."""
        if not self.enabled or self._pub is not None:
            return
        try:
            with self._lock:
                if self._pub is None:
                    self._ensure()
        except Exception as e:
            self._mark_fail(e, "预初始化")

    def _ensure(self):
        from zenoh_ros2_sdk import ROS2Publisher
        self._pub = ROS2Publisher(self.topic, "visualization_msgs/msg/MarkerArray",
                                  router_ip=self.router_ip, router_port=self.router_port)
        self._T = self._pub.session_mgr.store.types

    def _mark_fail(self, exc, where):
        self._dead = True
        self._dead_reason = str(exc)
        self._last_fail = time.time()
        print(f"[ros_pub] {where}失败: {exc}")

    def resolve_publish_indices(self, selected, all_indices):
        """返回要发布的关节索引列表.

        publish_joints 非空 → 只发布这些 (名称或索引); 否则 publish_all=True → 全部,
        False → 仅选中的. 无效名称打印警告并跳过.
        """
        if self.publish_joints:
            out = []
            for name in self.publish_joints:
                if isinstance(name, int) or str(name).strip().isdigit():
                    out.append(int(name))
                else:
                    try:
                        out.append(SKEL_JOINTS.index(str(name).strip()))
                    except ValueError:
                        print(f"[ros_pub] 忽略未知关节名: {name}")
            return sorted(set(i for i in out if 0 <= i < 24))
        return list(all_indices) if self.publish_all else list(selected)

    def publish(self, frame, joints):
        """发布一次 MarkerArray. joints: [{idx, position_m|None, R|None, depth_valid, selected}].

        失败仅打印, 绝不抛给调用方.
        """
        if not self.enabled:
            return
        try:
            with self._lock:
                if self._pub is None:
                    # 初始化失败后带间隔重试 (router 可能稍后才起来)
                    if self._dead and time.time() - self._last_fail < _RETRY_INTERVAL_S:
                        return
                    try:
                        self._ensure()
                        self._dead = False
                        self._dead_reason = None
                    except Exception as e:
                        self._mark_fail(e, "初始化")
                        return
                markers = self._build_markers(frame, joints)
                if markers:
                    self._pub.publish(markers=markers)
        except Exception as e:
            print(f"[ros_pub] 发布失败: {e}")

    def publish_annotated_image(self, frame, img_bgr):
        """把带标注的彩色图 (BGR ndarray) 编码 JPEG 发布为 CompressedImage (topic=annotated_topic).

        失败仅打印, 绝不抛给调用方.
        """
        if not self.enabled or not self.annotated_enabled:
            return
        try:
            import cv2 as _cv2
            import numpy as _np
            ok, buf = _cv2.imencode(".jpg", img_bgr)
            if not ok:
                print("[ros_pub] 标注图 JPEG 编码失败")
                return
            with self._lock:
                if self._img_pub is None:
                    from zenoh_ros2_sdk import ROS2Publisher
                    self._img_pub = ROS2Publisher(
                        self.annotated_topic, "sensor_msgs/msg/CompressedImage",
                        router_ip=self.router_ip, router_port=self.router_port)
                    self._img_T = self._img_pub.session_mgr.store.types
                T = self._img_T
                Time = T["builtin_interfaces/msg/Time"]
                Header = T["std_msgs/msg/Header"]
                CompressedImage = T["sensor_msgs/msg/CompressedImage"]
                hdr = Header(stamp=Time(sec=0, nanosec=0), frame_id=str(frame))
                data = _np.frombuffer(buf.tobytes(), dtype=_np.uint8)
                self._img_pub.publish(header=hdr, format="jpeg", data=data)
        except Exception as e:
            print(f"[ros_pub] 发布标注图失败: {e}")

    def _build_markers(self, frame, joints):
        T = self._T
        Time = T["builtin_interfaces/msg/Time"]
        Duration = T["builtin_interfaces/msg/Duration"]
        Header = T["std_msgs/msg/Header"]
        Point = T["geometry_msgs/msg/Point"]
        Quaternion = T["geometry_msgs/msg/Quaternion"]
        Pose = T["geometry_msgs/msg/Pose"]
        Vector3 = T["geometry_msgs/msg/Vector3"]
        ColorRGBA = T["std_msgs/msg/ColorRGBA"]
        CompressedImage = T["sensor_msgs/msg/CompressedImage"]
        MeshFile = T["visualization_msgs/msg/MeshFile"]
        Marker = T["visualization_msgs/msg/Marker"]

        zero = np.zeros(0, dtype=np.uint8)
        hdr = Header(stamp=Time(sec=0, nanosec=0), frame_id=str(frame))
        tex = CompressedImage(header=hdr, format="", data=zero)
        mesh = MeshFile(filename="", data=zero)
        q_ident = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        lifetime = Duration(sec=0, nanosec=0)

        def _mk(**kw):
            base = dict(header=hdr, ns="hsmr_joints", action=0, lifetime=lifetime,
                        frame_locked=False, points=[], colors=[], texture_resource="",
                        texture=tex, uv_coordinates=[], text="", mesh_resource="",
                        mesh_file=mesh, mesh_use_embedded_materials=False)
            base.update(kw)
            return Marker(**base)

        pos_of, R_of, sel_of = {}, {}, {}
        for jd in joints:
            idx = jd["idx"]
            if jd.get("depth_valid") and jd.get("position_m") is not None:
                pos_of[idx] = np.asarray(jd["position_m"], dtype=float)
            R = jd.get("R")
            if R is not None:
                R_of[idx] = np.asarray(R, dtype=float)
            sel_of[idx] = bool(jd.get("selected"))

        def _pt(v):
            return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))

        markers = []

        # 骨骼连线 (两端点都有效才画)
        for bi, (a, b) in enumerate(SKEL_BONES):
            if a not in pos_of or b not in pos_of:
                continue
            markers.append(_mk(
                id=500 + bi, type=Marker.LINE_LIST,
                pose=Pose(position=_pt([0, 0, 0]), orientation=q_ident),
                scale=Vector3(x=_BONE_W, y=1.0, z=1.0),
                color=ColorRGBA(*_BONE_COLOR, 1.0),
                points=[_pt(pos_of[a]), _pt(pos_of[b])],
            ))

        # 关节球
        for idx, p in pos_of.items():
            sel = sel_of.get(idx)
            r = _SPHERE_R_SEL if sel else _SPHERE_R_UNSEL
            col = _SEL_COLOR if sel else _UNSEL_COLOR
            markers.append(_mk(
                id=idx, type=Marker.SPHERE,
                pose=Pose(position=_pt(p), orientation=q_ident),
                scale=Vector3(x=r, y=r, z=r),
                color=ColorRGBA(*col, 1.0),
            ))

        # 每关节 3 根朝向轴 (rotation_matrix 的 3 列; 红=X前 绿=Y左 蓝=Z上)
        for idx, R in R_of.items():
            if idx not in pos_of:
                continue
            p = pos_of[idx]
            for ai in range(3):
                # 以 R[:,ai] 为第 0 列的旋转阵 (循环置换, det=+1) → 四元数 (wxyz→xyzw)
                M = np.column_stack([R[:, ai], R[:, (ai + 1) % 3], R[:, (ai + 2) % 3]])
                q = _rot_to_quat_wxyz(M)
                markers.append(_mk(
                    id=(ai + 1) * 1000 + idx, type=Marker.ARROW,
                    pose=Pose(position=_pt(p),
                              orientation=Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])),
                    scale=Vector3(x=_AXIS_LEN, y=_AXIS_W, z=_AXIS_W),
                    color=ColorRGBA(*_AXIS_COLORS[ai], 1.0),
                ))

        return markers


_PUB = None
_PUB_LOCK = threading.Lock()


def get_ros_publisher(cfg):
    """进程级单例 RosPublisher (读 config 的 ros_publish 段)."""
    global _PUB
    if _PUB is None:
        with _PUB_LOCK:
            if _PUB is None:
                _PUB = RosPublisher((cfg or {}).get("ros_publish"))
    return _PUB
