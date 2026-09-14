"""HsmrSdk 门面: 采集 → TF → 声纹人脸验证 → ROS 发布 的组合 (纯吃数据).

person 数据由外部提供 (如 HSMR-infer 的 /infer 输出或你自己的推理), SDK 只做外围.
缺依赖模块 (zenoh_ros2_sdk / websockets) 时对应功能自动禁用, 其余不受影响.
"""
import time

from hsmr_sdk.config import load_config
from hsmr_sdk.capture import MultiTopicCapture, build_topics, decode_color, decode_depth, camera_info_k
from hsmr_sdk.tf import TransformChain, joint_pos_ori
from hsmr_sdk.face import get_face_verifier, IdentifyError
from hsmr_sdk.ros import get_ros_publisher
from hsmr_sdk.infer import get_infer_client, InferError

# 说话人确认后取 3D 的关节 (骨盆/根 + head) —— SKEL_JOINTS 索引
PELVIS_JOINT_INDEX = 0
HEAD_JOINT_INDEX = 13


class HsmrSdk:
    """组合门面. 各模块可单独注入 (便于单测), 否则从 cfg 构建进程级单例."""

    def __init__(self, cfg=None, capture=None, tf=None, face=None, ros=None, infer=None):
        self.cfg = cfg or load_config()
        self.topics = self.cfg["topics"].get("list") or build_topics(self.cfg)
        self.capture = capture or MultiTopicCapture(self.cfg["bridge"]["url"], self.topics)
        self.tf = tf or TransformChain(self.cfg)
        self.face = face or get_face_verifier(self.cfg)
        self.ros = ros or get_ros_publisher(self.cfg)
        self.infer = infer or get_infer_client(self.cfg)
        self._capture_owned = capture is None
        self._capture_started = False

    # ── 采集生命周期 ──
    def start_capture(self, warmup=8.0, max_retries=2):
        """启动采集线程, 并阻塞到收到首帧 (彩色+深度) 或超时.

        foxglove bridge 订阅需先等 ~5s 的 advertise 收集窗口才真正订阅,
        所以 start_capture 默认为调用方等到首帧到达 (否则 capture_once 返回空).
        """
        if self._capture_owned and not self._capture_started:
            self.capture.start()
            self._capture_started = True
        deadline = time.time() + warmup
        while time.time() < deadline:
            snap = self.capture.snapshot()
            if snap.get(self.topics[0]) is not None and snap.get(self.topics[1]) is not None:
                return self
            time.sleep(0.2)
        # 收不到关键帧: 重连一轮 (bridge/订阅偶发) 后仍不行才放弃
        if max_retries > 0:
            print("[pipeline] 等待首帧超时, 重连一次")
            self.stop_capture()
            time.sleep(0.5)
            self.capture.start()
            self._capture_started = True
            return self.start_capture(warmup=warmup, max_retries=max_retries - 1)
        print(f"[pipeline] 警告: {warmup}s 内未收到彩色/深度帧 (检查 bridge 与话题)")
        return self

    def stop_capture(self):
        if self._capture_started:
            self.capture.stop()
            self._capture_started = False

    # ── 采集一次 → 解码帧数据 ──
    def capture_once(self):
        """snapshot + 解码 + 更新 TF 缓存. 返回 {rgb_bgr, depth_uint16, K, tf_msgs, raw_snapshot}.

        未订阅/未到的话题对应字段为 None.
        """
        snap = self.capture.snapshot()
        out = self.decode_snapshot(snap)
        self.tf.update_tf(out["tf_msgs"])
        return out

    def decode_snapshot(self, snap):
        """把 snapshot dict → 解码帧数据 (纯函数, 便于测试)."""
        color_t, depth_t, ci_t = self.topics[0], self.topics[1], self.topics[2]
        out = {
            "rgb_bgr": decode_color(snap[color_t]) if color_t in snap else None,
            "depth_uint16": decode_depth(snap[depth_t]) if depth_t in snap else None,
            "K": camera_info_k(snap[ci_t]) if ci_t in snap else None,
            "tf_msgs": [snap[t] for t in ("/tf_static", "/tf") if t in snap],
            "raw_snapshot": snap,
        }
        return out

    # ── 推理: 采集帧 → HSMR-infer 容器 → persons (一条龙) ──
    def infer_persons(self, frame=None, rgb_bgr=None, depth_uint16=None, K=None, **kwargs):
        """把采集帧发给 HSMR-infer /infer → persons[].

        frame = capture_once() 的输出 {rgb_bgr, depth_uint16, K}; 或显式传
        rgb_bgr / depth_uint16 / K. 失败抛 InferError (容器未起/超时/深度缺 K).
        """
        return self.infer.infer_persons(frame=frame, rgb_bgr=rgb_bgr,
                                        depth_uint16=depth_uint16, K=K, **kwargs)

    # ── 声纹 + 人脸交叉验证 → 说话人 3D 位置/朝向 ──
    def identify(self, audio_bytes, frame_bgr, persons, target="BASE",
                 include_pelvis_pose=False, verbose=False, top_k=None):
        """上传语音 → 声纹定说话人 → 人脸交叉确认 → 返回该 person 在 target 系的骨盆+头 3D.

        persons: 由外部 (如 HSMR-infer /infer 输出) 提供, 每条含
                 joints_2d / joints_optical_m / joints_ori / depth_valid / pelvis_pose.
        任一步失败抛 IdentifyError(msg, detail).
        """
        if not self.face.enabled:
            raise IdentifyError("face_verify 未启用")

        # ① 声纹 → 人脸 → 交叉验证 → 关联 person (纯 face 逻辑)
        confirmed = self.face.identify(audio_bytes, frame_bgr, persons,
                                       top_k=top_k, verbose=verbose)
        pi = confirmed["person_index"]
        person = persons[pi]

        # ② frame 别名 + 目标系变换
        FRAME_ALIAS = {
            "realsense_head_color_optical_frame": self.tf.optical_frame,
            "camera": self.tf.optical_frame, "光学系": self.tf.optical_frame,
            "头部相机": self.tf.optical_frame, "相机": self.tf.optical_frame,
            "HEAD": self.tf.head_frame, "头": self.tf.head_frame, "head": self.tf.head_frame,
        }
        target = FRAME_ALIAS.get(target, target)
        if (target not in ("BASE", "world", self.tf.head_frame, self.tf.optical_frame)
                and target.upper() not in ("BASE", "WORLD", "HEAD")):
            raise IdentifyError(
                f"frame 需为 BASE/world/HEAD/{self.tf.optical_frame}(或 头部相机/相机/光学系)")
        if self.tf.get_opt_to_target(target) is None:
            raise IdentifyError(f"无法从 optical 变换到 {target} (TF 找不到该帧)")

        # ③ 骨盆 + head 3D (目标系)
        pelvis_pos, pelvis_R, pelvis_dv = joint_pos_ori(person, PELVIS_JOINT_INDEX, self.tf, target)
        head_pos, head_R, head_dv = joint_pos_ori(person, HEAD_JOINT_INDEX, self.tf, target)

        resp = {
            "verified": True,
            "user_id": confirmed["user_id"],
            "name": confirmed["name"],
            "voice_score": confirmed["voice_score"],
            "face_score": confirmed["face_score"],
            "face_location": confirmed["face_location"],
            "person_index": pi,
            "frame": target,
            "position_m": pelvis_pos,
            "rotation_matrix": pelvis_R.tolist() if pelvis_R is not None else None,
            "depth_valid": pelvis_dv,
            "head_position_m": head_pos,
            "head_rotation_matrix": head_R.tolist() if head_R is not None else None,
            "head_depth_valid": head_dv,
            "timestamp": time.time(),
        }
        if include_pelvis_pose:
            resp["pelvis_pose"] = person.get("pelvis_pose")
        if verbose:
            resp["voice_candidates"] = confirmed.get("voice_candidates", [])
            resp["face_present"] = confirmed.get("face_present", [])
        return resp

    # ── 关节发布载荷 (供 ros.publish) ──
    def person_joints_payload(self, person, target="BASE", selected_joints=None):
        """单个 person → ros 发布用 joints 条目 [{idx, position_m(目标系), R(目标系), depth_valid, selected}]."""
        if selected_joints is None:
            selected_joints = self.ros.resolve_publish_indices(set(), list(range(24)))
        out = []
        for idx in range(24):
            pos, R, dv = joint_pos_ori(person, idx, self.tf, target)
            out.append({
                "idx": idx,
                "position_m": pos,
                "R": R.tolist() if R is not None else None,
                "depth_valid": dv,
                "selected": idx in set(selected_joints),
            })
        return out

    def publish(self, frame, joints, annotated_bgr=None):
        """发布关节 MarkerArray (+ 可选标注图). 失败静默 (ros 内部降级)."""
        self.ros.publish(frame, joints)
        if annotated_bgr is not None:
            self.ros.publish_annotated_image(frame, annotated_bgr)
