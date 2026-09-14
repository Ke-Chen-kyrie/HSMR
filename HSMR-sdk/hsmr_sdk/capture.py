"""Foxglove WebSocket 多主题采集 + 帧解码.

移植自原工程 deploy/joint_service/capture.py (复刻 docs/realtime_depth_3d.py):
连机器人的 foxglove bridge, 订阅 color / depth / camera_info / tf, 反序列化最新帧.
纯吃数据: snapshot() 返回各 topic 最新消息, 调用方自行解码.

依赖 websockets + rosbags (均懒加载, 仅实际订阅时导入); 无 rclpy / 无 ROS2 客户端.
"""
import asyncio
import json
import struct
import threading

SUB_PROTO = "foxglove.sdk.v1"
HDR = struct.Struct("<xIQ")          # opcode(1) + subscription_id(4) + receive_time(8)
DEPTH_PNG_OFFSET = 12                # compressedDepth 头部长度(剥掉再 imdecode)


class MultiTopicCapture:
    """订阅多个 topic, snapshot() 返回各 topic 最新反序列化消息."""

    def __init__(self, url, topics):
        self.url = url
        self.topics = topics
        self.latest = {}              # topic -> msg
        # /tf 与 /tf_static 都可能有多个发布者, 且单条消息常只含树的一部分边;
        # 按 (parent,child) 合并累积(取最新变换), snapshot 返回完整树, 避免图断连.
        self._tf_static_frames = {}   # (parent, child) -> TransformStamped
        self._tf_static_cls = None
        self.loop = asyncio.new_event_loop()
        self.thread_stop = False

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        self.loop.run_until_complete(self._receive_forever())

    async def _receive_forever(self):
        while not self.thread_stop:
            try:
                await self._receive()
            except Exception as e:
                print(f"[capture] 连接错误: {e}")
                await asyncio.sleep(1.5)

    async def _receive(self):
        import websockets
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg

        typestore = None
        for n in ("ROS2_JAZZY", "ROS2_HUMBLE"):
            s = getattr(Stores, n, None)
            if s:
                typestore = get_typestore(s)
                break

        ws = await websockets.connect(
            self.url, subprotocols=[SUB_PROTO], max_size=None, open_timeout=5
        )
        channels = {}
        deadline = self.loop.time() + 5
        while self.loop.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1)
            except asyncio.TimeoutError:
                continue
            if isinstance(raw, str):
                m = json.loads(raw)
                if m.get("op") == "advertise":
                    for ch in m.get("channels", []):
                        channels[ch["topic"]] = ch

        subs, sub_map = [], {}
        for i, topic in enumerate(self.topics):
            if topic in channels:
                subs.append({"id": i + 1, "channelId": channels[topic]["id"]})
                sub_map[i + 1] = topic
        print(f"[capture] 订阅 {len(subs)}/{len(self.topics)}: {list(sub_map.values())}")
        if not subs:
            raise RuntimeError("没有可订阅的主题")
        await ws.send(json.dumps({"op": "subscribe", "subscriptions": subs}))

        while not self.thread_stop:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if not isinstance(raw, bytes) or len(raw) < 13 or raw[0] != 0x01:
                continue
            sid, _ = HDR.unpack_from(raw)
            topic = sub_map.get(sid)
            if not topic:
                continue
            ch = channels[topic]
            sn = ch["schemaName"]
            if sn not in typestore.types:
                parts = ch["schema"].split("\n===\nMSG: ")
                result = {}
                if parts[0].strip():
                    result.update(get_types_from_msg(parts[0].strip(), sn))
                for part in parts[1:]:
                    lines = part.strip().split("\n")
                    if len(lines) >= 2:
                        result.update(get_types_from_msg("\n".join(lines[1:]).strip(), lines[0].strip()))
                new = {n: d for n, d in result.items() if n not in typestore.types}
                if new:
                    typestore.register(new)
            try:
                msg = typestore.deserialize_cdr(raw[13:], sn)
            except Exception:
                continue
            if topic in ("/tf_static", "/tf") and hasattr(msg, "transforms"):
                # 合并 /tf 与 /tf_static 的变换边(取最新), 保证完整 TF 树
                self._tf_static_cls = type(msg)
                for t in msg.transforms:
                    self._tf_static_frames[(t.header.frame_id, t.child_frame_id)] = t
                self.latest[topic] = self._tf_static_cls(
                    transforms=list(self._tf_static_frames.values()))
            else:
                self.latest[topic] = msg

    def snapshot(self):
        """返回各主题最新消息的引用."""
        return dict(self.latest)

    def stop(self):
        self.thread_stop = True


# ── 帧解码 (纯数据函数, 复刻原工程 inference.py) ──
def decode_color(msg):
    """彩色消息 → BGR ndarray (H,W,3)."""
    import numpy as _np
    import cv2 as _cv2
    d = _np.frombuffer(msg.data, dtype=_np.uint8)
    return _cv2.imdecode(d, _cv2.IMREAD_COLOR)


def decode_depth(msg):
    """深度消息 → uint16 毫米 (H,W). 支持 raw Image(16UC1) 与 compressedDepth(PNG)."""
    import numpy as _np
    import cv2 as _cv2
    if hasattr(msg, "height") and hasattr(msg, "width"):   # sensor_msgs/Image 原始 16UC1
        return _np.frombuffer(msg.data, dtype=_np.uint16).reshape(msg.height, msg.width)
    # 旧: CompressedImage PNG (compressedDepth)
    d = _np.frombuffer(msg.data, dtype=_np.uint8)
    return _cv2.imdecode(d[DEPTH_PNG_OFFSET:], _cv2.IMREAD_UNCHANGED)


def camera_info_k(msg):
    """CameraInfo 消息 → 3x3 内参矩阵."""
    import numpy as _np
    return _np.array(msg.k).reshape(3, 3)


def build_topics(cfg):
    """按 topics.prefix 组装 5 个订阅话题 (与原工程 main.py 一致)."""
    prefix = cfg["topics"]["prefix"]
    return [
        f"{prefix}/color/image_raw/compressed",
        f"{prefix}/aligned_depth_to_color/image_raw/compressedDepth",  # 需 realsense compressedDepth.format=png
        f"{prefix}/aligned_depth_to_color/camera_info",
        "/tf_static",
        "/tf",
    ]
