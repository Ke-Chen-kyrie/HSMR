"""Foxglove WebSocket 多主题订阅器.

复刻 docs/realtime_depth_3d.py 的 MultiTopicCapture:
连机器人的 foxglove bridge, 订阅 color / depth / camera_info, 反序列化最新帧.
渲染服务不需要 TF, 故剥离 /tf 相关状态与分支.
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
            self.latest[topic] = msg

    def snapshot(self):
        """返回各主题最新消息的引用."""
        return dict(self.latest)

    def stop(self):
        self.thread_stop = True


class RosbridgeCapture:
    """ROS1 rosbridge websocket 多主题订阅器.

    连机器人的 rosbridge (rosbridge_suite), 订阅 color / depth / camera_info,
    msg 为 rosbridge JSON 反序列化后的 dict (保持与 MultiTopicCapture 相同的
    snapshot() 接口, 上层 inference 解码时兼容 dict 形态).
    """

    def __init__(self, url, topics):
        self.url = url
        self.topics = topics
        self.latest = {}
        self.thread_stop = False

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.thread_stop:
            try:
                self._receive()
            except Exception as e:
                print(f"[rosbridge] 连接错误: {e}")
                import time as _t
                _t.sleep(1.5)

    def _receive(self):
        import json
        import websocket

        ws = websocket.create_connection(self.url, timeout=10)
        for topic in self.topics:
            ws.send(json.dumps({"op": "subscribe", "topic": topic,
                                "throttle_rate": 0, "queue_length": 1}))
        print(f"[rosbridge] 订阅 {len(self.topics)}: {self.topics}")
        while not self.thread_stop:
            try:
                raw = ws.recv()
            except Exception as e:
                print(f"[rosbridge] recv 错误: {e}")
                break
            try:
                m = json.loads(raw)
            except Exception:
                continue
            if m.get("op") == "publish":
                self.latest[m.get("topic")] = m.get("msg")
        ws.close()

    def snapshot(self):
        return dict(self.latest)

    def stop(self):
        self.thread_stop = True
