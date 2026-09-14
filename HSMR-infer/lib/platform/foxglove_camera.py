"""Latest-frame capture for ROS2 images exposed by Foxglove Bridge."""

import asyncio
import json
import struct
import threading
import time

from typing import Dict, Optional

import cv2
import numpy as np


FOXGLOVE_SUBPROTOCOL = "foxglove.sdk.v1"
MESSAGE_DATA_OPCODE = 0x01
BINARY_HEADER_SIZE = 13
_BINARY_HEADER = struct.Struct("<xIQ")


def _parse_foxglove_schema(
    schema_text: str,
    root_schema_name: str,
    get_types_from_msg,
) -> Dict:
    """Parse concatenated ROS2 schemas from a Foxglove advertisement."""

    parts = schema_text.split("\n===\nMSG: ")
    result = {}
    root_text = parts[0].strip()
    if root_text:
        result.update(get_types_from_msg(root_text, root_schema_name))

    for part in parts[1:]:
        lines = part.strip().split("\n")
        if len(lines) < 2:
            continue
        type_name = lines[0].strip()
        message_text = "\n".join(lines[1:]).strip()
        if message_text:
            result.update(get_types_from_msg(message_text, type_name))
    return result


class FoxgloveLatestFrameCapture:
    """Subscribe to a compressed ROS2 image and retain only its newest frame."""

    def __init__(
        self,
        url: str,
        topic: str,
        discovery_timeout: float = 3.0,
        reconnect_delay: float = 1.0,
    ):
        self.url = str(url)
        self.topic = str(topic)
        self.discovery_timeout = float(discovery_timeout)
        self.reconnect_delay = float(reconnect_delay)

        self.frame = None
        self.sequence = -1
        self.captured_monotonic = None
        self.captured_unix = None
        self.first_frame_monotonic = None
        self.error: Optional[str] = None
        self.eof = False
        self.connected = False

        self.stopped = threading.Event()
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self._run_thread, daemon=True)

    def properties(self):
        with self.condition:
            if self.frame is None:
                height = width = None
            else:
                height, width = self.frame.shape[:2]
            elapsed = (
                self.captured_monotonic - self.first_frame_monotonic
                if (
                    self.sequence > 0
                    and self.captured_monotonic is not None
                    and self.first_frame_monotonic is not None
                )
                else 0.0
            )
            measured_fps = self.sequence / elapsed if elapsed > 0 else None
            return {
                "type": "foxglove",
                "url": self.url,
                "topic": self.topic,
                "connected": self.connected,
                "width": width,
                "height": height,
                "fps": measured_fps,
            }

    def start(self):
        self.thread.start()
        return self

    def snapshot(self, after_sequence=None, timeout=10.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while (
                not self.stopped.is_set()
                and not self.eof
                and (
                    self.frame is None
                    or (
                        after_sequence is not None
                        and self.sequence <= after_sequence
                    )
                )
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)

            if self.frame is None:
                return None
            return {
                "frame_bgr": self.frame.copy(),
                "sequence": self.sequence,
                "captured_monotonic": self.captured_monotonic,
                "captured_unix": self.captured_unix,
            }

    def stop(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2.0)

    def _run_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._receive_forever())
        finally:
            loop.close()
            with self.condition:
                self.connected = False
                self.eof = True
                self.condition.notify_all()

    async def _receive_forever(self):
        while not self.stopped.is_set():
            try:
                await self._receive_connection()
            except Exception as error:
                with self.condition:
                    self.connected = False
                    self.error = f"{type(error).__name__}: {error}"
                    self.condition.notify_all()
            if not self.stopped.is_set():
                await asyncio.sleep(self.reconnect_delay)

    @staticmethod
    def _make_typestore():
        from rosbags.typesys import Stores, get_typestore

        for name in ("ROS2_JAZZY", "ROS2_HUMBLE"):
            store = getattr(Stores, name, None)
            if store is not None:
                return get_typestore(store)
        raise RuntimeError("Installed rosbags has no ROS2 Jazzy/Humble typestore.")

    async def _discover_channel(self, websocket, typestore):
        from rosbags.typesys import get_types_from_msg

        deadline = time.monotonic() + self.discovery_timeout
        channels = {}
        got_advertisement = False
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            wait = min(remaining, 0.5) if got_advertisement else remaining
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=wait)
            except asyncio.TimeoutError:
                if got_advertisement:
                    break
                continue
            if not isinstance(raw, str):
                continue

            message = json.loads(raw)
            if message.get("op") != "advertise":
                continue
            for channel in message.get("channels", []):
                channels[channel["topic"]] = channel
            got_advertisement = True
            if self.topic in channels:
                break

        channel = channels.get(self.topic)
        if channel is None:
            examples = sorted(channels)[:8]
            raise RuntimeError(
                f"Foxglove topic not advertised: {self.topic}. "
                f"Discovered {len(channels)} topics; examples: {examples}"
            )

        schema_name = channel["schemaName"]
        if schema_name not in typestore.types:
            schema_text = channel.get("schema", "")
            if not schema_text:
                raise RuntimeError(
                    f"Foxglove channel has no schema: {self.topic}"
                )
            types = _parse_foxglove_schema(
                schema_text,
                schema_name,
                get_types_from_msg,
            )
            new_types = {
                name: definition
                for name, definition in types.items()
                if name not in typestore.types
            }
            if new_types:
                typestore.register(new_types)
        return channel

    async def _receive_connection(self):
        try:
            import websockets
        except ImportError as error:
            raise RuntimeError(
                "Foxglove camera input requires `pip install websockets`."
            ) from error
        try:
            typestore = self._make_typestore()
        except ImportError as error:
            raise RuntimeError(
                "Foxglove camera input requires `pip install rosbags`."
            ) from error

        websocket = await websockets.connect(
            self.url,
            subprotocols=[FOXGLOVE_SUBPROTOCOL],
            max_size=None,
            open_timeout=5,
        )
        subscription_id = 1
        try:
            channel = await self._discover_channel(websocket, typestore)
            await websocket.send(json.dumps({
                "op": "subscribe",
                "subscriptions": [{
                    "id": subscription_id,
                    "channelId": channel["id"],
                }],
            }))
            with self.condition:
                self.connected = True
                self.error = None
                self.condition.notify_all()

            while not self.stopped.is_set():
                try:
                    raw = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    continue
                if (
                    not isinstance(raw, bytes)
                    or len(raw) < BINARY_HEADER_SIZE
                    or raw[0] != MESSAGE_DATA_OPCODE
                ):
                    continue

                received_subscription, _ = _BINARY_HEADER.unpack_from(raw)
                if received_subscription != subscription_id:
                    continue
                message = typestore.deserialize_cdr(
                    raw[BINARY_HEADER_SIZE:],
                    channel["schemaName"],
                )
                if not hasattr(message, "data"):
                    raise RuntimeError(
                        f"Topic is not an image message: {self.topic}"
                    )
                encoded = np.frombuffer(message.data, dtype=np.uint8)
                frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue

                now_monotonic = time.monotonic()
                with self.condition:
                    self.frame = frame_bgr
                    self.sequence += 1
                    self.captured_monotonic = now_monotonic
                    self.captured_unix = time.time()
                    if self.first_frame_monotonic is None:
                        self.first_frame_monotonic = now_monotonic
                    self.condition.notify_all()
        finally:
            try:
                if not websocket.closed:
                    await websocket.send(json.dumps({
                        "op": "unsubscribe",
                        "subscriptionIds": [subscription_id],
                    }))
            except Exception:
                pass
            await websocket.close()
            with self.condition:
                self.connected = False
                self.condition.notify_all()
