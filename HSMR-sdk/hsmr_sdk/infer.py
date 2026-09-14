"""HSMR-infer 容器客户端 (纯吃数据: 帧 → 24 关节 persons).

复用 HSMR-infer 容器 (默认 http://127.0.0.1:8010) 的 /infer: 把 SDK 采集的帧
(rgb + 原始深度 uint16 + 内参 K) 发过去, 取回 persons 供 tf/face/ros 消费.

读 config 的 infer 段 (enabled/url/timeout/max_instances).
"""
import cv2
import requests

__all__ = ["InferClient", "get_infer_client", "InferError", "encode_frame"]


class InferError(Exception):
    """infer_persons() 流程失败 (未启用/容器不可达/深度缺 K/接口错误)."""

    def __init__(self, msg, detail=None):
        super().__init__(msg)
        self.msg = msg
        self.detail = detail


def encode_frame(rgb_bgr, depth_uint16=None, K=None, max_instances=None):
    """帧 → (files, data) 纯函数 (便于测试, 不碰网络).

    files: multipart 的 {"image": (...)} (+ 有深度时 "depth")
    data:  内参 form 字段 {"fx","fy","cx","cy"} (+ 可选 max_instances)
    有深度但无 K → 抛 InferError (容器要求给 depth 时必须带 K).
    """
    if rgb_bgr is None:
        raise InferError("无彩色帧 rgb_bgr")
    ok, jpg = cv2.imencode(".jpg", rgb_bgr)
    if not ok:
        raise InferError("rgb 编码失败")
    files = {"image": ("color.jpg", jpg.tobytes(), "image/jpeg")}
    data = {}
    if depth_uint16 is not None:
        if K is None:
            raise InferError("有深度但无内参 K — 无法深度融合, 需等 camera_info 话题")
        ok, png = cv2.imencode(".png", depth_uint16)
        if not ok:
            raise InferError("depth 编码失败")
        files["depth"] = ("depth.png", png.tobytes(), "image/png")
        data = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2])}
    if max_instances is not None:
        data["max_instances"] = int(max_instances)
    return files, data


class InferClient:
    """读 config 的 infer 段."""

    def __init__(self, cfg_infer=None):
        cfg_infer = cfg_infer or {}
        self.enabled = bool(cfg_infer.get("enabled", True))
        self.url = (cfg_infer.get("url") or "http://127.0.0.1:8010").rstrip("/")
        self.timeout = float(cfg_infer.get("timeout", 120))
        self.max_instances = int(cfg_infer.get("max_instances", 5))

    def infer_persons(self, frame=None, rgb_bgr=None, depth_uint16=None, K=None,
                      timeout=None, max_instances=None):
        """把一帧发给 HSMR-infer /infer → persons[].

        入参二选一:
          frame = capture_once() 的输出 {rgb_bgr, depth_uint16, K}
          或显式 rgb_bgr / depth_uint16 / K (numpy 数组).
        容器不可达/超时/接口错误抛 InferError; 返回 persons 直接可喂 tf/face.
        """
        if not self.enabled:
            raise InferError("infer 未启用 (config.yaml 里 infer.enabled=false)")
        if frame is not None:
            rgb_bgr = rgb_bgr if rgb_bgr is not None else frame.get("rgb_bgr")
            depth_uint16 = depth_uint16 if depth_uint16 is not None else frame.get("depth_uint16")
            K = K if K is not None else frame.get("K")
        if max_instances is None:
            max_instances = self.max_instances

        files, data = encode_frame(rgb_bgr, depth_uint16, K, max_instances)
        try:
            r = requests.post(f"{self.url}/infer", files=files, data=data,
                              timeout=timeout or self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise InferError(f"/infer 请求失败 (容器在 {self.url}?): {e}", detail=str(e))
        body = r.json()
        if "persons" not in body:
            raise InferError(f"/infer 响应缺 persons 字段: {body}", detail=str(body)[:500])
        return body["persons"]


def get_infer_client(cfg=None):
    """从 config dict (或默认 load_config) 构建进程级单例 InferClient."""
    if cfg is None:
        from hsmr_sdk.config import load_config
        cfg = load_config()
    return InferClient(cfg.get("infer") or {})
