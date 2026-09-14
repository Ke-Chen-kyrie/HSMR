"""detector + HSMR(ONNX) 推理 → 24 个 SKEL 解剖关节 optical 坐标 + 世界朝向.

帧处理逻辑复刻 docs/realtime_depth_3d.py (改为 24 关节):
  彩色图 → detector 检人 → HSMR(onnx) → skel_model → joints_backup(24) + joints_ori(24x3x3)
  → 投影 2D → 真实深度采样 → 反投影到 optical 系 XYZ(米)
后台线程按 interval 循环处理最新帧, 缓存最新结果.
"""
import threading
import time

import numpy as np
import cv2
import torch

from lib.modeling.pipelines.vitdet import build_detector
from bbox_local import IMG_MEAN_255, IMG_STD_255, _img_det2patches

from capture import DEPTH_PNG_OFFSET

# ── 解码 / 投影 / 反投影 (复刻 realtime_depth_3d.py) ──
def decode_color(msg):
    """支持 foxglove(rosbags 对象) 与 rosbridge(dict) 两种形态.

    rosbridge 两种 payload:
      - CompressedImage: msg["data"] = base64 编码的 JPEG/PNG (imdecode)
      - sensor_msgs/Image 原始图: msg["encoding"] in (rgb8/bgr8/mono8) + data 为 base64 裸像素
    """
    import base64
    if isinstance(msg, dict):   # rosbridge
        raw = base64.b64decode(msg["data"])
        if msg.get("encoding") and msg.get("height") and msg.get("width"):
            enc = str(msg["encoding"]).lower()
            h, w = int(msg["height"]), int(msg["width"])
            if enc == "rgb8":
                return cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(h, w, 3),
                                    cv2.COLOR_RGB2BGR)
            if enc == "bgr8":
                return np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            if enc in ("mono8", "8uc1"):
                return cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(h, w),
                                    cv2.COLOR_GRAY2BGR)
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)  # BGR
    d = np.frombuffer(msg.data, dtype=np.uint8)
    return cv2.imdecode(d, cv2.IMREAD_COLOR)  # BGR


def decode_depth(msg):
    """支持 raw Image(16UC1) 与 compressedDepth(PNG). 返回 uint16 毫米.
    rosbridge(dict) 形态: compressedDepth 的 data 是 base64 PNG(剥 12B 头)."""
    if isinstance(msg, dict):   # rosbridge
        import base64
        d = np.frombuffer(base64.b64decode(msg["data"]), dtype=np.uint8)
        return cv2.imdecode(d[DEPTH_PNG_OFFSET:], cv2.IMREAD_UNCHANGED)
    if hasattr(msg, "height") and hasattr(msg, "width"):   # sensor_msgs/Image 原始 16UC1
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    # 旧: CompressedImage PNG (compressedDepth)
    d = np.frombuffer(msg.data, dtype=np.uint8)
    return cv2.imdecode(d[DEPTH_PNG_OFFSET:], cv2.IMREAD_UNCHANGED)


def camera_info_k(msg):
    if isinstance(msg, dict):   # rosbridge: msg["K"] 为 9 元素 list
        return np.array(msg["K"]).reshape(3, 3)
    return np.array(msg.k).reshape(3, 3)


def sample_depth_m(depth, u, v, radius=3):
    """在深度图(uint16 mm)取 (u,v) 周围窗口中位数, 返回米."""
    h, w = depth.shape
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < w and 0 <= v < h):
        return None
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)
    win = depth[v0:v1, u0:u1].astype(np.float32) / 1000.0
    valid = win[win > 0.1]
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def deproject(u, v, z_m, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([(u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m])


def compute_raw_cam_t(pd_cam_t, bbx_cs, img_w, img_h):
    """把 HSMR 相机平移缩放到整幅图虚拟相机系 (focal=5000, 中心=图心)."""
    raw_cam_t = pd_cam_t.clone().float()
    raw_cx, raw_cy = img_w / 2, img_h / 2
    bb = torch.as_tensor(np.asarray(bbx_cs), dtype=raw_cam_t.dtype)  # (N,3)
    bbx_s, bbx_cx, bbx_cy = bb[:, 2], bb[:, 0], bb[:, 1]
    raw_cam_t[:, 2] = pd_cam_t[:, 2] * 256 / bbx_s
    raw_cam_t[:, 1] += (bbx_cy - raw_cy) / 5000 * raw_cam_t[:, 2]
    raw_cam_t[:, 0] += (bbx_cx - raw_cx) / 5000 * raw_cam_t[:, 2]
    return raw_cam_t


# ── 模型加载 ──
def load_models(cfg, device):
    print("[加载] detector + HSMR(onnx)...")
    t0 = time.time()
    detector = build_detector(batch_size=1, max_img_size=cfg["detector"]["max_img_size"],
                              device=device, use_amp=cfg["detector"].get("use_amp", True))
    from deploy.onnx.runtime import HSMRONNXRuntimePipeline
    onnx_dev = cfg["model"].get("onnx_device", device)   # onnx 走 GPU, detector 走 CPU
    pipeline = HSMRONNXRuntimePipeline(
        model_path=cfg["model"]["onnx_model"],
        model_root=cfg["model"]["model_root"],
        device=onnx_dev,
    )
    print(f"[加载] 完成 {time.time() - t0:.1f}s")
    return detector, pipeline


# ── 模型单例 (detector + HSMR ONNX, 全进程只加载一次) ──
class RenderModelManager:
    """HSMR-render 推理模型单例: detector + HSMR(ONNX) 全进程只加载一次.

    服务启动时 (lifespan → worker.start()) 首次 get_instance() 完成加载;
    之后所有请求/后台线程都拿到同一个实例, 不会随请求重复加载模型.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.detector, self.pipeline = load_models(cfg, device)

    @classmethod
    def get_instance(cls, cfg=None, device=None):
        inst = cls._instance
        if inst is None:
            with cls._lock:
                inst = cls._instance
                if inst is None:
                    if cfg is None or device is None:
                        raise ValueError("首次调用必须传 cfg 和 device")
                    inst = cls(cfg, device)
                    cls._instance = inst
        return inst


# ── 单帧处理 ──
def process_frame(detector, pipeline, snap, topics, device, max_instances=5):
    """处理一帧 → dict 或 None. snap 为 capture.snapshot().

    与 SDK capture 一致: depth/camera_info 缺失时降级为彩色图纯推理
    (K=None 时模型虚拟相机投影, 3D 光学坐标以 NaN 输出)."""
    color_t, depth_t, info_t = topics[:3]
    if color_t not in snap:
        return None
    frame_bgr = decode_color(snap[color_t])
    if frame_bgr is None:
        return None
    depth, K = None, None
    if depth_t in snap and info_t in snap:
        try:
            K = camera_info_k(snap[info_t])
            depth = decode_depth(snap[depth_t])
        except Exception:
            depth, K = None, None
    if depth is not None and K is not None and depth.shape[:2] != frame_bgr.shape[:2]:
        depth, K = None, None
    return process_frame_arrays(detector, pipeline, frame_bgr, depth, K,
                                device, max_instances)


def process_frame_arrays(detector, pipeline, frame_bgr, depth, K, device,
                         max_instances=5):
    """处理已解码的彩色/深度数组；供相机抓帧和 HTTP 上传图片共同复用。"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    det_out = detector([frame_rgb])
    patches, bbx_cs = _img_det2patches(frame_rgb, det_out[0][0], det_out[1][0], max_instances)
    if len(patches) == 0:
        return {"persons": [], "timestamp": time.time(),
                "frame_bgr": frame_bgr, "K": K, "frame_rgb": frame_rgb}

    # 归一化 + HSMR 推理
    patches = patches.astype(np.float32)
    patches_n = (patches - IMG_MEAN_255) / IMG_STD_255
    patches_n = np.ascontiguousarray(patches_n.transpose(0, 3, 1, 2))
    with torch.inference_mode():
        outputs = pipeline(torch.from_numpy(patches_n))
        pd_params = {k: v.detach().cpu().clone() for k, v in outputs["pd_params"].items()}
        pd_cam_t = outputs["pd_cam_t"].detach().cpu().clone()
        skel_out = pipeline.skel_model(poses=pd_params["poses"].to(device),
                                       betas=pd_params["betas"].to(device), skelmesh=False)
        joints_body24 = skel_out.joints_backup.detach().cpu().numpy()   # (B,24,3)
        joints_ori = skel_out.joints_ori.detach().cpu().numpy()         # (B,24,3,3)

    H, W = frame_rgb.shape[:2]
    cx_v, cy_v = W / 2, H / 2
    f_v = 5000.0
    raw_cam_t = compute_raw_cam_t(pd_cam_t, np.asarray(bbx_cs), W, H)

    persons = []
    scores = det_out[0][0]["scores"]
    for i in range(len(patches)):
        # 2D 投影: 模型虚拟相机 (仅用于定位像素/画图)
        joints_cam = joints_body24[i] + raw_cam_t[i].numpy()
        u = f_v * joints_cam[:, 0] / joints_cam[:, 2] + cx_v
        v = f_v * joints_cam[:, 1] / joints_cam[:, 2] + cy_v
        # 逐关节: 在投影像素处采样真实深度 → 反投影 → 光学系真实 XYZ (米).
        # 采样失败/深度异常(采到背景/别人) → NaN + depth_valid=False.
        joints_opt = np.full((24, 3), np.nan)
        depth_valid = [False] * 24
        pelvis_z = None
        if depth is not None and K is not None:
            for j in range(24):
                z = sample_depth_m(depth, u[j], v[j], radius=3)
                if z is None:
                    continue
                if j == 0:
                    pelvis_z = z
                elif pelvis_z is not None and abs(z - pelvis_z) > 1.5:
                    continue   # 该关节与骨盆距离差太多 → 深度采到了背景/其他人
                joints_opt[j] = deproject(u[j], v[j], z, K)
                depth_valid[j] = True
        # 骨盆位姿(光学系): 位置=真实深度反投影, 朝向=SKELOutput骨盆朝向
        pelvis_pos_cam = joints_opt[0] if depth_valid[0] else None
        pelvis_ori = joints_ori[i, 0]                  # (3,3) SKELOutput 骨盆朝向
        persons.append({
            "person_index": int(i),
            "score": float(scores[i]) if len(scores) > i else None,
            "joints_optical_m": joints_opt.tolist(),   # (24,3) 光学系真实坐标(米, NaN=无效)
            "joints_2d": np.stack([u, v], axis=-1).tolist(),  # (24,2) 彩色图像素(画图用)
            "joints_ori": joints_ori[i].tolist(),      # (24,3,3) SKELOutput 原始朝向
            "depth_valid": depth_valid,                # 逐关节真实有效性
            "pelvis_pose": {
                # 相机→骨盆向量 = 骨盆3D位置 (光学系原点=相机原点)
                "translation_cam_to_pelvis": pelvis_pos_cam.tolist()
                    if pelvis_pos_cam is not None else None,
                "position_cam": pelvis_pos_cam.tolist() if pelvis_pos_cam is not None else None,
                "rotation_matrix": pelvis_ori.tolist(),   # 骨盆朝向 (SKEL 模型全局系≈相机系)
                "pose_4x4": (np.block([[pelvis_ori, pelvis_pos_cam.reshape(3,1)],
                                       [np.zeros((1,3)), 1.0]]).tolist()
                             if pelvis_pos_cam is not None else None),
                "note": "rotation 取自 SKEL 模型全局系(项目约定≈相机光学系), 平移为真实深度光学系坐标",
            },
        })
    return {"persons": persons, "timestamp": time.time(),
            "frame_bgr": frame_bgr, "K": K, "frame_rgb": frame_rgb,
            # SKEL 骨骼网格渲染所需: 模型姿态参数 + 全图虚拟相机平移 (B,).
            "poses": pd_params["poses"].numpy(),      # (B,46)
            "betas": pd_params["betas"].numpy(),      # (B,10)
            "raw_cam_t": raw_cam_t.numpy(),           # (B,3) 全图相机系平移
    }


# ── 后台推理线程 ──
class InferenceWorker:
    def __init__(self, cfg, capture, device="cuda:0", tf_chain=None):
        self.cfg = cfg
        self.capture = capture
        self.topics = cfg["topics"]["list"]      # 实际订阅 topic 列表
        self.device = device
        self.tf_chain = tf_chain                 # 坐标变换链 (光学→HEAD/BASE/world), 可选
        self.interval = cfg.get("inference_interval", 1.0)
        self.background_interval = cfg.get("background_interval", 0)  # 0=关闭后台线程
        self.max_instances = cfg.get("max_instances", 5)
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()   # 串行化 GPU 推理 (后台线程 + POST 实时推理)
        self._latest = {"persons": [], "timestamp": None, "infer_ms": None}
        self._model_loaded = False
        self._stop = False
        self._thread = None

    def start(self):
        # 模型单例: get_instance 启动时加载一次, 全进程常驻; 请求不复载
        mgr = RenderModelManager.get_instance(self.cfg, self.device)
        self.detector, self.pipeline = mgr.detector, mgr.pipeline
        self._model_loaded = True
        if self.background_interval > 0:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            print(f"[inference] 模型已加载, 后台推理线程已启动 (间隔{self.background_interval}s)")
        else:
            print("[inference] 模型已加载, 后台线程关闭, 由 POST 触发实时推理")

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        while not self._stop:
            t0 = time.time()
            try:
                with self._infer_lock:
                    res = self._process_once()
                if res is not None:
                    res["infer_ms"] = round((time.time() - t0) * 1000, 1)
                    with self._lock:
                        self._latest = res
            except Exception as e:
                print(f"[inference] 错误: {e}")
            # 推理耗时可能超过 interval, 用 max(interval - 耗时, 0) 控制
            elapsed = time.time() - t0
            wait = max(self.interval - elapsed, 0.0)
            time.sleep(wait)

    def _process_once(self):
        """抓当前帧并推理一帧 (需持有 _infer_lock). 返回结果 dict 或 None."""
        snap = self.capture.snapshot()
        # 每次抓帧先更新 TF 缓存 (光学→HEAD/BASE/world), 供位置坐标系变换使用
        if self.tf_chain is not None:
            self.tf_chain.update_tf([snap.get("/tf_static"), snap.get("/tf")])
        return process_frame(self.detector, self.pipeline, snap, self.topics,
                             self.device, self.max_instances)

    def infer_fresh(self):
        """POST 用: 当场抓当前帧并推理 (模型已加载, 不重载). 返回结果 dict 或 None."""
        if not self._model_loaded:
            raise RuntimeError("模型未加载")
        with self._infer_lock:
            t0 = time.time()
            res = self._process_once()
            if res is not None:
                res["infer_ms"] = round((time.time() - t0) * 1000, 1)
                with self._lock:
                    self._latest = res
            return res

    def infer_image(self, frame_bgr, depth=None, K=None, max_instances=None):
        """HTTP 上传帧用：对指定图片推理，不从 foxglove 抓取新帧。"""
        if not self._model_loaded:
            raise RuntimeError("模型未加载")
        with self._infer_lock:
            t0 = time.time()
            res = process_frame_arrays(
                self.detector,
                self.pipeline,
                frame_bgr,
                depth,
                K,
                self.device,
                int(max_instances or self.max_instances),
            )
            if res is not None:
                res["infer_ms"] = round((time.time() - t0) * 1000, 1)
                with self._lock:
                    self._latest = res
            return res

    def latest(self):
        with self._lock:
            return dict(self._latest)

    def ready(self):
        return self._model_loaded
