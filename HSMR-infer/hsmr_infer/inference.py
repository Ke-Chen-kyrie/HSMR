"""HSMR 推理核心: 单帧 detector + HSMR(ONNX) → 24 SKEL 关节.

与原工程 joint_service/inference.py 同源, 但**剥离 capture/话题/服务依赖**:
只接受 numpy 图像, 返回纯数据. 深度融合(真实深度反投影)可选:
  - 给 depth(uint16 mm) + K → 输出光学系 3D 米坐标 (joints_optical_m) + 逐关节 depth_valid
  - 不给 → 只出模型 2D/3D (joints_optical_m 为 NaN, depth_valid 全 False)

模型 I/O 契约 (与原工程一致):
  输入: 任意尺寸 BGR 图 (numpy, HxWx3)
  输出: persons[]:
    joints_2d        (24,2)  彩色图像素
    joints_optical_m (24,3)  光学系真实 3D 米坐标 (NaN=无效, 无深度融合时全 NaN)
    joints_ori       (24,3,3) SKEL 原始朝向 (X=右/Y=上/Z=前 的模型系, 见 README)
    depth_valid      [24]bool 逐关节深度有效
    pelvis_pose      {position_cam, rotation_matrix, pose_4x4}

模型常驻: 用 HSMRModelManager 单例持有 detector + pipeline, 全进程只加载一次,
每次请求调用 get_instance() 复用同一实例, 不会重复加载模型.
"""
import io
import os
import threading
import time

import numpy as np
import cv2
import torch
import yaml

from lib.modeling.pipelines.vitdet import build_detector
from hsmr_infer.bbox_local import IMG_MEAN_255, IMG_STD_255, _img_det2patches


# ── JSON 安全化 / 深度解码 ──
def json_safe(obj):
    """递归把 float NaN/Inf → None, 使输出为合法 JSON.

    Starlette 序列化用 allow_nan=False, 含 NaN 会直接 ValueError; 且 NaN 本就不是合法 JSON.
    无效深度关节 (joints_optical_m 中的 NaN) 由这里转为 null, 契约含义不变: null=无效.
    """
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    return obj


def validate_depth(arr):
    """校验/规范化深度图为单通道 uint16 (H,W) 毫米. 非法抛 ValueError 并说明原因.

    常见的坑: 深度相机同时会存一张彩色可视化 PNG (uint8 3通道), 通道值≠毫米.
    若放进来会被当成深度 → 3D 全错且无报错. 这里显式拒绝, 只收原始深度.
    """
    if arr.ndim == 3:
        raise ValueError(
            f"深度图是 {arr.shape[2]} 通道彩色图(可视化), 不是原始深度. "
            "请提供 16UC1 uint16 PNG 或 uint16 .npy")
    if arr.dtype != np.uint16:
        raise ValueError(
            f"深度图 dtype={arr.dtype}, 应为 uint16(毫米). "
            "彩色可视化/uint8 深度会被拒, 请用原始深度")
    return arr


def decode_depth(data, filename=""):
    """解码深度输入 → 校验过的单通道 uint16 (H,W).

    data: 文件字节 (bytes) 或已加载的 ndarray; filename 决定 bytes 的解码方式:
      .npy → np.load; 否则按 16UC1 PNG (cv2.imdecode UNCHANGED).
    非法输入抛 ValueError (含原因).
    """
    if isinstance(data, np.ndarray):
        return validate_depth(data)
    low = (filename or "").lower()
    if low.endswith(".npy"):
        arr = np.load(io.BytesIO(data))
    else:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError("无法解码深度图 (需 16UC1 PNG 或 uint16 .npy)")
    return validate_depth(arr)


# ── 投影 / 反投影 (复刻原工程) ──
def sample_depth_m(depth, u, v, radius=3):
    """在深度图(uint16 mm)取 (u,v) 周围窗口中位数, 返回米. 越界/无效返回 None."""
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


# ── 配置加载 ──
def load_default_cfg(cfg_path=None):
    """读取服务配置 config.yaml, 模型路径相对工程根解析 (不依赖 CWD).

    cfg_path 为 None 时依次取环境变量 HSMR_INFER_CONFIG → 工程根 config.yaml.
    server 与 CLI 共用同一套配置逻辑.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = cfg_path or os.environ.get("HSMR_INFER_CONFIG", os.path.join(root, "config.yaml"))
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("onnx_model", "model_root"):
        val = cfg["model"].get(key)
        if val and not os.path.isabs(val):
            cfg["model"][key] = os.path.join(root, val)
    return cfg


# ── 模型加载 ──
def load_models(cfg, device="cuda:0"):
    """加载 detector + HSMR(ONNX). cfg 需含 model/detector 段."""
    print("[加载] detector + HSMR(onnx)...")
    t0 = time.time()
    detector = build_detector(batch_size=1,
                              max_img_size=cfg["detector"]["max_img_size"],
                              device=device,
                              use_amp=cfg["detector"].get("use_amp", True))
    from deploy.onnx.runtime import HSMRONNXRuntimePipeline
    onnx_dev = cfg["model"].get("onnx_device", device)
    pipeline = HSMRONNXRuntimePipeline(
        model_path=cfg["model"]["onnx_model"],
        model_root=cfg["model"]["model_root"],
        device=onnx_dev,
    )
    print(f"[加载] 完成 {time.time() - t0:.1f}s")
    return detector, pipeline


class HSMRModelManager:
    """HSMR 模型单例管理器: detector + HSMR(ONNX) 全进程只加载一次.

    每次请求调用 get_instance() 都返回同一个实例, 模型常驻内存复用,
    不会每次请求重新加载. 线程安全 (双重检查锁): 多线程并发首次调用
    也只会真正加载一次, 其余线程直接拿到缓存实例.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.detector, self.pipeline = load_models(cfg, device)

    @classmethod
    def get_instance(cls, cfg=None, device=None):
        """返回全局唯一实例.

        cfg/device 仅在首次加载时生效, 之后直接返回缓存的同一实例.
        cfg 为 None 时自动读取 config.yaml (HSMR_INFER_CONFIG 可覆盖路径);
        device 为 None 时用 cfg 里的 device.
        """
        inst = cls._instance
        if inst is None:
            with cls._lock:
                inst = cls._instance
                if inst is None:
                    if cfg is None:
                        cfg = load_default_cfg()
                    if device is None:
                        device = cfg["model"].get("device", "cuda:0")
                    inst = cls(cfg, device)
                    cls._instance = inst
        return inst

    def infer(self, frame_bgr, depth=None, K=None, max_instances=5):
        """单帧推理. 内部串行使用同一 GPU 模型, 调用方需自行加锁."""
        return infer_frame(self.detector, self.pipeline, frame_bgr, depth, K,
                           max_instances, self.device)


# ── 单帧推理 ──
def infer_frame(detector, pipeline, frame_bgr, depth=None, K=None,
                max_instances=5, device="cuda:0"):
    """处理一帧 → dict. frame_bgr: BGR ndarray.

    depth (可选): uint16 毫米深度图, 与彩色图同尺寸对齐.
    K     (可选): 3x3 相机内参. depth 与 K 必须同时给出才做深度融合.
    返回 {"persons", "timestamp", "frame_bgr", "K", "frame_rgb",
          "poses", "betas", "raw_cam_t"}.
    """
    do_fusion = depth is not None and K is not None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    det_out = detector([frame_rgb])
    patches, bbx_cs = _img_det2patches(frame_rgb, det_out[0][0], det_out[1][0], max_instances)
    if len(patches) == 0:
        return {"persons": [], "timestamp": time.time(),
                "frame_bgr": frame_bgr, "K": K, "frame_rgb": frame_rgb,
                "depth_fused": do_fusion}

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
        # 逐关节: 真实深度采样 → 反投影 → 光学系 XYZ (米). 无深度融合时全 NaN.
        joints_opt = np.full((24, 3), np.nan)
        depth_valid = [False] * 24
        pelvis_z = None
        if do_fusion:
            for j in range(24):
                z = sample_depth_m(depth, u[j], v[j], radius=3)
                if z is None:
                    continue
                if j == 0:
                    pelvis_z = z
                elif pelvis_z is not None and abs(z - pelvis_z) > 1.5:
                    continue   # 与骨盆距离差太多 → 采到背景/其他人
                joints_opt[j] = deproject(u[j], v[j], z, K)
                depth_valid[j] = True
        pelvis_pos_cam = joints_opt[0].tolist() if depth_valid[0] else None
        pelvis_ori = joints_ori[i, 0]                  # (3,3) SKELOutput 骨盆朝向
        persons.append({
            "person_index": int(i),
            "score": float(scores[i]) if len(scores) > i else None,
            "joints_optical_m": joints_opt.tolist(),   # (24,3) 光学系真实坐标(米, NaN=无效)
            "joints_2d": np.stack([u, v], axis=-1).tolist(),  # (24,2) 彩色图像素
            "joints_ori": joints_ori[i].tolist(),      # (24,3,3) SKEL 原始朝向
            "depth_valid": depth_valid,                # 逐关节真实有效性
            "depth_fused": do_fusion,
            "pelvis_pose": {
                "translation_cam_to_pelvis": pelvis_pos_cam,
                "position_cam": pelvis_pos_cam,
                "rotation_matrix": pelvis_ori.tolist(),   # 骨盆朝向
                "pose_4x4": (np.block([[pelvis_ori, np.asarray(pelvis_pos_cam).reshape(3, 1)],
                                       [np.zeros((1, 3)), 1.0]]).tolist()
                             if pelvis_pos_cam is not None else None),
                "note": "rotation 取自 SKEL 模型全局系(项目约定≈相机光学系), 平移为真实深度光学系坐标",
            },
        })
    return {"persons": persons, "timestamp": time.time(),
            "frame_bgr": frame_bgr, "K": K, "frame_rgb": frame_rgb,
            "depth_fused": do_fusion,
            "poses": pd_params["poses"].numpy(),      # (B,46)
            "betas": pd_params["betas"].numpy(),      # (B,10)
            "raw_cam_t": raw_cam_t.numpy(),           # (B,3) 全图相机系平移
    }
