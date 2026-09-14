#!/usr/bin/env python3
"""逐阶段推理计时基准: detector / HSMR(onnx) / skel 分别计时.

用法:  python3 bench_timing.py <image_path> [width height]
输出每阶段耗时 + onnx provider 列表。
"""
import sys, os, time
import numpy as np
import cv2
import torch

from hsmr_infer.inference import (load_models, load_default_cfg, infer_frame,
                                  IMG_MEAN_255, IMG_STD_255)
from hsmr_infer.bbox_local import _img_det2patches
import deploy.onnx.runtime as ortmod
from deploy.onnx.runtime import HSMRONNXRuntimePipeline

img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 960
H = int(sys.argv[3]) if len(sys.argv) > 3 else 540

cfg = load_default_cfg()
device = cfg["model"].get("device", "cuda:0")

# onnx runtime provider 检查
onnx_dev = cfg["model"].get("onnx_device", device)
providers = HSMRONNXRuntimePipeline._available_providers() if hasattr(
    HSMRONNXRuntimePipeline, "_available_providers") else None
print(f"cfg device={device} onnx_device={onnx_dev}")

print("[load] 加载模型...")
t0 = time.time()
detector, pipeline = load_models(cfg, device)
print(f"[load] 完成 {time.time()-t0:.1f}s")

# session provider 信息
for k, sess in getattr(pipeline, "__dict__", {}).items():
    if hasattr(sess, "get_providers"):
        print(f"  session[{k}] providers={sess.get_providers()}")

img = cv2.imread(img_path)
if img is None:
    sys.exit(f"无法读取 {img_path}")
frame_bgr = cv2.resize(img, (W, H))
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
print(f"frame {frame_rgb.shape[1]}x{frame_rgb.shape[0]}")

# ── 检测器单独计时 ──
N_DET = 3
det_times = []
det_out = None
for i in range(N_DET):
    t = time.time()
    det_out = detector([frame_rgb])
    det_times.append(time.time() - t)
print(f"[detector] 单次: {[f'{x:.2f}' for x in det_times]}s  均值 {np.mean(det_times[1:]):.2f}s (去首次预热)")

# ── patches ──
patches, bbx_cs = _img_det2patches(frame_rgb, det_out[0][0], det_out[1][0], 5)
print(f"检测到 {len(patches)} 人, 框 bbx={np.asarray(bbx_cs).tolist()}")
if len(patches) == 0:
    sys.exit("未检测到人, 换图")

patches_n = patches.astype(np.float32)
patches_n = (patches_n - IMG_MEAN_255) / IMG_STD_255
patches_n = np.ascontiguousarray(patches_n.transpose(0, 3, 1, 2))
pt = torch.from_numpy(patches_n).to(device)
print(f"onnx 输入: {tuple(pt.shape)}")

# ── HSMR onnx pipeline 单独计时 ──
with torch.inference_mode():
    for i in range(N_DET):
        t = time.time()
        outputs = pipeline(pt)
        if i == 0:
            pd_params = {k: v.detach().cpu().clone() for k, v in outputs["pd_params"].items()}
            pd_cam_t = outputs["pd_cam_t"].detach().cpu().clone()
        el = time.time() - t
        print(f"[onnx] 第{i}次: {el:.2f}s")

# ── skel 单独计时 ──
with torch.inference_mode():
    for i in range(N_DET):
        t = time.time()
        skel_out = pipeline.skel_model(poses=pd_params["poses"].to(device),
                                       betas=pd_params["betas"].to(device),
                                       skelmesh=False)
        el = time.time() - t
        print(f"[skel] 第{i}次: {el:.2f}s")

# ── 端到端 infer_frame (含 skel) ──
for i in range(N_DET):
    t = time.time()
    r = infer_frame(detector, pipeline, frame_bgr, None, None, 5, device)
    print(f"[infer_frame] 端到端第{i}次: {time.time()-t:.2f}s  {len(r['persons'])}人")

print("== 计时完成 ==")
