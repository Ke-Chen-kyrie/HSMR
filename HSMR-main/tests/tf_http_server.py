#!/usr/bin/env python3
"""HSMR 关节 TF HTTP 服务 (轻量, stdlib only).

后台持续推理, 通过 HTTP GET 返回 head/thorax/lumbar 相对于相机的 TF (位置+朝向).

用法:
  .venv_orin/bin/python docs/tf_http_server.py [--device cuda:0] [--port 8003]

请求:
  GET http://<ip>:8003/tf

响应:
  {
    "timestamp": 1234567890.123,
    "persons": [{
      "person_index": 0, "score": 0.987,
      "tfs": {
        "head":         {"translation": {"x":...,"y":...,"z":...}, "rotation_4x4": [[...],...]},
        "thorax":       {"translation": {"x":...,"y":...,"z":...}, "rotation_4x4": [[...],...]},
        "lumbar_body":  {"translation": {"x":...,"y":...,"z":...}, "rotation_4x4": [[...],...]}
      }
    }]
  }

坐标系约定:
  - translation: 相机光学系 (X=右, Y=下, Z=前), 单位米, 来自真实深度
  - rotation_4x4: joint_local (remap后) → 相机光学系的齐次变换矩阵
"""
import os
import sys
import time
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, "deploy", "joint_service"))
sys.path.insert(0, os.path.join(_PROJECT, "thirdparty", "SKEL"))

from capture import MultiTopicCapture
from inference import (
    decode_color, decode_depth, camera_info_k, deproject, compute_raw_cam_t,
)
from bbox_local import IMG_MEAN_255, IMG_STD_255, _img_det2patches
import yaml

# ── 关节组 remap 矩阵 (与 test_capture_save.py 同) ──
_I = np.eye(3, dtype=np.float64)
_JOINT_GROUPS = {
    "pelvis":     {"joints": [0],                   "remap": np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)},
    "torso":      {"joints": [11, 12, 13],          "remap": np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)},
    "right_arm":  {"joints": [14, 15, 16, 17, 18],  "remap": np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)},
    "left_arm":   {"joints": [19, 20, 21, 22, 23],  "remap": np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)},
    "right_leg":  {"joints": [1, 2, 3, 4, 5],       "remap": np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)},
    "left_leg":   {"joints": [6, 7, 8, 9, 10],      "remap": np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)},
}

def _build_joint_remap():
    R = [None] * 24
    for g in _JOINT_GROUPS.values():
        for j in g["joints"]:
            R[j] = np.asarray(g["remap"], dtype=np.float64)
    return R

JOINT_REMAP = _build_joint_remap()

# 要发布的关节 (索引 → 名字)
TF_JOINTS = {11: "lumbar_body", 12: "thorax", 13: "head"}


def _sample_depth(depth, u, v, radius=3):
    """简单深度采样 (中位数), 与 inference.sample_depth_m 同."""
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    u0, u1 = max(0, ui - radius), min(w, ui + radius + 1)
    v0, v1 = max(0, vi - radius), min(h, vi + radius + 1)
    win = depth[v0:v1, u0:u1].astype(np.float32) / 1000.0
    valid = win[win > 0.1]
    return float(np.median(valid)) if len(valid) > 0 else None


class InferenceLoop:
    """后台推理循环: 持续抓帧 → 推理 → 缓存最新结果."""

    def __init__(self, cfg, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        self._lock = threading.Lock()
        self._latest = None
        self._stop = False

    def start(self):
        import torch
        print("[capture] 连接 bridge...", flush=True)
        prefix = self.cfg["topics"]["prefix"]
        topics = [
            f"{prefix}/color/image_raw/compressed",
            f"{prefix}/aligned_depth_to_color/image_raw/compressedDepth",
            f"{prefix}/aligned_depth_to_color/camera_info",
        ]
        self.cap = MultiTopicCapture(self.cfg["bridge"]["url"], topics).start()
        self._topics = topics  # 保存供 _loop 直接使用
        print("[capture] bridge 已连接", flush=True)

        print("[load] detector + HSMR (ONNX)...", flush=True)
        from lib.modeling.pipelines.vitdet import build_detector
        self.detector = build_detector(batch_size=1, max_img_size=512, device=self.device, use_amp=False)
        print("[load] detector OK", flush=True)
        from deploy.onnx.runtime import HSMRONNXRuntimePipeline
        self.pipeline = HSMRONNXRuntimePipeline(
            model_path=self.cfg["model"]["onnx_model"],
            model_root=self.cfg["model"]["model_root"],
            device=self.device,
        )
        print("[load] pipeline OK", flush=True)
        print("[ready] 开始推理循环", flush=True)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import torch
        frame_n = 0
        while not self._stop:
            t0 = time.time()
            try:
                snap = self.cap.snapshot()
                color_t, depth_t, info_t = self._topics[:3]
                if color_t not in snap or depth_t not in snap or info_t not in snap:
                    time.sleep(0.5)
                    continue

                K = camera_info_k(snap[info_t])
                depth = decode_depth(snap[depth_t])
                frame_bgr = decode_color(snap[color_t])
                if frame_bgr is None or depth is None:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                H, W = frame_rgb.shape[:2]

                det_out = self.detector([frame_rgb])
                patches, bbx_cs = _img_det2patches(frame_rgb, det_out[0][0], det_out[1][0], max_instances=5)

                if len(patches) == 0:
                    with self._lock:
                        self._latest = {"timestamp": time.time(), "persons": []}
                    time.sleep(0.3)
                    continue

                patches = patches.astype(np.float32)
                patches_n = (patches - IMG_MEAN_255) / IMG_STD_255
                patches_n = np.ascontiguousarray(patches_n.transpose(0, 3, 1, 2))

                with torch.inference_mode():
                    outputs = self.pipeline(torch.from_numpy(patches_n))
                    pd_params = {k: v.detach().cpu().clone() for k, v in outputs["pd_params"].items()}
                    pd_cam_t = outputs["pd_cam_t"].detach().cpu().clone()
                    skel = self.pipeline.skel_model(
                        poses=pd_params["poses"].to(self.device),
                        betas=pd_params["betas"].to(self.device),
                        skelmesh=False,
                    )
                    joints_body = skel.joints_backup.detach().cpu().numpy()
                    joints_ori = skel.joints_ori.detach().cpu().numpy()

                cx_v, cy_v = W / 2, H / 2
                f_v = 5000.0
                raw_cam_t = compute_raw_cam_t(pd_cam_t, np.asarray(bbx_cs), W, H)
                scores = det_out[0][0]["scores"]

                persons = []
                for i in range(len(patches)):
                    joints_cam = joints_body[i] + raw_cam_t[i].numpy()
                    u = f_v * joints_cam[:, 0] / joints_cam[:, 2] + cx_v
                    v = f_v * joints_cam[:, 1] / joints_cam[:, 2] + cy_v

                    tfs = {}
                    pelvis_z = _sample_depth(depth, u[0], v[0], radius=3)

                    for j, name in TF_JOINTS.items():
                        z = _sample_depth(depth, u[j], v[j], radius=3)
                        if z is None or (pelvis_z is not None and abs(z - pelvis_z) > 1.5):
                            z = None

                        if z is not None:
                            opt_pos = deproject(u[j], v[j], z, K)
                        else:
                            opt_pos = np.array([np.nan, np.nan, np.nan])

                        R_orig = joints_ori[i, j]
                        R_remap = JOINT_REMAP[j]
                        R_target = R_orig @ R_remap.T  # new_local → camera optical

                        tfs[name] = {
                            "translation": {"x": float(opt_pos[0]), "y": float(opt_pos[1]), "z": float(opt_pos[2])},
                            "rotation_4x4": [
                                [float(R_target[0,0]), float(R_target[0,1]), float(R_target[0,2]), float(joints_cam[j,0])],
                                [float(R_target[1,0]), float(R_target[1,1]), float(R_target[1,2]), float(joints_cam[j,1])],
                                [float(R_target[2,0]), float(R_target[2,1]), float(R_target[2,2]), float(joints_cam[j,2])],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        }

                    persons.append({
                        "person_index": i,
                        "score": float(scores[i]) if i < len(scores) else None,
                        "tfs": tfs,
                    })

                with self._lock:
                    self._latest = {"timestamp": time.time(), "persons": persons}

                frame_n += 1
                dt = (time.time() - t0) * 1000
                n_valid = sum(1 for p in persons for tf in p["tfs"].values() if not np.isnan(tf["translation"]["x"]))
                n_total = len(persons) * len(TF_JOINTS)
                print(f"[{frame_n}] {len(persons)}人, {n_valid}/{n_total} tf有效, {dt:.0f}ms", flush=True)

            except Exception as e:
                print(f"[inference] {e}")
            time.sleep(0.1)

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stop = True
        self.cap.stop()


class TFHandler(BaseHTTPRequestHandler):
    loop: InferenceLoop = None  # set by main

    def do_GET(self):
        if self.path == "/tf":
            data = self.loop.latest()
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # silent


def main():
    import argparse
    ap = argparse.ArgumentParser(description="HSMR TF HTTP Server")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    config_path = args.config or os.path.join(_PROJECT, "deploy", "joint_service", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    loop = InferenceLoop(cfg, args.device)
    loop.start()

    TFHandler.loop = loop
    server = HTTPServer(("0.0.0.0", args.port), TFHandler)
    print(f"[http] http://0.0.0.0:{args.port}/tf", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
