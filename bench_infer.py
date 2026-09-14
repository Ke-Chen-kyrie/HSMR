#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSMR 前向推理基准脚本 (独立文件, 不改动原工程任何代码).

读取 彩色图 + 深度图 + 相机内参 → 完整 HSMR 前向 (检测→裁剪→HSMR ONNX→SKEL→深度融合反投影),
分阶段计时. 模型经 HSMRModelManager 单例加载 (启动一次, 复现生产加载路径).

用法:
  python bench_infer.py --color docs/_depth_test_color.png \
      --depth docs/_depth3d_depth_mm.npy \
      --fx 910.677 --fy 910.280 --cx 653.789 --cy 374.077 --iters 5
  # 内参也可用 json 文件 (自动读 camera_intrinsics):
  python bench_infer.py --color ... --depth ... --camera-json docs/_depth3d_data.json

需要完整依赖 (torch/cv2/detectron2/onnxruntime) 的 python 解释器运行, 如远程:
  /home/naviai/hsmr_infer_venv/bin/python bench_infer.py ...
"""
import argparse
import json
import os
import statistics
import sys
import time

# ── 定位 HSMR-infer 工程根, 加入 sys.path (不改工程文件) ──
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(HERE, "HSMR-infer")
if not os.path.isdir(DEFAULT_ROOT):
    DEFAULT_ROOT = "/home/naviai/projects/HSMR/HSMR-infer"   # 远程布局兜底


def main():
    ap = argparse.ArgumentParser(description="HSMR 前向推理分阶段计时")
    ap.add_argument("--color", required=True, help="彩色图 (jpg/png/bmp)")
    ap.add_argument("--depth", required=True, help="深度图: uint16 .npy (毫米) 或 16UC1 PNG")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--camera-json", default=None,
                    help="含 camera_intrinsics 的 json (优先于 --fx/--fy/--cx/--cy)")
    ap.add_argument("--iters", type=int, default=5, help="计时轮数 (默认 5)")
    ap.add_argument("--warmup", type=int, default=1, help="计时前热身轮数 (默认 1, 抵消 CUDA warmup)")
    ap.add_argument("--max-instances", type=int, default=5)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="HSMR-infer 工程根")
    args = ap.parse_args()

    ROOT = os.path.abspath(args.root)
    sys.path.insert(0, ROOT)
    import hsmr_infer  # noqa: F401  引导 sys.path (ROOT + thirdparty/SKEL + SMPL)
    import numpy as np
    import cv2
    import torch
    from hsmr_infer.inference import (HSMRModelManager, compute_raw_cam_t,
                                       sample_depth_m, deproject, json_safe)
    from hsmr_infer.bbox_local import IMG_MEAN_255, IMG_STD_255, _img_det2patches

    # ── 读取输入 ──
    frame_bgr = cv2.imread(args.color, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        sys.exit(f"[FAIL] 无法读取彩色图: {args.color}")
    H, W = frame_bgr.shape[:2]

    if args.camera_json:
        with open(args.camera_json, "r", encoding="utf-8") as f:
            cj = json.load(f)
        c = cj.get("camera_intrinsics", {})
        fx, fy, cx, cy = c["fx"], c["fy"], c["cx"], c["cy"]
        print(f"  [内参] 来自 {args.camera_json}: fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")
    elif None not in (args.fx, args.fy, args.cx, args.cy):
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    else:
        sys.exit("[FAIL] 必须给 --camera-json 或 --fx/--fy/--cx/--cy")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    low = args.depth.lower()
    if low.endswith(".npy"):
        depth = np.load(args.depth)
    else:
        depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if depth is None:
        sys.exit(f"[FAIL] 无法读取深度图: {args.depth}")
    if depth.ndim == 3:
        sys.exit(f"[FAIL] 深度图是 {depth.shape[2]} 通道彩色可视化图, 需要 uint16 .npy 原始深度")
    if depth.dtype != np.uint16:
        sys.exit(f"[FAIL] 深度图 dtype={depth.dtype}, 应为 uint16(毫米)")
    if depth.shape[:2] != (H, W):
        sys.exit(f"[FAIL] 深度尺寸 {depth.shape[:2]} 与彩色图 {(H, W)} 不一致")
    print(f"  [输入] 彩色 {W}x{H} 深度 {depth.shape} dtype={depth.dtype}")

    # ── 加载模型 (单例, 一次) ──
    t0 = time.time()
    mgr = HSMRModelManager.get_instance()
    detector, pipeline, device = mgr.detector, mgr.pipeline, mgr.device
    print(f"[加载] 单例模型已就绪 {time.time()-t0:.1f}s")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 单次完整前向, 返回各阶段耗时 (毫秒) 与结果
    def run_once():
        d = {}
        t = time.time()
        det_out = detector([frame_rgb])
        d["detect"] = (time.time() - t) * 1000

        t = time.time()
        patches, bbx_cs = _img_det2patches(frame_rgb, det_out[0][0], det_out[1][0],
                                           args.max_instances)
        d["crop"] = (time.time() - t) * 1000

        n = len(patches)
        if n == 0:
            d.update(detect=0, crop=0, onnx=0, skel=0, post=0,
                     persons=0, joints=0, depth_valid=0)
            return d
        patches_n = (patches.astype(np.float32) - IMG_MEAN_255) / IMG_STD_255
        patches_n = np.ascontiguousarray(patches_n.transpose(0, 3, 1, 2))

        with torch.inference_mode():
            t = time.time()
            outputs = pipeline(torch.from_numpy(patches_n))
            pd_params = {k: v.detach().cpu().clone() for k, v in outputs["pd_params"].items()}
            pd_cam_t = outputs["pd_cam_t"].detach().cpu().clone()
            d["onnx"] = (time.time() - t) * 1000

            t = time.time()
            skel_out = pipeline.skel_model(poses=pd_params["poses"].to(device),
                                           betas=pd_params["betas"].to(device),
                                           skelmesh=False)
            joints_body24 = skel_out.joints_backup.detach().cpu().numpy()  # (B,24,3)
            joints_ori = skel_out.joints_ori.detach().cpu().numpy()        # (B,24,3,3)
            d["skel"] = (time.time() - t) * 1000

            t = time.time()
            raw_cam_t = compute_raw_cam_t(pd_cam_t, np.asarray(bbx_cs), W, H)
            scores = det_out[0][0]["scores"]
            f_v, cx_v, cy_v = 5000.0, W / 2, H / 2
            depth_valid_cnt = 0
            for i in range(n):
                joints_cam = joints_body24[i] + raw_cam_t[i].numpy()
                u = f_v * joints_cam[:, 0] / joints_cam[:, 2] + cx_v
                v = f_v * joints_cam[:, 1] / joints_cam[:, 2] + cy_v
                for j in range(24):
                    z = sample_depth_m(depth, u[j], v[j], radius=3)
                    if z is not None:
                        deproject(u[j], v[j], z, K)
                        depth_valid_cnt += 1
            d["post"] = (time.time() - t) * 1000

        d.update(persons=n, joints=n * 24,
                 depth_valid=depth_valid_cnt,
                 scores=[round(float(s), 3) for s in scores[:n]])
        return d

    # ── 热身 (CUDA/cuDNN 首次调用很慢, 不计时) ──
    for _ in range(args.warmup):
        run_once()
    print(f"[热身] {args.warmup} 轮完成")

    # ── 计时 ──
    runs = []
    for i in range(1, args.iters + 1):
        r = run_once()
        runs.append(r)
        r["iter"] = i
        print(f"  第{i:2d}次  total={r['detect']+r['crop']+r['onnx']+r['skel']+r['post']:7.1f}ms "
              f"detect={r['detect']:6.1f} crop={r['crop']:4.1f} "
              f"onnx={r['onnx']:6.1f} skel={r['skel']:4.1f} post={r['post']:4.1f} "
              f"persons={r['persons']}")

    # ── 汇总 (中位数最抗抖动) ──
    def med(key):
        vals = [r[key] for r in runs]
        return statistics.median(vals)

    p = runs[0]["persons"]
    print("\n==== 分阶段耗时汇总 (中位数, ms) ====")
    print(f"  persons   = {p}")
    print(f"  detect    = {med('detect'):7.1f}   (ViTDet 检测, 固定成本)")
    print(f"  crop      = {med('crop'):7.1f}   (裁剪 {p} 个 patch)")
    print(f"  onnx      = {med('onnx'):7.1f}   (HSMR ONNX 前向, B={p})")
    print(f"  skel      = {med('skel'):7.1f}   (SKEL 蒙皮参数)")
    print(f"  post      = {med('post'):7.1f}   (投影+深度采样+反投影, {runs[0]['depth_valid']} 个有效关节)")
    total = med('detect') + med('crop') + med('onnx') + med('skel') + med('post')
    print(f"  ──────────────────────────────")
    print(f"  合计      = {total:7.1f} ms/帧")
    print(f"  scores    = {runs[0].get('scores')}")


if __name__ == "__main__":
    main()
