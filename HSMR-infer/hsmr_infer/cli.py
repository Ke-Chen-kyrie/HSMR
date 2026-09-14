#!/usr/bin/env python3
"""CLI: 单张图片 → 24 关节 JSON. 与 server 同一套推理核心.

用法:
  python -m hsmr_infer.cli --image img.jpg [--depth d.png --fx 615 --fy 615 --cx 640 --cy 360]
                           [--json out.json] [--device cpu] [--no-amp]

不带 depth 时只出模型 2D/3D (深度融合关闭). 默认 device 从 config 读.
"""
import argparse
import json
import sys
import time

import numpy as np
import cv2

import hsmr_infer  # noqa: F401  (把 ROOT 加进 sys.path)
from hsmr_infer.inference import (decode_depth, json_safe, load_default_cfg,
                                  HSMRModelManager)


def _load_cfg(args):
    cfg = load_default_cfg(args.config)
    if args.device:
        cfg["model"]["device"] = args.device
        cfg["model"]["onnx_device"] = args.device
    if args.no_amp:
        cfg["detector"]["use_amp"] = False
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="HSMR 单帧推理 CLI")
    p.add_argument("--image", required=True, help="输入彩色图 (jpg/png/bmp)")
    p.add_argument("--depth", default=None, help="对齐深度图 16UC1 PNG (可选)")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--config", default=None, help="config.yaml 路径 (默认工程根)")
    p.add_argument("--json", default=None, help="结果另存 JSON 文件")
    p.add_argument("--device", default=None, help="覆盖 device, 如 cpu / cuda:0")
    p.add_argument("--no-amp", action="store_true", help="关闭 detector AMP (CPU 测试用)")
    p.add_argument("--max-instances", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = _load_cfg(args)
    device = cfg["model"]["device"]

    frame_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        sys.exit(f"无法读取图片: {args.image}")
    depth_arr, K = None, None
    if args.depth:
        if args.depth.lower().endswith(".npy"):
            depth_arr = np.load(args.depth)     # uint16 毫米原始深度
        else:
            depth_arr = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
            if depth_arr is None:
                sys.exit(f"无法读取深度图: {args.depth}")
        try:
            depth_arr = decode_depth(depth_arr)   # 拒彩色可视化/uint8 深度
        except ValueError as e:
            sys.exit(f"深度图校验失败: {e}")
        if None in (args.fx, args.fy, args.cx, args.cy):
            sys.exit("提供 --depth 时必须同时给 --fx/--fy/--cx/--cy")
        K = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]], dtype=float)

    # 单例: 首次调用加载, 进程内复用
    manager = HSMRModelManager.get_instance(cfg, device)
    m = int(args.max_instances or cfg.get("max_instances", 5))
    t0 = time.time()
    res = manager.infer(frame_bgr, depth_arr, K, m)
    infer_ms = round((time.time() - t0) * 1000, 1)

    out = {
        "image": args.image,
        "device": device,
        "num_persons": len(res["persons"]),
        "depth_fused": bool(res.get("depth_fused")),
        "infer_ms": infer_ms,
        "timestamp": res["timestamp"],
        "persons": res["persons"],
    }
    # NaN→null, allow_nan=False: 输出严格合法 JSON (无效深度关节为 null)
    text = json.dumps(json_safe(out), ensure_ascii=False, allow_nan=False, indent=1)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已保存: {args.json}")
    else:
        print(text)


if __name__ == "__main__":
    main()
