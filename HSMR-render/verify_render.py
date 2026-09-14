#!/usr/bin/env python3
"""离线渲染验证: 不连 foxglove / 不跑 detector / onnx, 直接用存帧 + 已保存的网格参数
(poses/betas/full_cam_t) 走 _render_and_enqueue 全路径 (EGL 渲染 + 存盘 + 入队 + pop).

素材: HSMR-main/data_outputs/infraed_demo/HSMR-*.jpg.npz (4 人份 poses/betas/full_cam_t)
     + 对应 .jpg (3 面板拼接源图).

用法 (本地, CPU 软件 EGL):
  cd /home/Kai/pose_model/HSMR-render
  /home/Kai/pose_model/HSMR-main/.venv/bin/python verify_render.py
"""
import argparse
import base64
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import cv2

import main as render_mod  # 触发 sys.path 引导 + _load_cfg (别名避免与下方 main() 冲突)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="网格参数 npz (默认 infraed_demo 第一组)")
    ap.add_argument("--jpg", default=None, help="源图 jpg (默认与 npz 同名)")
    ap.add_argument("--seq", type=int, default=0, help="标记当前帧序号 (默认 0)")
    args = ap.parse_args()

    demo = os.path.join(render_mod._HSRM_ROOT, "data_outputs", "infraed_demo")
    npz = args.npz or os.path.join(demo, "HSMR-20260812-172527.jpg.npz")
    jpg = args.jpg or (npz[:-4] + ".jpg")
    print(f"素材: npz={npz}\n      jpg={jpg}")

    d = np.load(npz)
    frame_bgr = cv2.imread(jpg, cv2.IMREAD_COLOR)
    assert frame_bgr is not None, f"读图失败: {jpg}"
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    print(f"源图 {frame_rgb.shape}  网格参数 {d['poses'].shape[0]} 人份")

    cfg = render_mod._load_cfg()
    cfg["model"]["device"] = "cpu"               # 本机无 GPU → SKEL 前向走 CPU
    cfg["model"]["onnx_device"] = "cpu"
    cfg["render"]["save_output"] = True          # 存盘验证
    renderer = render_mod._get_skel_renderer(cfg)   # 只加载 SKEL 渲染模型 (不加载 detector/onnx)
    render_mod._ensure_render_executor()

    pi = 0
    latest = {
        "frame_rgb": frame_rgb,
        "poses": d["poses"],           # (4,46)
        "betas": d["betas"],           # (4,10)
        "raw_cam_t": d["full_cam_t"],  # (4,3) 全图虚拟相机系平移 (语义等同 raw_cam_t)
        "persons": [{"person_index": i} for i in range(d["poses"].shape[0])],
        "timestamp": 0.0,
    }
    en, cn = ["head"], ["头"]

    # ① 带光晕渲染当前帧 (seq) → 入队
    item = render_mod._render_and_enqueue(latest, pi, {13}, en, cn, args.seq, cfg["render"])
    assert item is not None, "渲染/入队失败"
    assert item["seq"] == args.seq and item["joints"] == ["head"]
    assert item["image_base64"], "缺少 image_base64"

    # ② 校验图非空且与源图有差异 (蒙皮/骨骼叠加生效)
    out_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(item["image_base64"]), np.uint8),
                           cv2.IMREAD_COLOR)
    assert out_bgr is not None and out_bgr.shape == frame_bgr.shape
    diff = float(np.abs(out_bgr.astype(int) - frame_bgr.astype(int)).mean())
    assert diff > 1.0, f"渲染图与源图几乎无差异 (mean diff={diff:.2f}) — 蒙皮/骨骼可能没画上"

    # ③ 光晕生效: 带光晕 vs 不带光晕 应有差异 (seq=99 仅作对照, 不入断言队列).
    #    全图 mean 会被 2778 宽源图稀释, 用 max 差 + 显著变化像素数判断.
    item2 = render_mod._render_and_enqueue(latest, pi, set(), en, cn, 99, cfg["render"])
    out2 = cv2.imdecode(np.frombuffer(base64.b64decode(item2["image_base64"]), np.uint8),
                        cv2.IMREAD_COLOR)
    diff_px = np.abs(out_bgr.astype(int) - out2.astype(int))
    diff_glow = float(diff_px.mean())
    max_px = int(diff_px.max())
    n_changed = int((diff_px > 20).sum())
    assert max_px > 50 and n_changed > 500, \
        f"光晕差异过小 (mean={diff_glow:.2f} max={max_px} changed={n_changed})"

    # ④ 队列 pop 回当前条目 (先入先出, item 先于 item2)
    popped = render_mod._RENDER_QUEUE.get_nowait()
    assert popped["seq"] == args.seq, "队列 pop 回不到当前条目"
    assert popped["num_persons"] == 4

    out_dir = cfg["render"]["out_dir"]
    saved = sorted(f for f in os.listdir(out_dir) if f.endswith(f"_seq{args.seq}.jpg"))
    print(f"渲染 OK: shape={out_bgr.shape} mesh_diff={diff:.2f} glow[max={max_px} changed={n_changed}] "
          f"pop_seq={popped['seq']} 存盘目录={out_dir}/ ({len(saved)} 个 seq{args.seq} 文件)")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
