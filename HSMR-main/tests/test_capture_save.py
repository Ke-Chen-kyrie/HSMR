#!/usr/bin/env python3
"""一次性抓帧 + HSMR 推理 + 深度查询 + 坐标轴绘制 + 归档保存.

用途: 临时测试, 验证虚拟相机系关节位姿精度 + 深度图查询一致性.

输出 (时间戳子目录):
  rgb.jpg          原始彩色图
  depth.png        伪彩色深度图
  depth_raw.npy    原始 uint16 mm 深度数组
  annotated.jpg    叠加关节 + 坐标轴 (红=X前, 绿=Y左, 蓝=Z上)
  data.json        每关节像素/虚拟相机4×4/深度统计/光学系3D/BASE系位姿

坐标系: 输出使用统一约定 (X=前, Y=左, Z=上). 逐关节用 JOINT_REMAP 从 SKEL 原始转换.
BASE 系字段 (position_base_m/rotation_base_3x3) 通过订阅 /tf 查询 optical→BASE 得到,
机制同 tests/pose_cam_to_base.py; 无 TF 时该字段为 null.

用法:
  .venv_orin/bin/python docs/test_capture_save.py [--device cuda:0] [--out-dir ...]
"""
import os
import sys
import time
import json
import argparse
from datetime import datetime

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, "deploy", "joint_service"))
sys.path.insert(0, os.path.join(_PROJECT, "thirdparty", "SKEL"))

from capture import MultiTopicCapture
from inference import (
    decode_color,
    decode_depth,
    camera_info_k,
    deproject,
    compute_raw_cam_t,
)
from bbox_local import IMG_MEAN_255, IMG_STD_255, _img_det2patches
import yaml

# ═══════════════════════════════════════════════════════════════════
# 逐关节坐标系重映射矩阵 (SKEL 原始 → 统一约定 X=前, Y=左, Z=上).
# 6 组合部填 [[1,0,0],[0,0,-1],[0,1,0]]: new_x=old_x, new_y=-old_z, new_z=old_y.
# 本文件是 remap 的唯一来源; deploy/joint_service/main.py 的 _JOINT_REMAP 与之保持一致.
#   组: pelvis / torso / right_leg / left_leg / right_arm / left_arm
# 用法
#   R_unified = R_orig @ R_remap.T（作用在局部侧 不要忘记转置）
#   R_remap 是作用在关节局部坐标系上的变换，矩阵的第 i 行 = 新轴 i 在旧局部坐标系中的方向。
_I = np.eye(3, dtype=np.float64)
_JOINT_GROUPS = {
    # new_x=old_x, new_y=-old_z, new_z=old_y
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
    assert all(r is not None for r in R), f"Missing remap for joints: {[i for i,r in enumerate(R) if r is None]}"
    return R

JOINT_REMAP = _build_joint_remap()  # list of 24 (3,3) matrices

SKEL_JOINTS = [
    "pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
    "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l",
    "lumbar_body", "thorax", "head",
    "scapula_r", "humerus_r", "ulna_r", "radius_r", "hand_r",
    "scapula_l", "humerus_l", "ulna_l", "radius_l", "hand_l",
]

AXIS_LEN = 0.12        # 米, 坐标轴长度
AXIS_COLORS = [        # BGR
    (0, 0, 255),       # X=红 (前)
    (0, 255, 0),       # Y=绿 (左)
    (255, 0, 0),       # Z=蓝 (上)
]
# 统一约定的标准基向量 (X=前, Y=左, Z=上)
_STD_BASES = [
    np.array([1, 0, 0], dtype=np.float64),
    np.array([0, 1, 0], dtype=np.float64),
    np.array([0, 0, 1], dtype=np.float64),
]


# ═══════════════════════════════════════════════════════════════════
# 增强深度采样
# ═══════════════════════════════════════════════════════════════════
def sample_depth_stats(depth, u, v, radius=5, outlier_std=3.0):
    """在深度图 (uint16 mm) 的 (u,v) 邻域做统计采样, 剔除离群后取均值.

    Args:
        depth: uint16 mm 深度图
        u, v: 像素坐标 (浮点)
        radius: 半窗大小 (默认 5 → 11×11)
        outlier_std: 偏离中位数超过 Nσ 的视为离群

    Returns:
        {median_m, mean_m, std_m, valid_pixel_count}
        全部为 None/0 表示无有效深度.
    """
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return {"median_m": None, "mean_m": None, "std_m": None, "valid_pixel_count": 0}

    u0, u1 = max(0, ui - radius), min(w, ui + radius + 1)
    v0, v1 = max(0, vi - radius), min(h, vi + radius + 1)
    win = depth[v0:v1, u0:u1].astype(np.float32) / 1000.0  # mm → m

    valid = win[(win > 0.1) & np.isfinite(win)]
    if len(valid) == 0:
        return {"median_m": None, "mean_m": None, "std_m": None, "valid_pixel_count": 0}

    med = float(np.median(valid))
    std = float(np.std(valid))
    inliers = valid[np.abs(valid - med) <= outlier_std * std]
    if len(inliers) == 0:
        inliers = valid  # 退化为全部有效值

    return {
        "median_m": float(np.median(inliers)),
        "mean_m": float(np.mean(inliers)),
        "std_m": float(np.std(inliers)),
        "valid_pixel_count": int(len(inliers)),
    }


# ═══════════════════════════════════════════════════════════════════
# 标注图绘制
# ═══════════════════════════════════════════════════════════════════
def draw_annotated_image(frame_bgr, persons_data, K):
    """在彩色图上叠加相机参考轴 + 逐关节统一坐标轴 (X=前/红, Y=左/绿, Z=上/蓝).

    每个关节用其 JOINT_REMAP 矩阵将 SKEL 原始朝向转为统一约定后绘制.
    """
    img = frame_bgr.copy()
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    def _proj(p3):
        """光学系 3D → 像素. z <= 0.05 返回 None."""
        if p3[2] <= 0.05:
            return None
        return (int(fx * p3[0] / p3[2] + cx), int(fy * p3[1] / p3[2] + cy))

    # ── 相机参考轴 (左上角, 光学系: X=右, Y=下) ──
    ref_origin = (90, 90)
    ref_len = 55
    for ai, d in enumerate([(1, 0), (0, 1)]):  # x右, y下
        cv2.arrowedLine(
            img, ref_origin,
            (ref_origin[0] + d[0] * ref_len, ref_origin[1] + d[1] * ref_len),
            AXIS_COLORS[ai], 3, tipLength=0.2,
        )
    cv2.putText(img, "cam +x ->", (ref_origin[0] + ref_len + 5, ref_origin[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, AXIS_COLORS[0], 1)
    cv2.putText(img, "cam +y(down)", (ref_origin[0] + 8, ref_origin[1] + ref_len + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, AXIS_COLORS[1], 1)
    cv2.putText(img, "bone: R=X(fwd) G=Y(left) B=Z(up)",
                (ref_origin[0] - 30, ref_origin[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    person_colors = [(0, 255, 0), (255, 165, 0), (0, 255, 255), (255, 0, 255)]

    for pi, person in enumerate(persons_data):
        color = person_colors[pi % len(person_colors)]
        for jd in person.get("joints", []):
            u, v = jd.get("pixel_u"), jd.get("pixel_v")
            if u is None or v is None:
                continue
            u, v = int(round(u)), int(round(v))
            if not (0 <= u < img.shape[1] and 0 <= v < img.shape[0]):
                continue

            # 关节点 + 名字
            cv2.circle(img, (u, v), 4, color, -1)
            cv2.putText(img, jd.get("joint_name", ""), (u + 5, v - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            # 3D 坐标轴 (JOINT_REMAP 作用在关节局部侧: new_world = R_orig @ R_remap.T @ base)
            R_orig = jd.get("rotation_skel_3x3")
            opt_pos = jd.get("optical_position_m")
            j_idx = jd.get("joint_index", 0)
            if R_orig is None or opt_pos is None:
                continue
            p_origin = np.array(opt_pos, dtype=float)
            if np.isnan(p_origin).any():
                continue
            R_orig = np.array(R_orig, dtype=float)
            R_remap = JOINT_REMAP[j_idx]
            R_unified = R_orig @ R_remap.T

            for ai, base in enumerate(_STD_BASES):
                end3 = p_origin + AXIS_LEN * (R_unified @ base)
                p1 = _proj(p_origin)
                p2 = _proj(end3)
                if p1 is not None and p2 is not None:
                    cv2.line(img, p1, p2, AXIS_COLORS[ai], 2)

    return img


# ═══════════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════════
def save_outputs(out_dir, frame_bgr, depth, annotated, K, H, W, persons_data,
                 base_meta=None):
    """base_meta: {base_frame, base_transform_4x4, base_source} 或 None (无 TF)."""
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "rgb.jpg"), frame_bgr)
    np.save(os.path.join(out_dir, "depth_raw.npy"), depth)

    # 深度伪彩色
    d_vis = np.zeros(depth.shape, dtype=np.uint8)
    valid_d = depth > 0
    d_vis[valid_d] = np.clip(depth[valid_d] // 40, 0, 255).astype(np.uint8)
    depth_col = cv2.applyColorMap(d_vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(out_dir, "depth.png"), depth_col)

    if annotated is not None:
        cv2.imwrite(os.path.join(out_dir, "annotated.jpg"), annotated)

    data = {
        "schema": "test_capture_save_v1",
        "timestamp": os.path.basename(out_dir),
        "timestamp_unix": time.time(),
        "coord_convention": "x_forward_y_left_z_up_per_joint_remap",
        "base_frame": (base_meta or {}).get("base_frame"),
        "base_transform_4x4": (base_meta or {}).get("base_transform_4x4"),
        "base_source": (base_meta or {}).get("base_source"),
        "joint_groups": {name: {"joints": g["joints"], "remap": g["remap"].tolist()} for name, g in _JOINT_GROUPS.items()},
        "camera_intrinsics": {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "K_matrix": K.tolist(),
        },
        "image_size": {"width": W, "height": H},
        "num_persons_detected": len(persons_data) if persons_data else 0,
        "persons": persons_data or [],
    }
    with open(os.path.join(out_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  → rgb.jpg ({frame_bgr.shape[1]}x{frame_bgr.shape[0]})")
    print(f"  → depth.png + depth_raw.npy ({depth.shape[1]}x{depth.shape[0]})")
    if annotated is not None:
        print(f"  → annotated.jpg ({annotated.shape[1]}x{annotated.shape[0]})")
    n = len(persons_data) if persons_data else 0
    print(f"  → data.json ({n} person(s))")


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════
def run_capture_and_save(cfg, device, out_dir, detector=None, pipeline=None, cap=None):
    """抓一帧 → 推理 → 深度统计 → 坐标轴绘制 → 保存.

    如果传了 detector/pipeline/cap, 则复用已有实例 (loop 模式).
    """
    import torch

    _own_cap = (cap is None)
    if _own_cap:
        prefix = cfg["topics"]["prefix"]
        topics = [
            f"{prefix}/color/image_raw/compressed",
            f"{prefix}/aligned_depth_to_color/image_raw/compressedDepth",
            f"{prefix}/aligned_depth_to_color/camera_info",
            "/tf_static",
            "/tf",
        ]
        print("[capture] 连接 Foxglove bridge...")
        cap = MultiTopicCapture(cfg["bridge"]["url"], topics).start()
        t0 = time.time()
        snap = None
        while time.time() - t0 < 15:
            snap = cap.snapshot()
            if all(t in snap for t in topics):
                break
            time.sleep(0.5)
        if snap is None or not all(t in snap for t in topics):
            print("FAIL: 主题未就绪 (等15s)")
            cap.stop()
            return None

    t0 = time.time()
    snap = cap.snapshot()
    topics_keys = list(snap.keys())
    color_t = [t for t in topics_keys if "/color/" in t]
    depth_t = [t for t in topics_keys if "/aligned_depth" in t]
    info_t = [t for t in topics_keys if "camera_info" in t]
    if not (color_t and depth_t and info_t):
        print("FAIL: 缺少必需topic")
        if _own_cap: cap.stop()
        return None
    color_t, depth_t, info_t = color_t[0], depth_t[0], info_t[0]

    # ── 2. 解码 ──
    K = camera_info_k(snap[info_t])
    depth = decode_depth(snap[depth_t])
    frame_bgr = decode_color(snap[color_t])
    if frame_bgr is None or depth is None:
        print("FAIL: 无彩色图或深度图")
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_rgb.shape[:2]
    print(f"[decode] {W}x{H}, depth {depth[depth>0].min()}-{depth.max()} mm")

    # ── 2.5. base 变换链 (TF: optical → BASE, 复用 pose_cam_to_base.py 的机制) ──
    from tf_chain import TransformChain
    chain = TransformChain(cfg)
    chain.update_tf([snap.get("/tf_static"), snap.get("/tf")])
    r_base = chain.get_opt_to_target("BASE")
    base_source = chain.transform_source("BASE")
    base_meta = None
    if r_base is not None:
        R_ot_b, t_ot_b = r_base
        T_b = np.eye(4)
        T_b[:3, :3] = R_ot_b
        T_b[:3, 3] = t_ot_b
        base_meta = {
            "base_frame": "BASE",
            "base_transform_4x4": T_b.tolist(),
            "base_source": base_source,
        }
        print(f"[base] optical→BASE 可用 (source={base_source}), t={np.round(t_ot_b, 3).tolist()}")
    else:
        print("[base] 无 optical→BASE TF, base 字段置 null")

    # ── 3. 加载模型 (若未预加载) ──
    if detector is None or pipeline is None:
        print("[load] detector + HSMR (ONNX)...")
        from lib.modeling.pipelines.vitdet import build_detector
        detector = build_detector(
            batch_size=1, max_img_size=512, device=device, use_amp=True,
        )
        from deploy.onnx.runtime import HSMRONNXRuntimePipeline
        pipeline = HSMRONNXRuntimePipeline(
            model_path=cfg["model"]["onnx_model"],
            model_root=cfg["model"]["model_root"],
            device=device,
        )
        print(f"[load] done {time.time()-t0:.1f}s")

    # ── 4. 检测 ──
    t_det0 = time.time()
    det_out = detector([frame_rgb])
    t_det1 = time.time()
    patches, bbx_cs = _img_det2patches(
        frame_rgb, det_out[0][0], det_out[1][0], max_instances=5,
    )
    t_det2 = time.time()
    print(f"[detect] {len(patches)} person(s) | "
          f"ViTDet={(t_det1-t_det0)*1000:.0f}ms  crop={(t_det2-t_det1)*1000:.0f}ms")

    if len(patches) == 0:
        save_outputs(out_dir, frame_bgr, depth, None, K, H, W, None, base_meta)
        print(f"[done] 无人 → {out_dir}")
        return out_dir

    # ── 5. HSMR 推理 ──
    patches = patches.astype(np.float32)
    patches_n = (patches - IMG_MEAN_255) / IMG_STD_255
    patches_n = np.ascontiguousarray(patches_n.transpose(0, 3, 1, 2))

    t_hsmr0 = time.time()
    with torch.inference_mode():
        outputs = pipeline(torch.from_numpy(patches_n))
        t_hsmr1 = time.time()
        pd_params = {
            k: v.detach().cpu().clone()
            for k, v in outputs["pd_params"].items()
        }
        pd_cam_t = outputs["pd_cam_t"].detach().cpu().clone()
        skel_out = pipeline.skel_model(
            poses=pd_params["poses"].to(device),
            betas=pd_params["betas"].to(device),
            skelmesh=False,
        )
        t_hsmr2 = time.time()
        joints_body = skel_out.joints_backup.detach().cpu().numpy()  # (N,24,3)
        joints_ori = skel_out.joints_ori.detach().cpu().numpy()      # (N,24,3,3)
    print(f"[hsmr]  ONNX={(t_hsmr1-t_hsmr0)*1000:.0f}ms  skel={(t_hsmr2-t_hsmr1)*1000:.0f}ms")

    # ── 6. 虚拟相机投影 → 逐关节处理 ──
    cx_v, cy_v = W / 2, H / 2
    f_v = 5000.0
    raw_cam_t = compute_raw_cam_t(pd_cam_t, np.asarray(bbx_cs), W, H)
    scores = det_out[0][0]["scores"]

    persons_data = []
    for i in range(len(patches)):
        joints_cam = joints_body[i] + raw_cam_t[i].numpy()  # (24,3) 虚拟相机系
        u = f_v * joints_cam[:, 0] / joints_cam[:, 2] + cx_v
        v = f_v * joints_cam[:, 1] / joints_cam[:, 2] + cy_v

        joints_list = []
        pelvis_z = None
        for j in range(24):
            # 增强深度采样
            ds = sample_depth_stats(depth, u[j], v[j], radius=5, outlier_std=3.0)

            # 骨盆深度锚定
            if j == 0 and ds["median_m"] is not None:
                pelvis_z = ds["median_m"]

            # 校验: 与骨盆深度差 > 1.5m → 采到背景/别人
            skip = False
            if j > 0 and pelvis_z is not None and ds["median_m"] is not None:
                if abs(ds["median_m"] - pelvis_z) > 1.5:
                    skip = True

            # 光学系 3D (真实深度反投影)
            if skip or ds["median_m"] is None:
                opt_pos = [np.nan, np.nan, np.nan]
                depth_valid = False
            else:
                opt_pos_arr = deproject(u[j], v[j], ds["median_m"], K)
                opt_pos = opt_pos_arr.tolist()
                depth_valid = True

            # 虚拟相机系 4×4 (JOINT_REMAP 作用在局部侧: world = R_orig @ R_remap.T @ v_new_local)
            R_orig = joints_ori[i, j]     # (3,3) SKEL 原始 (local→world)
            t_orig = joints_cam[j]        # (3,) 虚拟相机系位置
            R_remap = JOINT_REMAP[j]
            R_target = R_orig @ R_remap.T  # new_local → world
            t_target = t_orig              # 平移不变 (关节原点位置不随局部轴重标而变)
            T_4x4 = np.eye(4)
            T_4x4[:3, :3] = R_target
            T_4x4[:3, 3] = t_target

            # BASE 系位姿 (统一约定朝向; R_ot_b/t_ot_b 来自 TF optical→BASE)
            if depth_valid and base_meta is not None:
                pos_base_arr = R_ot_b @ opt_pos_arr + t_ot_b
                R_base = R_ot_b @ R_target
                position_base_m = [float(round(x, 4)) for x in pos_base_arr]
                rotation_base_3x3 = [[float(round(y, 6)) for y in row] for row in R_base.tolist()]
            else:
                position_base_m = None
                rotation_base_3x3 = None

            joints_list.append({
                "joint_index": j,
                "joint_name": SKEL_JOINTS[j],
                "pixel_u": float(round(u[j], 2)),
                "pixel_v": float(round(v[j], 2)),
                "virtual_cam_position_m": [float(round(x, 4)) for x in t_target],
                "virtual_cam_transform_4x4": [
                    [float(round(y, 6)) for y in row] for row in T_4x4.tolist()
                ],
                "rotation_skel_3x3": R_orig.tolist(),  # 原始朝向 (供坐标轴绘制)
                "depth_stats": {
                    "median_m": (
                        round(ds["median_m"], 4) if ds["median_m"] is not None else None
                    ),
                    "mean_m": (
                        round(ds["mean_m"], 4) if ds["mean_m"] is not None else None
                    ),
                    "std_m": (
                        round(ds["std_m"], 4) if ds["std_m"] is not None else None
                    ),
                    "valid_pixel_count": ds["valid_pixel_count"],
                },
                "optical_position_m": [float(round(x, 4)) for x in opt_pos],
                "position_base_m": position_base_m,
                "rotation_base_3x3": rotation_base_3x3,
                "depth_valid": depth_valid,
            })

        persons_data.append({
            "person_index": i,
            "score": float(scores[i]) if i < len(scores) else None,
            "joints": joints_list,
        })

    # ── 7. 标注图 ──
    annotated = draw_annotated_image(frame_bgr, persons_data, K)

    # ── 8. 保存 ──
    t_save0 = time.time()
    save_outputs(out_dir, frame_bgr, depth, annotated, K, H, W, persons_data, base_meta)
    t_save1 = time.time()

    t_total = t_save1 - t0
    print(f"[timing] ViTDet={(t_det1-t_det0)*1000:.0f}ms | "
          f"HSMR_ONNX={(t_hsmr1-t_hsmr0)*1000:.0f}ms | "
          f"SKEL={(t_hsmr2-t_hsmr1)*1000:.0f}ms | "
          f"crop={(t_det2-t_det1)*1000:.0f}ms | "
          f"save={(t_save1-t_save0)*1000:.0f}ms",
          flush=True)

    if _own_cap:
        cap.stop()
    return out_dir


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(_PROJECT, "deploy", "joint_service", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(
        description="HSMR 抓帧+推理+深度查询+保存 (测试用)")
    ap.add_argument("--device", default="cuda:0",
                    help="推理设备 (default: cuda:0)")
    ap.add_argument("--out-dir", default=None,
                    help="输出根目录 (default: data_outputs/test_capture)")
    ap.add_argument("--config", default=None,
                    help="config.yaml 路径 (default: deploy/joint_service/config.yaml)")
    ap.add_argument("--timeout", type=float, default=15.0,
                    help="抓帧超时秒数 (default: 15)")
    ap.add_argument("--loop", action="store_true",
                    help="持续循环推理 (模型只加载一次)")
    ap.add_argument("--loop-interval", type=float, default=0,
                    help="循环间隔秒数 (default: 0, 推理完立刻下一帧)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    import torch

    prefix = cfg["topics"]["prefix"]
    topics = [
        f"{prefix}/color/image_raw/compressed",
        f"{prefix}/aligned_depth_to_color/image_raw/compressedDepth",
        f"{prefix}/aligned_depth_to_color/camera_info",
        "/tf_static",
        "/tf",
    ]

    # ── 加载模型 (只一次) ──
    print("[load] detector + HSMR (ONNX)...", flush=True)
    from lib.modeling.pipelines.vitdet import build_detector
    detector = build_detector(
        batch_size=1, max_img_size=512, device=args.device, use_amp=True,
    )
    from deploy.onnx.runtime import HSMRONNXRuntimePipeline
    pipeline = HSMRONNXRuntimePipeline(
        model_path=cfg["model"]["onnx_model"],
        model_root=cfg["model"]["model_root"],
        device=args.device,
    )
    print(f"[load] done", flush=True)

    # ── 连接相机 ──
    print("[capture] 连接 bridge...", flush=True)
    cap = MultiTopicCapture(cfg["bridge"]["url"], topics).start()
    t_wait = time.time()
    snap = None
    while time.time() - t_wait < args.timeout:
        snap = cap.snapshot()
        if all(t in snap for t in topics):
            break
        time.sleep(0.5)
    if snap is None or not all(t in snap for t in topics):
        print("FAIL: 主题未就绪"); cap.stop(); return

    if args.loop:
        print(f"[loop] 开始循环推理 (间隔{args.loop_interval}s), Ctrl+C 停止", flush=True)
        frame_n = 0
        try:
            while True:
                t_frame = time.time()
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                base_out = args.out_dir or os.path.join(_PROJECT, "data_outputs", "test_capture")
                out_dir = os.path.join(base_out, f"loop_{timestamp}")
                run_capture_and_save(cfg, args.device, out_dir,
                                     detector=detector, pipeline=pipeline, cap=cap)
                frame_n += 1
                elapsed = time.time() - t_frame
                print(f"[{frame_n}] {elapsed*1000:.0f}ms", flush=True)
                if args.loop_interval > 0:
                    time.sleep(args.loop_interval)
        except KeyboardInterrupt:
            print("\n[loop] 停止", flush=True)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_out = args.out_dir or os.path.join(_PROJECT, "data_outputs", "test_capture")
        out_dir = os.path.join(base_out, timestamp)
        run_capture_and_save(cfg, args.device, out_dir,
                             detector=detector, pipeline=pipeline, cap=cap)

    cap.stop()


if __name__ == "__main__":
    main()
