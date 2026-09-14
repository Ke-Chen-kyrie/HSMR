"""SKEL T-pose 探针: 以 poses=0 跑 skel_model, 打印每块骨骼基准朝向 (joints_ori).

用途: 确定 SKEL 骨骼局部坐标系约定 (哪个局部轴=前/上/左), 用于修订 JOINT_REMAP.
在远程项目根目录运行: .venv_orin/bin/python tests/probe_tpose.py
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, "deploy", "joint_service"))
sys.path.insert(0, os.path.join(_PROJECT, "thirdparty", "SKEL"))

import numpy as np
import torch

def main():
    cfg_path = os.path.join(_PROJECT, "deploy", "joint_service", "config.yaml")
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_root = os.path.join(_PROJECT, cfg["model"]["model_root"])
    onnx_path = os.path.join(_PROJECT, cfg["model"]["onnx_model"])
    device = "cuda:0"

    print(f"[load] pipeline model_root={model_root}", flush=True)
    from deploy.onnx.runtime import HSMRONNXRuntimePipeline
    pipeline = HSMRONNXRuntimePipeline(model_path=onnx_path, model_root=model_root, device=device)

    skel = pipeline.skel_model
    print(f"[skel] type={type(skel).__name__}")
    # 需要知道 pose 维度
    import inspect
    print(f"[skel] forward sig: {inspect.signature(skel.forward)}")

    B = 1
    poses = torch.zeros(B, 46, device=device)   # SKEL 46 维姿态
    betas = torch.zeros(B, 10, device=device)
    out = skel(poses=poses, betas=betas, skelmesh=False)
    J = out.joints_backup.detach().cpu().numpy()        # (B,24,3) 关节位置 (模型系)
    R = out.joints_ori.detach().cpu().numpy()           # (B,24,3,3) 骨骼朝向 (模型系)

    SKEL_JOINTS = ["pelvis","femur_r","tibia_r","talus_r","calcn_r","toes_r",
        "femur_l","tibia_l","talus_l","calcn_l","toes_l","lumbar_body","thorax","head",
        "scapula_r","humerus_r","ulna_r","radius_r","hand_r",
        "scapula_l","humerus_l","ulna_l","radius_l","hand_l"]

    j = J[0]
    print("\n=== T-pose 关节位置 (模型系) ===")
    print(f"pelvis: {j[0].round(3)}  head: {j[13].round(3)}  thorax: {j[12].round(3)}")
    print(f"hand_r: {j[18].round(3)}  hand_l: {j[23].round(3)}  tibia_r(膝): {j[2].round(3)}  talus_r(踝): {j[3].round(3)}")

    print("\n=== T-pose 骨骼局部轴在模型系的方向 (joints_ori 列) ===")
    print("格式: 每行 X/Y/Z 局部轴在世界(模型)系的三分量")
    for i, name in enumerate(SKEL_JOINTS):
        M = R[0, i]
        cx, cy, cz = M[:, 0], M[:, 1], M[:, 2]
        print(f"{name:>12}: X={cx.round(3)} Y={cy.round(3)} Z={cz.round(3)}")

    print("\n=== T-pose 骨骼朝向分析 ===")
    up = j[13] - j[0]                     # 骨盆→头
    up = up / np.linalg.norm(up)
    print(f"T-pose 骨盆→头方向(模型系 up): {up.round(3)}")
    for i in (0, 12, 18, 2):
        M = R[0, i]
        for ax, col in (("X", M[:,0]), ("Y", M[:,1]), ("Z", M[:,2])):
            print(f"{SKEL_JOINTS[i]:>12} 局部{ax}·up = {float(col @ up):+.3f}")

if __name__ == "__main__":
    main()
