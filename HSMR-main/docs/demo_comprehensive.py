"""
综合渲染演示 — 使用机器人 latest_3d.json 的真实 SKEL 关节姿态

用机器人已产出的真实人体姿态(44关节), 放到合理真实深度位置,
投影到彩色图, 合成深度图, 渲染 4 面板综合图.
这样即使当前镜头前没人, 也能立刻看到综合渲染的完整结构.

用法:
  python docs/demo_comprehensive.py [--data docs/_robot_latest_3d.json] [--color docs/_robot_latest.jpg]
"""
import argparse
import json
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_depth_comprehensive import render_comprehensive, JOINT_NAMES_44, BODY25_BONES

# 真实相机内参 (从 bridge 实测)
K = np.array([[910.677, 0, 653.789], [0, 910.280, 374.077], [0, 0, 1]], dtype=float)
H, W = 720, 1280


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='docs/_robot_latest_3d.json')
    ap.add_argument('--color', default='docs/_robot_latest.jpg')
    ap.add_argument('--out', default='docs/_depth_comprehensive_demo.png')
    ap.add_argument('--label', default='演示: 真实SKEL姿态 @ 模拟深度')
    args = ap.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    color = cv2.imread(args.color)
    if color is None:
        color = np.ones((H, W, 3), dtype=np.uint8) * 60
    else:
        color = cv2.resize(color, (W, H))

    # 每个检测到的人: 取真实 SKEL 根相对关节 (model_origin_relative_m)
    # 放到不同真实深度位置 (演示用: 2.0m, 3.5m)
    people = []
    for idx, p in enumerate(data.get('persons', [])):
        joints_rel = np.asarray(p['joints']['hsmr_44']['model_origin_relative_m'], dtype=float)  # [44,3]
        # 放置到光学系: 骨盆(mid_hip=8) 放到指定深度
        depth_place = [2.0, 3.5][idx % 2]
        # 骨盆在根相对坐标的位置
        pelvis_rel = joints_rel[8]
        # 要把整个人放到相机前方 depth_place 米处
        # 光学系: +Y下, +Z远离. 人体根相对是 +Y上. 需要翻转 Y
        # HSMR 模型坐标 +Y 下 (虚拟相机), 与光学系一致
        j_opt = joints_rel.copy()
        # 使骨盆在 (x_offset, y_offset, depth_place)
        x_off = [-0.35, 0.45][idx % 2]
        y_off = -0.1
        j_opt[:, 0] += (x_off - pelvis_rel[0])
        j_opt[:, 1] += (y_off - pelvis_rel[1])
        j_opt[:, 2] += (depth_place - pelvis_rel[2])

        # 投影到彩色像素 (真实内参)
        valid = j_opt[:, 2] > 0.1
        j2d = np.full((44, 2), np.nan)
        j2d[valid, 0] = K[0,0] * j_opt[valid, 0] / j_opt[valid, 2] + K[0,2]
        j2d[valid, 1] = K[1,1] * j_opt[valid, 1] / j_opt[valid, 2] + K[1,2]

        pelvis_opt = j_opt[8] if valid[8] else None
        orientation = p.get('orientation', {})
        people.append({
            'person_index': idx,
            'score': float(p.get('detection', {}).get('score', 0.99)),
            'joints_2d_color': j2d.tolist(),
            'joints_optical_m': j_opt.tolist(),
            'depth_valid': valid.tolist(),
            'pelvis_optical_m': pelvis_opt.tolist() if pelvis_opt is not None else None,
            'pelvis_depth_m': float(np.linalg.norm(pelvis_opt)) if pelvis_opt is not None else None,
            'orientation': {
                'rotation_matrix': orientation.get('rotation_matrix_body_canonical_to_model_camera'),
                'quaternion_wxyz': orientation.get('quaternion_body_canonical_to_model_camera_wxyz'),
                'forward_vector': orientation.get('body_forward_unit_model_camera'),
            },
        })

    # 合成深度图: 在每个关节处画深度值, 形成人体深度
    depth = np.zeros((H, W), dtype=np.uint16)
    for person in people:
        j2d = np.asarray(person['joints_2d_color'])
        valid = np.isfinite(j2d).all(axis=1)
        for j in range(44):
            if not valid[j]: continue
            u, v = int(round(j2d[j,0])), int(round(j2d[j,1]))
            if 0 <= u < W and 0 <= v < H:
                z = person['joints_optical_m'][j][2]
                if z > 0.1:
                    depth[max(0,v-8):min(H,v+8), max(0,u-8):min(W,u+8)] = int(z*1000)
    # 平滑 + 背景
    depth = cv2.GaussianBlur(depth, (11, 11), 0)
    bg_z = 4000  # 背景 4m
    depth[depth == 0] = bg_z

    render_comprehensive(color, depth, K, people, args.out, args.label)
    print(f'✅ 综合渲染演示完成: {args.out}')
    print(f'   共 {len(people)} 人:')
    for p in people:
        q = p['orientation'].get('quaternion_wxyz') or [0,0,0,0]
        print(f'   P{p["person_index"]}: 骨盆深度={p["pelvis_depth_m"]:.2f}m, '
              f'四元数(wxyz)={[round(x,4) for x in q]}')


if __name__ == '__main__':
    main()
