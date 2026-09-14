"""
从一帧机器人摄像头画面 → 3D 场景渲染（理解人体 3D 位置 + 朝向）

不改任何现有代码，只复用现有函数：
  build_detector / _img_det2patches / pipeline / prepare_mesh / params_q2rot / visualize_full_img

用法:
  python docs/render_frame_3d.py --image <frame.jpg> [--device cpu|--device cuda:0]
      [--max_instances 5] [--model_root data_inputs/released_models/HSMR-ViTH-r1d1]

输出:
  docs/_frame_3d_view.png    — 3D 场景: 虚拟相机 + 人体 + 朝向 + 坐标轴
  docs/_frame_2d_overlay.png — 2D 渲染叠加 (pyrender 原样)
  docs/_frame_3d_data.json   — 3D 数据: 每个关节位置 + 根朝向(四元数)

3D 场景含义 (虚拟相机坐标系):
  相机原点 (0,0,0), 朝向 +Z
  人体根节点在 camera_translation 处 (= raw_cam_t)
  +X=身体左侧, +Y=身体上方, +Z=身体前方
  金色箭头 = 人体前方向量 (root_R @ [0,0,1])
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── 中文字体 ──
_FONT_CANDIDATES = [
    '/usr/share/fonts/fonts-gb/GB_ST_GB18030.ttf',
    '/usr/share/fonts/opentype/source-han-cjk/SourceHanSansSC-Regular.otf',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
]
_zh_ok = False
for _f in _FONT_CANDIDATES:
    if os.path.exists(_f):
        try:
            font_manager.fontManager.addfont(_f)
            _name = font_manager.FontProperties(fname=_f).get_name()
            matplotlib.rcParams['font.family'] = _name
            _zh_ok = True
            break
        except Exception:
            continue

from lib.kits.hsmr_demo import IMG_MEAN_255, IMG_STD_255, _img_det2patches, prepare_mesh, visualize_full_img
from lib.modeling.pipelines.vitdet import build_detector
from lib.body_models.skel_utils.transforms import params_q2rot


# ============================================================
# 骨骼连接 (SKEL 24 解剖关节)
# ============================================================
BONES = [
    [0,11],[11,12],[12,13],
    [0,1],[1,2],[2,3],[3,4],[4,5],
    [0,6],[6,7],[7,8],[8,9],[9,10],
    [12,14],[14,15],[15,16],[16,17],[17,18],
    [12,19],[19,20],[20,21],[21,22],[22,23],
]
JOINT_NAMES = [
    'pelvis','femur-R','tibia-R','talus-R','calcn-R','toes-R',
    'femur-L','tibia-L','talus-L','calcn-L','toes-L',
    'lumbar','thorax','head','scapula-R','humerus-R','ulna-R','radius-R','hand-R',
    'scapula-L','humerus-L','ulna-L','radius-L','hand-L',
]



def rot_to_quat(M):
    """3x3 旋转矩阵 → 四元数 (w,x,y,z)."""
    M = np.asarray(M, dtype=float)
    tr = M[0,0]+M[1,1]+M[2,2]
    if tr > 0:
        s = np.sqrt(tr+1.0)*2
        q = [(0.25*s), (M[2,1]-M[1,2])/s, (M[0,2]-M[2,0])/s, (M[1,0]-M[0,1])/s]
    else:
        i = int(np.argmax([M[0,0],M[1,1],M[2,2]]))
        if i==0:
            s = np.sqrt(1.0+M[0,0]-M[1,1]-M[2,2])*2
            q = [(M[2,1]-M[1,2])/s, 0.25*s, (M[0,1]+M[1,0])/s, (M[0,2]+M[2,0])/s]
        elif i==1:
            s = np.sqrt(1.0+M[1,1]-M[0,0]-M[2,2])*2
            q = [(M[0,2]-M[2,0])/s, (M[0,1]+M[1,0])/s, 0.25*s, (M[1,2]+M[2,1])/s]
        else:
            s = np.sqrt(1.0+M[2,2]-M[0,0]-M[1,1])*2
            q = [(M[1,0]-M[0,1])/s, (M[0,2]+M[2,0])/s, (M[1,2]+M[2,1])/s, 0.25*s]
    q = np.array(q)/np.linalg.norm(q)
    return q.tolist()


def render_3d_scene(cam_t, joints_body, root_rot, img_path, out_png):
    """渲染 3D 场景: 虚拟相机 + 人体 + 朝向."""
    # 人体关节在相机坐标系
    joints_cam = joints_body + cam_t[None, :]

    # 前方向量 (SKEL 人体局部 +Z)
    fwd_local = np.array([0.0, 0.0, 1.0])
    fwd_world = (root_rot @ fwd_local).astype(float)
    quat = rot_to_quat(root_rot)

    fig = plt.figure(figsize=(21, 7))

    # ---------- 视角1: 3/4 透视图 ----------
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_title('3/4 透视图\n相机原点(0,0,0) 朝+Z | 人体在 camera_translation 处', fontsize=11, fontweight='bold')

    # 相机
    ax.scatter([0], [0], [0], s=200, c='#a371f7', marker='s', edgecolors='white', zorder=10)
    ax.text(0.05, 0.05, 0.05, '相机原点', fontsize=9, color='#a371f7')

    # 相机坐标轴
    axlen = 0.5
    _axis3d(ax, [0, 0, 0], axlen)

    # 人体骨架 (相机坐标系)
    for p, c in BONES:
        a = joints_cam[p]; b = joints_cam[c]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], '-', color='#f78166', lw=2)
    ax.scatter(joints_cam[:, 0], joints_cam[:, 1], joints_cam[:, 2],
               c='#f78166', s=30, edgecolors='#c04030', linewidths=0.5, zorder=5)

    # 人体前方向量 (金色)
    root_pos = joints_cam[0]
    ax.quiver(root_pos[0], root_pos[1], root_pos[2],
              fwd_world[0] * 0.8, fwd_world[1] * 0.8, fwd_world[2] * 0.8,
              color='#ffd166', linewidth=3, arrow_length_ratio=0.2)
    ax.text(root_pos[0] + fwd_world[0]*0.9, root_pos[1] + fwd_world[1]*0.9, root_pos[2] + fwd_world[2]*0.9,
            f'前向={np.round(fwd_world,2).tolist()}', fontsize=8, color='#ffd166', fontweight='bold')

    # 相机→人体连线
    ax.plot([0, root_pos[0]], [0, root_pos[1]], [0, root_pos[2]],
            '--', color='#d2991d', alpha=0.6)
    ax.text(root_pos[0]/2, root_pos[1]/2, root_pos[2]/2,
            f'camera_translation=[{cam_t[0]:.2f},{cam_t[1]:.2f},{cam_t[2]:.0f}]',
            fontsize=8, color='#d2991d')

    ax.set_xlabel('X (右→)'); ax.set_ylabel('Y (上→)'); ax.set_zlabel('Z (前→)')
    ax.view_init(elev=20, azim=-60)

    # ---------- 视角2: 侧面 ----------
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.set_title('侧面视图 (Y-Z 平面)\n看人体相对相机的距离和高度', fontsize=11, fontweight='bold')
    ax2.scatter([0], [0], [0], s=200, c='#a371f7', marker='s', edgecolors='white', zorder=10)
    _axis3d(ax2, [0, 0, 0], axlen)
    for p, c in BONES:
        a = joints_cam[p]; b = joints_cam[c]
        ax2.plot([a[1], b[1]], [a[2], b[2]], color='#f78166', lw=2)
    ax2.scatter(joints_cam[:, 1], joints_cam[:, 2], joints_cam[:, 0]*0, c='#f78166', s=30, zorder=5)
    # 人体高度范围
    y_top = joints_cam[:, 1].max(); y_bot = joints_cam[:, 1].min()
    ax2.plot([y_top, y_bot], [cam_t[2], cam_t[2]], 'k--', alpha=0.5)
    ax2.annotate(f'身高 {y_top - y_bot:.2f}', xy=(y_top, cam_t[2]))
    ax2.plot([cam_t[1], cam_t[1]], [0, cam_t[2]], '--', color='#d2991d', alpha=0.5)
    ax2.text(cam_t[1], cam_t[2]/2, 0, f'距离={cam_t[2]:.0f}', fontsize=8, color='#d2991d')
    ax2.set_xlabel('Y (上→)'); ax2.set_ylabel('Z (相机前方→)')
    ax2.view_init(elev=0, azim=-90)

    # ---------- 视角3: 俯视图 ----------
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_title('俯视图 (X-Z 平面)\n看人体左右位置和朝向', fontsize=11, fontweight='bold')
    ax3.scatter([0], [0], [0], s=200, c='#a371f7', marker='s', edgecolors='white', zorder=10)
    _axis3d(ax3, [0, 0, 0], axlen)
    for p, c in BONES:
        a = joints_cam[p]; b = joints_cam[c]
        ax3.plot([a[0], b[0]], [a[2], b[2]], color='#f78166', lw=2)
    ax3.scatter(joints_cam[:, 0], joints_cam[:, 2], c='#f78166', s=30, zorder=5)
    # 前方向量
    ax3.quiver(root_pos[0], root_pos[2], 0,
               fwd_world[0]*0.8, fwd_world[2]*0.8, 0,
               color='#ffd166', linewidth=3, arrow_length_ratio=0.2)
    ax3.text(root_pos[0]+fwd_world[0]*0.9, root_pos[2]+fwd_world[2]*0.9, 0,
             f'前向={np.round(fwd_world,2).tolist()}', fontsize=8, color='#ffd166', fontweight='bold')
    ax3.plot([0, root_pos[0]], [0, root_pos[2]], '--', color='#d2991d', alpha=0.6)
    ax3.set_xlabel('X (右→)'); ax3.set_ylabel('Z (前→)')
    ax3.view_init(elev=90, azim=-90)

    fig.suptitle(
        f'HSMR 3D 场景 | 来源: {os.path.basename(img_path)}\n'
        f'相机坐标系: 原点在相机, +Z=前方 | 人体根位置(相机坐标)={cam_t}\n'
        f'人体朝向: 前向={np.round(fwd_world, 3).tolist()}, 四元数={[round(x,3) for x in quat]}\n'
        f'{"" if _zh_ok else "[中文字体缺失, 显示英文]"}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(out_png, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    return fwd_world, quat


def _axis3d(ax, origin, length):
    o = np.array(origin)
    for d, c in [([1, 0, 0], 'red'), ([0, 1, 0], 'green'), ([0, 0, 1], 'blue')]:
        ax.quiver(o[0], o[1], o[2], d[0]*length, d[1]*length, d[2]*length,
                  color=c, linewidth=2, arrow_length_ratio=0.15)
    ax.text(o[0]+length*1.1, o[1], o[2], 'X', color='red', fontsize=10, fontweight='bold')
    ax.text(o[0], o[1]+length*1.1, o[2], 'Y', color='green', fontsize=10, fontweight='bold')
    ax.text(o[0], o[1], o[2]+length*1.1, 'Z', color='blue', fontsize=10, fontweight='bold')


def main():
    ap = argparse.ArgumentParser(description='一帧画面 → 3D 场景渲染')
    ap.add_argument('--image', required=True, help='输入画面 (jpg/png)')
    ap.add_argument('--device', default='cpu', help='cpu 或 cuda:0')
    ap.add_argument('--backend', default='pytorch', choices=['pytorch', 'onnx'],
                    help='pytorch 用完整 checkpoint; onnx 用导出的 onnx 模型')
    ap.add_argument('--onnx_model', default='deploy/onnx/artifacts/hsmr_full_fp16.onnx')
    ap.add_argument('--max_instances', type=int, default=5)
    ap.add_argument('--model_root', default='data_inputs/released_models/HSMR-ViTH-r1d1')
    ap.add_argument('--out_prefix', default='docs/_frame')
    args = ap.parse_args()

    img_path = args.image
    frame_bgr = cv2.imread(img_path)
    if frame_bgr is None:
        print(f'[错误] 无法读取图片: {img_path}')
        sys.exit(1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    print(f'[信息] 图片: {img_path} shape={frame_rgb.shape} device={args.device}')

    # ── 1. 加载模型 (一次) ──
    t0 = time.time()
    detector = build_detector(batch_size=1, max_img_size=512, device=args.device, use_amp=False)
    if args.backend == 'onnx':
        from deploy.onnx.runtime import HSMRONNXRuntimePipeline
        pipeline = HSMRONNXRuntimePipeline(model_path=args.onnx_model,
                                           model_root=args.model_root, device=args.device)
    else:
        from lib.modeling.pipelines.hsmr import build_inference_pipeline
        pipeline = build_inference_pipeline(model_root=args.model_root, device=args.device)
    print(f'[信息] 模型加载耗时 {time.time()-t0:.1f}s, backend={args.backend}')

    # ── 2. 检测 + 裁剪 ──
    t1 = time.time()
    detector_outputs = detector([frame_rgb])
    patches, bbx_cs = _img_det2patches(
        frame_rgb, detector_outputs[0][0], detector_outputs[1][0], args.max_instances)
    print(f'[信息] 检测+裁剪耗时 {time.time()-t1:.1f}s, 检测到 {len(patches)} 人')

    if len(patches) == 0:
        print('[信息] 无人, 退出')
        sys.exit(0)

    # ── 3. HSMR 推理 ──
    patches = patches.astype(np.float32)
    patches_normalized = (patches - IMG_MEAN_255) / IMG_STD_255
    patches_normalized = np.ascontiguousarray(patches_normalized.transpose(0, 3, 1, 2))
    import torch
    if args.backend == 'onnx':
        outputs = pipeline(patches_normalized)
    else:
        with torch.no_grad():
            outputs = pipeline(torch.from_numpy(patches_normalized))
    pd_params = {k: v.detach().cpu().clone() if hasattr(v, 'detach') else v
                 for k, v in outputs['pd_params'].items()}
    pd_cam_t = outputs['pd_cam_t'].detach().cpu().clone() if hasattr(outputs['pd_cam_t'], 'detach') else outputs['pd_cam_t']
    print(f'[信息] HSMR 完成, poses={pd_params["poses"].shape}, pd_cam_t={pd_cam_t.shape}')

    # ── 4. 网格 + 3D 数据 ──
    m_skin, m_skel = prepare_mesh(pipeline, pd_params, include_skeleton=True, batch_size=1)

    # 根相对 3D 关节 (body 坐标系)
    skel_out = pipeline.skel_model(poses=pd_params['poses'].to(args.device),
                                   betas=pd_params['betas'].to(args.device),
                                   skelmesh=False)
    joints_body = skel_out.joints.detach().cpu().numpy()  # [N,44,3]

    # 根朝向
    rotations = params_q2rot(pd_params['poses'])  # [N,24,3,3]
    root_rot = rotations[0, 0].numpy()            # [3,3] 根关节旋转矩阵

    # ── 5. full-image 相机平移 (渲染用) ──
    det_meta = {'n_patch_per_img': [len(patches)], 'bbx_cs_per_img': [bbx_cs],
                'bbx_cs': np.asarray(bbx_cs)}
    rendered, raw_cam_t = visualize_full_img(pd_cam_t, [frame_rgb], det_meta, m_skin, m_skel)

    # ── 6. 2D 叠加 + 3D 渲染 ──
    overlay = rendered[0]
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{args.out_prefix}_2d_overlay.png', overlay_bgr)
    print(f'[输出] 2D 渲染叠加: {args.out_prefix}_2d_overlay.png')

    # 每个检测到的人
    person_data = []
    for i in range(len(patches)):
        cam_t = raw_cam_t[i]              # 该人的相机平移
        jb = joints_body[i]               # 该人根相对关节
        rr = rotations[i, 0].numpy()      # 该人根旋转 [3,3]
        fwd, quat = render_3d_scene(cam_t, jb, rr, img_path, f'{args.out_prefix}_person{i}_3d_view.png')
        person_data.append({
            'person': i,
            'bbox_cs': bbx_cs[i].tolist() if isinstance(bbx_cs[i], np.ndarray) else bbx_cs[i],
            'camera_translation': cam_t.tolist(),
            'root_rotation': rr.tolist(),
            'forward_vector': fwd.tolist(),
            'quaternion_wxyz': quat,
            'joints_44_body': jb.tolist(),
        })
        print(f'[输出] Person{i}: camera_translation={cam_t}, 四元数={[round(x,3) for x in quat]}, 前向={np.round(fwd,3)}')

    with open(f'{args.out_prefix}_3d_data.json', 'w', encoding='utf-8') as f:
        json.dump(person_data, f, indent=2, ensure_ascii=False)
    print(f'[输出] 3D 数据: {args.out_prefix}_3d_data.json')


if __name__ == '__main__':
    main()
