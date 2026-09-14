"""
综合 4 面板渲染: 彩色图 | 深度图 | 3D真实场景 | 数据表
把所有关节位置 + orientation + 深度图信息一起渲染, 帮助理解.

用法 (被 realtime_depth_3d.py 调用, 也可独立测试):
  python docs/render_depth_comprehensive.py --color <color.png> --depth <depth.png> \
      --data <depth3d_data.json> --out <out.png>
"""
import argparse
import json
import os

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── 中文字体 (本地 GB + 机器人 Noto CJK 双候选) ──
_zh_ok = False
for _f in [
    '/usr/share/fonts/fonts-gb/GB_ST_GB18030.ttf',               # 本地
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',    # 机器人
    '/usr/share/fonts/opentype/source-han-cjk/SourceHanSansSC-Regular.otf',
    '/usr/share/fonts/opentype/source-han-cjk/SourceHanSansCN-Regular.otf',
]:
    if os.path.exists(_f):
        try:
            font_manager.fontManager.addfont(_f)
            _name = font_manager.FontProperties(fname=_f).get_name()
            matplotlib.rcParams['font.sans-serif'] = [_name, 'DejaVu Sans']
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['axes.unicode_minus'] = False
            _zh_ok = True
            break
        except Exception:
            pass
if not _zh_ok:
    for _n in ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'AR PL UMing CN']:
        try:
            matplotlib.rcParams['font.sans-serif'] = [_n, 'DejaVu Sans']
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['axes.unicode_minus'] = False
            _zh_ok = True
            break
        except Exception:
            pass
# CJK 等宽字体 (数据表对齐 + 中文)
for _mf in [
    '/usr/share/fonts/opentype/source-han-cjk/SourceHanMono.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansMonoCJKsc-Regular.otf',
]:
    if os.path.exists(_mf):
        try:
            font_manager.fontManager.addfont(_mf)
            matplotlib.rcParams['font.monospace'] = [font_manager.FontProperties(fname=_mf).get_name(), 'DejaVu Sans Mono']
            break
        except Exception:
            pass

# ── HSMR 44 关节名 (OpenPose BODY_25 + SMPL_to_J19) ──
JOINT_NAMES_44 = [
    # OpenPose BODY_25 (0-24)
    'nose','neck','R_shoulder','R_elbow','R_wrist','L_shoulder','L_elbow','L_wrist',
    'mid_hip','R_hip','R_knee','R_ankle','L_hip','L_knee','L_ankle',
    'R_eye','L_eye','R_ear','L_ear','L_big_toe','L_small_toe','L_heel',
    'R_big_toe','R_small_toe','R_heel',
    # SMPL_to_J19 (25-43)
    'R_ankle_s','R_knee_s','R_hip_s','L_hip_s','L_knee_s','L_ankle_s',
    'R_wrist_s','R_elbow_s','R_shoulder_s','L_shoulder_s','L_elbow_s','L_wrist_s',
    'neck_lsp','top_head_lsp','pelvis_mpii','thorax_mpii','spine_h36m','jaw_h36m','head_h36m',
]

# OpenPose BODY_25 骨架 (只画前25点的主骨架)
BODY25_BONES = [
    [1,8],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],
    [8,9],[9,10],[10,11],[8,12],[12,13],[13,14],
    [1,0],[0,15],[15,17],[0,16],[16,18],
    [14,21],[19,21],[20,21],[11,24],[22,24],[23,24],
]


def render_comprehensive(color_bgr, depth_mm, K, people, out_png, frame_label=''):
    """4 面板综合渲染.
    color_bgr: [H,W,3] uint8
    depth_mm: [H,W] uint16 毫米
    K: [3,3] 相机内参
    people: list of dicts, 每项含:
        person_index, score,
        joints_2d_color: [44,2] 像素,
        joints_optical_m: [44,3] 真实光学系坐标,
        depth_valid: [44] bool,
        pelvis_optical_m: [3],
        pelvis_depth_m: float,
        orientation: {rotation_matrix, quaternion_wxyz, body_forward, body_up, body_left},
        bbox_cs
    """
    H, W = depth_mm.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.18)

    colors = ['#f78166', '#58a6ff', '#3fb950', '#d2991d', '#a371f7']

    # ═══════════ 面板1: 彩色图 + 2D关节 ═══════════
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('① 彩色图 + HSMR 44 关节 (2D投影)', fontsize=12, fontweight='bold')
    ax1.imshow(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_xticks([]); ax1.set_yticks([])
    for i, person in enumerate(people):
        col = colors[i % len(colors)]
        j2d = np.asarray(person.get('joints_2d_color'), dtype=float) if person.get('joints_2d_color') else None
        if j2d is None: continue
        valid = np.isfinite(j2d).all(axis=1)
        ax1.plot(j2d[valid,0], j2d[valid,1], '.', color=col, markersize=3)
        # bbox
        bbox = person.get('bbox_cs')
        if bbox:
            cxs, cys, s = bbox
            ax1.add_patch(plt.Rectangle((cxs-s/2, cys-s/2), s, s,
                fill=False, edgecolor=col, linewidth=1.5))
        # 标关键关节号
        for j in [0, 8, 9, 12, 2, 5, 13, 14]:
            if valid[j]:
                ax1.annotate(str(j), (j2d[j,0], j2d[j,1]), fontsize=6, color=col,
                             fontweight='bold', textcoords='offset points', xytext=(3,3))
        ax1.set_title(f'① 彩色图 + 44关节 | P{i} score={person.get("score",0):.2f}', fontsize=11, fontweight='bold')

    # ═══════════ 面板2: 深度图 + 关节深度 ═══════════
    ax2 = fig.add_subplot(gs[0, 1])
    depth_vis = depth_mm.copy().astype(np.float32)
    depth_vis[depth_vis <= 0] = np.nan
    depth_vis = depth_vis / 1000.0  # m
    ax2.imshow(depth_vis, cmap='turbo', vmin=0, vmax=min(6.0, np.nanpercentile(depth_vis, 95)))
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title('② 深度图 (真实毫米→米) + 关节深度值', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(ax2.images[0], ax=ax2, fraction=0.04, pad=0.02)
    cbar.set_label('深度 (m)')
    for i, person in enumerate(people):
        col = colors[i % len(colors)]
        j2d = np.asarray(person.get('joints_2d_color'), dtype=float) if person.get('joints_2d_color') else None
        if j2d is None: continue
        valid = np.isfinite(j2d).all(axis=1)
        # 画关节
        ax2.plot(j2d[valid,0], j2d[valid,1], 'o', color=col, markersize=3, alpha=0.8)
        # 标注部分关键关节的深度值
        for j in [8, 9, 10, 11, 12, 13, 14, 0, 2, 5]:
            if not valid[j]: continue
            u, v = j2d[j]
            # 从原始深度采样 (越界判无效)
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < W and 0 <= vi < H):
                zm = np.nan
            else:
                zmm = depth_mm[vi, ui]
                zm = zmm/1000.0 if zmm > 0 else np.nan
            lbl = f'{JOINT_NAMES_44[j]}\n{zm:.2f}m' if np.isfinite(zm) else f'{JOINT_NAMES_44[j]}\n无效'
            ax2.annotate(lbl, (u, v), fontsize=6.5, color=col, fontweight='bold',
                         textcoords='offset points', xytext=(3, 3),
                         bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.55, ec='none'))

    # ═══════════ 面板3: 3D 真实场景 ═══════════
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    ax3.set_title('③ 3D 真实场景 (相机光学系, 米) + 所有关节 + 朝向', fontsize=12, fontweight='bold')
    # 相机
    ax3.scatter([0],[0],[0], s=300, c='#a371f7', marker='s', edgecolors='white', zorder=10)
    ax3.text(0.02, 0.02, 0.02, '相机', fontsize=9, color='#a371f7', fontweight='bold')
    for d, c, n in [([1,0,0],'red','X'), ([0,1,0],'green','Y↓'), ([0,0,1],'blue','Z→')]:
        ax3.quiver(0,0,0, d[0],d[1],d[2], color=c, linewidth=2, arrow_length_ratio=0.2)
        ax3.text(d[0]*1.15, d[1]*1.15, d[2]*1.15, n, color=c, fontsize=10, fontweight='bold')

    allpts = []
    for i, person in enumerate(people):
        col = colors[i % len(colors)]
        jopt = np.asarray(person.get('joints_optical_m'), dtype=float) if person.get('joints_optical_m') else None
        if jopt is None: continue
        finite = np.isfinite(jopt).all(axis=1)
        allpts.append(jopt[finite])
        # 骨骼
        for a, b in BODY25_BONES:
            if a < 25 and b < 25 and finite[a] and finite[b]:
                ax3.plot([jopt[a,0],jopt[b,0]], [jopt[a,1],jopt[b,1]], [jopt[a,2],jopt[b,2]],
                         '-', color=col, lw=2, alpha=0.8)
        # 所有关节点
        ax3.scatter(jopt[finite,0], jopt[finite,1], jopt[finite,2], c=col, s=22, zorder=5)
        # 关键关节标号
        for j in [0, 8, 9, 10, 11, 12, 13, 14, 2, 5, 17, 22]:
            if j < len(jopt) and finite[j]:
                ax3.text(jopt[j,0], jopt[j,1], jopt[j,2], f' {j}:{JOINT_NAMES_44[j]}',
                         fontsize=6, color=col)
        # 骨盆 + 距离
        pelvis = person.get('pelvis_optical_m')
        if pelvis and np.isfinite(pelvis).all():
            ax3.scatter([pelvis[0]],[pelvis[1]],[pelvis[2]], s=200, c=col, marker='*', edgecolors='white', zorder=8)
            ax3.plot([0,pelvis[0]],[0,pelvis[1]],[0,pelvis[2]], '--', color=col, alpha=0.5)
            d = np.linalg.norm(pelvis)
            ax3.text(pelvis[0], pelvis[1], pelvis[2], f' P{i} {d:.2f}m', fontsize=9, color=col)
        # orientation: 前向箭头
        ori = person.get('orientation') or {}
        fwd = ori.get('forward_vector')
        if fwd and pelvis and np.isfinite(pelvis).all():
            fwd = np.asarray(fwd)
            # 箭头从骨盆沿前向画 (0.6m)
            ax3.quiver(pelvis[0], pelvis[1], pelvis[2],
                       fwd[0]*0.6, fwd[1]*0.6, fwd[2]*0.6,
                       color='#ffd166', linewidth=3, arrow_length_ratio=0.2)
            quat = ori.get('quaternion_wxyz')
            qs = f'q=[{"".join(f"{x:+.2f}" for x in quat[:4])}]' if quat else 'q=[]'
            ax3.text(pelvis[0]+fwd[0]*0.7, pelvis[1]+fwd[1]*0.7, pelvis[2]+fwd[2]*0.7,
                     f' 前向={np.round(fwd,2).tolist()}\n {qs}', fontsize=7, color='#ffd166', fontweight='bold')
    if allpts:
        fin = np.vstack(allpts)
        m = np.percentile(fin, 5, axis=0); M = np.percentile(fin, 95, axis=0)
        ax3.set_xlim(m[0]-0.3, M[0]+0.3); ax3.set_ylim(m[1]-0.3, M[1]+0.3); ax3.set_zlim(m[2]-0.3, M[2]+0.3)
    ax3.set_xlabel('X (右→)'); ax3.set_ylabel('Y (下→)'); ax3.set_zlabel('Z (远离→)')
    ax3.view_init(elev=15, azim=-70)

    # ═══════════ 面板4: 数据表 ═══════════
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    lines = [f'════ 数据表 | {frame_label} ════',
             f'相机内参: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  (aligned depth=color)', '']
    for i, person in enumerate(people):
        j2d = np.asarray(person.get('joints_2d_color'), dtype=float) if person.get('joints_2d_color') else None
        jopt = np.asarray(person.get('joints_optical_m'), dtype=float) if person.get('joints_optical_m') else None
        ori = person.get('orientation') or {}
        lines += [f'【Person {i}】score={person.get("score",0):.2f}',
                  f'  骨盆: 深度={person.get("pelvis_depth_m",0):.2f}m, '
                  f'3D={None if person.get("pelvis_optical_m") is None else [round(x,2) for x in person["pelvis_optical_m"]]}',
                  f'  orientation:']
        if ori.get('rotation_matrix'):
            for row in ori['rotation_matrix']:
                lines.append(f'    [{", ".join(f"{x:+.2f}" for x in row)}]')
        q = ori.get('quaternion_wxyz')
        lines += [f'  四元数(wxyz): {None if not q else [round(x,4) for x in q]}',
                  f'  前向={[round(x,3) for x in ori.get("forward_vector", ori.get("body_forward",[0,0,0]))]}',
                  '']
        # 关节表头
        lines += [f'  关节     像素(u,v)    深度(m)    3D光学坐标(X,Y,Z)']
        for j in range(44):
            px = j2d[j] if (j2d is not None and np.isfinite(j2d[j]).all()) else None
            po = jopt[j] if (jopt is not None and np.isfinite(jopt[j]).all()) else None
            zm = None
            if px is not None:
                u, v = int(round(px[0])), int(round(px[1]))
                if 0 <= u < W and 0 <= v < H:
                    zmm = depth_mm[v, u]
                    zm = zmm/1000.0 if zmm > 0 else None
            name = JOINT_NAMES_44[j]
            pxs = f'({px[0]:.0f},{px[1]:.0f})' if px is not None else '( - )'
            zs = f'{zm:.2f}' if zm else '  -  '
            po_s = f'[{po[0]:+.2f},{po[1]:+.2f},{po[2]:+.2f}]' if po is not None else '( - )'
            lines.append(f'  {j:2d} {name:15s} {pxs:12s} {zs:8s} {po_s}')
        lines.append('')
    lines += ['════ 说明 ════',
              '• 3D坐标为真实深度反投影(相机光学系, 米)',
              '• 深度来自 aligned_depth 传感器, 非HSMR单目估计',
              '• orientation = 完整旋转矩阵(3x3)+四元数+前向向量',
              '• 金色箭头=人体前向(相机系), 完整姿态看旋转矩阵',
              '• +X右 +Y下 +Z远离相机',
              ]
    ax4.text(0.01, 0.995, '\n'.join(lines), transform=ax4.transAxes,
             va='top', ha='left', fontsize=7.5, linespacing=1.35)

    fig.suptitle(f'HSMR 关节 + Orientation + 真实深度 综合可视化 | {frame_label}',
                 fontsize=14, fontweight='bold')
    plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--color')
    ap.add_argument('--depth')
    ap.add_argument('--data')
    ap.add_argument('--out', default='docs/_depth_comprehensive.png')
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    color = cv2.imread(args.color)
    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        # 可能深度存的是米 float
        pass
    with open(args.data) as f:
        data = json.load(f)

    # K 从 data 或默认
    K = np.eye(3)
    intr = data.get('camera_intrinsics', {})
    if intr:
        K[0,0] = intr.get('fx', 910.68); K[1,1] = intr.get('fy', 910.28)
        K[0,2] = intr.get('cx', 653.79); K[1,2] = intr.get('cy', 374.08)

    # 补 joints_2d_color 若缺失 (从光学坐标反投影回像素)
    people = data.get('persons', [])
    for person in people:
        if not person.get('joints_2d_color') and person.get('joints_optical_m'):
            jo = np.asarray(person['joints_optical_m'], dtype=float)
            j2d = np.full((len(jo), 2), np.nan)
            valid = np.isfinite(jo).all(axis=1) & (jo[:,2] > 0.05)
            j2d[valid,0] = K[0,0]*jo[valid,0]/jo[valid,2] + K[0,2]
            j2d[valid,1] = K[1,1]*jo[valid,1]/jo[valid,2] + K[1,2]
            person['joints_2d_color'] = j2d.tolist()

    render_comprehensive(color, depth, K, people, args.out, args.label)
    print(f'✅ 综合渲染完成: {args.out}')


if __name__ == '__main__':
    main()
