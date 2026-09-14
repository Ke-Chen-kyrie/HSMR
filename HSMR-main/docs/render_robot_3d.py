"""
机器人摄像头预测 → 3D 场景渲染

读取机器人 (192.168.217.100) 上 run_webcam 产出的 latest_3d.json,
渲染每个人在虚拟相机坐标系中的 3D 位置和朝向。

两种用法:
  1) 本地文件:  python docs/render_robot_3d.py --local /tmp/robot_latest_3d.json
  2) SSH 拉取:  python docs/render_robot_3d.py --host 192.168.217.100 \
                       --user naviai --password 'naviai@2024' \
                       --remote data_outputs/webcam_head_onnx/latest_3d.json

输出:
  docs/_robot_3d_view.png   — 3D 场景 (相机 + 每个人 + 朝向 + 距离)
  docs/_robot_data.json     — 本地保存的机器人数据副本
"""
import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

# ── 中文字体 ──
_ZH_FONT = '/usr/share/fonts/fonts-gb/GB_ST_GB18030.ttf'
if os.path.exists(_ZH_FONT):
    try:
        font_manager.fontManager.addfont(_ZH_FONT)
        _ZH_NAME = font_manager.FontProperties(fname=_ZH_FONT).get_name()  # GB_SS_GB18030
        matplotlib.rcParams['font.sans-serif'] = [_ZH_NAME, 'DejaVu Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

# ── OpenPose BODY_25 骨架连接 (hsmr_44 前25点) ──
BODY25_BONES = [
    [1,8],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],
    [8,9],[9,10],[10,11],[8,12],[12,13],[13,14],
    [1,0],[0,15],[15,17],[0,16],[16,18],
    [14,21],[19,21],[20,21],[11,24],[22,24],[23,24],
]


def fetch_via_ssh(host, user, password, remote):
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=10)
    sftp = client.open_sftp()
    local = '/tmp/_robot_latest_3d.json'
    sftp.get(remote, local)
    sftp.close(); client.close()
    return local


def render_scene(data, out_png):
    cf = data.get('coordinate_frames', {})
    persons = data.get('persons', [])

    fig = plt.figure(figsize=(20, 11))
    # ---------- 3/4 透视图 ----------
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title('3/4 透视图\n相机在原点(0,0,0), +X右 +Y下 +Z远离相机', fontsize=12, fontweight='bold')

    # 相机
    ax.scatter([0],[0],[0], s=250, c='#a371f7', marker='s', edgecolors='white', zorder=10)
    ax.text(0.05, 0.05, 0.05, '相机', fontsize=9, color='#a371f7', fontweight='bold')
    _axes3d(ax, [0,0,0], 0.5)

    colors = ['#f78166', '#58a6ff', '#3fb950', '#d2991d', '#a371f7']
    for i, person in enumerate(persons):
        col = colors[i % len(colors)]
        joints = person['joints']['hsmr_44']['model_origin_relative_m']
        J = np.array(joints)  # [44,3] 模型原点相对 (虚拟相机系, +Y下)
        pos = person['position']
        pelvis_vcam = pos['pelvis_full_image_virtual_camera_m']
        fwd = person['orientation']['body_forward_unit_model_camera']
        quat = person['orientation']['quaternion_body_canonical_to_model_camera_wxyz']
        dist = pos['pelvis_distance_from_virtual_camera_m']

        # 人体 (模型原点相对)
        for a, b in BODY25_BONES:
            ax.plot([J[a,0], J[b,0]], [J[a,1], J[b,1]], [J[a,2], J[b,2]],
                    '-', color=col, lw=2, alpha=0.7)
        ax.scatter(J[:25,0], J[:25,1], J[:25,2], c=col, s=25, zorder=5)

        # 前方向量 (从模型原点处的骨盆画)
        root0 = J[8]  # mid_hip
        ax.quiver(root0[0], root0[1], root0[2],
                  fwd[0]*0.5, fwd[1]*0.5, fwd[2]*0.5,
                  color='#ffd166', linewidth=3, arrow_length_ratio=0.2)
        ax.text(root0[0]+fwd[0]*0.55, root0[1]+fwd[1]*0.55, root0[2]+fwd[2]*0.55,
                f'P{i} q=[{quat[0]:+.2f},{quat[1]:+.2f},{quat[2]:+.2f},{quat[3]:+.2f}]',
                fontsize=7, color='#ffd166', fontweight='bold')

        # 真实虚拟相机位置 (骨盆)
        ax.scatter([pelvis_vcam[0]], [pelvis_vcam[1]], [pelvis_vcam[2]],
                   s=150, c=col, marker='*', edgecolors='white', zorder=8)
        # 相机→骨盆连线
        ax.plot([0, pelvis_vcam[0]], [0, pelvis_vcam[1]], [0, pelvis_vcam[2]],
                '--', color=col, alpha=0.5)
        ax.text(pelvis_vcam[0]/2, pelvis_vcam[1]/2, pelvis_vcam[2]/2,
                f'P{i} 距离={dist:.1f}m', fontsize=8, color=col)

    ax.set_xlabel('X (图像右→)'); ax.set_ylabel('Y (图像下→)'); ax.set_zlabel('Z (远离相机→)')
    ax.view_init(elev=18, azim=-65)

    # ---------- 侧面 (Y-Z) ----------
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title('侧面视图 (Y-Z)\n看每个人相对相机的距离与高度', fontsize=12, fontweight='bold')
    ax2.scatter([0],[0],[0], s=250, c='#a371f7', marker='s', edgecolors='white', zorder=10)
    _axes3d(ax2, [0,0,0], 0.4)
    for i, person in enumerate(persons):
        col = colors[i % len(colors)]
        pos = person['position']
        pelvis_vcam = pos['pelvis_full_image_virtual_camera_m']
        dist = pos['pelvis_distance_from_virtual_camera_m']
        ax2.scatter([pelvis_vcam[1]], [pelvis_vcam[2]], [0], c=col, s=120, marker='*', zorder=8)
        ax2.plot([0, pelvis_vcam[1]], [0, pelvis_vcam[2]], '--', color=col, alpha=0.5)
        ax2.text(pelvis_vcam[1], pelvis_vcam[2], 0, f' P{i} d={dist:.1f}m', fontsize=9, color=col)
    ax2.set_xlabel('Y (下→)'); ax2.set_ylabel('Z (远离相机→)')
    ax2.view_init(elev=0, azim=-90)

    # ---------- 俯视 (X-Z) ----------
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_title('俯视图 (X-Z)\n看每个人左右位置与朝向', fontsize=12, fontweight='bold')
    ax3.scatter([0],[0],[0], s=250, c='#a371f7', marker='s', edgecolors='white', zorder=10)
    _axes3d(ax3, [0,0,0], 0.4)
    for i, person in enumerate(persons):
        col = colors[i % len(colors)]
        pos = person['position']
        pelvis_vcam = pos['pelvis_full_image_virtual_camera_m']
        fwd = person['orientation']['body_forward_unit_model_camera']
        ax3.scatter([pelvis_vcam[0]], [pelvis_vcam[2]], [0], c=col, s=120, marker='*', zorder=8)
        ax3.quiver(pelvis_vcam[0], pelvis_vcam[2], 0,
                   fwd[0]*0.8, fwd[2]*0.8, 0, color='#ffd166', linewidth=3, arrow_length_ratio=0.2)
        ax3.plot([0, pelvis_vcam[0]], [0, pelvis_vcam[2]], '--', color=col, alpha=0.5)
        ax3.text(pelvis_vcam[0]+fwd[0]*0.9, pelvis_vcam[2]+fwd[2]*0.9, 0,
                 f'P{i} 前向={np.round(fwd,2).tolist()}', fontsize=7, color=col)
    ax3.set_xlabel('X (右→)'); ax3.set_ylabel('Z (远离相机→)')
    ax3.view_init(elev=90, azim=-90)

    # ---------- 数据表格 ----------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    lines = ['===== 每个人 3D 数据 (虚拟相机坐标系) =====', '']
    for i, person in enumerate(persons):
        pos = person['position']; ori = person['orientation']
        det = person['detection']
        lines += [
            f'【Person {i}】 score={det["score"]:.2f}',
            f'  bbox(px): {person["crop_bbox_left_top_right_bottom_px"]}',
            f'  骨盆位置(相机坐标, m): {[round(x,3) for x in pos["pelvis_full_image_virtual_camera_m"]]}',
            f'  距相机距离: {pos["pelvis_distance_from_virtual_camera_m"]:.2f} m',
            f'  根旋转矩阵: {[row[:3] for row in ori["rotation_matrix_body_canonical_to_model_camera"]]}',
            f'  四元数(wxyz): {[round(x,4) for x in ori["quaternion_body_canonical_to_model_camera_wxyz"]]}',
            f'  前方向量: {[round(x,3) for x in ori["body_forward_unit_model_camera"]]}',
            '',
        ]
    lines += [
        '===== 坐标帧说明 =====',
        '模型系: +X图像右, +Y图像下, +Z远离相机',
        '人体位置 = model_origin_relative + model_origin_full_image_virtual_camera',
        '注意: 单目估计深度(虚拟焦距5000), 非RealSense/ROS/TF',
        '',
        f'来源: {data.get("schema")} | sample={data.get("sample_id")} | seq={data.get("source_sequence")} | 人={data.get("persons_count")}',
    ]
    ax4.text(0.02, 0.98, '\n'.join(lines), transform=ax4.transAxes,
             va='top', ha='left', fontsize=8.5, linespacing=1.5)

    _font_ok = os.path.exists(_ZH_FONT)
    fig.suptitle(
        f'机器人摄像头预测 → 3D 场景 | sample={data.get("sample_id")} | 时间={data.get("captured_unix")}\n'
        f'{"中文渲染正常" if _font_ok else "⚠中文字体缺失"}',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_png, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()


def _axes3d(ax, origin, length):
    o = np.array(origin)
    for d, c, name in [([1,0,0],'red','X'), ([0,1,0],'green','Y'), ([0,0,1],'blue','Z')]:
        ax.quiver(o[0], o[1], o[2], d[0]*length, d[1]*length, d[2]*length,
                  color=c, linewidth=2, arrow_length_ratio=0.15)
        ax.text(o[0]+d[0]*length*1.1, o[1]+d[1]*length*1.1, o[2]+d[2]*length*1.1,
                name, color=c, fontsize=10, fontweight='bold')


def main():
    ap = argparse.ArgumentParser(description='机器人3D预测渲染')
    ap.add_argument('--local', help='本地 latest_3d.json 路径')
    ap.add_argument('--host', default='192.168.217.100')
    ap.add_argument('--user', default='naviai')
    ap.add_argument('--password', default='naviai@2024')
    ap.add_argument('--remote', default='projects/HSMR-main/data_outputs/webcam_head_onnx/latest_3d.json')
    ap.add_argument('--out', default='docs/_robot_3d_view.png')
    args = ap.parse_args()

    if args.local:
        local = args.local
    else:
        print(f'[SSH] 拉取 {args.user}@{args.host}:{args.remote} ...')
        local = fetch_via_ssh(args.host, args.user, args.password, args.remote)
        print(f'[SSH] 已下载: {local}')

    with open(local) as f:
        data = json.load(f)

    # 保存副本
    with open('docs/_robot_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    render_scene(data, args.out)
    print(f'✅ 渲染完成: {args.out}')
    print(f'   人数: {data.get("persons_count")}, sample={data.get("sample_id")}')
    for i, p in enumerate(data.get('persons', [])):
        print(f'   P{i}: 距离={p["position"]["pelvis_distance_from_virtual_camera_m"]:.2f}m, '
              f'四元数={[round(x,4) for x in p["orientation"]["quaternion_body_canonical_to_model_camera_wxyz"]]}')


if __name__ == '__main__':
    main()
