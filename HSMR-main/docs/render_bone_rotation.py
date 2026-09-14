"""
骨骼旋转可视化 — 每块骨骼怎么转的

用 SKEL 模型 + params_q2rot, 对一组姿态算出:
  - 局部旋转矩阵 R_local [24,3,3]  (每块骨骼相对父骨骼)
  - 全局朝向 G [24,3,3]            (前向运动学链乘: G[child]=G[parent]@R_local)
每块骨骼画本地坐标轴 (全局朝向), 直观看到骨骼怎么旋转.

用法: python docs/render_bone_rotation.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

for _f in ['/usr/share/fonts/fonts-gb/GB_ST_GB18030.ttf']:
    if os.path.exists(_f):
        try:
            font_manager.fontManager.addfont(_f)
            matplotlib.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=_f).get_name(), 'DejaVu Sans']
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            pass

import torch
from lib.body_models.skel_wrapper import SKELWrapper
from lib.body_models.skel_utils.transforms import params_q2rot
from lib.body_models.skel_utils.definition import JOINTS_DEF, JID2QIDS

JNAMES = ['pelvis','femur-R','tibia-R','talus-R','calcn-R','toes-R',
          'femur-L','tibia-L','talus-L','calcn-L','toes-L',
          'lumbar','thorax','head','scapula-R','humerus-R','ulna-R','radius-R','hand-R',
          'scapula-L','humerus-L','ulna-L','radius-L','hand-L']

BONES = [
    [0,11],[11,12],[12,13],
    [0,1],[1,2],[2,3],[3,4],[4,5],
    [0,6],[6,7],[7,8],[8,9],[9,10],
    [12,14],[14,15],[15,16],[16,17],[17,18],
    [12,19],[19,20],[20,21],[21,22],[22,23],
]

POSE_NAMES = ['pelvis_tilt','pelvis_list','pelvis_rotation','hip_flex_r','hip_add_r','hip_rot_r',
 'knee_r','ankle_r','subtal_r','mtp_r','hip_flex_l','hip_add_l','hip_rot_l','knee_l','ankle_l','subtal_l','mtp_l',
 'lumbar_bend','lumbar_ext','lumbar_twist','thorax_bend','thorax_ext','thorax_twist','head_bend','head_ext','head_twist',
 'scap_abd_r','scap_elv_r','scap_rot_r','shl_r_x','shl_r_y','shl_r_z','elbow_flex_r','pro_sup_r','wrist_flex_r','wrist_dev_r',
 'scap_abd_l','scap_elv_l','scap_rot_l','shl_l_x','shl_l_y','shl_l_z','elbow_flex_l','pro_sup_l','wrist_flex_l','wrist_dev_l']


def load_model():
    model = SKELWrapper(
        model_path='data_inputs/body_models/skel', gender='male', num_betas=10,
        joint_regressor_extra='data_inputs/body_models/SMPL_to_J19.pkl',
        joint_regressor_custom='data_inputs/body_models/J_regressor_SKEL_mix_MALE.pkl',
        make_dense=True,
    )
    model.eval()
    return model


def compute(poses):
    """返回 joints [24,3], local_R [24,3,3], global_G [24,3,3]."""
    model = load_model()
    with torch.no_grad():
        out = model(poses=poses, betas=torch.zeros(1,10), skelmesh=True)
    joints = out.joints_backup.squeeze(0).numpy()       # [24,3]
    global_G = out.joints_ori.squeeze(0).numpy()        # [24,3,3] 全局骨骼朝向
    local_R = params_q2rot(poses).squeeze(0).numpy()    # [24,3,3] 局部旋转
    return joints, local_R, global_G


def pose_rest():
    return torch.zeros(1, 46)


def pose_action():
    """屈右膝 + 抬左臂 + 微侧身."""
    p = torch.zeros(1, 46)
    p[0, 6] = 1.2      # knee_r 屈膝
    p[0, 42] = 1.6     # elbow_flex_l 屈左肘
    p[0, 40] = -1.2    # shoulder_l_y 抬左臂
    p[0, 0] = 0.15     # pelvis_tilt 微前倾
    return p


def _axis3d(ax, origin, R, length=0.15):
    """在 origin 画旋转矩阵 R 的三轴 (本地X/Y/Z在全局的方向)."""
    o = np.asarray(origin)
    for vec, c in [(R[:,0], 'red'), (R[:,1], 'green'), (R[:,2], 'blue')]:
        ax.quiver(o[0], o[1], o[2], vec[0]*length, vec[1]*length, vec[2]*length,
                  color=c, linewidth=1.5, arrow_length_ratio=0.3)


def print_table(poses, label):
    local_R = params_q2rot(poses).squeeze(0).numpy()
    print(f'\n═══ {label} ═══')
    print(f"{'jid':3s} {'bone':12s} {'DOF':3s} {'axis(定义)':28s} {'flip':18s} {'q值(弧度)':40s}")
    for jid in range(24):
        j = JOINTS_DEF[jid]
        dof = j.nb_dof.item()
        # 轴/flip (部分关节无 axis 属性)
        if hasattr(j, 'axis'):
            axis = np.asarray(j.axis).round(2)
            axis_s = '[' + '; '.join(str(a.tolist()) for a in axis) + ']'
        else:
            axis_s = '(special)'
        if hasattr(j, 'axis_flip'):
            flip = np.asarray(j.axis_flip).flatten().round(2)
            flip_s = str(flip.tolist())
        else:
            flip_s = '(special)'
        qids = JID2QIDS[jid]
        q = poses[0, qids].numpy().round(2)
        q_s = ' '.join(f'{POSE_NAMES[qid]}={q[i]:+.2f}' for i, qid in enumerate(qids))
        print(f'{jid:3d} {JNAMES[jid]:12s} {dof:<3d} {axis_s:28s} {flip_s:18s} {q_s}')


def render(joints, global_G, title, out_png):
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=13, fontweight='bold')

    # 骨架
    for a, b in BONES:
        ax.plot([joints[a,0],joints[b,0]], [joints[a,1],joints[b,1]], [joints[a,2],joints[b,2]],
                '-', color='#888', lw=2)

    # 每块骨骼: 画关节点 + 本地坐标轴(全局朝向)
    for jid in range(24):
        col = '#f78166' if jid in [1,2,3,4,5,14,15,16,17,18] else '#58a6ff'
        ax.scatter([joints[jid,0]],[joints[jid,1]],[joints[jid,2]], c=col, s=40, edgecolors='black', zorder=5)
        _axis3d(ax, joints[jid], global_G[jid])

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    # 等比例
    mx = np.abs(joints).max() + 0.2
    ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx); ax.set_zlim(-mx, mx)
    ax.view_init(elev=12, azim=-60)
    # 图例说明
    ax.text2D(0.02, 0.95, '红=骨本地X轴\n绿=骨本地Y轴\n蓝=骨本地Z轴', transform=ax.transAxes,
              fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print('加载 SKEL...')
    model = load_model()

    # 静止
    print_table(pose_rest(), '静止姿态 (所有骨骼旋转=单位矩阵)')
    joints_r, local_R_r, G_r = compute(pose_rest())
    render(joints_r, G_r, '静止姿态\n每块骨骼本地坐标轴 (全局朝向) = 未旋转', 'docs/_bone_rot_rest.png')

    # 动作
    print_table(pose_action(), '动作姿态 (屈右膝+抬左臂+微前倾)')
    joints_a, local_R_a, G_a = compute(pose_action())
    render(joints_a, G_a, '动作姿态: 屈右膝+抬左臂+微前倾\n每块骨骼本地坐标轴 = 骨骼旋转', 'docs/_bone_rot_action.png')

    # 对比: 骨盆旋转矩阵
    print('\n═══ 骨盆(pelvis jid0) 局部旋转矩阵 R 对比 ═══')
    print('静止 R:'); print(local_R_r[0].round(3))
    print('动作 R:'); print(local_R_a[0].round(3))
    print('\n═══ 右膝(tibia jid2) 局部旋转矩阵 R 对比 ═══')
    print('静止 R:'); print(local_R_r[2].round(3))
    print('动作 R:'); print(local_R_a[2].round(3))
    print('\n═══ 左肱骨(humerus-l jid20) 局部旋转矩阵 R 对比 ═══')
    print('静止 R:'); print(local_R_r[20].round(3))
    print('动作 R:'); print(local_R_a[20].round(3))

    print('\n✅ 输出: docs/_bone_rot_rest.png, docs/_bone_rot_action.png')


if __name__ == '__main__':
    main()
