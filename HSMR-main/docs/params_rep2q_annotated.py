"""
params_rep2q 逐行注释版 — 6D旋转 → SKEL 46参数

整体逻辑:
  输入 [..., 24, 6] (24关节的6D旋转) → 按关节DOF分三类 → 输出 [..., 46]
    1-DOF关节(12个): 6D前2维 [cos, -sin] → atan2 → 1个角度
    2-DOF关节(2个):  直接取6D前2维
    3-DOF关节(10个): 6D → 旋转矩阵 → Euler角(按约定+逆序+flip)
"""
from typing import Union
import torch
import numpy as np


def params_rep2q(params_rot: Union[torch.Tensor, np.ndarray]):
    '''
    Transform the continuous representation back to the SKEL style euler-like representation.

    ### Args
    - params_rot: Union[torch.Tensor, np.ndarray], shape = (...B, 24, 6)
        # 输入: 24个关节的6D旋转表示
        #   每关节6个数 = 旋转矩阵前两个列向量(局部X和局部Y轴方向)
        #   第三列 = 前两列叉乘(隐含, 不用存)

    ### Returns
    - shape = (...B, 46)
        # 输出: SKEL的46个姿态参数(每块骨骼的旋转角, 弧度)
    '''

    with PM.time_monitor('params_rep2q'):
        with PM.time_monitor('preprocess'):
            # 统一转成 torch tensor (兼容 numpy 输入)
            params_rot, recover_type_back = to_tensor(params_rot, device=None, temporary=True)
            # 实际: params_rot.shape = [1, 24, 6]

            Bs = params_rot.shape[:-2]          # 批次维度, 实际 [1]
            params_q = params_rot.new_zeros((*Bs, 46))  # 初始化46维输出, [1, 46]

        with PM.time_monitor(f'dof1&dof2'):
            # ── ① 1-DOF 关节 (12个: 膝/踝/趾/肘/前臂) ──
            # DoF1_JIDS = [2,3,4,5,7,8,9,10,16,17,21,22]  关节编号
            # DoF1_QIDS = [6,7,8,9,13,14,15,16,32,33,42,43] 参数槽位
            # 取每关节6D的前2维 [cos, -sin], atan2 还原成角度
            params_q[..., DoF1_QIDS] = rotation_2d_to_angle(params_rot[..., DoF1_JIDS, :2]).squeeze(-1)
            # 结果: knee_r, ankle_r, ..., elbow_flex_l, pro_sup_l  (12个角)

            # ── ② 2-DOF 关节 (2个: 左右腕) ──
            # DoF2_JIDS = [18, 23]   关节编号 (hand_r, hand_l)
            # DoF2_QIDS = [34,35,44,45] 参数槽位 (每腕2个: 屈伸+侧摆)
            # 直接取6D前2维 (2-DOF参数就是2个值)
            params_q[..., DoF2_QIDS] = params_rot[..., DoF2_JIDS, :2].reshape(*Bs, -1)
            # 结果: wrist_flex/deviation_r, wrist_flex/deviation_l  (4个值)

        with PM.time_monitor(f'dof3'):
            # ── ③ 3-DOF 关节 (10个: 骨盆/髋/腰椎/胸/头/肩胛/肱骨) ──
            # DoF3_JIDS = [0,1,6,11,12,13,14,15,19,20]  关节编号
            # 取出10个3-DOF关节的6D → [1,10,6]
            dof3_6ds = params_rot[..., DoF3_JIDS, :].reshape(*Bs, len(DoF3_JIDS), 6)

            # 6D → 3×3 旋转矩阵 (Gram-Schmidt正交化)
            dof3_mats = rotation_6d_to_matrix(dof3_6ds)  # [1,10,3,3]
            # 用户实例 (10个旋转矩阵, 每个行列式≈1, 列=局部X/Y/Z轴在参考系方向):
            #   [0] pelvis     [0.699,0.068,-0.712; ...]  YXZ → tilt=-176°, list=2°, rot=-45°
            #   [1] femur-R    YXZ → hip_flex=17°, add=28°, rot=-18°
            #   [2] femur-L    YXZ → hip_flex=4°, add=3°, rot=-11°
            #   [3] lumbar     YZX → 前倾-6°, 侧弯-3°
            #   [4] thorax     YZX → 前倾14°, 侧弯-5°
            #   [5] head       YZX → 转头-39°
            #   [6] scapula-R  XZY → 内收-19°
            #   [7] humerus-R  ZYX → 抬臂-37°, 前伸-30°, 内旋62°
            #   [8] scapula-L  XZY → 内收28°
            #   [9] humerus-L  ZYX → 抬臂38°, 前伸4°, 内旋86°

            # 按 Euler 约定分组, 旋转矩阵 → 3个Euler角
            # CON_GROUP2JIDS: {'YXZ':[0,1,6], 'YZX':[11,12,13], 'XZY':[14,19], 'ZYX':[15,20]}
            for convention, jids in CON_GROUP2JIDS.items():
                idxs = [DoF3_JIDS.index(jid) for jid in jids]  # 这组关节在dof3_mats里的位置
                mats = dof3_mats[..., idxs, :, :]               # 取这组旋转矩阵

                # 旋转矩阵 → Euler角 (用该关节的约定)
                qs = matrix_to_euler_angles(mats, convention=convention)  # [1, J', 3]

                qs = qs[..., [2, 1, 0]]   # SKEL用逆序 (和JOINTS_DEF的axis顺序对齐)
                flips = qs.new_tensor(CON_GROUP2FLIPS[convention])  # 左右镜像符号
                qs = qs * flips           # 应用符号翻转

                qids = [qid for jid in jids for qid in JID2QIDS[jid]]  # 对应参数槽位
                params_q[..., qids] = qs.reshape(*Bs, -1)  # 填入46参数

        with PM.time_monitor('post_process'):
            # 如果是numpy输入, 转回numpy输出
            params_q = recover_type_back(params_q)
    return params_q  # [1, 46] SKEL姿态参数(弧度)
