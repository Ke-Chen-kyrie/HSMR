# SKEL 论文第五部分 ↔ 代码对照

> 论文：[Keller et al., "From Skin to Skeleton: Towards Biomechanically Accurate 3D Digital Humans", arXiv 2509.06607](https://arxiv.org/abs/2509.06607)（SIGGRAPH Asia 2023）
> 本文把**论文第五部分**逐段对应到 `skel_model.py` 代码，并解释此前验证结果。

---

## 0. 第五部分在讲什么

**标题《The SKEL model》** —— 把 BSM 生物力学骨骼 + SMPL 皮肤放进同一参照系，重绑 SMPL。

```
5.1  骨骼位置与朝向 (establishing bone locations & orientations)
      - 关节位置: 学习关节回归器 J (从 SMPL 顶点 → 24个解剖关节)
      - 骨骼朝向: R_i(β) = R_i^β(β) · R_i^base
5.2  构建 SKEL (single rig for skin and bones)
      - SKEL 函数: (β, q∈R⁴⁶) → (vskin, vskel, J)
      - 皮肤: LBS 方程 (5)(6), 骨骼: 全局变换 G_i^skin = 链乘
```

---

## 1. 论文 ↔ 代码 对照表

| 论文 (Sec 5) | 代码 | 验证 |
|---|---|---|
| **关节回归器 J** (5.1) | `skel_model.py:324` `J = einsum('bik,ji->bjk', v_shaped, J_regressor_osim)` | ✅ |
| **骨骼朝向** R_i(β)=R_i^β·R_i^base (式3) | `skel_model.py:637-688` `compute_bone_orientation()` → **Rk01**<br>`per_joint_rot` = R_i^base (learned)<br>`rotation_matrix_from_vectors(...)` = R_i^β (shape-dependent)<br>`Gk = Gk @ Gk_learned` = R_i^β·R_i^base | ✅ 代码顺序与式(3)一致 |
| **骨骼 rest 变换** T(R_k(β), J_k(β)) (式6) | `skel_model.py:512` `Gk01 = build_homog_matrix(Rk01, J)` | ✅ |
| **局部旋转** q→G_k^B (式6 中间项) | `osim_rot.py:23` `q_to_rot` + `skel_model.py:345` `pose_params_to_rot` | ✅ |
| **全局链乘** (式6 前导 Π) | `skel_model.py:430-438` `G_=[R\|t]`, `G[child]=G[parent]@G_[child]` | ✅ |
| **全局骨骼朝向** | `skel_model.py:527,545` `G_bones=G@Gk01`, `joints_ori=G_bones[:,:,:3,:3]` | ✅ |
| **皮肤 LBS** (式5) | `skel_model.py:470-476` skinning `T=W·Gskin`, `v_posed` | ✅ |

---

## 2. 论文的关键论述：为什么"局部/全局"要这样设计

> **SMPL**: "all joint orientations are defined in a **global T-pose space** with an **axis-aligned** frame of reference for each joint... the elbow rotation axis is aligned with the world y-axis, independent of the orientation of the humerus."
>
> **SKEL/BSM**: "requires the local frame on which the rotation is applied to be **precisely aligned with the anatomy**" — Fig 6 说明 *"In contrast to SMPL, which has axis-aligned rotation axes, SKEL's rotation axes are bone-aligned."*

**一句话**：
- **SMPL**：每个关节旋转轴对齐世界轴（Y=竖直），与实际骨骼方向无关 → 过参数化（72参数）才能做出合理姿态
- **SKEL**：每个关节旋转在**骨骼对齐的局部系**里 → 46 参数（更少）就能做生物力学合理姿态，因为局部系已经和骨骼对齐

这就解释了为什么 SKEL 的 46 参数是"每块骨骼在自身局部系里的旋转角"（局部旋转），而"全局朝向"需要沿运动学树链乘。

---

## 3. 论文如何解释此前的验证结果

### 3.1 为什么 `joints_ori` 静止时 ≠ 单位矩阵

论文式(3) 骨骼 rest 朝向 `R_i(β) = R_i^β(β) · R_i^base`：
- `R_i^base` 是**学习到的骨骼绕自身轴的基准朝向**（永远存在，静止也不消失）
- `joints_ori` 含 `Gk01 = T(Rk01, J)`（骨骼 rest 变换）
- → 静止时 `joints_ori` = 骨骼 rest 朝向，**不是单位矩阵**

**验证**：静止 `joints_ori[0] = per_joint_rot[0]`（差异 0.000000）✓

### 3.2 为什么骨盆的 rest 朝向 = `per_joint_rot[0]`

论文 5.1 说 `R_i^β(β)` 是 shape-dependent（随体型变）。代码 `skel_model.py:686`：
```python
Gk[:, self.joint_idx_fixed_beta] = I   # 骨盆/脚趾/头/手 的朝向不随体型变
```
骨盆在 `joint_idx_fixed_beta` 里 → `R_i^β = I` → `Rk01[0] = I · per_joint_rot[0] = per_joint_rot[0]`

**验证**：静止 `joints_ori[0] = per_joint_rot[0]` ✓（论文+代码双重解释）

### 3.3 为什么机器人的 `root_rotation = joints_ori[:,0] @ per_joint_rot[0].T`

代码 `skel_model.py:688` `Gk = Gk @ Gk_learned`，所以 `Rk01[0] = per_joint_rot[0]`（骨盆）。
`joints_ori[0] = R[0] @ Rk01[0]`（静止时 = Rk01[0]）。
去掉 rest 朝向：`joints_ori[0] @ per_joint_rot[0].T` → 理论上得到纯骨盆旋转 R[0]。

**注意**：我验证过 `joints_ori[0] @ per_joint_rot[0].T ≠ params_q2rot(poses)[:,0]`（差异 0.149）。
差异来源可能是 `Rk01[0]` 与 `per_joint_rot[0]` 的关系在非静止时不完全等于（因 `Gk = rotation_matrix_from_vectors(...) @ Gk_learned` 对骨盆 `Gk[:,fixed]=I` 但 `Gk_learned` 保留）。**此差异的精确解释尚未完全验证，不下断言。**

---

## 4. 局部 vs 全局（论文语境下的准确定义）

| | 局部旋转 q | 全局朝向 joints_ori |
|---|---|---|
| 论文 | 46 个 BSM bone angles，在**骨骼对齐的局部系** | G_i^skin 链乘（式6 前导 Π）的旋转部分 |
| 参照 | 骨骼局部系（父骨骼） | **SMPL T-pose 空间**（模型空间） |
| 代码 | `params_q2rot` / `pose_params_to_rot` | `(G @ Gk01)[:,:,:3,:3]` |
| 静止 | q=0 → R=I | joints_ori = 骨骼 rest 朝向 Rk01（≠I） |

**论文明确**：SKEL 的参照系是 "SMPL T-pose space"，即模型 T-pose 空间（trans=0，模板网格），**不是相机、不是机器人 base_link**。这与 `bone_rotation_viz.html` 里标注的"SKEL 模型空间"一致。

---

## 5. 一句话总结

```
论文: SKEL 用 46 个在"骨骼对齐局部系"里的角度 q 驱动骨架,
      全局朝向 = 沿运动学树链乘 + 骨骼 rest 朝向 Rk01.
代码: q→params_q2rot(局部R),  FK 链乘 G,  ×Gk01 → joints_ori.
参照: SMPL T-pose 空间 (模型空间), 不是相机/base_link.
```

---

## 来源

- 论文：arXiv [2509.06607](https://arxiv.org/abs/2509.06607) 第五部分（PDF 本地：`/tmp/skel_paper.pdf`，Sec 5 在第 547-787 行提取文本）
- 代码：`thirdparty/SKEL/skel/skel_model.py:324,512,527,545,637-688`
- 代码：`thirdparty/SKEL/skel/osim_rot.py:23`、`lib/body_models/skel_utils/transforms.py:200`
