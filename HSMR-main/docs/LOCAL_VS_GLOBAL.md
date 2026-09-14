# 局部旋转 vs 全局朝向 — 代码验证版（纠正版）

> 本文件所有结论均经过**实际运行 SKEL 模型验证**，并标注验证输出。
> 每一条都给出代码行号。此前文档中未被验证的推断已删除。

---

## 0. 重要更正（我之前讲错的）

| 此前说法 | 验证结果 | 状态 |
|---|---|---|
| "机器人 root_rotation = params_q2rot(poses)[:,0]" | `joints_ori[0] @ per_joint_rot[0].T` **≠** `params_q2rot(poses)[:,0]`（差异 0.149） | ❌ 已纠正 |
| "全局 G 静止=单位矩阵" | SKEL 的 `joints_ori` 静止时**≠单位矩阵**（含骨骼 rest 朝向 Rk01），静止时 `joints_ori[0] = per_joint_rot[0]` | ❌ 已纠正 |
| HTML 里 JS 自己算的"全局 G" = SKEL 的 joints_ori | JS 纯 FK 链乘(local_R) 与 joints_ori 差异 0.998，**不是一回事** | ❌ 已删除，改嵌入验证数据 |

---

## 1. 验证过的事实（附运行输出）

### 1.1 运动学树（`skel_model.py:142,156-158`）

运行输出（`model.parent` 映射）：
```
femur-R→pelvis, tibia-R→femur-R, talus-R→tibia-R, calcn-R→talus-R, toes-R→calcn-R
femur-L→pelvis, tibia-L→femur-L, talus-L→tibia-L, calcn-L→talus-L, toes-L→calcn-L
lumbar→pelvis, thorax→lumbar, head→thorax
scapula-R→thorax, humerus-R→scapula-R, ulna-R→humerus-R, radius-R→ulna-R, hand-R→radius-R
scapula-L→thorax, humerus-L→scapula-L, ulna-L→humerus-L, radius-L→ulna-L, hand-L→radius-L
```
**根 = 骨盆 (jid0)**。`parent` 数组长度 23（每个非根关节）。

### 1.2 根变换（`skel_model.py:430-438`）

```python
G_ = [R | t_posed]                # :430  局部变换 (旋转+平移)
G = [G_[:,0]]                      # :435  根 = 骨盆
G[i] = G[parent[i]] @ G_[:,i]      # :437  FK 链乘
```

**验证**：`G[0]` 的平移部分 = `t_posed[0] = J_[0] = J[0]`（骨盆位置）。

### 1.3 参照系 = SKEL 模型空间（trans=0）

运行输出：
```
骨盆 J[0]   = [ 0.0095, -0.2278,  0.1037]
头   J[13]  = [ 0.0010,  0.3813, -0.0074]
模板网格质心 = [0.0000,  0.0000,  0.0000]
```

结论：
- 原点 (0,0,0) = **模板网格质心**
- `joints_ori` 参照 **SKEL 模型空间**（trans=0）
- **不是相机系**（未加 camera_translation）
- **不是 base_link**（未加 TF）

### 1.4 joints_ori（全局骨骼朝向，`skel_model.py:545`）

```python
Gk01 = build_homog_matrix(Rk01, J)   # :512  骨骼 rest 轴对齐
G_bones = G @ Gk01                    # :527
joints_ori = G_bones[:,:,:3,:3]       # :545
```

**验证输出**（静止姿态）：
```
joints_ori[0] (骨盆) =
[[ 0.065 -0.003 -0.998]
 [-0.037  0.999 -0.005]
 [ 0.997  0.037  0.065]]
det = 1.0000 (旋转矩阵)
静止 joints_ori[0] vs per_joint_rot[0] 差异 = 0.000000  → 相同
```

**结论**：
- `joints_ori` = `(G @ Gk01)` 的旋转部分，**含骨骼 rest 朝向对齐（Rk01）**
- 静止时 `joints_ori[0] = per_joint_rot[0]`（骨盆骨骼的 rest 朝向，≠单位矩阵）
- `joints_ori` 与"纯 FK 链乘 local_R"**不同**（差异 0.998，因含 Rk01）

### 1.5 局部旋转（`transforms.py:200` + `osim_rot.py:23`）

```python
# params_q2rot: 每块骨骼 j 的局部旋转
params_rot[..., jid] = JOINTS_DEF[jid].q_to_rot(params_q[..., sid:eid])

# osim_rot.py CustomJoint.q_to_rot
Rp = I
for i in range(DOF):
    Rp = R(q[i]*flip[i] 绕 axis[i]) @ Rp
```

**验证输出**（动作姿态：屈右膝 knee=1.2, 抬左臂 shoulder_l_y=-1.2, 微前倾 tilt=0.15）：
```
骨盆 local_R[0] = [[0.989,-0.149,0],[0.149,0.989,0],[0,0,1]]   ← R_Z(0.15)
右膝 local_R[2] = [[0.362,0.932,0],[-0.932,0.362,0],[0,0,1]]    ← WalkerKnee R_Z(-1.2)
左肱 local_R[20]= [[0.362,0,-0.932],[0,1,0],[0.932,0,0.362]]     ← R_Y(-1.2)
```

---

## 2. 局部 vs 全局 — 已验证的区别

| 维度 | 局部 R | 全局 joints_ori |
|---|---|---|
| 参照系 | **父骨骼** | **SKEL 模型空间**（trans=0） |
| 含义 | 骨骼相对父骨骼的旋转 | 骨骼坐标轴在模型空间的朝向 |
| 求法 | `params_q2rot(poses)` | `(G @ Gk01)` 旋转部分，FK 链乘 |
| 静止 | R=I（每块骨骼沿父方向伸直） | joints_ori=骨骼 rest 朝向（Rk01, ≠I） |
| 代码 | `transforms.py:200`, `osim_rot.py:23` | `skel_model.py:435-438, 512, 527, 545` |
| 用途 | 关节弯曲角 | 骨骼模型空间朝向 |

**关键**：
- 局部 R 参照**父骨骼**（每块骨骼相对上一块怎么弯）
- 全局 joints_ori 参照**模型空间**（骨骼最终朝模型哪个方向）
- 全局 = FK 链乘（所有祖先旋转累积）+ 骨骼 rest 朝向

---

## 3. 未验证/存疑（不在此断言）

1. **机器人 `root_rotation = joints_ori[:,0] @ per_joint_rot[0].T` 的确切含义**：
   - 已验证它 **≠** `params_q2rot(poses)[:,0]`（差异 0.149）
   - 它是否等于"去掉骨骼 rest 朝向后的骨盆纯旋转"**未完全验证**，需要进一步分析 Rk01 构成
2. `joints_ori[0] @ per_joint_rot[0].T` 与 `R[0]` 的关系（差异恰好≈tilt 值 0.15），原因未深究

---

## 4. 机器人场景下参照系会变

bone_rotation_viz.html 展示的是**模型空间**。真实机器人要：
```
模型空间 + camera_translation(raw_cam_t) → 虚拟相机系   (hsmr_core.py:146)
虚拟相机系 + 真实深度反投影            → 真实光学系      (realtime_depth_3d.py)
真实光学系 + TF                       → base_link        (需完整 TF)
```
所以在机器人 `latest_3d.json` 里，`joints_ori` 相关输出若加了 camera_translation，参照就不再是纯模型空间。

---

## 来源

- 代码: `thirdparty/SKEL/skel/skel_model.py:142,156-158,430-438,512,527,545`
- 代码: `lib/body_models/skel_utils/transforms.py:200`
- 代码: `thirdparty/SKEL/skel/osim_rot.py:23`
- 验证: 本文件所有运行输出来自 `python` 直接调用 SKEL 模型
