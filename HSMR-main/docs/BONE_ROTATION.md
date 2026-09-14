# 骨骼旋转 (Bone Rotation) — 从代码讲清楚

> 你问的"骨骼旋转" = 每块骨骼相对父骨骼的**局部旋转**，由 46 个姿态参数驱动。
> 这是 HSMR 输出的核心：`poses [46]` 决定人体姿势。

---

## 1. 骨骼旋转是什么

人体骨架 = 24 块骨骼，串成一个树（骨盆是根）。

```
骨盆 → 腰椎 → 胸椎 → 头
       ├── 右髋 → 右膝 → 右踝 → 右脚
       ├── 左髋 → 左膝 → 左踝 → 左脚
       └── (胸椎) → 右肩胛 → 右肩 → 右肘 → 右腕 → 右手
                  → 左肩胛 → 左肩 → 左肘 → 左腕 → 左手
```

**每一块骨骼有一个局部旋转矩阵 R**，描述它相对父骨骼怎么转。

- `R = 单位矩阵` → 骨骼沿父骨骼方向伸直（T-pose）
- `R ≠ 单位矩阵` → 骨骼弯曲/扭转

**公式**（代码 `transforms.py:225` + `osim_rot.py:23`）：
```python
# 每块骨骼 j:
R_j = R_last @ ... @ R_1 @ R_0        # 左乘积累
# 其中每个 R_i = 绕 axis[i] 旋转 (q[i] × flip[i]) 弧度
```

**全局朝向**（前向运动学，代码 `skel_model.py`）：
```python
G[child] = G[parent] @ R_child
# G[骨盆] = R_骨盆（根）
```

---

## 2. 24 块骨骼怎么转（JOINTS_DEF，代码 `definition.py`）

| jid | 骨骼 | DOF | 旋转轴 axis | flip | 姿态参数(q) |
|---|---|---|---|---|---|
| 0 | 骨盆 | 3 | Z, X, Y | +,+,+ | pelvis_tilt, pelvis_list, pelvis_rotation |
| 1 | 右大腿 | 3 | Z, X, Y | +,+,+ | hip_flex, hip_add, hip_rot (右) |
| 2 | 右小腿 | 1 | (绕全局Z, -q) | - | knee_angle_r |
| 3 | 右距骨 | 1 | (特殊) | - | ankle_r |
| 4 | 右跟骨 | 1 | (特殊) | - | subtalar_r |
| 5 | 右脚趾 | 1 | (特殊) | - | mtp_r |
| 6 | 左大腿 | 3 | Z, X, Y | +,-,- | hip_flex, hip_add, hip_rot (左) |
| 7 | 左小腿 | 1 | (绕全局Z, -q) | - | knee_angle_l |
| 8-10 | 左脚踝/跟/趾 | 1 | (特殊) | - | ankle_l, subtalar_l, mtp_l |
| 11 | 腰椎 | 3 | X, Z, Y | +,+,+ | lumbar_bending, extension, twist |
| 12 | 胸椎 | 3 | X, Z, Y | +,+,+ | thorax_bending, extension, twist |
| 13 | 头 | 3 | X, Z, Y | +,+,+ | head_bending, extension, twist |
| 14 | 右肩胛 | 3 | Y, Z, X | +,-,- | scapula_abd, elv, rot (右) |
| 15 | 右肱骨 | 3 | X, Y, Z | +,+,+ | shoulder_r_x, y, z |
| 16 | 右尺骨 | 1 | [0.05, 0.04, 1.0] | + | elbow_flex_r |
| 17 | 右桡骨 | 1 | [-0.02, 0.99, -0.12] | + | pro_sup_r |
| 18 | 右手 | 2 | X, Z | +,- | wrist_flex, wrist_dev (右) |
| 19 | 左肩胛 | 3 | Y, Z, X | +,+,+ | scapula_abd, elv, rot (左) |
| 20 | 左肱骨 | 3 | X, Y, Z | +,+,+ | shoulder_l_x, y, z |
| 21 | 左尺骨 | 1 | [-0.05, -0.04, 1.0] | + | elbow_flex_l |
| 22 | 左桡骨 | 1 | [0.02, -0.99, -0.12] | + | pro_sup_l |
| 23 | 左手 | 2 | X, Z | -,- | wrist_flex, wrist_dev (左) |

---

## 3. 具体算例（已用 SKEL 模型验证）

### 骨盆 (jid0)：axis=[Z,X,Y], tilt=0.15 rad
```python
R = R_Y(0) @ R_X(0) @ R_Z(0.15)
```
矩阵：
```
[[ 0.989 -0.149  0.   ]
 [ 0.149  0.989  0.   ]
 [ 0.     0.     1.   ]]
```
`R_Z(0.15)` = 绕 Z 轴转 0.15 弧度（前倾），X/Y 分量变化，Z 不变。✓

### 右膝 (jid2)：WalkerKnee，knee_r=1.2 rad
```python
theta = [0, 0, -1.2]   # 绕全局Z轴负方向转
```
矩阵：
```
[[ 0.362  0.932  0.   ]
 [-0.932  0.362 -0.   ]
 [-0.     0.     1.   ]]
```
`R_Z(-1.2)`。正 knee_angle → 屈膝（小腿相对大腿绕 Z 负转）。✓

### 左肱骨 (jid20)：axis=[X,Y,Z]，shoulder_l_y=-1.2 rad
```python
R = R_Z(0) @ R_Y(-1.2) @ R_X(0)
```
矩阵：
```
[[ 0.362  0.    -0.932]
 [ 0.     1.     0.   ]
 [ 0.932  0.     0.362]]
```
`R_Y(-1.2)` = 绕 Y 轴（肩关节抬臂轴）转 -1.2 弧度（抬左臂）。✓

---

## 4. 旋转矩阵怎么读

对旋转矩阵 R，它的**三个列向量**是骨骼本地坐标系的三个轴在父坐标系里的方向：

```python
R = [ |  |  | ]
     [X  Y  Z ]   # 本地X轴在全局的方向 = R[:,0], 依此类推
     [ |  |  | ]
```

- 静止时 R=I：本地X→全局X，本地Y→全局Y，本地Z→全局Z
- 旋转后：本地轴指向旋转后的方向

**验证**：`render_bone_rotation.py` 在每个关节点画出 R 的三轴（红=本地X, 绿=本地Y, 蓝=本地Z）。

---

## 5. 关键理解

1. **局部 vs 全局**：R 是相对父骨骼的局部旋转；G 是链乘后的全局朝向。运动组要关节世界朝向用 G，要关节弯曲角用 R。
2. **DOF 决定自由度**：髋/肩/脊柱是 3-DOF（可任意方向转），膝/踝/肘是 1-DOF（只能屈伸），腕是 2-DOF。
3. **axis 决定绕哪转**：每块骨骼的 axis 定义了旋转轴，flip 决定正负方向。
4. **不是欧拉角**：poses 是"Euler-like"参数，必须经 params_q2rot 转成矩阵才能用。直接读 poses 的 3 个数当 yaw/pitch/roll 是错的（SKEL 有自己的轴序）。
5. **orientation 的两个层次**：
   - 每块骨骼的**局部旋转 R**（这就是"骨骼旋转"）
   - 骨盆根的**全局朝向**（我之前讲的 forward/yaw）

---

## 6. 文件

| 文件 | 内容 |
|---|---|
| [render_bone_rotation.py](render_bone_rotation.py) | 骨骼旋转可视化脚本（已验证矩阵） |
| [_bone_rot_rest.png](_bone_rot_rest.png) | 静止姿态：每块骨骼本地轴 = 不转 |
| [_bone_rot_action.png](_bone_rot_action.png) | 动作姿态：屈右膝+抬左臂，看本地轴怎么转 |

## 7. 代码依据

- `lib/body_models/skel_utils/transforms.py:200` — `params_q2rot`：46参数→24旋转矩阵
- `lib/body_models/skel_utils/definition.py:73` — `JOINTS_DEF`：每块骨骼的 axis/flip/DOF
- `thirdparty/SKEL/skel/osim_rot.py:23` — `CustomJoint.q_to_rot`：轴角左乘积累
- `thirdparty/SKEL/skel/skel_model.py` — 前向运动学链乘 → joints_ori（全局朝向）
