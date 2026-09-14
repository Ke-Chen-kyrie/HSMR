# HSMR 完整流程解析 — 从 RGB 到机器人 BASE 坐标

> 把之前所有讲解整合成一份。含全部关节名册。
> 数据来自代码 `lib/` + 实际运行验证。

---

## 0. 一句话总览

```
相机看到人 → 量出距离 → 算出3D → 换到机器人坐标 → 看人朝哪
    ①           ②          ③          ④           ⑤
```

完整链路：
```
RGB图
 → detector        人框(缩小图) + 所有类别 + 分数
 → _img_det2patches 筛人(0,>0.5)→top-k→还原框→扩192:256→方形→裁剪→256×256
 → HSMR网络         poses/betas/camera[scale,tx,ty]
 → params_rep2q     6D旋转 → SKEL 46参数
 → SKEL模型         44关节3D + 皮肤网格 + 根旋转矩阵
 → 2D投影           44关节2D像素 (虚拟相机 f=5000)
 → aligned_depth    2D像素处采样真实深度(米)
 → 反投影           X=(u-cx)Z/fx, Y=(v-cy)Z/fy → 光学系3D
 → TF               optical→HEAD→…→BASE → 机器人坐标
 → 输出             BASE坐标 + 旋转矩阵 + 四元数
```

---

## 1. 各环节详解

### ① detector (`vitdet/utils_detectron2.py:66`)

**输入**：RGB 图列表 → **输出**：`(preds, downsample_ratios)`

```python
preds[i] = {
  'pred_classes': [M],   # 每框类别 (0=人, COCO)
  'scores':       [M],   # 置信度
  'pred_boxes':   [M,4], # [左,上,右,下] ← 在缩小后图
}
downsample_ratios[i]     # 该图缩小比例
```

流程：
1. 最长边 > 512 → 缩小，记录 `downsample_ratio = 512/最长边`
2. 数据增强 → 转 [3,H,W] CPU tensor
3. 分批推理（batch_size，OOM 自动减半）
4. `pred_boxes` 要 `/downsample_ratio` 还原到原图

### ② `_img_det2patches` (`hsmr_demo.py:123`)

**输入**：原图 + 检测结果 → **输出**：`(patches[N,256,256,3], cs_all[N,3])`

```
筛人(class=0, score>0.5)
→ 超max_instances取top-k
→ 框/ratio还原原图
→ fit_bbox_to_aspect_ratio(192:256)   # ViT只吃中间256×192
→ lurb_to_cs 转方形[中心x,中心y,边长]
→ 裁剪 + 缩放256×256
```
`cs_all = [cx, cy, scale]` 之后用于 2D 关节映射回原图。

### ③ HSMR `forward` (`pipelines/hsmr.py:243`)

```
patch [B,3,256,256]
 → forward_step: ViT → poses_6d + betas + camera[scale,tx,ty]
 → _adapt_skel_params: poses_6d → poses(46)
 → skel_model → 44关节3D(根相对) + 皮肤网格
 → perspective_projection → 44关节2D
```

### ④ `forward_step` (`pipelines/hsmr.py:292`)

```
backbone(x[:,:,:,32:-32])   # 裁左右32 → 256×192
→ head(feats)               # transformer + decoder
→ camera → camera_translation:
    pd_cam_t = [tx, ty, 2×5000/(256×scale)]
```

**关键**：`pd_cam_t` 是虚拟相机平移（假深度）。`scale` 大→人近，小→人远（单目估计，非真实）。

### ⑤ ViT `forward_features` (`backbones/vit.py:307`)

```
[1,3,256,192] → patch_embed(16×16) → 192个token
→ 位置编码 → 12/32层transformer(自注意力)
→ last_norm → reshape [B,D,16,12] 特征图
```

### ⑥ head (`heads/skel_head.py:65`)

```
特征图 → 192 token
→ init先验(平均姿态/体型/相机)
→ 一个token与全图注意力 → 汇总
→ 3个decoder(残差) + init → poses_6d + betas + cam
→ 6D → params_rep2q → 46参数
```

**关键**：decoder 输出"偏移量"，加回 `init`（平均）→ 网络只学偏离多少，稳。

### ⑦ `params_rep2q` (`transforms.py:365`) — 6D → 46

```
输入 [1,24,6] (24关节6D旋转)
→ 按DOF分三类:
  1-DOF(12): 取6D前2维[cos,-sin] → atan2 → 角度
  2-DOF(2):  直接取6D前2维
  3-DOF(10): 6D→旋转矩阵→Euler角(按约定+逆序+flip)
→ 输出 [1,46]
```

---

## 2. 核心概念

### 6D 旋转表示

```
6D = [局部X轴在参考系方向(3个数) | 局部Y轴在参考系方向(3个数)]
     局部Z轴 = X×Y (叉积, 隐含)
```
**为什么用 6D**：无约束（网络随便输出）、连续（好回归），对比欧拉角有 gimbal lock。

### 弧度 ↔ 角度

```
角度° = 弧度 × 180/π   (1 rad ≈ 57.3°)
python: math.degrees(rad) / np.degrees(rad)
```

### DOF（自由度）

```
1-DOF = 只能绕一根轴转 (铰链): 膝/踝/趾/肘/前臂
2-DOF = 两根轴: 腕(屈伸+侧摆)
3-DOF = 任意方向 (球关节): 骨盆/髋/肩/脊柱/头/肩胛
12×1 + 2×2 + 10×3 = 46 自由度 = 46 参数
```

### JID vs QID

```
JID = 关节编号 (0~23, 24个)
QID = 参数编号 (0~45, 46个)
DoF1_JIDS/QIDS = 1-DOF关节的关节号/参数号
QIDS 数量 = DOF × 关节数
```

### Euler 约定（旋转轴顺序）

```
同一个旋转矩阵, 用不同轴顺序分解 → 不同角度!
YXZ: R = R_Y@R_X@R_Z
每块骨骼按 axis 选约定:
  pelvis(axis Z,X,Y) → YXZ
  lumbar(axis X,Z,Y) → YZX
  humerus(axis X,Y,Z)→ ZYX
```

### 局部 vs 全局

```
局部R = 骨骼相对父骨骼的旋转 (关节弯曲角)
全局G = 骨骼在世界/相机系的朝向 (FK链乘)
SKEL: +X=身体左侧, +Y=上, +Z=前 (右手系 X×Y=Z)
```

### 虚拟相机 vs 真实深度

```
虚拟相机: f=5000, 假深度 (单目估计, 只投影对齐)
真实深度: aligned_depth (红外立体测距, 米)
虚拟相机参数不能改成真实K (网络训练假设, 会错位12%)
正确: 虚拟算2D + 真实深度 + 真实K反投影 → 3D
```

---

## 3. 关节全名册

### 3.1 SKEL 24 解剖关节（`JOINTS_DEF`）

**DOF 含义** = 这个关节能做的动作（每个自由度对应一个解剖动作）。

| JID | 名称 | DOF | DOF 代表的意思（动作） | 旋转轴 | flip | 约定 | qids |
|---|---|---|---|---|---|---|---|
| 0 | **pelvis** | 3 | 骨盆前倾/后倾、侧倾、水平旋转 | Z,X,Y | +,+,+ | YXZ | 0-2 |
| 1 | **femur-R**（右大腿） | 3 | 髋：屈伸、内收外展、内外旋 | Z,X,Y | +,+,+ | YXZ | 3-5 |
| 2 | **tibia-R**（右小腿） | 1 | 膝：屈伸 | WalkerKnee | - | - | 6 |
| 3 | **talus-R**（右距骨/踝） | 1 | 踝：背屈/跖屈 | PinJoint | - | - | 7 |
| 4 | **calcn-R**（右跟骨） | 1 | 足：内翻/外翻（距下关节） | PinJoint | - | - | 8 |
| 5 | **toes-R**（右脚趾） | 1 | 趾：脚趾屈伸（跖趾关节） | PinJoint | - | - | 9 |
| 6 | **femur-L**（左大腿） | 3 | 髋：屈伸、内收外展、内外旋 | Z,X,Y | +,-,- | YXZ | 10-12 |
| 7 | **tibia-L**（左小腿） | 1 | 膝：屈伸 | WalkerKnee | - | - | 13 |
| 8 | **talus-L**（左距骨/踝） | 1 | 踝：背屈/跖屈 | PinJoint | - | - | 14 |
| 9 | **calcn-L**（左跟骨） | 1 | 足：内翻/外翻 | PinJoint | - | - | 15 |
| 10 | **toes-L**（左脚趾） | 1 | 趾：脚趾屈伸 | PinJoint | - | - | 16 |
| 11 | **lumbar**（腰椎） | 3 | 腰：屈伸、侧弯、扭转 | X,Z,Y | +,+,+ | YZX | 17-19 |
| 12 | **thorax**（胸椎） | 3 | 胸：屈伸、侧弯、扭转 | X,Z,Y | +,+,+ | YZX | 20-22 |
| 13 | **head**（头） | 3 | 头：屈伸、侧倾、旋转 | X,Z,Y | +,+,+ | YZX | 23-25 |
| 14 | **scapula-R**（右肩胛） | 3 | 肩胛：内收外展、升降、上回旋 | Y,Z,X | +,-,- | XZY | 26-28 |
| 15 | **humerus-R**（右肱骨/肩） | 3 | 肩：屈伸、外展内收、内外旋 | X,Y,Z | +,+,+ | ZYX | 29-31 |
| 16 | **ulna-R**（右尺骨/肘） | 1 | 肘：屈伸 | [0.05,0.04,1.0] | + | - | 32 |
| 17 | **radius-R**（右桡骨/前臂） | 1 | 前臂：旋前/旋后 | [-0.02,0.99,-0.12] | + | - | 33 |
| 18 | **hand-R**（右手） | 2 | 腕：屈伸、桡尺偏（侧摆） | X,Z | +,- | - | 34-35 |
| 19 | **scapula-L**（左肩胛） | 3 | 肩胛：内收外展、升降、上回旋 | Y,Z,X | +,+,+ | XZY | 36-38 |
| 20 | **humerus-L**（左肱骨/肩） | 3 | 肩：屈伸、外展内收、内外旋 | X,Y,Z | +,+,+ | ZYX | 39-41 |
| 21 | **ulna-L**（左尺骨/肘） | 1 | 肘：屈伸 | [-0.05,-0.04,1.0] | + | - | 42 |
| 22 | **radius-L**（左桡骨/前臂） | 1 | 前臂：旋前/旋后 | [0.02,-0.99,-0.12] | + | - | 43 |
| 23 | **hand-L**（左手） | 2 | 腕：屈伸、桡尺偏 | X,Z | -,- | - | 44-45 |

> **DOF 和动作的对应**：1-DOF = 一个动作（如膝只屈伸）；2-DOF = 两个动作（腕屈伸+侧摆）；3-DOF = 三个动作（如肩可做屈伸+外展+旋转）。

### 3.1b 1-DOF 旋转轴是怎么来的（生物力学来源）

1-DOF 关节（膝/踝/趾/肘/前臂）的旋转轴**来自 OpenSim 生物力学人体模型**（`.osim` 文件，代码注释明确）：

```python
# definition.py 注释:
PinJoint(parent_frame_ori = [0.175895, -0.105208, 0.0186622]),  # talus_r
# "Field taken from .osim Joint-> frames -> PhysicalOffsetFrame -> orientation"
```

四种 1-DOF 轴来源：

| 关节 | 类 | 轴怎么来 |
|---|---|---|
| **膝** (tibia) | `WalkerKnee` | 固定绕全局 Z 轴负向（Walker 膝关节简化模型） |
| **踝/跟/趾** (talus/calcn/toes) | `PinJoint` | 从 `.osim` 的 `parent_frame_ori` 算：`axis = R_euler(ori,'XYZ') @ [0,0,1]`（父系 Z 轴在关节系的方向） |
| **肘** (ulna) | `CustomJoint` | axis=[0.05,0.04,1.0]≈Z（肘屈伸轴，解剖提取） |
| **前臂** (radius) | `CustomJoint` | axis=[-0.02,0.99,-0.12]≈Y（旋前/旋后轴） |

**为什么只有 1 根轴**：这些是人体里的"铰链关节"——膝/踝/趾/肘本质上只绕一根解剖轴转动（屈伸），其余方向被关节结构限制。OpenSim 模型把这根轴精确提取出来。

**代码来源**：
- `data_inputs/body_models/skel/bsm.osim` / `tmp.osim`（OpenSim 模型）
- `lib/body_models/skel_utils/definition.py:77` 注释
- `thirdparty/SKEL/skel/osim_rot.py` `WalkerKnee`/`PinJoint` 实现

### 3.2 46 姿态参数（`pose_param_names`）

| qid | 名称 | 关节 | qid | 名称 | 关节 |
|---|---|---|---|---|---|
| 0 | pelvis_tilt | pelvis | 23 | head_bending | head |
| 1 | pelvis_list | pelvis | 24 | head_extension | head |
| 2 | pelvis_rotation | pelvis | 25 | head_twist | head |
| 3 | hip_flexion_r | femur-R | 26 | scapula_abduction_r | scapula-R |
| 4 | hip_adduction_r | femur-R | 27 | scapula_elevation_r | scapula-R |
| 5 | hip_rotation_r | femur-R | 28 | scapula_upward_rot_r | scapula-R |
| 6 | knee_angle_r | tibia-R | 29 | shoulder_r_x | humerus-R |
| 7 | ankle_angle_r | talus-R | 30 | shoulder_r_y | humerus-R |
| 8 | subtalar_angle_r | calcn-R | 31 | shoulder_r_z | humerus-R |
| 9 | mtp_angle_r | toes-R | 32 | elbow_flexion_r | ulna-R |
| 10 | hip_flexion_l | femur-L | 33 | pro_sup_r | radius-R |
| 11 | hip_adduction_l | femur-L | 34 | wrist_flexion_r | hand-R |
| 12 | hip_rotation_l | femur-L | 35 | wrist_deviation_r | hand-R |
| 13 | knee_angle_l | tibia-L | 36 | scapula_abduction_l | scapula-L |
| 14 | ankle_angle_l | talus-L | 37 | scapula_elevation_l | scapula-L |
| 15 | subtalar_angle_l | calcn-L | 38 | scapula_upward_rot_l | scapula-L |
| 16 | mtp_angle_l | toes-L | 39 | shoulder_l_x | humerus-L |
| 17 | lumbar_bending | lumbar | 40 | shoulder_l_y | humerus-L |
| 18 | lumbar_extension | lumbar | 41 | shoulder_l_z | humerus-L |
| 19 | lumbar_twist | lumbar | 42 | elbow_flexion_l | ulna-L |
| 20 | thorax_bending | thorax | 43 | pro_sup_l | radius-L |
| 21 | thorax_extension | thorax | 44 | wrist_flexion_l | hand-L |
| 22 | thorax_twist | thorax | 45 | wrist_deviation_l | hand-L |

### 3.3 HSMR 44 关节（OpenPose BODY_25 + SMPL extra）

**OpenPose 25（0-24）：**
| idx | 名称 | idx | 名称 | idx | 名称 |
|---|---|---|---|---|---|
| 0 | nose | 9 | R_hip | 18 | L_ear |
| 1 | neck | 10 | R_knee | 19 | L_big_toe |
| 2 | R_shoulder | 11 | R_ankle | 20 | L_small_toe |
| 3 | R_elbow | 12 | L_hip | 21 | L_heel |
| 4 | R_wrist | 13 | L_knee | 22 | R_big_toe |
| 5 | L_shoulder | 14 | L_ankle | 23 | R_small_toe |
| 6 | L_elbow | 15 | R_eye | 24 | R_heel |
| 7 | L_wrist | 16 | L_eye | | |
| 8 | mid_hip | 17 | R_ear | | |

**SMPL extra 19（25-43）：**
| idx | 名称 | idx | 名称 | idx | 名称 |
|---|---|---|---|---|---|
| 25 | R_ankle_s | 32 | R_shoulder_s | 39 | pelvis_mpii |
| 26 | R_knee_s | 33 | L_shoulder_s | 40 | thorax_mpii |
| 27 | R_hip_s | 34 | L_elbow_s | 41 | spine_h36m |
| 28 | L_hip_s | 35 | L_wrist_s | 42 | jaw_h36m |
| 29 | L_knee_s | 36 | neck_lsp | 43 | head_h36m |
| 30 | L_ankle_s | 37 | top_of_head_lsp | | |
| 31 | R_wrist_s | 38 | R_elbow_s | | |

> `joints_44` = OpenPose 25 + SMPL 19。`joints_backup` = SKEL 24 解剖。`joints_custom` = SMPL 风格 24。

---

## 4. 关键文件速查

| 功能 | 文件 |
|---|---|
| 检测 | `lib/modeling/pipelines/vitdet/utils_detectron2.py:66` |
| 裁剪 patch | `lib/kits/hsmr_demo.py:123` `_img_det2patches` |
| HSMR forward | `lib/modeling/pipelines/hsmr.py:243` |
| HSMR forward_step | `lib/modeling/pipelines/hsmr.py:292` |
| ViT | `lib/modeling/networks/backbones/vit.py:307` |
| head | `lib/modeling/networks/heads/skel_head.py:65` |
| 6D→46 | `lib/body_models/skel_utils/transforms.py:365` `params_rep2q` |
| 关节表 | `lib/body_models/skel_utils/definition.py` `JOINTS_DEF` |
| 参数名 | `thirdparty/SKEL/skel/kin_skel.py` `pose_param_names` |
| 相机平移 | `deploy/onnx/hsmr_onnx.py:141` |
| 真实深度融合 | `docs/realtime_depth_3d.py`（含 `--offline` 本地模式） |
