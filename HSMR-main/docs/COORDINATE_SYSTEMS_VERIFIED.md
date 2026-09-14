# HSMR 项目坐标系 — 逐项代码验证

> 每个结论都标注了来源：代码文件+行号，或标注「设计文档，未实现」。

---

## 0. 论文来源

| 模型 | 论文 | 出处 |
|---|---|---|
| **HSMR** | *Reconstructing Humans with a Biomechanically Accurate Skeleton*, CVPR 2025 Oral | [arXiv:2503.21751](https://arxiv.org/abs/2503.21751), Xia et al. |
| **SKEL** | *From Skin to Skeleton: Towards Biomechanically Accurate 3D Digital Humans*, SIGGRAPH Asia 2023 | Keller et al., [skel.is.tue.mpg.de](https://skel.is.tue.mpg.de/) |

HSMR 的 camera model 继承自 HMR/SPIN 系列的 weak-perspective 模型（focal length = 5000 是该系列的事实标准）。

---

## 1. SKEL 人体坐标系（✅ 从代码和实际模型提取）

### 1.1 坐标轴方向

从 `SKELWrapper.forward()` rest pose (`poses=0, betas=0`) 提取 24 个解剖关节点：

```
Head (anatomical 13):   [ 0.00, +0.38, -0.01]
Pelvis (anatomical 0):  [ 0.01, -0.23, +0.10]
Nose (OpenPose 0):      [ 0.00, +0.44, +0.14]
R-Shoulder (anat 15):   [-0.17, +0.24,  0.00]
L-Shoulder (anat 20):   [+0.18, +0.24,  0.00]
Toes-R (anatomical 5):  [-0.10, -1.20, +0.12]
Heel-R (anatomical 4):  [-0.11, -1.18, -0.08]
```

**计算方向向量**：

| 向量 | 计算 | 结果 | 含义 |
|---|---|---|---|
| 头−骨盆 | `[0.00,0.38,−0.01] − [0.01,−0.23,0.10]` | `[−0.01, **+0.61**, −0.11]` | **+Y = 上** |
| 鼻−骨盆 Z | `0.14 − 0.10` | `**+0.04**` | 鼻子在骨盆前方 |
| 脚尖−脚跟 Z | `0.12 − (−0.08)` | `**+0.20**` | 脚尖在脚跟前方 |
| R肩−L肩 X | `−0.17 − 0.18` | `**−0.35**` | 右肩在左肩左边（−X） |

**结论**：

```
+X = 身体左侧（左臂在 +X，右臂在 −X）
+Y = 身体上方（头在 +Y，脚在 −Y）
+Z = 身体前方（人体面朝 +Z，即面向观察者/相机）
```

⚠️ **关键点**：右臂在 −X，左臂在 +X。这是"观察者看人体"的视角——人体面向你(+Z)，你的右边就是他的左边(+X)。

**代码依据**：
- `transforms.py:98` 注释：*"the 'left & right' defined when the body is facing z+ direction"*
- `kin_skel.py:163-167` 注释确认 bone space axis 0=front-back, 1=up-down, 2=left-right

### 1.2 骨盆 ≠ 原点

```
Pelvis: [0.01, −0.23, +0.10]
Origin: [0.00,  0.00,  0.00]
```

SKEL 原点在骨盆**上方**约 0.23m（模板网格质心在 Y=0，骨盆在 Y=−0.23）。原点大致在肚脐/腰部位置。这个偏移来自 SKEL 模型模板网格的定义——网格顶点在设计时已居中。

### 1.3 关节运动学树

24 个关节，46 个姿态参数（`definition.py:73-98`）：

```
pelvis (root)
├── femur_r → tibia_r → talus_r → calcn_r → toes_r
├── femur_l → tibia_l → talus_l → calcn_l → toes_l
└── lumbar → thorax → head
              ├── scapula_r → humerus_r → ulna_r → radius_r → hand_r
              └── scapula_l → humerus_l → ulna_l → radius_l → hand_l
```

父-子关系从 `skel_male.pkl` 的 `osim_kintree_table` 提取（agent 验证）。

### 1.4 Orientation（身体朝向）定义在哪里？

身体朝向分**三个层次**编码在模型中：

**Level 1: 模板网格几何（最根本的来源）**

`skel_male.pkl` 中的 `skin_template_v` 顶点位置**已经**将人体摆放为面朝 +Z。这是朝向的最根本来源。验证：

```
网格质心: Y=0.000000 (恰好居中，设计使然)
鼻子 Z=+0.14 > 骨盆 Z=+0.10 → 鼻子在骨盆前方
脚尖 Z=+0.12 > 脚跟 Z=−0.08 → 脚尖在前
```

**Level 2: 关节回归矩阵（继承模板朝向）**

`J_regressor_osim` 从模板顶点回归关节位置。回归出的关节继承了模板的朝向。所有 24 个解剖关节的 Z 值分布与模板一致。

**Level 3: 代码约定（文档化朝向）**

- `transforms.py:98` 注释明确写："left & right defined when the body is facing z+ direction"
- `kin_skel.py:163-167` 注释：bone space 轴定义 "0: front-back, 1: up-down, 2: left-right"
- `definition.py:74`: `pelvis axis=[[0,0,1],[1,0,0],[0,1,0]]` → Z 轴=前后轴

**Level 4: 旋转编码**

Pelvis 关节旋转轴 `axis=[[0,0,1],[1,0,0],[0,1,0]]`（convention YXZ）：
- `axis[0] = [0,0,1]` = Z → pelvis_tilt（绕前后轴侧倾）
- `axis[1] = [1,0,0]` = X → pelvis_list（绕左右轴前后倾）
- `axis[2] = [0,1,0]` = Y → pelvis_rotation（绕上下轴旋转）

在 rest pose (q=[0,0,0])：R = R_Y(0)×R_X(0)×R_Z(0) = I（单位矩阵），身体面朝 +Z。

### 1.5 Pelvis 旋转轴

`definition.py:74`：
```python
CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1])
```

旋转顺序（`osim_rot.py:23-33` `q_to_rot` 从左乘积累）：
```
R = R_Y(θ₂) × R_X(θ₁) × R_Z(θ₀)

θ₀ = pelvis_tilt  → 绕 Z 轴（前后轴）→ 侧倾
θ₁ = pelvis_list  → 绕 X 轴（左右轴）→ 前后倾
θ₂ = pelvis_rotation → 绕 Y 轴（上下轴）→ 旋转
```

Convention (from `_DOF3_CONVENTIONS[0]`): YXZ, flip=[1,1,1]

---

## 2. HSMR 虚拟相机（✅ 从代码提取）

### 2.1 为什么叫"虚拟"相机

HSMR 虚拟相机**不是物理设备**。它是为了让三维人体投影与二维 RGB 图像对齐而建立的数学针孔模型。类比 Blender：先在 3D 场景中生成人体，再放置一台数学相机，使渲染轮廓与图片中的人重合。

### 2.2 Camera 输出

HSMR 核心网络输出（`hsmr_core.py:61`）：
```
camera = [scale, tx, ty]  # shape [B, 3]，weak-perspective 参数
```

### 2.3 camera_translation 公式

`hsmr_core.py:123-132` 和 `hsmr_onnx.py:141-148`（两者一致）：

```python
focal = 5000.0
patch_size = 256

camera_translation = [
    camera[:, 1],                                              # tx → X 平移
    camera[:, 2],                                              # ty → Y 平移
    2.0 * focal / (patch_size * camera[:, 0] + 1e-9),         # tz → Z 深度（从 scale 反算）
]
```

**精确公式**：`tz = 2 × 5000 / (256 × scale)`

- scale 越大 → tz 越小 → 人离相机越近
- scale 越小 → tz 越大 → 人离相机越远
- tx 控制人体在图像中的左右偏移
- ty 控制人体在图像中的上下偏移
- **没有任何正负号翻转**

### 2.4 虚拟相机坐标

`hsmr_core.py:146-148`（注释在 143-145）：
```python
# HSMR's body joints are root-relative. Adding the model-estimated
# translation gives HSMR camera coordinates. These are not a calibrated
# ROS optical frame and must not be treated as robot base coordinates.
joints_44_camera = joints_44_body + camera_translation[:, None, :]
```

### 2.5 投影公式

`camera.py:105-154` `perspective_projection`：
```python
# R = I (identity，默认)
points_camera = points + translation         # 3D 点平移到相机前方
u_norm = X_camera / Z_camera                 # 归一化坐标
v_norm = Y_camera / Z_camera
u_pixel = fx * u_norm + cx                   # 像素坐标
v_pixel = fy * v_norm + cy
```

其中 `fx = fy = focal / patch_size = 5000/256 ≈ 19.53`, `cx = 0, cy = 0`（在归一化坐标中，主点在原点）。

转为 patch 像素坐标（`hsmr_core.py:154-156`）：
```python
joints_2d_patch = (joints_2d_normalized + 0.5) * 256
# 等价于：u_patch = 5000 * X / Z + 128
#        v_patch = 5000 * Y / Z + 128
```

### 2.6 为什么 f=5000

这是 HMR/SPIN 系列方法的标准焦距。它来自 SMPL 训练数据的经验值——用这个焦距值和弱透视假设，可以在裁剪后的人物 patch（256×256）上获得合理的投影对齐。f=5000 使 `f/patch_size ≈ 19.5`，即归一化焦距。

### 2.7 pd_cam_t → raw_cam_t

`hsmr_demo.py:255-257` `visualize_full_img`：
```python
raw_cam_t_z = pd_cam_t_z * 256 / bbox_size          # 缩放修正
raw_cam_t_y += (bbox_cy - image_cy) / 5000 * raw_cam_t_z  # 中心偏移修正
raw_cam_t_x += (bbox_cx - image_cx) / 5000 * raw_cam_t_z
```

将 patch 虚拟相机平移映射回完整图像。仍然是虚拟相机坐标，**不涉及** RealSense。

### 2.8 渲染管线中的坐标转换

pyrender 渲染时有两个额外转换（`py_renderer/__init__.py:319-335` + `utils.py:13`）：

1. **cam_t[0] *= −1**（`utils.py:13`）：X 分量取反
2. **mesh 旋转 180° 绕 X 轴**（`__init__.py:332-335`）：Y 和 Z 取反

```
mesh.apply_transform(translation_matrix(cam_t))   # 先平移
mesh.apply_transform(rotation_matrix(180°, [1,0,0]))  # 再旋转
```

`R_180X = diag(1, -1, -1)` 的效果：
- 人体 +Z（面向相机）→ −Z（面向 OpenGL 相机方向）
- 人体 +Y（上方）→ −Y（在 OpenGL 图像中仍映射到上方）
- 最终图像中人体正向站立

---

## 3. RealSense 相机（⚠️ 设计文档，代码中仅订阅了彩色图）

### 3.1 当前代码实际订阅的内容

`run_webcam.py:19-25`：
```python
DEFAULT_CAMERA_TOPIC_HEAD = '/zj_humanoid/sensor/realsense_head/color/image_raw/compressed'
```

`foxglove_camera.py` 只做三件事：
- 通过 Foxglove WebSocket 订阅压缩图像
- `cv2.imdecode` 解码为 BGR
- 用 `time.time()` 记录接收时间（**不是 ROS header.stamp！**）

**没有订阅**：深度图、CameraInfo、`/tf`、`/tf_static`

### 3.2 设计文档中的规划

ROS3D 文档 §17-21 描述了未来集成方案：
- 对齐深度图（`aligned_depth_to_color`）
- 从 `CameraInfo` 读取真实内参 `K = [fx, 0, cx; 0, fy, cy; 0, 0, 1]`
- **反投影公式** §20：`X = (u − cx) × Z / fx`, `Y = (v − cy) × Z / fy`
- ROS optical frame：**X 右，Y 下，Z 前**

### 3.3 为什么虚拟相机 ≠ RealSense

| 项目 | HSMR 虚拟相机 | RealSense 真实相机 |
|---|---|---|
| 物理设备 | 否 | 是 |
| 焦距 | 固定 5000 | CameraInfo.K 标定值 |
| 畸变 | 无 | CameraInfo.D 描述 |
| 深度 | 从 RGB 估计（单位模糊） | 深度传感器逐像素测量（米） |
| 坐标原点 | 裁剪 patch 中心 | 相机光心 |
| 与机器人关系 | 无连接 | 可通过 TF 转换到 base_link |

---

## 4. Robot base_link（⚠️ 设计文档，代码中未实现）

### 4.1 TF 树（来自设计文档 §21）

```
camera_optical_frame → camera_link → head_link → base_link → odom → map
```

### 4.2 坐标轴

ROS `base_link` 标准（REP 103）：
- **+X = 机器人前方** (Forward)
- **+Y = 机器人左侧** (Left)
- **+Z = 机器人上方** (Up)

与相机 optical frame (X 右, Y 下, Z 前) **完全不同**！

### 4.3 变换公式（设计文档 §21.3-21.5）

```python
# 从 optical frame 到 base_link
p_base = R_base_optical @ p_optical + t_base_optical

# 使用图像采集时刻的 TF（不是查询时刻！）
transform = tf_buffer.lookup_transform(
    "base_link",           # target
    camera_info.frame_id,  # source (optical frame)
    stamp=image_stamp,     # 图像时间戳
)
```

### 4.4 Orin README 中的警告

`deploy/orin/README.md` §6：
> "Do not send HSMR output directly to the Pico x86 real-time controller...
> convert from the optical-camera frame to the robot base_link frame."

---

## 5. 关键误区总结

| 误区 | 事实 |
|---|---|
| 人体 +X = 右侧 | ❌ 右臂在 −X！+X = 身体左侧 |
| 骨盆 = 原点 | ❌ 骨盆在 `[0.01, −0.23, 0.10]` |
| pd_cam_t / raw_cam_t = RealSense 坐标 | ❌ 是 HSMR 虚拟相机坐标 |
| 虚拟相机和 RealSense 已连接 | ❌ 当前代码未接入 RealSense 深度/CameraInfo/TF |
| `poses[:,0:3]` = roll/pitch/yaw | ❌ 是 SKEL 的 YXZ Euler-like 参数 |
| f=5000 是 RealSense 的焦距 | ❌ 是 HSMR 内部分配的经验固定值 |

---

## 6. 数据来源一览

| 数据 | 来源 |
|---|---|
| SKEL 关节位置 | `SKELWrapper.forward()` rest pose 提取 |
| 关节点名称和父子关系 | `definition.py` + `kin_skel.py` + `skel_male.pkl` |
| 关节轴和 flip | `definition.py:73-98` JOINTS_DEF |
| camera_translation 公式 | `hsmr_core.py:123-132` / `hsmr_onnx.py:141-148` |
| perspective_projection | `camera.py:105-154` |
| pyrender 180° X 旋转 | `py_renderer/__init__.py:332-335` |
| pyrender cam_t[0] 翻转 | `py_renderer/utils.py:13` |
| f=5000 来源 | `hsmr_core.py:99` / `hsmr_onnx.py:99` / HMR 系列论文 |
| RealSense topic | `run_webcam.py:19-25` |
| Robot TF 树 | `docs/HSMR_ONNX_PIPELINE_ROS3D_ZH.md` §21 |
| HSMR 论文 | CVPR 2025 Oral, arXiv:2503.21751 |
| SKEL 论文 | SIGGRAPH Asia 2023, Keller et al. |
