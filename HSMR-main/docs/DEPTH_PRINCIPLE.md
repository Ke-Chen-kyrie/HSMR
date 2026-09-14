# HSMR → 机器人 BASE 完整流程（导航用）

> 所有数值来自机器人实跑。整条链路已打通：RGB → BASE 坐标 + 完整旋转矩阵。

---

## 1. 整体流程（一个部分看全）

```
┌─────────────────────────────────────────────────────────────────┐
│  RGB 彩色图                                                       │
│    │                                                             │
│    ▼                                                             │
│  ① HSMR 虚拟相机 (f=5000, 假深)                                   │
│     · 网络输出 scale/tx/ty → 2D 关节像素 (投影对齐)               │
│     · tz=2×5000/(256×scale) ← 假米, 不能用于测距                 │
│    │                                                             │
│    ▼                                                             │
│  ② aligned_depth (真实深度·毫米)                                  │
│     · 在 2D 关节像素 (u,v) 处采样 → Z(米)                        │
│     · 深度单位 uint16 毫米 → /1000                                │
│    │                                                             │
│    ▼                                                             │
│  ③ 反投影 (真实 K=910.68)                                         │
│     · X=(u-cx)Z/fx, Y=(v-cy)Z/fy, Z=Z                            │
│     · 得到 color_optical 真实 3D (米)                            │
│    │                                                             │
│    ▼                                                             │
│  ④ TF 转换                                                       │
│     · optical → HEAD → NECK → WAIST_PITCH → WAIST_YAW →          │
│                KNEE → ANKLE → BASE                                │
│     · p_base = T_base_optical @ p_optical                        │
│    │                                                             │
│    ▼                                                             │
│  输出: 44关节 BASE坐标(米) + 旋转矩阵(3×3) + 四元数 ✅             │
└─────────────────────────────────────────────────────────────────┘
```

**核心思想**：
- **虚拟相机算 2D**（投影对齐，网络训练好的，f=5000 不能改）
- **真实深度 + 真实 K 反投影算 3D**（测距）
- **TF 转 BASE**（机器人坐标）

---

## 2. 每步详解

### ① HSMR 虚拟相机 → 2D 关节像素

```python
# camera_translation (hsmr_onnx.py:141)
camera_translation = [tx, ty, 2*5000/(256*scale)]
# 2D 像素 (realtime_depth_3d.py)
u = 5000 * X_cam / Z_cam + cx
v = 5000 * Y_cam / Z_cam + cy
```

**为什么虚拟相机参数不能改**（实测）：网络在 f=5000 训练，改成真实 K 会导致 2D 关节**平均错位 21px（12%）**。虚拟相机只做投影对齐，真实测距交给深度图。

### ② aligned_depth 采样真实深度

```python
# compressedDepth 格式 (实测): [12字节头] + [PNG uint16 毫米]
depth = cv2.imdecode(data[12:], cv2.IMREAD_UNCHANGED)  # 720×1280 uint16
# 关节处小窗口取中位数 (抗噪)
z = median(depth[v-r:v+r, u-r:u+r]) / 1000.0   # 毫米 → 米
```

### ③ 反投影 → color_optical

```python
# 真实 CameraInfo K (实测)
K = [fx=910.68, 0, cx=653.79; 0, fy=910.28, cy=374.08; 0,0,1]
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z
```

### ④ TF → BASE

```python
# 完整 TF 树 (实测): world→root→BASE→ANKLE→KNEE→WAIST_YAW→
#   WAIST_PITCH→NECK→HEAD→realsense_head_color_optical_frame
# optical→BASE 路径 (自动 BFS):
#   optical → HEAD → NECK → WAIST_PITCH → WAIST_YAW → KNEE → ANKLE → BASE
T_base_optical = T_base_head @ T_head_neck @ ... @ T_head_optical
p_base = T_base_optical @ p_optical
```

---

## 3. 实测数据（P0，latest_3d.json）

```json
"pelvis_optical_m": [0.278, 0.068, 1.703],   // 相机前方1.7m (真实深度)
"pelvis_target_m":  [0.318, 1.577, 0.380],   // BASE系 (已TF转换)
"pelvis_depth_m": 1.73,                       // 距相机
"orientation": {
  "rotation_matrix": [[-0.510,-0.336,-0.792],[0.041,-0.929,0.367],[-0.859,0.155,0.487]],
  "quaternion_wxyz": [0.110,-0.483,0.153,0.855],
  "body_forward":    [-0.792, 0.367, 0.487]
}
"joints_target_m": [44关节 BASE坐标]           // 42/44 深度有效
"tf": {"target_frame": "BASE", "applied": true}
```

> **用完整旋转矩阵 + 四元数**（3×3 全姿态），**不用 yaw**（丢俯仰/侧倾）。

---

## 4. 导航怎么用

```
真实位置: 骨盆 [0.32, 1.58, 0.38]m (BASE系)   ← 真实深度 + TF
完整朝向: 旋转矩阵 R(3×3) + 四元数            ← 人相对机器人的朝向
```

- **深度+BASE坐标** → 避障、跟随、导航定位
- **旋转矩阵** → 判断人面朝/背对/侧对机器人
- 两者结合 → 完整的人体状态

---

## 5. 重要提醒

1. **不用 HSMR 假距离**（`pelvis_distance_from_virtual_camera_m` 实测 12.8m vs 真实 1.7m）
2. **真实深度** = aligned_depth 反投影 + 真实 K
3. **虚拟相机参数不能改**（网络训练假设 f=5000，改会错位 12%）
4. **BASE 坐标**需完整 TF 树（optical→HEAD→…→BASE），已打通
5. **深度单位** uint16 毫米 → /1000；0/NaN/越界判无效

---

## 6. 部署成果（机器人已跑通）

机器人在 `/home/naviai/projects/HSMR-main/docs/`：
- `realtime_depth_3d.py` — **run_webcam 模式**：每隔几秒采样，保存 latest.jpg / latest_depth.png / latest_3d.json（BASE+旋转矩阵）/ latest_comprehensive.png
- `base_link_demo.py` — 验证 optical→BASE TF 链
- `test_depth_validation.py` — 深度自洽测试
- `run_all_tests.sh` — 一键 3 项测试

运行命令（机器人上）：
```bash
cd /home/naviai/projects/HSMR-main
.venv_orin/bin/python docs/realtime_depth_3d.py \
  --device cuda:0 --backend onnx --interval 5 --out data_outputs/depth3d
```

输出（`data_outputs/depth3d/`）：latest.jpg（检测图）、latest_depth.png（深度图）、latest_3d.json（BASE坐标+旋转矩阵）、latest_comprehensive.png（综合4面板）、frames/（历史）
