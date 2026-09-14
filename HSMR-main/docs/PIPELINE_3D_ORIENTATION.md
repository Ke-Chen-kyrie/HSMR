# HSMR → 机器人 3D 位置 + Orientation 完整链路

> 目标：机器人通过 RealSense 摄像头 → 判断人的 3D 位置 + 朝向（给运动规划用）

---

## 0. 当前状态 vs 目标状态

```
当前 (已实现):
  RGB图 → ViTDet检测 → HSMR姿态 → SKEL骨骼网格 → OpenGL渲染
  输出: 骨骼渲染图（画在RGB上）

目标 (需要实现):
  RGB图 + 深度图 + CameraInfo + TF
    → 人体3D关节位置 (base_link坐标, 米)
    → 人体朝向 (base_link坐标, yaw角度)
  输出: 44个关节的 (x,y,z) + 根旋转矩阵
```

---

## 1. 虚拟相机的真正作用

虚拟相机**不是**为了输出 3D 位置。它的唯一作用：**让 3D 人体投影和 2D 图像对齐**。

```
网络做的事情：
  "我看到一个占据图像 60% 高度的人"
  → scale 参数 → tz ≈ 2×5000/(256×scale)
  → camera_translation = [tx, ty, tz]
  → 把人体放在相机前方 tz 处，偏移 (tx, ty)
  → 渲染出来刚好跟图像里的人重合

虚拟相机不能做：
  ✗ 告诉你人离机器人多少米（tz 不是真实米制深度）
  ✗ 告诉你人在 base_link 的坐标
  ✗ 标定 camera→robot 的外参
```

**但虚拟相机有一个重要用途**：`joints_44_body` 是**根相对**的 3D 关节位置。如果知道了根的真实深度（从深度图），就可以用这些相对位置重建完整 3D 骨架。

---

## 2. Orientation：怎么得到人的朝向

### 2.1 从 poses 提取根旋转矩阵

```python
from lib.body_models.skel_utils.transforms import params_q2rot

poses = outputs["pd_params"]["poses"]          # [N, 46]
rotations = params_q2rot(poses)                # [N, 24, 3, 3]
root_rotation = rotations[:, 0]                # [N, 3, 3]  ← 这就是人体朝向
```

### 2.2 人体前方向量

SKEL 人体局部前方 = +Z（已验证）：

```python
forward_local = np.array([0.0, 0.0, 1.0])      # SKEL body forward
forward_world = root_rotation @ forward_local   # 人体前方在世界坐标
```

### 2.3 验证数据

| pelvis_rotation | 人体前方向量 | 含义 |
|---|---|---|
| 0° | `[ 0, 0, +1]` | 面朝 +Z（默认前方） |
| +45° | `[+0.71, 0, +0.71]` | 左转 45° |
| +90° | `[+1, 0, 0]` | 左转 90°，面朝 +X |
| +180° | `[ 0, 0, -1]` | 转身 180°，背对 |
| −90° | `[-1, 0, 0]` | 右转 90°，面朝 −X |

### 2.4 转到机器人 base_link

```python
# Step 1: HSMR → RealSense optical（需标定！）
R_optical_hsmr = ???  # 待验证：HSMR 坐标轴 → 相机光学 coordinate

# Step 2: Camera optical → base_link（从 TF 获取）
R_base_optical = tf_transform.rotation  # 从 lookup_transform 获取

# Step 3: 合成
R_base_person = R_base_optical @ R_optical_hsmr @ root_rotation

# Step 4: 前方向量 + yaw
forward_base = R_base_person @ np.array([0, 0, 1])
forward_base /= np.linalg.norm(forward_base)
yaw = np.arctan2(forward_base[1], forward_base[0])  # rad
```

---

## 3. 3D 位置：两种策略

### 策略 A：逐关节深度采样（推荐先实现）

```python
# 1. 从 HSMR 获取 2D 关节像素
joints_2d_patch  # [N, 44, 2] — 在 256×256 patch 上的像素

# 2. 映射回原始彩色图
u_color = bbox_cx + (u_patch - 128) * bbox_size / 256
v_color = bbox_cy + (v_patch - 128) * bbox_size / 256

# 3. 读取对齐深度（小窗口取中位数）
depth_values = aligned_depth[v0:v1, u0:u1]
Z_meters = np.median(depth_values[depth_values > 0])

# 4. CameraInfo 反投影
fx, fy = camera_info.K[0], camera_info.K[4]
cx, cy = camera_info.K[2], camera_info.K[5]
X = (u_color - cx) * Z_meters / fx
Y = (v_color - cy) * Z_meters / fy
# → [X, Y, Z_meters] in camera optical frame

# 5. TF 变换
p_base = R_base_optical @ [X, Y, Z_meters] + t_base_optical
```

优点：每个关节都有真实深度测量
缺点：遮挡关节深度不可靠

### 策略 B：根深度 + 骨架（更鲁棒）

```python
# 只取骨盆/髋部区域的深度（更稳定）
root_depth = sample_depth(pelvis_pixel, aligned_depth)

# 反投影根位置
root_optical = deproject(pelvis_pixel, root_depth, camera_info)

# 用 HSMR 的根相对关节
joints_base = root_base + joints_44_body  # 需要尺度对齐
```

优点：骨架长度一致，遮挡关节也能估计
缺点：需要验证 HSMR body 尺度和 RealSense 尺度一致

---

## 4. 当前代码需要添加的内容

```python
# 需要订阅的新 topic：
# /.../aligned_depth_to_color/image_raw    ← 对齐深度图
# /.../color/camera_info                   ← 相机内参
# /tf, /tf_static                          ← 坐标变换

# 新依赖：
# tf2_ros (Buffer, TransformListener)
# image_geometry (PinholeCameraModel)

# 在 infer_latest_frame() 中添加：
def infer_latest_frame_with_depth(self, color_msg, depth_msg, camera_info):
    # 现有: detector → HSMR → SKEL
    outputs = pipeline(patches_normalized)
    
    # 新: 2D关节 → 深度 → 3D
    joints_3d_optical = []
    for i in range(44):
        u, v = joints_2d_color[i]
        z = sample_depth(depth_msg, u, v)
        xyz = deproject(u, v, z, camera_info)
        joints_3d_optical.append(xyz)
    
    # 新: TF → base_link
    transform = tf_buffer.lookup_transform(
        "base_link", camera_info.header.frame_id, color_msg.header.stamp)
    joints_3d_base = transform_points(joints_3d_optical, transform)
    
    # 新: Orientation
    root_rotation = params_q2rot(outputs["pd_params"]["poses"])[:, 0]
    # ... convert to base_link
    
    return {
        "joints_3d_base": joints_3d_base,      # [44, 3] 米
        "root_rotation_base": root_rotation_base,  # [3, 3]
        "forward_vector": forward_base,         # [3]
        "yaw_rad": yaw,
    }
```

## 5. 待标定项

| 项目 | 说明 | 标定方法 |
|---|---|---|
| `R_optical_hsmr` | HSMR 坐标轴 → RealSense optical | 用正面/背面/左侧/右侧样本验证 |
| 深度尺度 | HSMR body 尺度 vs 真实米 | 对比关节间距离 |
| TF 外参 | camera→base_link | 从 URDF/机器人标定文件获取 |
