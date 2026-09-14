# 真实深度 vs HSMR 单目估计 — 验证与接入

> 回答：latest_3d.json 的深度/朝向哪来的？真实 ROS 深度怎么接入？
> 所有结论均来自代码 + 实际桥接探测。

---

## 1. latest_3d.json 的深度是假的（纯单目估计）

**没有用任何深度图。** 代码证据 (`lib/platform/person_3d.py`)：

```python
# person_3d.py:466 构建 pelvis_distance
'pelvis_distance_from_virtual_camera_m': round(distance, 6),
# distance = ||pelvis_full_camera||
# pelvis_full_camera = pelvis_model + full_translation
# full_translation = HSMR raw_cam_t = [tx, ty, 2×5000/(256×scale)]
```

模型自己的 `units_note` 和 `coordinate_frames` 警告明确写了：

```python
'units_note': 'Absolute monocular translation and depth remain scale/focal-length dependent estimates.'
'warning': 'Monocular estimate using a fixed virtual focal length; not RealSense depth'
```

**实测对比（同一画面同一人）：**

| 人 | HSMR 假距离 (latest_3d.json) | 真实深度 (aligned_depth) |
|---|---|---|
| Person0 | 16.0 m | **1.90 m** |
| Person1 | 25.8 m | **1.25 m** |

单目假距离错 8~20 倍，完全不可用于机器人。

---

## 2. Orientation 哪来的

来自 **SKEL 前向运动学的根骨骼全局旋转**（机器人 run_webcam.py 增强版 prepare_mesh）：

```python
# hsmr_demo.py:233-237
root_rest_orientation = pipeline.skel_model.per_joint_rot[0]   # SKEL 固定 rest 对齐
root_rotation = skel_outputs.joints_ori[:, 0] @ root_rest_orientation.T
# joints_ori[:,0] = 前向运动学根骨骼全局朝向; 去掉 rest 对齐
```

然后 `describe_root_orientation` (person_3d.py) 把 canonical 轴旋转到相机系：

```python
body_forward = root_rotation @ [0, 0, 1]                    # 人体前向
camera_yaw = degrees(atan2(body_forward[0], -body_forward[2]))  # 0°=面向相机
coarse_facing: |yaw|≤45° → 'toward_camera'
```

**性质**：这是 HSMR 单张 RGB 推断的人体朝向（反映人相对相机画面的方向），不是 IMU/深度测量。相对几何可用，但坐标系是 HSMR 虚拟相机系。

---

## 3. 机器人实际有的 ROS 深度数据（已实测探测）

Foxglove bridge (192.168.217.100:8768) 广播 37 个主题，实测确认：

| 主题 | 实测值 |
|---|---|
| `.../color/image_raw/compressed` | 720×1280 BGR |
| `.../aligned_depth_to_color/image_raw/compressedDepth` | 720×1280 **uint16 毫米**，PNG 在 offset 12 |
| `.../aligned_depth_to_color/camera_info` | K=[910.68, 0, 653.79; 0, 910.28, 374.08; 0,0,1] |
| `.../color/camera_info` | 与深度同内参（已对齐） |
| `.../extrinsics/depth_to_color` | 深度→彩色外参 |
| `/tf_static` | `HEAD → realsense_head_color_optical_frame` |
| `/tf` | 动态（本次窗口无新增帧） |

**关键结论**：aligned_depth 与 color 已对齐（同内参同分辨率同 frame），可直接用彩色像素索引采样深度。

**⚠️ base_link 问题**：TF 树目前只有 `HEAD → realsense_head_color_optical_frame`，**没有 base_link/odom/map**。机器人没广播完整 TF。所以"base_link 坐标"暂时算不了——需要机器人发布完整 TF（含 head_link→base_link 链）才能转换。

---

## 4. 接入方案（已实现独立脚本）

**文件：`docs/realtime_depth_3d.py`**（不改任何现有代码）

```
订阅: color + aligned_depth(compressedDepth) + camera_info + tf_static + tf
  ↓
HSMR(现有管线) → 每人44关节 + 虚拟相机平移
  ↓
虚拟相机坐标 → 2D像素 u=5000·X/Z+cx, v=5000·Y/Z+cy
  ↓
aligned_depth 采样(中位数窗口) → 真实深度(米)
  ↓
CameraInfo K 反投影 → 相机光学系真实 XYZ(米)
  ↓
TF(optical→HEAD/base_link) → 目标系 XYZ
  ↓
输出 JSON + 3D 渲染
```

**运行**（机器人 GPU 上很快，本地 CPU ~60s/帧）：
```bash
python docs/realtime_depth_3d.py --max_frames 3 --interval 5 --device cuda:0 --backend onnx
```

**输出**：
- `docs/_depth3d_data.json` — 每人真实深度关节 XYZ（光学系 + TF 后）
- `docs/_depth3d_3d_view.png` — 3D 场景（真实米）
- `docs/_depth3d_2d_overlay.png` — HSMR 渲染叠加

**实测验证**（第1次运行，有人入镜时）：
```
[Person0] 骨盆真实深度=1.90m (HSMR单目假距离=16.0m)
[Person1] 骨盆真实深度=1.25m (HSMR单目假距离=25.8m)
```

---

## 5. 数据可信度总表

| 数据 | 来源 | 可信度 |
|---|---|---|
| joints_44_body（根相对骨架） | SKEL 模型 | ✅ 可靠（相对几何） |
| root_rotation / forward / yaw | SKEL 前向运动学 | ✅ 可靠（相对相机） |
| pelvis_distance (latest_3d.json) | HSMR 单目估计 | ❌ 假米（8~20倍误差） |
| 真实深度关节 XYZ | aligned_depth + CameraInfo | ✅ 真实物理米 |
| base_link 坐标 | 需要完整 TF | ⚠️ 目前 TF 不完整 |

---

## 6. 下一步建议

1. **在机器人上**部署 `realtime_depth_3d.py`（GPU + ONNX，快）
2. **让机器人发布完整 TF**（含 head_link → base_link → odom → map），否则 base_link 坐标无解
3. 若运动组要 base_link，需机器人把 `HEAD` 以下的 TF（脖子/头关节到 base_link）都广播出来
4. 深度窗口采样已做中位数滤波；可进一步对关节深度做时间平滑（Kalman）
