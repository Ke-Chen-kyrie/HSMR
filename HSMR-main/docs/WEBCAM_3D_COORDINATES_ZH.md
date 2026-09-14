# HSMR 摄像头三维坐标与人物朝向

## 先明确：当前输出不是 RealSense/机器人绝对坐标

HSMR 从单张 RGB 图像估计人体、姿态、形状和一个虚拟相机平移。当前程序为了与原图叠加渲染，使用固定虚拟焦距 `fx=fy=5000 px`。因此：

- 人体尺寸、骨骼和关节的单位采用 SKEL 模型米制尺度；
- `X/Y/Z`、距离和深度是单目模型在虚拟相机假设下的估计；
- 它们不是 RealSense 深度测量，也不是 ROS `camera_color_optical_frame`、`base_link`、`odom` 或 `map` 中的标定值；
- 单目 RGB 存在尺度—深度歧义，不应直接把估计的 `Z` 当作机器人安全距离。

## 每帧产生哪些文件

默认 head 启动脚本使用的目录是：

```text
/home/naviai/projects/HSMR-main/data_outputs/webcam_head_onnx/
├── latest.jpg
├── latest.json
├── latest_3d.json
└── frames/
    ├── 000001-时间戳.jpg
    └── 000001-时间戳.3d.json
```

`latest_3d.json` 始终覆盖为最新结果。没有使用 `--no_history` 时，每张历史检测图旁边会保存同名 `.3d.json`。

保存图会在每个检测裁剪框上标出 `P0/P1/...`、检测置信度、骨盆 `xyz`、朝向角和方向分类。颜色和编号与 JSON 中的 `person_index` 一致；编号只表示当前帧检测顺序，不是跨帧跟踪 ID。

## 三个需要区分的坐标概念

### 1. HSMR 模型坐标（平移为零）

JSON 字段：

```text
persons[i].joints.<joint_set>.model_origin_relative_m
```

SKEL 已经应用人物的全局根姿态，但没有应用相机平移。坐标轴与模型虚拟相机一致：

- `+X`：图像右侧；
- `+Y`：图像下方；
- `+Z`：远离相机、进入画面；
- 单位：SKEL 模型米。

特别注意：SKEL 模型原点并不严格等于骨盆关节点，所以不能假设第一个位置是 `[0,0,0]`。程序单独导出了骨盆位置。

### 2. 裁剪块虚拟相机平移

JSON 字段：

```text
persons[i].position.model_origin_crop_virtual_camera_m
```

它就是网络返回的 `pd_cam_t`，对应每个人的 256×256 裁剪块和焦距 5000 px，不适合直接比较原图中处于不同位置/不同框大小的人。

### 3. 全图虚拟相机坐标

JSON 字段：

```text
persons[i].position.model_origin_full_image_virtual_camera_m
persons[i].position.pelvis_full_image_virtual_camera_m
persons[i].joints.<joint_set>.full_image_virtual_camera_m
```

程序按照检测框中心、大小和原图中心，把裁剪块相机平移换算到全图渲染相机。任一关节满足：

```text
p_full_virtual = p_model_origin_relative + model_origin_full_image_virtual_camera_m
```

其中骨盆位置优先读取：

```text
persons[i].position.pelvis_full_image_virtual_camera_m
```

`pelvis_optical_axis_depth_m` 是骨盆的 `Z`，`pelvis_distance_from_virtual_camera_m` 是骨盆到虚拟相机原点的欧氏距离。

## 人物朝向字段

SKEL 的标准人体坐标定义为：`+X` 指向人体左侧、`+Y` 指向头顶、`+Z` 指向人体正前方。程序导出：

```text
persons[i].orientation.skel_root_pose_radians
persons[i].orientation.skel_root_pose_degrees
persons[i].orientation.rotation_matrix_body_canonical_to_model_camera
persons[i].orientation.quaternion_body_canonical_to_model_camera_wxyz
persons[i].orientation.body_forward_unit_model_camera
persons[i].orientation.body_up_unit_model_camera
persons[i].orientation.body_left_unit_model_camera
persons[i].orientation.body_right_unit_model_camera
persons[i].orientation.facing_camera_yaw_deg
persons[i].orientation.forward_elevation_deg
persons[i].orientation.coarse_facing
```

含义如下：

- 根姿态三个参数依次为 `pelvis_tilt`、`pelvis_list`、`pelvis_rotation`；原始单位是弧度；
- 旋转矩阵把标准人体坐标向量变换到 HSMR 模型/虚拟相机轴；
- 四元数顺序明确为 `[w, x, y, z]`；
- `body_forward_unit_model_camera` 是人物胸腹整体的正前方向单位向量；
- `facing_camera_yaw_deg=0°` 表示朝向相机；`+90°` 表示朝图像右侧；`-90°` 表示朝图像左侧；`±180°` 表示背向相机；
- `forward_elevation_deg>0°` 表示正前方向向图像上方抬起；
- `coarse_facing` 是以上精确角度的四分类结果，不应替代角度本身。

这是骨盆/身体根部的整体朝向，不是头部视线或眼睛注视方向。头部方向还需把腰椎、胸椎和头部的局部关节旋转沿运动链组合。

## 三套关节坐标

每个人都有三套数组；名称及下标在顶层 `joint_sets` 中：

- `hsmr_44`：前 25 个是 OpenPose BODY_25，后 19 个来自 `SMPL_to_J19.pkl`；
- `skel_24_anatomical`：SKEL 原生生物力学关节中心；
- `smpl_24_custom`：HSMR wrapper 使用的 SMPL 风格 24 关节。

要找机器人应用中的骨盆、膝、踝等常用点，通常从 `smpl_24_custom` 开始；要分析生物力学骨骼，使用 `skel_24_anatomical`。

检测器原始分数和框位于 `persons[i].detection.score` 与 `persons[i].detection.bbox_left_top_right_bottom_px`；实际送入 HSMR 的方形扩展框位于 `crop_bbox_left_top_right_bottom_px`。当一个小框落在另一个人物框内部，或者单目 `Z` 明显异常时，应结合分数和保存图排除局部/重复误检。

## Python 提取示例

```python
import json

path = '/home/naviai/projects/HSMR-main/data_outputs/webcam_head_onnx/latest_3d.json'
with open(path, 'r', encoding='utf-8') as file:
    frame = json.load(file)

for person in frame['persons']:
    person_id = person['person_index']
    position = person['position']['pelvis_full_image_virtual_camera_m']
    yaw = person['orientation']['facing_camera_yaw_deg']
    direction = person['orientation']['coarse_facing']
    forward = person['orientation']['body_forward_unit_model_camera']
    print(person_id, position, yaw, direction, forward)

    names = frame['joint_sets']['smpl_24_custom']['names']
    joints = person['joints']['smpl_24_custom'][
        'full_image_virtual_camera_m'
    ]
    left_wrist = joints[names.index('left_wrist')]
    print('left_wrist:', left_wrist)
```

命令行快速查看：

```bash
jq '.persons[] | {id: .person_index, position: .position.pelvis_full_image_virtual_camera_m, yaw: .orientation.facing_camera_yaw_deg, facing: .orientation.coarse_facing}' \
  data_outputs/webcam_head_onnx/latest_3d.json
```

## 对应代码位置

- `deploy/onnx/runtime.py`：ONNX 输出转换为 `pd_params` 和裁剪块 `pd_cam_t`；
- `lib/kits/hsmr_demo.py::prepare_mesh`：一次 SKEL forward 中提取 44/24/24 关节，并从 `joints_ori` 去掉固定骨骼初始对齐，得到精确根旋转；
- `lib/kits/hsmr_demo.py::visualize_full_img`：把裁剪块 `pd_cam_t` 换算成全图 `raw_cam_t`；
- `lib/platform/person_3d.py`：定义关节名称、坐标语义、旋转矩阵/四元数、人物转向角和 JSON schema；
- `exp/run_webcam.py::infer_latest_frame`：组装逐人三维记录；
- `exp/run_webcam.py::main`：保存 `latest_3d.json` 和历史 `.3d.json`。

## 如何得到真实 RealSense/机器人坐标

需要同步的深度图、真实彩色相机内参以及相机到机器人基座的外参。仅订阅当前的彩色压缩话题不够。

对与彩色图对齐后的深度 `Z_depth`，像素 `(u,v)` 可反投影到真实相机光学坐标：

```text
X = (u - cx) * Z_depth / fx
Y = (v - cy) * Z_depth / fy
Z = Z_depth
```

然后使用标定外参：

```text
p_base = R_base_camera * p_camera + t_base_camera
R_base_body = R_base_camera * R_camera_body
```

实际实现时应优先使用 RealSense SDK 的对齐与反投影函数，并通过 ROS TF2 查询带时间戳的相机到 `base_link` 变换。HSMR 可提供人体关节拓扑和朝向，真实深度负责确定可靠的位置尺度。
