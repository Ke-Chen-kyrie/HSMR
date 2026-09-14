# HSMR-sdk

HSMR 的**外围能力 SDK** — 采集 / TF 坐标变换 / 声纹+人脸验证 / ROS 发布。

**纯外围、吃数据**: SDK 不碰模型推理, 入参是 帧/深度/persons/音频, 输出是坐标、验证结果、发布动作。
persons 数据由外部提供 (典型来源: [HSMR-infer](../HSMR-infer) 的 `/infer` 输出)。

| 模块 | 能力 | 依赖 |
|---|---|---|
| `hsmr_sdk.capture` | Foxglove 多话题采集 (彩色/深度/内参/TF) + 帧解码 | websockets, rosbags |
| `hsmr_sdk.infer` | 帧 → HSMR-infer 容器 `/infer` → persons (一条龙取 24 关节) | requests |
| `hsmr_sdk.tf` | 光学系→BASE/HEAD/world 坐标变换 (TF 优先 + HEAD 兜底) | numpy |
| `hsmr_sdk.face` | 声纹 + 人脸交叉验证 (复用 user-identification 容器 :8001) | requests |
| `hsmr_sdk.ros` | zenoh→Foxglove 关节 MarkerArray + 标注图发布 | zenoh_ros2_sdk (可选) |
| `hsmr_sdk.pipeline` | `HsmrSdk` 门面: 采集→推理→TF→验证→发布 一条龙 | 上面各模块 |

依赖 `requirements.txt` 即可装; `zenoh_ros2_sdk` 缺包只禁用发布, 其余功能不受影响。

---

# 整体架构

```
[相机话题] ──帧──> 容器 HSMR-infer :8010 ──persons──> HSMR-sdk
 [foxglove bridge]   (/infer 推理出 24 关节)             ├─ tf   光学→BASE/HEAD/world 变换
         ↑                                                 ├─ face 声纹+人脸交叉验证 (:8001)
         └──────────────────采集──────────────────────────┤
                                                          └─ ros  关节 MarkerArray 发布
```

**分工**: 容器只做推理 (图 → 24 关节 3D), SDK 只做外围 (采集/TF/验证/发布)。
一条完整的实时链路 = **容器出 persons + SDK 做外围**。

---

# 一、容器 HSMR-infer (端口 8010)

推理服务。`/infer` 进彩色图 (+可选深度) → 出 24 个 SKEL 关节的光学系 3D 坐标/朝向。

## 部署

```bash
# 每台机器人: 一键脚本 (幂等, 自动清旧容器/宿主进程/等模型加载)
scp deploy.sh docker-compose.yml 机器人:/home/naviai/projects/hsmr-infer-image/
ssh 机器人 'cd /home/naviai/projects/hsmr-infer-image && bash deploy.sh'
```

- 镜像: `guopeilin-registry.cn-hangzhou.cr.aliyuncs.com/erban_agent/hsmr-infer:20260814`
- 网络: host 网络 (直绑机器人 IP), `restart: unless-stopped` 兜底被杀
- 前提: docker + nvidia-container-toolkit

## 接口

### `GET /health`

```bash
curl http://127.0.0.1:8010/health
# {"status":"ok","model_loaded":true,"device":"cuda:0"}   ← model_loaded:true 才算就绪
```

### `GET /joints`

24 关节名清单: `{"joints": [...], "count": 24}`。索引 `0`=骨盆, `13`=头。

### `POST /infer`

multipart/form-data:

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | 文件 | 彩色图 jpg/png/bmp |
| `depth` | 文件 | **原始深度** (见下), 可选 |
| `fx/fy/cx/cy` | form | 相机内参, 给 depth 时必须一起给 |

**⚠️ 深度格式坑**: 只收**原始深度 uint16 mm** (16UC1 PNG 或 uint16 ndarray 编码的 PNG)。
不要传 3 通道/uint8 的彩色可视化深度图 (如 `depth.png`) — 会被显式拒绝。

**响应**:

```json
{
  "num_persons": 3,
  "timestamp": "...",
  "depth_fused": true,
  "infer_ms": 2200.5,
  "persons": [
    {
      "joints_2d":        [[628.3, 219.6], ...],          // (24,2) 彩色图像素
      "joints_optical_m": [[-0.008, 0.261, 1.189], ...],  // (24,3) 光学系米坐标, 无效关节为 null
      "joints_ori":       [[[... 3x3 ...], ...]],         // (24,3,3) SKEL 朝向 (X=右 Y=上 Z=前)
      "depth_valid":      [true, true, false, ...],       // [24] 逐关节深度有效
      "pelvis_pose":      {"position_cam": [...], "rotation_matrix": [...], "pose_4x4": [...]}
    }
  ]
}
```

无深度融合时 `joints_optical_m` 全为 `null` (NaN→null, 契约含义不变: null=无效)。

### Python 调用示例

```python
import cv2, numpy as np, requests

SERVER = "http://127.0.0.1:8010"   # 本机; 远程机器人换成它的 IP

rgb   = cv2.imread("rgb.jpg", cv2.IMREAD_COLOR)
depth = np.load("depth_raw.npy")    # uint16 mm
K     = {"fx": 910.68, "fy": 910.28, "cx": 653.79, "cy": 374.08}

r = requests.post(f"{SERVER}/infer", timeout=180,
    files={"image": ("rgb.jpg", cv2.imencode(".jpg", rgb)[1].tobytes(), "image/jpeg"),
           "depth": ("depth.png", cv2.imencode(".png", depth)[1].tobytes(), "image/png")},
    data=K)
r.raise_for_status()
persons = r.json()["persons"]      # 直接可喂给 SDK
```

---

# 二、HSMR-sdk

## 安装

```bash
# 机器人上复用现有 venv (不用新建):
/home/naviai/projects/HSMR-main/.venv_orin/bin/pip install -e /home/naviai/projects/HSMR-sdk
# 或纯依赖:
/home/naviai/projects/HSMR-main/.venv_orin/bin/pip install -r /home/naviai/projects/HSMR-sdk/requirements.txt
```

## 配置 (config.yaml)

| 段 | 关键字段 | 默认 | 何时改 |
|---|---|---|---|
| `bridge` | `url` | `ws://localhost:8768` | bridge 在别的机器才改 |
| `topics` | `prefix` | `/zj_humanoid/sensor/realsense_head` | 相机话题名不同才改 |
| `frames` | `optical` / `head` | `realsense_head_color_optical_frame` / `HEAD` | 坐标系名不同才改 |
| `head_to_optical_fallback` | 4×4 | 已验证兜底矩阵 | TF 未到时 HEAD 用的兜底, 通常不动 |
| `ros_publish` | `enabled/topic/router_ip/router_port/publish_all/publish_joints/annotated_*` | 开 | 想发全部关节改 `publish_all: true` |
| `face_verify` | `enabled/service_url/voice_top_k/face_max_face_num/assoc_max_dist` | 开 | 验证容器不在 127.0.0.1:8001 才改 |
| `infer` | `enabled/url/timeout/max_instances` | 开 | HSMR-infer 容器不在 127.0.0.1:8010 才改 |

## 核心用法 (HsmrSdk 门面)

```python
from hsmr_sdk import HsmrSdk, load_config

cfg = load_config()                 # 默认读工程根 config.yaml (可用 HSMR_SDK_CONFIG 覆盖)
sdk = HsmrSdk(cfg).start_capture()  # 阻塞到首帧(彩色+深度)到达或超时(warmup=8s, 可调)
                                    # bridge 订阅需先等 ~5s advertise 窗口, 所以默认帮你等

# ① 采集一帧 → 解码 (彩色/深度/内参), 同时更新 TF 缓存
frame = sdk.capture_once()          # {rgb_bgr, depth_uint16, K, tf_msgs, raw_snapshot}

# ② 推理: SDK 直连 HSMR-infer 容器 /infer (config.infer.url), 帧 → persons
persons = sdk.infer_persons(frame)   # 容器未起/超时抛 InferError
#    也可以不传 frame, 显式给 rgb/depth/K:
#    persons = sdk.infer_persons(rgb_bgr=frame["rgb_bgr"], depth_uint16=frame["depth_uint16"], K=frame["K"])

# ③ 声纹+人脸交叉验证 → 说话人 3D 位置/朝向 (目标系)
resp = sdk.identify(audio_bytes, frame["rgb_bgr"], persons,
                    target="BASE", include_pelvis_pose=True, verbose=True)
# resp: {verified, user_id, name, voice_score, face_score, face_location,
#        person_index, frame, position_m, rotation_matrix, depth_valid,
#        head_position_m, head_rotation_matrix, head_depth_valid, pelvis_pose, ...}

# ④ 把说话人的 24 关节发布到 Foxglove (MarkerArray + 标注图)
joints = sdk.person_joints_payload(persons[resp["person_index"]], target="BASE")
sdk.publish(resp["frame"], joints, annotated_bgr=frame["rgb_bgr"])
```

### 现成示例

```bash
cd /home/naviai/projects/HSMR-sdk
# 采集 1 帧 + 说话人识别 + 发布 (真人音频)
/home/naviai/projects/HSMR-main/.venv_orin/bin/python \
    examples/realtime_identify.py one --audio /path/to/speech.wav

# 离线重放: 不连 bridge, 用存帧 + 存好的 persons JSON 走完整变换链
python examples/realtime_identify.py replay \
    --frame-rgb rgb.jpg --frame-depth depth_raw.npy --persons persons.json

# 跳过声纹人脸 (纯坐标变换+发布)
python examples/realtime_identify.py one --no-identify
```

### 模块 API 速览

- **capture**: `MultiTopicCapture(url, topics).start()/.snapshot()/.stop()`; `decode_color(msg)→BGR`, `decode_depth(msg)→uint16mm`, `camera_info_k(msg)→3x3`, `build_topics(cfg)→list[str]`
- **infer**: `InferClient(cfg_infer)`: `.infer_persons(frame_or_rgb, depth_uint16, K) → persons[]` (失败抛 `InferError`); 纯函数 `encode_frame(rgb, depth, K)→(files, data)` 供测试/复用
- **tf**: `TransformChain(cfg)`: `.update_tf(msgs)`, `.get_opt_to_target(target)`, `.to_target(pts, target)`, `.rot_to_target(R, target)`, `.transform_source(target)`, `.available_frames()`; 顶层 `joint_pos_ori(person, j, chain, target)`, `joint_orientation(R, is_whole=False)`
- **face**: `FaceVerifier(cfg_fv)`: `.voice_search(audio_bytes, top_k)`, `.face_detect(img_bgr, n)`, static `.cross_verify(voice_hits, face_hits)`, `.associate_face_to_person(persons, location)`, `.identify(audio, img, persons, verbose)` (失败抛 `IdentifyError`)
- **ros**: `RosPublisher(cfg_ros)`: `.publish(frame, joints)`, `.publish_annotated_image(frame, img)`, `.resolve_publish_indices(selected, all)`, `.prewarm()`
- **pipeline**: `HsmrSdk` 门面 (见上)

---

# 三、完整自检清单

```bash
# 1. 容器起了吗
curl 127.0.0.1:8010/health          # → model_loaded:true

# 2. 推理通吗 (直接调 /infer)
curl -s -X POST 127.0.0.1:8010/infer \
  -F "image=@rgb.jpg" -F "depth=@depth_raw.png" \
  -F "fx=910.68" -F "fy=910.28" -F "cx=653.79" -F "cy=374.08"

# 3. SDK 采集通吗
cd /home/naviai/projects/HSMR-sdk && \
  python examples/realtime_identify.py one --no-identify

# 4. 声纹+人脸验证通吗 (8001 容器)
python examples/realtime_identify.py one --audio speech.wav
```

**运行前提** (都在同一台机器人上): `foxglove_bridge`(8768) + `zenohd`(7447) +
user-identification 容器(8001) + hsmr-infer 容器(8010)。

---

# 部署到 Jetson

```bash
# 远端工程布局 (2026-08-14 重组): 都在 /home/naviai/projects/HSMR/ 下
rsync -az /path/to/HSMR-sdk/ naviai@<jetson>:/home/naviai/projects/HSMR/HSMR-sdk/
# 远程复用原工程 venv: /home/naviai/projects/HSMR/HSMR-main/.venv_orin/bin/python
```
