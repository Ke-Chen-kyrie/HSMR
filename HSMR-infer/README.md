# HSMR Infer

基于 HSMR(HSMR-ViTH-r1d1)的**干净推理工程**:只保留模型的输入输出契约 + 推理能力
(detector → HSMR ONNX → 24 SKEL 关节, 含深度融合), **不含任何机器人外围**。

原工程 [HSMR-main](../HSMR-main) 保留不动。本工程与它的边界:

| 保留 (模型 I/O + 推理) | 分开 (原工程在外围) |
|---|---|
| `lib/` HSMR 模型库 | `capture.py` 话题订阅 |
| `deploy/onnx/` ONNX 推理管线 (输入 `image [1,3,256,256] fp16` → joints/poses/betas) | `tf_chain.py` TF 变换 |
| `deploy/orin/hsmr_core.py` SKEL 后处理 | `main.py` 关节查询 HTTP 服务 |
| detector (ViTDet) + 深度融合 (深度反投影) | `ros_pub.py` ROS/Foxglove 发布 |
| `thirdparty/SKEL`、`configs/`、模型权重 (`data_inputs/`) | `face_verify.py` 声纹+人脸 |
| **`hsmr_infer/`** 干净推理包 (本工程) | SKEL 渲染 / docker / 训练脚本 |

## 模型 I/O 契约

```
输入:  任意尺寸 BGR 图 (numpy HxWx3)          [可选: depth uint16 mm + 内参 K]
输出:  persons[] (每检测到一人一个):
         joints_2d        (24,2)  彩色图像素
         joints_optical_m (24,3)  光学系真实 3D 米坐标 (null=无效; 无深度融合时全 null)
         joints_ori       (24,3,3) SKEL 原始朝向
         depth_valid      [24]bool 逐关节深度有效
         pelvis_pose      {position_cam, rotation_matrix, pose_4x4}
```

> **深度格式 (重要)**: 深度融合只收**原始深度** — 16UC1 uint16 PNG 或 uint16 `.npy`(毫米)。
> 深度相机常见会额外存一张 **彩色可视化 PNG (uint8 3通道)**, 通道值≠毫米。
> 放进来会被显式拒绝并报错 (不会静默产出错误 3D)。无效关节在 JSON 中输出为 `null`。

## 结构

```
HSMR-infer/
├── hsmr_infer/            # 本工程推理包
│   ├── inference.py       #   load_models + infer_frame (BGR→persons, 深度融合可选)
│   ├── bbox_local.py      #   detector 后处理 (纯 numpy/cv2)
│   ├── server.py          #   单端口 FastAPI: POST /infer + /health + /joints
│   └── cli.py             #   CLI: 单图 → JSON
├── lib/  configs/  deploy/onnx  deploy/orin  thirdparty/SKEL   # 模型核心 (拷自原工程)
├── data_inputs/           # 权重自包含: detector pkl + HSMR-ViTH-r1d1 + body_models
├── data_outputs/          # 占位 (lib ProjManager 断言需要)
├── config.yaml            # 模型/服务配置
└── requirements.txt
```

## 运行

环境: 复用原工程 venv (已有 detectron2/onnxruntime), 或按 `requirements.txt` 新建。

```bash
# ① 服务 (单端口, 默认 8010)
cd /path/to/HSMR-infer
.venv/bin/python -m hsmr_infer.server
# 或 uvicorn hsmr_infer.server:app

# ② 单图 CLI (不带 depth = 纯模型推理)
.venv/bin/python -m hsmr_infer.cli --image test.jpg --device cpu --no-amp

# ③ CLI + 深度融合 (需对齐 depth 16UC1 PNG 或 uint16 .npy + 内参)
.venv/bin/python -m hsmr_infer.cli --image c.jpg --depth d.png \
    --fx 615 --fy 615 --cx 640 --cy 360 --json out.json
#   或直接喂 .npy (工程 data_outputs 的 depth_raw.npy 即此格式)
.venv/bin/python -m hsmr_infer.cli --image c.jpg --depth d.npy \
    --fx 615 --fy 615 --cx 640 --cy 360 --json out.json
```

服务接口 (`POST /infer`):

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `image` | ✅ | 文件 | JPG/PNG/BMP 彩图 |
| `depth` | 可选 | 文件 | 16UC1 原始深度 PNG / uint16 `.npy`(需与图同尺寸对齐) |
| `fx` `fy` | 可选 | float | 相机内参焦距, 给 depth 时建议用 |
| `cx` `cy` | 可选 | float | 相机内参主点, 给 depth 时建议用 |

> 给 `depth` 时同时给 K(`fx/fy/cx/cy`), 否则不做深度融合, `joints_optical_m` 为 `null`。

```bash
# 纯模型推理
curl -s -X POST http://127.0.0.1:8010/infer -F 'image=@c.jpg'

# 推理并统计端到端耗时 (time_total = 整个 HTTP 请求, 即「推理速度」判定指标)
curl -s -o /dev/null -w "推理总耗时=%{time_total}s\n" \
     -X POST -F 'image=@c.jpg' http://127.0.0.1:8010/infer

# 深度融合
curl -s -X POST http://127.0.0.1:8010/infer \
     -F 'image=@c.jpg' -F 'depth=@d.png' \
     -F 'fx=615' -F 'fy=615' -F 'cx=640' -F 'cy=360'

# 健康 / 关节名
curl -s http://127.0.0.1:8010/health
curl -s http://127.0.0.1:8010/joints
```

> 服务端先返回推理结果、再异步转发给 8011 渲染, 所以 `8010` 的 `time_total` 只含推理不含渲染。

## 配置 (config.yaml)

| 字段 | 默认 | 说明 |
|---|---|---|
| `model.device` | cuda:0 | detector + SKEL 设备 |
| `model.onnx_device` | cuda:0 | ONNX Runtime 设备 |
| `model.onnx_model` | deploy/onnx/artifacts/hsmr_full_fp16.onnx | ONNX 图 |
| `model.model_root` | data_inputs/released_models/HSMR-ViTH-r1d1 | SKEL 配置根 |
| `detector.max_img_size` | 512 | detector 输入尺寸 |
| `detector.use_amp` | true | detector FP16 |
| `max_instances` | 5 | 每帧最多人数 |
| `server.port` | 8010 | 服务端口 |

模型路径相对工程根解析 (不依赖 CWD)。

## 部署到 Jetson

```bash
rsync -az --exclude='.venv' /path/to/HSMR-infer/ naviai@<jetson>:/home/naviai/projects/HSMR-infer/
```
远程用原工程 venv: `/home/naviai/projects/HSMR-main/.venv_orin/bin/python`。

## Docker 运维 (远程 192.168.217.100, 已上线)

infer(8010) + render(8011) 以容器部署, **restart=unless-stopped + docker 开机自启**,
宿主机重启后会自动拉起, 无需手动干预。常用命令 (在机器人本机或 `ssh naviai@192.168.217.100`):

```bash
# ① 查看运行状态 / 健康
docker ps --filter name=hsmr
curl -s http://127.0.0.1:8010/health
curl -s http://127.0.0.1:8011/health

# ② 重启推理容器 (改完代码/配置后最常用)
docker restart hsmr-infer

# ③ 重启渲染容器
docker restart hsmr-render

# ④ 同时重启两个
docker restart hsmr-infer hsmr-render

# ⑤ 看日志 (启动约 50s 加载模型, 等 health 返回 model_loaded:true 再发请求)
docker logs -f hsmr-infer
docker logs -f hsmr-render

# ⑥ 停止/启动 (注意: restart=unless-stopped, 手动 stop 后不会自动拉起, 需手动 start)
docker stop hsmr-infer && docker start hsmr-infer

# ⑦ 全部停掉
docker stop hsmr-infer hsmr-render
```

> 模型全进程**单例只加载一次**(启动 ~50s), 之后每个请求复用 GPU 常驻实例。
> 多人实测端到端 ~1.85s/2人 (detector≈1.4s + onnx+skel≈0.5s), 在 2s 预算内。
