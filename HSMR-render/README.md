# HSMR-render

独立渲染服务。与 [HSMR-infer](../HSMR-infer)(8010 纯推理) 各自收**同样的关节选择 POST**:

- **HSMR-infer**: 收到 POST 只返回关节 3D 位置 + 朝向, **不渲染**;
- **本服务**: 收到同样的 POST 只**渲染**不返 3D —— 渲染当前帧(2D 叠加在原图上) +
  后台渲染后 N 帧(自抓 ROS1 rosbridge 相机流), 结果入渲染队列, `GET /render/pop` 取走。

> **部署形态(2026-09 现状):** 远程以 **docker 容器 `hsmr-render`**(host 网络, 端口 8011) 运行,
> 相机流经 ROS1 **rosbridge**(`ws://localhost:9090`) 订阅, **不再用 foxglove**。
> 本 README 保留裸进程启动方式作参考, 当前线上运维命令见 [§ 部署与运维(Docker)](#部署与运维docker)。

## 为什么本服务也要自己跑完整推理

2D 蒙皮/骨骼网格叠加需要网格参数 `poses(46)/betas(10)/cam_t(3)` —— 这三个**只能在模型前向时拿到**,
无法从关节 3D 反推; 而 HSMR-infer `/infer` 响应不含它们, 且"保留原工程"禁改其契约。
→ 本服务自抓帧 + 完整 HSMR 推理(detector + ONNX), 与旧 joint_service 的 `InferenceWorker` 一致。

## 文件

| 文件 | 说明 |
|---|---|
| `main.py` | FastAPI 服务(端口 8011): `_SkelRenderer`(SKEL 蒙皮蓝+骨骼黄 0.7/0.3 + 光晕)、渲染队列、`POST /joint`、`GET /render/pop`、后台后 N 帧 |
| `capture.py` | ROS1 rosbridge 多话题订阅(节流 1Hz, 去 TF 版; 旧 foxglove 实现已弃用) |
| `inference.py` | detector + HSMR(ONNX) 推理 → persons + 网格参数(去 tf_chain 版) |
| `bbox_local.py` | 检测框 → 256 patch(纯 numpy) |
| `config.yaml` | 配置(见下) |
| `verify_render.py` | 离线渲染验证(本地 CPU 软件 EGL, 不连相机) |
| `verify_render_service.py` | 远程 HTTP 全链路验证 |

复用 HSMR-main 的 venv + 模型 + `lib/` `thirdparty/`(sys.path 指向 `hsrm_root`), **不改原工程一行**。

## 配置 (config.yaml)

- `hsrm_root`: HSMR-main 根(模型/lib/thirdparty 基准); 本机 `/home/Kai/pose_model/HSMR-main`, 远程 `/home/naviai/HSMR/HSMR-main`
- `bridge.url`: foxglove bridge(`ws://localhost:8768`, 旧实现, 已弃用)
- `bridge.rosbridge_url`: **ROS1 rosbridge**(`ws://localhost:9090`) —— 当前线上相机流来源
- `bridge.rosbridge_throttle_ms`: rosbridge 订阅节流(ms), **1000 = 1Hz**。⚠️ 必须开启: 不节流则 30fps
  持续 json 解析大 base64 会打满 CPU/内存带宽, 同进程 GPU 推理被拖慢 ~10x(性能优化关键项)
- `topics.prefix`: 相机话题前缀; `topics.ros1`: 实际订阅的 5 个话题名(color/depth/info/tf_static/tf)
- `model.onnx_model` / `model.model_root`: 相对 `hsrm_root` 解析成绝对, **启动 CWD 无关**
- `render.next_frames`: POST 后后台渲染帧数(默认 4; 0=关, 只渲染当前帧 1 张)
- `render.next_frame_interval`: 后台帧间隔秒(0=靠每帧 ~2.2s 推理自然间隔)
- `render.out_dir`: 渲染图存盘目录(相对本工程根, 保持 HSMR-main 干净)
- `render.half_res`: 半分辨率渲染再上采样(省 GL 填充/读回; 默认 false 保画质)
- `server.port`: 8011(避开旧 joint_service 8002 / hsmr-infer 8010)

## 接口

### `POST /joint`

body 为 key-value 关节选择(同旧 joint_service 契约): `{"hand_r":1, "head":0}`, value=1 选中;
支持中文(右腕/右手)与整体关键词(`{"整体":1}`)。可选 query `person_index`(默认 0)。

处理: 自抓当前帧 → 完整推理 → 渲染当前帧(同步等, 返回时已入队, seq=0) →
后台再渲染后 N 帧(seq=1..N, 自抓相机流)。**只返回渲染 ack, 不返回 3D 位置/朝向**(那是 HSMR 的事)。

```json
{"queued": true, "queue_size": 1, "timestamp": 1726..., "rendered_seq": 0,
 "next_frames": 4, "person_index": 0, "num_persons": 3,
 "joints": ["hand_r"], "joints_cn": ["右手"], "glow": true, "infer_ms": 2200.5, "bg_started": true}
```

### `GET /render/pop`

从队列取一张图(取走即移除)。条目含 `seq`(0=当前帧, 1..N=后台帧)、`joints`、`timestamp`、
`image_base64`(JPEG)。空队列返回 `{"rendered": null}`。

### `GET /render/queue` / `GET /health` / `GET /joints`

队列状态 / 服务状态(`model_loaded`)/ 24 关节名。

## 并发约束(务必遵守)

- **pyrender/EGL 必须单线程**: 所有渲染(当前帧 + 后 N 帧)一律经内部 `max_workers=1` 渲染执行器,
  外部线程只提交任务、不直接调渲染。
- 推理串行: 当前帧 + 后台帧都走同一把推理锁, GPU 前向不并发。
- 后台批次防重叠: 上一批后 N 帧未结束时再 POST, 只渲染当前帧(`bg_started:false`), 不叠开第二批。
- 队列满(maxsize=50): 丢新帧保当前帧, 不阻塞。

## 启动

```bash
# 本地(仅离线渲染验证 — 本机 torch 无 CUDA, 跑不了全链路)
cd /home/Kai/pose_model/HSMR-render
/home/Kai/pose_model/HSMR-main/.venv/bin/python verify_render.py     # 离线渲染验证

# 远程(Jetson GPU, 全链路) — 裸进程方式(旧, 已由 docker 替代, 见下节)
cd /home/naviai/projects/HSMR/HSMR-render
setsid nohup /home/naviai/projects/HSMR/HSMR-main/.venv_orin/bin/python main.py > service.log 2>&1 &
curl http://127.0.0.1:8011/health      # 等 model_loaded:true (模型加载 1-2 分钟)
```

## 部署与运维(Docker, 当前线上)

远程容器 `hsmr-render`, 镜像 `hsmr-render:20260901-fast`(本地 docker tag, host 网络), 端口 8011。

```bash
# ⚠️ 先确认 ROS 栈(rosbridge/相机)已就绪, 再重启本服务 —— 见下方「故障排查」
ssh naviai@192.168.217.100
docker restart hsmr-render           # 就绪需 ~40-50s(模型加载 33s + 预热渲染一帧)

# 看日志(重点看启动末两行: 预热)
docker logs -f hsmr-render
#   [main] 预热完成 (已渲染一帧, 不入队)  ← 正常: 订阅到相机, 渲染了一帧
#   [main] 预热: 当前帧无人, 跳过渲染      ← ⚠️ 异常: 相机订阅为空, 需排查(见下)

# 健康 / 触发 / 取图
curl http://127.0.0.1:8011/health              # 等 model_loaded:true
curl -X POST http://127.0.0.1:8011/joint -H "Content-Type: application/json" -d '{"右膝":1}'
curl http://127.0.0.1:8011/render/pop          # 取一张图(image_base64)

# 全链路验证(已适配异步: 会等 seq0..4 依次入队再取)
python verify_render_service.py --url http://127.0.0.1:8011 --joints "右膝"
```

代码同步(本工程 → 远程构建源):

```bash
rsync -az --exclude='.venv*' --exclude='__pycache__' --exclude='data_outputs' \
  /home/Kai/pose_model/HSMR/HSMR-render/ naviai@192.168.217.100:/home/naviai/HSMR/render_rebuild/app/
# 容器内实际跑的是 /app/ 下代码; 同步后 docker cp 进容器 或 重打镜像:
docker cp <文件> hsmr-render:/app/<文件>
docker restart hsmr-render
```

## 故障排查: 重启后 focus_on 503 / `/joint` "推理失败"

**现象**: 整机/ROS 栈重启后, `focus_on` 报 `⚠️ 关节聚焦服务暂不可用 (Status: 503)`;
直接 curl `/joint` 返回 `{"detail":"推理失败, 请检查相机/深度话题"}`。

**根因(订阅时序错位)**: 本服务启动时若撞上 ROS 栈未就绪(rosbridge/相机话题还没注册到 master),
首次连 rosbridge 被拒后重连订阅成功, 但订阅发生在相机话题注册**之前** → rosbridge 对"订阅时空话题、
之后才出现"的订阅**不会自动补发** → 永久空订阅, `capture.snapshot()` 无 color/depth/info →
`infer_fresh()` 返回 None → `/joint` 抛 503。

**判定**:
- `docker logs hsmr-render | tail`: 末行是 `预热: 当前帧无人, 跳过渲染` = 订阅空(有故障);
  `预热完成 (已渲染一帧)` = 正常。
- 手动订阅能收到数据(`6s 收到 N 条`)、但服务进程抓不到 = 确认是空订阅时序问题。

**修复**: 在 ROS 栈完全就绪后重启本服务, 让它重新订阅即可。

```bash
docker restart hsmr-render   # 就绪后看日志: 应出现「预热完成 (已渲染一帧)」
```

**根治建议(尚未实施)**: `capture.py` `RosbridgeCapture._receive()` 目前是阻塞 `while: ws.recv()`,
无数据时永不重连。建议改为 **recv 超时 + 长时间无 publish 自动断连重连**(如 `ws.recv(timeout=5)`,
累计 N 秒无数据则断开重连), 让话题在 ROS 就绪后出现时本服务自动恢复, 免去人肉重启。
实现时注意: 重连应复用 `_run` 循环的异常退出机制, 且重连前短暂 sleep 防空转。

**架构备忘(排查别踩坑)**: ROS master 在 `192.168.217.1:11311`(**不在本机** 192.168.217.100);
相机=`naviai_sensor` 容器、rosbridge=`naviai_rosbridge` 容器(均 host 网络), 都连 192.168.217.1 master。
本机 11311 是空 roscore, 别查错层。相关记录: 桌面 `HSMR-render服务3s优化记录_2026-09-01.md` §7。

## 已知取舍

- 8010(HSMR-infer)与 8011(本服务)同机共享 GPU, 同帧双份推理 —— 契约冻结决定, 非重复建设。
- 后台每帧推理 ~2.2s, 后 N 帧约 9-11s 填满队列; 消费方按 `seq` 容忍间隔。
- 无人生成: 当前帧 POST 返回 404; 后台帧跳过该 seq, 不产生坏图。
