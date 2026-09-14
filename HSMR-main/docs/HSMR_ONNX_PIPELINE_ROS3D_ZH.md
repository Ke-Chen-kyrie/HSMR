# HSMR Webcam ONNX 推理、三维坐标、RealSense 深度与 ROS TF 学习笔记

本文对应当前项目中的以下主要代码：

- `exp/run_webcam.py`
- `lib/modeling/pipelines/vitdet/`
- `lib/kits/hsmr_demo.py`
- `deploy/onnx/runtime.py`
- `deploy/orin/hsmr_core.py`
- `deploy/onnx/hsmr_onnx.py`
- `deploy/onnx/export_hsmr_onnx.py`
- `lib/body_models/skel_wrapper.py`

本文的目标有两个：

1. 解释当前 `infer_latest_frame()` 从人体检测到 ONNX、SKEL 和渲染的完整执行过程。
2. 解释怎样在未来把 HSMR 的人体结果与 RealSense 对齐深度、`CameraInfo` 和 ROS TF 结合，得到真实相机坐标及 `base_link` 坐标。

> 重要：当前项目只订阅了压缩彩色图。深度、`CameraInfo`、`/tf` 和 `/tf_static` 尚未接入 `run_webcam.py`。本文后半部分描述的是推荐扩展方案，不代表当前代码已经实现。

---

## 1. 最先记住的结论

当前程序同时存在四种不同含义的数据：

| 数据 | 形状 | 是否为三维位置 | 参考坐标系 |
|---|---:|---|---|
| `poses_6d` | `[N,24,6]` | 否 | 24 个关节的连续旋转表示 |
| `poses` | `[N,46]` | 否 | SKEL 的 46 个广义关节参数 |
| `joints_44_body` | `[N,44,3]` | 是 | 人体根节点相对坐标 |
| `joints_44_camera` | `[N,44,3]` | 是 | HSMR 裁剪图虚拟相机坐标 |
| `pd_cam_t` | `[N,3]` | 是 | 人体根节点相对裁剪图虚拟相机的估计平移 |
| `raw_cam_t` | `[N,3]` | 是 | 经过原图 bbox 修正的 HSMR 虚拟相机平移 |
| 深度反投影坐标 | `[N,J,3]` | 是 | RealSense 真实光学坐标系，通常以米为单位 |
| TF 转换后坐标 | `[N,J,3]` | 是 | `base_link`、`odom` 或 `map` |

`pd_cam_t`、`raw_cam_t` 和 `joints_44_camera` 是为了让单目 HSMR 人体投影与图像对齐而估计的模型坐标。它们没有使用真实相机内参、RealSense 深度或 ROS TF，因此不能直接作为机器人世界坐标。

---

## 2. 当前 `infer_latest_frame()` 总调用链

入口位于 `exp/run_webcam.py:485`：

```text
frame_bgr
  │
  ├─ BGR → RGB
  │
  ├─ detector([frame_rgb])
  │    ├─ ViTDet/Cascade Mask R-CNN
  │    ├─ pred_classes
  │    ├─ scores
  │    ├─ pred_boxes
  │    └─ downsample_ratio
  │
  ├─ _img_det2patches(...)
  │    ├─ 只保留 person class=0
  │    ├─ score > 0.5
  │    ├─ 最多 max_instances 人
  │    ├─ 原图 bbox → 方形 bbox
  │    └─ 每个人裁剪为 [256,256,3]
  │
  ├─ ImageNet 归一化 + NHWC → NCHW
  │    └─ [N,3,256,256]
  │
  ├─ pipeline(patches_normalized)
  │    └─ HSMRONNXRuntimePipeline.__call__
  │         ├─ 校验输入
  │         ├─ 每人切成静态 B=1
  │         ├─ session.run(...)
  │         ├─ 拼回 N 人
  │         └─ 返回 poses、betas、pd_cam_t
  │
  ├─ prepare_mesh(...)
  │    └─ PyTorch SKEL
  │         ├─ skin_verts
  │         ├─ skel_verts
  │         └─ joints（当前函数未返回 joints）
  │
  ├─ visualize_full_img(...)
  │    ├─ pd_cam_t → raw_cam_t
  │    ├─ 皮肤网格渲染
  │    ├─ 骨骼网格渲染
  │    └─ 拼接结果图
  │
  └─ rendered image + 性能数据
```

---

## 3. `detector_outputs = detector([frame_rgb])`

### 3.1 为什么先把 BGR 转成 RGB

OpenCV 解码的摄像头帧是 BGR：

```python
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
```

项目的 Detectron2 mapper 明确要求输入格式为 RGB，见：

```text
lib/modeling/pipelines/vitdet/utils_detectron2.py:63
```

### 3.2 检测器是什么

检测器由 `build_detector()` 创建，位置：

```text
lib/modeling/pipelines/vitdet/__init__.py:7
```

使用的是 ViTDet-H Cascade Mask R-CNN。构建时 Detectron2 自身的测试阈值设置为 `0.25`，但 `_img_det2patches()` 后面还会使用更严格的 `0.5` 阈值再次筛选。

### 3.3 `detector([frame_rgb])` 内部做什么

`DefaultPredictor_Lazy.__call__()` 位于：

```text
lib/modeling/pipelines/vitdet/utils_detectron2.py:66
```

对每张 RGB 图依次执行：

1. 如果图像最长边超过 `det_mis`，按比例缩小。
2. 保存 `downsample_ratio`。
3. 执行 Detectron2 augmentation。
4. 把 `[H,W,3]` 转成 `[3,H,W]` 的 `float32` Tensor。
5. 移动到 CUDA。
6. 调用 Detectron2 模型。
7. 把结果复制回 CPU，只保留类别、置信度和框。

返回：

```python
detector_outputs = (preds, downsample_ratios)
```

当前只输入了一张图，所以：

```python
detector_outputs[0][0]  # 第0张图的检测结果字典
detector_outputs[1][0]  # 第0张图的缩放比例
```

检测结果字典的结构是：

```python
{
    "pred_classes": Tensor[M],
    "scores": Tensor[M],
    "pred_boxes": Tensor[M,4],
}
```

其中 `M` 是所有类别检测框的数量，还没有只保留人。

`pred_boxes` 的顺序为：

```text
[left, upper, right, bottom]
```

框坐标对应检测器缩放后的图像，所以后面必须除以 `downsample_ratio` 才能回到原始彩色图坐标。

### 3.4 推荐 Debug 内容

在 `exp/run_webcam.py:511` 后检查：

```python
detector_outputs[0][0].keys()
detector_outputs[0][0]["pred_classes"]
detector_outputs[0][0]["scores"]
detector_outputs[0][0]["pred_boxes"]
detector_outputs[1][0]
```

---

## 4. `_img_det2patches()` 怎样筛人和裁剪

函数位于：

```text
lib/kits/hsmr_demo.py:123
```

调用参数：

```python
patches, bbx_cs = _img_det2patches(
    frame_rgb,
    detector_outputs[0][0],
    detector_outputs[1][0],
    args.max_instances,
)
```

### 4.1 只保留可信人体

```python
is_human_mask = pred_classes == 0
reliable_mask = scores > 0.5
active_mask = is_human_mask & reliable_mask
```

这里的 `0` 是 COCO person 类别。

如果可信人体超过 `max_instances`，则选择人体检测分数最高的前 K 个。

### 4.2 检测框恢复到原图

```python
lurb_all = pred_boxes[valid_mask].numpy() / downsample_ratio
```

例如原图最长边为 `1280`，检测器最大边为 `512`：

```text
downsample_ratio = 512 / 1280 = 0.4
```

检测图上的坐标 `[200,100,300,400]` 恢复到原图就是：

```text
[500,250,750,1000]
```

### 4.3 为什么先调整到 `192:256`，又扩成方形

HSMR 的 ViT 实际只使用人物 patch 中间的 `256×192` 区域。代码先扩展检测框，使人体主要内容满足 `192:256` 宽高比，再把它转换成方形框以兼容当前数据处理接口。

最终保存：

```python
bbx_cs = [center_x, center_y, square_size]
```

### 4.4 裁剪和缩放

`crop_with_lurb()` 支持 bbox 超出图像边界。超出部分使用 0 填充。裁剪后再缩放为：

```text
[256,256,3]
```

多人时：

```text
patches.shape = [N,256,256,3]
len(bbx_cs) = N
```

### 4.5 一个容易误解的注释

`_img_det2patches()` 的函数注释写了“Normalize”，但函数本身只负责裁剪和缩放。真正归一化发生在 `run_webcam.py:540-545`。

### 4.6 `detection_meta` 是否进入 HSMR

如果远程版本增加了：

```python
patches, bbx_cs, detection_meta = _img_det2patches(...)
```

`detection_meta` 通常用于保存检测框、分数、类别或跟踪信息。它不进入神经网络。HSMR 的实际输入仍只有 `patches_normalized`。

---

## 5. HSMR 输入归一化

位置：

```text
exp/run_webcam.py:540-545
```

首先转为 `float32`：

```python
patches = patches.astype(np.float32)
```

然后使用 ImageNet RGB 均值和标准差：

```python
patches_normalized = (patches - IMG_MEAN_255) / IMG_STD_255
```

具体数值：

```text
mean = [123.675, 116.280, 103.530]
std  = [ 58.395,  57.120,  57.375]
```

最后从 NHWC 转为 NCHW：

```python
patches_normalized = patches_normalized.transpose(0,3,1,2)
```

输入形状变为：

```text
[N,3,256,256]
```

`np.ascontiguousarray()` 保证底层内存连续，避免 ONNX Runtime 处理非连续视图。

推荐检查：

```python
patches_normalized.shape
patches_normalized.dtype
np.isfinite(patches_normalized).all()
patches_normalized.min(), patches_normalized.max()
```

---

## 6. `HSMRONNXRuntimePipeline` 初始化

类位于：

```text
deploy/onnx/runtime.py:73
```

在 `run_webcam.py:608-616` 创建一次，后续每帧复用同一个 ONNX Session。

### 6.1 Provider 选择

`provider=auto` 且 `device=cuda:0` 时优先：

```text
CUDAExecutionProvider → CPUExecutionProvider
```

如果明确指定 TensorRT：

```text
TensorrtExecutionProvider → CUDAExecutionProvider → CPUExecutionProvider
```

### 6.2 Session 配置

当前代码：

- 开启 `ORT_ENABLE_ALL` 图优化。
- Jetson 上禁用 `GatherSliceToSplitFusion`。
- 模型必须只有一个名为 `image` 的输入。
- 输入类型只接受 FP16 或 FP32。

### 6.3 判断 full/core 图

如果模型输出至少包含：

```text
poses, betas, camera_translation
```

则：

```python
self.graph_scope = "full"
self.run_output_names = ["poses", "betas", "camera_translation"]
```

如果模型只包含核心输出：

```text
poses_6d, betas, camera
```

则：

```python
self.graph_scope = "core"
self.run_output_names = ["poses_6d", "betas", "camera"]
```

### 6.4 为什么 ONNX pipeline 里仍加载 PyTorch SKEL

初始化最后调用：

```python
self.skel_model = load_skel_model(...)
```

原因是 `prepare_mesh()` 需要 SKEL 的皮肤拓扑、骨骼拓扑和顶点。当前架构不是“全程序纯 ONNX”，而是：

```text
人体检测：PyTorch Detectron2
HSMR视觉回归：ONNX Runtime
人体网格：PyTorch SKEL
渲染：OpenGL
```

---

## 7. `outputs = pipeline(patches_normalized)` 具体执行

这行位于：

```text
exp/run_webcam.py:548
```

实际进入：

```text
deploy/onnx/runtime.py:198
HSMRONNXRuntimePipeline.__call__()
```

### 7.1 输入校验

要求：

```text
ndim = 4
shape[1:] = [3,256,256]
N > 0
```

如果传入 PyTorch Tensor，会先复制到 CPU NumPy。

### 7.2 为什么 `_run_static_batch_one()` 是线性的

导出的 ONNX 使用固定 batch size 1。导出样例就是：

```text
[1,3,256,256]
```

所以 `_run_static_batch_one()` 不能一次送入 N 个人，而是：

```python
for index in range(len(patches)):
    image = patches[index:index+1]
    values = session.run(...)
```

假设有 3 个人：

```text
原始 patches：[3,3,256,256]

第1次 session.run：[1,3,256,256]
第2次 session.run：[1,3,256,256]
第3次 session.run：[1,3,256,256]

最后把三次输出 concatenate 回 batch=3
```

因此推理耗时近似：

```text
T(N) ≈ N × T_onnx_batch1 + T_concat
```

它是串行循环，不是动态 batch，也没有并行执行不同人物。

优点是单次 GPU 峰值显存接近 B=1；缺点是人数越多，ONNX 调用开销和网络推理时间近似线性增加。

### 7.3 每个人送入 Session 前做什么

```python
image = np.ascontiguousarray(
    patches[index:index+1],
    dtype=self.input_dtype,
)
```

如果模型输入是 FP16，这里把归一化后的 NumPy FP32 转为 FP16。

此时典型形状：

```text
image.shape = [1,3,256,256]
image.dtype = float16
```

---

## 8. `session.run()` 到底做什么

当前调用：

```python
values = self.session.run(
    self.run_output_names,
    {self.input_name: image},
)
```

两个参数的含义：

```python
self.run_output_names
```

表示希望 ONNX Runtime 返回哪些命名输出，例如：

```python
["poses", "betas", "camera_translation"]
```

输入字典：

```python
{"image": image}
```

表示把 NumPy 数组绑定到 ONNX 图的 `image` 输入。

### 8.1 Session 内部执行过程

从当前代码视角可以理解为：

1. 检查输入名称、形状和数据类型。
2. 根据 Execution Provider 把输入送到 CUDA、TensorRT 或 CPU。
3. 按 ONNX 计算图拓扑执行算子，例如 Conv、MatMul、Reshape、Gather、三角函数等。
4. 把请求的输出组成 NumPy 数组返回。
5. Python 调用在 `session.run()` 返回前是阻塞的。

当前没有使用 ONNX Runtime I/O Binding，因此输入是 CPU NumPy，ORT 自己负责把输入复制到 GPU；返回的 `values` 也是 CPU NumPy。多人时每个人都会发生一次 `session.run()` 边界调用。

`session.run()` 返回顺序与 `self.run_output_names` 顺序一致，因此代码可以：

```python
for name, value in zip(self.run_output_names, values):
    chunks[name].append(value)
```

最后：

```python
np.concatenate(values, axis=0)
```

恢复成 N 人的 batch。

### 8.2 输出列表是否会保证跳过其他分支

`run_output_names` 首先控制“哪些结果返回给 Python”。不能仅凭这个列表假定图内所有其他节点一定不会执行。你的实际错误：

```text
/postprocessor/skel_model/Reshape_185
```

已经证明当前 full 图的执行计划进入了 SKEL 后处理节点，即使 Python 最后只接收三个输出。

### 8.3 推荐 Debug

在 `deploy/onnx/runtime.py:186` 前查看：

```python
self.graph_scope
self.providers
self.input_name
self.input_dtype
self.run_output_names
image.shape
image.dtype
np.isfinite(image).all()
```

在 `session.run()` 后查看：

```python
[(name, value.shape, value.dtype)
 for name, value in zip(self.run_output_names, values)]
```

检查整个 ONNX 模型声明的输入输出：

```python
[(x.name, x.shape, x.type) for x in self.session.get_inputs()]
[(x.name, x.shape, x.type) for x in self.session.get_outputs()]
```

---

## 9. `HSMRONNXPostprocessor.forward()` 在哪里被调用

这是最容易混淆的地方。

### 9.1 导出 full ONNX 时发生的 Python 调用

导出程序在：

```text
deploy/onnx/export_hsmr_onnx.py
```

full 模型构建过程：

```python
postprocessor = HSMRONNXPostprocessor(skel_model)
model = HSMREndToEndONNX(core, postprocessor)
```

`HSMREndToEndONNX.forward()` 位于 `deploy/onnx/hsmr_onnx.py:189`：

```python
poses_6d, betas, camera = self.core(image)
skeleton_outputs = self.postprocessor(
    poses_6d,
    betas,
    camera,
)
```

调用 `self.postprocessor(...)` 时，PyTorch Module 的 `__call__` 会转入：

```text
HSMRONNXPostprocessor.forward()
```

之后 `torch.onnx.export()` 在 `export_hsmr_onnx.py:294` 运行/追踪这个模型，把 forward 中的运算固化成 ONNX 节点。

### 9.2 实际 webcam 运行时不会再调用这个 Python forward

模型导出完成后，webcam 使用的是 `.onnx` 文件：

```python
self.session = ort.InferenceSession(...)
self.session.run(...)
```

这时不会再进入 Python 的：

```python
HSMRONNXPostprocessor.forward()
```

ONNX Runtime 直接执行由它导出的 ONNX 节点。

因此报错名称：

```text
/postprocessor/skel_model/Reshape_185
```

表示这个 ONNX 节点最初来自：

```text
HSMREndToEndONNX.postprocessor.skel_model
```

但它不是当前 Python 调用栈里现场执行的 `forward()`。

### 9.3 core ONNX 的不同

core ONNX 只导出：

```text
image → poses_6d, betas, camera
```

不包含 `HSMRONNXPostprocessor`。core 模式下 `runtime.py` 在 Python 中执行：

```python
poses = params_rep2q(poses_6d)
camera_translation = ...
```

随后 `prepare_mesh()` 再调用 PyTorch SKEL。

---

## 10. HSMR 核心神经网络

核心定义在：

```text
deploy/orin/hsmr_core.py:21
```

输入：

```text
image：[B,3,256,256]
```

### 10.1 中间裁剪

```python
features = self.backbone(image[:,:,:,32:-32])
```

左右各裁掉 32 像素，真正进入 ViT 的是：

```text
[B,3,256,192]
```

### 10.2 ViT + Transformer

```python
context = features.flatten(2).transpose(1,2)
token = zeros([B,1,1])
token = transformer(token, context=context)
```

Transformer token 汇总整个人体视觉特征，然后三个 decoder 分别回归：

```python
poses_6d = poses_decoder(token) + init_poses
betas = betas_decoder(token) + init_betas
camera = camera_decoder(token) + init_camera
```

输出：

```text
poses_6d：[B,144] = [B,24×6]
betas：[B,10]
camera：[B,3]
```

---

## 11. `HSMRONNXPostprocessor.forward()` 每个变量的含义

函数位置：

```text
deploy/onnx/hsmr_onnx.py:116-170
```

### 11.1 `poses_6d`

形状：

```text
[B,144] 或 reshape 后 [B,24,6]
```

它是 24 个关节的连续旋转表示。每组 6 个数不是 XYZ、不是欧拉角，也不是“位置3维+旋转3维”。它是便于神经网络稳定回归旋转矩阵的表示。

### 11.2 `poses`

```python
poses = params_rep2q_onnx(poses_6d.float())
```

输出：

```text
[B,46]
```

它把 24 个关节旋转转换为 SKEL 使用的 46 个广义姿态参数：

```text
10个三自由度关节 × 3 = 30
 2个二自由度关节 × 2 = 4
12个一自由度关节 × 1 = 12
合计 46
```

46 个参数的区间：

| 索引 | 部位 |
|---|---|
| `0:3` | pelvis，骨盆/根姿态 |
| `3:6` | 右大腿 |
| `6` | 右小腿/膝 |
| `7` | 右距骨 |
| `8` | 右跟骨 |
| `9` | 右脚趾 |
| `10:13` | 左大腿 |
| `13` | 左小腿/膝 |
| `14` | 左距骨 |
| `15` | 左跟骨 |
| `16` | 左脚趾 |
| `17:20` | 腰椎 |
| `20:23` | 胸部 |
| `23:26` | 头部 |
| `26:29` | 右肩胛 |
| `29:32` | 右上臂 |
| `32` | 右尺骨/肘 |
| `33` | 右桡骨/前臂 |
| `34:36` | 右手 |
| `36:39` | 左肩胛 |
| `39:42` | 左上臂 |
| `42` | 左尺骨/肘 |
| `43` | 左桡骨/前臂 |
| `44:46` | 左手 |

这些值主要是弧度制的 Euler-like/广义关节量。尤其 `poses[:,0:3]` 不能不经转换就直接命名为普通 `roll/pitch/yaw`；SKEL 使用自己的轴顺序。

### 11.3 `betas` 与 `betas_fp32`

```python
betas_fp32 = betas.float()
```

形状：

```text
[B,10]
```

`betas` 是人体形状空间的 10 个统计系数，综合影响身材比例、胖瘦、四肢长度和躯干形状。每一个 beta 不对应一个可以直接命名的身体尺寸，也不是三维坐标。

转为 FP32 只是提高 SKEL 蒙皮和三角函数计算的稳定性，不改变其语义。

### 11.4 `camera` 与 `camera_fp32`

```python
camera_fp32 = camera.float()
```

形状：

```text
[B,3]
```

语义是弱透视相机参数：

```text
camera = [scale, tx, ty]
```

- `scale`：人体在裁剪图中的尺度。
- `tx`：裁剪图虚拟相机横向平移。
- `ty`：裁剪图虚拟相机纵向平移。

它不是 RealSense 外参，也不是 ROS 中相机的物理位姿。

### 11.5 `camera_translation`

代码使用：

```python
camera_translation = [
    camera_fp32[:,1],
    camera_fp32[:,2],
    2 * focal / (patch_size * camera_fp32[:,0] + 1e-9),
]
```

其中：

```text
focal = 5000
patch_size = 256
```

输出形状：

```text
[B,3]
```

可理解为：

```text
[虚拟相机x平移, 虚拟相机y平移, 虚拟深度z]
```

`scale` 越大，计算得到的 `z` 越小；`scale` 越小，`z` 越大。

这个 `z` 主要用于投影对齐，不是深度相机测得的物理 Z。

### 11.6 `joints_44_body`

```python
joints_44_body = skel_output.joints
```

形状：

```text
[B,44,3]
```

这是人体根相对三维关节点。根据 `SKELWrapper.forward()`，44 点由两部分拼成：

```text
前25点：映射到 OpenPose BODY_25 风格的点
后19点：由额外 joint regressor 从皮肤顶点回归的点
```

因此它不应简单理解为“44根骨头的解剖中心”。它是为训练、评估和 2D/3D 关键点接口组织的联合关键点集合。

### 11.7 `joints_24_anatomical`

```python
joints_24_anatomical = skel_output.joints_backup
```

形状：

```text
[B,24,3]
```

这是 SKEL 原始的 24 个解剖/运动学关节点，顺序是：

```text
0 pelvis
1 femur-r
2 tibia-r
3 talus-r
4 calcn-r
5 toes-r
6 femur-l
7 tibia-l
8 talus-l
9 calcn-l
10 toes-l
11 lumbar
12 thorax
13 head
14 scapula-r
15 humerus-r
16 ulna-r
17 radius-r
18 hand-r
19 scapula-l
20 humerus-l
21 ulna-l
22 radius-l
23 hand-l
```

如果目标是分析 SKEL 生物力学骨架，这组点比 `joints_44_body` 更直接。

### 11.8 `joints_24_custom`

```python
joints_24_custom = skel_output.joints_custom
```

形状：

```text
[B,24,3]
```

这是把 SKEL 点映射/回归成 SMPL 风格的 24 点顺序后的结果。它位于 44 点组合之前，主要用于与 SMPL/HMR 数据接口兼容。

### 11.9 `joints_44_camera`

```python
joints_44_camera = (
    joints_44_body + camera_translation[:,None,:]
)
```

含义是：

```text
人体根相对关节点 + 整个人的虚拟相机平移
```

得到的是 HSMR 裁剪图虚拟相机坐标。它不是经 CameraInfo 标定的 RealSense 光学坐标。

### 11.10 `joints_2d_normalized`

```python
joints_2d_normalized = (
    joints_44_camera[:,:,:2]
    / joints_44_camera[:,:,2:3]
    * (5000 / 256)
)
```

形状：

```text
[B,44,2]
```

这是以裁剪图中心为原点的归一化投影坐标。这里 `(0,0)` 表示裁剪图中心，不保证所有点都落在固定的 `[-0.5,0.5]` 范围内。

### 11.11 `joints_2d_patch`

```python
joints_2d_patch = (
    joints_2d_normalized + 0.5
) * 256
```

形状：

```text
[B,44,2]
```

它是 `256×256` 人物 patch 上的像素坐标：

```text
(0,0) normalized → (128,128) patch pixel
```

点可能落在 `[0,255]` 之外，这表示预测关节点位于裁剪区域外。

### 11.12 full ONNX 输出顺序

full 图声明的输出顺序为：

```text
poses_6d
betas
camera
poses
camera_translation
joints_44_body
joints_44_camera
joints_24_anatomical
joints_24_custom
joints_2d_normalized
joints_2d_patch
```

只有导出时使用 `--include-skin` 才会额外包含：

```text
skin_vertices：[B,6890,3]
```

---

## 12. 当前 runtime 为什么拿不到所有 joint 输出

虽然 full ONNX 声明了 joint 输出，`runtime.py` 当前只请求：

```python
["poses", "betas", "camera_translation"]
```

然后统一返回：

```python
{
    "pd_params": {
        "poses": poses,
        "betas": betas,
    },
    "pd_cam_t": camera_translation,
}
```

所以 `run_webcam.py` 的 `outputs` 中没有：

```text
joints_44_body
joints_44_camera
joints_2d_patch
```

如果只想 Debug，不改代码，可以在 `runtime.py:186` 的断点里查看：

```python
[x.name for x in self.session.get_outputs()]
```

并理解完整输出是否真实存在。正式提取时需要调整 runtime 的请求输出和返回结构，但这不属于本文对现有代码的修改。

---

## 13. `prepare_mesh()` 和 full ONNX 的关系

`run_webcam.py` 从 pipeline 取出参数后，调用：

```text
lib/kits/hsmr_demo.py:161
prepare_mesh()
```

它重新把 CPU 参数移到 GPU：

```python
poses = pd_params["poses"].to(pipeline.device)
betas = pd_params["betas"].to(pipeline.device)
```

然后调用 PyTorch SKEL：

```python
skel_outputs = pipeline.skel_model(
    poses=poses,
    betas=betas,
    skelmesh=include_skeleton,
)
```

可得到：

```text
skin_verts：[B,6890,3]
skel_verts：[B,Ve,3]
joints：[B,44,3]
joints_backup：[B,24,3]
joints_custom：[B,24,3]
```

但当前 `prepare_mesh()` 只保留皮肤和骨骼网格顶点：

```python
m_skin = {"v": skin_verts, "f": skin_faces}
m_skel = {"v": skel_verts, "f": skel_faces}
```

这意味着当前使用 full ONNX 时可能出现：

```text
full ONNX 内部包含一次 SKEL joint 后处理
                         +
prepare_mesh 再执行一次 PyTorch SKEL 生成渲染网格
```

这也是理解性能和 `/postprocessor/skel_model/...` 报错的重要背景。

---

## 14. `pd_cam_t` 怎样转换成 `raw_cam_t`

`pd_cam_t` 对应 `256×256` 人物 patch，而渲染需要回到完整彩色图。

`visualize_full_img()` 位于：

```text
lib/kits/hsmr_demo.py:236
```

使用 bbox `[cx,cy,size]` 修正：

```python
raw_cam_t_z = pd_cam_t_z * 256 / bbox_size
raw_cam_t_y += (bbox_cy - image_cy) / 5000 * raw_cam_t_z
raw_cam_t_x += (bbox_cx - image_cx) / 5000 * raw_cam_t_z
```

作用：

- bbox 越大，完整图上的估计深度越小。
- bbox 越小，完整图上的估计深度越大。
- bbox 不在图像中心时，修正 x/y 平移。

渲染仍使用假设内参：

```text
fx = fy = 5000
cx = raw_width / 2
cy = raw_height / 2
```

所以 `raw_cam_t` 仍然是 HSMR 渲染坐标，不是 RealSense 实测坐标。

当前调用写成：

```python
rendered, _ = visualize_full_img(...)
```

Python 中 `_` 只是普通变量。在 `run_webcam.py:579` 打断点后，`_` 就是返回的 `raw_cam_t`。

### 14.1 “虚拟相机”到底是什么

虚拟相机不是另一台真实摄像头，也不是 RealSense 内部的一路传感器。它是程序为了把 HSMR 生成的三维人体重新投影到二维图片上而建立的一套数学针孔相机。可以类比 Blender：先在三维场景中生成一个以骨盆为根的人体，再放置一台数学相机，调整整个人体相对相机的位置，使渲染轮廓与 RGB 图片中的人重合。

当前 patch 虚拟相机使用以下固定假设：

| 参数 | 当前值/含义 |
|---|---|
| 图像大小 | `256×256` 人物 patch |
| 焦距 | `fx=fy=5000` |
| 主点 | patch 中心 `(128,128)` |
| 相机旋转 | 单位旋转，即投影阶段不额外旋转人体 |
| 相机平移 | 模型根据人体在 patch 中的尺度和位置估计 |

三维人体最初是根相对的：

```text
joints_44_body：以人体根节点/骨盆附近为参考的关节点
camera_translation：把整个人放到虚拟相机前面的平移
joints_44_camera = joints_44_body + camera_translation
```

将关节点投影到 patch 的公式等价于：

```text
u_patch = 5000 × X_camera / Z_camera + 128
v_patch = 5000 × Y_camera / Z_camera + 128
```

这里的 `camera_translation=[tx,ty,tz]` 可通俗理解为：`tx` 控制整个人在 patch 中左右移动，`ty` 控制上下移动，`tz` 控制投影后人体的大小。模型核心先预测弱透视参数：

```text
camera = [scale, tx, ty]
```

再计算：

```text
tz = 2 × 5000 / (256 × scale + 1e-9)
```

因此图像中的人越大，模型一般预测越大的 `scale`，对应越小的虚拟 `tz`；人越小则相反。这是在单目 RGB 的尺度歧义下，为了让三维人体投影尺寸与图片一致而作的估计，不是深度传感器测量。

### 14.2 为什么它不等于真实 RealSense 相机

两者区别如下：

| 项目 | HSMR 虚拟相机 | RealSense 真实相机 |
|---|---|---|
| 是否为物理设备 | 否，只是数学模型 | 是 |
| 焦距和主点 | 使用 HSMR 固定假设 | 来自 `CameraInfo` 标定 |
| 镜头畸变 | 不按当前 RealSense 的真实畸变处理 | 由 `CameraInfo.D` 描述 |
| 深度 | 根据单张 RGB 中人体大小估计 | 深度传感器逐像素测量 |
| 坐标单位 | 跟随人体模型尺度，不能直接认定为精确米制 | 反投影后通常使用米 |
| 与机器人关系 | 未连接 `base_link` | 可通过 ROS TF 转换 |

只有彩色 RGB 时，一个“小人”既可能是正常身高的人站得远，也可能是较矮的人站得近，单张图片不能唯一确定真实尺度。所以虚拟相机的主要任务是“投影对齐”，不是“物理测距”。

### 14.3 `pd_cam_t` 和 `raw_cam_t` 属于哪台相机

两者都属于 HSMR 虚拟相机链路：

```text
pd_cam_t：相对 256×256 人物 patch 的虚拟相机平移
raw_cam_t：加入 bbox 位置和大小后，对齐完整 RGB 图的虚拟相机平移
```

`raw_cam_t` 只是撤销人物裁剪带来的平移和尺度变化，并没有加入 RealSense 深度或真实内参，因此不会因为映射回原图就自动变成真实相机坐标。

### 14.4 orientation 相对虚拟相机是什么意思

根 orientation 表示人在输入画面中整体是正对、背对、左转还是右转；它与虚拟相机/模型坐标轴有关。局部关节 orientation 则描述小腿相对大腿、前臂相对上臂等人体内部旋转。要得到机器人 `base_link` 下的朝向，还必须确认 HSMR 坐标轴到 RealSense optical frame 的固定轴映射，再与图像采集时刻的 TF 旋转相乘。详细公式见第 29 节。

### 14.5 最简单的记忆方式

```text
虚拟相机：让三维人体正确画回 RGB 图片，位置和深度主要是模型估计
真实相机：用 CameraInfo + 对齐深度得到真实相机 XYZ
机器人坐标：用 ROS TF 把真实相机 XYZ/orientation 转到 base_link
```

---

## 15. 把 `joints_2d_patch` 对齐回完整彩色图

要用深度图查询关节点深度，必须先把 patch 像素变回原始彩色图像素。

设：

```text
patch 点 = (u_patch, v_patch)
bbox = (cx, cy, s)
patch_size = 256
```

方形裁剪框左上角近似为：

```text
left = cx - s/2
upper = cy - s/2
```

映射回彩色图：

```text
u_color = left  + u_patch × s / 256
v_color = upper + v_patch × s / 256
```

也可以写成以 patch 中心为基准：

```text
u_color = cx + (u_patch - 128) × s / 256
v_color = cy + (v_patch - 128) × s / 256
```

### 15.1 当前实现中的整数取整误差

`crop_with_lurb()` 在裁剪时把 bbox 转为 `int64`，而 `bbx_cs` 保存的是转换前的浮点中心和尺度。因此上面的映射通常只有亚像素到约 1 像素量级误差。

如果后续要求严格像素对齐，建议未来保存“实际用于裁剪的整数 `lurb` 和实际 crop 宽高”，再使用完全相同的参数做逆变换。

### 15.2 越界检查

映射后必须检查：

```text
0 <= u_color < color_width
0 <= v_color < color_height
```

如果超出，说明关节点落在原图之外，不能直接读取深度。

---

## 16. 当前 Foxglove 输入为什么还不能做 RGB-D

当前 `FoxgloveLatestFrameCapture` 位于：

```text
lib/platform/foxglove_camera.py
```

它只订阅一个 topic，并假定 `message.data` 是 JPEG/PNG 压缩数据：

```python
encoded = np.frombuffer(message.data, dtype=np.uint8)
frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
```

这适用于：

```text
/color/image_raw/compressed
```

但不适用于原始深度 `sensor_msgs/Image`。原始深度需要根据：

```text
height
width
encoding
is_bigendian
step
data
```

恢复为二维 `uint16` 或 `float32` 数组。

此外当前 capture 保存的是本机接收时间：

```python
time.time()
```

没有保存 ROS 消息 `header.stamp`。做彩色—深度—TF 融合时必须使用传感器采集时间进行同步，不能只用 WebSocket 到达本机的时间。

---

## 17. 什么叫“对齐到彩色图的深度图”

RealSense 的彩色传感器和深度传感器不是同一个成像器：

```text
彩色相机有自己的分辨率、内参和光心
深度相机有自己的分辨率、内参和光心
两者之间还有固定旋转和平移外参
```

所以原始彩色像素 `(u,v)` 与原始深度图同一索引 `(u,v)` 不一定表示同一条空间射线。

“深度对齐到彩色图”表示驱动利用：

```text
深度内参
+ 深度值
+ 深度到彩色外参
+ 彩色内参
```

把深度重新投影到彩色图像素网格。对齐后，才可以对彩色图中的关节点 `(u_color,v_color)` 读取：

```python
depth_aligned[v_color, u_color]
```

RealSense ROS2 驱动通常通过类似参数启用：

```text
enable_color:=true
enable_depth:=true
enable_sync:=true
align_depth.enable:=true
```

常见话题形式是：

```text
/.../color/image_raw
/.../color/camera_info
/.../aligned_depth_to_color/image_raw
/.../aligned_depth_to_color/camera_info
```

具体前缀取决于机器人 launch 配置，必须在目标设备上确认，不能直接假定。

建议检查：

```bash
ros2 topic list | grep -E 'color|depth|camera_info'
```

```bash
ros2 topic info <aligned_depth_topic> -v
```

```bash
ros2 topic echo <color_camera_info_topic> --once
```

---

## 18. 彩色、深度和 CameraInfo 的时间同步

正确的一组输入必须尽量来自同一时刻：

```text
color.header.stamp ≈ depth.header.stamp ≈ camera_info.header.stamp
```

推荐原则：

1. 优先由 RealSense 驱动启用硬件/frameset 同步。
2. ROS2 节点中使用 exact 或 approximate message synchronization。
3. TF 查询使用这组图像的采集时间，而不是当前系统时间。
4. 如果 RGB 和 depth 时间差过大，人在运动时会出现“关节点像素对应到了背景深度”。

当前 Foxglove capture 只保留最新彩色帧。要通过 Foxglove 完成同步，需要同时订阅多个 channel，保存每条消息的 `header.stamp`，再按时间戳配对。

更稳妥的工程方式是把 RGB-D 融合封装成 ROS2 `rclpy` 节点，在 ROS 内使用消息同步和 tf2；HSMR 推理仍然可以复用当前 Python pipeline。

---

## 19. 深度值怎么读取

### 19.1 常见编码

检查：

```text
depth_msg.encoding
```

常见情况：

| encoding | NumPy 类型 | 常见单位 |
|---|---|---|
| `16UC1` | `uint16` | 通常毫米或设备 depth unit |
| `32FC1` | `float32` | 通常米 |

RealSense D400 系列常见 depth unit 是 `0.001 m`，但工程代码应从驱动/设备配置确认，不应把所有相机永久写死为 0.001。

深度为 `0`、NaN、Inf 或超出有效量程时，应视为无效。

### 19.2 不要只读单个像素

HSMR 的关节点只是视觉估计，像素可能落在衣服边缘、遮挡区域或背景。推荐在关节点周围取一个小窗口：

```text
半径 2~4 像素
过滤 0、NaN、Inf 和量程外值
取有效值中位数
```

示例逻辑：

```python
def sample_depth_m(depth, encoding, u, v, radius=3, depth_scale=0.001):
    u = int(round(u))
    v = int(round(v))
    h, w = depth.shape
    if not (0 <= u < w and 0 <= v < h):
        return None

    u0 = max(0, u - radius)
    u1 = min(w, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(h, v + radius + 1)
    values = depth[v0:v1, u0:u1].astype(np.float32)

    if encoding == "16UC1":
        values *= depth_scale
    elif encoding != "32FC1":
        raise ValueError(f"Unsupported depth encoding: {encoding}")

    valid = np.isfinite(values) & (values > 0.1) & (values < 10.0)
    if not valid.any():
        return None
    return float(np.median(values[valid]))
```

`0.1~10.0 m` 只是示例范围，应根据相机型号和机器人场景配置。

---

## 20. `CameraInfo` 是什么，如何反投影到真实相机 XYZ

`sensor_msgs/CameraInfo` 提供真实相机标定信息。最重要的是内参矩阵：

```text
K = [fx  0 cx
      0 fy cy
      0  0  1]
```

如果得到彩色图像素：

```text
(u,v)
```

以及对齐深度：

```text
Z，单位米
```

针孔相机反投影公式为：

```text
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
Z = depth
```

代码形式：

```python
def deproject_pixel(u, v, z_m, camera_info):
    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]
    x = (u - cx) * z_m / fx
    y = (v - cy) * z_m / fy
    return np.array([x, y, z_m], dtype=np.float64)
```

### 20.1 K 还是 P

- 原始、带畸变图像对应 `K` 和畸变系数 `D`。
- 已校正图像通常使用投影矩阵 `P` 左上角的 `fx',fy',cx',cy'`。
- 如果输入仍有明显镜头畸变，直接套针孔公式会产生偏差，应先用 `image_geometry`/OpenCV 校正点或直接使用 rectified color 图。

必须保证彩色图、对齐深度和 CameraInfo 描述的是同一个像素网格、同一分辨率和同一相机光学 frame。

### 20.2 RealSense 光学坐标方向

ROS 相机光学 frame 通常是：

```text
X：图像右方
Y：图像下方
Z：摄像头向前
```

所以一个点：

```text
[0.2, 0.1, 2.0] m
```

表示它在相机右侧 0.2 m、下方 0.1 m、前方 2.0 m。

来源 frame 应优先读取：

```python
camera_info.header.frame_id
```

而不是硬编码一个猜测名称。

---

## 21. `/tf`、`/tf_static` 和 `base_link` 转换

### 21.1 TF 不是 TensorFlow

这里的 TF 是 ROS 坐标变换系统。它描述：

```text
camera optical frame
→ camera_link
→ head_link
→ base_link
→ odom
→ map
```

`/tf_static` 通常保存固定安装关系，例如彩色光学 frame 到相机 link；`/tf` 保存会随时间变化的关系，例如头部关节、机器人里程计和底盘运动。

### 21.2 坐标轴不同

典型 optical frame：

```text
x 向右，y 向下，z 向前
```

典型 `base_link`：

```text
x 向前，y 向左，z 向上
```

所以转换不只是加相机安装高度，还包括旋转坐标轴。

### 21.3 TF 数学公式

查询得到从 source camera frame 到 target `base_link` 的变换：

```text
T_base_camera = [R t
                 0 1]
```

对相机点：

```text
p_base = R_base_camera × p_camera + t_base_camera
```

### 21.4 ROS2 Python 查询方式

节点初始化时：

```python
from tf2_ros import Buffer, TransformListener

self.tf_buffer = Buffer()
self.tf_listener = TransformListener(self.tf_buffer, self)
```

处理一组同步 RGB-D 消息时：

```python
from rclpy.duration import Duration
from rclpy.time import Time

source_frame = camera_info.header.frame_id
stamp = Time.from_msg(color_msg.header.stamp)

transform = self.tf_buffer.lookup_transform(
    "base_link",       # target frame
    source_frame,      # source frame
    stamp,             # 图像采集时刻
    timeout=Duration(seconds=0.1),
)
```

记忆顺序：

```text
lookup_transform(target_frame, source_frame, time)
```

查询的结果就是把 source 中的点转换到 target 所需要的变换。

创建 `TransformListener` 后，它会接收 `/tf` 和 `/tf_static`；通常不需要业务代码分别手写两个订阅器。

### 21.5 把 TransformStamped 应用到 NumPy 点

下面是学习用的纯 NumPy 示例：

```python
def quaternion_to_matrix(x, y, z, w):
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm == 0:
        raise ValueError("zero quaternion")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

def transform_points(points_camera, transform_stamped):
    q = transform_stamped.transform.rotation
    t = transform_stamped.transform.translation
    rotation = quaternion_to_matrix(q.x, q.y, q.z, q.w)
    translation = np.array([t.x, t.y, t.z], dtype=np.float64)
    return points_camera @ rotation.T + translation
```

输入：

```text
points_camera：[J,3]
```

输出：

```text
points_base_link：[J,3]
```

### 21.6 TF Debug

先确认 frame 名称：

```bash
ros2 topic echo <color_camera_info_topic> --once
```

然后检查变换链：

```bash
ros2 run tf2_ros tf2_echo base_link <camera_optical_frame>
```

如果查不到，常见原因是：

- frame 名拼错。
- 机器人没有发布相机安装外参。
- 查询时间超出 TF buffer。
- 使用最新 TF 处理旧图像，时间不匹配。
- 只在 Foxglove 客户端收到图像，但当前 Python 进程不是 ROS2 节点，也没有 TF buffer。

---

## 22. 从 HSMR 到真实 `base_link` 关节点的完整推荐流程

### 22.1 输入数据

每个同步样本至少包含：

```text
color image
aligned depth image
color CameraInfo
ROS timestamp
camera optical frame_id
TF buffer
```

### 22.2 HSMR 推理

```text
color image
→ detector
→ bbox
→ person patch
→ HSMR
→ joints_2d_patch
```

### 22.3 patch 像素回原始彩色图

对每个人、每个关节点：

```text
(u_patch,v_patch) + bbox[cx,cy,s]
→ (u_color,v_color)
```

### 22.4 读取对应深度

```text
(u_color,v_color)
→ aligned_depth 小窗口
→ 中位数 Z，单位米
```

### 22.5 CameraInfo 反投影

```text
(u_color,v_color,Z,K)
→ [X_camera,Y_camera,Z_camera]
```

这一步得到 RealSense 光学 frame 中的真实测量点。

### 22.6 TF 转到机器人

```text
p_camera
+ T_base_camera(at image timestamp)
→ p_base_link
```

最终输出可以设计为：

```python
{
    "person_index": 0,
    "stamp": "...",
    "camera_frame": "..._color_optical_frame",
    "target_frame": "base_link",
    "joints_2d_color": [[u,v], ...],
    "joints_depth_valid": [True, False, ...],
    "joints_camera_m": [[x,y,z], ...],
    "joints_base_link_m": [[x,y,z], ...],
}
```

### 22.7 推荐保留有效性标记

不要用 `[0,0,0]` 代表无效关节，因为 `[0,0,0]` 本身可能是合法坐标原点。应单独保存：

```text
valid
depth_valid
tf_valid
```

---

## 23. 两种三维融合策略

### 策略 A：每个关节分别读取深度

```text
每个2D关节点 → 深度 → 反投影 → 真实XYZ
```

优点：

- 每个有效关节都有真实深度测量。
- 不依赖 HSMR 单目深度尺度。

缺点：

- 遮挡关节没有可靠深度。
- 手腕、脚踝等小部位容易读到背景。
- 深度噪声会破坏骨骼长度一致性。

适合先实现和验证。

### 策略 B：深度确定人体根位置，HSMR 提供相对骨架

```text
骨盆/躯干区域深度 → 真实根位置
HSMR joints_body → 相对骨架
根位置 + 旋转/尺度对齐 → 完整骨架
```

优点：

- 骨架长度和姿态更连续。
- 对部分关节深度缺失更鲁棒。

缺点：

- 必须确认 HSMR body 坐标的轴、尺度和根节点定义。
- 必须处理人体朝向旋转。
- 不能简单写成 `joints_body + real_root` 就认为已经正确。

推荐先完成策略 A，确认像素、深度、CameraInfo 和 TF 全部对齐后，再研究策略 B。

---

## 24. 多人情况下还缺少“身份跟踪”

当前 `max_instances` 只是每帧保留前 K 个高分人体。检测顺序不保证跨帧稳定：

```text
上一帧 person 0 可能是甲
下一帧 person 0 可能变成乙
```

如果要输出持续的三维人物轨迹，还需要 tracker，根据：

- bbox IoU
- 外观特征
- 2D/3D位置连续性
- 时间戳

维护稳定 `track_id`。

这与 ONNX 静态 batch=1 是两个不同问题：

```text
static batch=1：影响一次怎样推理多人
tracking：影响跨帧怎样认出同一个人
```

---

## 25. 建议的代码集成位置，但本文不修改现有代码

未来实现时可以按职责拆分：

### 25.1 输入层

扩展或新增 RGB-D capture，保存：

```text
frame_bgr
aligned_depth
depth_encoding
camera_info
header_stamp
frame_id
```

如果继续使用 Foxglove，需要支持多 channel 和 ROS 时间戳配对；如果使用 ROS2 `rclpy`，推荐用消息同步器。

### 25.2 ONNX runtime 输出层

正式需要关节点时，让 full ONNX runtime 返回：

```text
joints_44_body
joints_2d_patch
```

或者继续只返回参数，在 `prepare_mesh()` 的 PyTorch SKEL 调用处提取 `skel_outputs.joints` 并执行 2D 投影。

不要同时计算两套 joint 却只使用一套，否则会增加性能开销和维护复杂度。

### 25.3 `infer_latest_frame()` 融合层

在获得：

```text
joints_2d_patch + bbx_cs
```

之后执行：

```text
patch_to_color
→ sample_depth
→ deproject
→ tf_transform
```

渲染仍可使用现有 `pd_cam_t/raw_cam_t`，机器人坐标输出则使用深度+CameraInfo+TF，两个链路不要混用。

---

## 26. 完整 Debug 路线

### 26.1 检测器输出

断点：`exp/run_webcam.py:511`

```python
detector_outputs[0][0]
detector_outputs[1][0]
```

### 26.2 人体裁剪

断点：`exp/run_webcam.py:523`

```python
persons
patches.shape
bbx_cs
```

### 26.3 HSMR 输入

断点：`exp/run_webcam.py:545`

```python
patches_normalized.shape
patches_normalized.dtype
np.isfinite(patches_normalized).all()
```

### 26.4 ONNX 模型元数据

断点：`deploy/onnx/runtime.py:144`

```python
[(x.name,x.shape,x.type) for x in self.session.get_inputs()]
[(x.name,x.shape,x.type) for x in self.session.get_outputs()]
self.graph_scope
self.providers
```

### 26.5 每个人的静态 batch

断点：`deploy/onnx/runtime.py:186`

```python
index
len(patches)
image.shape
image.dtype
self.run_output_names
```

### 26.6 ONNX 原始返回值

断点：`deploy/onnx/runtime.py:190`

```python
[(n,v.shape,v.dtype) for n,v in zip(self.run_output_names,values)]
```

### 26.7 pipeline 标准返回

断点：`exp/run_webcam.py:549`

```python
outputs.keys()
outputs["pd_params"]["poses"].shape
outputs["pd_params"]["betas"].shape
outputs["pd_cam_t"].shape
```

### 26.8 PyTorch SKEL 真正三维数据

断点：`lib/kits/hsmr_demo.py:186`

```python
skel_outputs.joints.shape
skel_outputs.joints_backup.shape
skel_outputs.joints_custom.shape
skel_outputs.skin_verts.shape
skel_outputs.skel_verts.shape
```

### 26.9 原图虚拟相机平移

断点：`lib/kits/hsmr_demo.py:259`

```python
pd_cam_t
bbx_cs
raw_cam_t_i
```

### 26.10 RGB-D 和 TF

未来接入后检查：

```text
color stamp
depth stamp
stamp difference
color resolution
depth resolution
camera_info resolution
camera_info frame_id
depth encoding
depth unit
joint patch pixel
joint color pixel
sampled depth meter
camera XYZ
TF source/target/stamp
base_link XYZ
```

---

## 27. `/postprocessor/skel_model/Reshape_185` 错误如何放在整个流程里理解

报错：

```text
input shape {0}, requested shape {}
```

发生在：

```text
session.run()
→ full ONNX postprocessor
→ exported SKEL graph
→ Reshape_185
```

它不是：

- detector 输出格式错误。
- `_img_det2patches()` 没检测到人，因为无人时根本不会调用 pipeline。
- `prepare_mesh()` 的 PyTorch SKEL 错误，因为异常在 `session.run()` 内已经抛出。

这个节点路径说明 full ONNX 内部某个 SKEL 中间张量在当前输入下变成空 tensor，随后试图 reshape 成标量或固定形状。

Debug 时优先确认：

```text
输入确实是 [1,3,256,256]
输入 finite
模型确实是 static B=1
错误是否只发生在特定人物/姿态
core ONNX 是否能通过
full ONNX 的全部输出和节点版本是否与导出环境一致
```

core 模型可以用来区分“视觉核心网络问题”与“ONNX SKEL 后处理问题”，但这只是诊断思路，不等于最终必须改用 core。

---

## 28. 常见误区检查表

- [ ] 不把 `poses` 当成 XYZ 坐标。
- [ ] 不把 `poses[:,0:3]` 直接命名为普通 yaw/pitch/roll。
- [ ] 不把 `betas` 当成身高、体重等一一对应指标。
- [ ] 不把 `pd_cam_t` 当成 RealSense 实测位置。
- [ ] 不把 `raw_cam_t` 当成 `base_link` 坐标。
- [ ] 不直接用原始深度图的同一像素索引查询彩色关节点。
- [ ] 使用对齐到彩色图的深度。
- [ ] 使用与彩色像素网格匹配的 CameraInfo。
- [ ] 处理镜头畸变或使用 rectified 图。
- [ ] 把 `16UC1` 深度单位正确换算成米。
- [ ] 过滤深度 0、NaN、Inf 和背景值。
- [ ] 彩色、深度、CameraInfo 按 ROS header 时间同步。
- [ ] TF 使用图像采集时间查询。
- [ ] source frame 从消息 header/CameraInfo 读取。
- [ ] 明确 optical frame 与 `base_link` 坐标轴不同。
- [ ] 多人跨帧需要 tracker，不能依赖数组顺序。

---

## 29. Orientation：能不能得到人的朝向

可以得到，但必须区分“整体朝向”“关节局部旋转”和“机器人坐标下的朝向”。

| Orientation 类型 | 当前模型是否提供 | 相对于谁 |
|---|---|---|
| 骨盆/人体根 orientation | 提供 | HSMR 虚拟相机/模型坐标轴 |
| 四肢关节 orientation | 提供 | 每个关节的父关节 |
| RealSense 光学 frame 下的人体 orientation | 没有直接标定输出 | 需要确认 HSMR 到 optical frame 的轴转换 |
| `base_link` 下的人体 orientation/yaw | 没有直接输出 | 需要相机 orientation + ROS TF 旋转合成 |

### 29.1 根 orientation 在哪个变量里

核心网络预测：

```text
poses_6d：[B,24,6]
```

其中第 0 个关节是 pelvis：

```python
root_pose_6d = poses_6d[:,0,:]
```

经过 `params_rep2q_onnx()` 转换后，根 orientation 位于：

```python
root_q = poses[:,0:3]
```

所以当前 `run_webcam.py` 即使没有返回 `poses_6d`，仍然可以在 `pipeline()` 之后看到根姿态信息：

```python
poses = outputs["pd_params"]["poses"]  # [N,46]
root_q = poses[:,0:3]                   # [N,3]
```

但 `root_q` 是 SKEL 的 Euler-like 广义参数，不应直接把三个数重命名为：

```text
[roll, pitch, yaw]
```

因为 SKEL pelvis 使用自己的旋转轴和顺序。

### 29.2 怎样得到真正可使用的旋转矩阵

项目已经提供 SKEL 参数到旋转矩阵的转换：

```python
from lib.body_models.skel_utils.transforms import params_q2rot

rotations = params_q2rot(poses)  # [N,24,3,3]
root_rotation = rotations[:,0]   # [N,3,3]
```

`root_rotation` 是骨盆/整个人体的根旋转矩阵。后面的 23 个矩阵是各运动学关节的局部旋转。

如果直接拿到 core/full ONNX 的 `poses_6d`，也可以：

```python
from lib.utils.geometry.rotation import rotation_6d_to_matrix

poses_6d = poses_6d.reshape(-1,24,6)
root_rotation_6d = rotation_6d_to_matrix(poses_6d[:,0])
```

当前 full ONNX 文件声明了 `poses_6d` 输出，但 `runtime.py` 的 `run_output_names` 没有请求它；当前标准 `outputs` 中可直接使用的是 `[N,46]` 的 `poses`。

### 29.3 它是相对人自己，还是相对摄像头

答案分两类。

#### 人体根 orientation

骨盆根旋转描述整个人怎样朝向 HSMR 的模型/虚拟相机坐标轴。原因是：

1. HSMR 从相机图像估计人的整体姿态。
2. SKEL 顶点先由根 orientation 旋转。
3. 投影代码的相机 rotation 默认是单位矩阵。
4. 再加入 `camera_translation` 将人体放到虚拟相机前方。

因此根 orientation 不是“只相对人自己而没有外部参考”。它携带了人相对输入相机画面方向的整体朝向信息。

但是它的参考 frame 是 HSMR/SKEL 约定的虚拟坐标轴，并未通过 `CameraInfo` 或 TF 声明为真实的 RealSense `color_optical_frame`。

#### 四肢关节 orientation

例如膝、肘、肩的旋转属于人体运动学树中的局部旋转：

```text
小腿相对大腿
前臂相对上臂
头部相对胸部/颈部
```

这部分是“相对人体父关节”的，不是相对机器人摄像头。

经过前向运动学把所有父子旋转连乘以后，才得到某个部位在根坐标或相机坐标中的最终 orientation。

### 29.4 裁剪 bbox 会不会改变 orientation

当前 `_img_det2patches()` 只做：

```text
平移裁剪
尺度缩放
边界填充
```

它没有旋转图像。因此 bbox 中心和 bbox 大小会影响人物平移、尺度以及 `raw_cam_t`，但不会引入额外的图像平面旋转。

不过 HSMR 使用固定焦距和虚拟相机约定，所以仍不能把根旋转矩阵不加验证地当作已经标定的 RealSense orientation。

### 29.5 HSMR orientation 到真实相机 orientation

设：

```text
R_hsmr_person：HSMR 根旋转
R_optical_hsmr：HSMR 坐标轴到 RealSense optical frame 的固定轴转换
```

则真实相机光学 frame 中的人体 orientation 是：

```text
R_optical_person = R_optical_hsmr × R_hsmr_person
```

当前代码没有显式定义 `R_optical_hsmr`。不能仅根据变量名字假定它一定是单位矩阵。确定它的方法包括：

1. 检查 SKEL 中人体零姿态的前、上、右方向。
2. 用已知正对、背对、左转、右转的人体样本验证旋转矩阵。
3. 将旋转后的身体 forward vector 投影到图像，检查方向是否符合画面。
4. 确认是否需要固定轴交换或符号翻转。

项目的 SKEL 变换代码注明，人体左右定义以人体面向 `+Z` 为基准。因此可以先把 SKEL 局部 forward axis 设为：

```python
forward_hsmr_local = np.array([0.0, 0.0, 1.0])
```

再计算：

```python
forward_hsmr = R_hsmr_person @ forward_hsmr_local
```

但在用于机器人控制前，仍应通过实际样本验证这个方向和符号。

### 29.6 从相机 orientation 转到 `base_link`

TF 中从相机光学 frame 到 `base_link` 的变换包含：

```text
R_base_optical：旋转
t_base_optical：平移
```

点坐标转换需要旋转和平移：

```text
p_base = R_base_optical × p_optical + t_base_optical
```

orientation 转换只需要旋转相乘，不需要平移：

```text
R_base_person = R_base_optical × R_optical_person
```

合并 HSMR 到 optical 的固定轴变换后：

```text
R_base_person
    = R_base_optical
    × R_optical_hsmr
    × R_hsmr_person
```

由于头部摄像头可能随机器人头部运动，`R_base_optical` 必须使用图像采集时间对应的 TF，不能一直使用启动时的一次结果。

### 29.7 怎样计算机器人坐标下“人朝哪个方向”

使用人体 forward vector 比直接读取 `root_q[2]` 更清楚。

假设已经得到：

```text
R_base_person
```

并确认 SKEL 人体局部前方是 `+Z`：

```python
forward_person = np.array([0.0, 0.0, 1.0])
forward_base = R_base_person @ forward_person
```

典型 ROS `base_link` 使用：

```text
x 向机器人前方
y 向机器人左方
z 向上
```

那么忽略俯仰，只看地面平面的 yaw：

```python
yaw_base = np.arctan2(forward_base[1], forward_base[0])
yaw_base_deg = np.degrees(yaw_base)
```

可解释为：

```text
0°：人朝机器人 base_link 的 +X 方向
+90°：朝 base_link 的 +Y，即机器人左侧
-90°：朝 base_link 的 -Y，即机器人右侧
±180°：朝 base_link 的后方
```

这些角度解释成立的前提是：

- `R_optical_hsmr` 已经验证正确。
- TF 的 source/target 顺序正确。
- forward axis 已通过样本确认。

### 29.8 只有 orientation，能不能知道人是否面向机器人

可以在 `base_link` 中比较两个方向：

```text
人的前向向量 forward_base
人指向机器人的向量 robot_position - person_position
```

归一化后计算点积：

```python
cos_angle = np.dot(
    forward_base_normalized,
    person_to_robot_normalized,
)
```

```text
cos_angle 接近 1：人基本面向机器人
cos_angle 接近 0：人大致侧对机器人
cos_angle 接近 -1：人基本背对机器人
```

这里必须同时具有：

1. 深度 + CameraInfo + TF 得到的真实人体位置。
2. 转换到同一个 `base_link` frame 的人体 orientation。

不能把 `pd_cam_t` 位置与 `base_link` orientation 混合计算。

### 29.9 orientation 的可靠性限制

HSMR orientation 来自单张 RGB 图像推断，不是 IMU 或运动捕捉系统直接测量。以下情况可能发生前后方向或角度误差：

- 人被遮挡。
- 只看到上半身或背影。
- 左右肢体重叠。
- 多人互相遮挡。
- 图像模糊。
- 人体接近正侧面，前后信息不足。

深度可以改善真实位置，但单独的深度值不会自动修正 HSMR 根 orientation。实际系统建议对 orientation 做时间滤波，并根据肩、髋关节点几何方向做一致性检查。

### 29.10 orientation 推荐输出格式

在未完成真实轴标定时，应明确标记 frame：

```python
{
    "orientation_frame": "hsmr_virtual_camera",
    "root_q": [q0, q1, q2],
    "root_rotation_matrix": [[...], [...], [...]],
    "calibrated_to_ros_optical": False,
}
```

完成 HSMR 轴验证和 TF 转换后，可以输出：

```python
{
    "orientation_frame": "base_link",
    "root_rotation_matrix": [[...], [...], [...]],
    "forward_vector": [fx, fy, fz],
    "yaw_rad": yaw,
    "yaw_deg": yaw_deg,
    "calibrated_to_ros_optical": True,
}
```

### 29.11 orientation Debug 位置

在 `exp/run_webcam.py:549` 查看当前已返回的根姿态：

```python
poses = outputs["pd_params"]["poses"]
poses.shape
poses[:,0:3]
```

在 `deploy/onnx/runtime.py:190` 检查 full ONNX 是否声明并能够返回 `poses_6d`：

```python
[(x.name,x.shape,x.type) for x in self.session.get_outputs()]
```

在获得旋转矩阵后，至少用四类图片检查：

```text
正对摄像头
背对摄像头
向左侧身
向右侧身
```

不要只用一帧判断轴映射和正负号。

---

## 30. 官方参考资料

- ROS `CameraInfo` 消息和内参矩阵：<https://docs.ros.org/en/humble/p/sensor_msgs/msg/CameraInfo.html>
- ROS `Image` 光学 frame 方向和时间戳约定：<https://docs.ros.org/en/humble/p/sensor_msgs/msg/Image.html>
- ROS2 tf2 Python listener 教程：<https://docs.ros.org/en/iron/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html>
- ROS2 `tf2_ros.Buffer.lookup_transform` API：<https://docs.ros.org/en/kilted/p/tf2_ros_py/tf2_ros.buffer.html>
- `image_geometry::PinholeCameraModel`：<https://docs.ros.org/en/ros2_packages/humble/api/image_geometry/generated/classimage__geometry_1_1PinholeCameraModel.html>
- RealSense ROS2 对齐深度配置和话题说明：<https://github.com/realsenseai/realsense-ros>
- RealSense 2D/3D 投影、深度单位和 frame alignment：<https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0>

---

## 31. 一句话总结

```text
当前 HSMR：RGB → 人框 → 单人patch → ONNX姿态/形状/虚拟相机 → SKEL网格 → 图像渲染

真实机器人三维：HSMR二维关节点 → 原图像素 → 对齐深度 → CameraInfo反投影 → 相机XYZ → TF → base_link XYZ

真实机器人朝向：HSMR根旋转 → 验证HSMR到光学坐标轴映射 → TF旋转合成 → base_link前向向量/yaw
```
