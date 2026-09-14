# HSMR-render Dockerfile — x86_64, CUDA 12.4, Ubuntu 22.04
#
# 职责: SKEL 蒙皮+骨骼 2D 叠加渲染服务 (EGL headless)
# 构建: docker build -t hsmr-render:latest .
# 运行: 见下方说明 — 需挂载 HSMR-main 和模型数据
#
# 挂载说明:
#   -v /home/kai/HSMR/HSMR-main:/app/HSMR-main:ro          (lib/deploy/thirdparty)
#   -v /home/kai/HSMR/HSMR-render:/app/HSMR-render          (本服务代码)
#   -v /home/kai/HSMR/HSMR-infer/data_inputs:/app/HSMR-main/data_inputs:ro    (模型)
#   -v /home/kai/HSMR/HSMR-infer/deploy/onnx/artifacts:/app/HSMR-main/deploy/onnx/artifacts:ro
#   -v /home/kai/HSMR/HSMR-render/data_outputs:/app/HSMR-render/data_outputs

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYOPENGL_PLATFORM=egl

# ── 系统依赖 (含 EGL/OpenGL 渲染运行库) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils \
    curl git wget cmake build-essential \
    libegl1-mesa libegl1 libgl1-mesa-glx libgles2-mesa \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libosmesa6 mesa-utils \
    libgomp1 libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# ── venv ──
RUN python3.10 -m venv /opt/hsmr-venv
ENV PATH=/opt/hsmr-venv/bin:$PATH

# ── PyTorch 2.1.0 + CUDA 12.1 (兼容 detectron2 v0.6) ──
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── detectron2 (需 torch 已安装, 从源码编译) ──
RUN pip install --no-cache-dir \
    'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# ── ONNX Runtime GPU ──
RUN pip install --no-cache-dir \
    onnx>=1.16 onnxruntime-gpu>=1.18

# ── 渲染相关 (pyrender + PyOpenGL EGL) ──
RUN pip install --no-cache-dir \
    pyrender PyOpenGL PyOpenGL-accelerate \
    trimesh scipy networkx

# ── 服务依赖 ──
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 uvicorn[standard]>=0.30.0 python-multipart>=0.0.9 \
    opencv-python-headless numpy>=1.24 PyYAML>=6.0 \
    websocket-client \
    hydra-core omegaconf pytorch-lightning rich smplx

# 容器默认工作目录和启动命令由 docker-compose / docker run 覆盖
WORKDIR /app/HSMR-render
EXPOSE 8011

CMD ["python", "main.py"]