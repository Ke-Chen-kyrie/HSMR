# HSMR-infer Dockerfile — x86_64, CUDA 12.4, Ubuntu 22.04
#
# 构建: docker build -t hsmr-infer:latest .
# 运行: docker run --gpus all -p 8010:8010 -v ./data_inputs:/app/HSMR-infer/data_inputs:ro -v ./deploy/onx/artifacts:/app/HSMR-infer/deploy/onx/artifacts:ro hsmr-infer:latest
#
# HSMR-infer 自包含: lib/ deploy/ thirdparty/ data_inputs/ 都在本目录下
# 只需额外挂载模型文件（若未随 data_inputs COPY 进镜像）

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# ── 系统依赖 ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils \
    curl git wget cmake build-essential \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# ── 创建 venv (与旧工程一致, 便于调试) ──
RUN python3.10 -m venv /opt/hsmr-venv
ENV PATH=/opt/hsmr-venv/bin:$PATH

# ── PyTorch 2.1.0 + CUDA 12.1 (兼容 detectron2 v0.6) ──
# 注: CUDA 12.4 驱动向下兼容 CUDA 12.1 编译的 torch
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── detectron2 (需 torch 已安装, 从源码编译) ──
RUN pip install --no-cache-dir \
    'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# ── ONNX Runtime GPU ──
RUN pip install --no-cache-dir \
    onnx>=1.16 onnxruntime-gpu>=1.18

# ── 服务依赖 ──
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 uvicorn[standard]>=0.30.0 python-multipart>=0.0.9 \
    opencv-python-headless numpy>=1.24 PyYAML>=6.0 \
    hydra-core omegaconf pytorch-lightning rich smplx \
    aiofiles

# ── 应用代码 ──
COPY . /app/HSMR-infer

WORKDIR /app/HSMR-infer

# 暴露端口 (实际端口由 config.yaml 的 server.port 控制, 默认 8010)
EXPOSE 8010

# 默认启动命令; 要改端口请设环境变量 HSMR_INFER_PORT
CMD ["python", "-m", "hsmr_infer.server"]