#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export FOXGLOVE_BRIDGE_URL="${FOXGLOVE_BRIDGE_URL:-ws://192.168.217.100:8768}"
export CAMERA_TOPIC_HEAD="${CAMERA_TOPIC_HEAD:-/zj_humanoid/sensor/realsense_head/color/image_raw/compressed}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hsmr-mpl-cache}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

PYTHON_BIN="${HSMR_PYTHON:-${PROJECT_ROOT}/.venv_orin/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Jetson Python environment not found: ${PYTHON_BIN}" >&2
    exit 1
fi

exec "${PYTHON_BIN}" exp/run_webcam.py \
    --backend onnx \
    --onnx_model deploy/onnx/artifacts/hsmr_full_fp16.onnx \
    --onnx_provider auto \
    --source head \
    --device cuda:0 \
    --interval "${HSMR_INTERVAL:-8}" \
    --max_instances "${HSMR_MAX_INSTANCES:-3}" \
    --mesh_bs 1 \
    --output_path data_outputs/webcam_head_onnx \
    "$@"
