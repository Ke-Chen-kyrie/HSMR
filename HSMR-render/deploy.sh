#!/usr/bin/env bash
# hsmr-render 一键部署 — 每台机器人一次
# 用法: bash deploy.sh
# 与 docker-compose.yml 同目录; 镜像缺失时本地构建 (构建需 --network=host), 已有则直接用
set -uo pipefail

IMAGE="guopeilin-registry.cn-hangzhou.cr.aliyuncs.com/erban_agent/hsmr-render:20260814"
SERVICE="hsmr-render"
HEALTH_URL="http://127.0.0.1:8011/health"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
WAIT_TIMEOUT=150

echo "=== hsmr-render 一键部署 ==="
echo "镜像: $IMAGE"

command -v docker >/dev/null 2>&1 || { echo "[FAIL] docker 未安装或不在 PATH"; exit 1; }

if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    echo "[FAIL] docker 未配置 nvidia runtime。见 hsmr-infer-image/deploy.sh 的报错提示"
    exit 1
fi
echo "[OK] nvidia runtime 就绪"

# 镜像存在则直接用, 否则本地构建
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[OK] 镜像已存在, 直接使用"
else
    echo "[WARN] 镜像不存在, 本地构建 (docker build --network=host, 约 10-20 分钟)..."
    docker build --network=host -t "$IMAGE" "$SCRIPT_DIR" || { echo "[FAIL] 构建失败"; exit 1; }
    echo "[OK] 镜像构建完成"
fi

# 释放 8011: 先移除旧容器, 再清宿主旧进程 (按端口精准处理, 绝不 pkill)
old_cids=$(docker ps -aq --filter "name=${SERVICE}" 2>/dev/null)
if [ -n "$old_cids" ]; then
    echo "[WARN] 移除旧容器: $(docker ps -a --filter "name=${SERVICE}" --format '{{.Names}}' | tr '\n' ' ')"
    docker rm -f $old_cids >/dev/null 2>&1 || { echo "[FAIL] 移除旧容器失败"; exit 1; }
fi
for _ in 1 2 3 4 5; do
    if ! ss -tlnH 2>/dev/null | grep -q ':8011'; then
        echo "[OK] 8011 空闲"
        break
    fi
    host_pid=$(ss -tlnpH 2>/dev/null | grep ':8011' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u | head -1)
    if [ -z "$host_pid" ]; then
        echo "[FAIL] 8011 被占用但拿不到 pid。请手动: sudo ss -tlnp | grep 8011"
        exit 1
    fi
    echo "[WARN] 8011 被宿主进程 pid=$host_pid 占用, 停止..."
    kill "$host_pid" 2>/dev/null && sleep 1 || { echo "[FAIL] 杀 pid=$host_pid 失败"; exit 1; }
done

cd "$SCRIPT_DIR" || exit 1
docker compose up -d || { echo "[FAIL] docker compose up -d 失败, 日志: docker logs $SERVICE"; exit 1; }
echo "[OK] 容器已启动"

echo "等待模型加载 (ViTDet + HSMR ONNX + SKEL)..."
for i in $(seq 1 "$WAIT_TIMEOUT"); do
    sleep 5
    resp="$(curl -s --max-time 5 "$HEALTH_URL" 2>/dev/null)"
    if echo "$resp" | grep -q '"model_loaded":true'; then
        echo "[OK] 服务就绪: $resp"
        echo "=== 部署完成 ==="
        echo "容器:   $(docker ps --filter "name=^/${SERVICE}$" --format '{{.Status}}')"
        echo "地址:   ${HEALTH_URL}"
        exit 0
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$SERVICE" 2>/dev/null)" != "true" ]; then
        echo "[FAIL] 容器未在运行! 最近日志:"
        docker logs --tail 50 "$SERVICE" 2>&1 | tail -50
        exit 1
    fi
    [ $((i % 4)) -eq 0 ] && echo "  ...仍在等模型加载 (${i}x5s)"
done

echo "[FAIL] 等待超时 (${WAIT_TIMEOUT}x5s)。排查: docker logs $SERVICE"
exit 1
