#!/usr/bin/env bash
# hsmr-infer(8010) + hsmr-render(8011) 一键重新部署 —— 在"本地开发机"上运行
#
# 原理 (与旧 deploy.sh 一致):
#   远程 Jetson 上已有构建目录 (hsmr-infer-image / hsmr-render-image),
#   内含 Dockerfile + venv_site(aarch64 site-packages 1.4G) + hsrm_root(模型/库)。
#   本脚本只做两件事:
#     ① rsync 把"新代码"推进远程 app/  (infer 只有 hsmr_infer/, render 是全部 py)
#     ② 远程 docker build --network=host 重打镜像(新 TAG) → 改 compose tag → up -d → 健康检查
#   镜像基础层 (user-identification:latest + venv_site) 不变, build 只有 app 层变化, 秒级。
#
# 用法:
#   bash deploy_hsmr.sh                # 全流程 (默认 TAG=今天日期 20260831)
#   HSMR_SSH_PASS=xxx bash deploy_hsmr.sh   # 指定远程密码 (默认 naviai@2024)
#   bash deploy_hsmr.sh --sync-only    # 只推代码, 不构建 (先 review)
#   bash deploy_hsmr.sh --build-only   # 只构建+起容器 (假设代码已同步)
#
# 需要: 本地装了 sshpass + rsync; 远程 user 无 sudo 也能 docker(已在 docker 组)
set -uo pipefail

# ── 配置 ──
REMOTE_HOST="${HSMR_REMOTE_HOST:-192.168.217.100}"
REMOTE_USER="${HSMR_REMOTE_USER:-naviai}"
REMOTE_PASS="${HSMR_SSH_PASS:-naviai@2024}"
REMOTE_BASE="/home/naviai/projects/HSMR"
REG="guopeilin-registry.cn-hangzhou.cr.aliyuncs.com/erban_agent"
TAG="$(date +%Y%m%d)"

LOCAL_HSMR="/home/Kai/pose_model/HSMR"     # 本地三个工程根
INFER_SRC="$LOCAL_HSMR/HSMR-infer"
RENDER_SRC="$LOCAL_HSMR/HSMR-render"
INFER_BUILD="$REMOTE_BASE/hsmr-infer-image"
RENDER_BUILD="$REMOTE_BASE/hsmr-render-image"

MODE="${1:---all}"
SSH="sshpass -p $REMOTE_PASS ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 $REMOTE_USER@$REMOTE_HOST"
RSH="sshpass -p $REMOTE_PASS ssh -o StrictHostKeyChecking=no"

C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'; C_NC=$'\033[0m'
ok()  { echo -e "${C_GRN}[OK]${C_NC}   $*"; }
warn(){ echo -e "${C_YEL}[WARN]${C_NC} $*"; }
err() { echo -e "${C_RED}[FAIL]${C_NC} $*"; }

echo "=== HSMR 一键部署  (${REMOTE_USER}@${REMOTE_HOST})  镜像 TAG=${TAG} ==="
command -v sshpass >/dev/null || { err "本机缺 sshpass (sudo apt install sshpass)"; exit 1; }
command -v rsync  >/dev/null || { err "本机缺 rsync"; exit 1; }

# ── 0. 连通性 + 前置检查 ──
$SSH 'true' 2>/dev/null || { err "连不上远程, 请检查网络/密码"; exit 1; }
ok "SSH 连通"
$SSH "docker info --format '{{json .Runtimes}}' | grep -q '\"nvidia\"'" \
  || { err "远程 docker 未配 nvidia runtime"; exit 1; }
ok "远程 nvidia runtime 就绪"
$SSH "docker image inspect user-identification:latest >/dev/null 2>&1" \
  || { err "远程缺少基座镜像 user-identification:latest"; exit 1; }
ok "基座镜像 user-identification:latest 存在"

# ── 1. 推代码 ──
if [ "$MODE" != "--build-only" ]; then
  echo
  echo "── [1/3] rsync 推送新代码 ──"
  # infer: 整个工程 (模型/lib/configs 已一致, rsync 只传变化的; 排除构建相关与缓存)
  rsync -az --delete -e "$RSH" \
    --exclude='Dockerfile' --exclude='deploy.sh' --exclude='data_outputs' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    "$INFER_SRC/" "$REMOTE_USER@$REMOTE_HOST:$INFER_BUILD/app/" \
    || { err "infer 代码同步失败"; exit 1; }
  ok "infer 代码已同步 (hsmr_infer/server.py 等)"

  # render: 全部代码; 但 config.yaml 要换容器路径(见下), 先不推, 由脚本生成
  rsync -az --delete -e "$RSH" \
    --exclude='Dockerfile' --exclude='deploy.sh' --exclude='config.yaml' \
    --exclude='data_outputs' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='*.log' --exclude='.git' \
    "$RENDER_SRC/" "$REMOTE_USER@$REMOTE_HOST:$RENDER_BUILD/app/" \
    || { err "render 代码同步失败"; exit 1; }
  ok "render 代码已同步 (main.py/inference.py/tf_chain.py 等)"

  # render config.yaml: 本地新配置(带 frames/兜底矩阵)但 hsrm_root 是本地路径,
  # 容器里必须 /app/hsrm_root → 由本地生成远程配置
  sed -E 's#^(\s*hsrm_root:).*#\1 "/app/hsrm_root"#' \
    "$RENDER_SRC/config.yaml" \
    | $SSH "cat > $RENDER_BUILD/app/config.yaml"
  ok "render config.yaml 已按容器路径 (/app/hsrm_root) 生成"
fi

if [ "$MODE" = "--sync-only" ]; then
  echo
  warn "已按 --sync-only 结束, 未构建。review 后执行: bash deploy_hsmr.sh --build-only"
  exit 0
fi

# ── 2. 远程构建 (Dockerfile 用 --network=host) ──
echo
echo "── [2/3] 远程构建镜像 (TAG=$TAG) ──"
$SSH "cd $INFER_BUILD && docker build --network=host -t $REG/hsmr-infer:$TAG ." \
  || { err "infer 镜像构建失败"; exit 1; }
ok "hsmr-infer:$TAG 构建完成"
$SSH "cd $RENDER_BUILD && docker build --network=host -t $REG/hsmr-render:$TAG ." \
  || { err "render 镜像构建失败"; exit 1; }
ok "hsmr-render:$TAG 构建完成"

# ── 3. 改 compose tag + up -d + 健康检查 ──
echo
echo "── [3/3] 更新 compose 并启动 ──"
$SSH "cd $INFER_BUILD  && sed -i 's#image: .*hsmr-infer:.*#image: $REG/hsmr-infer:$TAG#' docker-compose.yml && docker compose up -d" \
  || { err "infer compose 启动失败"; exit 1; }
$SSH "cd $RENDER_BUILD && sed -i 's#image: .*hsmr-render:.*#image: $REG/hsmr-render:$TAG#' docker-compose.yml && docker compose up -d" \
  || { err "render compose 启动失败"; exit 1; }
ok "容器已启动 (restart: unless-stopped)"

# 健康检查在远程执行 (host 网络, 端口绑在远程 127.0.0.1)
echo
echo "── 等待模型加载 (infer=8010 / render=8011, 约 1-2 分钟) ──"
wait_health() {
  local url=$1 name=$2
  for i in $(seq 1 60); do
    resp="$($SSH "curl -s --max-time 5 $url/health 2>/dev/null")"
    if echo "$resp" | grep -q '"model_loaded":true'; then
      ok "$name 就绪: $resp"; return 0
    fi
    [ $((i % 6)) -eq 0 ] && echo "  ...仍在加载 $name (${i}x5s)"
    sleep 5
  done
  err "$name 等待超时。远程排查: docker logs hsmr-$name"
  return 1
}
wait_health "http://127.0.0.1:8010" "infer"
wait_health "http://127.0.0.1:8011" "render"

echo
echo "=== 部署完成 ==="
$SSH 'docker ps --filter "name=^/hsmr-" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"'
echo "验证:"
echo "  curl http://${REMOTE_HOST}:8010/health"
echo "  curl http://${REMOTE_HOST}:8011/health"