#!/usr/bin/env bash
# bootstrap_remote.sh —— 在"全新/空白 Jetson"上执行: 从打包产物恢复 HSMR 构建环境
#
# 前置: 本机已收到 pack_hsmr_bundle.sh 的两个产物 (默认放 ~/):
#   hsmr_base_image.tar       基座镜像
#   hsmr-build-bundle.tar.gz  两个构建目录
# 本脚本只还原"构建材料", 不构建镜像; 之后用 deploy_hsmr.sh (本地机) 推送新代码并构建启动。
#
# 用法:  bash bootstrap_remote.sh
set -uo pipefail

SRC="${1:-$HOME}"
BASE="/home/naviai/projects/HSMR"
BASE_TAR="$SRC/hsmr_base_image.tar"
BUNDLE="$SRC/hsmr-build-bundle.tar.gz"

C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_NC=$'\033[0m'
ok()  { echo -e "${C_GRN}[OK]${C_NC}   $*"; }
err() { echo -e "${C_RED}[FAIL]${C_NC} $*"; }

echo "=== 空白 Jetson 还原 HSMR 构建环境 ==="
command -v docker >/dev/null || { err "需要 docker"; exit 1; }

# 1. 基座镜像
[ -f "$BASE_TAR" ] || { err "缺 $BASE_TAR"; exit 1; }
if docker image inspect user-identification:latest >/dev/null 2>&1; then
    ok "基座镜像已存在, 跳过 load"
else
    echo "-- docker load 基座镜像 (约 2-5 分钟) --"
    docker load -i "$BASE_TAR" || { err "docker load 失败"; exit 1; }
    # 若 save 的 tag 不是 latest, 补 tag
    id=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -i 'user-identification' | head -1)
    if [ -n "$id" ] && [ "$id" != "user-identification:latest" ]; then
        docker tag "$id" user-identification:latest
    fi
    ok "基座镜像就绪: user-identification:latest"
fi

# 2. 构建目录
[ -f "$BUNDLE" ] || { err "缺 $BUNDLE"; exit 1; }
mkdir -p "$BASE"
if [ -d "$BASE/hsmr-infer-image" ]; then
    ok "构建目录已存在, 跳过解包 (如需覆盖: rm -rf $BASE/hsmr-*-image 后重跑)"
else
    echo "-- 解包构建目录 --"
    tar xzf "$BUNDLE" -C "$BASE" || { err "解包失败"; exit 1; }
    ok "构建目录已还原: hsmr-infer-image / hsmr-render-image"
fi

# 3. 检查结构
for d in hsmr-infer-image/app hsmr-infer-image/venv_site \
         hsmr-render-image/app hsmr-render-image/venv_site hsmr-render-image/hsrm_root; do
    [ -d "$BASE/$d" ] || { err "结构不完整: 缺 $BASE/$d"; exit 1; }
done
ok "目录结构完整 (app + venv_site + hsrm_root)"

# 4. nvidia runtime
docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"' \
    || { err "docker 未配 nvidia runtime (见原 deploy.sh 提示)"; exit 1; }
ok "nvidia runtime 就绪"

echo
echo "=== 还原完成 ==="
echo "下一步 (在本地开发机):"
echo "  bash deploy_hsmr.sh        # rsync 新代码 + 远程构建 + compose up + 健康检查"
echo "  bash deploy_hsmr.sh --sync-only   # 先只推代码 review"