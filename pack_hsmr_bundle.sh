#!/usr/bin/env bash
# pack_hsmr_bundle.sh —— 在一台"已装好 HSMR 的 Jetson"上打包全套, 供空白机器人部署
#
# 产物 (放在当前目录):
#   hsmr_base_image.tar       基座镜像 user-identification:latest (~18.5GB, docker save)
#   hsmr-build-bundle.tar.gz  两个构建目录 hsmr-infer-image/ + hsmr-render-image/
#                             (含 venv_site aarch64 依赖 + hsrm_root 模型库 + app + Dockerfile + compose)
#
# 用法:  bash pack_hsmr_bundle.sh
# 需要:  在已部署的 Jetson 上执行 (有 docker + ~/projects/HSMR/{hsmr-infer-image,hsmr-render-image})
set -uo pipefail

BASE_DIR="${1:-$HOME/projects/HSMR}"
OUT="${2:-$HOME}"
TAR=()

command -v docker >/dev/null || { echo "[FAIL] 需要 docker"; exit 1; }

echo "=== 打包 HSMR 全套 (源: $BASE_DIR) ==="

echo "-- 1/2 基座镜像 (docker save) --"
if [ ! -f "$OUT/hsmr_base_image.tar" ]; then
    docker save user-identification:latest -o "$OUT/hsmr_base_image.tar" \
        || { echo "[FAIL] docker save 基座镜像失败"; exit 1; }
else
    echo "已存在, 跳过"
fi
echo "[OK] $OUT/hsmr_base_image.tar ($(du -sh "$OUT/hsmr_base_image.tar" | cut -f1))"

echo "-- 2/2 两个构建目录 --"
for d in hsmr-infer-image hsmr-render-image; do
    [ -d "$BASE_DIR/$d" ] || { echo "[FAIL] 缺 $BASE_DIR/$d"; exit 1; }
    TAR+=("$BASE_DIR/$d")
done
tar czf "$OUT/hsmr-build-bundle.tar.gz" -C "$BASE_DIR" "${TAR[@]##*/}" \
    || { echo "[FAIL] tar 构建目录失败"; exit 1; }
echo "[OK] $OUT/hsmr-build-bundle.tar.gz ($(du -sh "$OUT/hsmr-build-bundle.tar.gz" | cut -f1))"

echo
echo "=== 打包完成, 需传到空白机器人 ==="
echo "  scp $OUT/hsmr_base_image.tar      <blank-jetson>:~/"
echo "  scp $OUT/hsmr-build-bundle.tar.gz <blank-jetson>:~/"
echo "然后在其上执行: bash ~/bootstrap_remote.sh"
du -sh "$OUT/hsmr_base_image.tar" "$OUT/hsmr-build-bundle.tar.gz"