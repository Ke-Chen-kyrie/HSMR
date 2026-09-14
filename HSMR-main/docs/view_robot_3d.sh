#!/bin/bash
# 一键查看机器人摄像头预测的 3D 场景
#
# 流程:
#   1. SSH 连接到机器人 (192.168.217.100)
#   2. 拉取 latest_3d.json (3D预测) + latest.jpg (原始画面)
#   3. 渲染 3D 场景 (相机 + 每个人 + 朝向 + 距离)
#   4. 拼合: 原始画面 | 2D叠加 | 3D场景
#
# 用法: bash docs/view_robot_3d.sh

set -e
HOST="192.168.217.100"
USER="naviai"
PASS="naviai@2024"
REMOTE_BASE="/home/naviai/projects/HSMR-main/data_outputs/webcam_head_onnx"
OUT_DIR="docs"
PYTHON=".venv/bin/python"

echo "=============================================="
echo " 🤖 机器人 3D 场景查看器"
echo "=============================================="

# 1. 用 python+paramiko 拉取数据
echo "[1/4] SSH 拉取机器人预测数据..."
$PYTHON - "$HOST" "$USER" "$PASS" "$REMOTE_BASE" "$OUT_DIR" << 'PYEOF'
import sys, paramiko, os
host, user, password, remote_base, out_dir = sys.argv[1:6]
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password, timeout=10)
sftp = client.open_sftp()

files = ['latest_3d.json', 'latest.jpg', 'latest.json']
for f in files:
    try:
        sftp.get(f'{remote_base}/{f}', f'{out_dir}/_robot_{f}')
        print(f'  ✅ {f} ({os.path.getsize(f"{out_dir}/_robot_{f}")//1024} KB)')
    except Exception as e:
        print(f'  ⚠️  {f}: {e}')
sftp.close(); client.close()
PYEOF

# 2. 渲染 3D 场景
echo "[2/4] 渲染 3D 场景..."
$PYTHON docs/render_robot_3d.py --local "$OUT_DIR/_robot_latest_3d.json" --out "$OUT_DIR/_robot_3d_view.png"

# 3. 拼合原始画面 + 3D场景
echo "[3/4] 拼合视图..."
$PYTHON - "$OUT_DIR" << 'PYEOF'
import sys, os
sys.path.insert(0, '.')
import cv2, numpy as np
out_dir = sys.argv[1]

# 原始机器人画面 (latest.jpg 是 pyrender 叠加后的)
orig = cv2.imread(f'{out_dir}/_robot_latest.jpg')
if orig is None:
    print('  ⚠️ 无原始画面, 跳过拼合')
    sys.exit(0)
H_TGT = 800
orig = cv2.resize(orig, (int(H_TGT*orig.shape[1]/orig.shape[0]), H_TGT))

# 3D 场景图 (缩放到等高)
three = cv2.imread(f'{out_dir}/_robot_3d_view.png')
scale = H_TGT / three.shape[0]
three = cv2.resize(three, (int(three.shape[1]*scale), H_TGT))

# 上下补白到等高
max_h = max(orig.shape[0], three.shape[0])
def pad(img, target_h):
    if img.shape[0] >= target_h: return img
    pad_h = target_h - img.shape[0]
    return np.vstack([img, np.ones((pad_h, img.shape[1], 3), dtype=np.uint8)*30])
orig = pad(orig, max_h); three = pad(three, max_h)

# 左右拼
gap = np.ones((max_h, 6, 3), dtype=np.uint8)*80
combined = np.hstack([orig, gap, three])
cv2.imwrite(f'{out_dir}/_robot_combined.png', combined)
print(f'  ✅ 拼合完成: {out_dir}/_robot_combined.png ({combined.shape[1]}x{combined.shape[0]})')
PYEOF

echo "[4/4] 完成!"
echo ""
echo "输出文件:"
echo "  $OUT_DIR/_robot_combined.png    ← 原始画面 + 3D场景"
echo "  $OUT_DIR/_robot_3d_view.png     ← 纯3D场景"
echo "  $OUT_DIR/_robot_latest_3d.json  ← 机器人3D数据"
echo "  $OUT_DIR/_robot_latest.jpg      ← 机器人渲染帧"
echo ""
echo "打开看图: eog $OUT_DIR/_robot_combined.png &"
