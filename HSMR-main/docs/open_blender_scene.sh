#!/bin/bash
# 一键打开 HSMR Blender 场景
# Usage: bash docs/open_blender_scene.sh

BLENDER=/home/Kai/blender-4.2/blender
SCENE=/home/Kai/pose_model/HSMR-main/docs/hsmr_scene.blend

echo "🚀 正在打开 Blender 场景..."
echo ""
echo "══════════════════════════════════════"
echo "  操作指南 (笔记本，无需数字键盘)"
echo "══════════════════════════════════════"
echo ""
echo "  🖱️  旋转视角     鼠标中键拖拽 (或用触控板双指)"
echo "  🔍 缩放          滚轮 (或触控板捏合)"
echo "  ✋ 平移          Shift + 鼠标中键"
echo ""
echo "  📷 切换预设视角:"
echo "     右侧 Outliner 面板 → 07_Preset_Views"
echo "     双击 VIEW_Front   → 正面 (人体面对你)"
echo "     双击 VIEW_Right   → 侧面"
echo "     双击 VIEW_Top     → 俯视"
echo "     双击 VIEW_3Quarter → 3/4 角"
echo ""
echo "  👁️  显示/隐藏部件:"
echo "     Outliner 中点击各 Collection 的眼睛图标"
echo "     例如: 关闭 01_Skin_Mesh 只看骨骼"
echo ""
echo "  🎨 颜色说明:"
echo "     红色 = +X (身体左侧)"
echo "     绿色 = +Y (身体上方)"
echo "     蓝色 = +Z (身体前方)"
echo "     金色 = 人体前方向量"
echo "     白色球 = SKEL 坐标原点"
echo "     紫色块 = HSMR 虚拟相机"
echo "══════════════════════════════════════"
echo ""

# Enable emulate numpad (number row works as numpad) + open scene
$BLENDER $SCENE --python-expr "
import bpy
# Enable numpad emulation (use number row 1-0)
bpy.context.preferences.inputs.use_emulate_numpad = True
# Save preference so it persists
bpy.ops.wm.save_userpref()
print('Numpad emulation enabled: use 1-0 keys on number row')
" 2>/dev/null &

echo "Blender 启动中..."
sleep 2
echo "✅ 场景已打开！按照上方指南操作即可。"
