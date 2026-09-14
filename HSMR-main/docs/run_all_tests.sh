#!/bin/bash
# 一键测试: 深度自洽 + base_link TF链 + 完整融合
# 用法: bash docs/run_all_tests.sh
cd /home/naviai/projects/HSMR-main
PY=.venv_orin/bin/python

echo "=============================================="
echo " 🤖 HSMR 真实深度测试 (3 项)"
echo "=============================================="

echo ""
echo "===== 测试1: 深度反投影自洽 (不需要人) ====="
echo "看: 回投误差应≈0px, 深度中位数合理"
timeout 60 $PY docs/test_depth_validation.py 2>&1 | grep -vE 'WARNING|INFO' | grep -A2 '【A\|【B\|【C\|结论\|中心\|采样\|合理'

echo ""
echo "===== 测试2: base_link TF 链 (不需要人) ====="
echo "看: optical→BASE 路径完整, 示例点转换合理"
timeout 60 $PY docs/base_link_demo.py 2>&1 | grep -vE 'WARNING|INFO' | grep -A2 '═══ [1234]'

echo ""
echo "===== 测试3: 完整融合 (需要站到相机前!) ====="
echo "⚠️  现在请站到机器人相机前约 2 米, 面朝相机"
echo "    (脚本会在 5 秒后开始采样)"
sleep 5
timeout 120 $PY docs/realtime_depth_3d.py \
  --device cuda:0 --backend onnx --max_frames 3 --interval 5 \
  --max_instances 3 --out data_outputs/depth3d 2>&1 | grep -vE 'WARNING|INFO|Batch Detection|it/s|meshgrid' | grep -E 'sample|P[0-9]|骨盆|前向|TF|渲染'

echo ""
echo "=============================================="
echo " ✅ 测试完成. 结果看:"
echo "   data_outputs/depth3d/latest_comprehensive.png  综合4面板"
echo "   data_outputs/depth3d/latest_3d.json            BASE坐标+旋转矩阵"
echo "   docs/_depth_test_*.png                        深度验证图"
echo ""
echo " 对照方法:"
echo "   · 距离: 你站 ~2m, 看 P 骨盆深度是否≈2m"
echo "   · 朝向: 面朝相机 → 前向≈[0,0,-1], 背对 → [0,0,1]"
echo "=============================================="
