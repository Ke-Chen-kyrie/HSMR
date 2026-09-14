"""HSMR 推理包: 模型输入输出 + 推理能力, 不含任何机器人外围 (capture/TF/ROS/服务依赖).

导入本包即把工程根加入 sys.path (lib/、deploy/、thirdparty/SKEL 均以根为基准),
因此无论从哪个目录 `import hsmr_infer` 都能正确解析模型库依赖.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)          # 工程根 (HSMR-infer)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# SKEL 包 (thirdparty/SKEL/skel)
sys.path.insert(0, os.path.join(ROOT, "thirdparty", "SKEL"))
# 若存在 SMPL 依赖
_smpl = os.path.join(ROOT, "thirdparty", "SMPL")
if os.path.isdir(_smpl):
    sys.path.insert(0, _smpl)
