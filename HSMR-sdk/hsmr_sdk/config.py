"""配置加载. SDK 各模块构造时收 dict 段 (bridge/topics/frames/ros_publish/face_verify).

默认读工程根 config.yaml, 可用环境变量 HSMR_SDK_CONFIG 覆盖.
"""
import os

import yaml

_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config.yaml")


def load_config(path=None):
    """返回 yaml 配置 dict. path=None → 环境变量 HSMR_SDK_CONFIG 或工程根 config.yaml."""
    path = path or os.environ.get("HSMR_SDK_CONFIG") or _DEFAULT
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
