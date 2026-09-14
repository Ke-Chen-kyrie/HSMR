"""HSMR 外围能力 SDK.

纯外围、吃数据 —— 不碰模型推理, 入参是帧/深度/persons/音频, 输出是坐标/验证/发布结果.
模块:
  capture   Foxglove 多话题采集 (彩色/深度/内参/TF) + 帧解码
  tf        光学系→BASE/HEAD/world 坐标变换 (TF 优先 + HEAD 兜底)
  face      声纹 + 人脸交叉验证 (复用 user-identification 容器 127.0.0.1:8001)
  ros       zenoh/Foxglove 关节 MarkerArray + 标注图发布
  pipeline  HsmrSdk 门面: 采集→TF→验证→发布 一条龙 (persons 由外部提供)
"""
__version__ = "0.1.0"

from hsmr_sdk.config import load_config
from hsmr_sdk.capture import (MultiTopicCapture, decode_color, decode_depth,
                              camera_info_k, DEPTH_PNG_OFFSET)
from hsmr_sdk.tf import (TransformChain, quat_to_rotmat, inv_homog, transform_points,
                         joint_pos_ori, joint_orientation)
from hsmr_sdk.face import FaceVerifier, get_face_verifier, IdentifyError
from hsmr_sdk.ros import RosPublisher, get_ros_publisher, SKEL_JOINTS, SKEL_BONES
from hsmr_sdk.infer import InferClient, get_infer_client, InferError, encode_frame
from hsmr_sdk.pipeline import HsmrSdk

__all__ = [
    "__version__",
    "load_config",
    "MultiTopicCapture", "decode_color", "decode_depth", "camera_info_k", "DEPTH_PNG_OFFSET",
    "TransformChain", "quat_to_rotmat", "inv_homog", "transform_points",
    "joint_pos_ori", "joint_orientation",
    "FaceVerifier", "get_face_verifier", "IdentifyError",
    "RosPublisher", "get_ros_publisher", "SKEL_JOINTS", "SKEL_BONES",
    "InferClient", "get_infer_client", "InferError", "encode_frame",
    "HsmrSdk",
]
