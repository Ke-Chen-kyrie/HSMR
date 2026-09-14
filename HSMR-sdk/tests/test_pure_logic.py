#!/usr/bin/env python3
"""HSMR-sdk 纯逻辑单元测试 (无网络 / 无硬件 / 无 zenoh).

覆盖:
  tf:    quat_to_rotmat / inv_homog / TransformChain (TF 图连乘 + HEAD 兜底) / joint_pos_ori / joint_orientation
  face:  cross_verify (两路交集) / associate_face_to_person (头关节 2D 最近邻 + 阈值)
  pipeline.identify: 注入 mock FaceVerifier + 合成 persons, 走通 BASE 系骨盆/头 3D 与响应形状
        (IdentifyError 分支)

运行:
  python -m tests.test_pure_logic        (或 pytest tests/)
"""
import sys
import numpy as np
from pathlib import Path

SDK_ROOT = str(Path(__file__).resolve().parent.parent)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from hsmr_sdk.tf import (TransformChain, joint_pos_ori, joint_orientation,
                         quat_to_rotmat, inv_homog, _JOINT_REMAP)
from hsmr_sdk.face import FaceVerifier, IdentifyError
from hsmr_sdk import load_config

# ── 合成 TF 消息 (最小桩) ──
class _Q:  # geometry_msgs/Quaternion
    def __init__(s, x, y, z, w): s.x, s.y, s.z, s.w = x, y, z, w
class _V:  # geometry_msgs/Vector3
    def __init__(s, x, y, z): s.x, s.y, s.z = x, y, z
class _Hdr:
    def __init__(s, frame_id): s.frame_id = frame_id
class _Tr:
    def __init__(s, frame_id, child_frame_id, R, t):
        s.header = _Hdr(frame_id)
        s.child_frame_id = child_frame_id
        s.transform = type("T", (), {"rotation": _Q(*_rot_to_q(R)), "translation": _V(*t)})
class _TfMsg:
    def __init__(s, transforms): s.transforms = transforms

def _rot_to_q(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        s2 = np.sqrt(tr+1)*2
        return (R[2,1]-R[1,2])/s2, (R[0,2]-R[2,0])/s2, (R[1,0]-R[0,1])/s2, 0.25*s2
    raise NotImplementedError("测试只用 tr>0 的旋转")

def _cfg():
    return load_config(str(Path(SDK_ROOT)/"config.yaml"))

def _person(joints_optical, ori, depth_valid, joints_2d=None):
    return {"joints_optical_m": joints_optical, "joints_ori": ori,
            "depth_valid": depth_valid, "joints_2d": joints_2d or [[0,0]]*24}

def _person_default():
    I = np.eye(3)
    pos = [[0.5, 0.0, 1.2]]*24
    return _person([list(p) for p in pos],
                   [I.tolist()]*24,
                   [True]*24)

# ══════════════════════════════════════════════════════════════════
# tf
# ══════════════════════════════════════════════════════════════════
def test_quat_to_rotmat_identity():
    R = quat_to_rotmat(0, 0, 0, 1)
    assert np.allclose(R, np.eye(3)), R

def test_quat_to_rotmat_z90():
    # 绕 Z 旋转 90°: 四元数 w=cos45, z=sin45
    R = quat_to_rotmat(0, 0, np.sqrt(0.5), np.sqrt(0.5))
    v = R @ np.array([1.0, 0, 0])
    assert np.allclose(v, [0, 1, 0], atol=1e-6), v

def test_inv_homog():
    R, t = quat_to_rotmat(0, 0, 0.5, np.sqrt(0.75)), [1.0, 2.0, 3.0]
    M = np.eye(4); M[:3,:3], M[:3,3] = R, t
    assert np.allclose(M @ inv_homog(M), np.eye(4), atol=1e-9)
    assert np.allclose(inv_homog(M) @ M, np.eye(4), atol=1e-9)

def test_chain_bfs_multi_hop():
    cfg = _cfg()
    chain = TransformChain(cfg)
    # 物理: HEAD 在 optical 上方 0.1, base_link 在 HEAD 下方 0.9 ⇒ base_link 在 optical 下方 0.8.
    # get_opt_to_target 返回 optical→target 映射 (p_target = R@p_optical + t),
    # 故 optical 原点(在 base 系上方 0.8) ⇒ t = [0,0,0.8].
    chain.update_tf([_TfMsg([_Tr(cfg["frames"]["optical"], cfg["frames"]["head"],
                                 np.eye(3), [0,0,0.1])]),
                     _TfMsg([_Tr(cfg["frames"]["head"], "base_link", np.eye(3), [0,0,-0.9])])])
    r = chain.get_opt_to_target("BASE")
    assert r is not None, "TF 图中 base_link 存在, BASE 应可解"
    R, t = r
    assert np.allclose(R, np.eye(3)), R
    assert np.allclose(t, [0, 0, 0.8], atol=1e-9), t
    # 点变换: optical [0,0,1] → base 系 +0.8 = [0,0,1.8]
    p = chain.to_target(np.array([[0,0,1.0]]), "BASE")
    assert np.allclose(p, [[0, 0, 1.8]], atol=1e-9), p

def test_chain_rot_to_target():
    cfg = _cfg()
    chain = TransformChain(cfg)
    # TF 声明 optical→base_link 的朝向 = Rz90 (base_link 在 optical 系里的朝向);
    # get_opt_to_target 返回其逆作为 optical→target 映射, 即 Rz90.T.
    Rz90 = quat_to_rotmat(0, 0, np.sqrt(0.5), np.sqrt(0.5))
    chain.update_tf([_TfMsg([_Tr(cfg["frames"]["optical"], "base_link", Rz90, [0,0,0])])])
    Rr = chain.rot_to_target(np.eye(3), "BASE")
    assert Rr is not None and np.allclose(Rr, Rz90.T, atol=1e-9), Rr
    # 关节朝向也从 optical → target: R_target = R_map @ R_joint_optical
    assert np.allclose(chain.rot_to_target(Rz90, "BASE"), Rz90.T @ Rz90, atol=1e-9)

def test_chain_head_fallback_and_missing():
    cfg = _cfg()
    chain = TransformChain(cfg)          # 无任何 TF
    assert chain.get_opt_to_target("BASE") is None, "无 TF 时 BASE 不可解 (不瞎算)"
    r = chain.get_opt_to_target(cfg["frames"]["head"])
    assert r is not None, "无 TF 时 HEAD 走 verified 兜底矩阵"
    R, t = r
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "兜底矩阵旋转部分须正交"
    assert chain.transform_source(cfg["frames"]["head"]) == "head_matrix"
    assert chain.transform_source("BASE") is None

def test_remap_invariant_unified_identity():
    # joints_ori = _JOINT_REMAP ⇒ 统一 remap 后为 identity (X=前)
    cfg = _cfg()
    chain = TransformChain(cfg)          # optical → optical = identity
    person = _person_default()
    person["joints_ori"] = [_JOINT_REMAP.tolist()]*24
    pos, R, dv = joint_pos_ori(person, 0, chain, cfg["frames"]["optical"])
    assert dv and np.allclose(R, np.eye(3), atol=1e-9), R
    o = joint_orientation(R)
    assert np.allclose(o["body_forward"], [1,0,0], atol=1e-9), o["body_forward"]
    assert np.isclose(o["euler_deg"]["yaw_deg"], 0.0)

def test_joint_pos_ori_nan_depth_invalid():
    cfg = _cfg()
    chain = TransformChain(cfg)
    person = _person_default()
    person["joints_optical_m"] = [[float("nan"), 0, 1.0]]*24
    person["depth_valid"] = [True]*24
    pos, R, dv = joint_pos_ori(person, 0, chain, cfg["frames"]["optical"])
    assert dv is False, "NaN 坐标应把 depth_valid 置 False"
    assert pos is None and R is None

# ══════════════════════════════════════════════════════════════════
# face (纯逻辑, 无网络)
# ══════════════════════════════════════════════════════════════════
def test_cross_verify_intersection():
    fv = FaceVerifier()
    voice = [{"user_id": "peilin", "name": "A", "score": 0.9},
             {"user_id": "kai", "name": "B", "score": 0.6}]
    face = [{"user_id": "kai", "name": "B", "score": 0.8, "matched": True,
             "location": {"x": 100, "y": 50, "width": 40, "height": 40}},
            {"user_id": "unknown_1", "matched": True, "location": {}}]
    out = FaceVerifier.cross_verify(voice, face)
    assert out and out["user_id"] == "kai", out     # 声纹第1(peilin)人脸不在, 取第2(kai)
    assert abs(out["voice_score"] - 0.6) < 1e-9
    # unknown_ 排除 + 无交集
    assert FaceVerifier.cross_verify(voice, [{"user_id": "unknown_1", "matched": True}]) is None
    assert FaceVerifier.cross_verify([], face) is None
    assert FaceVerifier.cross_verify(voice, []) is None

def test_associate_face_to_person():
    fv = FaceVerifier({"assoc_max_dist": 80})
    # person0 头 2D (400,300), person1 头 2D (700,500)
    p0 = _person_default(); p0["joints_2d"] = [[400,300]]*24
    p1 = _person_default(); p1["joints_2d"] = [[700,500]]*24
    # 人脸框中心 (410,310) → person0 距离 √(10²+10²)=14
    assert fv.associate_face_to_person([p0, p1], {"x":390,"y":290,"width":40,"height":40}) == 0
    # 离最近头关节 >80 → None
    assert fv.associate_face_to_person([p0, p1], {"x":0,"y":0,"width":10,"height":10}) is None
    # 无 joints_2d → None
    assert fv.associate_face_to_person([_person_default()], {"x":400,"y":300,"width":10,"height":10}) is None

# ══════════════════════════════════════════════════════════════════
# pipeline.identify (mock face, 真实 TransformChain)
# ══════════════════════════════════════════════════════════════════
class _MockFace:
    enabled = True
    def __init__(self, person_index=0, fail=False): self._pi, self._fail = person_index, fail
    def identify(self, audio, frame, persons, top_k=None, verbose=False):
        if self._fail:
            raise IdentifyError("声纹未匹配到注册用户")
        return {"user_id": "kai", "name": "kai", "voice_score": 0.8, "face_score": 0.9,
                "face_location": {"x": 390, "y": 290, "width": 40, "height": 40},
                "person_index": self._pi}

def _make_sdk(face):
    from hsmr_sdk import HsmrSdk
    cfg = _cfg()
    sdk = HsmrSdk(cfg)
    sdk.face = face
    return sdk

def test_pipeline_identify_shape_and_tf():
    cfg = _cfg()
    sdk = _make_sdk(_MockFace(0))
    # optical → base_link: t=[0,0,-0.8]
    sdk.tf.update_tf([_TfMsg([_Tr(cfg["frames"]["optical"], "base_link", np.eye(3), [0,0,-0.8])])])
    person = _person_default()    # 全关节 optical [0.5, 0.0, 1.2]
    resp = sdk.identify(b"fake-audio", np.zeros((64,64,3), np.uint8), [person],
                        target="BASE", include_pelvis_pose=True, verbose=True)
    assert resp["verified"] is True and resp["user_id"] == "kai"
    assert resp["person_index"] == 0 and resp["frame"] == "BASE"
    # base_link 在 optical 下方 0.8 ⇒ 光学点 [0.5,0,1.2] → BASE [0.5,0,1.2+0.8]
    assert np.allclose(resp["position_m"], [0.5, 0.0, 2.0], atol=1e-9), resp["position_m"]
    assert np.allclose(resp["head_position_m"], [0.5, 0.0, 2.0], atol=1e-9)
    assert resp["depth_valid"] is True and resp["head_depth_valid"] is True
    assert resp["rotation_matrix"] is not None
    assert "voice_candidates" in resp and "pelvis_pose" in resp
    assert "timestamp" in resp

def test_pipeline_identify_frame_alias_and_error():
    from hsmr_sdk.pipeline import HsmrSdk
    cfg = _cfg()
    sdk = _make_sdk(_MockFace(0))
    sdk.tf.update_tf([_TfMsg([_Tr(cfg["frames"]["optical"], "base_link", np.eye(3), [0,0,0])])])
    person = _person_default()
    # 别名 "相机"/"光学系" → optical_frame
    resp = sdk.identify(b"a", np.zeros((8,8,3),np.uint8), [person], target="相机")
    assert resp["frame"] == cfg["frames"]["optical"]
    # face 抛 IdentifyError → 原样传播
    sdk2 = _make_sdk(_MockFace(0, fail=True))
    try:
        sdk2.identify(b"a", np.zeros((8,8,3),np.uint8), [person], target="BASE")
        raise AssertionError("应抛 IdentifyError")
    except IdentifyError as e:
        assert e.msg == "声纹未匹配到注册用户"

def test_pipeline_person_joints_payload():
    from hsmr_sdk import HsmrSdk
    cfg = _cfg()
    sdk = _make_sdk(_MockFace(0))
    sdk.tf.update_tf([_TfMsg([_Tr(cfg["frames"]["optical"], "base_link", np.eye(3), [0,0,-0.8])])])
    joints = sdk.person_joints_payload(_person_default(), target="BASE", selected_joints={0,13})
    assert len(joints) == 24
    assert all(j["idx"] == i for i, j in enumerate(joints))
    assert np.allclose(joints[0]["position_m"], [0.5, 0.0, 2.0], atol=1e-9)
    assert joints[0]["selected"] is True and joints[1]["selected"] is False
    assert joints[0]["R"] is not None and joints[0]["depth_valid"] is True


# ══════════════════════════════════════════════════════════════════
# infer (纯逻辑: encode_frame, 不碰网络)
# ══════════════════════════════════════════════════════════════════
def test_infer_encode_frame_with_depth():
    from hsmr_sdk.infer import encode_frame
    rgb = np.zeros((16, 16, 3), np.uint8)
    depth = np.full((16, 16), 1000, np.uint16)
    K = np.array([[910.68, 0, 653.79], [0, 910.28, 374.08], [0, 0, 1]], float)
    files, data = encode_frame(rgb, depth, K, max_instances=3)
    assert "image" in files and "depth" in files
    assert files["depth"][2] == "image/png"
    assert data["fx"] == 910.68 and data["fy"] == 910.28
    assert data["cx"] == 653.79 and data["cy"] == 374.08
    assert data["max_instances"] == 3

def test_infer_encode_frame_no_depth():
    from hsmr_sdk.infer import encode_frame
    rgb = np.zeros((8, 8, 3), np.uint8)
    files, data = encode_frame(rgb)          # 无深度 → 无 K 也不报错
    assert "image" in files and "depth" not in files
    assert data == {}

def test_infer_encode_frame_depth_without_k_raises():
    from hsmr_sdk.infer import encode_frame, InferError
    rgb = np.zeros((8, 8, 3), np.uint8)
    depth = np.zeros((8, 8), np.uint16)
    try:
        encode_frame(rgb, depth)             # 有深度无 K → 必须抛
        raise AssertionError("应抛 InferError")
    except InferError as e:
        assert "K" in e.msg

def test_infer_client_from_config():
    from hsmr_sdk import get_infer_client
    c = get_infer_client(_cfg())
    assert c.url == "http://127.0.0.1:8010"
    assert c.enabled is True and c.max_instances == 5

def test_pipeline_infer_persons_passthrough():
    from hsmr_sdk import HsmrSdk
    class _MockInfer:
        def __init__(self): self.called = None
        def infer_persons(self, frame=None, rgb_bgr=None, depth_uint16=None, K=None, **kw):
            self.called = (frame, rgb_bgr, depth_uint16, K)
            return ["p1", "p2"]
    cfg = _cfg()
    sdk = HsmrSdk(cfg)
    mock = _MockInfer()
    sdk.infer = mock
    frame = {"rgb_bgr": np.zeros((8,8,3),np.uint8), "depth_uint16": np.zeros((8,8),np.uint16),
             "K": np.eye(3)}
    persons = sdk.infer_persons(frame)
    assert persons == ["p1", "p2"] and mock.called[0] is frame
    # 显式参数走同样路径
    sdk.infer_persons(rgb_bgr=frame["rgb_bgr"])
    assert mock.called[1] is frame["rgb_bgr"] and mock.called[0] is None


def _run_all():
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} 通过")
    return failed

if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
