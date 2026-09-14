"""声纹 + 人脸交叉验证客户端.

移植自原工程 deploy/joint_service/face_verify.py, 复用宿主机 user-identification
容器 (默认 127.0.0.1:8001), 不重复部署 insightface/声纹引擎:
  POST /api/voice/search  → 声纹候选 [{user_id, name, score}] (分数降序, 已按 VOICE_MIN_SCORE 过滤)
  POST /api/face/detect   → 人脸 [{user_id, name, score, matched, location{x,y,width,height}}]

交叉验证: 声纹候选(分数降序)中第一个被当前帧人脸匹配到的 user_id = 说话人.
人脸 bbox 通过头关节 2D 最近邻关联到 person.
纯吃数据: 入参是 音频bytes / 帧BGR / persons; 不碰推理.
"""
import threading

import requests

# 关节名索引常量 (associate_face_to_person 用 head=13 关联) —— 与 tf.SKEL_JOINTS 一致
HEAD_JOINT_INDEX = 13


class IdentifyError(Exception):
    """identify() 流程分支失败 (声纹无匹配/交叉未通过/人脸关联失败/TF 不可达)."""

    def __init__(self, msg, detail=None):
        super().__init__(msg)
        self.msg = msg
        self.detail = detail


class FaceVerifier:
    """读 config 的 face_verify 段."""

    def __init__(self, cfg_fv=None):
        cfg_fv = cfg_fv or {}
        self.enabled = bool(cfg_fv.get("enabled", True))
        self.service_url = (cfg_fv.get("service_url") or "http://127.0.0.1:8001").rstrip("/")
        self.voice_top_k = int(cfg_fv.get("voice_top_k", 3))
        self.face_max_face_num = int(cfg_fv.get("face_max_face_num", 10))
        self.assoc_max_dist = float(cfg_fv.get("assoc_max_dist", 80))
        self.timeout = 10.0

    def voice_search(self, audio_bytes, top_k=None):
        """音频 → 声纹候选 [{user_id, name, score}] 分数降序. 异常抛给调用方."""
        top_k = int(top_k or self.voice_top_k)
        r = requests.post(
            f"{self.service_url}/api/voice/search",
            files={"audio": ("audio.wav", audio_bytes, "audio/wav")},
            data={"top_k": top_k},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json().get("results", [])

    def face_detect(self, img_bgr, max_face_num=None):
        """BGR 帧 → 人脸 [{user_id, name, score, matched, location}]. 异常抛给调用方."""
        import cv2 as _cv2
        max_face_num = int(max_face_num or self.face_max_face_num)
        ok, buf = _cv2.imencode(".jpg", img_bgr)
        if not ok:
            raise RuntimeError("人脸检测图 JPEG 编码失败")
        r = requests.post(
            f"{self.service_url}/api/face/detect",
            files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            data={"max_face_num": max_face_num},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json().get("faces", [])

    @staticmethod
    def cross_verify(voice_hits, face_hits):
        """声纹候选里第一个被当前帧人脸匹配的 user_id.

        返回 {"user_id", "voice_score", "face_item"} 或 None.
        """
        if not voice_hits or not face_hits:
            return None
        # 当前帧人脸匹配到的注册用户 (排除 unknown_XXX)
        present = {}
        for f in face_hits:
            uid = f.get("user_id", "")
            if f.get("matched") and uid and not uid.startswith("unknown_"):
                if uid not in present or f.get("score", 0) > present[uid].get("score", 0):
                    present[uid] = f
        if not present:
            return None
        for v in voice_hits:      # 容器已按分数降序
            uid = v.get("user_id", "")
            if uid in present:
                return {"user_id": uid,
                        "voice_score": v.get("score", 0.0),
                        "face_item": present[uid]}
        return None

    def associate_face_to_person(self, persons, face_location):
        """人脸 bbox 中心 ↔ 各 person 头关节(2D, 索引13)最近邻.

        距离 ≤ assoc_max_dist 才返回 person_index, 否则 None.
        """
        import numpy as _np
        loc = face_location or {}
        cx = float(loc.get("x", 0)) + float(loc.get("width", 0)) / 2
        cy = float(loc.get("y", 0)) + float(loc.get("height", 0)) / 2
        best_idx, best_d = None, None
        for pi, p in enumerate(persons):
            j2d = p.get("joints_2d")
            if not j2d:
                continue
            hx, hy = j2d[HEAD_JOINT_INDEX][0], j2d[HEAD_JOINT_INDEX][1]
            d = ((hx - cx) ** 2 + (hy - cy) ** 2) ** 0.5
            if best_idx is None or d < best_d:
                best_idx, best_d = pi, d
        if best_idx is None or best_d > self.assoc_max_dist:
            return None
        return best_idx

    def identify(self, audio_bytes, frame_bgr, persons, top_k=None, verbose=False):
        """声纹+人脸交叉验证 → 说话人 person_index (纯数据进出).

        流程: voice_search → face_detect → cross_verify → associate_face_to_person.
        返回 {"user_id", "name", "voice_score", "face_score", "face_location",
              "person_index"} (+ verbose 时附 voice_candidates/face_present).
        任一步失败抛 IdentifyError(msg, detail).
        """
        voice_hits = self.voice_search(audio_bytes, top_k or None)
        if not voice_hits:
            raise IdentifyError("声纹未匹配到注册用户")
        face_hits = self.face_detect(frame_bgr)
        confirmed = self.cross_verify(voice_hits, face_hits)
        if confirmed is None:
            raise IdentifyError(
                "声纹与人脸交叉验证未通过 (说话人不在画面中或两路身份不一致)",
                detail={
                    "voice_candidates": [{"user_id": v.get("user_id"), "name": v.get("name"),
                                          "score": round(float(v.get("score", 0.0)), 4)}
                                         for v in voice_hits],
                    "face_present": [{"user_id": f.get("user_id"), "name": f.get("name"),
                                      "score": round(float(f.get("score", 0.0)), 4),
                                      "matched": f.get("matched"),
                                      "location": f.get("location")} for f in face_hits],
                })
        face_item = confirmed["face_item"]
        pi = self.associate_face_to_person(persons, face_item.get("location"))
        if pi is None:
            raise IdentifyError("人脸与骨骼 person 关联失败 (头关节 2D 距离超限)")
        out = {
            "user_id": confirmed["user_id"],
            "name": face_item.get("name") or "",
            "voice_score": round(float(confirmed["voice_score"]), 4),
            "face_score": round(float(face_item.get("score", 0.0)), 4),
            "face_location": face_item.get("location"),
            "person_index": pi,
        }
        if verbose:
            out["voice_candidates"] = [{"user_id": v.get("user_id"), "name": v.get("name"),
                                        "score": round(float(v.get("score", 0.0)), 4)}
                                       for v in voice_hits]
            out["face_present"] = [{"user_id": f.get("user_id"), "name": f.get("name"),
                                    "score": round(float(f.get("score", 0.0)), 4),
                                    "matched": f.get("matched"),
                                    "location": f.get("location")} for f in face_hits]
        return out


_VERIFIER = None
_VERIFIER_LOCK = threading.Lock()


def get_face_verifier(cfg):
    """进程级单例 FaceVerifier (读 config 的 face_verify 段)."""
    global _VERIFIER
    if _VERIFIER is None:
        with _VERIFIER_LOCK:
            if _VERIFIER is None:
                _VERIFIER = FaceVerifier((cfg or {}).get("face_verify"))
    return _VERIFIER
