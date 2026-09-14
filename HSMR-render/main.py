"""HSMR-render 独立渲染服务 (FastAPI).

职责: 与 HSMR-infer 各自收同样的关节选择 POST —— HSMR 只返回 3D 位置/朝向 (不渲染),
本服务只渲染不返 3D: 自抓 foxglove 帧 → 完整 HSMR 推理拿网格参数 (poses/betas/cam_t,
/infer 契约不含它们, 故本服务自备模型) → SKEL 蒙皮蓝+骨骼黄 2D 叠加渲染 + 选中关节光晕
→ 入渲染队列; 后台再渲染后 N 帧 (自抓相机流).

POST /joint  body = key-value 关节选择 ({"hand_l":1, "head":0}, value=1选中/0不选)
            query person_index 第几个人 (0 起); all/-1/全部/所有人 = 一次渲染所有检测到的人 (批量合成)
            → 渲染当前帧入队 (同步等), 返回渲染 ack (不含 3D 位置/朝向)
            → 后台线程再渲染后 N 帧 (seq=1..N) 入队
GET  /render/pop  从渲染队列取一张图 (取走即移除); 空队列返回 {"rendered": null}
GET  /render/queue 渲染队列状态 (调试)
GET  /joints 24 关节名列表
GET  /health 服务状态
"""
import base64
import io
import os
import queue
import threading
import time
import concurrent.futures
# pyrender 首次 import 前必须指定 EGL 平台 (Jetson headless, 无 X display)
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
from contextlib import asynccontextmanager
from typing import Dict, Union

import numpy as np
import yaml
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile

# ── 路径引导: 先读 config 拿 hsrm_root, 再插 sys.path (lib/deploy/thirdparty 以 hsrm_root 为基准) ──
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_cfg_raw():
    path = os.environ.get("RENDER_SERVICE_CONFIG", os.path.join(_HERE, "config.yaml"))
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(root, p):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(root, p)


_CFG_BOOT = _load_cfg_raw()
_HSRM_ROOT = os.path.abspath(os.path.expanduser(_CFG_BOOT["hsrm_root"]))
sys.path.insert(0, _HSRM_ROOT)
sys.path.insert(0, _HERE)
# SKEL 包 (thirdparty/SKEL/skel)
sys.path.insert(0, os.path.join(_HSRM_ROOT, "thirdparty", "SKEL"))
# 若 SKEL 有 SMPL 依赖
_smpl = os.path.join(_HSRM_ROOT, "thirdparty", "SMPL")
if os.path.isdir(_smpl):
    sys.path.insert(0, _smpl)

def _load_cfg():
    """读配置并解析相对路径: 模型路径相对 hsrm_root, out_dir 相对本工程 (启动 CWD 无关)."""
    cfg = _load_cfg_raw()
    cfg["model"]["onnx_model"] = _resolve(_HSRM_ROOT, cfg["model"]["onnx_model"])
    cfg["model"]["model_root"] = _resolve(_HSRM_ROOT, cfg["model"]["model_root"])
    cfg["render"]["out_dir"] = _resolve(_HERE, cfg["render"]["out_dir"])
    return cfg


from capture import MultiTopicCapture, RosbridgeCapture
from inference import InferenceWorker
from tf_chain import TransformChain

# 24 SKEL 解剖关节名 (thirdparty/SKEL/skel/kin_skel.py)
SKEL_JOINTS = [
    "pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
    "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l",
    "lumbar_body", "thorax", "head",
    "scapula_r", "humerus_r", "ulna_r", "radius_r", "hand_r",
    "scapula_l", "humerus_l", "ulna_l", "radius_l", "hand_l",
]
JOINT_INDEX = {n: i for i, n in enumerate(SKEL_JOINTS)}

# 中文关节名 (与 SKEL_JOINTS 一一对应)
JOINT_NAMES_CN = [
    "骨盆", "右髋", "右膝", "右踝", "右跟骨", "右趾",
    "左髋", "左膝", "左踝", "左跟骨", "左趾",
    "腰椎", "胸椎", "头",
    "右肩胛", "右肩", "右肘", "右前臂", "右手",
    "左肩胛", "左肩", "左肘", "左前臂", "左手",
]
# 中文 → 索引 (接受中文查询)
JOINT_CN_INDEX = {cn: i for i, cn in enumerate(JOINT_NAMES_CN)}
# 别名: 手腕/手 都选 hand_r(18)/hand_l(23), 而不是 radius (其在肘部)
JOINT_CN_INDEX.update({
    "右腕": 18, "右手腕": 18, "手腕": 18, "手": 18, "手部": 18,
    "左腕": 23, "左手腕": 23,
})

# 服务状态 (全局, 由 lifespan 填充)
STATE = {
    "capture": None,
    "worker": None,
    "cfg": None,
    "tf_chain": None,
    "start_time": time.time(),
}


def _build_topics(cfg):
    if cfg.get("source", "foxglove") == "rosbridge":
        ros1 = cfg["topics"].get("ros1", {})
        return [
            ros1.get("color", f"{cfg['topics']['prefix']}/color/image_raw/compressed"),
            ros1.get("depth", f"{cfg['topics']['prefix']}/aligned_depth_to_color/image_raw/compressed"),
            ros1.get("info", f"{cfg['topics']['prefix']}/aligned_depth_to_color/camera_info"),
            ros1.get("tf_static", "/tf_static"),
            ros1.get("tf", "/tf"),
        ]
    prefix = cfg["topics"]["prefix"]
    return [
        f"{prefix}/color/image_raw/compressed",
        f"{prefix}/aligned_depth_to_color/image_raw/compressedDepth",  # 需 realsense compressedDepth.format=png
        f"{prefix}/aligned_depth_to_color/camera_info",
        "/tf_static",   # 坐标系变换链 (光学→HEAD/BASE/world)
        "/tf",
    ]


def _build_capture(cfg):
    """按 source 选择采集器: foxglove(ROS2 bridge) 或 rosbridge(ROS1)."""
    if cfg.get("source", "foxglove") == "rosbridge":
        return RosbridgeCapture(cfg["bridge"]["rosbridge_url"], cfg["topics"]["list"]).start()
    return MultiTopicCapture(cfg["bridge"]["url"], cfg["topics"]["list"]).start()


# ── SKEL 骨骼网格渲染 (复用旧 joint_service 实现, pyrender/EGL 必须固定单线程) ──
_SKIN_BLEND = 0.7
_SKEL_BLEND = 0.3
_GLOW_ALPHA = 0.7        # 光圈中心最大不透明度 (低一点更柔和)


class _SkelRenderer:
    """模型自带 SKEL 骨骼网格渲染 (pyrender EGL). 必须固定单线程调用."""

    def __init__(self, model_root, device):
        import torch
        from deploy.orin.hsmr_core import load_skel_model
        self._torch = torch
        self.device = device
        self.skel = load_skel_model(model_root=model_root, device=device)
        self.skel.eval()
        self.skel_f = self.skel.skel_f.cpu().numpy()    # (NF,3) 骨骼网格面片
        self.skin_f = self.skel.skin_f.cpu().numpy()    # (NF_skin,3) 蒙皮网格面片
        # osim 父子骨架: skel.parent 是 23 项 (索引 k → 关节 k+1 的父); 根关节(0)标 -1.
        try:
            _p = self.skel.parent.cpu().numpy()
            self.joint_parent = np.concatenate([[-1], np.asarray(_p, dtype=np.int64)])
        except Exception:
            self.joint_parent = None

    def render(self, frame_rgb, poses, betas, cam_t, glow_joints):
        """单人渲染 (兼容旧接口): 委托 render_all 单元素批."""
        return self.render_all(frame_rgb, [poses], [betas], [cam_t], glow_joints)

    def render_all(self, frame_rgb, poses_list, betas_list, cam_t_list, glow_joints):
        """多人渲染: 所有人蒙皮(蓝)+骨骼(黄)网格批量离屏渲染合成一次,
        再 0.7/0.3 混合 (同 lib.kits.hsmr_demo 的 front_blend), 最后逐人给选中关节画光晕."""
        n = len(poses_list)
        if n == 0:
            return frame_rgb.copy()
        with self._torch.no_grad():
            poses_t = self._torch.as_tensor(np.asarray(poses_list, dtype=np.float32),
                                            device=self.device)          # (N,72)
            betas_t = self._torch.as_tensor(np.asarray(betas_list, dtype=np.float32),
                                            device=self.device)          # (N,16)
            skel_out = self.skel(poses=poses_t, betas=betas_t, skelmesh=True)
            v_skin = skel_out.skin_verts.detach().cpu().numpy()   # (N,6890,3) 蒙皮
            v_skel = skel_out.skel_verts.detach().cpu().numpy()   # (N,Nv,3) 骨骼
            # SKELWrapper 会把 out.joints 替换成 44×3 OpenPose 顺序, 必须用
            # joints_backup (原始 SKEL 24×3, osim 顺序 = SKEL_JOINTS) 才能对上关节名.
            jskel = getattr(skel_out, "joints_backup", skel_out.joints)
            joints = jskel.detach().cpu().numpy()                  # (N,24,3) 与 skel_verts 同坐标系

        H, W = frame_rgb.shape[:2]
        K4 = [5000.0, 5000.0, W / 2, H / 2]
        cam_t = np.asarray(cam_t_list, dtype=np.float32)          # (N,3)
        from lib.utils.vis.py_renderer import render_meshes_overlay_img
        skin_img = render_meshes_overlay_img(
            faces_all=np.tile(self.skin_f[None], (n, 1, 1)), verts_all=v_skin,
            cam_t_all=cam_t, K4=K4, img=frame_rgb, mesh_color="blue", view="front")
        skel_img = render_meshes_overlay_img(
            faces_all=np.tile(self.skel_f[None], (n, 1, 1)), verts_all=v_skel,
            cam_t_all=cam_t, K4=K4, img=frame_rgb, mesh_color="human_yellow", view="front")
        import cv2 as _cv2
        blend = _cv2.addWeighted(skin_img, _SKIN_BLEND, skel_img, _SKEL_BLEND, 0)

        if not glow_joints:
            return blend

        # 逐人叠加选中关节光晕 (纯 2D 合成, 各自用自己帧的 24 关节投影)
        out = blend
        for i in range(n):
            out = _overlay_joint_glows(out, cam_t[i], joints[i], set(glow_joints),
                                       self.joint_parent, _GLOW_ALPHA)
        return out


# 光晕颜色 (RGB 0~1) — 橘色 (与蓝蒙皮/黄骨骼都有色差, 柔和不刺眼)
_GLOW_RGB = (1.0, 0.55, 0.12)


def _overlay_joint_glows(img_rgb, cam_t, joints, joint_set, parent=None, alpha=0.9):
    """以每个选中关节投影点为圆心, 画径向渐变光晕 (中心深→向外渐淡, 无轮廓/无边界).
    纯 2D 合成, 不经过 pyrender. 光晕半径 = 该关节→父关节骨骼长度投影 × f/z × 0.35,
    再 clamp 到 [10,55]px."""
    H, W = img_rgb.shape[:2]
    out = img_rgb.astype(np.float32).copy()
    fx = fy = 5000.0
    cx, cy = W / 2, H / 2
    cam_t = np.asarray(cam_t, dtype=np.float32)
    glow_rgb = 255.0 * np.asarray(_GLOW_RGB, dtype=np.float32)

    parent = np.asarray(parent, dtype=np.int64) if parent is not None else None
    bone_len = np.zeros(24, dtype=np.float32)
    for jidx in range(24):
        if parent is not None and parent[jidx] >= 0:
            bone_len[jidx] = float(np.linalg.norm(joints[jidx] - joints[parent[jidx]]))
        else:
            dd = np.linalg.norm(joints - joints[jidx], axis=1)
            dd[jidx] = np.inf
            bone_len[jidx] = float(dd.min())

    glows = []   # (u, v, radius_px, sigma_px)
    for jidx in sorted(joint_set):
        p = joints[jidx] + cam_t   # 全图相机系 (与 inference 投影一致, z 正)
        if p[2] <= 0:
            continue
        u = fx * p[0] / p[2] + cx
        vy = fy * p[1] / p[2] + cy
        r = float(bone_len[jidx] * fx / p[2]) * 0.35
        r = float(np.clip(r, 10.0, 55.0))
        glows.append((u, vy, r, max(r * 0.75, 12.0)))
    if not glows:
        return np.asarray(img_rgb).copy()

    yy, xx = np.mgrid[0:H, 0:W]
    for u, vy, r, sigma in glows:
        d = np.hypot(xx - u, yy - vy)
        w = np.clip(np.exp(-0.5 * (d / sigma) ** 2), 0.0, 1.0) * alpha
        out = glow_rgb[None, None, :] * w[..., None] + out * (1.0 - w[..., None])
    return out.astype(np.uint8)


# 渲染结果队列 + 单线程渲染执行器 (pyrender/EGL 必须固定单线程) + 后台批次防重叠锁
_RENDER_QUEUE = queue.Queue(maxsize=50)
_RENDER_EXECUTOR = None
_SKEL_RENDERER = None
_SKEL_RENDERER_LOCK = threading.Lock()   # SKEL 渲染模型单例双检锁 (全进程只加载一次)
_BG_GENERATION_LOCK = threading.Lock()


def _get_skel_renderer(cfg=None):
    """SKEL 骨骼网格渲染器单例: 双检锁, 全进程只加载一次.
    启动时 (lifespan) 首次加载; 请求/后台渲染不复载, 只复用同一实例."""
    global _SKEL_RENDERER
    if _SKEL_RENDERER is None:
        with _SKEL_RENDERER_LOCK:
            if _SKEL_RENDERER is None:
                if cfg is None:
                    cfg = STATE["cfg"]
                _SKEL_RENDERER = _SkelRenderer(cfg["model"]["model_root"],
                                               cfg["model"]["device"])
    return _SKEL_RENDERER


def _ensure_render_executor():
    global _RENDER_EXECUTOR
    if _RENDER_EXECUTOR is None:
        _RENDER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="skel-render")
    return _RENDER_EXECUTOR


def _resolve_frame(chain, frame):
    """解析目标坐标系 → (resolved, R_opt_target, t_opt_target, requested, fallback, tf_source).

    别名: camera/光学系/头部相机/相机 → 光学系; HEAD/头 → HEAD。
    若请求的坐标系在 TF 树中无对应变换 (BASE/world 纯 TF 找不到), 先兜底展示 HEAD
    (HEAD 有 verified 兜底矩阵, 无 /tf 也可用); HEAD 兜底也缺失才退回光学系."""
    requested = frame
    if frame in ("camera", "光学系", "头部相机", "相机", "realsense_head_color_optical_frame"):
        frame = chain.optical_frame
    elif frame in ("HEAD", "头", "head"):
        frame = chain.head_frame
    r = chain.get_opt_to_target(frame)
    if r is None:
        frame = chain.head_frame
        r = chain.get_opt_to_target(frame)
        fallback = True
        if r is None:
            frame = chain.optical_frame
            r = (np.eye(3), np.zeros(3))
    else:
        fallback = False
    R, t = r
    return frame, R, t, requested, fallback, chain.transform_source(frame)


def _person_positions(person, R, t):
    """24 关节位置 → 目标坐标系 (米). 无效深度/NaN → None."""
    pos_opt = np.asarray(person["joints_optical_m"], dtype=float)  # (24,3)
    dv = person["depth_valid"]
    pos = (pos_opt @ R.T) + t
    out = []
    for j in range(24):
        if not dv[j] or not np.isfinite(pos[j]).all():
            out.append(None)
        else:
            out.append([float(x) for x in pos[j]])
    return out


def _frame_ctx_from_state(frame="HEAD"):
    """从 STATE 取 tf_chain 解析 frame; 无 TF 链时退回光学系. 返回 dict."""
    chain = STATE.get("tf_chain")
    if chain is None:
        return {"frame": "realsense_head_color_optical_frame", "R": np.eye(3),
                "t": np.zeros(3), "requested_frame": frame,
                "frame_fallback": False, "tf_source": "identity"}
    resolved, R, t, requested, fallback, src = _resolve_frame(chain, frame)
    return {"frame": resolved, "R": R, "t": t, "requested_frame": requested,
            "frame_fallback": fallback, "tf_source": src}


def _render_and_enqueue(latest, person_index, glow_joints, en_names, cn_names, seq,
                        cfg_render, all_persons=False, frame_ctx=None):
    """在单线程渲染执行器里渲染一帧并入队 (pyrender/EGL 单线程硬约束).
    返回入队条目 dict 或 None. 供当前帧 (seq=0) 与后台后 N 帧共用.
    all_persons=True 时一次渲染所有检测到的人 (批量合成), person_index 被忽略.
    frame_ctx 非空时把 24 关节位置(目标坐标系)一起放进条目."""
    def _task():
        frame_rgb = latest.get("frame_rgb")
        poses = latest.get("poses")
        betas = latest.get("betas")
        raw_cam_t = latest.get("raw_cam_t")
        if frame_rgb is None or poses is None or betas is None or raw_cam_t is None:
            print(f"[render] seq={seq}: 缺少渲染输入 (poses/betas/raw_cam_t/frame)")
            return None
        if all_persons:
            img = _get_skel_renderer().render_all(frame_rgb, list(poses), list(betas),
                                                  list(raw_cam_t), set(glow_joints))
            pi = None
        else:
            img = _get_skel_renderer().render(frame_rgb, poses[person_index],
                                              betas[person_index], raw_cam_t[person_index],
                                              set(glow_joints))
            pi = int(person_index)
        import cv2 as _cv2
        img_bgr = img[:, :, ::-1]
        if cfg_render.get("save_output", True):
            try:
                os.makedirs(cfg_render["out_dir"], exist_ok=True)
                _cv2.imwrite(os.path.join(cfg_render["out_dir"],
                                          f"render_{int(time.time())}_seq{seq}.jpg"), img_bgr)
            except Exception as e:
                print(f"[render] 存盘失败: {e}")
        ok, buf = _cv2.imencode(".jpg", img_bgr)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        persons = latest.get("persons") or []
        # 位置 (目标坐标系): 每人都给 24 关节坐标
        if frame_ctx is not None:
            persons_positions = [
                {"person_index": i, "score": p.get("score"),
                 "positions": _person_positions(p, frame_ctx["R"], frame_ctx["t"])}
                for i, p in enumerate(persons)
            ]
        else:
            persons_positions = []
        item = {
            "joints": list(en_names),
            "joints_cn": list(cn_names),
            "timestamp": latest.get("timestamp"),
            "image_base64": b64,
            "seq": int(seq),
            "person_index": pi,
            "all_persons": bool(all_persons),
            "num_persons": len(persons),
            "frame": frame_ctx["frame"] if frame_ctx else None,
            "requested_frame": frame_ctx["requested_frame"] if frame_ctx else None,
            "frame_fallback": frame_ctx["frame_fallback"] if frame_ctx else None,
            "tf_source": frame_ctx["tf_source"] if frame_ctx else None,
            "positions": persons_positions,
            "queued_at": time.time(),
        }
        try:
            _RENDER_QUEUE.put(item, timeout=5)
        except queue.Full:
            print(f"[render] 渲染队列满, 丢弃 seq={seq}")
            return None
        return item

    try:
        fut = _ensure_render_executor().submit(_task)
        return fut.result(timeout=30)
    except Exception as e:
        print(f"[render] seq={seq} 渲染/入队失败: {e}")
        return None


def _schedule_next_frames(cfg, worker, person_index, glow_joints, en_names, cn_names,
                          all_persons=False):
    """后台渲染后 N 帧 (自抓相机流). 上一批次未结束时再 POST 不重复启动. 返回是否启动.
    all_persons=True 时每帧渲染所有检测到的人."""
    n = int(cfg["render"].get("next_frames", 0) or 0)
    if n <= 0:
        return False
    if not _BG_GENERATION_LOCK.acquire(blocking=False):
        print("[bg] 上一批后帧生成中, 跳过新批次")
        return False

    def _run():
        try:
            interval = float(cfg["render"].get("next_frame_interval", 0) or 0)
            for i in range(1, n + 1):
                if interval > 0:
                    time.sleep(interval)
                try:
                    latest = worker.infer_fresh()   # 每次抓最新 live 帧 → 推理
                    if latest is None or not (latest.get("persons") or []):
                        print(f"[bg] seq={i}: 无人生成, 跳过")
                        continue
                    if all_persons:
                        _render_and_enqueue(latest, 0, glow_joints, en_names, cn_names,
                                            i, cfg["render"], all_persons=True)
                    else:
                        pi = min(person_index, len(latest["persons"]) - 1)
                        _render_and_enqueue(latest, pi, glow_joints, en_names, cn_names,
                                            i, cfg["render"])
                except Exception as e:
                    print(f"[bg] seq={i} 失败: {e}")
        finally:
            _BG_GENERATION_LOCK.release()

    threading.Thread(target=_run, daemon=True, name="render-next-frames").start()
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = _load_cfg()
    STATE["cfg"] = cfg
    device = cfg["model"]["device"]
    os.chdir(_HSRM_ROOT)   # 兜底 (路径已绝对化; 仅防御历史相对路径依赖)

    topics = _build_topics(cfg)
    cfg["topics"]["list"] = topics

    # 坐标系变换链: optical→HEAD/BASE/world (HEAD 有 verified 兜底矩阵, 无 TF 也可用)
    try:
        tf_chain = TransformChain(cfg)
        STATE["tf_chain"] = tf_chain
    except Exception as e:
        print(f"[main] TransformChain 初始化失败: {e}")
        STATE["tf_chain"] = None

    cap = _build_capture(cfg)
    STATE["capture"] = cap

    # 等订阅就绪 (最多 15s)
    t_wait = time.time()
    while time.time() - t_wait < 15:
        snap = cap.snapshot()
        if all(t in snap for t in topics[:3]):
            break
        time.sleep(0.5)

    worker = InferenceWorker(cfg, cap, device=device, tf_chain=STATE["tf_chain"])
    worker.start()
    STATE["worker"] = worker

    # SKEL 骨骼网格渲染器单例 + 单线程执行器 (pyrender/EGL 必须单线程; 启动时加载一次, 请求不复载)
    try:
        _get_skel_renderer(cfg)
        _ensure_render_executor()
        print("[main] SKEL 骨骼渲染器就绪")
    except Exception as e:
        print(f"[main] SKEL 渲染器初始化失败: {e}")

    print(f"[main] 服务就绪, device={device}, next_frames={cfg['render'].get('next_frames', 0)}")
    yield

    worker.stop()
    cap.stop()


app = FastAPI(title="HSMR-render", lifespan=lifespan)


@app.get("/health")
def health():
    worker = STATE["worker"]
    return {
        "status": "ok" if (worker is not None and worker.ready()) else "loading",
        "model_loaded": worker.ready() if worker is not None else False,
        "device": (STATE["cfg"] or {}).get("model", {}).get("device"),
        "queue_size": _RENDER_QUEUE.qsize(),
        "uptime_s": round(time.time() - STATE["start_time"], 1),
    }


@app.get("/joints")
def joints():
    return {"joints": SKEL_JOINTS, "joints_cn": JOINT_NAMES_CN, "count": 24}


def _decode_uploaded_depth(payload, filename):
    """解码 8010 转发来的 uint16 PNG/.npy 深度图。"""
    if not payload:
        return None
    if (filename or "").lower().endswith(".npy"):
        depth = np.load(io.BytesIO(payload), allow_pickle=False)
    else:
        import cv2 as _cv2
        depth = _cv2.imdecode(np.frombuffer(payload, np.uint8), _cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
        raise HTTPException(status_code=400,
                            detail="depth 必须是 uint16 单通道 PNG 或 .npy（毫米）")
    return depth


def _nearest_person_index(latest):
    """优先按真实骨盆深度选最近者；无有效深度时退回模型虚拟相机 z。"""
    persons = latest.get("persons") or []
    raw_cam_t = latest.get("raw_cam_t")
    candidates = []
    for i, person in enumerate(persons):
        z = None
        joints = person.get("joints_optical_m") or []
        if joints and joints[0] is not None and len(joints[0]) >= 3:
            pelvis_z = joints[0][2]
            if pelvis_z is not None and np.isfinite(pelvis_z) and pelvis_z > 0:
                z = float(pelvis_z)
        if z is None and raw_cam_t is not None and i < len(raw_cam_t):
            virtual_z = float(raw_cam_t[i][2])
            if np.isfinite(virtual_z) and virtual_z > 0:
                z = virtual_z
        if z is not None:
            candidates.append((z, i))
    return min(candidates)[1] if candidates else 0


@app.post("/infer")
def render_forwarded_infer(
    image: UploadFile = File(..., description="由 8010 转发的彩色图"),
    depth: UploadFile = File(None, description="可选 uint16 深度 PNG/.npy"),
    fx: float = Form(None),
    fy: float = Form(None),
    cx: float = Form(None),
    cy: float = Form(None),
    max_instances: int = Form(None),
):
    """接收 8010 的同一份 /infer 输入，渲染其中距离相机最近的人。"""
    worker = STATE["worker"]
    if worker is None or not worker.ready():
        raise HTTPException(status_code=503, detail="渲染服务尚未就绪")

    import cv2 as _cv2
    frame_bgr = _cv2.imdecode(
        np.frombuffer(image.file.read(), np.uint8), _cv2.IMREAD_COLOR
    )
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="无法解码 image")

    depth_arr, K = None, None
    if depth is not None and depth.filename:
        depth_arr = _decode_uploaded_depth(depth.file.read(), depth.filename)
        if None in (fx, fy, cx, cy):
            raise HTTPException(status_code=400,
                                detail="提供 depth 时必须同时给 fx/fy/cx/cy")
        if depth_arr.shape[:2] != frame_bgr.shape[:2]:
            raise HTTPException(status_code=400,
                                detail="depth 与彩色图尺寸不一致")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    try:
        latest = worker.infer_image(frame_bgr, depth_arr, K, max_instances)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    if latest is None:
        raise HTTPException(status_code=503, detail="转发图片推理失败")
    persons = latest.get("persons") or []
    if not persons:
        raise HTTPException(status_code=404, detail="图片中未检测到人")

    person_index = _nearest_person_index(latest)
    cfg_render = STATE["cfg"]["render"]
    frame_ctx = _frame_ctx_from_state("HEAD")
    item = _render_and_enqueue(
        latest,
        person_index,
        set(),
        SKEL_JOINTS,
        JOINT_NAMES_CN,
        0,
        cfg_render,
        all_persons=False,
        frame_ctx=frame_ctx,
    )
    return {
        "queued": item is not None,
        "queue_size": _RENDER_QUEUE.qsize(),
        "timestamp": latest.get("timestamp"),
        "person_index": person_index,
        "num_persons": len(persons),
        "infer_ms": latest.get("infer_ms"),
        "source": "infer8010_forward",
    }


@app.post("/joint")
def query_joint(
    body: Dict[str, int] = Body(..., description="key-value 关节选择: 关节名→1选中/0不选 (中英文均可, 如 {\"hand_l\":1,\"head\":0})"),
    person_index: Union[int, str] = Query(0, description="第几个人 (0 起); all/-1/everyone/全部/所有人 = 一次渲染所有检测到的人"),
):
    """POST = 触发渲染: 自抓当前帧推理 → SKEL 蒙皮+骨骼 2D 叠加渲染 → 入渲染队列;
    后台再渲染后 N 帧 (自抓相机流). 本服务只渲染, 不返回 3D 位置/朝向 (那是 HSMR 的事).
    person_index=all 时一次渲染所有检测到的人 (批量合成)."""
    WHOLE_KEYWORDS = {"整体", "全身", "whole", "whole_body", "body", "person", "人"}

    # ① 解析选中关节 (value=1 的 key; 中英文关节名均可)
    selected = []
    is_whole = False
    unknown = []
    for k, v in body.items():
        if v != 1:
            continue
        if k in WHOLE_KEYWORDS:
            is_whole = True
            continue
        idx = JOINT_INDEX.get(k)
        if idx is None:
            idx = JOINT_CN_INDEX.get(k)
        if idx is None:
            unknown.append(k)
            continue
        selected.append(idx)
    if unknown:
        raise HTTPException(status_code=404,
                            detail=f"未知关节名: {unknown}, 可选: {SKEL_JOINTS} / {JOINT_NAMES_CN} / 整体")
    if is_whole:
        selected = list(range(24))
    if not selected:
        raise HTTPException(status_code=400, detail="请至少选中一个关节 (value=1)")
    en_names = [SKEL_JOINTS[j] for j in selected]
    cn_names = [JOINT_NAMES_CN[j] for j in selected]

    worker = STATE["worker"]
    if worker is None or not worker.ready():
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    # ② 当场抓当前帧并推理 (不读缓存, 不重载模型)
    try:
        latest = worker.infer_fresh()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="模型尚未加载完成, 请稍候")
    if latest is None:
        raise HTTPException(status_code=503, detail="推理失败, 请检查相机/深度话题")
    persons = latest.get("persons") or []
    if not persons:
        raise HTTPException(status_code=404, detail="当前帧未检测到人")

    # 解析 person_index: 整数=第几个人; all/-1/全部/所有人=一次渲染所有检测到的人
    all_persons = False
    if isinstance(person_index, str):
        _pix = person_index.strip().lower()
        if _pix in ("all", "-1", "everyone", "全部", "所有人"):
            all_persons = True
        else:
            try:
                person_index = int(_pix)
            except ValueError:
                raise HTTPException(status_code=400,
                                    detail=f"person_index 需为整数或 all, 收到: {person_index!r}")
    if not all_persons and person_index == -1:
        all_persons = True
    if not all_persons and person_index >= len(persons):
        raise HTTPException(status_code=404,
                            detail=f"person_index {person_index} 超出检测人数 {len(persons)}")

    cfg_render = STATE["cfg"]["render"]
    glow = set() if is_whole else set(selected)
    if not cfg_render.get("glow_enabled", True):
        glow = set()

    # ③ 渲染当前帧 (同步等, 保证返回时图已入队)
    try:
        item = _render_and_enqueue(latest, person_index, glow, en_names, cn_names,
                                   0, cfg_render, all_persons=all_persons)
    except Exception as e:
        print(f"[main] 渲染当前帧失败: {e}")
        item = None

    # ④ 后台后 N 帧 (自抓相机流)
    bg_started = _schedule_next_frames(STATE["cfg"], worker, person_index,
                                       glow, en_names, cn_names,
                                       all_persons=all_persons)

    return {
        "queued": item is not None,
        "queue_size": _RENDER_QUEUE.qsize(),
        "timestamp": latest.get("timestamp"),
        "rendered_seq": 0,
        "next_frames": cfg_render.get("next_frames", 0),
        "person_index": None if all_persons else person_index,
        "all_persons": all_persons,
        "num_persons": len(persons),
        "joints": en_names,
        "joints_cn": cn_names,
        "glow": bool(glow),
        "infer_ms": latest.get("infer_ms"),
        "bg_started": bg_started,
    }


@app.get("/render/pop")
def pop_rendered():
    """从渲染队列取一张图 (取走即移除). 空队列返回 {"rendered": null}."""
    try:
        item = _RENDER_QUEUE.get_nowait()
    except queue.Empty:
        return {"rendered": None}
    return {"rendered": item}


@app.get("/render/queue")
def render_queue_status():
    """渲染队列状态 (调试用)."""
    return {"queue_size": _RENDER_QUEUE.qsize()}


if __name__ == "__main__":
    cfg = _load_cfg()
    uvicorn.run(app, host=cfg["server"]["host"], port=cfg["server"]["port"])
