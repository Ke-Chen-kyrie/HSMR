"""
真实深度 3D 人体追踪 (独立脚本, 不改现有代码)

从 Foxglove bridge 同时订阅:
  - 彩色图   /.../color/image_raw/compressed
  - 对齐深度 /.../aligned_depth_to_color/image_raw/compressedDepth
  - 深度内参 /.../aligned_depth_to_color/camera_info
  - TF      /tf_static, /tf

流程:
  1. 抓最新彩色图 → 现有 HSMR 管线 → 每人 44 关节 + 虚拟相机平移
  2. 虚拟相机关节 → 2D 像素 (u = 5000*X/Z + cx)
  3. 在真实深度图采样 → 深度(米) → 真实内参反投影 → 相机光学系 XYZ(米)
  4. TF 转换 → HEAD 坐标 (base_link 若存在则转换)
  5. 输出 JSON + 3D 渲染

用法:
  python docs/realtime_depth_3d.py [--interval 5] [--max_frames 1]
      [--device cpu] [--backend pytorch] [--out docs/_depth3d]

注意: 本地 CPU 推理慢(~1分钟/帧), 机器人在 GPU 上快。
"""
import argparse
import asyncio
import json
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
for _f in ['/usr/share/fonts/fonts-gb/GB_ST_GB18030.ttf']:
    if os.path.exists(_f):
        try:
            font_manager.fontManager.addfont(_f)
            matplotlib.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=_f).get_name(), 'DejaVu Sans']
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            pass

BRIDGE = "ws://192.168.217.100:8768"
SUB_PROTO = "foxglove.sdk.v1"
HDR = struct.Struct("<xIQ")

# ── 主题 (可通过 --prefix 覆盖) ──
DEPTH_PNG_OFFSET = 12  # compressedDepth 头部长度

# ── HSMR 依赖 (模块顶层, process_frame 需要) ──
import torch
from lib.modeling.pipelines.vitdet import build_detector
from lib.kits.hsmr_demo import IMG_MEAN_255, IMG_STD_255, _img_det2patches, prepare_mesh, visualize_full_img
from lib.body_models.skel_utils.transforms import params_q2rot

# TF 跨帧累积缓存 (base_link 静态链不变)
_TF_CACHE = {}

# OpenPose BODY_25 骨架
BODY25_BONES = [
    [1,8],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],
    [8,9],[9,10],[10,11],[8,12],[12,13],[13,14],
    [1,0],[0,15],[15,17],[0,16],[16,18],
    [14,21],[19,21],[20,21],[11,24],[22,24],[23,24],
]


def quat_to_rotmat(x, y, z, w):
    n = float(np.sqrt(x*x + y*y + z*z + w*w))
    if n == 0: return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


class MultiTopicCapture:
    """Foxglove 多主题订阅: color + depth + camera_info + tf."""

    def __init__(self, url, topics):
        self.url = url
        self.topics = topics
        self.latest = {}   # topic -> deserialized msg
        self.loop = asyncio.new_event_loop()
        self.thread_stop = False

    def start(self):
        import threading
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        self.loop.run_until_complete(self._receive_forever())

    async def _receive_forever(self):
        while not self.thread_stop:
            try:
                await self._receive()
            except Exception as e:
                print(f'[capture] 连接错误: {e}')
                await asyncio.sleep(1.5)

    async def _receive(self):
        import websockets
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg
        typestore = None
        for n in ("ROS2_JAZZY", "ROS2_HUMBLE"):
            s = getattr(Stores, n, None)
            if s: typestore = get_typestore(s); break

        ws = await websockets.connect(self.url, subprotocols=[SUB_PROTO], max_size=None, open_timeout=5)
        channels = {}
        deadline = self.loop.time() + 5
        while self.loop.time() < deadline:
            try: raw = await asyncio.wait_for(ws.recv(), timeout=1)
            except asyncio.TimeoutError: continue
            if isinstance(raw, str):
                m = json.loads(raw)
                if m.get("op") == "advertise":
                    for ch in m.get("channels", []):
                        channels[ch["topic"]] = ch

        subs, sub_map = [], {}
        for i, topic in enumerate(self.topics):
            if topic in channels:
                subs.append({"id": i+1, "channelId": channels[topic]["id"]})
                sub_map[i+1] = topic
        print(f"[capture] 订阅 {len(subs)}/{len(self.topics)}: {list(sub_map.values())}")
        if not subs:
            raise RuntimeError('没有可订阅的主题')
        await ws.send(json.dumps({"op": "subscribe", "subscriptions": subs}))

        while not self.thread_stop:
            try: raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError: continue
            if not isinstance(raw, bytes) or len(raw) < 13 or raw[0] != 0x01: continue
            sid, _ = HDR.unpack_from(raw)
            topic = sub_map.get(sid)
            if not topic: continue
            ch = channels[topic]; sn = ch["schemaName"]
            if sn not in typestore.types:
                parts = ch["schema"].split("\n===\nMSG: ")
                result = {}
                if parts[0].strip(): result.update(get_types_from_msg(parts[0].strip(), sn))
                for part in parts[1:]:
                    lines = part.strip().split("\n")
                    if len(lines) >= 2:
                        result.update(get_types_from_msg("\n".join(lines[1:]).strip(), lines[0].strip()))
                new = {n: d for n, d in result.items() if n not in typestore.types}
                if new: typestore.register(new)
            try:
                msg = typestore.deserialize_cdr(raw[13:], sn)
            except Exception:
                continue
            self.latest[topic] = msg

    def snapshot(self):
        """返回各主题最新消息的引用."""
        return dict(self.latest)

    def stop(self):
        self.thread_stop = True


# ── 解码函数 ──
def decode_color(msg):
    d = np.frombuffer(msg.data, dtype=np.uint8)
    return cv2.imdecode(d, cv2.IMREAD_COLOR)  # BGR


def decode_depth(msg):
    d = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(d[DEPTH_PNG_OFFSET:], cv2.IMREAD_UNCHANGED)
    return img  # uint16 mm


def camera_info_k(msg):
    return np.array(msg.k).reshape(3, 3)


def sample_depth_m(depth, u, v, radius=3):
    """在深度图 (uint16 mm) 取 (u,v) 周围窗口的中位数, 返回米."""
    h, w = depth.shape
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < w and 0 <= v < h): return None
    u0, u1 = max(0, u-radius), min(w, u+radius+1)
    v0, v1 = max(0, v-radius), min(h, v+radius+1)
    win = depth[v0:v1, u0:u1].astype(np.float32) / 1000.0  # mm → m
    valid = win[win > 0.1]
    if len(valid) == 0: return None
    return float(np.median(valid))


def deproject(u, v, z_m, K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    return np.array([(u-cx)*z_m/fx, (v-cy)*z_m/fy, z_m])


def transform_points(pts, R, t):
    return (pts @ R.T) + t


# ═══════════════════════════════════════════════════════
# Orientation: 完整旋转表示 (矩阵+四元数+向量, 无 yaw)
# ═══════════════════════════════════════════════════════
def rot_to_quat_wxyz(M):
    """3x3 旋转矩阵 → 四元数 (w,x,y,z)."""
    M = np.asarray(M, dtype=float)
    tr = M[0,0]+M[1,1]+M[2,2]
    if tr > 0:
        s = np.sqrt(tr+1.0)*2
        q = [(0.25*s), (M[2,1]-M[1,2])/s, (M[0,2]-M[2,0])/s, (M[1,0]-M[0,1])/s]
    else:
        i = int(np.argmax([M[0,0],M[1,1],M[2,2]]))
        if i==0:
            s = np.sqrt(1.0+M[0,0]-M[1,1]-M[2,2])*2
            q = [(M[2,1]-M[1,2])/s, 0.25*s, (M[0,1]+M[1,0])/s, (M[0,2]+M[2,0])/s]
        elif i==1:
            s = np.sqrt(1.0+M[1,1]-M[0,0]-M[2,2])*2
            q = [(M[0,2]-M[2,0])/s, (M[0,1]+M[1,0])/s, 0.25*s, (M[1,2]+M[2,1])/s]
        else:
            s = np.sqrt(1.0+M[2,2]-M[0,0]-M[1,1])*2
            q = [(M[1,0]-M[0,1])/s, (M[0,2]+M[2,0])/s, (M[1,2]+M[2,1])/s, 0.25*s]
    q = np.array(q)/np.linalg.norm(q)
    return q.tolist()


def describe_orientation(root_rot):
    """root_rot: [3,3], 返回完整旋转表示 (无 yaw)."""
    root_rot = np.asarray(root_rot, dtype=float)
    body_forward = root_rot @ np.array([0.0, 0.0, 1.0])
    body_up     = root_rot @ np.array([0.0, 1.0, 0.0])
    body_left   = root_rot @ np.array([1.0, 0.0, 0.0])
    return {
        'rotation_matrix': root_rot.tolist(),          # 完整 3x3 旋转矩阵
        'quaternion_wxyz': rot_to_quat_wxyz(root_rot), # 四元数
        'body_forward': body_forward.tolist(),         # 人体前向(相机系)
        'body_up': body_up.tolist(),
        'body_left': body_left.tolist(),
        'reference': 'body_canonical(+Z前,+Y上,+X左) → camera optical',
    }


# ═══════════════════════════════════════════════════════
# 3D 渲染
# ═══════════════════════════════════════════════════════
def render_3d(people, K, out_png, frame_label):
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title(f'真实深度 3D (相机光学系, 米)\n{frame_label}', fontsize=12, fontweight='bold')
    ax.scatter([0],[0],[0], s=250, c='#a371f7', marker='s', edgecolors='white', zorder=10)
    ax.text(0.02, 0.02, 0.02, '相机', fontsize=9, color='#a371f7')
    for d, c, n in [([1,0,0],'red','X'), ([0,1,0],'green','Y↓'), ([0,0,1],'blue','Z→')]:
        ax.quiver(0,0,0, d[0],d[1],d[2], color=c, linewidth=2, arrow_length_ratio=0.2)
        ax.text(d[0]*1.1, d[1]*1.1, d[2]*1.1, n, color=c, fontsize=9)

    colors = ['#f78166', '#58a6ff', '#3fb950', '#d2991d']
    valid_pts = []
    for i, person in enumerate(people):
        col = colors[i % 4]
        opt = person['joints_optical_m']  # [44,3] or None
        if opt is None: continue
        J = np.asarray(opt, dtype=float)
        valid_pts.append(J)
        for a, b in BODY25_BONES:
            if a < len(J) and b < len(J):
                ax.plot([J[a,0],J[b,0]], [J[a,1],J[b,1]], [J[a,2],J[b,2]], '-', color=col, lw=2, alpha=0.8)
        ax.scatter(J[:,0], J[:,1], J[:,2], c=col, s=25, zorder=5)
        pelvis = person.get('pelvis_optical_m')
        if pelvis is not None:
            ax.scatter([pelvis[0]],[pelvis[1]],[pelvis[2]], s=200, c=col, marker='*', edgecolors='white', zorder=8)
            ax.plot([0,pelvis[0]],[0,pelvis[1]],[0,pelvis[2]], '--', color=col, alpha=0.5)
            d = np.linalg.norm(pelvis)
            ax.text(pelvis[0], pelvis[1], pelvis[2], f' P{i} {d:.2f}m', fontsize=9, color=col)
    ax.set_xlabel('X(右→)'); ax.set_ylabel('Y(下→)'); ax.set_zlabel('Z(远离→)')
    # 视图范围 (过滤 NaN)
    if valid_pts:
        allp = np.vstack(valid_pts)
        finite = allp[np.isfinite(allp).all(axis=1)]
        if len(finite):
            m = np.percentile(finite, 5, axis=0); M = np.percentile(finite, 95, axis=0)
            ax.set_xlim(m[0]-0.3, M[0]+0.3); ax.set_ylim(m[1]-0.3, M[1]+0.3); ax.set_zlim(m[2]-0.3, M[2]+0.3)
    ax.view_init(elev=15, azim=-70)

    # 数据表
    ax2 = fig.add_subplot(1, 2, 2); ax2.axis('off')
    lines = [f'相机内参 K (aligned depth = color):', f'  fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}', '',
             f'真实深度: {frame_label}', '']
    for i, person in enumerate(people):
        pos = person.get('position') or {}
        opt_pelvis = person.get('pelvis_optical_m')
        lines += [
            f'【Person {i}】score={person.get("score", 0):.2f}',
            f'  骨盆真实深度(光学系, m): {None if opt_pelvis is None else [round(x,3) for x in opt_pelvis]}',
            f'  距相机距离: {person.get("pelvis_depth_m", 0):.2f} m',
        ]
        if pos.get('hsmr_virtual'):
            lines.append(f'  (对比) HSMR单目估计距离: {pos["hsmr_virtual"]:.1f} m ← 假米!')
        if person.get('tf_applied'):
            lines.append(f'  TF: {person["tf_from"]} → {person["tf_to"]} 已应用')
        lines.append('')
    lines += [
        '════════ 结论 ════════',
        '真实深度来自 aligned_depth 传感器(毫米→米)',
        '反投影用 CameraInfo K, 得到相机光学系真实XYZ',
        'HSMR单目距离(假米)仅用于理解, 不能替代真实深度',
        '当前 bridge 只广播 HEAD→optical TF, 无 base_link',
    ]
    ax2.text(0.02, 0.98, '\n'.join(lines), transform=ax2.transAxes, va='top', ha='left', fontsize=8.5, linespacing=1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_png, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════
def atomic_save_rgb_jpg(img_bgr, path):
    tmp = str(path) + '.tmp.jpg'
    cv2.imwrite(tmp, img_bgr)
    os.replace(tmp, str(path))

def atomic_save_json(obj, path):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, str(path))

def depth_to_vis(depth):
    """uint16 毫米深度 → turbo 伪彩色 8bit BGR."""
    d8 = np.zeros(depth.shape, dtype=np.uint8)
    valid = depth > 0
    d8[valid] = np.clip(depth[valid] // 40, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)


def process_frame(detector, pipeline, args, topics, snap):
    """处理一帧: 检测+HSMR+真实深度融合+orientation.
    返回 dict 或 None."""
    K = camera_info_k(snap[topics[2]])
    depth = decode_depth(snap[topics[1]])
    frame_bgr = decode_color(snap[topics[0]])
    if frame_bgr is None or depth is None:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    det_out = detector([frame_rgb])
    patches, bbx_cs = _img_det2patches(frame_rgb, det_out[0][0], det_out[1][0], args.max_instances)

    # 带检测框标注图 (保存 latest.jpg, 即使无人也保存)
    annotated = frame_bgr.copy()
    for cs in bbx_cs:
        cx, cy, s = float(cs[0]), float(cs[1]), float(cs[2])
        p1 = (int(cx - s/2), int(cy - s/2)); p2 = (int(cx + s/2), int(cy + s/2))
        cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
        cv2.putText(annotated, 'person', (p1[0], max(0, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if len(patches) == 0:
        cv2.putText(annotated, 'No person', (24, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)
        return {'annotated': annotated, 'K': K, 'depth': depth, 'frame_bgr': frame_bgr,
                'people': [], 'out_data': None}

    # ── HSMR 推理 ──
    patches = patches.astype(np.float32)
    patches_n = (patches - IMG_MEAN_255) / IMG_STD_255
    patches_n = np.ascontiguousarray(patches_n.transpose(0, 3, 1, 2))
    if args.backend == 'onnx':
        outputs = pipeline(patches_n)
    else:
        with torch.no_grad():
            outputs = pipeline(torch.from_numpy(patches_n))
    pd_params = {k: v.detach().cpu().clone() for k, v in outputs['pd_params'].items()}
    pd_cam_t = outputs['pd_cam_t'].detach().cpu().clone()

    m_skin, m_skel = prepare_mesh(pipeline, pd_params, include_skeleton=True, batch_size=1)
    skel_out = pipeline.skel_model(poses=pd_params['poses'].to(args.device),
                                   betas=pd_params['betas'].to(args.device), skelmesh=False)
    joints_body = skel_out.joints.detach().cpu().numpy()

    det_meta = {'n_patch_per_img': [len(patches)], 'bbx_cs_per_img': [bbx_cs], 'bbx_cs': np.asarray(bbx_cs)}
    rendered, raw_cam_t = visualize_full_img(pd_cam_t, [frame_rgb], det_meta, m_skin, m_skel)

    # ── 2D 关节 → 真实深度 → 光学 XYZ ──
    H, W = frame_rgb.shape[:2]
    cx_v, cy_v = W/2, H/2
    f_v = 5000.0
    rotations = params_q2rot(pd_params['poses'])

    people = []
    for i in range(len(patches)):
        joints_cam = joints_body[i] + raw_cam_t[i]
        u = f_v * joints_cam[:, 0] / joints_cam[:, 2] + cx_v
        v = f_v * joints_cam[:, 1] / joints_cam[:, 2] + cy_v
        joints_2d = np.stack([u, v], axis=-1)
        joints_opt = np.zeros_like(joints_cam)
        depth_valid = []
        for j in range(len(joints_cam)):
            z = sample_depth_m(depth, u[j], v[j], radius=3)
            if z is None:
                joints_opt[j] = np.nan; depth_valid.append(False)
            else:
                joints_opt[j] = deproject(u[j], v[j], z, K); depth_valid.append(True)
        pelvis_opt = joints_opt[8]
        pelvis_depth = float(np.linalg.norm(pelvis_opt)) if depth_valid[8] else None
        orientation = describe_orientation(rotations[i, 0].numpy())

        person = {
            'person_index': i,
            'score': float(det_out[0][0]['scores'][i]) if len(det_out[0][0]['scores']) > i else None,
            'bbox_cs': np.asarray(bbx_cs[i]).tolist() if i < len(bbx_cs) else None,
            'pelvis_optical_m': pelvis_opt.tolist() if depth_valid[8] else None,
            'pelvis_depth_m': pelvis_depth,
            'joints_optical_m': joints_opt.tolist(),
            'joints_2d_color': joints_2d.tolist(),
            'depth_valid': depth_valid,
            'orientation': orientation,
            'position': {'hsmr_virtual': float(np.linalg.norm(raw_cam_t[i]))},
        }
        people.append(person)
        fwd = orientation['body_forward']
        print(f'  [P{i}] 骨盆深度={pelvis_depth:.2f}m (HSMR假={np.linalg.norm(raw_cam_t[i]):.1f}m) '
              f'前向={np.round(fwd,3)}')

    # ── TF: optical → base_link/HEAD (跨帧累积, 静态链不变) ──
    global _TF_CACHE
    for msgs in (snap.get('/tf_static'), snap.get('/tf')):
        if msgs is None or not hasattr(msgs, 'transforms'):
            continue
        for t in msgs.transforms:
            hdr = t.header; q = t.transform.rotation; tr = t.transform.translation
            R = quat_to_rotmat(q.x, q.y, q.z, q.w)
            _TF_CACHE[(hdr.frame_id, t.child_frame_id)] = (R, np.array([tr.x, tr.y, tr.z]))
    tf_chain = _TF_CACHE

    optical = 'realsense_head_color_optical_frame'
    target = args.target_frame
    if target is None:
        frames = set()
        for (a, b) in tf_chain: frames.update([a, b])
        for cand in ('base_link', 'BASE', 'base_footprint', 'world', 'map', 'odom', 'HEAD'):
            if cand in frames: target = cand; break
    if target is None:
        target = 'HEAD'

    graph = {}
    for (a, b) in tf_chain:
        graph.setdefault(a, []).append(b); graph.setdefault(b, []).append(a)

    def edge_tf(a, b):
        if (a, b) in tf_chain:
            R, t = tf_chain[(a, b)]; return R.T, -R.T @ t
        if (b, a) in tf_chain: return tf_chain[(b, a)]
        return None

    def find_path(start, goal):
        from collections import deque
        q = deque([(start, [])]); seen = {start}
        while q:
            node, path = q.popleft()
            for nxt in graph.get(node, []):
                if nxt in seen: continue
                np_ = path + [(node, nxt)]
                if nxt == goal: return np_
                seen.add(nxt); q.append((nxt, np_))
        return None

    R_opt_target, t_opt_target = np.eye(3), np.zeros(3)
    tf_applied = False
    if target == optical:
        tf_applied = True
    else:
        path = find_path(optical, target)
        if path is not None:
            R_acc, t_acc = np.eye(3), np.zeros(3)
            ok = True
            for (a, b) in path:
                etf = edge_tf(a, b)
                if etf is None: ok = False; break
                R, p = etf
                t_acc = R @ t_acc + p; R_acc = R @ R_acc
            if ok:
                R_opt_target, t_opt_target = R_acc, t_acc; tf_applied = True
        if not tf_applied:
            print(f'  [TF] 找不到 optical→{target}, 仅输出光学系')

    if tf_applied:
        for person in people:
            if person['pelvis_optical_m'] is not None:
                person['pelvis_target_m'] = transform_points(
                    np.array(person['pelvis_optical_m']), R_opt_target, t_opt_target).tolist()
            jo = person['joints_optical_m']
            jo_t = []
            for j in jo:
                if np.isnan(j).any(): jo_t.append(None)
                else: jo_t.append(transform_points(np.array(j), R_opt_target, t_opt_target).tolist())
            person['joints_target_m'] = jo_t
            person['tf_from'] = optical; person['tf_to'] = target

    out_data = {
        'schema': 'realtime_depth_3d_v1',
        'captured_unix': time.time(),
        'camera_intrinsics': {'fx': float(K[0,0]), 'fy': float(K[1,1]), 'cx': float(K[0,2]), 'cy': float(K[1,2])},
        'depth_units': 'meters (decoded from uint16 mm)',
        'tf': {'optical_frame': optical, 'target_frame': target, 'applied': tf_applied},
        'persons': people,
    }
    return {'annotated': annotated, 'K': K, 'depth': depth, 'frame_bgr': frame_bgr,
            'frame_rgb': frame_rgb, 'rendered': rendered[0], 'people': people, 'out_data': out_data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='/zj_humanoid/sensor/realsense_head')
    ap.add_argument('--interval', type=float, default=5.0)
    ap.add_argument('--max_frames', type=int, default=0, help='0=无限循环')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--backend', default='pytorch', choices=['pytorch', 'onnx'])
    ap.add_argument('--onnx_model', default='deploy/onnx/artifacts/hsmr_full_fp16.onnx')
    ap.add_argument('--model_root', default='data_inputs/released_models/HSMR-ViTH-r1d1')
    ap.add_argument('--max_instances', type=int, default=5)
    ap.add_argument('--out', default='data_outputs/depth3d')
    ap.add_argument('--target_frame', default=None, help='TF目标坐标系')
    ap.add_argument('--no_history', action='store_true', help='不保存历史帧')
    args = ap.parse_args()

    topics = [
        f'{args.prefix}/color/image_raw/compressed',
        f'{args.prefix}/aligned_depth_to_color/image_raw/compressedDepth',
        f'{args.prefix}/aligned_depth_to_color/camera_info',
        '/tf_static', '/tf',
    ]

    outputs_root = Path(args.out)
    outputs_root.mkdir(parents=True, exist_ok=True)
    history_root = outputs_root / 'frames'
    if not args.no_history:
        history_root.mkdir(parents=True, exist_ok=True)

    # ── 加载模型 (一次) ──
    print('[加载] detector + HSMR...')
    t0 = time.time()
    detector = build_detector(batch_size=1, max_img_size=512, device=args.device, use_amp=False)
    if args.backend == 'onnx':
        from deploy.onnx.runtime import HSMRONNXRuntimePipeline
        pipeline = HSMRONNXRuntimePipeline(model_path=args.onnx_model,
                                           model_root=args.model_root, device=args.device)
    else:
        from lib.modeling.pipelines.hsmr import build_inference_pipeline
        pipeline = build_inference_pipeline(model_root=args.model_root, device=args.device)
    print(f'[加载] 完成 {time.time()-t0:.1f}s, backend={args.backend}')

    # ── 连接 capture ──
    cap = MultiTopicCapture(BRIDGE, topics).start()
    t_wait = time.time()
    while time.time() - t_wait < 15:
        snap = cap.snapshot()
        if all(t in snap for t in topics[:3]): break
        time.sleep(0.5)
    missing = [t for t in topics[:3] if t not in cap.snapshot()]
    if missing:
        print(f'[错误] 缺少主题: {missing}'); cap.stop(); return

    # ── 预取 TF (攒 base_link 链, /tf 动态低频) ──
    global _TF_CACHE
    t_tf = time.time()
    while time.time() - t_tf < 10:
        snap_tf = cap.snapshot()
        for msgs in (snap_tf.get('/tf_static'), snap_tf.get('/tf')):
            if msgs is None or not hasattr(msgs, 'transforms'): continue
            for t in msgs.transforms:
                hdr = t.header; q = t.transform.rotation; tr = t.transform.translation
                R = quat_to_rotmat(q.x, q.y, q.z, q.w)
                _TF_CACHE[(hdr.frame_id, t.child_frame_id)] = (R, np.array([tr.x, tr.y, tr.z]))
        if len(_TF_CACHE) > 5: break
        time.sleep(0.5)
    print(f'[TF] 预取 {len(_TF_CACHE)} 条边')

    print(f'[运行] interval={args.interval}s, max_frames={args.max_frames or "无限"}')
    print(f'[运行] 输出: {outputs_root}/')
    print('  latest.jpg           — 检测图片(带人框)')
    print('  latest_render.jpg    — HSMR 渲染叠加')
    print('  latest_depth.png     — 深度图(伪彩色)')
    print('  latest_3d.json       — 深度3D+orientation(无yaw)')
    print('  latest_comprehensive.png — 综合4面板')
    print('  frames/              — 历史(时间戳命名)')
    print('  按 Ctrl+C 停止')

    sample_id = 0
    next_due = time.monotonic()
    try:
        while args.max_frames == 0 or sample_id < args.max_frames:
            wait = next_due - time.monotonic()
            if wait > 0: time.sleep(wait)

            # 抓最新同步帧
            t_c = time.time()
            while time.time() - t_c < max(2.0, args.interval):
                snap = cap.snapshot()
                if all(t in snap for t in topics[:3]): break
                time.sleep(0.5)

            sample_id += 1
            t_f = time.time()
            res = process_frame(detector, pipeline, args, topics, snap)
            if res is None:
                print(f'[sample {sample_id}] 解码失败')
                next_due += args.interval; continue

            # 保存: 检测图片 + 深度图 (总是保存)
            atomic_save_rgb_jpg(res['annotated'], outputs_root / 'latest.jpg')
            atomic_save_rgb_jpg(depth_to_vis(res['depth']), outputs_root / 'latest_depth.png')

            if res['out_data'] is None:
                empty = {'schema': 'realtime_depth_3d_v1', 'sample_id': sample_id,
                         'captured_unix': time.time(), 'persons_count': 0, 'persons': []}
                atomic_save_json(empty, outputs_root / 'latest_3d.json')
                print(f'[sample {sample_id}] 无人 ({time.time()-t_f:.1f}s)')
                next_due += args.interval; continue

            out_data = res['out_data']
            atomic_save_json(out_data, outputs_root / 'latest_3d.json')
            atomic_save_rgb_jpg(cv2.cvtColor(res['rendered'], cv2.COLOR_RGB2BGR),
                                outputs_root / 'latest_render.jpg')
            try:
                from render_depth_comprehensive import render_comprehensive
                render_comprehensive(res['frame_bgr'], res['depth'], res['K'], res['people'],
                                     str(outputs_root / 'latest_comprehensive.png'),
                                     time.strftime('%H:%M:%S'))
            except Exception as e:
                print(f'[渲染] 综合渲染失败: {e}')

            if not args.no_history:
                ts = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S-%f')[:-3]
                stem = f'{sample_id:06d}-{ts}'
                atomic_save_rgb_jpg(res['annotated'], history_root / f'{stem}.jpg')
                atomic_save_json(out_data, history_root / f'{stem}.3d.json')

            print(f'[sample {sample_id}] {len(res["people"])}人, 耗时{time.time()-t_f:.1f}s')
            next_due += args.interval

    except KeyboardInterrupt:
        print('\n[停止] Ctrl+C')
    finally:
        cap.stop()
        print('[结束]')


if __name__ == '__main__':
    main()
