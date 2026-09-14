"""
Base_link 转换验证 — 把光学系坐标转到机器人 BASE

链路: color_optical → HEAD → NECK → WAIST_PITCH → WAIST_YAW → KNEE → ANKLE → BASE
用法(机器人上): .venv_orin/bin/python docs/base_link_demo.py
"""
import asyncio, json, struct, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_depth_3d import MultiTopicCapture, BRIDGE

TOPICS = ['/tf_static', '/tf']

def quat_to_rotmat(x, y, z, w):
    n = float(np.sqrt(x*x+y*y+z*z+w*w))
    if n == 0: return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def T4(R, t):
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=np.array(t); return T

def main():
    cap = MultiTopicCapture(BRIDGE, TOPICS).start()
    t0 = time.time()
    tf_chain = {}
    while time.time()-t0 < 12:
        snap = cap.snapshot()
        tf_chain = {}
        for msgs in (snap.get('/tf_static'), snap.get('/tf')):
            if msgs is None or not hasattr(msgs, 'transforms'): continue
            for t in msgs.transforms:
                q = t.transform.rotation; tr = t.transform.translation
                R = quat_to_rotmat(q.x, q.y, q.z, q.w)
                tf_chain[(t.header.frame_id, t.child_frame_id)] = T4(R, [tr.x, tr.y, tr.z])
        if len(tf_chain) > 5: break
        time.sleep(0.5)
    cap.stop()
    print(f"TF 边数: {len(tf_chain)}")

    # 1. 用户给的 HEAD→color_optical (对照)
    if ('HEAD', 'realsense_head_color_optical_frame') in tf_chain:
        T = tf_chain[('HEAD', 'realsense_head_color_optical_frame')]
        print("═══ 1. HEAD → color_optical (TF 实测) ═══")
        print(np.round(T, 6))

    # 2. BFS 找 optical → BASE 路径
    optical = 'realsense_head_color_optical_frame'
    target = 'BASE'
    graph = {}
    for (a,b) in tf_chain:
        graph.setdefault(a,[]).append(b); graph.setdefault(b,[]).append(a)
    def find_path(start, goal):
        from collections import deque
        q = deque([(start, [])]); seen={start}
        while q:
            node, path = q.popleft()
            for nxt in graph.get(node, []):
                if nxt in seen: continue
                np_ = path + [(node, nxt)]
                if nxt == goal: return np_
                seen.add(nxt); q.append((nxt, np_))
        return None
    path = find_path(optical, target)
    print(f"\n═══ 2. 路径 {optical} → {target} ═══")
    print("  " + " → ".join([a for a,b in path]) + f" → {target}" if path else "  无路径!")

    # 3. 组合变换: p_optical → p_base
    if path:
        T_accum = np.eye(4)  # 累积: 每步 p_child = T_edge @ p_parent
        for a, b in path:
            if (a,b) in tf_chain:
                Te = tf_chain[(a,b)]        # p_b = Te @ p_a
            elif (b,a) in tf_chain:
                Te = np.linalg.inv(tf_chain[(b,a)])  # p_a = Te' @ p_b → 逆
            else:
                print(f"  缺边 {a}→{b}"); break
            T_accum = Te @ T_accum
        print(f"\n═══ 3. optical → BASE 变换矩阵 ═══")
        print(np.round(T_accum, 6))
        print(f"平移部分 = [{T_accum[0,3]:.3f}, {T_accum[1,3]:.3f}, {T_accum[2,3]:.3f}] m")

        # 4. 示例点: 相机前方 1.5m 的人 (光学系)
        print(f"\n═══ 4. 示例点转换 (人@相机前方1.5m) ═══")
        for name, p_opt in [("光学系原点前方1.5m", np.array([0,0,1.5,1.0])),
                            ("图像中心(相机光轴)", np.array([0,0,1.0,1.0]))]:
            p_base = T_accum @ p_opt
            print(f"  {name}:")
            print(f"    optical = [{p_opt[0]:.2f},{p_opt[1]:.2f},{p_opt[2]:.2f}]m")
            print(f"    BASE    = [{p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f}]m")

    print(f"\n═══ 5. 完整链路 (HSMR→真实→base) ═══")
    print("  HSMR虚拟(假深) → 2D像素 → 真实深度(aligned_depth) →")
    print("  反投影(真实K=910.68) → color_optical(米) →")
    print("  TF → HEAD → NECK → WAIST_PITCH → WAIST_YAW → KNEE → ANKLE → BASE")

if __name__ == '__main__':
    main()
