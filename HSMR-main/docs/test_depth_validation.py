"""
深度验证测试 — 检查真实深度反投影是否正确

方法:
  A. 深度自洽: 画面中心像素反投影 → X,Y 应≈0 (因为中心在主点附近)
  B. 回投自洽: 任取若干像素 → 反投影3D → 再投影回2D → 误差应≈0像素
  C. 合理性:   深度范围 0.1~10m, 非零占比
  D. 可视化:   彩色 + 深度 标注中心点

用法(机器人上):
  .venv_orin/bin/python docs/test_depth_validation.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_depth_3d import MultiTopicCapture, decode_color, decode_depth, camera_info_k, BRIDGE

TOPICS = [
    '/zj_humanoid/sensor/realsense_head/color/image_raw/compressed',
    '/zj_humanoid/sensor/realsense_head/aligned_depth_to_color/image_raw/compressedDepth',
    '/zj_humanoid/sensor/realsense_head/aligned_depth_to_color/camera_info',
]

def deproject(u, v, z_m, K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    return np.array([(u-cx)*z_m/fx, (v-cy)*z_m/fy, z_m])

def project(p, K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    return np.array([p[0]*fx/p[2]+cx, p[1]*fy/p[2]+cy])

def sample_z(depth, u, v, r=2):
    h, w = depth.shape
    u, v = int(round(u)), int(round(v))
    if not (0<=u<w and 0<=v<h): return None
    win = depth[max(0,v-r):min(h,v+r+1), max(0,u-r):min(w,u+r+1)].astype(np.float32)/1000.0
    ok = win[win>0.1]
    return float(np.median(ok)) if len(ok) else None

def main():
    cap = MultiTopicCapture(BRIDGE, TOPICS).start()
    t0 = time.time()
    snap = None
    while time.time()-t0 < 15:
        snap = cap.snapshot()
        if all(t in snap for t in TOPICS): break
        time.sleep(0.5)
    if not snap or not all(t in snap for t in TOPICS):
        print("❌ 捕获失败"); cap.stop(); return
    cap.stop()

    frame = decode_color(snap[TOPICS[0]])
    depth = decode_depth(snap[TOPICS[1]])     # uint16 毫米
    K = camera_info_k(snap[TOPICS[2]])
    H, W = depth.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    print(f"深度图: {W}x{H}  uint16 毫米")
    print(f"相机内参 K: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print("="*55)

    # A. 深度自洽: 中心像素
    print("\n【A. 画面中心深度自洽】(中心应≈主点, X/Y≈0)")
    for (u, v, tag) in [(cx, cy, '中心'), (W//2, H//2, '图像中心'), (int(cx), int(cy), '主点')]:
        z = sample_z(depth, u, v)
        if z is None:
            print(f"  {tag}({u:.0f},{v:.0f}): 深度无效"); continue
        p = deproject(u, v, z, K)
        # 回投
        p2 = project(p, K)
        err = np.linalg.norm(p2 - np.array([u, v]))
        print(f"  {tag}({u:.0f},{v:.0f}): Z={z:.2f}m → 3D=[{p[0]:+.2f},{p[1]:+.2f},{p[2]:.2f}] → 回投误差{err:.2f}px {'✓' if err<0.5 else '✗'}")

    # B. 回投自洽: 随机采样 10 个有效像素
    print("\n【B. 随机像素 3D→2D 回投自洽】(误差应≈0px)")
    valid_mask = depth > 100
    ys, xs = np.nonzero(valid_mask)
    if len(ys) > 0:
        idx = np.random.RandomState(0).choice(len(ys), min(10, len(ys)), replace=False)
        errs = []
        for i in idx:
            u, v = xs[i], ys[i]
            z = depth[v,u]/1000.0
            if not 0.1<z<10: continue
            p = deproject(u, v, z, K)
            p2 = project(p, K)
            e = np.linalg.norm(p2-np.array([u,v]))
            errs.append(e)
        if errs:
            print(f"  采样 {len(errs)} 点, 平均回投误差 = {np.mean(errs):.4f}px  {'✓ 自洽' if np.mean(errs)<0.5 else '✗ 有问题'}")
    else:
        print("  无有效深度")

    # C. 深度合理性
    print("\n【C. 深度合理性】")
    nz = depth[depth>0]/1000.0
    print(f"  非零像素: {len(nz)/depth.size*100:.1f}%")
    if len(nz):
        print(f"  深度范围: {nz.min():.2f} ~ {nz.max():.2f} m, 中位数 {np.median(nz):.2f} m")
        ok = ((nz>0.1)&(nz<10)).mean()
        print(f"  合理范围(0.1~10m)占比: {ok*100:.1f}%  {'✓' if ok>0.8 else '⚠ 检查'}")

    # D. 可视化
    print("\n【D. 保存可视化】")
    # 深度伪彩色
    depth_vis = depth.copy().astype(np.float32)/1000.0
    depth_vis[depth_vis<=0] = np.nan
    d8 = np.zeros_like(depth, dtype=np.uint8)
    valid = depth>0
    d8[valid] = np.clip(depth[valid]/40, 0, 255).astype(np.uint8)  # 40mm→1 简化映射
    dcol = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    # 中心标注
    for (u, v, tag, col) in [(int(cx), int(cy), '主点', (0,255,0)), (W//2, H//2, '中心', (255,0,0))]:
        cv2.circle(frame, (u, v), 8, col, 2)
        cv2.putText(frame, tag, (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.circle(dcol, (u, v), 8, col, 2)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    cv2.imwrite(f'{out_dir}/_depth_test_color.png', frame)
    cv2.imwrite(f'{out_dir}/_depth_test_depth.png', dcol)
    print(f"  {out_dir}/_depth_test_color.png (彩色+中心标注)")
    print(f"  {out_dir}/_depth_test_depth.png (深度伪彩色+中心标注)")

    print("\n" + "="*55)
    print("结论: 看 A/B/C 是否 ✓。")
    print("  已知距离对照: 站相机前 ~1m/2m/3m, 看中心 Z 是否接近")
    print("  旋转矩阵: 跑 realtime_depth_3d.py, 面朝相机 yaw≈0, 背对 yaw≈±180")

if __name__ == '__main__':
    main()
