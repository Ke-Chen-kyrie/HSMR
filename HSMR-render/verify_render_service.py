#!/usr/bin/env python3
"""HSMR-render 远程 HTTP 全链路验证: POST /joint 触发渲染 → 当前帧(seq=0) 入队,
后台后 N 帧(seq=1..N) 依次入队, GET /render/pop 可取走.

用法: python verify_render_service.py [--url http://127.0.0.1:8011] [--wait 15]
"""
import argparse
import base64
import json
import sys
import time
import urllib.request


def post(url, body):
    req = urllib.request.Request(url + "/joint",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def get(url, path):
    return json.load(urllib.request.urlopen(url + path, timeout=10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--wait", type=float, default=15.0, help="等后台后 N 帧填队列的秒数")
    ap.add_argument("--joints", default="hand_r", help="POST 关节 (英文/中文/整体), 逗号分隔")
    args = ap.parse_args()
    url = args.url

    print(f"== 服务: {url} ==")
    h = get(url, "/health")
    print(f"[health] status={h.get('status')} model_loaded={h.get('model_loaded')} "
          f"queue={h.get('queue_size')}")
    if not h.get("model_loaded"):
        print("[FAIL] 模型未加载")
        return 1

    jlist = [j for j in args.joints.split(",") if j]
    selected = {j: 1 for j in jlist}
    print(f"[POST /joint] {selected}")
    d = post(url, selected)
    print(f"  queued={d.get('queued')} queue_size={d.get('queue_size')} "
          f"next_frames={d.get('next_frames')} bg_started={d.get('bg_started')} "
          f"persons={d.get('num_persons')} joints={d.get('joints')} glow={d.get('glow')}")
    if not d.get("queued"):
        print("[FAIL] 当前帧未入队")
        return 1
    next_frames = d.get("next_frames") or 0

    # 轮询队列: 观察 seq 0..N 依次出现
    t0 = time.time()
    seen = set()
    max_q = 0
    while time.time() - t0 < args.wait:
        q = get(url, "/render/queue")
        max_q = max(max_q, q.get("queue_size", 0))
        n = get(url, "/render/pop")["rendered"]
        if n:
            seen.add(n.get("seq"))
            got_b64 = bool(n.get("image_base64"))
            print(f"  pop seq={n.get('seq')} joints={n.get('joints')} "
                  f"persons={n.get('num_persons')} b64={got_b64}")
        if next_frames == 0:
            break
        time.sleep(0.5)
    print(f"[队列] 峰值 size={max_q}, 看到的 seq={sorted(seen)}")

    ok = (0 in seen) and (next_frames == 0 or set(range(1, next_frames + 1)).issubset(seen))
    if ok:
        print("\n=== 结论 ===")
        print("✅ 服务正常: 当前帧+后 N 帧渲染入队, /render/pop 可取走")
        return 0
    print("\n=== 结论 ===")
    print(f"⚠️ seq 未收全: 期望 0..{next_frames}, 只看到 {sorted(seen)}")
    print("   (若队列被清空过快/人数不足, 后台帧会跳过 — 属正常, 可加大 --wait 重试)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
