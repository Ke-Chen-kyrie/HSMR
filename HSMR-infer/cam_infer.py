#!/usr/bin/env python3
"""抓当前摄像头一帧 → 调 infer(8010) → 打印按名字的关节.

用法: python3 cam_infer.py [video_dev] [width] [height]
默认自动扫 /dev/video0..11 找第一个能出帧的.
"""
import sys, os, json, time, urllib.request, cv2

def grab(vdev, w=1280, h=720):
    cap = cv2.VideoCapture(vdev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, None
    if w and h:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(5):  # 跳过首几帧黑帧/暖机
        ok, frame = cap.read()
        if ok:
            break
    else:
        cap.release(); return None, None
    cap.release()
    return frame, cap.get(cv2.CAP_PROP_FPS)

def infer(img_bytes):
    boundary = "----hsmrcam"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="image"; filename="cam.jpg"\r\n',
        b"Content-Type: image/jpeg\r\n\r\n",
        img_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request("http://127.0.0.1:8010/infer", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req, timeout=120))

JOINTS = ["pelvis","femur_r","tibia_r","talus_r","calcn_r","toes_r",
          "femur_l","tibia_l","talus_l","calcn_l","toes_l",
          "lumbar_body","thorax","head",
          "scapula_r","humerus_r","ulna_r","radius_r","hand_r",
          "scapula_l","humerus_l","ulna_l","radius_l","hand_l"]

def main():
    W, H = 1280, 720
    frame, fps = None, None
    vdevs = [int(sys.argv[1])] if len(sys.argv) > 1 else range(12)
    for v in vdevs:
        f, fp = grab(v, W, H)
        if f is not None:
            print(f"[cam] 使用 /dev/video{v}  fps~{fp:.0f}  {f.shape[1]}x{f.shape[0]}")
            frame = f
            vdev = v
            break
    if frame is None:
        sys.exit("[cam] 所有 /dev/video* 都打不开, 检查相机")
    os.makedirs("/tmp", exist_ok=True)
    cv2.imwrite("/tmp/cam_frame.jpg", frame)

    t = time.time()
    d = infer(open("/tmp/cam_frame.jpg","rb").read())
    el = time.time() - t
    print(f"[infer] 人数={d.get('num_persons')} | infer_ms={d.get('infer_ms'):.0f} | 端到端{el*1000:.0f}ms")
    os.makedirs("/home/naviai/projects/HSMR/HSMR-infer/data_outputs", exist_ok=True)
    cv2.imwrite("/home/naviai/projects/HSMR/HSMR-infer/data_outputs/cam_frame.jpg", frame)
    print(f"[存图] /home/naviai/projects/HSMR/HSMR-infer/data_outputs/cam_frame.jpg")
    for i, p in enumerate(d["persons"]):
        print(f"-- person[{i}] score={p['score']:.2f} pelvis2D={p['joints_2d'][0]} pelvism={p['joints_optical_m'][0]}")

if __name__ == "__main__":
    main()