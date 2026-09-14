#!/usr/bin/env python3
"""关节查询: 传一张图, 返回按关节名的 2D 像素 / 3D 光学坐标 + 推理耗时.

用法:
  python3 joint_query.py <image> [person_index] [joint_list,...]
  默认查全部 24 关节的第 1 个人; 可指定第几人/只查某几个关节.

示例:
  python3 joint_query.py img.jpg                 # 全部 24 关节, 第1人
  python3 joint_query.py img.jpg 0 humerus_r     # 第1人 humerus_r
  python3 joint_query.py img.jpg 1 tail          # 第2人 名字含tail的关节
"""
import sys, json, urllib.request

JOINTS = ["pelvis","femur_r","tibia_r","talus_r","calcn_r","toes_r",
          "femur_l","tibia_l","talus_l","calcn_l","toes_l",
          "lumbar_body","thorax","head",
          "scapula_r","humerus_r","ulna_r","radius_r","hand_r",
          "scapula_l","humerus_l","ulna_l","radius_l","hand_l"]

URL = "http://127.0.0.1:8010/infer"

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    img = sys.argv[1]
    person = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    want = sys.argv[3].split(",") if len(sys.argv) > 3 else None

    # multipart
    boundary = "----hsmrq"
    with open(img, "rb") as f:
        body = b"".join([
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="image"; filename="i.jpg"\r\n',
            b"Content-Type: image/jpeg\r\n\r\n",
            f.read(),
            f"\r\n--{boundary}--\r\n".encode(),
        ])
    req = urllib.request.Request(URL, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    d = json.load(urllib.request.urlopen(req, timeout=120))

    n = d.get("num_persons", 0)
    print(f"人数={n} | 推理耗时={d.get('infer_ms'):.0f}ms | person[{person}]")
    if person >= len(d["persons"]):
        print(f"  该帧只有 {n} 人, 无第 {person+1} 人")
        return
    p = d["persons"][person]
    for i, name in enumerate(JOINTS):
        if want and not any(w in name for w in want):
            continue
        print(f"  {name:12s} 2D={p['joints_2d'][i]}  opt_m={p['joints_optical_m'][i]}")

if __name__ == "__main__":
    main()