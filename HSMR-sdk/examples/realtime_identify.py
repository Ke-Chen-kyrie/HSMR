#!/usr/bin/env python3
"""realtime_identify.py — 组合示例: 采集 → infer_persons → 声纹+人脸 → 发布.

一条龙: SDK 采集帧 → sdk.infer_persons(frame) 直连 HSMR-infer 容器 (config.infer.url)
取 persons → 坐标变换 + 声纹人脸交叉验证 → 关节 MarkerArray/标注图发布.

用法:
  # 采集 1 帧, 推理 + 说话人识别 + 发布
  python examples/realtime_identify.py one --audio /path/to/speech.wav

  # 采集 1 帧, 跳过声纹人脸 (纯推理 + 坐标变换 + 发布)
  python examples/realtime_identify.py one --no-identify

  # 用已存帧离线跑一次 (不连 bridge, 从已存 persons JSON 读, 走完整变换链)
  python examples/realtime_identify.py replay --frame-rgb rgb.jpg --frame-depth depth_raw.npy --persons persons.json

需要: foxglove_bridge (ws://localhost:8768), HSMR-infer 容器 (127.0.0.1:8010),
user-identification 容器 (127.0.0.1:8001), zenoh router (127.0.0.1:7447, 可选).
"""
import argparse
import json
import sys
import time

from hsmr_sdk import HsmrSdk, load_config


def run_one(cfg, args):
    sdk = HsmrSdk(cfg).start_capture()
    try:
        frame = sdk.capture_once()
        if frame["rgb_bgr"] is None:
            print("[!!] 未采集到彩色帧 — 确认 foxglove_bridge 在", cfg["bridge"]["url"], "且相机在发布")
            return 1
        print(f"[ok] 帧 {frame['rgb_bgr'].shape}, depth "
              f"{'OK' if frame['depth_uint16'] is not None else 'N/A'}, "
              f"K={'OK' if frame['K'] is not None else 'N/A'}, tf={len(frame['tf_msgs'])} 条")

        # ① 推理: SDK 直连 HSMR-infer 容器 /infer, 帧 → persons (容器未起/超时抛 InferError)
        if args.persons:
            persons = json.load(open(args.persons))
        elif args.no_infer:
            persons = []
        else:
            persons = sdk.infer_persons(frame)
        print(f"[ok] persons: {len(persons)}")

        if not args.no_identify:
            audio = open(args.audio, "rb").read() if args.audio else None
            if not audio:
                print("[!!] 需要 --audio <语音文件> (或加 --no-identify 跳过声纹)")
                return 1
            resp = sdk.identify(audio, frame["rgb_bgr"], persons,
                                target=args.target, include_pelvis_pose=True, verbose=True)
            pi = resp["person_index"]
            print(json.dumps({k: resp[k] for k in ("verified", "user_id", "name",
                                                   "voice_score", "face_score", "person_index")}, ensure_ascii=False))
            print(f"  说话人 BASE 系 骨盆: {resp['position_m']} (valid={resp['depth_valid']})")
            print(f"                 head: {resp['head_position_m']} (valid={resp['head_depth_valid']})")
        else:
            pi = args.person if args.person is not None else 0
            if not persons:
                print("[ok] 无 persons, 跳过发布")
                return 0

        # 发布
        if sdk.ros.enabled and sdk.ros.available:
            joints = sdk.person_joints_payload(persons[pi], target=args.target)
            sdk.publish(resp["frame"] if not args.no_identify else args.target, joints,
                        annotated_bgr=frame["rgb_bgr"])
            print(f"[ok] 已发布 {len(joints)} 关节到 {cfg['ros_publish']['topic']} (目标系 {args.target})")
        else:
            print("[ok] ros 发布未启用 (缺 zenoh_ros2_sdk 或 enabled=false) — 跳过")

        time.sleep(1)
        return 0
    finally:
        sdk.stop_capture()


def run_replay(cfg, args):
    """离线: 从已存文件重放一帧 + persons, 走完整变换链路 (不连 bridge)."""
    import cv2
    import numpy as np

    from hsmr_sdk import HsmrSdk
    from hsmr_sdk.tf import joint_pos_ori

    rgb = cv2.imread(args.frame_rgb, cv2.IMREAD_COLOR)
    depth = np.load(args.frame_depth)  # uint16 mm
    persons = json.load(open(args.persons))
    sdk = HsmrSdk(cfg)  # 不 start_capture (纯坐标变换, 不连 bridge)
    print(f"[ok] replay: rgb={rgb.shape}, depth={depth.shape} dtype={depth.dtype} median={int(np.median(depth))}mm, persons={len(persons)}")

    for i, p in enumerate(persons):
        pos, R, dv = joint_pos_ori(p, 0, sdk.tf, args.target)
        head, hR, hdv = joint_pos_ori(p, 13, sdk.tf, args.target)
        print(f"  person{i}: pelvis={pos} (valid={dv}), head={head} (valid={hdv})  <- {args.target} 系")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["one", "replay"])
    ap.add_argument("--audio")
    ap.add_argument("--persons", help="跳过推理, 从 JSON 读 persons")
    ap.add_argument("--no-infer", action="store_true", help="不调 HSMR-infer 容器 (无 persons)")
    ap.add_argument("--no-identify", action="store_true", help="跳过声纹+人脸验证")
    ap.add_argument("--target", default="BASE", help="BASE / world / HEAD / 光学系")
    ap.add_argument("--person", type=int, default=None)
    ap.add_argument("--frame-rgb")
    ap.add_argument("--frame-depth")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode == "one":
        return run_one(cfg, args)
    return run_replay(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
