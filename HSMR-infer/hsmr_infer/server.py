"""单端口 FastAPI 推理服务.

POST /infer    multipart: image (jpg/png/bmp) [+ depth (16UC1 PNG) + fx/fy/cx/cy]
               → {num_persons, timestamp, depth_fused, persons[]}
GET  /health   服务状态 (模型是否加载)
GET  /joints   24 关节名

图片进 → 24 关节数据出；同时把同一份 multipart 请求异步转发给独立渲染服务。
渲染服务失败不会影响本接口的推理结果。
"""
import concurrent.futures
import os
import threading
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import hsmr_infer  # noqa: F401  (把 ROOT 加进 sys.path)
from hsmr_infer.inference import (decode_depth, json_safe, load_default_cfg,
                                  HSMRModelManager)

# 24 SKEL 关节名 (与 lib/ 的 SKEL_JOINTS 一致)
SKEL_JOINTS = [
    "pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
    "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l",
    "lumbar_body", "thorax", "head",
    "scapula_r", "humerus_r", "ulna_r", "radius_r", "hand_r",
    "scapula_l", "humerus_l", "ulna_l", "radius_l", "hand_l",
]

# 模型单例常驻在 HSMRModelManager 里; 这里只存服务级状态.
STATE = {
    "runtime": None,   # HSMRModelManager 单例 (全进程只加载一次)
    "ready": False,
    "infer_lock": threading.Lock(),
}

# 渲染转发使用独立单线程，避免阻塞 /infer 响应线程，也避免并发请求把 8011 压满。
RENDER_FORWARD_URL = os.environ.get(
    "HSMR_RENDER_URL", "http://127.0.0.1:8011/infer"
)
RENDER_FORWARD_TIMEOUT = float(os.environ.get("HSMR_RENDER_TIMEOUT", "120"))
_RENDER_FORWARD_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="hsmr-render-forward"
)


def _multipart_body(image_meta, image_bytes, depth_meta, depth_bytes, fields):
    """用标准库构造转发给 render 的 multipart/form-data 请求体。"""
    boundary = f"----hsmr-{uuid.uuid4().hex}"
    body = bytearray()

    def add_bytes(value):
        body.extend(value)
        body.extend(b"\r\n")

    for name, value in fields.items():
        if value is None:
            continue
        add_bytes(f"--{boundary}".encode("ascii"))
        add_bytes(f'Content-Disposition: form-data; name="{name}"'.encode("utf-8"))
        add_bytes(b"")
        add_bytes(str(value).encode("utf-8"))

    for field_name, meta, payload in (
        ("image", image_meta, image_bytes),
        ("depth", depth_meta, depth_bytes),
    ):
        if not payload:
            continue
        filename, content_type = meta
        add_bytes(f"--{boundary}".encode("ascii"))
        add_bytes(
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="{filename}"'.encode("utf-8")
        )
        add_bytes(f"Content-Type: {content_type}".encode("ascii"))
        add_bytes(b"")
        add_bytes(payload)

    body.extend(f"--{boundary}--\r\n".encode("ascii"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _forward_to_render(image_meta, image_bytes, depth_meta, depth_bytes, fields):
    """后台转发同一输入到 8011；异常只记日志，不影响 8010。"""
    try:
        body, content_type = _multipart_body(
            image_meta, image_bytes, depth_meta, depth_bytes, fields
        )
        request = urllib.request.Request(
            RENDER_FORWARD_URL,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=RENDER_FORWARD_TIMEOUT) as response:
            response.read()
            print(f"[render-forward] {RENDER_FORWARD_URL} -> {response.status}")
    except Exception as exc:
        print(f"[render-forward] 转发失败（不影响主推理）: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动即加载: get_instance 在 yield 前执行, uvicorn 只有等这里完成后
    # 才开始接受请求, 因此模型一定在启动阶段就常驻内存.
    t0 = time.time()
    STATE["runtime"] = HSMRModelManager.get_instance()
    print(f"[启动] 模型已加载 ({time.time() - t0:.1f}s), 服务就绪")
    STATE["ready"] = True
    yield


app = FastAPI(title="HSMR Infer",
              description="HSMR 推理服务: 图片进 → 24 SKEL 关节数据出",
              version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    rt = STATE["runtime"]
    return {"status": "ok" if STATE["ready"] else "warming_up",
            "model_loaded": STATE["ready"],
            "device": rt.device if rt else None}


@app.get("/joints")
def joints():
    return {"joints": SKEL_JOINTS, "count": len(SKEL_JOINTS)}


@app.post("/infer")
def infer(
    image: UploadFile = File(..., description="彩色图 (jpg/png/bmp)"),
    depth: UploadFile = File(None, description="对齐深度图, 16UC1 PNG 或 uint16 .npy (毫米)"),
    fx: float = Form(None, description="相机内参 fx (给 depth 时必填)"),
    fy: float = Form(None, description="相机内参 fy (给 depth 时必填)"),
    cx: float = Form(None, description="相机内参 cx (给 depth 时必填)"),
    cy: float = Form(None, description="相机内参 cy (给 depth 时必填)"),
    max_instances: int = Form(None, description="每帧最多人数 (默认用 config)"),
):
    """图片进 → 24 关节数据出. 带 depth+K 时做深度融合(光学系 3D 米坐标)."""
    rt = STATE["runtime"]
    if rt is None:
        # 启动预热未完成的兜底: 懒加载单例 (仍只加载一次)
        rt = HSMRModelManager.get_instance()
        STATE["runtime"] = rt
        STATE["ready"] = True

    # 解码彩色图
    image_meta = (
        image.filename or "image.jpg",
        image.content_type or "application/octet-stream",
    )
    img_bytes = image.file.read()
    frame_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="无法解码 image (支持 jpg/png/bmp)")

    # 解码深度 (可选) + K
    depth_arr, K = None, None
    depth_bytes = None
    depth_meta = ("depth.png", "application/octet-stream")
    if depth is not None and depth.filename:
        depth_bytes = depth.file.read()
        depth_meta = (
            depth.filename,
            depth.content_type or "application/octet-stream",
        )
        try:
            depth_arr = decode_depth(depth_bytes, depth.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if None in (fx, fy, cx, cy):
            raise HTTPException(status_code=400,
                                detail="提供 depth 时必须同时给 fx/fy/cx/cy")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        if depth_arr.shape[:2] != frame_bgr.shape[:2]:
            raise HTTPException(status_code=400,
                                detail=f"depth 尺寸 {depth_arr.shape[:2]} 与彩色图 "
                                       f"{frame_bgr.shape[:2]} 不一致")

    m = int(max_instances or rt.cfg.get("max_instances", 5))

    # 先提交后台转发，再做本地推理，使 8010 与 8011 尽可能并行处理同一帧。
    # future 不等待：render 超时/失败不会改变 /infer 的返回状态和响应体。
    _RENDER_FORWARD_EXECUTOR.submit(
        _forward_to_render,
        image_meta,
        img_bytes,
        depth_meta,
        depth_bytes,
        {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "max_instances": m},
    )

    with STATE["infer_lock"]:
        t0 = time.time()
        res = rt.infer(frame_bgr, depth_arr, K, m)
        infer_ms = round((time.time() - t0) * 1000, 1)

    return {
        "num_persons": len(res["persons"]),
        "timestamp": res["timestamp"],
        "depth_fused": bool(res.get("depth_fused")),
        "infer_ms": infer_ms,
        # NaN→null: 无效深度关节以 null 输出 (Starlette allow_nan=False 会拒 NaN)
        "persons": [json_safe(p) for p in res["persons"]],
    }


if __name__ == "__main__":
    cfg = load_default_cfg()
    uvicorn.run(app, host=cfg["server"]["host"], port=cfg["server"]["port"])
