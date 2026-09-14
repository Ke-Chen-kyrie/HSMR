"""检测框 → 256x256 patch (纯 numpy 重写).

复刻 lib/kits/hsmr_demo.py 的 _img_det2patches 逻辑, 但**不 import hsmr_demo**
(它会拉进 pyrender/OpenGL 渲染链, 容器里易挂). 只依赖 numpy/cv2/torch.
"""
import numpy as np
import cv2

IMG_MEAN_255 = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMG_STD_255 = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


# ── bbox 格式转换 (纯 numpy, 等价 lib/utils/bbox.py) ──
def lurb_to_cwh(lurb):
    lurb = np.asarray(lurb, dtype=float)
    l, u, r, b = lurb[..., 0], lurb[..., 1], lurb[..., 2], lurb[..., 3]
    c = np.stack([(l + r) / 2, (u + b) / 2], axis=-1)
    wh = np.stack([r - l, b - u], axis=-1)
    return np.concatenate([c, wh], axis=-1)


def cwh_to_cs(cwh):
    cwh = np.asarray(cwh, dtype=float)
    c = cwh[..., :2]
    s = np.maximum(cwh[..., 2], cwh[..., 3])
    return np.concatenate([c, s[..., None]], axis=-1)


def cs_to_cwh(cs):
    cs = np.asarray(cs, dtype=float)
    s = cs[..., 2]
    return np.concatenate([cs[..., :2], s[..., None], s[..., None]], axis=-1)


def cwh_to_lurb(cwh):
    cwh = np.asarray(cwh, dtype=float)
    c = cwh[..., :2]
    wh = cwh[..., 2:]
    return np.concatenate([c - wh / 2, c + wh / 2], axis=-1)


def lurb_to_cs(lurb):
    return cwh_to_cs(lurb_to_cwh(lurb))


def cs_to_lurb(cs):
    return cwh_to_lurb(cs_to_cwh(cs))


def expand_wh_to_aspect_ratio(bbx_wh, tgt_ratio):
    if tgt_ratio is None:
        return np.asarray(bbx_wh, dtype=float)
    bbx_w, bbx_h = float(bbx_wh[0]), float(bbx_wh[1])
    tgt_w, tgt_h = tgt_ratio
    if bbx_h / bbx_w < tgt_h / tgt_w:
        new_h = max(bbx_w * tgt_h / tgt_w, bbx_h)
        new_w = bbx_w
    else:
        new_h = bbx_h
        new_w = max(bbx_h * tgt_w / tgt_h, bbx_w)
    return np.array([new_w, new_h])


def fit_bbox_to_aspect_ratio(bbox, tgt_ratio=None, bbox_type="lurb"):
    bbox = np.asarray(bbox, dtype=float).copy()
    if bbox_type == "lurb":
        bbx_cwh = lurb_to_cwh(bbox)
    elif bbox_type == "cwh":
        bbx_cwh = bbox
    else:
        raise ValueError(bbox_type)
    bbx_cwh[2:] = expand_wh_to_aspect_ratio(bbx_cwh[2:], tgt_ratio)
    return cwh_to_lurb(bbx_cwh) if bbox_type == "lurb" else bbx_cwh


def crop_with_lurb(img, lurb, padding=0):
    img = np.asarray(img)
    l, u, r, b = [int(round(x)) for x in np.asarray(lurb, dtype=float)]
    H, W = img.shape[:2]
    H_patch, W_patch = max(b - u, 1), max(r - l, 1)
    out = np.full((H_patch, W_patch) + img.shape[2:], padding, dtype=img.dtype)
    vl, vu = max(0, l), max(0, u)
    vr, vb = min(W, r), min(H, b)
    tl, tu = vl - l, vu - u
    tr, tb = tl + (vr - vl), tu + (vb - vu)
    if tr > tl and tb > tu:
        out[tu:tb, tl:tr] = img[vu:vb, vl:vr]
    return out


def _img_det2patches(img, det_instances, downsample_ratio, max_instances=5):
    """检测结果 → 256x256 patch + center-scale 框. 复刻 hsmr_demo._img_det2patches."""
    if det_instances is None:
        return np.empty((0, 256, 256, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    import torch
    CLASS_HUMAN_ID, DET_THRESHOLD_SCORE = 0, 0.5
    is_human = det_instances["pred_classes"] == CLASS_HUMAN_ID
    reliable = det_instances["scores"] > DET_THRESHOLD_SCORE
    active = is_human & reliable
    if active.sum().item() > max_instances:
        humans_scores = det_instances["scores"] * is_human.float()
        _, top_idx = humans_scores.topk(max_instances)
        valid = torch.zeros_like(active).bool()
        valid[top_idx] = True
    else:
        valid = active
    boxes = det_instances["pred_boxes"][valid].numpy() / downsample_ratio  # (N,4) lurb
    boxes = [fit_bbox_to_aspect_ratio(b, (192, 256)) for b in boxes]
    cs = np.asarray([lurb_to_cs(b) for b in boxes], dtype=np.float32).reshape(-1, 3) if boxes else np.empty((0, 3), dtype=np.float32)
    lurbs = [cs_to_lurb(c) for c in cs]
    cropped = [crop_with_lurb(img, b) for b in lurbs]
    patches = np.stack(
        [cv2.resize(c, (256, 256), interpolation=cv2.INTER_LINEAR) for c in cropped]
    ).astype(np.float32) if cropped else np.empty((0, 256, 256, 3), dtype=np.float32)
    return patches, cs
