"""2D body pose for every tracked person in every frame.

Players are 50–150 px tall in wide shots, too small for a full-frame pose model, so each
tracked box is cropped with some margin, upscaled, and the crops of a frame are run as one
batch through YOLO11-pose. Keypoints are mapped back to full-frame pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
import polars as pl
from tqdm import tqdm
from ultralytics import YOLO

CROP_H = 288     # px, crops are upscaled to this height
N_KPT = 17       # COCO


def _crop(frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, float, int, int] | None:
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    if h < 16:
        return None
    # pad, then make the crop 1:1 so limbs held out sideways fit
    cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(h * 1.25, w * 1.4)
    X1, Y1 = round(cx - side / 2), round(cy - side / 2)
    X2, Y2 = X1 + round(side), Y1 + round(side)
    H, W = frame.shape[:2]
    pad = cv2.copyMakeBorder(frame[max(Y1, 0):min(Y2, H), max(X1, 0):min(X2, W)],
                             max(0, -Y1), max(0, Y2 - H), max(0, -X1), max(0, X2 - W),
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    s = CROP_H / side
    return cv2.resize(pad, (CROP_H, CROP_H), interpolation=cv2.INTER_CUBIC), s, X1, Y1


def run(video: str, people: pl.DataFrame, weights: str, batch: int = 64) -> pl.DataFrame:
    """Return one row per (frame, track_id) with kx/ky/kc arrays (17 each, full-frame px)."""
    model = YOLO(weights)
    by_frame = {f: g for (f,), g in people.group_by("frame")}
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows: list[tuple] = []
    pending: list[tuple] = []  # (frame, track_id, scale, X1, Y1, crop)

    def flush():
        if not pending:
            return
        res = model([p[5] for p in pending], imgsz=CROP_H, conf=0.15, verbose=False, half=True)
        for (f, tid, s, X1, Y1, _), r in zip(pending, res, strict=True):
            if r.keypoints is None or len(r.boxes) == 0:
                continue
            # the person we cropped for is the one closest to the crop centre
            b = r.boxes.xyxy.cpu().numpy()
            c = (b[:, :2] + b[:, 2:]) / 2
            i = int(np.argmin(np.linalg.norm(c - CROP_H / 2, axis=1)
                              - 0.2 * (b[:, 3] - b[:, 1])))
            k = r.keypoints.data[i].cpu().numpy()
            rows.append((f, tid, (k[:, 0] / s + X1).tolist(), (k[:, 1] / s + Y1).tolist(),
                         k[:, 2].tolist()))
        pending.clear()

    for f in tqdm(range(n), desc="pose"):
        ok, frame = cap.read()
        if not ok:
            break
        g = by_frame.get(f)
        if g is None:
            continue
        for tid, x1, y1, x2, y2 in g.select("track_id", "x1", "y1", "x2", "y2").iter_rows():
            c = _crop(frame, np.array([x1, y1, x2, y2]))
            if c is not None:
                pending.append((f, tid, *c[1:], c[0]))
        if len(pending) >= batch:
            flush()
    flush()
    cap.release()
    return pl.DataFrame(rows, schema={"frame": pl.Int32, "track_id": pl.Int32,
                                      "kx": pl.List(pl.Float32), "ky": pl.List(pl.Float32),
                                      "kc": pl.List(pl.Float32)}, orient="row")
