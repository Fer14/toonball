"""Pre-pass over an arbitrary video: find camera cuts and keep the wide, calibratable shots.

Cheap on purpose (histograms every frame, pitch keypoints every `stride` frames) so that long
highlight reels / full matches only pay for the expensive detection on usable footage.
"""

from __future__ import annotations

from itertools import pairwise

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from toonball.calib.homography import fit_raw, project


def _hist(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    return cv2.normalize(h, h).flatten(), gray


def _is_cut(prev, cur, diffs: list[float], hist_thresh: float, diff_ratio: float = 4.0) -> bool:
    """Colour histogram jump, or (for cuts between two similar wide views) a pixel difference
    far above the recent frame-to-frame level."""
    if cv2.compareHist(prev[0], cur[0], cv2.HISTCMP_BHATTACHARYYA) > hist_thresh:
        return True
    d = float(np.abs(cur[1] - prev[1]).mean())
    recent = np.median(diffs[-12:]) if len(diffs) >= 4 else None
    diffs.append(d)
    return recent is not None and d > 12.0 and d > diff_ratio * max(recent, 1.5)


def prepass(video: str, pitch_w: str, stride: int = 3, cut_thresh: float = 0.4):
    """Return (cuts, keypoints, n_frames, fps, size); keypoints is (F, 32, 3), zero where skipped."""
    cap = cv2.VideoCapture(video)
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO(pitch_w)
    kps = np.zeros((n, 32, 3), np.float32)
    sampled = np.zeros(n, bool)
    cuts, prev, diffs = [0], None, []
    for f in tqdm(range(n), desc="shots"):
        ok, frame = cap.read()
        if not ok:
            n = f
            break
        h = _hist(frame)
        if prev is not None and _is_cut(prev, h, diffs, cut_thresh):
            cuts.append(f)
            diffs.clear()
        prev = h
        if f % stride == 0 or cuts[-1] == f:
            k = model(frame, imgsz=640, conf=0.3, verbose=False, half=True)[0].keypoints
            if k is not None and len(k.data):
                kps[f] = k.data[0].cpu().numpy()
            sampled[f] = True
    cap.release()
    return cuts + [n], kps[:n], sampled[:n], n, fps, size


def wide_shots(cuts: list[int], kps: np.ndarray, sampled: np.ndarray, fps: float,
               size: tuple[int, int], min_s: float = 3.0, min_ok: float = 0.6,
               view_m: tuple[float, float] = (20.0, 110.0)) -> list[dict]:
    """Shots long enough, mostly calibrated, and showing a broad view of the pitch."""
    w, h = size
    H = fit_raw(kps)
    out = []
    for a, b in pairwise(cuts):
        # trim a few frames at each end: dissolves and wipes around cuts
        a2, b2 = a + int(0.2 * fps), b - int(0.2 * fps)
        if (b2 - a2) < min_s * fps:
            continue
        s = np.flatnonzero(sampled[a2:b2]) + a2
        ok = s[np.isfinite(H[s, 0, 0])]
        if len(s) == 0 or len(ok) / len(s) < min_ok:
            continue
        mid = np.array([[0, 0.55 * h], [w, 0.55 * h]])
        widths = [np.linalg.norm(np.subtract(*project(H[f:f + 1], np.zeros(2, int), mid)))
                  for f in ok]
        view = float(np.median(widths))
        if not view_m[0] <= view <= view_m[1]:
            continue
        out.append({"start": int(a2), "end": int(b2), "calibrated": round(len(ok) / len(s), 2),
                    "view_m": round(view, 1)})
    return out
