"""Frame-to-frame camera motion as an image homography (broadcast cameras pan/tilt/zoom about a
fixed centre, so the whole scene — pitch and stands — moves by one homography).

Players are masked out; LK optical flow on corners + RANSAC.
"""

from __future__ import annotations

import cv2
import numpy as np
import polars as pl
from tqdm import tqdm


def _plausible(h: np.ndarray, shape: tuple[int, int], max_shift: float = 0.12) -> bool:
    hh, ww = shape
    c = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]])[None]
    d = np.linalg.norm(cv2.perspectiveTransform(c, h)[0] - c[0], axis=1)
    return bool(d.max() < max_shift * ww)


def player_mask(shape: tuple[int, int], boxes) -> np.ndarray:
    """255 = background (stands, grass, lines), 0 = a player's box grown by 4 px."""
    mask = np.full(shape, 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (int(x1) - 4, int(y1) - 4), (int(x2) + 4, int(y2) + 4), 0, -1)
    return mask


def flow(prev: np.ndarray, gray: np.ndarray, boxes, max_corners: int = 500) -> dict:
    """One frame step: corners on the background of prev (players masked), Lucas–Kanade to
    gray, one homography by RANSAC. Returns mask, p0 (corners), p1 (where LK found them), st
    (LK succeeded), inl (RANSAC inliers among st), h (prev → gray, working pixels), ok."""
    mask = player_mask(gray.shape, boxes)
    out = {"mask": mask, "p0": np.zeros((0, 2), np.float32), "p1": np.zeros((0, 2), np.float32),
           "st": np.zeros(0, bool), "inl": np.zeros(0, bool), "h": None, "ok": False}
    p0 = cv2.goodFeaturesToTrack(prev, max_corners, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 12:
        return out
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
    st = st.ravel() == 1
    out.update(p0=p0[:, 0], p1=p1[:, 0], st=st, inl=np.zeros(len(st), bool))
    if st.sum() < 12:
        return out
    h, inl = cv2.findHomography(p0[st], p1[st], cv2.RANSAC, 1.5)
    if inl is not None:
        out["inl"][np.flatnonzero(st)] = inl.ravel() == 1
    out["h"] = h
    # reject weak fits: few inliers, or a jump no broadcast camera makes
    out["ok"] = bool(h is not None and inl.sum() >= max(20, 0.5 * st.sum())
                     and _plausible(h, gray.shape))
    return out


def estimate(video: str, people: pl.DataFrame, scale: float = 0.5,
             max_corners: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Return R (F, 3, 3) with R[t] mapping image t-1 → image t (full-res pixels), and ok (F,)."""
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    S = np.diag([scale, scale, 1.0])
    S_inv = np.linalg.inv(S)
    boxes = {f: g.select("x1", "y1", "x2", "y2").to_numpy() * scale
             for (f,), g in people.group_by("frame")}
    R = np.repeat(np.eye(3)[None], n, 0)
    ok = np.zeros(n, bool)
    prev = None
    for f in tqdm(range(n), desc="camera motion"):
        good, frame = cap.read()
        if not good:
            R, ok = R[:f], ok[:f]
            break
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            fl = flow(prev, gray, boxes.get(f - 1, ()), max_corners)
            if fl["ok"]:
                R[f] = S_inv @ fl["h"] @ S
                ok[f] = True
        prev = gray
    cap.release()
    return R, ok
