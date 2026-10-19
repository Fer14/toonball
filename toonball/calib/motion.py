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
            mask = np.full(gray.shape, 255, np.uint8)
            for x1, y1, x2, y2 in boxes.get(f - 1, ()):
                cv2.rectangle(mask, (int(x1) - 4, int(y1) - 4), (int(x2) + 4, int(y2) + 4), 0, -1)
            p0 = cv2.goodFeaturesToTrack(prev, max_corners, 0.01, 8, mask=mask)
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=(21, 21),
                                                     maxLevel=3)
                st = st.ravel() == 1
                if st.sum() >= 12:
                    h, inl = cv2.findHomography(p0[st], p1[st], cv2.RANSAC, 1.5)
                    # reject weak fits: few inliers, or a jump no broadcast camera makes
                    if (h is not None and inl.sum() >= max(20, 0.5 * st.sum())
                            and _plausible(h, gray.shape)):
                        R[f] = S_inv @ h @ S
                        ok[f] = True
        prev = gray
    cap.release()
    return R, ok
