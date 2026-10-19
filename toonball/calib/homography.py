"""Per-frame image→pitch homography: keypoint fits fused with camera motion (see `fuse`)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import savgol_filter

from toonball.calib import pitch


def fit_raw(keypoints: np.ndarray, min_conf: float = 0.5, min_pts: int = 4,
            max_err_m: float = 1.5) -> np.ndarray:
    """(F, 32, 3) keypoints → (F, 3, 3) homographies, NaN where rejected."""
    ref = pitch.vertices()
    H = np.full((len(keypoints), 3, 3), np.nan, np.float64)
    for f, kp in enumerate(keypoints):
        m = kp[:, 2] > min_conf
        if m.sum() < min_pts:
            continue
        src, dst = kp[m, :2].astype(np.float64), ref[m].astype(np.float64)
        h, inl = cv2.findHomography(src, dst, cv2.RANSAC, 1.0)
        if h is None or inl.sum() < min_pts:
            continue
        proj = cv2.perspectiveTransform(src[None], h)[0]
        if np.median(np.linalg.norm(proj - dst, axis=1)) > max_err_m:
            continue
        H[f] = h
    return H


def _grid(w: int, h: int) -> np.ndarray:
    xs = np.linspace(0.05 * w, 0.95 * w, 4)
    ys = np.linspace(0.4 * h, 0.98 * h, 3)
    return np.array([(x, y) for y in ys for x in xs], np.float64)


def _apply(Hs: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Batched perspective transform: (n, 3, 3) × (m, 2) → (n, m, 2)."""
    ph = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    q = np.einsum("nij,mj->nmi", Hs, ph)
    return q[..., :2] / q[..., 2:3]


def _filter(Z: np.ndarray, R: np.ndarray, R_ok: np.ndarray, grid: np.ndarray, q: float,
            r: float, gate_m: float, relock: int, x0: np.ndarray | None = None,
            v0: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
    """One pass of a Kalman-style filter on the pitch positions of a fixed image grid.

    Z (F, n, 2): keypoint measurement of the grid per frame (NaN = none).
    R[t]: image t-1 → image t. Prediction carries the previous state through the camera motion.
    """
    F = len(Z)
    X = np.full_like(Z, np.nan)
    V = np.full(F, np.inf)
    x, v, rejected = (None, np.inf, []) if x0 is None else (x0.copy(), v0, [])
    for t in range(F):
        if x is not None:
            if R_ok[t]:
                h, _ = cv2.findHomography(grid, x)
                x = _apply((h @ np.linalg.inv(R[t]))[None], grid)[0]
                v += q
            else:
                v += 4.0
        z = Z[t]
        if np.isfinite(z[0, 0]):
            if x is None:
                x, v = z.copy(), r
            else:
                d = np.median(np.linalg.norm(z - x, axis=1))
                if d < gate_m + 2 * np.sqrt(v):
                    k = v / (v + r)
                    x, v = x + k * (z - x), (1 - k) * v
                    rejected = []
                else:
                    # carry earlier rejected measurements along with the camera, and re-lock
                    # when they keep agreeing with each other (the state had drifted)
                    rejected = [(zz, tt) for zz, tt in rejected if t - tt < 2 * relock]
                    rejected.append((z, t))
                    if len(rejected) >= relock:
                        tail = np.stack([zz for zz, _ in rejected[-relock:]])
                        spread = np.median(np.linalg.norm(tail - tail.mean(0), axis=-1))
                        if spread < gate_m:
                            x, v, rejected = z.copy(), r, []
        X[t], V[t] = (x, v) if x is not None else (np.nan, np.inf)
    return X, V


def fuse(H_kp: np.ndarray, R: np.ndarray, R_ok: np.ndarray, size: tuple[int, int], fps: float,
         q: float = 0.01, r: float = 1.0, gate_m: float = 3.0, relock: int = 15,
         smooth_s: float = 0.5) -> np.ndarray:
    """Per-frame homography from keypoint fits + frame-to-frame camera motion.

    Forward and backward filters over the pitch positions of a fixed image grid: camera motion
    gives smooth frame-to-frame changes, keypoint fits correct the drift, and fits that disagree
    with the motion-predicted state are rejected (mislabelled keypoints). The two passes are
    blended by their variances.
    """
    F = len(H_kp)
    grid = _grid(*size)
    Z = np.full((F, len(grid), 2), np.nan)
    ok = np.isfinite(H_kp[:, 0, 0])
    if ok.any():
        Z[ok] = _apply(H_kp[ok], grid)
    # backward first: R_b[t] maps image t+1 → image t, i.e. inverse of R[t+1]
    R_b = np.linalg.inv(np.concatenate([R[1:], np.eye(3)[None]]))[::-1]
    ok_b = np.concatenate([R_ok[1:], [False]])[::-1]
    Xb, Vb = _filter(Z[::-1], R_b, ok_b, grid, q, r, gate_m, relock)
    Xb, Vb = Xb[::-1], Vb[::-1]
    # forward pass starts from the backward estimate, so a bad first frame can't seed it
    start = (Xb[0], max(Vb[0], r)) if np.isfinite(Vb[0]) else (None, np.inf)
    Xf, Vf = _filter(Z, R, R_ok, grid, q, r, gate_m, relock, *start)
    wf = np.where(np.isfinite(Vf), 1 / np.maximum(Vf, 1e-6), 0.0)
    wb = np.where(np.isfinite(Vb), 1 / np.maximum(Vb, 1e-6), 0.0)
    has = wf + wb > 0
    X = np.full_like(Z, np.nan)
    X[has] = ((np.nan_to_num(Xf) * wf[:, None, None] + np.nan_to_num(Xb) * wb[:, None, None])
              / (wf + wb)[has][:, None, None])[has] if has.any() else X[has]
    # final light smoothing of the grid trajectories removes residual steps (re-locks)
    win = int(smooth_s * fps) | 1
    idx = np.flatnonzero(has)
    if len(idx) > win:
        X[idx] = savgol_filter(X[idx], win, 2, axis=0)
    out = np.full_like(H_kp, np.nan)
    for t in idx:
        out[t], _ = cv2.findHomography(grid, X[t])
    return out


def project(H: np.ndarray, frames: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Project image points xy[i] at frame frames[i]; NaN where H is missing."""
    out = np.full((len(xy), 2), np.nan)
    for f in np.unique(frames):
        h = H[f]
        if np.isnan(h[0, 0]):
            continue
        m = frames == f
        out[m] = cv2.perspectiveTransform(xy[m][None].astype(np.float64), h)[0]
    return out
