"""Full pinhole cameras from the per-frame pitch homographies.

Broadcast cameras stand on a fixed tripod and only pan / tilt / zoom, so one camera centre is
estimated per shot and every frame solves only rotation + focal length (square pixels,
principal point at image centre). That is far better conditioned than decomposing each
homography on its own.

World frame: pitch x, y in metres, z = height above the grass (up).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from toonball.calib.homography import _apply, _grid

FLIP = np.diag([1.0, 1.0, -1.0])  # pitch (x, y) with z = x × y points down; we want z up


@dataclass
class Cameras:
    P: np.ndarray       # (F, 3, 4) world (x, y, z_up) → pixels; NaN where uncalibrated
    f: np.ndarray       # (F,) focal length, px
    centre: np.ndarray  # (3,) camera position, metres


def _focal(G: np.ndarray) -> float:
    g1, g2 = G[:, 0], G[:, 1]
    est = []
    if abs(g1[2] * g2[2]) > 1e-12:
        est.append(-(g1[0] * g2[0] + g1[1] * g2[1]) / (g1[2] * g2[2]))
    if abs(g1[2] ** 2 - g2[2] ** 2) > 1e-12:
        est.append(-(g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2) / (g1[2] ** 2 - g2[2] ** 2))
    est = [e for e in est if e > 0]
    return float(np.sqrt(np.median(est))) if est else np.nan


def _K(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def _P(K: np.ndarray, Rm: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Camera in OpenCV's (z-down) pitch frame → projection of (x, y, z_up) points."""
    Cd = FLIP @ C
    return K @ np.concatenate([Rm, -(Rm @ Cd)[:, None]], 1) @ np.diag([1, 1, -1, 1.0])


def estimate(H: np.ndarray, size: tuple[int, int], fps: float) -> Cameras:
    w, h = size
    F = len(H)
    grid = _grid(w, h)
    ok = np.flatnonzero(np.isfinite(H[:, 0, 0]))
    P = np.full((F, 3, 4), np.nan)
    f_out = np.full(F, np.nan)
    if len(ok) < 3:
        return Cameras(P, f_out, np.full(3, np.nan))
    pitch_pts = _apply(H[ok], grid)                              # (n, 12, 2)
    obj = np.concatenate([pitch_pts, np.zeros((*pitch_pts.shape[:2], 1))], -1)
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])
    f0 = np.array([_focal(T @ np.linalg.inv(H[t])) for t in ok])
    f0 = np.where(np.isfinite(f0) & (f0 > 0.3 * w) & (f0 < 20 * w), f0, np.nan)
    if np.isfinite(f0).sum() == 0:
        return Cameras(P, f_out, np.full(3, np.nan))
    f0 = np.where(np.isfinite(f0), f0, np.nanmedian(f0))

    # 1. per-frame planar PnP → camera centres; the tripod position is their robust median
    rots, centres = [], []
    for i, t in enumerate(ok):
        good, rvec, tvec = cv2.solvePnP(obj[i], grid, _K(f0[i], w, h), None,
                                        flags=cv2.SOLVEPNP_IPPE)
        Rm = cv2.Rodrigues(rvec)[0] if good else np.eye(3)
        Cd = -Rm.T @ tvec.ravel() if good else np.full(3, np.nan)
        rots.append(Rm)
        centres.append(FLIP @ Cd)
    centres = np.array(centres)
    plausible = (centres[:, 2] > 4) & (centres[:, 2] < 80)
    if plausible.sum() == 0:
        return Cameras(P, f_out, np.full(3, np.nan))
    C = np.median(centres[plausible], axis=0)

    # 2. per frame, with the centre fixed: solve rotation + focal (4 dof) against the grid
    def residual(p, o):
        Rm = cv2.Rodrigues(p[:3])[0]
        Pm = _P(_K(np.exp(p[3]), w, h), Rm, C)
        q = (Pm @ np.concatenate([o, np.ones((len(o), 1))], 1).T).T
        return (q[:, :2] / q[:, 2:3] - grid).ravel()

    params = np.full((len(ok), 4), np.nan)
    p = None
    for i in range(len(ok)):
        o = obj[i] * np.array([1, 1, 0.0])  # grid points lie on the grass, z = 0
        init = p if p is not None else np.r_[cv2.Rodrigues(rots[i])[0].ravel(), np.log(f0[i])]
        sol = least_squares(residual, init, args=(o,), loss="soft_l1", f_scale=5.0)
        p = sol.x
        params[i] = p

    # 3. joint refinement of the tripod position with every frame's rotation + focal
    n, m = len(ok), len(grid)
    sub = np.arange(0, n, max(1, n // 150))  # ≤150 frames keep the solve small

    def joint(z):
        Cz = z[:3]
        out = []
        for j, i in enumerate(sub):
            q = z[3 + 4 * j: 7 + 4 * j]
            Rm = cv2.Rodrigues(q[:3])[0]
            Pm = _P(_K(np.exp(q[3]), w, h), Rm, Cz)
            o = obj[i] * np.array([1, 1, 0.0])
            qq = (Pm @ np.concatenate([o, np.ones((m, 1))], 1).T).T
            out.append((qq[:, :2] / qq[:, 2:3] - grid).ravel())
        return np.concatenate(out)

    from scipy.sparse import lil_matrix
    J = lil_matrix((2 * m * len(sub), 3 + 4 * len(sub)), dtype=int)
    for j in range(len(sub)):
        J[2 * m * j: 2 * m * (j + 1), :3] = 1
        J[2 * m * j: 2 * m * (j + 1), 3 + 4 * j: 7 + 4 * j] = 1
    sol = least_squares(joint, np.r_[C, params[sub].ravel()], jac_sparsity=J,
                        loss="soft_l1", f_scale=5.0, max_nfev=50)
    if 4 < sol.x[2] < 80:
        C = sol.x[:3]
        p = None
        for i in range(n):  # re-solve every frame against the refined centre
            o = obj[i] * np.array([1, 1, 0.0])
            sol_i = least_squares(residual, params[i] if p is None else p, args=(o,),
                                  loss="soft_l1", f_scale=5.0)
            p = params[i] = sol_i.x
    # zoom changes slowly: smooth log focal, then re-solve rotation only
    win = int(fps) | 1
    lf = savgol_filter(params[:, 3], win, 2) if len(ok) > win else params[:, 3]
    for i, t in enumerate(ok):
        o = obj[i] * np.array([1, 1, 0.0])
        sol = least_squares(lambda r, o=o, lf=lf[i]: residual(np.r_[r, lf], o), params[i, :3],
                            loss="soft_l1", f_scale=5.0)
        P[t] = _P(_K(np.exp(lf[i]), w, h), cv2.Rodrigues(sol.x)[0], C)
        f_out[t] = np.exp(lf[i])
    return Cameras(P, f_out, C)


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(3, 4) × (n, 3) world points → (n, 2) pixels."""
    q = (P @ np.concatenate([X, np.ones((len(X), 1))], 1).T).T
    return q[:, :2] / q[:, 2:3]


def grid_error(cams: Cameras, H: np.ndarray, size: tuple[int, int], step: int = 5) -> float:
    """Median over frames of the worst grid-point disagreement (px) between P and H."""
    grid = _grid(*size)
    errs = []
    for t in range(0, len(H), step):
        if not (np.isfinite(H[t, 0, 0]) and np.isfinite(cams.P[t, 0, 0])):
            continue
        pp = _apply(H[t][None], grid)[0]
        errs.append(np.abs(project(cams.P[t], np.c_[pp, np.zeros(len(pp))]) - grid).max())
    return float(np.median(errs)) if errs else np.inf
