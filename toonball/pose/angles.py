"""2D keypoints → joint angles for the avatar rig, per tracked person and frame.

Single-camera approximation: limbs are assumed to swing in the person's sagittal plane. The
image shows that plane foreshortened by s = forward · image-right (both on the ground), so the
forward component of a limb is its horizontal image component divided by s. When a person runs
straight at / away from the camera (|s| small) the swing is unobservable and the weight drops,
letting the viewer fall back to its procedural cycle. Sideways tilt of the torso (keeper dives)
is read the same way through the person's lateral axis.

Exported per frame: gait phase locked to the footsteps seen in the video, leg swing amplitude,
mean knee and elbow bend, torso lean (forward +) and roll (towards the person's left +), and
reliability weights w (sagittal) and wr (roll). Left/right limbs are not exported separately:
on 50–150 px players the pose model can't reliably tell them apart.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

from toonball.calib.homography import project

# COCO-17
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16
KEYS = ("lh", "rh", "lk", "rk", "ls", "rs", "le", "re", "lean", "roll")  # per-frame estimates
MIN_CONF = 0.3


def index(pose: pl.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    """(frame, raw track id) → (17, 3) keypoints."""
    out = {}
    for f, tid, kx, ky, kc in pose.select("frame", "track_id", "kx", "ky", "kc").iter_rows():
        out[(f, tid)] = np.stack([kx, ky, kc], 1)
    return out


def _ang(k: np.ndarray, a: int, b: int, s: float) -> float:
    """Angle of limb a→b from straight down, forward positive (image px → sagittal plane)."""
    if k[a, 2] < MIN_CONF or k[b, 2] < MIN_CONF:
        return np.nan
    dx, dy = k[b, 0] - k[a, 0], k[b, 1] - k[a, 1]
    return float(np.arctan2(dx / s, dy))


def _frame_angles(k: np.ndarray, s: float, lr: float) -> dict[str, float]:
    out = {}
    for side, (hip, kn, an, sh, el, wr) in (("l", (L_HIP, L_KN, L_AN, L_SH, L_EL, L_WR)),
                                            ("r", (R_HIP, R_KN, R_AN, R_SH, R_EL, R_WR))):
        thigh, shin = _ang(k, hip, kn, s), _ang(k, kn, an, s)
        upper, fore = _ang(k, sh, el, s), _ang(k, el, wr, s)
        out[f"{side}h"] = thigh
        out[f"{side}k"] = np.clip(thigh - shin, 0, 2.4)
        out[f"{side}s"] = upper
        out[f"{side}e"] = np.clip(fore - upper, 0, 2.6)
    ok = k[[L_SH, R_SH, L_HIP, R_HIP], 2].min() >= MIN_CONF
    if ok:
        dx = (k[L_SH, 0] + k[R_SH, 0] - k[L_HIP, 0] - k[R_HIP, 0]) / 2
        dy = (k[L_SH, 1] + k[R_SH, 1] - k[L_HIP, 1] - k[R_HIP, 1]) / 2
        out["lean"] = float(np.arctan2(dx / s, -dy)) if abs(s) > 0.5 else np.nan
        # image-right component of the person's left axis is lr: tilting left moves the
        # shoulders towards +lr in the image
        out["roll"] = float(np.arctan2(dx / lr, -dy)) if abs(lr) > 0.5 else np.nan
    else:
        out["lean"] = out["roll"] = np.nan
    return out


def _smooth(a: np.ndarray, max_gap: int = 6, win: int = 7) -> np.ndarray:
    ok = np.isfinite(a)
    if ok.sum() < 3:
        return np.full_like(a, np.nan)
    idx = np.arange(len(a))
    v = np.flatnonzero(ok)
    filled = np.interp(idx, v, a[v])
    nxt = np.clip(np.searchsorted(v, idx), 1, len(v) - 1)
    gap = v[nxt] - v[nxt - 1]
    keep = ok | ((idx > v[0]) & (idx < v[-1]) & (gap <= max_gap))
    if len(a) >= win:
        filled = savgol_filter(filled, win, 2)
    return np.where(keep, filled, np.nan)


def track_angles(track, kps: dict, H: np.ndarray, fps: float, ball_x: np.ndarray,
                 ball_y: np.ndarray) -> dict[str, np.ndarray] | None:
    n = len(track.x)
    A = {k: np.full(n, np.nan) for k in KEYS}
    w = np.zeros(n)
    wr = np.zeros(n)
    h = 0.0
    for i in range(n):
        f = track.start + i
        # facing: running direction, or towards the ball when standing (as the viewer does)
        a, b = max(0, i - 4), min(n - 1, i + 4)
        dx, dy = track.x[b] - track.x[a], track.y[b] - track.y[a]
        if track.speed[i] > 1.0:
            h = np.arctan2(dx, dy)
        elif np.isfinite(ball_x[f]):
            h = np.arctan2(ball_x[f] - track.x[i], ball_y[f] - track.y[i])
        if track.alpha[i] < 0.999 or not np.isfinite(H[f, 0, 0]):
            continue
        cands = [kps[(f, r)] for r in track.raw_ids if (f, r) in kps]
        if not cands:
            continue
        k = max(cands, key=lambda c: c[:, 2].mean())
        foot = np.array([[k[[L_AN, R_AN], 0].mean(), k[[L_AN, R_AN], 1].max()]])
        p = project(H, np.array([f, f]), np.concatenate([foot, foot + [10, 0]]))
        r = p[1] - p[0]
        if not np.isfinite(r).all() or np.linalg.norm(r) < 1e-6:
            continue
        r /= np.linalg.norm(r)
        fwd = np.array([np.sin(h), np.cos(h)])
        left = np.array([np.cos(h), -np.sin(h)])
        s, lr = float(fwd @ r), float(left @ r)
        conf = float(k[[L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN], 2].mean())
        w[i] = np.clip((abs(s) - 0.3) / 0.3, 0, 1) * conf
        wr[i] = np.clip((abs(lr) - 0.3) / 0.3, 0, 1) * conf
        if abs(s) < 0.3:
            s = 0.3 * (np.sign(s) or 1.0)
        for key, val in _frame_angles(k, s, lr).items():
            A[key][i] = val
    if w.max() == 0:
        return None
    A = {k: _smooth(v) for k, v in A.items()}
    kern = np.ones(5) / 5
    w = np.where(np.isfinite(A["lh"]) & np.isfinite(A["rh"]), np.convolve(w, kern, "same"), 0.0)
    wr = np.where(np.isfinite(A["roll"]), np.convolve(wr, kern, "same"), 0.0)

    # small players: the model can't tell left from right leg, so use what doesn't depend on
    # it — the stride opening |lh - rh|, which opens and closes once per footstep
    opening = np.abs(A["lh"] - A["rh"])
    phase = _gait_phase(opening, w, track.speed, fps)
    out = {
        "phase": phase,                                   # gait cycle, radians
        "amp": np.clip(opening / 2, 0, 0.9),              # leg swing amplitude
        "knee": _mean2(A["lk"], A["rk"]),                 # mean knee bend
        "elbow": _mean2(A["le"], A["re"]),                # mean elbow bend
        "lean": A["lean"], "roll": A["roll"], "w": w, "wr": wr,
    }
    return out


def stride_length(speed: np.ndarray | float) -> np.ndarray | float:
    """Metres per gait cycle (two steps): ~1.8 m walking, ~3.2 m jogging at 5 m/s, ~4.4 m
    sprinting — about 3 footsteps/s at 4–5 m/s, as measured in the Bundesliga clips."""
    return 1.2 + 0.4 * np.asarray(speed)


def _mean2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean of two series, using whichever side was seen."""
    n = np.isfinite(a).astype(float) + np.isfinite(b)
    return np.where(n > 0, (np.nan_to_num(a) + np.nan_to_num(b)) / np.maximum(n, 1), np.nan)


def _gait_phase(opening: np.ndarray, w: np.ndarray, speed: np.ndarray, fps: float,
                gain: float = 0.2, freq_gain: float = 0.02) -> np.ndarray:
    """Phase-locked gait cycle: advances at the stride rate expected for the running speed
    (as the viewer's procedural cycle does) and is pulled towards the footstep timing seen in
    the stride opening wherever pose is reliable. Second-order loop: it also learns the
    runner's own cadence, which decays back to the default when pose is missing.

    Opening ∝ |sin φ| ≈ c − cos 2φ, so the Hilbert phase ψ of the detrended opening satisfies
    ψ ≈ 2φ + π (φ is only defined modulo π: which leg leads is unobservable)."""
    from scipy.signal import hilbert

    n = len(opening)
    psi = np.full(n, np.nan)
    ok = np.isfinite(opening) & (w > 0.25)
    i = 0
    while i < n:  # Hilbert transform per contiguous reliable run
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= 12:
            seg = opening[i:j]
            trend = np.convolve(seg, np.ones(9) / 9, "same")
            psi[i:j] = np.angle(hilbert(seg - trend))
        i = j
    phi = np.zeros(n)
    p, df = 0.0, 0.0
    for t in range(n):
        sp = speed[t]
        p += 2 * np.pi * sp / fps / stride_length(sp) + df
        if np.isfinite(psi[t]) and sp > 1.5:
            err = np.angle(np.exp(1j * (psi[t] - np.pi - 2 * p))) / 2
            p += gain * w[t] * err
            df += freq_gain * w[t] * err
        else:
            df *= 0.97
        phi[t] = p
    return phi
