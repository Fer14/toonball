"""Ball trajectory with game logic instead of raw detections.

- detections projected to the pitch; spikes rejected against a local median
- possession: nearest player within 1.5 m, with hysteresis
- while someone has the ball it sits at their feet (the ball is often hidden by legs there)
- between possessions it flies: a 3D parabola under gravity is fitted to the 2D detections
  through the full camera model; accepted only when it explains the pixels clearly better than
  a ball rolling on the grass
- short gaps are bridged, so the ball doesn't flicker in and out
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from toonball.calib import camera as cam
from toonball.calib.homography import project
from toonball.track.players import Track, _on_pitch

G = 9.81
R_BALL = 0.11


@dataclass
class Ball:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    owner: np.ndarray                  # track id holding the ball, -1 none
    kicks: list[tuple[int, int]]       # (release frame, track id)
    flights: list[dict]                # accepted 3D flights (for inspection)
    hands: np.ndarray                  # (F,) bool: held in a keeper's hands


def _detections(ball: pl.DataFrame, H: np.ndarray, fps: float):
    frames = ball["frame"].to_numpy()
    uv = np.stack([(ball["x1"] + ball["x2"]).to_numpy() / 2,
                   (ball["y1"] + ball["y2"]).to_numpy() / 2], 1)
    xy = project(H, frames, uv)
    ok = np.isfinite(xy[:, 0]) & _on_pitch(np.nan_to_num(xy, nan=-99), 3.0)
    frames, uv, xy = frames[ok], uv[ok], xy[ok]
    if len(frames) < 3:
        return frames, uv, xy
    # spikes: far from the median of the neighbouring detections
    k = min(len(frames) // 2 * 2 - 1, int(0.4 * fps) | 1)
    med = np.stack([median_filter(xy[:, i], size=k, mode="nearest") for i in (0, 1)], 1)
    keep = np.linalg.norm(xy - med, axis=1) < 6.0
    return frames[keep], uv[keep], xy[keep]


def _positions(tracks: list[Track], F: int) -> np.ndarray:
    pos = np.full((len(tracks), F, 2), np.nan)
    for k, t in enumerate(tracks):
        if t.role == "referee":
            continue
        sl = slice(t.start, t.start + len(t.x))
        seen = t.alpha >= 0.999
        pos[k, sl, 0] = np.where(seen, t.x, np.nan)
        pos[k, sl, 1] = np.where(seen, t.y, np.nan)
    return pos


def _image_near(frames: np.ndarray, uv: np.ndarray, people: pl.DataFrame,
                raw_to_k: dict[int, int], roles: list[str], F: int):
    """Who the ball belongs to from the image: a ball inside / touching a person's box is
    theirs whatever its height (a ball held at the chest projects metres behind the keeper on
    the grass). Returns near (F,) track index / -1 none / -3 no detection, and hands (F,):
    a keeper holding it in the upper part of his body."""
    near = np.full(F, -3)
    hands = np.zeros(F, bool)
    boxes = {f: g.select("track_id", "x1", "y1", "x2", "y2").to_numpy()
             for (f,), g in people.filter(pl.col("frame").is_in(frames.tolist()))
             .group_by("frame")}
    for f, (u, v) in zip(frames, uv, strict=True):
        near[f] = -1
        best = np.inf
        for tid, x1, y1, x2, y2 in boxes.get(f, ()):
            k = raw_to_k.get(int(tid))
            if k is None or roles[k] == "referee":
                continue
            w, h = x2 - x1, y2 - y1
            if x1 - 0.25 * w <= u <= x2 + 0.25 * w and y1 - 0.1 * h <= v <= y2 + 0.15 * h:
                d = abs(u - (x1 + x2) / 2) + abs(v - y2)
                if d < best:
                    best, near[f] = d, k
                    hands[f] = roles[k] == "goalkeeper" and v < y1 + 0.65 * h
    return near, hands


def _possession(pos: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                near_img: np.ndarray | None = None, radius: float = 1.5,
                switch: int = 4) -> np.ndarray:
    F = len(gx)
    d = np.hypot(pos[..., 0] - gx, pos[..., 1] - gy)
    d = np.where(np.isnan(d), np.inf, d)
    near = np.where(d.min(0) < radius, d.argmin(0), -1) if len(pos) else np.full(F, -1)
    if near_img is not None:  # where the ball is detected, the image decides
        near = np.where(near_img >= -1, np.where(near_img >= 0, near_img, near), near)
    near = np.where(np.isfinite(gx), near, -2)  # -2: ball unknown → keep current owner
    owner = np.full(F, -1)
    cur, cand, run = -1, -1, 0
    for f in range(F):
        n = near[f]
        if n == -2:
            owner[f] = cur
            continue
        run = run + 1 if n == cand else 1
        cand = n
        if cand != cur and run >= (switch if cand != -1 else 2 * switch):
            cur = cand
        owner[f] = cur
    # hysteresis delays every change by a few frames: date each possession back to the first /
    # last frame the ball was really with that player, so a pass leaves the foot on time
    f = 0
    while f < F:
        k = owner[f]
        if k < 0:
            f += 1
            continue
        a = f
        while f < F and owner[f] == k:
            f += 1
        b = f - 1
        seen = np.flatnonzero(near[max(0, a - 3 * switch): b + 1] == k) + max(0, a - 3 * switch)
        if len(seen):
            owner[seen[0]:a] = np.where(owner[seen[0]:a] == -1, k, owner[seen[0]:a])
            owner[seen[-1] + 1: b + 1] = -1
    return owner


def _fit_flight(P: np.ndarray, frames: np.ndarray, uv: np.ndarray, xy: np.ndarray, fps: float,
                start_xy: np.ndarray | None, end_xy: np.ndarray | None):
    """Fit X(t) = X0 + V t + (0, 0, -g/2 t²) to pixel detections. Returns params, rms, rms_ground."""
    t = (frames - frames[0]) / fps
    T = t[-1]

    def resid(p, ground=False):
        x0, y0, z0, vx, vy, vz = p if not ground else (*p[:2], R_BALL, *p[2:4], 0.0)
        X = np.stack([x0 + vx * t, y0 + vy * t,
                      (z0 + vz * t - 0.5 * G * t ** 2) if not ground else np.full_like(t, z0)], 1)
        out = np.concatenate([cam.project(P[f], X[i:i + 1])[0] - uv[i]
                              for i, f in enumerate(frames)])
        # the flight starts/ends at the passer's / receiver's feet when we know them
        extra = []
        if start_xy is not None:
            extra.append(5.0 * (X[0, :2] - start_xy))
        if end_xy is not None:
            extra.append(5.0 * (X[-1, :2] - end_xy))
        if not ground:
            extra.append([20.0 * min(0.0, X[:, 2].min() - R_BALL)])  # stays above the grass
        return np.concatenate([out, *[np.ravel(e) for e in extra]])

    v0 = (xy[-1] - xy[0]) / max(T, 1e-3)
    g0 = least_squares(lambda p: resid(p, True), np.r_[xy[0], v0], loss="soft_l1", f_scale=4.0)
    best = None
    for vz0 in (0.5 * G * T, 0.25 * G * T, 2.0):  # lob, driven ball, low ball
        s = least_squares(resid, np.r_[g0.x[:2], R_BALL, g0.x[2:], vz0], loss="soft_l1",
                          f_scale=4.0)
        if best is None or s.cost < best.cost:
            best = s
    n = len(frames)
    rms = float(np.sqrt(np.mean(resid(best.x)[: 2 * n] ** 2)))
    rms_g = float(np.sqrt(np.mean(resid(g0.x, True)[: 2 * n] ** 2)))
    return best.x, rms, rms_g


def _z(p: np.ndarray, t):
    return p[2] + p[5] * t - 0.5 * G * np.asarray(t) ** 2


def _despike(bx: np.ndarray, by: np.ndarray, bz: np.ndarray, fps: float, vmax: float = 40.0,
             pad: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bx, by, bz = bx.copy(), by.copy(), bz.copy()
    for it in range(8):
        step = np.hypot(np.diff(bx), np.diff(by)) * fps
        bad = np.flatnonzero(np.nan_to_num(step) > vmax)
        if len(bad) == 0:
            break
        drop = np.zeros(len(bx), bool)
        for f in bad:
            drop[max(0, f - pad - it + 1): f + pad + it + 1] = True
        good = np.isfinite(bx) & ~drop
        if good.sum() < 2:
            break
        idx = np.arange(len(bx))
        fill = drop & np.isfinite(bx)
        for arr in (bx, by, bz):
            arr[fill] = np.interp(idx[fill], idx[good], arr[good])
    return bx, by, bz


def _kinks(frames: np.ndarray, uv: np.ndarray, k: int = 4, angle: float = 30.0,
           ratio: float = 2.2) -> list[int]:
    """Frames where the image velocity of the ball changes abruptly."""
    if len(frames) < 2 * k + 7:
        return []
    v = np.gradient(savgol_filter(uv, 7, 2, axis=0), axis=0)
    out = []
    for i in range(k, len(frames) - k):
        if frames[i + k] - frames[i - k] != 2 * k:
            continue
        a, b = v[i - k], v[i + k]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if max(na, nb) < 2:
            continue
        ang = np.degrees(np.arccos(np.clip(a @ b / (na * nb + 1e-9), -1, 1)))
        if (ang > angle or max(na, nb) / (min(na, nb) + 0.5) > ratio) and (
                not out or frames[i] - out[-1] > k):
            out.append(int(frames[i]))
    return out


def build_ball(ball_raw: pl.DataFrame, H: np.ndarray, P: np.ndarray | None,
               tracks: list[Track], fps: float, people: pl.DataFrame | None = None,
               max_gap_s: float = 2.0) -> Ball:
    F = len(H)
    frames, uv, xy = _detections(ball_raw, H, fps)
    gx, gy = np.full(F, np.nan), np.full(F, np.nan)
    gx[frames], gy[frames] = xy[:, 0], xy[:, 1]

    # ground track with short gaps bridged (possession decisions use this)
    idx = np.arange(F)
    if len(frames) >= 2:
        ix, iy = np.interp(idx, frames, xy[:, 0]), np.interp(idx, frames, xy[:, 1])
        nxt = np.clip(np.searchsorted(frames, idx), 1, len(frames) - 1)
        gap = frames[nxt] - frames[nxt - 1]
        inside = (idx >= frames[0]) & (idx <= frames[-1])
        bridge = inside & ((gap <= max_gap_s * fps) | np.isin(idx, frames))
        ix[~bridge], iy[~bridge] = np.nan, np.nan
    else:
        ix, iy = gx.copy(), gy.copy()

    pos = _positions(tracks, F)
    near_img, hands = None, np.zeros(F, bool)
    if people is not None and len(frames):
        raw_to_k = {r: k for k, t in enumerate(tracks) for r in t.raw_ids}
        near_img, hands = _image_near(frames, uv, people, raw_to_k, [t.role for t in tracks], F)
    owner_k = _possession(pos, ix, iy, near_img)
    ids = np.array([t.id for t in tracks])
    owner = np.where(owner_k >= 0, ids[np.maximum(owner_k, 0)] if len(ids) else -1, -1)
    # a keeper keeps it in his hands until he releases it (detections drop while it's held):
    # the hold lasts while the same keeper owns it and no detection shows it at his feet
    hold = np.zeros(F, bool)
    for f in range(F):
        if owner_k[f] < 0:
            continue
        at_feet = near_img is not None and near_img[f] == owner_k[f] and not hands[f]
        continuing = f > 0 and hold[f - 1] and owner_k[f] == owner_k[f - 1]
        hold[f] = hands[f] or (continuing and not at_feet)

    bx, by, bz = ix.copy(), iy.copy(), np.zeros(F)

    # with the ball: at the owner's feet, on the side the detections put it (≤ 0.8 m);
    # in a keeper's hands: at his body, chest height (the viewer moves it to the hands)
    off = np.zeros(2)
    for f in range(F):
        k = owner_k[f]
        if k < 0 or not np.isfinite(pos[k, f, 0]):
            continue
        if hold[f]:
            bx[f], by[f], bz[f] = pos[k, f, 0], pos[k, f, 1], 1.0
            continue
        if np.isfinite(gx[f]):
            d = np.array([gx[f], gy[f]]) - pos[k, f]
            n = np.linalg.norm(d)
            off = d if n < 0.8 else d / max(n, 1e-6) * 0.8
        bx[f], by[f] = pos[k, f] + off

    # flights: split the image trajectory at kinks (kicks, bounces, deflections); a piece is
    # airborne when it bends away from a straight image line (a ball rolling straight on the
    # grass stays on one — homographies map lines to lines) and a gravity parabola explains it
    flights, accepted = [], []
    if P is not None and np.isfinite(P[:, 0, 0]).any() and len(frames) > 16:
        cuts = [int(frames[0]), *_kinks(frames, uv), int(frames[-1])]
        for a, b in itertools.pairwise(cuts):
            m = (frames >= a) & (frames <= b)
            if (b - a) < 0.4 * fps or m.sum() < 8 or not np.isfinite(P[frames[m], 0, 0]).all():
                continue
            if np.mean(owner[a:b + 1] == -1) < 0.6:
                continue
            pts = uv[m]
            line_rms = np.linalg.svd(pts - pts.mean(0), compute_uv=False)[1] / np.sqrt(len(pts))
            if line_rms < 1.5:
                continue
            s_xy = np.array([bx[a - 1], by[a - 1]]) if a > 0 and owner[a - 1] != -1 else None
            p, rms, rms_g = _fit_flight(P, frames[m], pts, xy[m], fps, s_xy, None)
            t = (np.arange(a, b + 1) - frames[m][0]) / fps
            z = p[2] + p[5] * t - 0.5 * G * t ** 2
            apex = float(z.max())
            if apex > 0.8 and rms < max(2.0, 0.35 * rms_g):
                accepted.append((a, b, int(frames[m][0]), p, apex, rms, rms_g))
        # extend each flight to launch / landing (z back on the grass) over free frames, so it
        # joins the ground track instead of jumping along the camera ray — never into a
        # neighbouring flight
        for j, (a, b, t0, p, apex, rms, rms_g) in enumerate(accepted):
            lo = accepted[j - 1][1] + 1 if j else 0
            hi = accepted[j + 1][0] - 1 if j + 1 < len(accepted) else F - 1
            while a > lo and owner[a - 1] == -1 and _z(p, (a - 1 - t0) / fps) > R_BALL:
                a -= 1
            while b < hi and owner[b + 1] == -1 and _z(p, (b + 1 - t0) / fps) > R_BALL:
                b += 1
            accepted[j] = (a, b, t0, p, apex, rms, rms_g)
            t = (np.arange(a, b + 1) - t0) / fps
            bx[a:b + 1], by[a:b + 1] = p[0] + p[3] * t, p[1] + p[4] * t
            bz[a:b + 1] = np.clip(_z(p, t) - R_BALL, 0, None)
            owner[a:b + 1] = -1
            flights.append({"start": int(a), "end": int(b), "apex_m": round(apex, 2),
                            "rms_px": round(rms, 1), "rms_ground_px": round(rms_g, 1)})

    # no teleports: steps faster than any kick are re-interpolated from the frames around them
    bx, by, bz = _despike(bx, by, bz, fps)

    # light smoothing of ground motion (flights are already smooth parabolas)
    ok = np.isfinite(bx)
    ground = ok & (bz == 0)
    if ground.sum() > 7:
        run_start = None
        for i in range(F + 1):
            if i < F and ground[i]:
                run_start = i if run_start is None else run_start
            elif run_start is not None:
                if i - run_start >= 7:
                    bx[run_start:i] = savgol_filter(bx[run_start:i], 7, 2)
                    by[run_start:i] = savgol_filter(by[run_start:i], 7, 2)
                run_start = None

    kicks = []
    bs = np.hypot(np.gradient(np.nan_to_num(bx)), np.gradient(np.nan_to_num(by))) * fps
    for f in range(1, F):
        if (owner[f - 1] != -1 and owner[f] != owner[f - 1]
                and np.nanmax(bs[f:f + int(0.3 * fps)], initial=0) > 6.0):
            kicks.append((f - 1, int(owner[f - 1])))  # release frame: the ball leaves the foot
    return Ball(bx, by, bz, owner, kicks, flights, hold)
