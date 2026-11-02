"""`toonball show pitch`: the image→pitch homography drawn on the real clip (ch04).

- the detected pitch keypoints with their number (1–32 of calib/pitch.py, + 1), green when
  RANSAC kept them for the frame's fit, red (with the line to where the fit puts them) when not;
- the pitch model lines projected back onto the video with the homography;
- a panel with keypoints found, RANSAC inliers and the reprojection error in metres;
- a top-down radar: the pitch diagram, the part the camera sees, and the players' feet as dots
  in their team colour.

hom="fused" draws the final homography (homography.npy: keypoint fits fused with camera
motion); hom="raw" the per-frame keypoint fit alone (homography_raw.npy, or fit_raw again).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from toonball.calib import homography, pitch
from toonball.render.debug import ACCENT, DROP, FONT, INK, OUT, PAPER, RED, _text

LINE_C = (40, 225, 255)    # projected pitch lines (yellow, BGR)
IN_C = (90, 200, 60)       # RANSAC inlier
OUT_C = RED                # RANSAC outlier / rejected
RADAR_PX = 5.0             # radar scale, px per metre


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)


def model_lines(step: float = 0.5) -> list[np.ndarray]:
    """The pitch markings as polylines in metres (pitch x along the length, y across)."""
    L, W = pitch.LENGTH, pitch.WIDTH
    pb, gb, R = pitch.PENALTY_BOX_W, pitch.GOAL_BOX_W, pitch.CIRCLE_R

    def seg(a, b):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
        return np.linspace(a, b, n)

    def rect(x0, y0, x1, y1):
        return np.concatenate([seg((x0, y0), (x1, y0)), seg((x1, y0), (x1, y1)),
                               seg((x1, y1), (x0, y1)), seg((x0, y1), (x0, y0))])

    def arc(cx, cy, a0, a1):
        t = np.linspace(a0, a1, max(8, int(R * abs(a1 - a0) / step)))
        return np.stack([cx + R * np.cos(t), cy + R * np.sin(t)], 1)

    out = [rect(0, 0, L, W), seg((L / 2, 0), (L / 2, W)), arc(L / 2, W / 2, 0, 2 * np.pi)]
    for x0, sgn in ((0, 1), (L, -1)):
        x_pb, x_gb = x0 + sgn * pitch.PENALTY_BOX_L, x0 + sgn * pitch.GOAL_BOX_L
        out += [np.concatenate([seg((x0, (W - pb) / 2), (x_pb, (W - pb) / 2)),
                                seg((x_pb, (W - pb) / 2), (x_pb, (W + pb) / 2)),
                                seg((x_pb, (W + pb) / 2), (x0, (W + pb) / 2))]),
                np.concatenate([seg((x0, (W - gb) / 2), (x_gb, (W - gb) / 2)),
                                seg((x_gb, (W - gb) / 2), (x_gb, (W + gb) / 2)),
                                seg((x_gb, (W + gb) / 2), (x0, (W + gb) / 2))])]
        # penalty arc: the part of the circle round the spot outside the box
        a = np.arccos((pitch.PENALTY_BOX_L - pitch.PENALTY_SPOT) / R)
        cx = x0 + sgn * pitch.PENALTY_SPOT
        out.append(arc(cx, W / 2, -a, a) if sgn > 0 else arc(cx, W / 2, np.pi - a, np.pi + a))
    return out


def to_image(H: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pitch metres → image px with the inverse of an image→pitch H; ok = in front of camera."""
    Hi = np.linalg.inv(H)
    q = np.c_[xy, np.ones(len(xy))] @ Hi.T
    w = q[:, 2] * np.sign(np.linalg.det(Hi))
    p = q[:, :2] / q[:, 2:3]
    return p, (w > 1e-9) & (np.abs(p) < 2e4).all(1)


def draw_lines(img, H, s: float, colour=LINE_C, thick: int = 3) -> None:
    for poly in model_lines():
        p, ok = to_image(H, poly)
        p = p * s
        for i in range(1, len(p)):
            if ok[i - 1] and ok[i]:
                cv2.line(img, tuple(np.int32(p[i - 1])), tuple(np.int32(p[i])), colour, thick,
                         cv2.LINE_AA)


def radar_base(px: float = RADAR_PX, margin: int = 14) -> np.ndarray:
    w, h = int(pitch.LENGTH * px) + 2 * margin, int(pitch.WIDTH * px) + 2 * margin
    img = np.full((h, w, 3), (96, 148, 92), np.uint8)
    for poly in model_lines():
        cv2.polylines(img, [np.int32(poly * px + margin)], False, PAPER, 2, cv2.LINE_AA)
    return img


def load(run_dir: Path, hom: str = "fused") -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    kp = np.load(run_dir / "keypoints.npy")
    fits = [homography.fit_frame(k) for k in kp]
    if hom == "raw":
        p = run_dir / "homography_raw.npy"
        H = np.load(p) if p.exists() else homography.fit_raw(kp)
    else:
        H = np.load(run_dir / "homography.npy")
    people = pl.read_parquet(run_dir / "people.parquet")
    team = {}
    tp = run_dir / "tracks.parquet"
    if tp.exists():
        team = {int(r): (role, t) for r, role, t in
                pl.read_parquet(tp).select("raw_id", "role", "team").iter_rows()}
    cp = run_dir / "colours.json"
    colours = [_hex(c) for c in json.loads(cp.read_text())] if cp.exists() else []
    feet: dict[int, list] = {}
    for f, tid, cls, x1, y2, x2 in people.select("frame", "track_id", "cls", "x1", "y2",
                                                   "x2").iter_rows():
        role, t = team.get(int(tid), (cls, None))
        if t is not None and 0 <= t < len(colours) and role in ("player", "goalkeeper"):
            col = colours[t]
        elif role == "referee" or cls == "referee":
            col = INK
        else:
            col = (200, 200, 200)
        feet.setdefault(f, []).append(((x1 + x2) / 2, y2, col))
    return {"n": len(kp), "fps": meta["fps"], "size": tuple(meta["size"]), "kp": kp,
            "fits": fits, "H": H, "hom": hom, "feet": feet, "radar": radar_base()}


def frame_error(H: np.ndarray, kp: np.ndarray, min_conf: float = 0.5) -> float:
    """Median distance (m) on the pitch between the detected keypoints sent through H and the
    model points they should land on."""
    m = kp[:, 2] > min_conf
    if not m.any() or not np.isfinite(H[0, 0]):
        return np.nan
    p = cv2.perspectiveTransform(kp[m, :2][None].astype(np.float64), H)[0]
    return float(np.median(np.linalg.norm(p - pitch.vertices()[m], axis=1)))


def draw_keypoints(img, kp, fit, s: float, H=None, scale: float = 1.0) -> None:
    ref = pitch.vertices()
    for i in np.flatnonzero(fit["used"]):
        p = tuple(np.int32(kp[i, :2] * s))
        col = IN_C if fit["inliers"][i] else OUT_C
        if not fit["inliers"][i] and fit["H"] is not None:
            q, ok = to_image(fit["H"], ref[i:i + 1])
            if ok[0]:
                cv2.line(img, p, tuple(np.int32(q[0] * s)), OUT_C, 2, cv2.LINE_AA)
                cv2.circle(img, tuple(np.int32(q[0] * s)), int(6 * scale), OUT_C, 2, cv2.LINE_AA)
        cv2.circle(img, p, int(9 * scale), INK, -1, cv2.LINE_AA)
        cv2.circle(img, p, int(7 * scale), col, -1, cv2.LINE_AA)
        t = str(i + 1)
        (w, h), _ = cv2.getTextSize(t, FONT, 0.6 * scale, 2)
        org = (p[0] + int(10 * scale), p[1] - int(10 * scale))
        cv2.rectangle(img, (org[0] - 4, org[1] - h - 4), (org[0] + w + 4, org[1] + 6), INK, -1)
        cv2.putText(img, t, org, FONT, 0.6 * scale, PAPER, 2, cv2.LINE_AA)


def draw_radar(img, ctx, f: int, H) -> None:
    rad = ctx["radar"].copy()
    m, px = 14, RADAR_PX
    if np.isfinite(H[0, 0]):
        w, h = ctx["size"]
        # the part of the pitch the camera sees: the picture's outline sent through H
        edge = np.concatenate([np.linspace((0, h * .35), (w, h * .35), 20),
                               np.linspace((w, h * .35), (w, h), 10),
                               np.linspace((w, h), (0, h), 20),
                               np.linspace((0, h), (0, h * .35), 10)])
        v = cv2.perspectiveTransform(edge[None], H)[0]
        if np.isfinite(v).all() and np.abs(v).max() < 1e3:
            ov = rad.copy()
            cv2.fillPoly(ov, [np.int32(v * px + m)], (150, 190, 150))
            cv2.addWeighted(ov, 0.45, rad, 0.55, 0, rad)
        feet = ctx["feet"].get(f, [])
        if feet:
            xy = np.array([(x, y) for x, y, _ in feet], np.float64)
            p = cv2.perspectiveTransform(xy[None], H)[0]
            for (x, y), (_, _, col) in zip(p, feet):
                if np.isfinite(x) and -5 < x < pitch.LENGTH + 5 and -5 < y < pitch.WIDTH + 5:
                    c = (int(x * px + m), int(y * px + m))
                    cv2.circle(rad, c, 8, INK, -1, cv2.LINE_AA)
                    cv2.circle(rad, c, 6, col, -1, cv2.LINE_AA)
    rh, rw = rad.shape[:2]
    x0, y0 = OUT[0] - rw - 30, OUT[1] - rh - 30
    cv2.rectangle(img, (x0 - 6, y0 - 46), (x0 + rw + 6, y0 + rh + 6), INK, -1)
    cv2.putText(img, "TOP-DOWN RADAR (105 x 68 m)", (x0 + 6, y0 - 14), FONT, 0.7, PAPER, 2,
                cv2.LINE_AA)
    img[y0:y0 + rh, x0:x0 + rw] = rad


def draw(img: np.ndarray, f: int, ctx: dict, opts: dict) -> None:
    s = OUT[0] / ctx["size"][0]
    kp, fit, H = ctx["kp"][f], ctx["fits"][f], ctx["H"][f]
    ok = np.isfinite(H[0, 0])
    if opts.get("lines", True) and ok:
        draw_lines(img, H, s)
    if opts.get("keypoints", True):
        draw_keypoints(img, kp, fit, s)
    if opts.get("radar", True):
        draw_radar(img, ctx, f, H)

    title = ("PITCH KEYPOINTS: DETECTOR + RANSAC" if not opts.get("lines", True)
             else "RAW FIT: THIS FRAME'S KEYPOINTS ONLY" if ctx["hom"] == "raw"
             else "FINAL: KEYPOINTS + CAMERA MOTION")
    x, y = 40, 70
    _text(img, title, (x, y), 1.0, PAPER, 2, bg=INK)
    rows = [f"PITCH KEYPOINTS: {int(fit['used'].sum())} / 32",
            f"RANSAC INLIERS: {int(fit['inliers'].sum())} / {int(fit['used'].sum())}"]
    err = frame_error(H, kp) if ok else np.nan
    rows.append(f"REPROJECTION ERROR: {err:.2f} m" if np.isfinite(err) else
                "REPROJECTION ERROR: -")
    for i, r in enumerate(rows):
        _text(img, r, (x, y + 60 + 50 * i), 0.85, INK, 2, bg=PAPER)
    if not ok:
        _text(img, "FIT REJECTED", (x, y + 60 + 50 * len(rows)), 0.9, PAPER, 2, bg=OUT_C)
    _text(img, f"frame {f}  {f / ctx['fps']:5.2f} s", (40, OUT[1] - 40), 0.7, PAPER, 2, bg=INK)


def stats(run_dir: Path) -> dict:
    """Keypoints per frame, rejected fits, RANSAC inliers, reprojection error (m), and how much
    the projected pitch moves between frames with the raw vs the final homography."""
    ctx = load(run_dir, "raw")
    kp, fits = ctx["kp"], ctx["fits"]
    Hf = np.load(run_dir / "homography.npy")
    Hr = np.stack([f["H"] if f["H"] is not None else np.full((3, 3), np.nan) for f in fits])
    n_used = np.array([f["used"].sum() for f in fits])
    n_inl = np.array([f["inliers"].sum() for f in fits])
    reasons = [f["reason"] for f in fits]

    def q(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if not len(a):
            return None
        return {"median": round(float(np.median(a)), 3), "p90": round(float(np.percentile(a, 90)), 3),
                "mean": round(float(a.mean()), 3), "max": round(float(a.max()), 3)}

    # jitter: how far a fixed grid of image points moves on the pitch from frame to frame
    w, h = ctx["size"]
    grid = homography._grid(w, h)

    def jump(Hs):
        okf = np.isfinite(Hs[:, 0, 0])
        P = np.full((len(Hs), len(grid), 2), np.nan)
        P[okf] = homography._apply(Hs[okf], grid)
        return np.median(np.linalg.norm(np.diff(P, axis=0), axis=-1), axis=1)

    err_raw = [frame_error(Hr[i], kp[i]) for i in range(len(kp))]
    err_fused = [frame_error(Hf[i], kp[i]) for i in range(len(kp))]
    inl_err = [np.median(f["err_m"][f["inliers"]]) if f["inliers"].any() else np.nan for f in fits]
    out_n = n_used - n_inl
    return {
        "frames": len(kp), "fps": ctx["fps"],
        "keypoints_per_frame": {"mean": round(float(n_used.mean()), 2),
                                "median": float(np.median(n_used)), "min": int(n_used.min()),
                                "max": int(n_used.max()),
                                "frames_under_4": int((n_used < 4).sum())},
        "keypoints_low_conf_per_frame": round(float(((kp[:, :, 2] > 0.05)
                                                     & (kp[:, :, 2] <= 0.5)).sum(1).mean()), 2),
        "raw_fit": {"accepted": int(sum(r == "" for r in reasons)),
                    "rejected": {k: int(sum(r == k for r in reasons))
                                 for k in ("few points", "ransac", "error")}},
        "ransac_inliers_per_frame": {"mean": round(float(n_inl.mean()), 2),
                                     "median": float(np.median(n_inl)),
                                     "inlier_share": round(float(n_inl.sum() / max(n_used.sum(), 1)), 4),
                                     "frames_with_outliers": int((out_n > 0).sum()),
                                     "outliers_total": int(out_n.sum())},
        "reprojection_error_m": {"raw_all_points": q(err_raw), "raw_inliers_only": q(inl_err),
                                 "final_all_points": q(err_fused)},
        "frame_to_frame_jump_m": {"raw": q(jump(Hr)), "final": q(jump(Hf)),
                                  "note": "median movement on the pitch of 12 fixed image points "
                                          "between consecutive frames"},
        "final_frames_with_homography": int(np.isfinite(Hf[:, 0, 0]).sum()),
    }
