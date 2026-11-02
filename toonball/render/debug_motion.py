"""`toonball show motion` and `toonball show fusion`: camera motion and its fusion with the
per-frame keypoint fits (ch05).

motion: the background corners Lucas–Kanade follows from one frame to the next (players masked,
the mask tinted lightly), drawn as arrows (magnified) — green when RANSAC keeps them for the
frame's homography, red when not — and a readout of the frame-to-frame motion of the picture
centre (pan, tilt in px per frame) and its zoom (% per frame).

fusion: the pitch lines projected with three homographies at once: the raw per-frame keypoint
fit (red), camera motion alone chained from the first frame shown (blue: drifts), and the fused
one (yellow), plus a plot of where the centre spot lands on screen for the three and the
gating decision of the frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from toonball.calib import homography, motion
from toonball.render.debug import FONT, INK, OUT, PAPER, RED, _text, _Writer
from toonball.render.debug_pitch import draw_lines, to_image

IN_C = (90, 200, 60)
RAW_C = (60, 60, 235)        # raw fit, red (BGR)
CHAIN_C = (235, 150, 40)     # motion only, blue
FUSED_C = (40, 225, 255)     # fused, yellow
ARROW_GAIN = 8.0
SCALE = 0.5                  # motion.estimate works on half-size frames


def decompose(R: np.ndarray, size: tuple[int, int]) -> tuple[float, float, float]:
    """Frame-to-frame homography (full-res px) → pan, tilt of the picture centre (px/frame) and
    zoom (% per frame)."""
    w, h = size
    c = np.array([[[w / 2, h / 2]]], np.float64)
    d = cv2.perspectiveTransform(c, R)[0, 0] - c[0, 0]
    A = R[:2, :2] / R[2, 2]
    return float(d[0]), float(d[1]), float((np.sqrt(abs(np.linalg.det(A))) - 1) * 100)


def chain(H0: np.ndarray, R: np.ndarray, a: int) -> np.ndarray:
    """Motion-only homographies: start from H0 (image→pitch) at frame a and carry it with the
    camera motion alone, H[t] = H[t-1] · R[t]⁻¹."""
    H = np.full((len(R), 3, 3), np.nan)
    H[a] = H0
    for t in range(a + 1, len(R)):
        H[t] = H[t - 1] @ np.linalg.inv(R[t])
        H[t] /= H[t, 2, 2]
    return H


def grid_gap(Ha: np.ndarray, Hb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Per frame: median distance (m) on the pitch between where Ha and Hb send the fusion grid
    (the image points fuse() follows)."""
    g = homography._grid(*size)
    out = np.full(len(Ha), np.nan)
    ok = np.isfinite(Ha[:, 0, 0]) & np.isfinite(Hb[:, 0, 0])
    if ok.any():
        pa, pb = homography._apply(Ha[ok], g), homography._apply(Hb[ok], g)
        out[ok] = np.median(np.linalg.norm(pa - pb, axis=-1), axis=1)
    return out


def landmark(H: np.ndarray, pt: tuple[float, float]) -> np.ndarray:
    """(F, 2) pitch position (m) of one image point under each frame's homography."""
    out = np.full((len(H), 2), np.nan)
    ok = np.isfinite(H[:, 0, 0])
    if ok.any():
        out[ok] = homography._apply(H[ok], np.array([pt], np.float64))[:, 0]
    return out


def spot_on_screen(H: np.ndarray, xy=(52.5, 34.0)) -> np.ndarray:
    """(F, 2) image position (px) of one pitch point (default the centre spot) under each
    frame's homography."""
    out = np.full((len(H), 2), np.nan)
    for t in np.flatnonzero(np.isfinite(H[:, 0, 0])):
        p, ok = to_image(H[t], np.array([xy], np.float64))
        if ok[0]:
            out[t] = p[0]
    return out


def load(run_dir: Path, a: int = 0) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    size, fps = tuple(meta["size"]), meta["fps"]
    mo = np.load(run_dir / "motion.npz")
    R, R_ok = mo["R"], mo["ok"]
    p = run_dir / "homography_raw.npy"
    Hr = np.load(p) if p.exists() else homography.fit_raw(np.load(run_dir / "keypoints.npy"))
    Hf = np.load(run_dir / "homography.npy")
    log: dict = {}
    homography.fuse(Hr, R, R_ok, size, fps, log=log)
    Hc = chain(Hf[a], R, a)
    people = pl.read_parquet(run_dir / "people.parquet")
    boxes = {f: g.select("x1", "y1", "x2", "y2").to_numpy()
             for (f,), g in people.group_by("frame")}
    return {"n": len(R), "fps": fps, "size": size, "R": R, "ok": R_ok, "Hr": Hr, "Hf": Hf,
            "Hc": Hc, "a": a, "log": log, "boxes": boxes,
            "pan": np.array([decompose(r, size) for r in R]),
            "lm": {k: spot_on_screen(H) for k, H in (("raw", Hr), ("chain", Hc), ("fused", Hf))},
            "gap_chain": grid_gap(Hc, Hf, size), "gap_raw": grid_gap(Hr, Hf, size)}


def _panel(img, x, y, rows, w=560) -> None:
    h = 20 + 44 * len(rows)
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), INK, -1)
    cv2.addWeighted(ov, 0.82, img, 0.18, 0, img)
    for i, (s, col) in enumerate(rows):
        cv2.putText(img, s, (x + 18, y + 44 + 44 * i), FONT, 0.85, col, 2, cv2.LINE_AA)


def _plot(img, x, y, w, h, series, a, b, f, title, unit, fps) -> None:
    """Small time plot of several (values, colour) over frames a..b with a playhead at f."""
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), INK, -1)
    cv2.addWeighted(ov, 0.82, img, 0.18, 0, img)
    cv2.putText(img, title, (x + 14, y + 30), FONT, 0.7, PAPER, 2, cv2.LINE_AA)
    allv = np.concatenate([v[a:b][np.isfinite(v[a:b])] for v, _ in series])
    lo, hi = (allv.min(), allv.max()) if len(allv) else (0, 1)
    pad = max(0.5, 0.08 * (hi - lo))
    lo, hi = lo - pad, hi + pad
    x0, x1, y0, y1 = x + 70, x + w - 16, y + 46, y + h - 34
    for v_ in np.linspace(lo, hi, 3):
        yy = int(y1 - (v_ - lo) / (hi - lo) * (y1 - y0))
        cv2.line(img, (x0, yy), (x1, yy), (90, 90, 90), 1)
        cv2.putText(img, f"{v_:.0f}", (x + 10, yy + 6), FONT, 0.5, PAPER, 1, cv2.LINE_AA)
    cv2.putText(img, unit, (x + 10, y1 + 26), FONT, 0.5, PAPER, 1, cv2.LINE_AA)
    cv2.putText(img, f"{a / fps:.0f}s", (x0, y1 + 26), FONT, 0.5, PAPER, 1, cv2.LINE_AA)
    cv2.putText(img, f"{b / fps:.0f}s", (x1 - 40, y1 + 26), FONT, 0.5, PAPER, 1, cv2.LINE_AA)
    for v, col in series:
        t = np.arange(a, min(f + 1, b))
        vv = np.clip(v[t], lo, hi)
        px = x0 + (t - a) / max(1, b - 1 - a) * (x1 - x0)
        py = y1 - (vv - lo) / (hi - lo) * (y1 - y0)
        good = np.isfinite(py)
        pts = np.stack([px, py], 1)
        for i in range(1, len(t)):
            if good[i] and good[i - 1]:
                cv2.line(img, tuple(np.int32(pts[i - 1])), tuple(np.int32(pts[i])), col, 2,
                         cv2.LINE_AA)
    xf = int(x0 + (f - a) / max(1, b - 1 - a) * (x1 - x0))
    cv2.line(img, (xf, y0), (xf, y1), RED, 2)


def draw_motion(img, f: int, ctx: dict, fl: dict | None) -> None:
    s = OUT[0] / ctx["size"][0] / SCALE      # working px → output px
    if fl is not None:
        m = cv2.resize(fl["mask"], OUT, interpolation=cv2.INTER_NEAREST) == 0
        tint = img.copy()
        tint[m] = (0.55 * tint[m] + 0.45 * np.array((70, 70, 200))).astype(np.uint8)
        img[:] = tint
        cnt, _ = cv2.findContours((m * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnt, -1, (70, 70, 200), 1, cv2.LINE_AA)
        for p0, p1, st, inl in zip(fl["p0"], fl["p1"], fl["st"], fl["inl"]):
            if not st:
                continue
            a = p0 * s
            b = a + (p1 - p0) * s * ARROW_GAIN
            col = IN_C if inl else RED
            cv2.circle(img, tuple(np.int32(a)), 3, col, -1, cv2.LINE_AA)
            cv2.arrowedLine(img, tuple(np.int32(a)), tuple(np.int32(b)), col, 2, cv2.LINE_AA,
                            tipLength=0.3)
    pan, tilt, zoom = ctx["pan"][f]
    n_pts = int(fl["st"].sum()) if fl else 0
    n_in = int(fl["inl"].sum()) if fl else 0
    okm = bool(ctx["ok"][f])
    _panel(img, 30, 30, [
        ("CAMERA MOTION (frame to frame)", PAPER),
        (f"FLOW POINTS  {n_pts}   KEPT {n_in}", IN_C),
        (f"PAN   {pan:+6.1f} px/frame", PAPER),
        (f"TILT  {tilt:+6.1f} px/frame", PAPER),
        (f"ZOOM  {zoom:+6.2f} % /frame", PAPER),
        ("ONE HOMOGRAPHY: OK" if okm else "ONE HOMOGRAPHY: REJECTED", IN_C if okm else RED),
    ])
    _text(img, f"arrows x{ARROW_GAIN:g}   red tint = players masked", (30, OUT[1] - 250), 0.75,
          PAPER, 2, bg=INK)
    a, b = ctx["win"]
    _plot(img, 30, OUT[1] - 210, 760, 180,
          [(ctx["pan"][:, 0], (235, 150, 40)), (ctx["pan"][:, 1], (60, 200, 230))], a, b, f,
          "PAN (blue)  TILT (yellow)", "px/fr", ctx["fps"])


def _event(ctx, f) -> tuple[str, tuple]:
    ef, eb = ctx["log"]["forward"][f][0], ctx["log"]["backward"][f][0]
    if ef == "reject" and eb == "reject":
        return "RAW FIT REJECTED (both passes)", RED
    if "reject" in (ef, eb):
        return "RAW FIT REJECTED (one pass)", (60, 140, 235)
    if "relock" in (ef, eb):
        return "RE-LOCK on the raw fits", (60, 140, 235)
    return "RAW FIT ACCEPTED (blended)", IN_C


def draw_fusion(img, f: int, ctx: dict) -> None:
    s = OUT[0] / ctx["size"][0]
    for key, col, th in (("Hc", CHAIN_C, 3), ("Hr", RAW_C, 3), ("Hf", FUSED_C, 3)):
        H = ctx[key][f]
        if np.isfinite(H[0, 0]):
            draw_lines(img, H, s, col, th)
    ev, evc = _event(ctx, f)
    gr, gc = ctx["gap_raw"][f], ctx["gap_chain"][f]
    _panel(img, 30, 30, [
        ("PITCH LINES FROM THREE HOMOGRAPHIES", PAPER),
        ("RAW: keypoint fit of this frame", RAW_C),
        (f"MOTION ONLY: chained from {ctx['a'] / ctx['fps']:.0f} s", CHAIN_C),
        ("FUSED: both, forward + backward", FUSED_C),
        (f"raw vs fused     {gr:5.1f} m", RAW_C),
        (f"drift (motion only) {gc:5.1f} m", CHAIN_C),
        (ev, evc),
    ], w=640)
    a, b = ctx["win"]
    _plot(img, 30, OUT[1] - 250, 900, 220,
          [(ctx["lm"]["chain"][:, 0], CHAIN_C), (ctx["lm"]["raw"][:, 0], RAW_C),
           (ctx["lm"]["fused"][:, 0], FUSED_C)], a, b, f,
          "CENTRE SPOT ON SCREEN (x position)", "px", ctx["fps"])


def render(stage: str, run_dir: Path, out: Path, start: float = 0.0, dur: float = 0.0,
           speed: float = 1.0) -> None:
    meta = json.loads((run_dir / "meta.json").read_text())
    fps = meta["fps"]
    a = int(start * fps)
    ctx = load(run_dir, a)
    n = ctx["n"]
    b = n if not dur else min(n, a + int(round(dur * fps)))
    ctx["win"] = (a, b)
    cap = cv2.VideoCapture(str(run_dir / "clip.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, a - 1))
    prev = None
    if a > 0:
        _, fr = cap.read()
        prev = cv2.cvtColor(cv2.resize(fr, None, fx=SCALE, fy=SCALE), cv2.COLOR_BGR2GRAY)
    w = _Writer(out, fps * speed)
    for f in range(a, b):
        good, frame = cap.read()
        if not good:
            break
        img = cv2.resize(frame, OUT, interpolation=cv2.INTER_AREA)
        if stage == "motion":
            gray = cv2.cvtColor(cv2.resize(frame, None, fx=SCALE, fy=SCALE), cv2.COLOR_BGR2GRAY)
            fl = None
            if prev is not None:
                fl = motion.flow(prev, gray, ctx["boxes"].get(f - 1, np.zeros((0, 4))) * SCALE)
            draw_motion(img, f, ctx, fl)
            prev = gray
        else:
            draw_fusion(img, f, ctx)
        w.write(img)
    cap.release()
    w.close()


def _q(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return {"median": round(float(np.median(x)), 3), "p90": round(float(np.percentile(x, 90)), 3),
            "mean": round(float(x.mean()), 3), "max": round(float(x.max()), 3)}


def stats(run_dir: Path) -> dict:
    """Flow points per frame (LK re-run on the clip), motion magnitude, motion-only drift,
    gating counts and the frame-to-frame jump of raw vs fused."""
    ctx = load(run_dir, 0)
    n, fps, size = ctx["n"], ctx["fps"], ctx["size"]
    cap = cv2.VideoCapture(str(run_dir / "clip.mp4"))
    prev, pts, kept, R_diff = None, [], [], []
    S = np.diag([SCALE, SCALE, 1.0])
    for f in range(n):
        good, frame = cap.read()
        if not good:
            break
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=SCALE, fy=SCALE), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            fl = motion.flow(prev, gray, ctx["boxes"].get(f - 1, np.zeros((0, 4))) * SCALE)
            pts.append(int(fl["st"].sum()))
            kept.append(int(fl["inl"].sum()))
            if fl["ok"]:
                R_diff.append(float(np.abs(np.linalg.inv(S) @ fl["h"] @ S - ctx["R"][f]).max()))
        prev = gray
    cap.release()
    pan = ctx["pan"][1:][ctx["ok"][1:]]
    speed = np.hypot(pan[:, 0], pan[:, 1])
    drift = ctx["gap_chain"]
    lf, lb = ctx["log"]["forward"], ctx["log"]["backward"]
    ef, eb = np.array([e for e, _, _ in lf]), np.array([e for e, _, _ in lb])
    kf = np.array([k for e, k, _ in lf if e == "accept"])
    rej = np.flatnonzero((ef == "reject") | (eb == "reject"))

    def jumps(H):
        g = homography._grid(*size)
        ok = np.isfinite(H[:, 0, 0])
        p = np.full((n, len(g), 2), np.nan)
        p[ok] = homography._apply(H[ok], g)
        return np.median(np.linalg.norm(np.diff(p, axis=0), axis=-1), axis=1)

    runs, start = [], None
    for i in range(n + 1):
        on = i < n and i in set(rej.tolist())
        if on and start is None:
            start = i
        if not on and start is not None:
            runs.append([start, i - 1])
            start = None
    return {
        "frames": n, "fps": fps,
        "flow_points_per_frame": {"tracked": _q(pts), "kept_by_ransac": _q(kept),
                                  "min_tracked": int(min(pts)), "min_kept": int(min(kept))},
        "flow_recomputed_matches_motion_npz_max_abs": round(max(R_diff), 9) if R_diff else None,
        "motion_ok_frames": int(ctx["ok"][1:].sum()), "motion_rejected_frames":
            int((~ctx["ok"][1:]).sum()),
        "motion_magnitude": {"centre_speed_px_per_frame": _q(speed),
                             "pan_px_per_frame_abs": _q(np.abs(pan[:, 0])),
                             "tilt_px_per_frame_abs": _q(np.abs(pan[:, 1])),
                             "zoom_pct_per_frame_abs": _q(np.abs(pan[:, 2])),
                             "total_pan_px": round(float(pan[:, 0].sum()), 1),
                             "total_tilt_px": round(float(pan[:, 1].sum()), 1),
                             "total_zoom_pct": round(float((np.prod(1 + pan[:, 2] / 100) - 1)
                                                           * 100), 2)},
        "motion_only_drift_m": {
            "note": "chained from the fused homography at frame 0; median over the fusion grid "
                    "of the distance on the pitch to the fused homography",
            **{f"at_{t}s": round(float(drift[min(n - 1, int(t * fps))]), 2)
               for t in (1, 5, 10, 20, 29.9)},
            "max": round(float(np.nanmax(drift)), 2)},
        "raw_vs_fused_m": _q(ctx["gap_raw"]),
        "gating": {"gate_m": 3.0, "raw_fits": int(np.isfinite(ctx["Hr"][:, 0, 0]).sum()),
                   "rejected_forward": int((ef == "reject").sum()),
                   "rejected_backward": int((eb == "reject").sum()),
                   "rejected_both": int(((ef == "reject") & (eb == "reject")).sum()),
                   "rejected_either": int(len(rej)),
                   "relocks": int((ef == "relock").sum() + (eb == "relock").sum()),
                   "rejected_frame_runs": runs,
                   "accepted_gain_forward": _q(kf),
                   "note": "an accepted raw fit is blended with gain k = v/(v+r): k is the weight "
                           "it gets against the motion prediction (always < 1, i.e. every raw fit "
                           "is down-weighted)"},
        "frame_to_frame_jump_m": {"raw": _q(jumps(ctx["Hr"])), "motion_only": _q(jumps(ctx["Hc"])),
                                  "fused": _q(jumps(ctx["Hf"]))},
    }
