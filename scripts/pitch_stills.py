"""Stills for ch04 (pitch model + homography), 1920x1080 PNG + JSON with the numbers.

    uv run python scripts/pitch_stills.py model  runs/<clip> OUT_DIR --frame F
    uv run python scripts/pitch_stills.py ransac runs/<clip> OUT_DIR --frame F

model:  the 32-point pitch model (numbers 1-32) next to the video frame, the keypoints found
        in that frame joined to their model point (RANSAC inliers green, outliers red).
ransac: one frame with the RANSAC inliers (green) and outliers (red, line to where the fit puts
        them), the pitch lines of the RANSAC fit (yellow) and of a plain least-squares fit
        through all points (red), with each point's error in metres.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from toonball.calib import homography, pitch
from toonball.render.debug import FONT, INK, PAPER, _text
from toonball.render.debug_pitch import (IN_C, LINE_C, OUT_C, draw_keypoints, draw_lines,
                                         model_lines)


def frame_at(run: Path, f: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(run / "clip.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    cap.release()
    assert ok, f
    return cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)


def model(run: Path, out: Path, f: int) -> dict:
    kp = np.load(run / "keypoints.npy")[f]
    fit = homography.fit_frame(kp)
    ref = pitch.vertices()
    c = np.full((1080, 1920, 3), PAPER, np.uint8)
    # left: the model, 8.4 px per metre
    px, ox, oy = 8.4, 30, 300
    W, H = int(pitch.LENGTH * px), int(pitch.WIDTH * px)
    cv2.rectangle(c, (ox - 12, oy - 12), (ox + W + 12, oy + H + 12), (96, 148, 92), -1)
    for poly in model_lines():
        cv2.polylines(c, [np.int32(poly * px + (ox, oy))], False, PAPER, 2, cv2.LINE_AA)
    mp = ref * px + (ox, oy)
    # right: the frame at 0.5x
    fx, fy, s = 960, 270, 0.5
    img = frame_at(run, f)
    small = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
    draw_keypoints(small, kp, fit, s, scale=0.8)
    c[fy:fy + 540, fx:fx + 960] = small
    for i, p in enumerate(mp):
        hit = fit["used"][i]
        col = (IN_C if fit["inliers"][i] else OUT_C) if hit else (200, 200, 200)
        cv2.circle(c, tuple(np.int32(p)), 9 if hit else 6, INK, -1, cv2.LINE_AA)
        cv2.circle(c, tuple(np.int32(p)), 7 if hit else 4, col, -1, cv2.LINE_AA)
        cv2.putText(c, str(i + 1), (int(p[0]) + 8, int(p[1]) - 8), FONT, 0.55,
                    INK if hit else (110, 110, 110), 2 if hit else 1, cv2.LINE_AA)
    for i in np.flatnonzero(fit["used"]):
        q = kp[i, :2] * s + (fx, fy)
        col = IN_C if fit["inliers"][i] else OUT_C
        cv2.line(c, tuple(np.int32(mp[i])), tuple(np.int32(q)), col, 1, cv2.LINE_AA)
    _text(c, "THE PITCH MODEL: 32 KNOWN POINTS (METRES)", (30, 150), 1.0, PAPER, 2, bg=INK)
    _text(c, f"THE VIDEO: {int(fit['used'].sum())} OF THEM FOUND IN THIS FRAME", (960, 150), 1.0,
          PAPER, 2, bg=INK)
    _text(c, "105 x 68 m", (ox, oy + H + 60), 0.8, INK, 2)
    _text(c, f"frame {f}: {int(fit['inliers'].sum())} kept by RANSAC (green), "
             f"{int(fit['used'].sum() - fit['inliers'].sum())} rejected (red). "
             "4 points are enough for the 8 numbers of a homography.",
          (30, 1000), 0.75, INK, 2)
    cv2.imwrite(str(out / f"still_pitch_model_frame{f:04d}.png"), c)
    return {"frame": f, "found": [int(i + 1) for i in np.flatnonzero(fit["used"])],
            "inliers": [int(i + 1) for i in np.flatnonzero(fit["inliers"])],
            "model_xy_m": {int(i + 1): [round(float(v), 2) for v in ref[i]]
                           for i in np.flatnonzero(fit["used"])},
            "image_xy_px": {int(i + 1): [round(float(v), 1) for v in kp[i, :2]]
                            for i in np.flatnonzero(fit["used"])}}


def ransac(run: Path, out: Path, f: int) -> dict:
    kp = np.load(run / "keypoints.npy")[f]
    fit = homography.fit_frame(kp)
    ref = pitch.vertices().astype(np.float64)
    m = fit["used"]
    src, dst = kp[m, :2].astype(np.float64), ref[m]
    h_ls, _ = cv2.findHomography(src, dst, 0)  # least squares through every point
    err_ls = np.linalg.norm(cv2.perspectiveTransform(src[None], h_ls)[0] - dst, axis=1)
    img = frame_at(run, f)
    draw_lines(img, h_ls, 1.0, OUT_C, 2)
    draw_lines(img, fit["H"], 1.0, LINE_C, 3)
    draw_keypoints(img, kp, fit, 1.0, scale=1.2)
    for i in np.flatnonzero(m):
        e = fit["err_m"][i]
        p = kp[i, :2]
        _text(img, f"{e:.2f} m", (int(p[0]) + 14, int(p[1]) + 30), 0.6,
              PAPER, 1, bg=IN_C if fit["inliers"][i] else OUT_C, pad=4)
    x, y = 40, 830  # bottom left, clear of the far touchline points
    _text(img, f"RANSAC ON ONE FRAME ({f})", (x, y), 1.0, PAPER, 2, bg=INK)
    rows = [(f"INLIERS: {int(fit['inliers'].sum())}", IN_C),
            (f"OUTLIERS: {int(m.sum() - fit['inliers'].sum())}", OUT_C),
            ("YELLOW: RANSAC FIT (INLIERS ONLY)", LINE_C),
            ("RED: LEAST SQUARES THROUGH ALL POINTS", OUT_C)]
    for i, (r, col) in enumerate(rows):
        cv2.rectangle(img, (x - 10, y + 40 + 50 * i), (x + 20, y + 70 + 50 * i), col, -1)
        _text(img, r, (x + 34, y + 66 + 50 * i), 0.8, INK, 2, bg=PAPER, pad=6)
    cv2.imwrite(str(out / f"still_ransac_frame{f:04d}.png"), img)
    ids = np.flatnonzero(m)
    return {"frame": f, "points": [
        {"id": int(i + 1), "inlier": bool(fit["inliers"][i]),
         "err_ransac_m": round(float(fit["err_m"][i]), 3),
         "err_least_squares_m": round(float(err_ls[j]), 3)} for j, i in enumerate(ids)],
        "median_err_ransac_m": round(fit["median_err_m"], 3),
        "median_err_inliers_m": round(float(np.median(fit["err_m"][fit["inliers"]])), 3),
        "median_err_least_squares_m": round(float(np.median(err_ls)), 3),
        "ransac_threshold_m": 1.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["model", "ransac"])
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frame", type=int, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = (model if a.mode == "model" else ransac)(a.run, a.out, a.frame)
    (a.out / f"still_{'pitch_model' if a.mode == 'model' else 'ransac'}.json").write_text(
        json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
