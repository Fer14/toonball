"""Stills for the tracking lesson (ch03), from a run's cached tables.

    uv run python scripts/track_stills.py iou RUN_DIR OUT_DIR --frame 205 --around 4 [--centre X Y]
    uv run python scripts/track_stills.py occlusion RUN_DIR OUT_DIR --old 4 --new 27

iou: frame t with its tracked boxes and ByteTrack's guess for frame t+1 (its constant-velocity
Kalman filter one step ahead, from replaying the tracker on the cached detections), then frame
t+1 with those guesses (dashed) against the detector's new boxes (solid, D1..Dn), the IoU of
every pair and the Hungarian assignment; plus the IoU matrix and a JSON with the numbers. The
cost is ByteTrack's: 1 - IoU x detection confidence, and a pair only matches below 0.8. The
assignment is solved over every track and box of the frame (only the window is drawn), and the
JSON says whether it picks the same pairs ByteTrack did.

occlusion: three crops of one person disappearing behind another: the last frame under the old
ByteTrack id, a frame while hidden (the dashed guess), the first frame under the new id, with
what ByteTrack says ("NEW ID") and what track/players.py says ("SAME PERSON").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toonball.render.debug import (
    ACCENT,
    DROP,
    INK,
    LOST_C,
    PAPER,
    _chip,
    _dashed_rect,
    _text,
    id_colour,
)
from toonball.track.replay import replay

W, H = 1920, 1080


def _frame(video: Path, f: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {f} of {video}")
    return img


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return float(i / u) if u > 0 else 0.0


def _boxes(df: pl.DataFrame, f: int) -> dict[int, np.ndarray]:
    return {int(r["track_id"]): np.array([r["x1"], r["y1"], r["x2"], r["y2"]])
            for r in df.filter(pl.col("frame") == f).iter_rows(named=True)}


class Crop:
    """A window of the frame blown up to fill the output."""

    def __init__(self, cx: float, cy: float, w: int, h: int, size: tuple[int, int],
                 out: tuple[int, int]):
        self.x0 = int(np.clip(cx - w / 2, 0, size[0] - w))
        self.y0 = int(np.clip(cy - h / 2, 0, size[1] - h))
        self.w, self.h, self.out = w, h, out
        self.z = out[0] / w

    def img(self, frame: np.ndarray) -> np.ndarray:
        c = frame[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        return cv2.resize(c, self.out, interpolation=cv2.INTER_CUBIC)

    def pt(self, x: float, y: float) -> tuple[int, int]:
        return round((x - self.x0) * self.z), round((y - self.y0) * self.z)

    def box(self, b) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.pt(b[0], b[1]), self.pt(b[2], b[3])

    def inside(self, b, margin: float = 0.0) -> bool:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return (self.x0 - margin < cx < self.x0 + self.w + margin
                and self.y0 - margin < cy < self.y0 + self.h + margin)

    def whole(self, b, pad: float = 4.0) -> bool:
        """The whole box (and its label) is in the window, below the title."""
        return (b[0] > self.x0 + pad and b[2] < self.x0 + self.w - pad
                and b[1] > self.y0 + 0.17 * self.h and b[3] < self.y0 + self.h - 0.06 * self.h)


def _title(img, s: str, sub: str = "") -> None:
    _text(img, s, (40, 70), 1.3, PAPER, 2, bg=INK)
    if sub:
        _text(img, sub, (40, 130), 0.9, INK, 2, bg=PAPER, pad=8)


def iou_stills(run: Path, out: Path, t: int, around: int, win: int,
               centre: list[float] | None = None) -> None:
    people = pl.read_parquet(run / "people.parquet")
    dets = pl.read_parquet(run / "detections.parquet").filter(pl.col("cls") != "ball")
    meta = json.loads((run / "meta.json").read_text())
    video = run / "clip.mp4"
    cur, nxt = _boxes(people, t), _boxes(people, t + 1)
    rp = replay(run, meta["fps"], t + 2)
    if around not in cur:
        raise SystemExit(f"id {around} not tracked in frame {t}")
    a = cur[around]
    cx, cy = centre or ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    crop = Crop(cx, cy, win, win * 9 // 16, tuple(meta["size"]), (W, H))

    # ByteTrack's first association over the whole frame: every track (seen or lost) with its
    # Kalman guess for t+1 against every new box
    pred = rp.pred[t + 1]
    pool = sorted(pred)
    d_all = dets.filter(pl.col("frame") == t + 1).sort("x1")
    D_all = d_all.select("x1", "y1", "x2", "y2").to_numpy()
    c_all = d_all["conf"].to_numpy()
    cost_all = np.array([[1 - _iou(pred[i], d) * c for d, c in zip(D_all, c_all)] for i in pool])
    ri, ci = linear_sum_assignment(np.where(cost_all < 0.8, cost_all, 10.0))
    match_all = {pool[r]: c for r, c in zip(ri, ci) if cost_all[r, c] < 0.8}

    # the window: tracks seen in it at t (or lost there), boxes in it at t+1
    ids = sorted(i for i in pool if crop.whole(cur.get(i, pred[i])))
    cols = [c for c in range(len(D_all)) if crop.whole(D_all[c])]
    D, conf = [D_all[c] for c in cols], c_all[cols]
    iou = np.array([[_iou(pred[i], d) for d in D] for i in ids])
    cost = 1 - iou * conf[None, :]
    match = {i: cols.index(match_all[i]) for i in ids if match_all.get(i) in cols}

    # what ByteTrack really did: the id carried by the same box in frame t+1
    actual, actual_all = {}, {}
    for c, d in enumerate(D_all):
        for i, b in nxt.items():
            if np.allclose(b, d, atol=0.01):
                actual_all[c] = i
                if c in cols:
                    actual[cols.index(c)] = i
    agree = sum(actual_all.get(c) == i for i, c in match_all.items())

    # 1. frame t: tracked boxes and their guess for the next frame
    img = crop.img(_frame(video, t))
    for i in ids:
        col = id_colour(i)
        q1, q2 = crop.box(pred[i])
        _dashed_rect(img, q1, q2, col, 3, 12)
        if i not in cur:  # lost: only the guess exists
            _chip(img, f"#{i} LOST", q1, LOST_C, 1.1)
            continue
        p1, p2 = crop.box(cur[i])
        cv2.rectangle(img, p1, p2, col, 4, cv2.LINE_AA)
        _chip(img, f"#{i}", p1, col, 1.1)
        c0 = crop.pt((cur[i][0] + cur[i][2]) / 2, cur[i][3])
        v = (pred[i] - cur[i])
        c1 = crop.pt((cur[i][0] + cur[i][2]) / 2 + 6 * (v[0] + v[2]) / 2, cur[i][3] + 6 * v[3])
        cv2.arrowedLine(img, c0, c1, col, 4, cv2.LINE_AA, tipLength=0.3)
    _title(img, f"FRAME {t}: TRACKS AND THEIR GUESS FOR FRAME {t + 1}",
           f"solid = box now   dashed = guess (constant velocity)   arrow = motion x6   "
           f"(window x{crop.z:.0f})")
    cv2.imwrite(str(out / f"still_iou_frame{t:04d}_tracks.png"), img)

    # 2. frame t+1: guesses vs new detections, IoU on every overlapping pair, the matches
    img = crop.img(_frame(video, t + 1))
    labels = []
    for c, d in enumerate(D):
        p1, p2 = crop.box(d)
        cv2.rectangle(img, p1, p2, PAPER, 3, cv2.LINE_AA)
        _chip(img, f"D{c + 1} {conf[c]:.2f}", (p1[0], p2[1] + 34), PAPER, 0.9)
    for r, i in enumerate(ids):
        col = id_colour(i)
        q1, q2 = crop.box(pred[i])
        _dashed_rect(img, q1, q2, col, 4, 12)
        _chip(img, f"#{i}", q1, col, 1.1)
        for c in range(len(D)):
            if iou[r, c] <= 0.02:
                continue
            pa = crop.pt((pred[i][0] + pred[i][2]) / 2, (pred[i][1] + pred[i][3]) / 2)
            pb = crop.pt((D[c][0] + D[c][2]) / 2, (D[c][1] + D[c][3]) / 2)
            hit = match.get(i) == c
            cv2.line(img, pa, pb, ACCENT if hit else DROP, 6 if hit else 2, cv2.LINE_AA)
            labels.append(((pa[0] + pb[0]) // 2 + 10, (pa[1] + pb[1]) // 2 - 10,
                           f"#{i}-D{c + 1} IoU {iou[r, c]:.2f}", ACCENT if hit else DROP))
    placed: list[tuple[int, int, int, int]] = []
    for x, y, lab, col in sorted(labels, key=lambda v: v[1]):
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
        while any(x < b[2] and x + tw + 8 > b[0] and y > b[1] and y - th - 8 < b[3]
                  for b in placed):
            y += th + 14
        placed.append((x, y - th - 8, x + tw + 8, y))
        _chip(img, lab, (x, y), col, 0.8)
    _title(img, f"FRAME {t + 1}: MATCH GUESSES TO NEW BOXES BY OVERLAP",
           "dashed = guess   white = detector's boxes   green line = Hungarian match")
    cv2.imwrite(str(out / f"still_iou_frame{t + 1:04d}_match.png"), img)

    # 3. the IoU matrix with the assignment
    img = np.full((H, W, 3), PAPER, np.uint8)
    _title(img, "IoU OF EVERY GUESS WITH EVERY NEW BOX",
           "green = the Hungarian algorithm's pick (one box per track, best total overlap)")
    cw, ch = min(240, (W - 400) // max(len(D), 1)), min(110, 760 // max(len(ids), 1))
    x0, y0 = 320, 260
    for c in range(len(D)):
        _text(img, f"D{c + 1}", (x0 + c * cw + cw // 2 - 30, y0 - 20), 1.2, INK, 2)
    for r, i in enumerate(ids):
        y = y0 + r * ch
        cv2.rectangle(img, (80, y + 10), (80 + 180, y + ch - 10), id_colour(i), -1)
        _text(img, f"#{i}", (110, y + ch // 2 + 16), 1.3, INK if sum(id_colour(i)) > 360 else PAPER, 3)
        for c in range(len(D)):
            hit = match.get(i) == c
            cv2.rectangle(img, (x0 + c * cw + 6, y + 6), (x0 + (c + 1) * cw - 6, y + ch - 6),
                          ACCENT if hit else (220, 226, 223), -1)
            _text(img, f"{iou[r, c]:.2f}", (x0 + c * cw + cw // 2 - 50, y + ch // 2 + 16), 1.3,
                  PAPER if hit else (INK if iou[r, c] > 0 else DROP), 3 if hit else 2)
    cv2.imwrite(str(out / "still_iou_matrix.png"), img)

    res = {"run": run.name, "frame_t": t, "frame_t1": t + 1, "window_xywh": [crop.x0, crop.y0,
           crop.w, crop.h], "tracks": ids,
           "track_box_t": {i: cur[i].round(1).tolist() for i in ids},
           "predicted_box_t1": {i: pred[i].round(1).tolist() for i in ids},
           "detections_t1": [{"name": f"D{c + 1}", "box": D[c].round(1).tolist(),
                              "conf": round(float(conf[c]), 3)} for c in range(len(D))],
           "iou": iou.round(3).tolist(), "cost_1_minus_iou_x_conf": cost.round(3).tolist(),
           "hungarian": {i: f"D{c + 1}" for i, c in match.items()},
           "bytetrack_actual": {f"D{c + 1}": i for c, i in actual.items()},
           "same_as_bytetrack_whole_frame": f"{agree}/{len(match_all)}",
           "tracks_in_frame": len(pool), "boxes_in_frame": len(D_all)}
    (out / "still_iou.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def occlusion_stills(run: Path, out: Path, old: int, new: int, size_px: int, tag: str,
                     title: str) -> None:
    people = pl.read_parquet(run / "people.parquet")
    video = run / "clip.mp4"
    a = people.filter(pl.col("track_id") == old).sort("frame")
    b = people.filter(pl.col("track_id") == new).sort("frame")
    fa, fb = int(a["frame"][-1]), int(b["frame"][0])
    fm = (fa + fb) // 2
    la, lb = a.row(-1, named=True), b.row(0, named=True)
    box_a = np.array([la["x1"], la["y1"], la["x2"], la["y2"]])
    box_b = np.array([lb["x1"], lb["y1"], lb["x2"], lb["y2"]])
    meta = json.loads((run / "meta.json").read_text())
    rp = replay(run, meta["fps"], fb + 1)
    guess_m, guess_b = rp.pred[fm][old], rp.pred[fb][old]
    iou_b = _iou(guess_b, box_b)
    c = (box_a + box_b) / 2
    pw = 600
    crop = Crop((c[0] + c[2]) / 2, (c[1] + c[3]) / 2, size_px, size_px, tuple(meta["size"]),
                (pw, pw))
    canvas = np.full((H, W, 3), PAPER, np.uint8)
    _text(canvas, title, (40, 80), 1.3, PAPER, 2, bg=INK)
    panels = [(fa, f"frame {fa}: last seen", f"BYTETRACK #{old}", box_a, id_colour(old), None),
              (fm, f"frame {fm}: not detected", f"LOST #{old}: guess", guess_m, LOST_C, None),
              (fb, f"frame {fb}: back", f"NEW ID #{new}", box_b, id_colour(new),
               f"toonball: SAME PERSON #{old}")]
    others = {f: _boxes(people, f) for f in (fa, fm, fb)}
    for k, (f, head, lab, box, col, after) in enumerate(panels):
        img = crop.img(_frame(video, f))
        for i, ob in others[f].items():
            if i in (old, new) or not crop.inside(ob, 60):
                continue
            p1, p2 = crop.box(ob)
            cv2.rectangle(img, p1, p2, id_colour(i), 2, cv2.LINE_AA)
            _chip(img, f"#{i}", p1, id_colour(i), 0.7)
        p1, p2 = crop.box(box)
        if k == 2:  # where the lost track was guessed to be when he came back
            g1, g2 = crop.box(guess_b)
            _dashed_rect(img, g1, g2, LOST_C, 3, 12)
            _chip(img, f"guess for #{old}: IoU {iou_b:.2f}", (g1[0], g2[1] + 30), LOST_C, 0.6)
        if k == 1:
            _dashed_rect(img, p1, p2, col, 4, 12)
        else:
            cv2.rectangle(img, p1, p2, col, 4, cv2.LINE_AA)
        _chip(img, lab, (p1[0], p1[1]), col, 0.8)
        cv2.imwrite(str(out / f"still_occlusion{tag}_{k + 1}_frame{f:04d}.png"), img)
        x = 40 + k * (pw + 30)
        canvas[200:200 + pw, x:x + pw] = img
        _text(canvas, head, (x, 180), 1.0, INK, 2)
        if after:
            _text(canvas, after, (x, 200 + pw + 70), 1.0, INK, 2, bg=ACCENT, pad=10)
            _text(canvas, after, (x, 200 + pw + 70), 1.0, PAPER, 2)
    score = iou_b * lb["conf"]
    _text(canvas, f"Not detected for {fb - fa - 1} frames. ByteTrack keeps a lost track 2 s, but when he "
                  f"comes back the guess overlaps him by IoU {iou_b:.2f}", (40, 200 + pw + 140),
          0.8, INK, 2)
    _text(canvas, f"(x confidence {lb['conf']:.2f} = {score:.2f}, a match needs more than 0.2), "
                  f"so he gets a new id.", (40, 200 + pw + 180), 0.8, INK, 2)
    cv2.imwrite(str(out / f"still_occlusion{tag}.png"), canvas)
    res = {"run": run.name, "old_id": old, "new_id": new, "last_frame_old": fa,
           "hidden_frame_shown": fm, "first_frame_new": fb, "frames_hidden": fb - fa - 1,
           "guess_at_return": guess_b.round(1).tolist(), "new_box": box_b.round(1).tolist(),
           "iou_guess_vs_new_box": round(iou_b, 3), "new_box_conf": round(lb["conf"], 3),
           "match_score_iou_x_conf": round(score, 3), "needed": 0.2}
    (out / f"still_occlusion{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["iou", "occlusion"])
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frame", type=int, default=205)
    ap.add_argument("--around", type=int, default=4)
    ap.add_argument("--win", type=int, default=480, help="iou: window width in source px")
    ap.add_argument("--centre", type=float, nargs=2, help="iou: window centre (default: --around)")
    ap.add_argument("--old", type=int)
    ap.add_argument("--new", type=int)
    ap.add_argument("--size", type=int, default=240, help="occlusion: crop size in source px")
    ap.add_argument("--tag", default="")
    ap.add_argument("--title", default="OCCLUSION: ONE PLAYER HIDDEN BEHIND ANOTHER")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mode == "iou":
        iou_stills(a.run, a.out, a.frame, a.around, a.win, a.centre)
    else:
        occlusion_stills(a.run, a.out, a.old, a.new, a.size, a.tag, a.title)


if __name__ == "__main__":
    main()
