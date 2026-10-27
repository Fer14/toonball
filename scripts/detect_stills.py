"""Stills for explaining detection, 1920x1080 PNGs.

    uv run python scripts/detect_stills.py nms CLIP OUT_DIR --frames 100 400
    uv run python scripts/detect_stills.py ball RUN_DIR OUT_DIR [--frame F]

nms: per frame, the raw frame, every box the player model scores >= 0.3 before non-max
suppression, and the boxes left after it (the model's per-class NMS at IoU 0.7 plus the
class-agnostic NMS at 0.5 of detect/run.py, i.e. what the pipeline keeps). Counts -> nms.json.

The raw boxes of one person lie almost on top of each other, so each kept box is also labelled
with how many raw boxes it replaced, and nms_zoom.png shows the most crowded area blown up x3
before and after.

ball: the ball cut out at full resolution next to the same patch after the frame is shrunk to the
1280 px model input, both blown up with square pixels, and the whole frame with the patch marked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
INK, PAPER = (34, 34, 34), (236, 241, 238)
RED = (40, 40, 230)
FONT = cv2.FONT_HERSHEY_DUPLEX
CLS = {"ball": RED, "goalkeeper": (0, 150, 255), "player": (235, 190, 60),
       "referee": (210, 70, 210)}
NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}


def _frame(video: Path, f: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {f} of {video}")
    return img


def _label(img, s, org, scale=1.2, fg=PAPER, bg=INK, thick=2, pad=12):
    (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad), bg, -1)
    cv2.putText(img, s, (x, y), FONT, scale, fg, thick, cv2.LINE_AA)


def _boxes(img, xyxy, cls, conf, thick=2, labels=True):
    for b, c, p in zip(xyxy, cls, conf):
        x1, y1, x2, y2 = map(int, b)
        col = CLS[NAMES[int(c)]]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)
        if labels:
            lab = f"{NAMES[int(c)]} {p:.2f}"
            (tw, th), _ = cv2.getTextSize(lab, FONT, 0.45, 1)
            cv2.rectangle(img, (x1 - 1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
            cv2.putText(img, lab, (x1 + 3, y1 - 5), FONT, 0.45, INK, 1, cv2.LINE_AA)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def _zoom(img, box, k=3, size=(1920, 1080)):
    x0, y0, x1, y1 = box
    return cv2.resize(img[y0:y1, x0:x1], ((x1 - x0) * k, (y1 - y0) * k),
                      interpolation=cv2.INTER_NEAREST)


def nms(clip: Path, out: Path, frames: list[int], conf: float = 0.3) -> None:
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "weights" / "football-player-detection.pt"))
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for f in frames:
        img = _frame(clip, f)
        # iou=1.0: no box overlaps another by more than 100 %, so NMS keeps everything
        raw = model(img, imgsz=1280, conf=conf, iou=1.0, max_det=5000, verbose=False,
                    half=True)[0]
        kept = model(img, imgsz=1280, conf=conf, verbose=False, half=True)[0]
        d = sv.Detections.from_ultralytics(kept)
        people = d[d.class_id != 0].with_nms(threshold=0.5, class_agnostic=True)
        d = sv.Detections.merge([people, d[d.class_id == 0]])
        rb = raw.boxes
        rxy = rb.xyxy.cpu().numpy()
        before = img.copy()
        _boxes(before, rxy, rb.cls.cpu().numpy(), rb.conf.cpu().numpy(), thick=1, labels=False)
        # how many raw boxes each kept box stands for (raw box assigned to its best kept box)
        own = _iou(rxy, d.xyxy).argmax(1) if len(d) and len(rxy) else np.zeros(0, int)
        for j, b in enumerate(d.xyxy):
            k = int((own == j).sum())
            x1, y1 = int(b[0]), int(b[1])
            cv2.putText(before, f"x{k}", (x1, y1 - 6), FONT, 0.55, INK, 4, cv2.LINE_AA)
            cv2.putText(before, f"x{k}", (x1, y1 - 6), FONT, 0.55, PAPER, 1, cv2.LINE_AA)
        _label(before, f"BEFORE NMS: {len(rb)} boxes (confidence >= {conf})", (40, 70))
        after = img.copy()
        _boxes(after, d.xyxy, d.class_id, d.confidence)
        _label(after, f"AFTER NMS: {len(d)} boxes", (40, 70))
        # most crowded 480x270 window (by raw boxes) -> x3 zoom, before | after
        cxy = (rxy[:, :2] + rxy[:, 2:]) / 2
        H, W = img.shape[:2]
        best, bx = -1, (0, 0)
        for x0 in range(0, W - 480 + 1, 40):
            for y0 in range(0, H - 270 + 1, 30):
                m = ((cxy[:, 0] > x0 + 40) & (cxy[:, 0] < x0 + 440) & (cxy[:, 1] > y0 + 40)
                     & (cxy[:, 1] < y0 + 230)).sum()
                if m > best:
                    best, bx = m, (x0, y0)
        win = (bx[0], bx[1], bx[0] + 480, bx[1] + 270)
        zb, za = _zoom(before, win), _zoom(after, win)
        zoom = np.full((1080, 1920, 3), PAPER, np.uint8)
        zoom[135:135 + 810, 0:955] = zb[:, 240:240 + 955]
        zoom[135:135 + 810, 965:1920] = za[:, 240:240 + 955]
        _label(zoom, f"BEFORE NMS ({len(rb)} boxes in the frame)", (40, 90), 1.1)
        _label(zoom, f"AFTER NMS ({len(d)} boxes)", (1005, 90), 1.1)
        cv2.imwrite(str(out / f"frame{f:04d}_nms_zoom.png"), zoom)
        cv2.imwrite(str(out / f"frame{f:04d}_raw.png"), img)
        cv2.imwrite(str(out / f"frame{f:04d}_before_nms.png"), before)
        cv2.imwrite(str(out / f"frame{f:04d}_after_nms.png"), after)
        by = lambda c: {NAMES[k]: int((np.asarray(c) == k).sum()) for k in NAMES}
        counts[f] = {"before_nms": len(rb), "after_nms": len(d),
                     "before_by_class": by(rb.cls.cpu().numpy()),
                     "after_by_class": by(d.class_id)}
        print(f, counts[f])
    (out / "nms.json").write_text(json.dumps(counts, indent=1))


def _pick_ball_frame(run: Path) -> int:
    """A clear, isolated ball of typical size: high confidence, away from people, width close
    to the median ball width of the clip."""
    ball = pl.read_parquet(run / "ball_raw.parquet").with_columns(
        ((pl.col("x1") + pl.col("x2")) / 2).alias("cx"),
        ((pl.col("y1") + pl.col("y2")) / 2).alias("cy"),
        (pl.col("x2") - pl.col("x1")).alias("w"))
    ppl = pl.read_parquet(run / "people.parquet")
    med = float(ball["w"].median())
    best, score = None, -1e9
    for r in ball.filter(pl.col("conf") > 0.6).iter_rows(named=True):
        p = ppl.filter(pl.col("frame") == r["frame"])
        d = np.hypot((p["x1"] + p["x2"]).to_numpy() / 2 - r["cx"],
                     p["y2"].to_numpy() - r["cy"]).min() if p.height else 999
        s = min(d, 120) - 30 * abs(r["w"] - med) + 40 * r["conf"]
        if s > score:
            best, score = r["frame"], s
    return int(best)


def ball(run: Path, out: Path, frame: int | None = None, patch: int = 72,
         model_w: int = 1280, tag: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    f = _pick_ball_frame(run) if frame is None else frame
    b = pl.read_parquet(run / "ball_raw.parquet").filter(pl.col("frame") == f).row(0, named=True)
    img = _frame(run / "clip.mp4", f)
    H, W = img.shape[:2]
    cx, cy = (b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2
    bw = b["x2"] - b["x1"]
    k = model_w / W
    x0 = int(np.clip(cx - patch / 2, 0, W - patch))
    y0 = int(np.clip(cy - patch / 2, 0, H - patch))
    full = img[y0:y0 + patch, x0:x0 + patch]
    # the same resize ultralytics' letterbox does
    small = cv2.resize(img, (model_w, round(H * k)), interpolation=cv2.INTER_LINEAR)
    sp = round(patch * k)
    sx0, sy0 = round(x0 * k), round(y0 * k)
    low = small[sy0:sy0 + sp, sx0:sx0 + sp]

    canvas = np.full((1080, 1920, 3), PAPER, np.uint8)
    S = 600
    big_full = cv2.resize(full, (S, S), interpolation=cv2.INTER_NEAREST)
    big_low = cv2.resize(low, (S, S), interpolation=cv2.INTER_NEAREST)
    # thumbnail of the whole frame with the patch marked
    th = cv2.resize(img, (560, 315), interpolation=cv2.INTER_AREA)
    t = 560 / W
    cv2.rectangle(th, (int(x0 * t) - 3, int(y0 * t) - 3),
                  (int((x0 + patch) * t) + 3, int((y0 + patch) * t) + 3), RED, 2)
    ys = 300
    canvas[ys:ys + 315, 60:620] = th
    canvas[ys - 100:ys - 100 + S, 660:660 + S] = big_full
    canvas[ys - 100:ys - 100 + S, 1290:1290 + S] = big_low
    _label(canvas, "The ball at full size vs at the model input", (60, 80), 1.5)
    cv2.putText(canvas, f"frame {f}, {W}x{H}", (60, ys + 360), FONT, 0.8, INK, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{patch}x{patch} px patch", (60, ys + 400), FONT, 0.8, INK, 2,
                cv2.LINE_AA)
    cv2.putText(canvas, f"full resolution ({W} px wide)", (660, ys - 120), FONT, 0.9, INK, 2,
                cv2.LINE_AA)
    cv2.putText(canvas, f"shrunk to the {model_w} px model input", (1290, ys - 120), FONT, 0.9,
                INK, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"ball {bw:.0f} px wide", (660, ys + S - 50), FONT, 1.2, RED, 2,
                cv2.LINE_AA)
    cv2.putText(canvas, f"ball {bw * k:.0f} px wide", (1290, ys + S - 50), FONT, 1.2, RED, 2,
                cv2.LINE_AA)
    cv2.putText(canvas, f"same patch: {patch * patch} pixels -> {sp * sp} pixels", (1290,
                ys + S - 5), FONT, 0.8, INK, 2, cv2.LINE_AA)
    cv2.imwrite(str(out / f"ball_full_vs_1280{tag}.png"), canvas)
    cv2.imwrite(str(out / f"ball_patch_full{tag}.png"), big_full)
    cv2.imwrite(str(out / f"ball_patch_1280{tag}.png"), big_low)
    info = {"frame": f, "ball_w_px": round(bw, 1), "ball_w_px_at_1280": round(bw * k, 1),
            "patch": patch, "patch_at_1280": sp, "conf": round(b["conf"], 3)}
    (out / f"ball{tag}.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("nms")
    a.add_argument("clip", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--frames", type=int, nargs="+", default=[100])
    a.add_argument("--conf", type=float, default=0.3)
    b = sub.add_parser("ball")
    b.add_argument("run", type=Path)
    b.add_argument("out", type=Path)
    b.add_argument("--frame", type=int)
    b.add_argument("--tag", default="", help="suffix for the output file names")
    args = ap.parse_args()
    if args.cmd == "nms":
        nms(args.clip, args.out, args.frames, args.conf)
    else:
        ball(args.run, args.out, args.frame, tag=args.tag)
