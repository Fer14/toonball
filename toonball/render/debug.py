"""Debug overlays: a stage's cached tables drawn on the real clip → 1920×1080 MP4.

The pipeline itself never draws on the video (stages write tables, not pixels); this is only
for explaining and checking a stage. `toonball show STAGE RUN_DIR`.

- shots: shot number and wide / dropped label, a CUT flash on each cut, the two cut scores of
  io/shots.py (colour histogram, pixel difference) against their threshold, and a timeline
  strip of the whole video with the wide (kept) part of every shot.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from toonball.io.shots import _hist

OUT = (1920, 1080)
INK, PAPER = (34, 34, 34), (236, 241, 238)
ACCENT = (69, 94, 47)      # #2f5e45 (BGR)
DROP = (150, 150, 150)
RED, BLUE = (40, 40, 220), (138, 108, 74)  # BLUE = #4a6c8a
FONT = cv2.FONT_HERSHEY_DUPLEX


def _source(run_dir: Path, info: dict) -> Path:
    if "video" in info and Path(info["video"]).exists():
        return Path(info["video"])
    root = Path(__file__).resolve().parents[2]
    return root / "clips" / f"{run_dir.name.split('@')[0]}.mp4"


def _text(img, s: str, org, scale: float, colour=INK, thick: int = 2, bg=None, pad: int = 10):
    (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad), bg, -1)
    cv2.putText(img, s, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)
    return w


class _Writer:
    def __init__(self, out: Path, fps: float):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
             f"{OUT[0]}x{OUT[1]}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset",
             "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)

    def write(self, img: np.ndarray) -> None:
        self.p.stdin.write(img.tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait()


def _shot_scores(video: Path, n: int, hist_thresh: float = 0.4,
                 diff_ratio: float = 4.0) -> np.ndarray:
    """(n, 2) the two cut tests of io/shots._is_cut as scores, cut when either reaches 1:
    colour-histogram (Bhattacharyya) jump, and pixel difference far above its recent level."""
    cap, prev, diffs = cv2.VideoCapture(str(video)), None, []
    out = np.zeros((n, 2))
    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        h = _hist(frame)
        if prev is not None:
            out[f, 0] = cv2.compareHist(prev[0], h[0], cv2.HISTCMP_BHATTACHARYYA) / hist_thresh
            d = float(np.abs(h[1] - prev[1]).mean())
            recent = np.median(diffs[-12:]) if len(diffs) >= 4 else None
            diffs.append(d)
            if recent is not None:
                out[f, 1] = min(d / 12.0, d / (diff_ratio * max(recent, 1.5)))
            if out[f].max() >= 1:
                diffs.clear()
        prev = h
    cap.release()
    return out


def _shots(img: np.ndarray, f: int, info: dict, score: np.ndarray) -> None:
    n, fps, cuts, wide = info["n_frames"], info["fps"], info["cuts"], info["wide"]
    W, H = OUT
    k = int(np.searchsorted(cuts, f, side="right")) - 1
    is_wide = any(s["start"] <= f < s["end"] for s in wide)
    in_kept_shot = any(cuts[k] <= s["start"] < cuts[k + 1] for s in wide)

    # cut flash: a thick frame and a label for half a second after each cut
    since = f - cuts[k]
    if k > 0 and since < 0.5 * fps:
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), PAPER, 14)
        _text(img, "CUT", (W // 2 - 70, 140), 3.0, PAPER, 6, bg=INK, pad=18)

    # top-left: shot number, kept / dropped, frame and time
    label = f"SHOT {k + 1}/{len(cuts) - 1}"
    state = "WIDE: KEPT" if is_wide else ("TRIMMED EDGE" if in_kept_shot else "DROPPED")
    x = 40
    x += _text(img, label, (x, 70), 1.3, PAPER, 2, bg=INK) + 40
    _text(img, state, (x, 70), 1.3, PAPER, 2, bg=ACCENT if is_wide else DROP)
    _text(img, f"frame {f:4d}   {f / fps:5.2f} s   {fps:g} fps", (40, 130), 0.9, PAPER, 2,
          bg=INK, pad=8)

    # bottom: timeline of the whole video, histogram distance above it
    x0, x1, y0, y1 = 40, W - 40, H - 70, H - 36
    px = lambda fr: int(x0 + (x1 - x0) * fr / max(n, 1))
    cv2.rectangle(img, (x0 - 12, H - 220), (x1 + 12, H - 20), PAPER, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), DROP, -1)
    for s in wide:
        cv2.rectangle(img, (px(s["start"]), y0), (px(s["end"]), y1), ACCENT, -1)
    for c in cuts[1:-1]:
        cv2.line(img, (px(c), y0 - 8), (px(c), y1 + 8), INK, 3)
    for j in range(len(cuts) - 1):
        a, b = px(cuts[j]), px(cuts[j + 1])
        if b - a > 60:
            _text(img, str(j + 1), ((a + b) // 2 - 8, y1 - 8), 0.7, PAPER, 2)
    # the two cut scores (1 = cut threshold), clipped at 2
    gy0, gy1 = H - 200, H - 90
    gy = lambda v: int(gy1 - (gy1 - gy0) * min(v, 2.0) / 2)
    for xx in range(x0, x1, 24):
        cv2.line(img, (xx, gy(1)), (xx + 12, gy(1)), RED, 2)
    _text(img, "cut threshold", (x1 - 140, gy(1) - 8), 0.6, RED, 1)
    for c, (name, col) in enumerate([("colour histogram change", BLUE),
                                     ("pixel difference vs recent frames", INK)]):
        _text(img, name, (x0 + 380 * c + 26, gy0 + 4), 0.6, col, 1)
        cv2.line(img, (x0 + 380 * c, gy0 - 2), (x0 + 380 * c + 18, gy0 - 2), col, 3)
        pts = np.array([(px(i), gy(score[i, c])) for i in range(f + 1)], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
    cv2.line(img, (px(f), gy0), (px(f), y1 + 8), RED, 3)


STAGES = {"shots": _shots}


def render(stage: str, run_dir: Path, out: Path, start: float = 0.0, dur: float = 0.0,
           speed: float = 1.0) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; one of {', '.join(STAGES)}")
    info = json.loads((run_dir / "shots.json").read_text())
    video = _source(run_dir, info)
    n, fps = info["n_frames"], info["fps"]
    score = _shot_scores(video, n)
    a = int(start * fps)
    b = n if not dur else min(n, a + int(dur * fps))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    w = _Writer(out, fps * speed)
    for f in range(a, b):
        ok, frame = cap.read()
        if not ok:
            break
        img = cv2.resize(frame, OUT, interpolation=cv2.INTER_AREA)
        STAGES[stage](img, f, info, score)
        w.write(img)
    cap.release()
    w.close()
