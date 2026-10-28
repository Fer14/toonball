"""Debug overlays: a stage's cached tables drawn on the real clip → 1920×1080 MP4.

The pipeline itself never draws on the video (stages write tables, not pixels); this is only
for explaining and checking a stage. `toonball show STAGE RUN_DIR`.

- shots: shot number and wide / dropped label, a CUT flash on each cut, the two cut scores of
  io/shots.py (colour histogram, pixel difference) against their threshold, and a timeline
  strip of the whole video with the wide (kept) part of every shot.
- detect: every box of detect/run.py before tracking, coloured by class with its confidence;
  the ball the pipeline keeps, the 640 px tile(s) searched for it in that frame, a magnified
  ball inset, per-class counts, and a strip of the frames where the ball was found.
- track: one colour and label per id with a short trail of the feet; NEW ID, LOST (ByteTrack's
  Kalman guess while a track is unmatched, up to 2 s), FOUND AGAIN, GONE, ID CHANGE / DOUBLE
  (ByteTrack ids that track/players.py joins) and JOINED / MERGED markers; the id count and an
  id timeline.
  ids="raw" draws ByteTrack's ids, ids="final" the person tracks of tracks.parquet.
"""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import polars as pl

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


# ---- detect ---------------------------------------------------------------------------------

# class colours (BGR), picked to stay readable on grass and on both kits
CLS = {"player": (235, 190, 60),      # sky blue
       "goalkeeper": (0, 150, 255),   # orange
       "referee": (210, 70, 210),     # magenta
       "ball": (40, 40, 230)}         # red
SEARCH = (60, 230, 250)               # yellow: ball search tiles
TILE = 640


def _load_detect(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    n = len(np.load(run_dir / "keypoints.npy", mmap_mode="r"))
    ball = pl.read_parquet(run_dir / "ball_raw.parquet")
    if (run_dir / "detections.parquet").exists():
        dets = pl.read_parquet(run_dir / "detections.parquet")
    else:  # older runs: only the tracked boxes and the kept ball
        dets = pl.concat([
            pl.read_parquet(run_dir / "people.parquet").drop("track_id")
            .with_columns(pl.lit("full").alias("source")),
            ball.with_columns(pl.lit("ball").alias("cls"), pl.lit("?").alias("source"))
            .select("frame", "cls", "conf", "x1", "y1", "x2", "y2", "source")])
    search = (pl.read_parquet(run_dir / "ball_search.parquet")
              if (run_dir / "ball_search.parquet").exists() else None)
    by_frame: dict[int, list] = {}
    for r in dets.iter_rows(named=True):
        by_frame.setdefault(r["frame"], []).append(r)
    kept = {r["frame"]: r for r in ball.iter_rows(named=True)}
    found = np.zeros(n, bool)
    found[[f for f in kept if f < n]] = True
    from toonball.detect.run import _tile_origins
    w, h = meta["size"]
    return {"n": n, "fps": meta["fps"], "size": (w, h), "dets": by_frame, "ball": kept,
            "found": found, "tiles": meta.get("ball_tiles", True),
            "search": None if search is None else {r["frame"]: r for r in
                                                    search.iter_rows(named=True)},
            "grid": _tile_origins(w, h)}


def _dashed_rect(img, p1, p2, colour, thick=2, dash=18):
    (x1, y1), (x2, y2) = p1, p2
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)),
                 ((x1, y2), (x1, y1))):
        L = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        for t in range(0, L, 2 * dash):
            u, v = t / max(L, 1), min(t + dash, L) / max(L, 1)
            cv2.line(img, (int(a[0] + (b[0] - a[0]) * u), int(a[1] + (b[1] - a[1]) * u)),
                     (int(a[0] + (b[0] - a[0]) * v), int(a[1] + (b[1] - a[1]) * v)),
                     colour, thick, cv2.LINE_AA)


def _detect(img: np.ndarray, f: int, ctx: dict, opts: dict) -> None:
    W, H = OUT
    s = W / ctx["size"][0]
    fps, n = ctx["fps"], ctx["n"]
    clean = img.copy()
    rows = ctx["dets"].get(f, [])
    kept = ctx["ball"].get(f)
    sc = lambda r: tuple(int(round(v * s)) for v in (r["x1"], r["y1"], r["x2"], r["y2"]))

    # ball search: the tile(s) the ball model looked at in this frame
    srch = ctx["search"].get(f) if ctx["search"] else None
    if opts.get("tiles", True) and srch:
        mode = srch["mode"]
        if "grid" in mode:
            layer = img.copy()
            for x, y in ctx["grid"]:
                cv2.rectangle(layer, (int(x * s), int(y * s)),
                              (int((x + TILE) * s) - 1, int((y + TILE) * s) - 1), SEARCH, 2)
            cv2.addWeighted(layer, 0.45, img, 0.55, 0, img)
            why = "not in the tile" if mode == "near+grid" else "ball lost"
            _text(img, f"{why}: search all {len(ctx['grid'])} tiles", (W // 2 - 220, 300),
                  1.0, INK, 2, bg=SEARCH)
        if "near" in mode:
            x, y = srch["near_x"] * s, srch["near_y"] * s
            _dashed_rect(img, (int(x), int(y)), (int(x + TILE * s) - 1, int(y + TILE * s) - 1),
                         SEARCH, 3)
            _text(img, f"search tile {TILE} px", (int(x) + 14, int(y + TILE * s) - 16), 0.7, INK,
                  2, bg=SEARCH, pad=6)

    # people: one box per detection, label = class + confidence
    for r in rows:
        if r["cls"] == "ball":
            continue
        x1, y1, x2, y2 = sc(r)
        col = CLS[r["cls"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        lab = f"{r['cls']} {r['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(lab, FONT, 0.45, 1)
        cv2.rectangle(img, (x1 - 1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
        cv2.putText(img, lab, (x1 + 3, y1 - 5), FONT, 0.45, INK, 1, cv2.LINE_AA)

    # ball: every candidate thin, the one the pipeline keeps circled
    for r in rows:
        if r["cls"] == "ball" and not (kept and abs(r["x1"] - kept["x1"]) < 1e-3
                                       and abs(r["y1"] - kept["y1"]) < 1e-3):
            x1, y1, x2, y2 = sc(r)
            cv2.rectangle(img, (x1, y1), (x2, y2), CLS["ball"], 1, cv2.LINE_AA)
    if kept:
        x1, y1, x2, y2 = sc(kept)
        c = ((x1 + x2) // 2, (y1 + y2) // 2)
        rad = max(x2 - x1, y2 - y1) // 2 + 10
        cv2.circle(img, c, rad, CLS["ball"], 3, cv2.LINE_AA)
        _text(img, f"ball {kept['conf']:.2f}", (c[0] + rad + 6, c[1] + 6), 0.6, PAPER, 1,
              bg=CLS["ball"], pad=4)

    # top-left: stage, ball search mode, frame and per-class counts
    x = 40
    x += _text(img, "DETECTION", (x, 70), 1.3, PAPER, 2, bg=INK) + 40
    if not ctx["tiles"]:
        mode_lab = "BALL: WHOLE FRAME AT 1280 PX"
    elif srch is None:
        mode_lab = "BALL: 640 PX TILES"
    else:
        mode_lab = {"near": "BALL: 1 TILE AT LAST POSITION", "grid": "BALL LOST: ALL TILES",
                    "near+grid": "BALL MISSED: ALL TILES"}[srch["mode"]]
    _text(img, mode_lab, (x, 70), 1.0, INK, 2, bg=SEARCH if ctx["tiles"] else DROP)
    _text(img, f"frame {f:4d}   {f / fps:5.2f} s", (40, 130), 0.9, PAPER, 2, bg=INK, pad=8)
    cnt = {k: 0 for k in CLS}
    for r in rows:
        if r["cls"] != "ball":
            cnt[r["cls"]] += 1
    cnt["ball"] = int(kept is not None)
    x = 40
    for k, col in CLS.items():
        lab = f"{k} {cnt[k]}" if k != "ball" else ("ball found" if kept else "ball: none")
        (tw, th), _ = cv2.getTextSize(lab, FONT, 0.8, 2)
        cv2.rectangle(img, (x - 8, 160), (x + tw + 40, 200), PAPER, -1)
        cv2.rectangle(img, (x, 168), (x + 24, 192), col, -1)
        cv2.putText(img, lab, (x + 32, 190), FONT, 0.8, INK, 2, cv2.LINE_AA)
        x += tw + 64

    # bottom: strip of the frames with a ball, playhead
    x0, x1, y0, y1 = 40, W - 40, H - 52, H - 28
    px = lambda fr: int(x0 + (x1 - x0) * fr / max(n, 1))
    cv2.rectangle(img, (x0 - 12, H - 96), (x1 + 12, H - 16), PAPER, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), DROP, -1)
    fd = ctx["found"]
    for i in np.flatnonzero(fd):
        cv2.rectangle(img, (px(i), y0), (px(i + 1), y1), ACCENT, -1)
    pct = fd[:f + 1].mean() if f >= 0 else 0
    _text(img, f"ball found in {fd.mean():.0%} of frames  (so far {pct:.0%})", (x0, H - 64),
          0.7, INK, 2)
    _text(img, "green = ball found   grey = no ball", (x1 - 430, H - 64), 0.6, INK, 1)
    cv2.line(img, (px(f), y0 - 8), (px(f), y1 + 8), RED, 3)

    # bottom-right: the ball magnified (x4) from the clean frame
    if opts.get("zoom", True):
        S, Z = 80, 4
        bx0, by0 = W - 40 - S * Z, H - 110 - S * Z - 34
        cv2.rectangle(img, (bx0 - 6, by0 - 40), (bx0 + S * Z + 6, by0 + S * Z + 6), INK, -1)
        if kept:
            cx = int((kept["x1"] + kept["x2"]) / 2 * s)
            cy = int((kept["y1"] + kept["y2"]) / 2 * s)
            ax = int(np.clip(cx - S // 2, 0, W - S))
            ay = int(np.clip(cy - S // 2, 0, H - S))
            crop = cv2.resize(clean[ay:ay + S, ax:ax + S], (S * Z, S * Z),
                              interpolation=cv2.INTER_NEAREST)
            bw = (kept["x2"] - kept["x1"]) * s
            k1 = (int((kept["x1"] * s - ax) * Z), int((kept["y1"] * s - ay) * Z))
            k2 = (int((kept["x2"] * s - ax) * Z), int((kept["y2"] * s - ay) * Z))
            cv2.rectangle(crop, k1, k2, CLS["ball"], 2)
            img[by0:by0 + S * Z, bx0:bx0 + S * Z] = crop
            _text(img, f"BALL x{Z}   {bw:.0f} px wide", (bx0 + 4, by0 - 12), 0.6, PAPER, 1)
        else:
            cv2.rectangle(img, (bx0, by0), (bx0 + S * Z, by0 + S * Z), (60, 60, 60), -1)
            _text(img, "BALL x4", (bx0 + 4, by0 - 12), 0.6, PAPER, 1)
            _text(img, "NOT FOUND", (bx0 + 70, by0 + S * Z // 2 + 12), 1.1, PAPER, 2)


def detect_stats(run_dir: Path) -> dict:
    """Numbers for a detect run: detections per frame by class, mean confidence, ball found %,
    ball size, how the ball was searched."""
    ctx = _load_detect(run_dir)
    n = ctx["n"]
    dets = (pl.read_parquet(run_dir / "detections.parquet")
            if (run_dir / "detections.parquet").exists() else None)
    people = pl.read_parquet(run_dir / "people.parquet")
    ball = pl.read_parquet(run_dir / "ball_raw.parquet")
    out = {"run": run_dir.name, "frames": n, "ball_tiles": ctx["tiles"]}
    if dets is not None:
        out["detections_per_frame"] = {
            k: round(dets.filter(pl.col("cls") == k).height / n, 2) for k in CLS}
        out["detections_per_frame"]["ball_candidates"] = out["detections_per_frame"].pop("ball")
        out["mean_conf"] = {k: round(float(dets.filter(pl.col("cls") == k)["conf"].mean() or 0), 3)
                            for k in ("player", "goalkeeper", "referee")}
    out["tracked_boxes_per_frame"] = {
        k: round(people.filter(pl.col("cls") == k).height / n, 2)
        for k in ("player", "goalkeeper", "referee")}
    out.setdefault("mean_conf", {})["ball_kept"] = round(float(ball["conf"].mean() or 0), 3)
    fd = ctx["found"]
    gaps, run = [], 0
    for v in fd:
        run = 0 if v else run + 1
        gaps.append(run)
    out["ball_found_pct"] = round(100 * fd.mean(), 1)
    out["ball_longest_gap_frames"] = int(max(gaps) if gaps else 0)
    if ball.height:
        bw, bh = (ball["x2"] - ball["x1"]), (ball["y2"] - ball["y1"])
        out["ball_size_px"] = {"median_w": round(float(bw.median()), 1),
                               "median_h": round(float(bh.median()), 1),
                               "p10_w": round(float(bw.quantile(0.1)), 1),
                               "p90_w": round(float(bw.quantile(0.9)), 1),
                               "median_w_at_1280_input": round(float(bw.median()) * 1280
                                                               / ctx["size"][0], 1)}
    if ctx["search"]:
        modes = [r["mode"] for r in ctx["search"].values()]
        out["ball_search_frames"] = {m: modes.count(m) for m in sorted(set(modes))}
        if dets is not None:
            tb = (dets.filter(pl.col("cls") == "ball").sort("source", descending=True)
                  .unique(subset=["frame", "x1", "y1"], keep="first"))
            kept_src = ball.join(tb, on=["frame", "x1", "y1"], how="left")["source"]
            out["ball_kept_source"] = {str(k): int(v) for k, v in
                                       zip(*np.unique(kept_src.fill_null("?").to_numpy(),
                                                      return_counts=True))}
    return out


# ---- track ----------------------------------------------------------------------------------

LOST_C, NEW_C, JOIN_C = (40, 40, 220), (90, 200, 90), (60, 230, 250)  # red, green, yellow (BGR)
TRAIL = 30         # frames of trail behind each person (1.2 s at 25 fps)
EVENT_SHOW = 0.8   # s: NEW / ID CHANGE / FOUND / GONE chips
MIN_GAP = 3        # frames: shorter holes in a track get no FOUND AGAIN chip

# 16 colours that stay apart on grass and on both kits (BGR); ids step through them by 5 so
# neighbouring ids never share one
PALETTE = [(255, 255, 255), (0, 230, 255), (0, 128, 255), (40, 40, 230), (230, 60, 230),
           (255, 230, 0), (250, 170, 80), (220, 80, 20), (180, 40, 120), (190, 150, 255),
           (40, 80, 150), (25, 25, 25), (250, 190, 200), (60, 20, 140), (170, 220, 240),
           (160, 160, 0)]


def id_colour(i: int) -> tuple[int, int, int]:
    """Stable colour per id."""
    return PALETTE[(i * 5) % len(PALETTE)]


def _on(col) -> tuple[int, int, int]:
    """Readable text colour on a chip of colour col."""
    b, g, r = col
    return INK if 0.114 * b + 0.587 * g + 0.299 * r > 120 else PAPER


def _runs(frames: np.ndarray) -> list[tuple[int, int]]:
    """Sorted frame numbers → [(first, last)] of consecutive runs."""
    if not len(frames):
        return []
    br = np.flatnonzero(np.diff(frames) > 1)
    starts = np.r_[frames[0], frames[br + 1]]
    ends = np.r_[frames[br], frames[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _load_track(run_dir: Path, ids: str = "final") -> dict:
    """ids="raw": ByteTrack's ids as they come out of detect/run.py. ids="final": the person
    tracks of track/players.py (tracks.parquet), each labelled with its first ByteTrack id.
    ByteTrack is replayed on the cached detections for its lost tracks and Kalman guesses."""
    from toonball.track.replay import replay

    meta = json.loads((run_dir / "meta.json").read_text())
    n = len(np.load(run_dir / "keypoints.npy", mmap_mode="r"))
    fps = meta["fps"]
    people = pl.read_parquet(run_dir / "people.parquet").sort("frame", "track_id")
    tp = run_dir / "tracks.parquet"
    if ids == "final" and not tp.exists():
        raise FileNotFoundError(f"{tp} missing: run `toonball run CLIP --upto track` first")
    tmap = pl.read_parquet(tp) if tp.exists() else None
    agg = people.group_by("track_id").agg(pl.col("frame").min().alias("a"),
                                          pl.col("frame").max().alias("b"))
    raw_first = dict(zip(agg["track_id"].to_list(), agg["a"].to_list()))
    raw_last = dict(zip(agg["track_id"].to_list(), agg["b"].to_list()))
    raw_to_t, members = {}, {}
    if tmap is not None:
        for t, r in tmap.select("track_id", "raw_id").iter_rows():
            raw_to_t[int(r)] = int(t)
            members.setdefault(int(t), []).append(int(r))
        for rs in members.values():
            rs.sort(key=lambda r: raw_first[r])

    def disp(r: int) -> int | None:
        """Display id of a ByteTrack id: itself, or its person (keyed by the first raw id)."""
        if ids == "raw":
            return r
        t = raw_to_t.get(r)
        return None if t is None else members[t][0]

    by_frame: dict[int, list] = {}
    pos: dict[int, dict[int, tuple]] = {}
    for r in people.iter_rows(named=True):
        d = disp(int(r["track_id"]))
        box = (r["x1"], r["y1"], r["x2"], r["y2"])
        by_frame.setdefault(r["frame"], []).append((d, int(r["track_id"]), r["conf"], box))
        if d is not None:
            cur = pos.setdefault(d, {}).get(r["frame"])
            if cur is None or r["conf"] > cur[1]:
                pos[d][r["frame"]] = (box, r["conf"])

    # ByteTrack's lost tracks with their guessed box, by display id (a person seen again under
    # another id is no longer drawn as lost)
    rp = replay(run_dir, fps, n)
    lost: dict[int, dict[int, tuple]] = {}
    for f in range(n):
        out = {}
        for r, k in sorted(rp.lost.get(f, {}).items(), key=lambda x: (x[1], x[0])):
            d = disp(r)
            if d is None or f in pos.get(d, {}) or d in out or r not in rp.pred.get(f, {}):
                continue
            out[d] = (tuple(rp.pred[f][r]), k, r)
        lost[f] = out

    # joins of track/players.py. change: a ByteTrack id ends and the same person comes back
    # under a new one (_stitch); double: two ids on one person at the same time (_merge)
    joins = []
    for t, rs in members.items():
        for i, b in enumerate(rs[1:], 1):
            kind = "change" if raw_first[b] > max(raw_last[x] for x in rs[:i]) else "double"
            joins.append((raw_first[b], kind, rs[i - 1], b))
    join_of = {b: (k, a) for _, k, a, b in joins}

    events: list[tuple[int, str, int, str]] = []   # frame, kind, display id, text
    for d, fr in pos.items():
        runs = _runs(np.array(sorted(fr)))
        if runs[0][0] > 0:
            if ids == "raw" and d in join_of:
                k, a = join_of[d]
                txt = f"NEW ID #{d} (was #{a})" if k == "change" else f"DOUBLE: #{d} = #{a}"
                events.append((runs[0][0], k, d, txt))
            else:
                events.append((runs[0][0], "new", d, f"NEW ID #{d}"))
        for (_, e0), (s1, _) in itertools.pairwise(runs):
            if s1 - e0 - 1 < MIN_GAP:
                continue
            back = {x[1] for x in by_frame.get(s1, []) if x[0] == d}
            if ids == "final" and any(raw_first.get(r) == s1 and r != d for r in back):
                events.append((s1, "joined", d, f"SAME #{d}: IDS JOINED"))
            else:
                events.append((s1, "found", d, "FOUND AGAIN"))
    if ids == "final":
        events += [(f, "merged", disp(a), "DOUBLE MERGED") for f, k, a, _ in joins
                   if k == "double"]
    # gone: ByteTrack gives up on a lost track (2 s) and the person never comes back
    for f in range(1, n):
        for d, (box, _, r) in lost[f - 1].items():
            if d not in lost[f] and f not in pos.get(d, {}) and \
                    not any(g > f for g in pos.get(d, {})):
                events.append((f, "gone", d, f"GONE #{d}: lost 2 s"))
    # one chip at a time per person: drop an event that starts while the previous one shows
    kept, last_on = [], {}
    for e in sorted(events):
        if e[0] < last_on.get(e[2], -1e9) + EVENT_SHOW * fps:
            continue
        kept.append(e)
        last_on[e[2]] = e[0]
    events = kept
    first = {d: min(fr) for d, fr in pos.items()}
    order = sorted(first, key=lambda d: (first[d], d))
    return {"n": n, "fps": fps, "size": tuple(meta["size"]), "ids": ids,
            "stitch": meta.get("stitch", True), "by_frame": by_frame, "pos": pos,
            "lost": lost, "events": events, "joins": joins, "first": first, "order": order,
            "replay_same": rp.same, "replay": rp,
            "runs": {d: _runs(np.array(sorted(fr))) for d, fr in pos.items()}}


def _chip(img, s: str, org, colour, scale: float, text=None):
    text = _on(colour) if text is None else text
    t = max(1, int(round(1.6 * scale)))
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, t)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 1, y - th - int(10 * scale)), (x + tw + int(8 * scale), y), colour, -1)
    cv2.putText(img, s, (x + int(4 * scale), y - int(6 * scale)), FONT, scale, text, t,
                cv2.LINE_AA)


def _track(img: np.ndarray, f: int, ctx: dict, opts: dict) -> None:
    W, H = OUT
    s = W / ctx["size"][0]
    fps, n, k = ctx["fps"], ctx["n"], float(opts.get("label_scale", 1.0))
    th = max(2, int(round(2 * k)))
    sc = lambda b: tuple(int(round(v * s)) for v in b)
    pos = ctx["pos"]

    # trails: the feet (bottom centre) of the last TRAIL frames
    for d, fr in pos.items():
        if f not in fr:
            continue
        pts = [((b[0] + b[2]) / 2 * s, b[3] * s) for g in range(f - TRAIL, f + 1)
               if (b := fr.get(g, (None,))[0]) is not None]
        col = id_colour(d)
        for i in range(1, len(pts)):
            cv2.line(img, tuple(map(int, pts[i - 1])), tuple(map(int, pts[i])), col,
                     max(1, int(round((1 + 3 * i / len(pts)) * k))), cv2.LINE_AA)

    # lost tracks: ByteTrack's guess of where they are, kept for up to 2 s
    for d, (box, lf, r) in ctx["lost"].get(f, {}).items():
        x1, y1, x2, y2 = sc(box)
        _dashed_rect(img, (x1, y1), (x2, y2), LOST_C, th, 8)
        _chip(img, f"LOST #{d}  {lf / fps:.1f} s", (x1, y2 + int(22 * k)), LOST_C, 0.5 * k)

    # boxes: dropped tracklets grey, a second box of the same person thin, the rest by id
    rows = ctx["by_frame"].get(f, [])
    for d, r, conf, box in rows:
        x1, y1, x2, y2 = sc(box)
        if d is None:
            cv2.rectangle(img, (x1, y1), (x2, y2), DROP, 1, cv2.LINE_AA)
            continue
        col = id_colour(d)
        if pos[d][f][0] != box:
            _dashed_rect(img, (x1, y1), (x2, y2), col, 1, 6)
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), col, th, cv2.LINE_AA)
        _chip(img, f"#{d}", (x1, y1), col, 0.55 * k)

    # events around this frame
    for e0, kind, d, txt in ctx["events"]:
        if not e0 <= f < e0 + EVENT_SHOW * fps:
            continue
        if kind == "gone":
            last = ctx["lost"].get(e0 - 1, {}).get(d)
            if last is None:
                continue
            x1, y1, x2, y2 = sc(last[0])
            cv2.line(img, (x1, y1), (x2, y2), LOST_C, th + 1, cv2.LINE_AA)
            cv2.line(img, (x1, y2), (x2, y1), LOST_C, th + 1, cv2.LINE_AA)
            _chip(img, txt, (x1, y1 - 4), LOST_C, 0.55 * k)
            continue
        if f not in pos[d]:
            continue
        x1, y1, x2, y2 = sc(pos[d][f][0])
        col = {"new": NEW_C, "found": NEW_C, "change": LOST_C, "double": LOST_C,
               "joined": JOIN_C, "merged": JOIN_C}[kind]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rad = int(max(x2 - x1, y2 - y1) * 0.8 + 10 + 20 * (f - e0) / (EVENT_SHOW * fps))
        cv2.circle(img, (cx, cy), rad, col, th + 1, cv2.LINE_AA)
        _chip(img, txt, (x1, y1 - int(30 * k)), col, 0.6 * k)

    # top-left: stage and mode; top-right: the id count (scaled with the labels, up to 1.5)
    hk = min(k, 1.5)
    hy = lambda y: int(y * hk)
    raw = ctx["ids"] == "raw"
    mode = ("RAW BYTETRACK IDS" if raw else
            "AFTER STITCHING + MERGING" if ctx["stitch"] else "NO STITCHING (ABLATION)")
    x = 40
    x += _text(img, "TRACKING", (x, hy(70)), 1.3 * hk, PAPER, 2, bg=INK) + int(40 * hk)
    _text(img, mode, (x, hy(70)), 1.0 * hk, INK if raw or not ctx["stitch"] else PAPER, 2,
          bg=DROP if (raw or not ctx["stitch"]) else ACCENT)
    _text(img, f"frame {f:4d}   {f / fps:5.2f} s", (40, hy(130)), 0.9 * hk, PAPER, 2, bg=INK,
          pad=8)
    seen = sum(1 for d in ctx["first"] if ctx["first"][d] <= f)
    now = len({d for d, _, _, _ in rows if d is not None})
    nl = len(ctx["lost"].get(f, {}))
    nj = sum(1 for g, kk, _, _ in ctx["joins"] if g <= f and kk == "change")
    nd = sum(1 for g, kk, _, _ in ctx["joins"] if g <= f and kk == "double")
    if raw:
        line = f"on screen {now}   lost {nl}   id changes {nj}   double ids {nd}"
    elif ctx["stitch"]:
        line = f"on screen {now}   lost {nl}   ids joined {nj}   doubles merged {nd}"
    else:
        line = f"on screen {now}   lost {nl}   grey = dropped (too short / off the pitch)"
    _text(img, line, (40, hy(186)), 0.8 * hk, INK, 2, bg=PAPER, pad=8)
    lab = "IDS SO FAR" if (raw or not ctx["stitch"]) else "PEOPLE SO FAR"
    (tw, _), _ = cv2.getTextSize(lab, FONT, 0.8 * hk, 2)
    bx = W - 40 - max(tw, int(150 * hk))
    cv2.rectangle(img, (bx - 16, 28), (W - 24, hy(172)), INK, -1)
    _text(img, lab, (bx, hy(64)), 0.8 * hk, PAPER, 2)
    _text(img, str(seen), (bx, hy(152)), 2.6 * hk, PAPER, 5)

    # bottom: id timeline, one row per id in order of appearance
    if opts.get("timeline", True):
        order = ctx["order"]
        x0, x1, y1 = 40, W - 40, H - 22
        rh = max(2, min(8, 130 // max(len(order), 1)))
        y0 = y1 - rh * len(order)
        px = lambda fr_: int(x0 + (x1 - x0) * fr_ / max(n, 1))
        layer = img.copy()
        cv2.rectangle(layer, (x0 - 12, y0 - hy(40)), (x1 + 12, H - 10), INK, -1)
        cv2.addWeighted(layer, 0.75, img, 0.25, 0, img)
        _text(img, f"id timeline: one row per {'ByteTrack id' if raw else 'track'} "
                   f"({len(order)})", (x0, y0 - hy(14)), 0.6 * hk, PAPER, 1 if hk < 1.2 else 2)
        for i, d in enumerate(order):
            col = id_colour(d)
            for a, b in ctx["runs"][d]:
                cv2.rectangle(img, (px(a), y0 + i * rh), (max(px(b + 1) - 1, px(a)),
                              y0 + (i + 1) * rh - 1), col, -1)
        cv2.line(img, (px(f), y0 - 6), (px(f), y1 + 4), PAPER, 2)


def track_stats(run_dir: Path) -> dict:
    """ByteTrack ids vs person tracks, fragments per person, track lengths, lost tracks."""
    ctx = _load_track(run_dir, "final")
    raw = _load_track(run_dir, "raw")
    fps, n = ctx["fps"], ctx["n"]
    people = pl.read_parquet(run_dir / "people.parquet")
    tmap = pl.read_parquet(run_dir / "tracks.parquet")

    def lengths(runs: dict) -> np.ndarray:
        return np.array([r[-1][1] - r[0][0] + 1 for r in runs.values()])

    # lost episodes of ByteTrack: refound under the same id within 2 s, or given up
    lost_eps, refound, gone = [], 0, 0
    rl_ = raw["replay"].lost
    for f in range(n):
        for r, k in rl_.get(f, {}).items():
            if k == 1:
                lost_eps.append((r, f))
    for r, f0 in lost_eps:
        g = f0
        while g + 1 < n and r in rl_.get(g + 1, {}):
            g += 1
        if g + 1 < n and any(x[1] == r for x in raw["by_frame"].get(g + 1, [])):
            refound += 1
        elif g + 1 < n:
            gone += 1
    per = tmap.group_by("track_id").len()["len"].to_numpy()
    rl, fl = lengths(raw["runs"]), lengths(ctx["runs"])
    dropped = sorted(set(people["track_id"].unique().to_list()) - set(tmap["raw_id"].to_list()))
    joins = sorted(ctx["joins"])
    return {
        "run": run_dir.name, "frames": n, "fps": fps, "stitch": ctx["stitch"],
        "replay_identical_to_detect": ctx["replay_same"],
        "bytetrack_ids": len(raw["pos"]),
        "bytetrack_ids_by_class": {k: people.filter(pl.col("cls") == k)["track_id"].n_unique()
                                   for k in ("player", "goalkeeper", "referee")},
        "person_tracks": len(ctx["pos"]),
        "person_tracks_by_role": {k: tmap.filter(pl.col("role") == k)["track_id"].n_unique()
                                  for k in ("player", "goalkeeper", "referee")},
        "bytetrack_ids_dropped": len(dropped),
        "dropped_why": "shorter than 8 frames, a track under 0.6 s, or staff standing outside "
                       "the lines",
        "ids_joined_after_a_gap": sum(1 for j in joins if j[1] == "change"),
        "doubles_merged": sum(1 for j in joins if j[1] == "double"),
        "joins": [{"frame": f, "time_s": round(f / fps, 2), "kind": k, "old_id": a, "new_id": b,
                   "gap_frames": f - max(raw["runs"][a][-1][1], 0) - 1}
                  for f, k, a, b in joins],
        "bytetrack_ids_per_track": {"mean": round(float(per.mean()), 2), "max": int(per.max()),
                                    "tracks_with_more_than_one": int((per > 1).sum())},
        "people_boxes_per_frame": round(people.height / n, 2),
        "mean_track_length_s": {"bytetrack_id": round(float(rl.mean()) / fps, 2),
                                "track": round(float(fl.mean()) / fps, 2)},
        "median_track_length_s": {"bytetrack_id": round(float(np.median(rl)) / fps, 2),
                                  "track": round(float(np.median(fl)) / fps, 2)},
        "spanning_whole_clip": {"bytetrack_id": int((rl >= n - 1).sum()),
                                "track": int((fl >= n - 1).sum())},
        "bytetrack_lost": {"episodes": len(lost_eps), "refound_same_id": refound,
                           "given_up": gone, "still_lost_at_end": len(lost_eps) - refound - gone},
    }


STAGES = {"shots": _shots, "detect": _detect, "track": _track}


def render(stage: str, run_dir: Path, out: Path, start: float = 0.0, dur: float = 0.0,
           speed: float = 1.0, **opts) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; one of {', '.join(STAGES)}")
    if stage == "shots":
        info = json.loads((run_dir / "shots.json").read_text())
        video = _source(run_dir, info)
        n, fps = info["n_frames"], info["fps"]
        score = _shot_scores(video, n)
        draw = lambda img, f: _shots(img, f, info, score)
    elif stage == "track":
        ctx = _load_track(run_dir, opts.get("ids", "final"))
        video, n, fps = run_dir / "clip.mp4", ctx["n"], ctx["fps"]
        draw = lambda img, f: _track(img, f, ctx, opts)
    else:
        ctx = _load_detect(run_dir)
        video, n, fps = run_dir / "clip.mp4", ctx["n"], ctx["fps"]
        draw = lambda img, f: _detect(img, f, ctx, opts)
    a = int(start * fps)
    b = n if not dur else min(n, a + int(round(dur * fps)))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    w = _Writer(out, fps * speed)
    for f in range(a, b):
        ok, frame = cap.read()
        if not ok:
            break
        img = cv2.resize(frame, OUT, interpolation=cv2.INTER_AREA)
        draw(img, f)
        w.write(img)
    cap.release()
    w.close()
