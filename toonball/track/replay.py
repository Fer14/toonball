"""Re-run ByteTrack on the cached detections to see inside it (for `toonball show track`).

detect/run.py feeds the tracker the boxes it stores in detections.parquet, so feeding them
again in the same order gives the same ids (checked against people.parquet). On the way we keep
what the tracker knew but never wrote down: its Kalman guess of every track's box in each frame
before matching, and which tracks were lost (unmatched, kept for up to 2 s).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import supervision as sv

from toonball.detect.run import CLS_NAMES, make_tracker

CLS_ID = {v: k for k, v in CLS_NAMES.items()}


@dataclass
class Replay:
    pred: dict[int, dict[int, np.ndarray]]   # frame -> id -> Kalman guess x1 y1 x2 y2
    lost: dict[int, dict[int, int]]          # frame -> id -> frames lost so far
    same: bool                               # ids identical to people.parquet


def _tlbr(mean: np.ndarray) -> np.ndarray:
    cx, cy, a, h = mean[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def replay(run_dir: Path, fps: float, n: int) -> Replay:
    dets = pl.read_parquet(run_dir / "detections.parquet").filter(pl.col("cls") != "ball")
    people = pl.read_parquet(run_dir / "people.parquet")
    by_f = {f: g for (f,), g in dets.group_by("frame")}
    ref = {f: g for (f,), g in people.group_by("frame")}
    tr = make_tracker(fps)
    pred, lost, same = {}, {}, True
    for f in range(n):
        # the guess for frame f: each track's state after frame f-1, one Kalman step ahead
        g = {}
        for t in tr.tracked_tracks + tr.lost_tracks:
            if t.mean is None or not t.is_activated:
                continue
            m = t.mean.copy()
            if t.state != 1:  # TrackState.Tracked
                m[7] = 0
            m, _ = t.kalman_filter.predict(m, t.covariance)
            g[int(t.external_track_id)] = _tlbr(m)
        pred[f] = g
        d = by_f.get(f)
        det = sv.Detections.empty() if d is None else sv.Detections(
            xyxy=d.select("x1", "y1", "x2", "y2").to_numpy().astype(np.float32),
            confidence=d["conf"].to_numpy().astype(np.float32),
            class_id=np.array([CLS_ID[c] for c in d["cls"]]))
        out = tr.update_with_detections(det)
        lost[f] = {int(t.external_track_id): tr.frame_id - t.frame_id for t in tr.lost_tracks
                   if t.is_activated}
        r = ref.get(f)
        a = sorted(zip(np.round(out.xyxy[:, 0].astype(float), 1).tolist(),
                       out.tracker_id.tolist()))
        b = [] if r is None else sorted(zip(r["x1"].round(1).to_list(), r["track_id"].to_list()))
        same &= a == b
    return Replay(pred, lost, same)
