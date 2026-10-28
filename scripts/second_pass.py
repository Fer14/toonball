"""What ByteTrack's second (low-confidence) pass would add on a clip.

    uv run python scripts/second_pass.py runs/121364_0 OUT.json [--device cpu]

detect/run.py keeps player boxes from 0.3 up, and ByteTrack's split between "high" and "low"
boxes is also 0.3, so the second pass (unmatched tracks against boxes of 0.1-0.3) never gets a
box. This runs the player model again at 0.1 with the same NMS, then ByteTrack twice with
toonball's settings: on the boxes from 0.3 up (today; checked against people.parquet) and on
every box from 0.1 up (second pass active). The ball and pitch models are not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toonball.detect.run import make_tracker


def _track(frames: list[sv.Detections], fps: float, low: bool) -> dict:
    tr = make_tracker(fps)
    rows, low_used = [], 0
    for f, det in enumerate(frames):
        d = det if low else det[det.confidence >= 0.3]
        out = tr.update_with_detections(d)
        for b, c, t in zip(out.xyxy, out.confidence, out.tracker_id):
            rows.append((f, int(t), float(c), *map(float, b)))
            low_used += c < 0.3
    df = pl.DataFrame(rows, schema=["frame", "track_id", "conf", "x1", "y1", "x2", "y2"],
                      orient="row")
    gaps = 0
    for (_,), g in df.sort("frame").group_by("track_id"):
        gaps += int((np.diff(g["frame"].to_numpy()) > 1).sum())
    lens = df.group_by("track_id").len()["len"].to_numpy()
    return {"df": df, "ids": df["track_id"].n_unique(),
            "boxes_per_frame": round(df.height / len(frames), 2),
            "low_boxes_kept": int(low_used), "holes_inside_ids": gaps,
            "mean_boxes_per_id": round(float(lens.mean()), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    meta = json.loads((a.run / "meta.json").read_text())
    fps = meta["fps"]
    model = YOLO(str(ROOT / "weights" / "football-player-detection.pt"))
    frames, n_low = [], 0
    video = str(a.run / "clip.mp4")
    n = sv.VideoInfo.from_video_path(video).total_frames
    for frame in tqdm(sv.get_video_frames_generator(video), total=n, desc="detect 0.1"):
        r = model(frame, imgsz=1280, conf=0.1, verbose=False, half=a.device != "cpu",
                  device=a.device)[0]
        det = sv.Detections.from_ultralytics(r)
        det = det[det.class_id != 0].with_nms(threshold=0.5, class_agnostic=True)
        n_low += int((det.confidence < 0.3).sum())
        frames.append(det)
    today, both = _track(frames, fps, False), _track(frames, fps, True)
    ref = pl.read_parquet(a.run / "people.parquet")
    same = today["df"].height == ref.height and today["ids"] == ref["track_id"].n_unique()
    res = {"run": a.run.name, "frames": len(frames), "device": a.device,
           "low_boxes_per_frame_0.1_to_0.3": round(n_low / len(frames), 2),
           "today_boxes_from_0.3": {k: v for k, v in today.items() if k != "df"},
           "second_pass_boxes_from_0.1": {k: v for k, v in both.items() if k != "df"},
           "today_matches_people_parquet": same,
           "note": "ids = ByteTrack ids handed out; holes = places where an id is missing for "
                   "one or more frames and comes back; low_boxes_kept = boxes under 0.3 that "
                   "the second pass attached to a track"}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
