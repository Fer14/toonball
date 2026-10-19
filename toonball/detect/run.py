"""Single pass over the video: players (tracked), ball (sliced), pitch keypoints, crops.

Writes nothing itself — returns arrays for the store to cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import polars as pl
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

CLS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
TILE = 640
CROP = 64  # px, square crop of the shirt for team embedding / kit colour


@dataclass
class DetectOutput:
    people: pl.DataFrame      # frame, track_id, cls, conf, x1, y1, x2, y2
    ball: pl.DataFrame        # frame, conf, x1, y1, x2, y2 (best detection per frame)
    keypoints: np.ndarray     # (F, 32, 3) x, y, conf
    crops: np.ndarray         # (N, CROP, CROP, 3) RGB
    crop_index: pl.DataFrame  # row i of crops -> frame, track_id
    fps: float
    size: tuple[int, int]     # w, h


def _shirt_crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = box.astype(int)
    h, w = y2 - y1, x2 - x1
    if h < 20 or w < 8:
        return None
    # upper body only: skip head, shorts and grass
    sx1, sx2 = x1 + int(0.2 * w), x2 - int(0.2 * w)
    sy1, sy2 = y1 + int(0.15 * h), y1 + int(0.5 * h)
    c = frame[max(sy1, 0):sy2, max(sx1, 0):sx2]
    if c.size == 0:
        return None
    return cv2.cvtColor(cv2.resize(c, (CROP, CROP)), cv2.COLOR_BGR2RGB)


def _tile_origins(w: int, h: int, tile: int = TILE, overlap: int = 64) -> list[tuple[int, int]]:
    def starts(size: int) -> list[int]:
        n = max(1, int(np.ceil((size - overlap) / (tile - overlap))))
        return [round(i * (size - tile) / max(n - 1, 1)) for i in range(n)]
    return [(x, y) for y in starts(h) for x in starts(w)]


def _sliced_ball(model: YOLO, frame: np.ndarray, origins, conf: float) -> sv.Detections:
    """Small-object pass: all tiles in one batch, boxes shifted back to frame coords."""
    tiles = [frame[y:y + TILE, x:x + TILE] for x, y in origins]
    out = []
    for (x, y), r in zip(origins, model(tiles, imgsz=TILE, conf=conf, verbose=False, half=True)):
        d = sv.Detections.from_ultralytics(r)
        if len(d):
            d.xyxy += np.array([x, y, x, y], dtype=d.xyxy.dtype)
            out.append(d)
    return sv.Detections.merge(out) if out else sv.Detections.empty()


def run(
    video: str,
    player_w: str,
    ball_w: str,
    pitch_w: str,
    conf: float = 0.3,
    ball_conf: float = 0.2,
    crop_every: int = 3,
    max_frames: int | None = None,
) -> DetectOutput:
    info = sv.VideoInfo.from_video_path(video)
    n = info.total_frames if max_frames is None else min(info.total_frames, max_frames)
    players, balls, pitch = YOLO(player_w), YOLO(ball_w), YOLO(pitch_w)

    origins = _tile_origins(info.width, info.height)
    tracker = sv.ByteTrack(
        frame_rate=int(info.fps),
        track_activation_threshold=0.3,
        lost_track_buffer=int(info.fps * 2),
        minimum_matching_threshold=0.8,
    )

    rows, ball_rows, kps, crops, crop_rows = [], [], [], [], []
    last_ball_f, last_ball_xy = -100, (0.0, 0.0)
    frames = sv.get_video_frames_generator(video)
    for f, frame in enumerate(tqdm(frames, total=n, desc="detect")):
        if f >= n:
            break
        r = players(frame, imgsz=1280, conf=conf, verbose=False, half=True)[0]
        det = sv.Detections.from_ultralytics(r)
        people = det[det.class_id != 0]
        people = people.with_nms(threshold=0.5, class_agnostic=True)
        people = tracker.update_with_detections(people)
        for box, cid, c, tid in zip(people.xyxy, people.class_id, people.confidence,
                                    people.tracker_id):
            rows.append((f, int(tid), CLS_NAMES[int(cid)], float(c), *map(float, box)))
            if f % crop_every == 0 and cid in (1, 2):
                crop = _shirt_crop(frame, box)
                if crop is not None:
                    crops.append(crop)
                    crop_rows.append((f, int(tid)))

        # track the ball with one tile around its last position; full tiling only when lost
        if f - last_ball_f <= 5:
            cx, cy = last_ball_xy
            near = [(int(np.clip(cx - TILE / 2, 0, max(0, info.width - TILE))),
                     int(np.clip(cy - TILE / 2, 0, max(0, info.height - TILE))))]
            b = _sliced_ball(balls, frame, near, ball_conf)
            if not len(b):
                b = _sliced_ball(balls, frame, origins, ball_conf)
        else:
            b = _sliced_ball(balls, frame, origins, ball_conf)
        b_full = det[det.class_id == 0]
        if len(b_full):
            b = sv.Detections.merge([b, b_full]) if len(b) else b_full
        if len(b):
            i = int(np.argmax(b.confidence))
            ball_rows.append((f, float(b.confidence[i]), *map(float, b.xyxy[i])))
            last_ball_f = f
            last_ball_xy = (b.xyxy[i, 0] + b.xyxy[i, 2]) / 2, (b.xyxy[i, 1] + b.xyxy[i, 3]) / 2

        k = pitch(frame, imgsz=640, conf=0.3, verbose=False, half=True)[0]
        if k.keypoints is not None and len(k.keypoints.data):
            kps.append(k.keypoints.data[0].cpu().numpy())
        else:
            kps.append(np.zeros((32, 3), np.float32))

    box_cols = ["x1", "y1", "x2", "y2"]
    return DetectOutput(
        people=pl.DataFrame(rows, schema=["frame", "track_id", "cls", "conf", *box_cols],
                            orient="row"),
        ball=pl.DataFrame(ball_rows, schema=["frame", "conf", *box_cols], orient="row"),
        keypoints=np.stack(kps).astype(np.float32),
        crops=np.stack(crops) if crops else np.zeros((0, CROP, CROP, 3), np.uint8),
        crop_index=pl.DataFrame(crop_rows, schema=["frame", "track_id"], orient="row"),
        fps=float(info.fps),
        size=(info.width, info.height),
    )
