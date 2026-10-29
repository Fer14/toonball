"""Players + ball + camera view → one World ready for the game renderer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from toonball.calib.homography import project
from toonball.track.ball import build_ball
from toonball.track.players import Track, build_tracks


@dataclass
class World:
    fps: float
    n_frames: int
    tracks: list[Track]
    ball_x: np.ndarray
    ball_y: np.ndarray
    ball_z: np.ndarray
    owner: np.ndarray                  # track id holding the ball, -1 none
    kicks: list[tuple[int, int]]       # (frame, track id)
    cam: np.ndarray                    # (F, 3) centre x, centre y, view width (m)
    calibrated: np.ndarray             # (F,) bool
    colours: list[str] = field(default_factory=list)
    flights: list[dict] = field(default_factory=list)
    hands: np.ndarray | None = None    # (F,) ball held in a keeper's hands


def _camera(H: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    w, h = size
    ok = ~np.isnan(H[:, 0, 0])
    cam = np.full((len(H), 3), np.nan)
    pts = np.array([[w / 2, 0.55 * h], [0, 0.55 * h], [w, 0.55 * h]])
    for f in np.flatnonzero(ok):
        p = project(H[f:f + 1], np.zeros(3, int), pts)
        cam[f] = (p[0, 0], p[0, 1], np.linalg.norm(p[2] - p[1]))
    if ok.any():
        v = np.flatnonzero(ok)
        for c in range(3):
            cam[:, c] = np.interp(np.arange(len(H)), v, cam[v, c])
    return cam, ok


def build(people: pl.DataFrame, teams: pl.DataFrame, ball: pl.DataFrame, H: np.ndarray,
          P: np.ndarray | None, fps: float, size: tuple[int, int], colours: list[str],
          colour_of: Callable[[list[int]], str | None] | None = None,
          stitch: bool = True) -> World:
    F = len(H)
    tracks = build_tracks(people, teams, H, size, fps, F, colour_of, stitch=stitch)
    b = build_ball(ball, H, P, tracks, fps, people)
    cam, ok = _camera(H, size)
    return World(fps, F, tracks, b.x, b.y, b.z, b.owner, b.kicks, cam, ok, colours, b.flights,
                 b.hands)


def to_frame(world: World) -> pl.DataFrame:
    rows = []
    for t in world.tracks:
        f = np.arange(t.start, t.start + len(t.x))
        rows.append(pl.DataFrame({
            "frame": f, "track_id": t.id, "cls": t.role, "team": t.team,
            "pitch_x": t.x, "pitch_y": t.y, "speed_ms": t.speed, "alpha": t.alpha,
            "has_ball": world.owner[f] == t.id,
        }))
    return pl.concat(rows) if rows else pl.DataFrame()
