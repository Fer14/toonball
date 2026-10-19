"""World → scene.json consumed by the Three.js game viewer (viewer/)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from toonball.calib import pitch
from toonball.track.world import World


def _r(a: np.ndarray, nd: int = 2) -> list:
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


def export(world: World, out: Path, video: str) -> Path:
    scene = {
        "fps": world.fps,
        "n_frames": world.n_frames,
        "video": video,
        "pitch": {"length": pitch.LENGTH, "width": pitch.WIDTH},
        "teams": world.colours,
        "camera": {
            "x": _r(world.cam[:, 0], 1), "y": _r(world.cam[:, 1], 1),
            "w": _r(world.cam[:, 2], 1),
        },
        "calibrated": world.calibrated.astype(int).tolist(),
        "players": [
            {
                "id": t.id, "role": t.role, "team": t.team, "start": t.start,
                "x": _r(t.x), "y": _r(t.y), "s": _r(t.speed, 1), "a": _r(t.alpha),
                **({"colour": t.colour} if t.colour else {}),
            }
            for t in world.tracks
        ],
        "ball": {"x": _r(world.ball_x), "y": _r(world.ball_y), "z": _r(world.ball_z)},
        "owner": world.owner.tolist(),
        "kicks": world.kicks,
    }
    out.write_text(json.dumps(scene, separators=(",", ":")))
    return out
