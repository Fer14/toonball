"""Image tracklets → one smooth pitch trajectory per real person.

- feet projected to metres, with a per-detection noise that grows with distance from the camera
- tracklets stitched across occlusions / leaving the frame; duplicate tracklets of the same
  person (ByteTrack double tracks) merged
- roles decided per person, not per frame: at most one goalkeeper per goal, near that goal
- constant-velocity Kalman + RTS smoother with human acceleration limits
- fade out when a person leaves the camera view instead of leaving ghosts behind
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
import polars as pl

from toonball.calib import pitch
from toonball.calib.homography import project


@dataclass
class Track:
    id: int
    role: str          # player | goalkeeper | referee
    team: int          # 0 | 1 | -1 referee / unknown
    start: int         # first frame of the arrays below
    x: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    alpha: np.ndarray  # 1 = seen, fading towards 0 around entries / exits
    colour: str | None = None
    raw_ids: list[int] = field(default_factory=list)


def _on_pitch(xy: np.ndarray, margin: float) -> np.ndarray:
    return ((xy[:, 0] > -margin) & (xy[:, 0] < pitch.LENGTH + margin)
            & (xy[:, 1] > -margin) & (xy[:, 1] < pitch.WIDTH + margin))


def project_people(people: pl.DataFrame, H: np.ndarray) -> pl.DataFrame:
    frames = people["frame"].to_numpy()
    u = (people["x1"] + people["x2"]).to_numpy() / 2
    v = people["y2"].to_numpy()
    xy = project(H, frames, np.stack([u, v], 1))
    # box bottoms wobble by a few pixels; a pixel is worth more metres far from the camera
    xy1 = project(H, frames, np.stack([u, v + 1], 1))
    m_per_px = np.linalg.norm(xy1 - xy, axis=1)
    sigma = np.hypot(0.25, 3.0 * m_per_px)
    df = people.with_columns(pl.Series("px", xy[:, 0]), pl.Series("py", xy[:, 1]),
                             pl.Series("sig", sigma))
    return df.filter(pl.col("px").is_not_nan() & pl.Series(_on_pitch(xy, 4.0)))


def _segments(df: pl.DataFrame, teams: pl.DataFrame, min_len: int) -> list[dict]:
    df = df.join(teams.select("track_id", "team"), on="track_id", how="left")
    segs = []
    for (tid,), g in df.sort("frame").group_by("track_id", maintain_order=True):
        if g.height < min_len:
            continue
        cls = g["cls"].to_numpy()
        segs.append({
            "ids": [int(tid)],
            "f": g["frame"].to_numpy(), "x": g["px"].to_numpy(), "y": g["py"].to_numpy(),
            "sig": g["sig"].to_numpy(),
            "gk": int((cls == "goalkeeper").sum()), "ref": int((cls == "referee").sum()),
            "n": g.height,
            "team": None if g["team"][0] is None else int(g["team"][0]),
        })
    return segs


def _is_ref(s: dict) -> bool:
    return s["ref"] > 0.5 * s["n"]


def _compatible(a: dict, b: dict) -> bool:
    if _is_ref(a) != _is_ref(b):
        return False
    gk_like = a["gk"] > 0.3 * a["n"] or b["gk"] > 0.3 * b["n"]  # GK kits cluster unreliably
    return gk_like or a["team"] is None or b["team"] is None or a["team"] == b["team"]


def _merge(a: dict, b: dict) -> dict:
    f = np.concatenate([a["f"], b["f"]])
    x = np.concatenate([a["x"], b["x"]])
    y = np.concatenate([a["y"], b["y"]])
    sig = np.concatenate([a["sig"], b["sig"]])
    uf, inv = np.unique(f, return_inverse=True)
    w = 1 / sig ** 2
    xs = np.bincount(inv, w * x) / np.bincount(inv, w)
    ys = np.bincount(inv, w * y) / np.bincount(inv, w)
    ss = 1 / np.sqrt(np.bincount(inv, w))
    team = a["team"] if a["team"] is not None else b["team"]
    if a["team"] is not None and b["team"] is not None and a["n"] < b["n"]:
        team = b["team"]
    return {"ids": a["ids"] + b["ids"], "f": uf, "x": xs, "y": ys, "sig": ss,
            "gk": a["gk"] + b["gk"], "ref": a["ref"] + b["ref"], "n": a["n"] + b["n"],
            "team": team}


def _stitch(segs: list[dict], fps: float, max_gap_s: float = 3.0,
            max_overlap_s: float = 0.6) -> list[dict]:
    """Greedy chaining by start time. A segment continues a chain if the person could have run
    there in the gap, or — for overlapping segments — if both follow the same path (duplicate)."""
    chains: list[dict] = []
    for s in sorted(segs, key=lambda s: s["f"][0]):
        best, best_cost = None, np.inf
        for i, c in enumerate(chains):
            if not _compatible(c, s):
                continue
            gap = s["f"][0] - c["f"][-1]
            if gap > max_gap_s * fps:
                continue
            if gap > 0:
                d = np.hypot(s["x"][0] - c["x"][-1], s["y"][0] - c["y"][-1])
                if d > 2.5 + 7.0 * gap / fps:
                    continue
                cost = d + gap / fps
            else:
                if -gap > max_overlap_s * fps:
                    continue
                common, ia, ib = np.intersect1d(c["f"], s["f"], return_indices=True)
                if len(common) == 0:
                    continue
                d = np.median(np.hypot(c["x"][ia] - s["x"][ib], c["y"][ia] - s["y"][ib]))
                if d > 1.2:
                    continue
                cost = d
            if cost < best_cost:
                best, best_cost = i, cost
        if best is None:
            chains.append(s)
        else:
            chains[best] = _merge(chains[best], s)
    return chains


def _rts(f: np.ndarray, x: np.ndarray, y: np.ndarray, sig: np.ndarray, fps: float,
         acc: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant-velocity Kalman filter + Rauch–Tung–Striebel smoother over the dense frame range
    (gaps are bridged by the motion model)."""
    n = f[-1] - f[0] + 1
    dt = 1.0 / fps
    A = np.array([[1, dt], [0, 1]])
    Q = acc ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    z = np.full((n, 2), np.nan)
    r = np.full(n, np.nan)
    z[f - f[0]] = np.stack([x, y], 1)
    r[f - f[0]] = sig ** 2
    out = []
    for ax in range(2):
        m = np.array([z[0, ax], 0.0])
        P = np.diag([r[0], 4.0])
        ms, Ps, mp, Pp = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2)), np.zeros((n, 2, 2))
        for t in range(n):
            if t:
                m, P = A @ m, A @ P @ A.T + Q
            mp[t], Pp[t] = m, P
            if np.isfinite(z[t, ax]):
                k = P[:, 0] / (P[0, 0] + r[t])
                m = m + k * (z[t, ax] - m[0])
                P = P - np.outer(k, P[0])
            ms[t], Ps[t] = m, P
        for t in range(n - 2, -1, -1):
            C = Ps[t] @ A.T @ np.linalg.inv(Pp[t + 1])
            ms[t] = ms[t] + C @ (ms[t + 1] - mp[t + 1])
            Ps[t] = Ps[t] + C @ (Ps[t + 1] - Pp[t + 1]) @ C.T
        out.append(ms)
    px, py = out[0][:, 0], out[1][:, 0]
    speed = np.hypot(out[0][:, 1], out[1][:, 1])
    return px, py, np.minimum(speed, 10.0)


def _view_polygons(H: np.ndarray, size: tuple[int, int]) -> list[np.ndarray | None]:
    w, h = size
    corners = np.array([[0, 0.12 * h], [w, 0.12 * h], [w, h], [0, h]], np.float64)
    polys = []
    for t in range(len(H)):
        polys.append(None if not np.isfinite(H[t, 0, 0])
                     else cv2.perspectiveTransform(corners[None], H[t])[0].astype(np.float32))
    return polys


def _near_edge(poly: np.ndarray | None, x: float, y: float, margin: float = 4.0) -> bool:
    if poly is None:
        return True
    return cv2.pointPolygonTest(poly, (float(x), float(y)), True) < margin


def build_tracks(people: pl.DataFrame, teams: pl.DataFrame, H: np.ndarray,
                 size: tuple[int, int], fps: float, n_frames: int,
                 colour_of: Callable[[list[int]], str | None] | None = None,
                 min_len: int = 8) -> list[Track]:
    df = project_people(people, H)
    chains = _stitch(_segments(df, teams, min_len), fps)
    chains = [c for c in chains if len(c["f"]) >= int(0.6 * fps)]  # drop flickers

    # team sides: which team stands closer to each goal on average
    team_x = {}
    for t in (0, 1):
        xs = [c["x"].mean() for c in chains if c["team"] == t and c["gk"] < 0.3 * c["n"]]
        team_x[t] = float(np.mean(xs)) if xs else pitch.LENGTH / 2
    left_team = 0 if team_x[0] < team_x[1] else 1

    # goalkeepers: per goal, the strongest GK-labelled chain(s) that stay near that goal and
    # don't overlap in time; every other "goalkeeper" tracklet is an outfield player
    roles = ["referee" if _is_ref(c) else "player" for c in chains]
    for side in (0, 1):
        cand = [i for i, c in enumerate(chains)
                if roles[i] == "player" and c["gk"] >= 0.3 * c["n"]
                and (np.median(c["x"]) < 30 if side == 0 else np.median(c["x"]) > pitch.LENGTH - 30)]
        taken: list[tuple[int, int]] = []
        for i in sorted(cand, key=lambda i: -chains[i]["gk"]):
            a, b = chains[i]["f"][0], chains[i]["f"][-1]
            if all(b < s + fps * 0.5 or a > e - fps * 0.5 for s, e in taken):
                roles[i] = "goalkeeper"
                chains[i]["team"] = left_team if side == 0 else 1 - left_team
                taken.append((a, b))

    polys = _view_polygons(H, size)
    out = []
    for k, c in enumerate(chains):
        x, y, speed = _rts(c["f"], c["x"], c["y"], c["sig"], fps)
        start, end = int(c["f"][0]), int(c["f"][-1])
        # fade in/out: slowly at the edge of the camera view (they walked out of shot),
        # quickly when lost in the middle of the view (occlusion we failed to bridge)
        fin = int((0.5 if _near_edge(polys[start], x[0], y[0]) else 0.25) * fps)
        fout = int((0.8 if _near_edge(polys[end], x[-1], y[-1]) else 0.25) * fps)
        pre, post = min(fin, start), min(fout, n_frames - 1 - end)
        x = np.concatenate([np.full(pre, x[0]), x, np.full(post, x[-1])])
        y = np.concatenate([np.full(pre, y[0]), y, np.full(post, y[-1])])
        speed = np.concatenate([np.zeros(pre), speed, np.zeros(post)])
        alpha = np.concatenate([np.linspace(0, 1, pre + 2)[1:-1], np.ones(end - start + 1),
                                np.linspace(1, 0, post + 2)[1:-1]])
        role = roles[k]
        team = -1 if role == "referee" else (-1 if c["team"] is None else int(c["team"]))
        colour = colour_of(c["ids"]) if (colour_of and role == "goalkeeper") else None
        out.append(Track(k, role, team, start - pre, x, y, speed, alpha, colour, c["ids"]))
    return out
