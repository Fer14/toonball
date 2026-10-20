"""Full teams: every player exists for the whole clip, also while the camera doesn't see him.

The broadcast camera shows about a third of the pitch, so most players are out of shot most of
the time. Per team:

1. **Identities.** Track pieces of the same team are chained into at most 11 people: a piece
   continues the person whose predicted position is closest when it starts. (Pieces are
   already stitched across short gaps in players.py; this joins them across long ones.)
2. **Unseen stretches.** While out of shot a player moves with his team: he keeps his offset to
   the team's centre (measured from the visible players), and between two sightings the
   offset blends from where he left to where he reappears.
3. **Never in shot.** A player that wasn't detected can't be inside the camera's view, so an
   inferred position inside it is pushed just outside.
4. **Never seen.** Remaining places are filled from a 4-4-2 shape around the team's centre
   (a missing keeper stands in his goal and shadows the ball).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import savgol_filter

from toonball.calib import pitch
from toonball.track.players import Track, _view_polygons

TEAM_SIZE = 11
# 4-4-2 offsets from the team centre (metres): (towards the opponent's goal, across the pitch)
FORMATION = [(-14, -24), (-16, -8), (-16, 8), (-14, 24),
             (0, -22), (-2, -7), (-2, 7), (0, 22),
             (13, -7), (13, 7)]


def _span(t: Track) -> tuple[int, int]:
    seen = np.flatnonzero(t.alpha >= 0.999)
    return (t.start + int(seen[0]), t.start + int(seen[-1])) if len(seen) else (t.start, t.start)


def _at(t: Track, f: int) -> np.ndarray:
    i = int(np.clip(f - t.start, 0, len(t.x) - 1))
    return np.array([t.x[i], t.y[i]])


def _centres(tracks: list[Track], F: int) -> dict[int, np.ndarray]:
    """Team centre per frame from the visible outfield players (held when nobody is seen)."""
    out = {}
    for team in (0, 1):
        acc, n = np.zeros((F, 2)), np.zeros(F)
        for t in tracks:
            if t.team != team or t.role != "player":
                continue
            seen = t.alpha >= 0.999
            f = t.start + np.flatnonzero(seen)
            acc[f] += np.stack([t.x[seen], t.y[seen]], 1)
            n[f] += 1
        c = np.full((F, 2), np.nan)
        c[n > 0] = acc[n > 0] / n[n > 0, None]
        ok = np.flatnonzero(n > 0)
        if len(ok):
            for k in range(2):
                c[:, k] = np.interp(np.arange(F), ok, c[ok, k])
            win = min(F - (1 - F % 2), 51)
            if win >= 5:
                c = savgol_filter(c, win, 2, axis=0)
        else:
            c[:] = (pitch.LENGTH / 2, pitch.WIDTH / 2)
        out[team] = c
    return out


def _chain(pieces: list[Track], centre: np.ndarray, slots: int) -> list[list[Track]]:
    """Assign track pieces to at most `slots` people: a piece continues the person (free at its
    start) whose team-relative position predicts its start best."""
    people: list[list[Track]] = []
    for p in sorted(pieces, key=lambda t: _span(t)[0]):
        s0, _ = _span(p)
        best, best_d = None, np.inf
        for k, chain in enumerate(people):
            last = chain[-1]
            e = _span(last)[1]
            if e >= s0:
                # overlapping pieces are the same person only if they follow the same path
                if e - s0 > 50:
                    continue
                d = np.median([np.linalg.norm(_at(last, f) - _at(p, f)) for f in range(s0, e + 1)])
                if d > 1.5:
                    continue
            else:
                # where the last piece's player would be now, moving with the team
                pred = _at(last, e) + (centre[s0] - centre[e])
                d = np.linalg.norm(pred - _at(p, s0))
            if d < best_d:
                best, best_d = k, d
        if best is not None and (len(people) >= slots or best_d < 15.0):
            people[best].append(p)
        else:
            people.append([p])
    return people


def _outside_view(xy: np.ndarray, polys: list, margin: float = 2.0) -> np.ndarray:
    """Push inferred positions that fall inside the camera view just outside it."""
    out = xy.copy()
    for f, poly in enumerate(polys):
        if poly is None or not np.isfinite(out[f]).all():
            continue
        x, y = out[f]
        if cv2.pointPolygonTest(poly, (float(x), float(y)), True) < -margin:
            continue
        c = poly.mean(0)
        d = out[f] - c
        d = d / (np.linalg.norm(d) + 1e-6)
        for _ in range(80):  # walk outwards in 1 m steps
            x, y = out[f]
            if cv2.pointPolygonTest(poly, (float(x), float(y)), True) < -margin:
                break
            out[f] += d
    out[:, 0] = np.clip(out[:, 0], -3, pitch.LENGTH + 3)
    out[:, 1] = np.clip(out[:, 1], -3, pitch.WIDTH + 3)
    return out


def _person(chain: list[Track], centre: np.ndarray, F: int, polys: list,
            fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Whole-clip trajectory for one person: seen frames from the tracks, team-relative
    offsets in between / before / after."""
    xy = np.full((F, 2), np.nan)
    seen = np.zeros(F, bool)
    for t in chain:
        i = np.flatnonzero(t.alpha >= 0.999)
        f = t.start + i
        xy[f] = np.stack([t.x[i], t.y[i]], 1)
        seen[f] = True
    off = xy - centre
    v = np.flatnonzero(seen)
    for k in range(2):  # team-relative offset interpolated across gaps, held at the ends
        off[:, k] = np.interp(np.arange(F), v, off[v, k])
    inferred = centre + off
    inferred = np.where(seen[:, None], xy, _outside_view(inferred, polys))
    win = int(0.8 * fps) | 1
    if F > win:  # smooth only the inferred stretches' joins
        smooth = savgol_filter(inferred, win, 2, axis=0)
        inferred = np.where(seen[:, None], inferred, smooth)
    return inferred, seen


def _merge_pose(chain: list[Track], F: int) -> tuple[dict | None, dict | None]:
    pose = pose3d = None
    for t in chain:
        sl = slice(t.start, t.start + len(t.x))
        if t.pose:
            pose = pose or {k: np.full(F, np.nan) for k in t.pose}
            for k, v in t.pose.items():
                pose[k][sl] = v
        if t.pose3d:
            pose3d = pose3d or {"joints": np.full((F, 17, 3), np.nan), "conf": np.zeros(F)}
            pose3d["joints"][sl] = t.pose3d["joints"]
            pose3d["conf"][sl] = t.pose3d["conf"]
    if pose:
        for k in ("w", "wr"):
            if k in pose:
                pose[k] = np.nan_to_num(pose[k])
    return pose, pose3d


def fill(tracks: list[Track], H: np.ndarray, size: tuple[int, int], fps: float,
         ball_xy: tuple[np.ndarray, np.ndarray]) -> tuple[list[Track], dict[int, int]]:
    """Return whole-clip people (11 per team + referees as they were) and old id → new id."""
    F = len(H)
    polys = _view_polygons(H, size)
    centre = _centres(tracks, F)
    # which goal each team defends: its keeper's side, else the side its centre is nearer
    side = {}
    for team in (0, 1):
        gk = [t for t in tracks if t.team == team and t.role == "goalkeeper"]
        x = np.mean([t.x.mean() for t in gk]) if gk else np.nanmean(centre[team][:, 0])
        side[team] = 0 if x < pitch.LENGTH / 2 else 1
    if side[0] == side[1]:
        side[1] = 1 - side[0]

    out: list[Track] = []
    remap: dict[int, int] = {}

    def add(role, team, xy, seen, chain, colour=None):
        nid = len(out)
        speed = np.hypot(*np.gradient(xy, axis=0).T) * fps
        pose, pose3d = _merge_pose(chain, F) if chain else (None, None)
        alpha = np.ones(F)
        out.append(Track(nid, role, team, 0, xy[:, 0], xy[:, 1], np.minimum(speed, 10.0), alpha,
                         colour, [r for t in chain for r in t.raw_ids], pose, pose3d, seen))
        for t in chain:
            remap[t.id] = nid

    for team in (0, 1):
        towards = 1.0 if side[team] == 0 else -1.0   # +x attacks for the team defending x=0
        gks = [t for t in tracks if t.team == team and t.role == "goalkeeper"]
        outfield = [t for t in tracks if t.team == team and t.role == "player"]
        chains = _chain(outfield, centre[team], TEAM_SIZE - 1)
        for chain in chains:
            xy, seen = _person(chain, centre[team], F, polys, fps)
            add("player", team, xy, seen, chain)
        # never-seen outfield players: free 4-4-2 places, away from everyone we know about
        known = [o for o in out if o.team == team and o.role == "player"]
        for dx, dy in FORMATION:
            if len(known) >= TEAM_SIZE - 1:
                break
            slot = centre[team] + np.array([towards * dx, dy])
            slot[:, 1] = np.clip(slot[:, 1], 2, pitch.WIDTH - 2)
            near = min((np.nanmedian(np.hypot(o.x - slot[:, 0], o.y - slot[:, 1]))
                        for o in known), default=np.inf)
            if near < 8.0:
                continue
            xy = _outside_view(slot, polys)
            add("player", team, xy, np.zeros(F, bool), [])
            known.append(out[-1])
        # keeper: his track pieces, else standing in his goal shadowing the ball
        goal_x = 4.0 if side[team] == 0 else pitch.LENGTH - 4.0
        if gks:
            c = np.tile([goal_x, pitch.WIDTH / 2], (F, 1)).astype(float)
            xy, seen = _person(gks, c, F, polys, fps)
            add("goalkeeper", team, xy, seen, gks, gks[0].colour)
        else:
            by = np.nan_to_num(ball_xy[1], nan=pitch.WIDTH / 2)
            xy = np.stack([np.full(F, goal_x), pitch.WIDTH / 2 + 0.25 * (by - pitch.WIDTH / 2)], 1)
            add("goalkeeper", team, _outside_view(xy, polys), np.zeros(F, bool), [])

    for t in tracks:  # referees stay as they were
        if t.role == "referee":
            remap[t.id] = len(out)
            t.id = len(out)
            out.append(t)
    return out, remap
