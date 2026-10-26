"""toonball make | run | record | show."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
MAX_W = 1920  # shots are re-encoded to at most this width
LIFTER = os.environ.get("TOONBALL_LIFTER", "motionagformer")  # or "videopose3d"


def _variant(name: str, flags: dict[str, bool]) -> str:
    """Ablations ("before" clips) write to runs/<clip>@<variant>/ so cached stages are kept."""
    on = [k for k, v in flags.items() if v]
    return f"{name}@{'+'.join(on)}" if on else name


def _ffmpeg_cut(src: Path, dst: Path, start_f: int | None = None, end_f: int | None = None,
                start_s: float = 0.0, dur_s: float = 0.0) -> None:
    vf = [f"scale='min({MAX_W},iw)':-2"]
    if start_f is not None:
        vf.insert(0, f"trim=start_frame={start_f}:end_frame={end_f},setpts=PTS-STARTPTS")
    cut = (["-ss", str(start_s)] if start_s else []) + (["-t", str(dur_s)] if dur_s else [])
    subprocess.run(["ffmpeg", "-v", "error", "-y", *cut, "-i", str(src), "-an", "-vf",
                    ",".join(vf), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", str(dst)], check=True)


def _pipeline(store, force: bool = False, pose_on: bool = True) -> dict:
    """clip.mp4 in a store dir → detections → teams → calibration → world → scene.json."""
    from toonball.calib import camera, homography, motion
    from toonball.detect import run as detect
    from toonball.render.export import export
    from toonball.team import classifier
    from toonball.track import world

    video = store.path("clip.mp4")
    if force or not store.has("people.parquet", "keypoints.npy"):
        d = detect.run(str(video), str(WEIGHTS / "football-player-detection.pt"),
                       str(WEIGHTS / "football-ball-detection.pt"),
                       str(WEIGHTS / "football-pitch-detection.pt"))
        store.put_df("people", d.people)
        store.put_df("ball_raw", d.ball)
        store.put_np("keypoints", d.keypoints)
        store.put_np("crops", d.crops)
        store.put_df("crop_index", d.crop_index)
        store.put_json("meta", {"fps": d.fps, "size": list(d.size)})
    meta = store.json("meta")
    fps, size = meta["fps"], tuple(meta["size"])

    if force or not store.has("teams.parquet", "colours.json"):
        teams, colours = classifier.assign(store.np("crops"), store.df("crop_index"))
        store.put_df("teams", teams)
        store.put_json("colours", colours)

    if force or not store.has("motion.npz"):
        R, R_ok = motion.estimate(str(video), store.df("people"))
        np.savez(store.path("motion.npz"), R=R, ok=R_ok)
    mo = np.load(store.path("motion.npz"))
    H = homography.fuse(homography.fit_raw(store.np("keypoints")), mo["R"], mo["ok"], size, fps)
    store.put_np("homography", H)

    # full camera (for ball height) only when one tripod position explains the homographies
    cams = camera.estimate(H, size, fps)
    err = camera.grid_error(cams, H, size)
    P = cams.P if err < 30 else None
    store.put_json("camera", {"centre_m": np.round(cams.centre, 2).tolist(),
                              "grid_error_px": round(err, 1), "used_for_ball": P is not None})

    crops, crop_index = store.np("crops"), store.df("crop_index")

    def colour_of(track_ids: list[int]) -> str | None:
        m = crop_index["track_id"].is_in(track_ids).to_numpy()
        return classifier.kit_colour(crops[m]) if m.sum() >= 3 else None

    w = world.build(store.df("people"), store.df("teams"), store.df("ball_raw"), H, P, fps,
                    size, store.json("colours"), colour_of)

    if pose_on:
        from toonball.pose import angles, estimate
        if force or not store.has("pose.parquet"):
            estimate.run(str(video), store.df("people"),
                         str(WEIGHTS / "yolo11m-pose.pt")).write_parquet(store.path("pose.parquet"))
        kps = angles.index(pl.read_parquet(store.path("pose.parquet")))
        for t in w.tracks:
            t.pose = angles.track_angles(t, kps, H, fps, w.ball_x, w.ball_y)
        # full-body 3D pose: lift each person's 2D sequence, rotate into pitch axes with the
        # broadcast camera (its rotation is usable even when the tripod fit is too loose for
        # ball heights)
        if np.isfinite(cams.P[:, 0, 0]).any():
            import torch

            from toonball.pose import lift
            device = "cuda" if torch.cuda.is_available() else "cpu"
            lifter = lift.load(LIFTER, WEIGHTS, device)
            R_cam = lift.camera_rotations(cams.P, size)
            for t in w.tracks:
                t.pose3d = lift.lift_track(lifter, t, kps, R_cam, fps)

    # full teams: every player exists for the whole clip, also while off camera
    from toonball.track import offscreen
    w.tracks, remap = offscreen.fill(w.tracks, H, size, fps, (w.ball_x, w.ball_y))
    w.owner = np.array([remap.get(int(o), -1) if o >= 0 else -1 for o in w.owner])
    w.kicks = [(f, remap[i]) for f, i in w.kicks if i in remap]
    store.put_df("state", world.to_frame(w))
    store.put_json("flights", w.flights)
    export(w, store.path("scene.json"), "clip.mp4")
    ball_ok = float(np.isfinite(w.ball_x[w.calibrated]).mean()) if w.calibrated.any() else 0.0
    gks = sum(t.role == "goalkeeper" for t in w.tracks)
    per_team = [sum(t.team == k and t.role != "referee" for t in w.tracks) for k in (0, 1)]
    summary = {"calibrated": float(np.isfinite(H[:, 0, 0]).mean()), "tracks": len(w.tracks),
               "ball": ball_ok, "kicks": len(w.kicks), "flights": len(w.flights)}
    typer.echo(f"  {store.dir.name}: calibrated {summary['calibrated']:.0%}  tracks "
               f"{summary['tracks']} (gk {gks})  ball {summary['ball']:.0%}  kicks "
               f"{summary['kicks']}  3D flights {len(w.flights)}  camera err {err:.0f}px  "
               f"team sizes {per_team[0]}+{per_team[1]}")
    return summary


def _align_teams(scene_path: Path, ref: list[str] | None) -> list[str]:
    """Clustering labels teams arbitrarily per shot; keep HOME/AWAY = first shot's kits."""
    import json

    scene = json.loads(scene_path.read_text())
    cols = scene["teams"]
    if ref is None:
        return cols
    rgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)
    same = np.linalg.norm(rgb(cols[0]) - rgb(ref[0])) + np.linalg.norm(rgb(cols[1]) - rgb(ref[1]))
    swap = np.linalg.norm(rgb(cols[0]) - rgb(ref[1])) + np.linalg.norm(rgb(cols[1]) - rgb(ref[0]))
    if swap < same:
        scene["teams"] = cols[::-1]
        for p in scene["players"]:
            if p["team"] in (0, 1):
                p["team"] = 1 - p["team"]
        scene_path.write_text(json.dumps(scene, separators=(",", ":")))
    return ref


def _record(run_dir: Path, theme: str, layout: str) -> Path:
    subprocess.run(["node", str(ROOT / "scripts" / "record.mjs"), str(run_dir.resolve()),
                    theme, layout], check=True, cwd=ROOT)
    return run_dir / f"{layout}_{theme}.mp4"


@app.command()
def make(
    video: Path,
    theme: str = typer.Option("wii", help="wii | toy"),
    layout: str = typer.Option("pip", help="pip | side | game"),
    min_shot: float = typer.Option(3.0, help="shortest wide shot kept (s)"),
    force: bool = typer.Option(False, help="recompute cached stages"),
    record: bool = typer.Option(True, help="render MP4s (off: only scene.json for the viewer)"),
    no_shots: bool = typer.Option(False, help="ablation: the whole video is one shot"),
):
    """Any football video → cartoon video. Cuts, replays and close-ups are skipped."""
    from toonball.io import shots
    from toonball.io.store import Store

    top = Store(_variant(video.stem, {"no-shots": no_shots}), ROOT / "runs")
    if no_shots and (force or not top.has("shots.json")):
        import cv2
        cap = cv2.VideoCapture(str(video))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        cap.release()
        top.put_json("shots", {"video": str(video), "fps": fps, "n_frames": n, "size": size,
                               "cuts": [0, n], "wide": [{"start": 0, "end": n}]})
    elif force or not top.has("shots.json"):
        cuts, kps, sampled, n, fps, size = shots.prepass(
            str(video), str(WEIGHTS / "football-pitch-detection.pt"))
        wide = shots.wide_shots(cuts, kps, sampled, fps, size, min_s=min_shot)
        top.put_json("shots", {"video": str(video), "fps": fps, "n_frames": n,
                               "size": list(size), "cuts": cuts, "wide": wide})
    info = top.json("shots")
    wide = info["wide"]
    kept = sum(s["end"] - s["start"] for s in wide) / info["fps"]
    typer.echo(f"{len(info['cuts']) - 1} shots, {len(wide)} wide "
               f"({kept:.0f}s of {info['n_frames'] / info['fps']:.0f}s)")
    if not wide:
        raise typer.Exit("no wide, calibratable shot found — is the pitch visible?")

    parts, ref = [], None
    for k, s in enumerate(wide):
        st = Store(f"{k:02d}", top.dir / "shots")
        if force or not st.has("clip.mp4"):
            _ffmpeg_cut(video, st.path("clip.mp4"), s["start"], s["end"])
        st.put_json("source", {"video": str(video), **s})
        _pipeline(st, force)
        ref = _align_teams(st.path("scene.json"), ref)
        if record:
            parts.append(_record(st.dir, theme, layout))
    if not record:
        typer.echo(f"viewer: /viewer/index.html?run=/runs/{top.dir.name}/shots/NN")
        return

    out = top.dir / f"{top.dir.name}_{layout}_{theme}.mp4"
    lst = top.path("concat.txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)
    typer.echo(f"wrote {out}")


@app.command()
def run(
    clip: Path,
    start: float = typer.Option(0.0, help="start time (s); clip is cut to [start, start+dur]"),
    dur: float = typer.Option(0.0, help="duration (s), 0 = whole clip"),
    force: bool = typer.Option(False, help="recompute cached stages"),
):
    """Single continuous shot → scene.json (no cut detection)."""
    from toonball.io.store import Store

    store = Store(clip.stem if not (start or dur) else f"{clip.stem}_{int(start)}s",
                  ROOT / "runs")
    if not store.path("clip.mp4").exists():
        _ffmpeg_cut(clip, store.path("clip.mp4"), start_s=start, dur_s=dur)
    _pipeline(store, force)


@app.command()
def record(
    run_dir: Path,
    theme: str = typer.Option("wii", help="wii | toy"),
    layout: str = typer.Option("pip", help="pip | side | game"),
):
    """Render a run's scene.json in headless Chromium and mux with the source footage."""
    typer.echo(f"wrote {_record(run_dir, theme, layout)}")


@app.command()
def show(
    stage: str,
    run_dir: Path,
    start: float = typer.Option(0.0, help="start time (s)"),
    dur: float = typer.Option(0.0, help="duration (s), 0 = to the end"),
    speed: float = typer.Option(1.0, help="playback speed, e.g. 0.5 for slow motion"),
    out: str = typer.Option("", help="output MP4 (default RUN_DIR/show_<STAGE>.mp4)"),
):
    """Debug overlay of a stage on the real clip → 1920×1080 MP4. STAGE: shots (needs the
    shots.json of `toonball make`)."""
    from toonball.render import debug

    out = Path(out) if out else run_dir / f"show_{stage}.mp4"
    debug.render(stage, run_dir, out, start, dur, speed)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
