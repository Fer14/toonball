# toonball — plan

**Goal:** take a real football broadcast clip and re-render the match as a cartoon / video-game
world, in the spirit of:

- NFL *Toy Story Funday Football* (Disney+, Oct 2023) — live game re-skinned as Andy's room.
- Australian Open 2025 *Wii-Sports style* stream — Hawk-Eye tracking driving cartoon avatars.

Both are the same recipe: **tracking data → game engine → characters**. They had stadium-wide
multi-camera tracking; we have one broadcast camera, so we must recover the tracking from video.
That recovery is what futstats already started (detection, tracking, teams, homography);
toonball rebuilds it with better models and adds the 3D "game" renderer on top.

Hardware: RTX 3060 12 GB, 14 GB RAM, ~19 GB free disk.

---

## 1. Data

| Source | What | How we get it |
|---|---|---|
| DFL Bundesliga Data Shootout (Kaggle competition, the futstats clips) | 30 s Bundesliga broadcast clips, tactical wide camera | the 5 clips mirrored by [roboflow/sports](https://github.com/roboflow/sports) (`0bfacc_0`, `2e57b9_0`, `08fd33_0`, `573e61_0`, `121364_0`) via `gdown` — no Kaggle login needed. Full set via `kaggle competitions download` if needed later |
| [shreyamainkar/football-soccer-videos-dataset](https://www.kaggle.com/datasets/shreyamainkar/football-soccer-videos-dataset) | assorted broadcast clips (`106.mp4`, …) | public Kaggle API per-file download (`/api/v1/datasets/download/<ds>/<file>`) |

`scripts/get_data.sh` downloads clips to `clips/` and weights to `weights/` (both gitignored).
Disk is tight, so we download single clips, never the whole competition (~30 GB).

## 2. Pipeline

Entry point is **any video file**: `toonball make video.mp4`.

```
video.mp4
  └─[0] shots      cut detection (HSV histograms) + pitch keypoints every 3rd frame;
                   keep shots ≥3 s, ≥60 % calibratable, view width 20–110 m
                   (drops replays, close-ups, crowd, studio) → runs/<video>/shots/NN/clip.mp4
for each wide shot:
clip.mp4
  ├─[1] detect     players / GK / referee / ball         YOLOv8 football weights (roboflow/sports)
  ├─[2] track      persistent IDs                        supervision ByteTrack
  ├─[3] team       2 kits + GK + ref                     SigLIP → UMAP → KMeans, per-track vote
  ├─[4] calibrate  image → pitch metres                  32-keypoint pitch model → homography,
  │                                                      RANSAC + temporal smoothing + gating
  ├─[5] ball       small-object recovery                 slice inference + interpolation
  ├─[6] state      one tidy parquet table                frame, track_id, cls, team, pitch_x/y, …
  ├─[7] smooth     world-space motion                    per-track Savitzky-Golay, speed, heading
  └─[8] export     scene.json for the game renderer      players + ball per frame, kit colours
                         │
                         ▼
              [9] game renderer (Three.js, browser)
                   themes: "toy" (toy-room) and "wii" (Mii stadium)
                   procedural run cycle from speed, kick when the ball leaves a player
                   side-by-side with the original broadcast, synced
                         │
                         ▼
              [10] record → MP4 (headless Chromium, frame-stepped, ffmpeg)
```

Rule kept from the original plan: **stages write tables, not pixels.** Each stage caches to
`runs/<clip>/<stage>.parquet`; the renderer only reads `scene.json`.

### Model choices

| Role | v1 (ship) | v2 (upgrade) |
|---|---|---|
| Detection | roboflow/sports `football-player-detection.pt` (YOLOv8x, 4 classes) | RF-DETR fine-tune |
| Ball | `football-ball-detection.pt` + `sv.InferenceSlicer` | Kalman + 3D parabola fit |
| Pitch keypoints | `football-pitch-detection.pt` (YOLOv8-pose, 32 pts) | PnLCalib (full camera matrix → ball height) |
| Tracking | ByteTrack | BoT-SORT + ReID |
| Teams | SigLIP + UMAP + KMeans | — |

## 3. Single-camera problems and how they are handled

| Problem | Now | Next |
|---|---|---|
| Wrong pitch keypoints (mislabelled lines in mid-field, e.g. "halfway line" off screen) | per-frame keypoint fits fused with frame-to-frame camera motion (LK flow on the background, players masked): forward/backward filter, gated outlier fits, re-lock only on persistent agreement (`calib/homography.fuse`) | PnLCalib lines + points |
| Hidden cuts between two similar wide shots | cut = colour-histogram jump **or** pixel difference ≫ recent level (`io/shots._is_cut`) | — |
| Ball height (homography = ground only) | one fixed tripod per shot + per-frame rotation/zoom (`calib/camera`); image trajectory split at kinks; a piece is airborne when it bends off a straight image line and a gravity parabola explains it (≪ px error than rolling) — then extended to launch/landing. Skipped when the camera fit is poor (>30 px) | bounces, drag, spin |
| Ball occluded at feet / flicker | possession with hysteresis; owned ball sits at the owner's feet; gaps ≤2 s bridged; steps >40 m/s re-interpolated | — |
| Goalkeeper label flicker | roles decided per person: ≥30 % GK-labelled, stays within 30 m of a goal, one per goal at a time; kit colour measured from that person's crops | jersey/kit classifier |
| Track ID fragmentation, duplicates | tracklets stitched by reachable distance; overlapping tracklets on the same path merged | ReID across cuts |
| Jitter | constant-velocity Kalman + RTS smoother, noise ∝ metres-per-pixel at the player's position | — |
| Players off-screen (camera sees ~⅓ of pitch) | fade out at the view edge (no ghosts); renderer camera ≈ broadcast view | team-shape prior to place unseen players |
| Body pose (players 50–150 px tall) | procedural: gait cycle from real stride length, keeper stance / side-steps / dives, kick on possession change | pose estimation — see [FUTURE.md](FUTURE.md) |

## 4. Layout

```
toonball/
  scripts/get_data.sh        clips + weights
  toonball/
    io/        store.py, shots.py
    detect/    run.py (players + sliced ball + pitch keypoints)
    track/     players.py (stitch, roles, RTS smoothing), ball.py (possession, 3D flights), world.py
    team/      classifier.py
    calib/     pitch.py (105×68 model, 32 keypoints), homography.py, motion.py, camera.py
    render/    export.py (scene.json)
    cli.py     toonball make | run | record
  viewer/      index.html, game.js (Three.js, wii + toy themes)
  scripts/record.mjs         headless frame-stepped capture → MP4
  runs/<video>/shots.json, shots/NN/{clip.mp4,*.parquet,scene.json,pip_wii.mp4}, <video>_pip_wii.mp4
```

## 5. Phases

| # | Deliverable | Check |
|---|---|---|
| 0 | Clean repo, uv env, `get_data.sh`, clips + weights on disk | `ls clips weights` |
| 1 | detect + track → `track.parquet`, debug overlay video | boxes stick to players |
| 2 | pitch keypoints → homography → `pitch_x/y`, radar render | radar dots move plausibly, stay in 105×68 |
| 3 | team classifier → `team` column | radar colours match kits |
| 4 | ball slicing + interpolation, smoothing, speed, `scene.json` | ball present ≥90 % of calibrated frames |
| 5 | Three.js game viewer, "wii" theme, side-by-side with video | looks like the AO stream |
| 6 | "toy" theme (toy-room floor, plastic figures, giant furniture) | looks like Funday Football |
| 7 | headless recorder → `side_by_side.mp4` for every clip | shareable MP4 |
| 8 | `toonball make`: any video → shot split → per-shot pipeline → concatenated render | Kaggle highlights reel renders end-to-end |
| 9 | v2 upgrades from §2/§3 (3D ball, off-screen players, ReID across cuts) | — |

## 6. Done means

`scripts/get_data.sh && uv run toonball make clips/<any>.mp4` produces
`runs/<any>/<any>_pip_wii.mp4`: broadcast on one side, a cartoon match on the
other, with players in kit colours running where the real players run and the ball moving
between them.
