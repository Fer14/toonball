# toonball — ideas to consider later

## Known issues to follow up (noted 2026-10-20)

Measured with the motion diagnostics after the calibration / goalkeeper / ball rewrite.

- [ ] **No ball height on the Kaggle video** (`clips/kaggle_106.mp4`). Its frames don't fit a
  single fixed camera well (47–572 px of grid error, against 17–30 px for the Bundesliga clips;
  see `runs/kaggle_106/shots/NN/camera.json`). Rather than invent heights, the ball stays on the
  ground there (`P = None` when error > 30 px in `toonball/cli.py`). Ideas: allow the camera
  centre to move slowly within a shot, model lens distortion, or calibrate with PnLCalib.
- [ ] **Kaggle shot 07 is weak**: the ball is found in only 45 % of frames, and there is about
  1 duplicate same-team player pair (< 1 m apart) per frame.
- [ ] **Kaggle shot 06 still jumps**: 99th-percentile player acceleration 64 m/s²
  (Bundesliga clips: ~5).
- [ ] **Mid-field ball flights can be missed.** The airborne test (`toonball/track/ball.py`) only
  finds flights between clear kinks in the ball's image path; a long chain of passes with no
  clear break stays on the ground. Ideas: sliding-window parabola fits, or segmenting at
  player–ball contacts instead of image kinks.

## Pose estimation for player animation

**Problem:** players are animated procedurally (run cycle from speed, facing from heading).
Legs swing too much and every player moves the same way; real body language is lost.

**Idea:** estimate a body pose per player and per frame, and drive the avatar's skeleton with it
instead of the procedural cycle.

- **2D keypoints** on each tracked crop: RTMPose / ViTPose (COCO-17), or YOLO11-pose run on the
  player boxes. Players are 50–150 px tall in wide shots, so upscale the crops and smooth the
  keypoints over time per track (One-Euro filter / Savitzky–Golay).
- **2D → 3D lifting:** MotionBERT / VideoPose3D on the per-track 2D sequence, or world-grounded
  mesh recovery (GVHMR, 4DHumans / HMR2.0, CameraHMR) on closer shots. Use the pitch homography /
  camera matrix to put the root on the ground plane and keep feet planted (no foot sliding).
- **Retargeting:** map joint rotations onto the avatar rig (hips, knees, shoulders, elbows,
  head). Fall back to the procedural cycle when keypoint confidence is low (occlusion, tiny
  players).
- **Cheaper middle ground:** use pose only for events (kick leg, header, goalkeeper dive,
  throw-in arms), detected from keypoints, and keep the procedural cycle otherwise.
- **Cost:** about 5–10 ms per crop for RTMPose-m on the 3060, so roughly 22 crops × 25 fps
  → an offline pass, cached as `pose.parquet` next to `state.parquet`.
