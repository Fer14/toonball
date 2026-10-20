# toonball — ideas to consider later

## Known issues to follow up (noted 2026-10-20)

- [ ] **Other broadcasts look far off** (Kaggle highlight reels from other leagues):
  calibration, kit colours and shot selection. Tracked in
  [#1](https://github.com/Fer14/toonball/issues/1).
- [ ] **3D pose exaggerates crouches**: in 573e61_0 the keeper briefly lies flat after his
  catch, while the real keeper is crouched low (see #1).

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

### v2 — full-body 3D pose (branch `pose-estimation`)

- **2D:** YOLO11m-pose on upscaled crops of every tracked box (`toonball/pose/estimate.py` →
  `pose.parquet`).
- **3D:** VideoPose3D (Human3.6M, COCO input), reimplemented in `toonball/pose/lift.py` and
  loaded from the released checkpoint. It lifts each person's 2D sequence (±121 frames) to 17
  joints and rotates them into pitch axes with the estimated broadcast camera. The constant
  tilt between H36M's near-horizontal cameras and the elevated broadcast camera is removed by
  making the median spine vertical. Left/right flips of the 2D model are fixed over time
  before lifting.
- **Viewer** (`applyPose3d`): skeleton body → pelvis → chest → head / shoulders → elbows,
  pelvis → hips → knees. Every bone is aimed at its 3D direction, body facing comes from hips
  and shoulders, and the lower foot stays on the grass. Fully pose-driven on 83–94 % of
  visible player-frames. The scripted kick, keeper stance and dives only run where the pose
  is missing.
- **Ball in the keeper's hands:** possession is now also decided in the image (ball inside a
  player's box), because a held ball projects ~5 m behind the keeper on the grass. A keeper
  holding it in the upper half of his box carries it in his hands.
- **Pass timing:** each possession is dated back to the first / last frame the ball was
  really with that player (the 4-frame hysteresis used to keep it glued to the foot after
  the kick).
- **Known weakness:** poses are looser than the real ones (e.g. a crouched catch comes out
  fairly upright); H36M has no football movements and our players are 50–150 px tall.

### Done since v2

- [x] **Foot planting** (#3): stepping cycle with physics-based swing while moving, standing
  foot locked by two-bone IK. Standing-foot sliding 2.0 → 0.4 m/s.
- [x] **Ball contact from the body** (#4): kicks / headers / throw-ins / keeper throws timed
  to the release frame; the ball leaves from the foot, head or hands.
- [x] **Stronger lifter** (#5): MotionAGFormer-B (WACV 2024) is the default. 2D agreement
  −16 %, jitter −10 %, crouches deeper.
- [x] **Full teams** (#6): 11 + 11 for the whole clip, inferred off camera.
- [x] **Look and feel** (#7): shirt numbers by position, 3D crowd reacting to danger, goal
  celebrations.

### Next

- **Left/right:** resolve which leg leads from the foot on the ground (the lowest ankle, which
  stays still in pitch coordinates) instead of the model's L/R labels.
- **Stronger lifting model:** MotionAGFormer / MotionBERT, or fine-tune the lifter on
  football motion (e.g. synthetic data from a game engine).
- **Foot locking:** keep a planted foot fixed on the grass in pitch coordinates (removes
  sliding).
- **Events from pose:** kicks (one leg far forward), headers (both ankles off the ground),
  throw-ins (both hands above the head), goalkeeper dives (torso near horizontal) — instead of
  inferring them from the ball alone.
- **Mesh recovery on closer shots:** GVHMR, 4DHumans or CameraHMR for hand and head
  detail.
- **Faster crops:** RTMPose-m (≈5 ms per crop) if the pose pass becomes the bottleneck on long
  videos.
- **Real shirt numbers:** jersey OCR on the clearest crops of each person (e.g. PARSeq, or a
  small VLM), voted over the whole track; today numbers follow position.
- **More footage:** needs the Kaggle token for the DFL competition (dozens of clips + full
  halves).
- **Sound:** crowd noise following the excitement curve, a whistle at the goal.
