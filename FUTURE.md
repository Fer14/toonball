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

### v1 (branch `pose-estimation`)

- **2D pose:** YOLO11m-pose on upscaled crops of every tracked box (`toonball/pose/estimate.py`,
  cached as `pose.parquet`, about 1 min per 30 s clip).
- **Angles** (`toonball/pose/angles.py`): limbs are assumed to swing in the sagittal plane, and
  foreshortening is undone with s = facing · image-right. The weight drops when a player runs
  at or away from the camera.
- **Left/right legs are unreliable on 50–150 px players** (hip–hip correlation ≈ 0 while
  running), so v1 exports only side-independent cues:
  - gait phase, locked to the stride opening |lh − rh| with a second-order PLL (legs-apart
    timing vs video: corr 0.35–0.53, against ≈ 0 for the procedural cycle)
  - stride amplitude, mean knee and elbow bend, torso lean and roll
- **Viewer:** knees and elbows added to the rig. Everything is blended with the procedural
  cycle by reliability. The "pose from video" checkbox, or `&pose=0` in the URL, turns it off
  for comparison.
- **Found along the way:** the procedural cadence was about 5 steps/s at a jog. The stride
  length is now 1.2 m + 0.4 × speed, about 3 steps/s, as measured.

### Next

- **Left/right:** resolve which leg leads from the foot on the ground (the lowest ankle, which
  stays still in pitch coordinates) instead of the model's L/R labels.
- **Events from pose:** kicks (one leg far forward), headers (both ankles off the ground),
  throw-ins (both hands above the head), goalkeeper dives (torso near horizontal) — instead of
  inferring them from the ball alone.
- **Better 3D:** lift 2D → 3D (MotionBERT / VideoPose3D) on per-track sequences, or
  world-grounded mesh recovery (GVHMR, 4DHumans, CameraHMR) on closer shots, using the camera
  matrix to keep feet planted.
- **Retargeting** full joint rotations onto the rig once left/right is reliable.
- **Faster crops:** RTMPose-m (≈5 ms per crop) if the pose pass becomes the bottleneck on long
  videos.
