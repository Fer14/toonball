"""ch05 stats.json: debug_motion.stats on the five Bundesliga runs (also RUN_DIR/stats_motion.json).

    uv run python scripts/motion_stats.py OUT.json
"""
import json, sys
from pathlib import Path
from toonball.render import debug_motion as dm
out = {"clip_shown": "121364_0", "window_shown_s": [10, 30],
       "slowmo": {"file": "fusion_flip_slowmo.mp4", "source_s": [19.5, 23.5], "speed": 0.5,
                  "raw_fit_rejected_frames": [536, 540], "at_s_in_file": 3.9},
       "clips": {}}
for c in ["121364_0", "573e61_0", "08fd33_0", "0bfacc_0", "2e57b9_0"]:
    r = Path("runs") / c
    if not (r / "homography_raw.npy").exists():
        print("skip", c); continue
    st = dm.stats(r)
    (r / "stats_motion.json").write_text(json.dumps(st, indent=1))
    out["clips"][c] = st
    print(c, st["flow_points_per_frame"]["tracked"]["median"], st["motion_only_drift_m"]["max"],
          st["gating"]["rejected_either"], st["frame_to_frame_jump_m"]["raw"], st["frame_to_frame_jump_m"]["fused"], st["motion_magnitude"]["centre_speed_px_per_frame"])
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
