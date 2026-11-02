"""ch05 stills: the player mask with the background corners (frame 375 of 121364_0) and the
centre spot over the whole clip for raw / motion-only / fused, with drift and the gate.

    uv run python scripts/motion_stills.py OUT_DIR
"""
import sys, json
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toonball.calib import motion
from toonball.render import debug_motion as dm
run, A = Path("runs/121364_0"), Path(sys.argv[1])
ctx = dm.load(run, 0)
f = 375  # 15 s, same frame as still_flow_arrows.png (10 s + 5 s)
cap = cv2.VideoCapture(str(run / "clip.mp4")); cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
_, a = cap.read(); _, b = cap.read()
g = lambda x: cv2.cvtColor(cv2.resize(x, None, fx=.5, fy=.5), cv2.COLOR_BGR2GRAY)
fl = motion.flow(g(a), g(b), ctx["boxes"].get(f - 1, np.zeros((0, 4))) * .5)
m = cv2.resize(fl["mask"], (1920, 1080), interpolation=cv2.INTER_NEAREST) == 0
img = b.copy()
img[~m] = (0.75 * img[~m] + 0.25 * 255).astype(np.uint8)          # background: kept, lightened
img[m] = (0.45 * img[m] + 0.55 * np.array((70, 70, 200))).astype(np.uint8)  # players: masked out
for p in fl["p0"]:
    cv2.circle(img, tuple(np.int32(p * 2)), 4, (90, 200, 60), -1, cv2.LINE_AA)
dm._text(img, f"PLAYERS MASKED (red)   {len(fl['p0'])} CORNERS ON THE BACKGROUND (green)", (30, 60), 1.0, (236, 241, 238), 2, bg=(34, 34, 34))
cv2.imwrite(str(A / "still_mask.png"), img)
# landmark over the whole clip, motion-only chained from frame 0
fps = ctx["fps"]; t = np.arange(ctx["n"]) / fps
fig, ax = plt.subplots(2, 1, figsize=(19.2, 10.8), dpi=100, sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
for k, c, lab in (("chain", "#2896eb", "camera motion only (chained from 0 s)"),
                  ("raw", "#eb3c3c", "raw keypoint fit, each frame alone"),
                  ("fused", "#e0b000", "fused (forward + backward, gated)")):
    ax[0].plot(t, ctx["lm"][k][:, 0], c, lw=2.5 if k != "raw" else 1.8, label=lab)
ax[0].set_ylabel("centre spot on screen, x (px)", fontsize=16)
ax[0].legend(fontsize=16, loc="upper left"); ax[0].grid(alpha=.3)
rej = np.array([e == "reject" for e, _, _ in ctx["log"]["forward"]])
for i in np.flatnonzero(rej):
    ax[0].axvspan(t[i] - .02, t[i] + .02, color="#eb3c3c", alpha=.12, lw=0)
ax[1].plot(t, ctx["gap_chain"], "#2896eb", lw=2.5, label="motion only vs fused (drift)")
ax[1].plot(t, ctx["gap_raw"], "#eb3c3c", lw=1.5, label="raw vs fused")
ax[1].axhline(3, color="k", ls="--", lw=1, label="gate 3 m")
ax[1].set_ylim(0, 8); ax[1].set_ylabel("distance on pitch (m)", fontsize=16)
ax[1].set_xlabel("time (s)   clip 121364_0", fontsize=16); ax[1].legend(fontsize=14, loc="upper left"); ax[1].grid(alpha=.3)
for x in ax: x.tick_params(labelsize=14)
ax[0].set_title("Keeping the pitch still: one landmark, three homographies (red bands = raw fit rejected by the gate)", fontsize=18)
fig.tight_layout(); fig.savefig(A / "still_landmark_over_time.png"); 
