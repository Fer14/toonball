"""Two 1920x1080 clips side by side on a 1920x1080 paper canvas, each at 960x540 with a title.

    uv run python scripts/side_by_side.py LEFT.mp4 RIGHT.mp4 OUT.mp4 "LEFT TITLE" "RIGHT TITLE"

For before/after overlays (e.g. toonball show detect on runs/<clip> and runs/<clip>@no-ball-tiles).
Stops at the shorter clip; silent H.264 at the left clip's fps.
"""

import subprocess
import sys

import cv2
import numpy as np

INK, PAPER, FONT = (34, 34, 34), (236, 241, 238), cv2.FONT_HERSHEY_DUPLEX

left, right, out, t_left, t_right = sys.argv[1:6]
a, b = cv2.VideoCapture(left), cv2.VideoCapture(right)
fps = a.get(cv2.CAP_PROP_FPS)
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                      "1920x1080", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-crf", "18",
                      "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
while True:
    ok1, x = a.read()
    ok2, y = b.read()
    if not (ok1 and ok2):
        break
    c = np.full((1080, 1920, 3), PAPER, np.uint8)
    c[270:810, 0:960] = cv2.resize(x, (960, 540), interpolation=cv2.INTER_AREA)
    c[270:810, 960:1920] = cv2.resize(y, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.line(c, (960, 260), (960, 820), PAPER, 6)
    for t, x0 in ((t_left, 40), (t_right, 1000)):
        (w, h), _ = cv2.getTextSize(t, FONT, 1.2, 2)
        cv2.rectangle(c, (x0 - 14, 200 - h - 14), (x0 + w + 14, 214), INK, -1)
        cv2.putText(c, t, (x0, 200), FONT, 1.2, PAPER, 2, cv2.LINE_AA)
    p.stdin.write(c.tobytes())
p.stdin.close()
p.wait()
