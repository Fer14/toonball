"""Pitch model in metres. Keypoint order matches the roboflow/sports 32-keypoint pitch model."""

from __future__ import annotations

import numpy as np

LENGTH = 105.0
WIDTH = 68.0
PENALTY_BOX_W = 40.32
PENALTY_BOX_L = 16.5
GOAL_BOX_W = 18.32
GOAL_BOX_L = 5.5
CIRCLE_R = 9.15
PENALTY_SPOT = 11.0


def vertices() -> np.ndarray:
    L, W = LENGTH, WIDTH
    pb, gb = PENALTY_BOX_W, GOAL_BOX_W
    return np.array([
        (0, 0),
        (0, (W - pb) / 2),
        (0, (W - gb) / 2),
        (0, (W + gb) / 2),
        (0, (W + pb) / 2),
        (0, W),
        (GOAL_BOX_L, (W - gb) / 2),
        (GOAL_BOX_L, (W + gb) / 2),
        (PENALTY_SPOT, W / 2),
        (PENALTY_BOX_L, (W - pb) / 2),
        (PENALTY_BOX_L, (W - gb) / 2),
        (PENALTY_BOX_L, (W + gb) / 2),
        (PENALTY_BOX_L, (W + pb) / 2),
        (L / 2, 0),
        (L / 2, W / 2 - CIRCLE_R),
        (L / 2, W / 2 + CIRCLE_R),
        (L / 2, W),
        (L - PENALTY_BOX_L, (W - pb) / 2),
        (L - PENALTY_BOX_L, (W - gb) / 2),
        (L - PENALTY_BOX_L, (W + gb) / 2),
        (L - PENALTY_BOX_L, (W + pb) / 2),
        (L - PENALTY_SPOT, W / 2),
        (L - GOAL_BOX_L, (W - gb) / 2),
        (L - GOAL_BOX_L, (W + gb) / 2),
        (L, 0),
        (L, (W - pb) / 2),
        (L, (W - gb) / 2),
        (L, (W + gb) / 2),
        (L, (W + pb) / 2),
        (L, W),
        (L / 2 - CIRCLE_R, W / 2),
        (L / 2 + CIRCLE_R, W / 2),
    ], dtype=np.float32)
