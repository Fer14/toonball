"""2D keypoint sequences → 3D body pose per tracked person (VideoPose3D, Pavllo et al. CVPR'19).

The pretrained model (Human3.6M, COCO-format 2D input) looks at ±121 frames of a person's 2D
pose and outputs 17 Human3.6M joints in camera coordinates, relative to the pelvis. Those are
rotated into the pitch frame with the estimated broadcast camera, the constant tilt between the
model's near-horizontal training cameras and an elevated broadcast camera is removed, and the
result is exported per frame for the viewer to retarget onto the avatar.

H36M joints: 0 pelvis, 1 r_hip, 2 r_knee, 3 r_foot, 4 l_hip, 5 l_knee, 6 l_foot, 7 spine,
8 thorax, 9 neck, 10 head, 11 l_shoulder, 12 l_elbow, 13 l_wrist, 14 r_shoulder, 15 r_elbow,
16 r_wrist.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from toonball.pose.angles import L_AN, L_HIP, R_AN, R_HIP

# COCO left/right pairs (eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
_LR = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
_ARMS = [(5, 6), (7, 8), (9, 10)]
_LEGS = [(11, 12), (13, 14), (15, 16)]


class TemporalModel(nn.Module):
    """VideoPose3D TemporalModel (non-causal, non-dense), matching the released checkpoints."""

    def __init__(self, joints_in: int = 17, joints_out: int = 17, widths=(3, 3, 3, 3, 3),
                 channels: int = 1024):
        super().__init__()
        self.pad = [widths[0] // 2]
        self.expand_conv = nn.Conv1d(joints_in * 2, channels, widths[0], bias=False)
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        convs, bns = [], []
        d = widths[0]
        for w in widths[1:]:
            self.pad.append((w - 1) * d // 2)
            convs += [nn.Conv1d(channels, channels, w, dilation=d, bias=False),
                      nn.Conv1d(channels, channels, 1, bias=False)]
            bns += [nn.BatchNorm1d(channels, momentum=0.1), nn.BatchNorm1d(channels, momentum=0.1)]
            d *= w
        self.layers_conv = nn.ModuleList(convs)
        self.layers_bn = nn.ModuleList(bns)
        self.shrink = nn.Conv1d(channels, joints_out * 3, 1)
        self.joints_out = joints_out

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * sum(self.pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, T, 17, 2) → (N, T', 17, 3)
        n, t = x.shape[:2]
        x = x.reshape(n, t, -1).permute(0, 2, 1)
        x = torch.relu(self.expand_bn(self.expand_conv(x)))
        for i in range(len(self.pad) - 1):
            p = self.pad[i + 1]
            res = x[:, :, p: x.shape[2] - p]
            x = torch.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x)))
            x = res + torch.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x)))
        x = self.shrink(x).permute(0, 2, 1)
        return x.reshape(n, -1, self.joints_out, 3)


def load(weights: str, device: str) -> TemporalModel:
    m = TemporalModel()
    m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["model_pos"])
    return m.to(device).eval()


def _fix_lr_swaps(K: np.ndarray) -> np.ndarray:
    """Keep left/right consistent over time: per frame, swap a limb group (arms / legs) when the
    swapped assignment is closer to the previous frame (pose models flip sides on small,
    side-on players). K: (T, 17, 3) with NaN rows for missing frames."""
    K = K.copy()
    prev = None
    for t in range(len(K)):
        if not np.isfinite(K[t, 0, 0]):
            continue
        if prev is not None:
            c = (K[t, [L_HIP, R_HIP], :2].mean(0) - prev[[L_HIP, R_HIP], :2].mean(0))
            for group in (_ARMS, _LEGS):
                a = [i for i, _ in group]
                b = [j for _, j in group]
                cur = K[t, :, :2] - c
                keep = np.linalg.norm(cur[a] - prev[a, :2], axis=1).sum() + \
                    np.linalg.norm(cur[b] - prev[b, :2], axis=1).sum()
                swap = np.linalg.norm(cur[b] - prev[a, :2], axis=1).sum() + \
                    np.linalg.norm(cur[a] - prev[b, :2], axis=1).sum()
                if swap < 0.8 * keep:
                    K[t, a + b] = K[t, b + a]
        prev = K[t].copy()
    return K


def _normalise(K: np.ndarray, fps: float) -> np.ndarray:
    """Pixel keypoints → the model's input space: pelvis-centred, scaled so the person is as
    tall as a Human3.6M subject in its normalised screen coordinates (~0.9)."""
    xy = K[..., :2]
    extent = (np.nanmax(xy[..., 1], axis=1) - np.nanmin(xy[..., 1], axis=1))
    # zoom changes slowly: scale from a rolling median of the body's pixel height
    win = int(2 * fps)
    scale = np.array([np.nanmedian(extent[max(0, t - win): t + win + 1]) for t in range(len(K))])
    centre = xy[:, [L_HIP, R_HIP]].mean(1, keepdims=True)
    return (xy - centre) / scale[:, None, None] * 0.9


def _segments(ok: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(ok)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= min_len:
            out.append((i, j))
        i = j
    return out


def _fill(K: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation of missing frames inside gaps ≤ max_gap."""
    K = K.copy()
    idx = np.arange(len(K))
    ok = np.isfinite(K[:, 0, 0])
    v = np.flatnonzero(ok)
    if len(v) < 2:
        return K
    nxt = np.clip(np.searchsorted(v, idx), 1, len(v) - 1)
    gap = v[nxt] - v[nxt - 1]
    fill = ~ok & (idx > v[0]) & (idx < v[-1]) & (gap <= max_gap)
    for j in range(K.shape[1]):
        for c in range(K.shape[2]):
            K[fill, j, c] = np.interp(idx[fill], v, K[v, j, c])
    return K


@torch.no_grad()
def lift_track(model: TemporalModel, track, kps: dict, R_cam: np.ndarray, fps: float,
               device: str) -> dict | None:
    """3D joints (n, 17, 3) in pitch axes (x, y, z-up), pelvis-relative metres, + confidence."""
    n = len(track.x)
    K = np.full((n, 17, 3), np.nan)
    for i in range(n):
        f = track.start + i
        if track.alpha[i] < 0.999:
            continue
        cands = [kps[(f, r)] for r in track.raw_ids if (f, r) in kps]
        if cands:
            k = max(cands, key=lambda c: c[:, 2].mean())
            if k[[L_HIP, R_HIP, L_AN, R_AN], 2].min() > 0.2:
                K[i] = k
    K = _fill(_fix_lr_swaps(K), max_gap=int(0.4 * fps))
    ok = np.isfinite(K[:, 0, 0])
    segs = _segments(ok, min_len=int(0.6 * fps))
    if not segs:
        return None
    J = np.full((n, 17, 3), np.nan)
    conf = np.zeros(n)
    pad = model.receptive_field // 2
    for a, b in segs:
        x = _normalise(K[a:b], fps)
        x = np.nan_to_num(x)
        x = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)])
        out = model(torch.as_tensor(x[None], dtype=torch.float32, device=device))[0].cpu().numpy()
        cam = out - out[:, :1]                                  # pelvis-relative, camera axes
        frames = track.start + np.arange(a, b)
        Rw = R_cam[frames]                                      # world → camera, per frame
        world = np.einsum("tji,tkj->tki", Rw, cam)              # Rᵀ · X  (camera → world)
        # remove the constant tilt between the model's training cameras and ours: rotate so the
        # median spine (pelvis → thorax) is vertical
        spine = np.nanmedian(world[:, 8] - world[:, 0], axis=0)
        world = world @ _align(spine, np.array([0, 0, 1.0])).T
        J[a:b] = world
        conf[a:b] = np.nan_to_num(K[a:b, :, 2]).mean(1)
    return {"joints": J, "conf": conf}


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking direction a onto direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def camera_rotations(P: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Per-frame orthogonal map world (x, y, z-up) → camera axes (x right, y down, z forward)
    from (F, 3, 4) projection matrices P = K [M | t] with square pixels, centred principal point.
    M may be a reflection (our world is z-up, OpenCV's pitch frame z-down); its transpose still
    maps camera directions back to the world."""
    w, h = size
    F = len(P)
    R = np.full((F, 3, 3), np.nan)
    for t in range(F):
        if not np.isfinite(P[t, 0, 0]):
            continue
        M = P[t, :, :3]
        f = np.linalg.norm(M[0] - (w / 2) * M[2]) / np.linalg.norm(M[2])
        Rm = np.linalg.inv(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])) @ M
        Rm /= np.linalg.norm(Rm[2])
        if Rm[2, 2] > 0:      # the camera's forward axis must point down towards the grass
            Rm = -Rm
        U, _, Vt = np.linalg.svd(Rm)
        R[t] = U @ Vt
    ok = np.isfinite(R[:, 0, 0])
    if ok.any():  # fill gaps with the nearest valid rotation
        v = np.flatnonzero(ok)
        R[~ok] = R[v[np.clip(np.searchsorted(v, np.flatnonzero(~ok)), 0, len(v) - 1)]]
    return R
