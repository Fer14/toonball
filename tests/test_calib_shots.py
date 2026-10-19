import cv2
import numpy as np

from toonball.calib import homography, pitch
from toonball.io.shots import wide_shots

SIZE = (1920, 1080)


def _camera_h(scale: float) -> np.ndarray:
    """A plausible pitch→image homography; larger scale = tighter zoom."""
    src = np.float32([[30, 10], [75, 10], [90, 60], [15, 60]])
    c = np.float32([[400, 300], [1500, 300], [1900, 1000], [0, 1000]])
    centre = c.mean(0)
    return cv2.getPerspectiveTransform(src, (c - centre) * scale + centre)


def _keypoints(h_img: np.ndarray, n: int) -> np.ndarray:
    img = cv2.perspectiveTransform(pitch.vertices()[None], h_img)[0]
    inside = (img[:, 0] > 0) & (img[:, 0] < SIZE[0]) & (img[:, 1] > 0) & (img[:, 1] < SIZE[1])
    kp = np.concatenate([img, inside[:, None].astype(np.float32)], 1)
    return np.repeat(kp[None], n, 0)


def test_fit_raw_recovers_pitch_coordinates():
    h_img = _camera_h(1.0)
    H = homography.fit_raw(_keypoints(h_img, 3))
    assert np.isfinite(H[:, 0, 0]).all()
    pt = np.float32([[[52.5, 34.0]]])
    back = cv2.perspectiveTransform(cv2.perspectiveTransform(pt, h_img), H[0])
    assert np.allclose(back, pt, atol=0.05)


def test_wide_shots_keeps_wide_and_drops_uncalibrated():
    fps, n = 25, 250
    kps = np.concatenate([_keypoints(_camera_h(1.0), 150), np.zeros((100, 32, 3), np.float32)])
    shots = wide_shots([0, 150, n], kps, np.ones(n, bool), fps, SIZE)
    assert len(shots) == 1
    assert shots[0]["start"] < 150 <= shots[0]["end"] + int(0.2 * fps)
    assert 20 <= shots[0]["view_m"] <= 110


def test_fuse_rejects_a_mislabelled_keypoint_frame():
    fps, n = 25, 60
    # camera pans: pitch→image homography shifts a little every frame
    pan = [np.array([[1, 0, -4.0 * t], [0, 1, 0], [0, 0, 1]]) for t in range(n)]
    truth = [pan[t] @ _camera_h(1.0) for t in range(n)]
    kps = np.stack([_keypoints(h, 1)[0] for h in truth])
    kps[30, :, :2] += 150.0                    # one frame of badly placed keypoints
    R = np.stack([np.eye(3)] + [pan[t] @ np.linalg.inv(pan[t - 1]) for t in range(1, n)])
    H = homography.fuse(homography.fit_raw(kps), R, np.r_[False, np.ones(n - 1, bool)], SIZE, fps)
    pt = np.float32([[[960, 700]]])
    for t in (10, 30, 50):
        true_xy = cv2.perspectiveTransform(pt, np.linalg.inv(truth[t]))
        assert np.linalg.norm(cv2.perspectiveTransform(pt, H[t]) - true_xy) < 0.5
