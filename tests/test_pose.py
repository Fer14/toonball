import numpy as np

from toonball.pose.angles import _gait_phase


def test_gait_phase_locks_to_observed_footsteps():
    fps, n = 25, 250
    speed = np.full(n, 5.0)
    # the real runner steps faster than the speed-based guess (1.56 Hz), and out of phase
    true_phi = 2 * np.pi * 1.9 * np.arange(n) / fps + 1.0
    opening = 0.5 * np.abs(np.sin(true_phi))
    phi = _gait_phase(opening, np.ones(n), speed, fps)
    # legs-apart moments (|sin φ| = 1) line up once locked; φ is defined modulo π
    err = np.angle(np.exp(2j * (phi - true_phi))) / 2
    assert np.abs(err[150:]).mean() < 0.25


def test_gait_phase_without_pose_follows_speed():
    fps, n = 25, 100
    speed = np.full(n, 4.0)
    phi = _gait_phase(np.full(n, np.nan), np.zeros(n), speed, fps)
    rate = 2 * np.pi * 4.0 / fps / (1.2 + 0.4 * 4.0)
    assert np.allclose(np.diff(phi), rate)


def test_videopose_model_shapes():
    import torch

    from toonball.pose.lift import TemporalModel

    m = TemporalModel().eval()
    assert m.receptive_field == 243
    out = m(torch.zeros(1, 243 + 9, 17, 2))
    assert out.shape == (1, 10, 17, 3)


def test_camera_rotation_maps_up_to_image_up():
    from toonball.pose.lift import camera_rotations

    # camera at (52, 100, 18) looking at the centre spot, f = 1500, 1920×1080
    C = np.array([52.0, 100.0, 18.0])
    fwd = np.array([52.5, 34, 0]) - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])          # world (z-up) → camera (x right, y down, z fwd)
    K = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1.0]])
    P = K @ np.c_[R, -R @ C]
    Rc = camera_rotations(P[None], (1920, 1080))[0]
    assert np.allclose(Rc, R, atol=1e-6)
    assert (Rc @ [0, 0, 1.0])[1] < 0           # world up points up in the image


def test_ball_in_keeper_hands_and_release_timing():
    import polars as pl

    from toonball.track.ball import _image_near, _possession

    F = 40
    people = pl.DataFrame({"frame": list(range(F)), "track_id": [7] * F,
                           "x1": [100.0] * F, "y1": [100.0] * F,
                           "x2": [140.0] * F, "y2": [200.0] * F})
    frames = np.arange(F)
    uv = np.array([[120.0, 130.0]] * 20 + [[300.0, 300.0]] * 20)  # held, then thrown away
    near, hands = _image_near(frames, uv, people, {7: 0}, ["goalkeeper"], F)
    assert hands[:20].all() and not hands[20:].any()
    pos = np.zeros((1, F, 2))
    owner = _possession(pos, np.full(F, 50.0), np.full(F, 50.0), near)  # far on the ground
    assert (owner[:20] == 0).all()             # the image test wins over ground distance
    assert (owner[20:] == -1).all()            # released exactly when it left the hands
