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
