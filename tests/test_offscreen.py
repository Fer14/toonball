import numpy as np

from toonball.track.offscreen import _chain, fill
from toonball.track.players import Track


def _piece(i, team, start, n, x0, y0, vx=0.0, role="player"):
    t = np.arange(n)
    return Track(i, role, team, start, x0 + vx * t / 25, np.full(n, float(y0)),
                 np.full(n, abs(vx)), np.ones(n), raw_ids=[i])


def test_chain_joins_a_player_who_reappears_where_his_team_carried_him():
    F = 200
    centre = np.stack([np.linspace(40, 60, F), np.full(F, 34.0)], 1)  # team moves +20 m
    a = _piece(0, 0, 0, 50, 30, 20)          # seen 0–49 at x=30
    b = _piece(1, 0, 150, 50, 45, 20)        # reappears at 150, ~ +15 m further
    c = _piece(2, 0, 150, 50, 30, 60)        # someone else, far away
    people = _chain([a, b, c], centre, 10)
    assert [len(p) for p in people] == [2, 1]
    assert people[0][1].id == 1


def test_fill_gives_eleven_per_team_with_inferred_players_outside_the_view():
    F = 100
    tracks = [_piece(k, k % 2, 0, F, 40 + k, 20 + 2 * k) for k in range(8)]
    H = np.tile(np.eye(3), (F, 1, 1)) * np.nan                    # no camera: nothing visible
    out, remap = fill(tracks, H, (1920, 1080), 25.0, (np.full(F, 52.0), np.full(F, 34.0)))
    for team in (0, 1):
        members = [t for t in out if t.team == team]
        assert len(members) == 11
        assert sum(t.role == "goalkeeper" for t in members) == 1
        assert all(len(t.x) == F and np.isfinite(t.x).all() for t in members)
    assert set(remap) == {t.id for t in tracks}
