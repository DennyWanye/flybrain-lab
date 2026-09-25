import numpy as np
from gymnasium.utils.env_checker import check_env

from flydrone.contracts import FlightAction, FlightConfig
from flydrone.env import FlyToTargetEnv


def make_env(**kwargs):
    return FlyToTargetEnv(FlightConfig(**kwargs))


def test_gymnasium_contract():
    check_env(make_env(), skip_render_check=True)


def test_same_seed_and_actions_are_deterministic():
    a, b = make_env(), make_env()
    obs_a, _ = a.reset(seed=42); obs_b, _ = b.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)
    for action in [1, 4, 0, 2]:
        result_a, result_b = a.step(action), b.step(action)
        np.testing.assert_allclose(result_a[0], result_b[0])
        assert result_a[1:4] == result_b[1:4]


def test_action_direction_and_hold_inertia():
    env = make_env(curriculum="C0")
    env.reset(seed=1, options={"initial_state": {"position_xy": [0., 0.], "goal_xy": [.8, 0.]}})
    env.step(FlightAction.POS_Y)
    assert env.state.position_xy[1] > 0
    before = env.state.velocity_xy.copy()
    env.step(FlightAction.HOLD)
    assert np.linalg.norm(before) > 0
    assert np.linalg.norm(env.state.velocity_xy) > 0


def test_speed_accel_and_boundary_are_physical():
    env = make_env(deadline_s=5.)
    env.reset(seed=1, options={"initial_state": {"position_xy": [1.84, 0.], "goal_xy": [-1., 0.]}})
    _, _, terminated, truncated, info = env.step(FlightAction.POS_X)
    assert terminated and not truncated and info["end_reason"] == "boundary"
    assert np.linalg.norm(env.state.velocity_xy) <= env.config.speed_max_mps + 1e-9
    assert env.state.position_xy[0] >= env.config.safe_half_extent_m


def test_dwell_success_and_deadline():
    env = make_env(dwell_s=.04, action_duration_s=.02, deadline_s=.2)
    env.reset(seed=1, options={"initial_state": {"position_xy": [0., 0.], "goal_xy": [0., 0.]}})
    _, _, terminal, _, info = env.step(FlightAction.HOLD)
    assert not terminal
    _, _, terminal, _, info = env.step(FlightAction.HOLD)
    assert terminal and info["end_reason"] == "success"
    env = make_env(action_duration_s=.02, deadline_s=.02)
    env.reset(seed=1, options={"initial_state": {"position_xy": [0., 0.], "goal_xy": [1., 0.]}})
    _, _, terminal, truncated, info = env.step(FlightAction.HOLD)
    assert terminal and not truncated and info["end_reason"] == "deadline"
