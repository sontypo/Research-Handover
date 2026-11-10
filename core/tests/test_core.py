"""Sanity tests: run with `python3 -m pytest core/tests -q`."""
import numpy as np
import pytest

from socialnav_core import (SCENARIOS, Scenario, make_controller, make_scenario,
                            run_episode, episode_metrics)
from socialnav_core.env import CrowdNavEnv
from socialnav_core.observation import to_vector
from socialnav_core.prediction import ConstantVelocityPredictor, ade_fde
from socialnav_core.robot import Robot


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_are_deterministic(name):
    a, b = make_scenario(name, 3), make_scenario(name, 3)
    assert [p.start for p in a.peds] == [p.start for p in b.peds]
    assert len(a.peds) > 0


def test_empty_scene_goal_seeker_succeeds():
    sc = Scenario("empty", (0, 0, 0), (6.0, 2.0), [])
    log = run_episode(sc, make_controller("goal_seeker"))
    assert log.outcome == "success"


def test_head_on_collision_is_detected():
    from socialnav_core.scenarios import PedSpec
    sc = Scenario("head_on", (0, 0, 0), (11.0, 0.0), [PedSpec((6.0, 0.0), (-3.0, 0.0), 1.2)])
    log = run_episode(sc, make_controller("goal_seeker"))
    assert log.outcome == "collision"


@pytest.mark.parametrize("ctrl", ["goal_seeker", "social_force", "predictive_sf"])
def test_episodes_are_finite(ctrl):
    log = run_episode(make_scenario("crossing", 0), make_controller(ctrl))
    assert np.isfinite(log.robot).all() and np.isfinite(log.peds).all()
    m = episode_metrics(log)
    assert m["outcome"] in ("success", "collision", "timeout")


def test_robot_respects_limits():
    r = Robot()
    for _ in range(100):
        r.step(5.0, 5.0, 0.05)
    assert r.v <= r.lim.v_max + 1e-9 and abs(r.w) <= r.lim.w_max + 1e-9


def test_constant_velocity_prediction():
    p = ConstantVelocityPredictor().predict([[0, 0]], [[1, 0]], 1.0, 0.5)
    assert p.shape == (1, 2, 1, 2) and np.allclose(p[0, -1, 0], [1, 0])
    assert ade_fde(p, p[0]) == (0.0, 0.0)


def test_env_and_observation_shapes():
    env = CrowdNavEnv("dense_social", seed=1)
    o = env.reset()
    assert o.shape == (env.obs_dim,)
    o, r, done, info = env.step([0.5, 0.0])
    assert o.shape == (env.obs_dim,) and isinstance(r, float)


def test_learned_policy_slot_is_empty():
    with pytest.raises(NotImplementedError):
        run_episode(make_scenario("corridor", 0), make_controller("learned"))


def test_rays_see_walls():
    from socialnav_core.observation import ray_cast
    r = ray_cast([0, 0], 0.0, [((-5, 1.0), (5, 1.0))], 4, 4.0)
    # rays at -135, -45, 45, 135 deg: the two upper ones hit the wall y = 1
    assert np.allclose(r[2:], np.sqrt(2), atol=1e-6) and np.allclose(r[:2], 4.0)


def test_learned_policy_with_meta_runs():
    from socialnav_core.controllers import LearnedPolicy
    from socialnav_core.env import CrowdNavEnv
    cfg = CrowdNavEnv().config()
    pol = LearnedPolicy(model=lambda x: np.array([1.0, 0.0]), meta=cfg)
    log = run_episode(make_scenario("crossing", 0), pol, dt=cfg["dt"])
    assert log.outcome in ("success", "collision", "timeout")
    assert pol.vector(_obs()).shape == (cfg_dim(cfg),)


def _obs():
    from socialnav_core.observation import Obs
    return Obs(0.0, 0.1, np.zeros(2), 0.0, 0.0, 0.0, np.array([5.0, 0.0]),
               np.zeros((0, 2)), np.zeros((0, 2)))


def cfg_dim(cfg):
    return 5 + 6 * cfg["obs"]["k"] + cfg["obs"]["n_rays"]


def test_numpy_policy_roundtrip(tmp_path):
    from socialnav_core.policy_io import numpy_mlp
    rng = np.random.default_rng(0)
    W0, b0, W1, b1 = rng.normal(size=(4, 3)), rng.normal(size=4), rng.normal(size=(2, 4)), rng.normal(size=2)
    np.savez(tmp_path / "p.npz", n_layers=2, activation="tanh", squash=False, W0=W0, b0=b0, W1=W1, b1=b1)
    x = rng.normal(size=3)
    assert np.allclose(numpy_mlp(str(tmp_path / "p.npz"))(x), W1 @ np.tanh(W0 @ x + b0) + b1)


def test_gym_env_passes_sb3_checker():
    pytest.importorskip("gymnasium")
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    from socialnav_core.gym_env import SocialNavGymEnv
    check_env(SocialNavGymEnv(scenario="mixed", seed=0))
    assert sb3 is not None


def test_torchscript_policy_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from socialnav_core.policy_io import load_policy
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)).eval()
    torch.jit.trace(net, torch.zeros(1, 3)).save(str(tmp_path / "policy.pt"))
    f, meta = load_policy(str(tmp_path / "policy.pt"))
    x = np.array([0.1, -0.2, 0.3], np.float32)
    with torch.no_grad():
        ref = net(torch.as_tensor(x)[None])[0].numpy()
    assert np.allclose(f(x), ref, atol=1e-6) and meta == {}
