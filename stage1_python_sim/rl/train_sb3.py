#!/usr/bin/env python3
"""Train a crowd-navigation policy with stable-baselines3 (PPO by default).

A minimal, generic setup: an MLP on the observation vector of socialnav_core
(goal, k nearest people, LiDAR-like rays), the baseline reward of
``socialnav_core.env``, and a mixture of scenarios. During training the policy is
evaluated on the four benchmark scenarios; the best model is kept and exported
(policy.pt / policy.npz / policy_meta.json) at the end.

Usage:
  python3 train_sb3.py                                  # PPO, 1.5 M steps, CPU
  python3 train_sb3.py --timesteps 200000 --run-name quick
  python3 train_sb3.py --algo sac --n-envs 4 --timesteps 300000
  tensorboard --logdir runs                             # learning curves

Output: runs/<run-name>/ with best_model.zip, final_model.zip, progress.csv,
learning_curve.png, train_config.json and the exported policy.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "core"))
sys.path.insert(0, HERE)

from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from socialnav_core import SCENARIOS, episode_metrics, make_scenario  # noqa: E402
from socialnav_core.controllers import LearnedPolicy  # noqa: E402
from socialnav_core.env import CrowdNavEnv  # noqa: E402
from socialnav_core.gym_env import SocialNavGymEnv  # noqa: E402
from socialnav_core.sim import Simulator  # noqa: E402


def evaluate(model, cfg, episodes=10, seed0=10_000):
    """Deterministic policy on the benchmark scenarios (seeds never used in training)."""
    pol = LearnedPolicy(model=lambda x: model.predict(x, deterministic=True)[0], meta=cfg)
    out = {}
    for scn in sorted(SCENARIOS):
        ms = [episode_metrics(Simulator(make_scenario(scn, seed0 + k), pol, dt=cfg["dt"],
                                        robot_visible=cfg["robot_visible"]).run())
              for k in range(episodes)]
        out[scn] = {k: float(np.mean([m[k] for m in ms])) for k in ("success", "collision", "timeout")}
    out["mean"] = {k: float(np.mean([out[s][k] for s in SCENARIOS]))
                   for k in ("success", "collision", "timeout")}
    return out


class EvalCallback(BaseCallback):
    def __init__(self, run_dir, cfg, every, episodes):
        super().__init__()
        self.run_dir, self.cfg, self.every, self.episodes = run_dir, cfg, every, episodes
        self.next, self.best = every, -np.inf
        self.t0 = time.time()
        self.csv = os.path.join(run_dir, "progress.csv")
        with open(self.csv, "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "minutes", "success", "collision", "timeout",
                                    *[f"success_{s}" for s in sorted(SCENARIOS)]])

    def _on_step(self):
        if self.num_timesteps < self.next:
            return True
        self.next += self.every
        r = evaluate(self.model, self.cfg, self.episodes)
        m = r["mean"]
        score = m["success"] - 0.5 * m["collision"]
        with open(self.csv, "a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, round((time.time() - self.t0) / 60, 2),
                                    m["success"], m["collision"], m["timeout"],
                                    *[r[s]["success"] for s in sorted(SCENARIOS)]])
        tag = ""
        if score > self.best:
            self.best = score
            self.model.save(os.path.join(self.run_dir, "best_model"))
            tag = "  <- best"
        print(f"[eval] {self.num_timesteps:>9d} steps  success {100 * m['success']:5.1f}%  "
              f"collision {100 * m['collision']:5.1f}%  timeout {100 * m['timeout']:5.1f}%  "
              + "  ".join(f"{s[:5]} {100 * r[s]['success']:3.0f}%" for s in sorted(SCENARIOS)) + tag,
              flush=True)
        self.logger.record("eval/success", m["success"])
        self.logger.record("eval/collision", m["collision"])
        return True


def plot_curve(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(os.path.join(run_dir, "progress.csv"))))
    if not rows:
        return
    t = [int(r["timesteps"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for k, c in (("success", "#16a34a"), ("collision", "#dc2626"), ("timeout", "#f59e0b")):
        ax.plot(t, [100 * float(r[k]) for r in rows], color=c, lw=2, label=k)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("evaluation episodes (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "learning_curve.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--timesteps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--scenario", default="mixed",
                    help="mixed (default) or one of: " + ", ".join(sorted(SCENARIOS)) + ", random_crowd")
    ap.add_argument("--robot-visible", action="store_true", help="pedestrians react to the robot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu is fastest for small MLPs")
    ap.add_argument("--eval-every", type=int, default=100_000)
    ap.add_argument("--eval-episodes", type=int, default=10, help="per scenario")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    a = ap.parse_args()

    run = a.run_name or f"{a.algo}_{a.scenario}_s{a.seed}"
    run_dir = os.path.join(a.runs_dir, run)
    os.makedirs(run_dir, exist_ok=True)
    env_kw = dict(scenario=a.scenario, robot_visible=a.robot_visible)
    cfg = CrowdNavEnv(**env_kw).config()
    with open(os.path.join(run_dir, "train_config.json"), "w") as f:
        json.dump({**cfg, "algo": a.algo, "timesteps": a.timesteps, "seed": a.seed}, f, indent=2)

    venv = make_vec_env(SocialNavGymEnv, n_envs=a.n_envs, seed=a.seed, env_kwargs=env_kw,
                        vec_env_cls=SubprocVecEnv if a.n_envs > 1 else None)
    try:
        import tensorboard  # noqa: F401
        tb = os.path.join(a.runs_dir, "tensorboard")
    except ImportError:
        tb = None
    if a.algo == "ppo":
        model = PPO("MlpPolicy", venv, n_steps=256, batch_size=1024, n_epochs=10, gamma=0.99,
                    gae_lambda=0.95, learning_rate=3e-4, clip_range=0.2, ent_coef=0.0,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                    tensorboard_log=tb, seed=a.seed, device=a.device, verbose=0)
    else:
        model = SAC("MlpPolicy", venv, buffer_size=300_000, batch_size=256, learning_rate=3e-4,
                    gamma=0.99, train_freq=1, gradient_steps=1, learning_starts=5_000,
                    policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log=tb, seed=a.seed,
                    device=a.device, verbose=0)
    print(f"training {a.algo.upper()} on '{a.scenario}' for {a.timesteps:,} steps "
          f"({a.n_envs} envs) -> {run_dir}", flush=True)
    t0 = time.time()
    model.learn(a.timesteps, callback=EvalCallback(run_dir, cfg, a.eval_every, a.eval_episodes),
                tb_log_name=run, progress_bar=False)
    model.save(os.path.join(run_dir, "final_model"))
    venv.close()
    print(f"training done in {(time.time() - t0) / 60:.1f} min", flush=True)
    plot_curve(run_dir)

    best = os.path.join(run_dir, "best_model.zip")
    if not os.path.isfile(best):
        model.save(os.path.join(run_dir, "best_model"))
    from export_policy import export
    export(best, run_dir)
    print(f"next: python3 play.py --policy {os.path.relpath(os.path.join(run_dir, 'policy.npz'))}")


if __name__ == "__main__":
    main()
