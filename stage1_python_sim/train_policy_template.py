#!/usr/bin/env python3
"""Template for training your own learned policy (exercise, intentionally incomplete).

This file shows *where* a learning-based controller plugs in, not *how* to build
a good one. It runs a random policy in ``CrowdNavEnv`` so you can check the
interface, then stops at the points you are expected to fill in:

  1. Choose an RL algorithm (e.g. PPO from stable-baselines3, CleanRL, or RSL-RL
     with a GPU-parallel simulator) and wrap ``CrowdNavEnv`` for it.
  2. Design the network: how should a variable number of pedestrians be encoded?
     (flat MLP over the k nearest, attention, graph networks, ...)
  3. Design the reward: the provided one is a standard sparse baseline.
  4. Evaluate with ``run_benchmark.py`` through ``LearnedPolicy`` and compare with
     the baselines on all four scenarios, with pedestrians that do not yield.
  5. Save the actor as a .pt model (rl/export_policy.py) so that stages 2-3 can run it.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))

from socialnav_core.env import CrowdNavEnv  # noqa


def rollout_random(episodes=5):
    env = CrowdNavEnv(randomise_scenario=True, seed=0)
    outcomes = []
    for _ in range(episodes):
        obs, done, ret = env.reset(), False, 0.0
        while not done:
            obs, r, done, info = env.step(env.sample_action())
            ret += r
        outcomes.append((info["outcome"], round(ret, 2)))
    print("observation size:", env.obs_dim, " action size:", env.act_dim)
    print("random policy:", outcomes)


def train():
    # TODO(successor): replace with your training loop.
    raise NotImplementedError("Training is the research task of the next student.")


if __name__ == "__main__":
    rollout_random()
    print("\nNext step: implement train() and plug the result into "
          "socialnav_core.controllers.LearnedPolicy(model=...).")
