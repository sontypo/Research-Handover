#!/usr/bin/env python3
"""Play a trained policy in the 2-D simulator: metrics, plots and GIFs.

Accepts any format (best_model.zip, policy.pt, policy.npz); the settings come
from policy_meta.json / train_config.json in the same folder.

Usage:
  python3 play.py --policy runs/ppo_mixed_s0/policy.pt                  # all scenarios
  python3 play.py --policy runs/ppo_mixed_s0/policy.npz --scenario corridor --episodes 50
  python3 play.py --policy runs/ppo_mixed_s0/best_model.zip --gif --seed 3
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "core"))

from socialnav_core import SCENARIOS, episode_metrics, make_scenario  # noqa: E402
from socialnav_core.controllers import LearnedPolicy  # noqa: E402
from socialnav_core.policy_io import load_policy  # noqa: E402
from socialnav_core.sim import Simulator  # noqa: E402
from socialnav_core import viz  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--scenario", default="all", help="all, or one of " + ", ".join(sorted(SCENARIOS)))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--gif", action="store_true", help="animate the first episode of each scenario")
    ap.add_argument("--robot-visible", action="store_true")
    ap.add_argument("--out", default=None, help="default: <policy folder>/play")
    a = ap.parse_args()

    model, meta = load_policy(a.policy)
    if not meta:
        cfg = os.path.join(os.path.dirname(os.path.abspath(a.policy)), "train_config.json")
        meta = json.load(open(cfg)) if os.path.isfile(cfg) else {}
    pol = LearnedPolicy(model=model, meta=meta)
    dt = float(meta.get("dt", 0.1))
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.policy)), "play")
    os.makedirs(out, exist_ok=True)
    scns = sorted(SCENARIOS) if a.scenario == "all" else [a.scenario]

    print(f"policy {a.policy}  (dt = {dt} s, obs = {meta.get('obs', '?')})\n")
    print(f"{'scenario':13s} {'success':>8s} {'collision':>10s} {'timeout':>8s} "
          f"{'time (s)':>9s} {'min dist':>9s} {'exposure':>9s}")
    for scn in scns:
        ms, first = [], None
        for k in range(a.episodes):
            sc = make_scenario(scn, a.seed + k)
            log = Simulator(sc, pol, dt=dt, robot_visible=a.robot_visible).run()
            log.controller = "learned"
            ms.append(episode_metrics(log))
            first = first or (sc, log)
        succ = [m for m in ms if m["outcome"] == "success"]
        ttg = np.mean([m["time"] for m in succ]) if succ else float("nan")
        print(f"{scn:13s} {100 * np.mean([m['success'] for m in ms]):7.0f}% "
              f"{100 * np.mean([m['collision'] for m in ms]):9.0f}% "
              f"{100 * np.mean([m['timeout'] for m in ms]):7.0f}% {ttg:9.1f} "
              f"{np.mean([m['min_dist'] for m in ms]):9.2f} {np.mean([m['exposure_pct'] for m in ms]):8.1f}%")
        sc, log = first
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        viz.plot_episode(log, sc, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{scn}_seed{a.seed}.png"), dpi=130)
        plt.close(fig)
        if a.gif:
            viz.animate_episode(log, sc, os.path.join(out, f"{scn}_seed{a.seed}.gif"))
    print(f"\nfigures in {out}")


if __name__ == "__main__":
    main()
