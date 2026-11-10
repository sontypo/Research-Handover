#!/usr/bin/env python3
"""Stage 1 benchmark: every scenario x controller x seed in the 2-D simulator.

Writes to --out (default stage1_python_sim/outputs):
  results.csv                one row per episode
  summary.md                 success / collision / timeout and comfort metrics
  summary.png                bar chart of the outcomes
  trajectories_<scn>.png     one example episode per controller
  logs/<scn>_<ctrl>_<seed>.npz  raw episode logs (only with --save-logs)

Usage:
  python3 run_benchmark.py                      # 20 seeds, the three baselines
  python3 run_benchmark.py --seeds 5 --robot-visible
  python3 run_benchmark.py --controllers social_force predictive_sf
  python3 run_benchmark.py --policy rl/runs/ppo_mixed_s0/policy.npz   # adds the learned policy
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))

from socialnav_core import SCENARIOS, episode_metrics, make_controller, make_scenario  # noqa
from socialnav_core.sim import Simulator  # noqa
from socialnav_core import viz  # noqa
import matplotlib.pyplot as plt  # noqa

BASELINES = ["goal_seeker", "social_force", "predictive_sf"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--scenarios", nargs="+", default=sorted(SCENARIOS))
    ap.add_argument("--controllers", nargs="+", default=BASELINES)
    ap.add_argument("--robot-visible", action="store_true",
                    help="pedestrians react to the robot (easier, less honest)")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--save-logs", action="store_true")
    ap.add_argument("--policy", default=None,
                    help="exported policy (.pt/.npz/.zip) to add as controller 'learned'")
    args = ap.parse_args()
    if args.policy and "learned" not in args.controllers:
        args.controllers = list(args.controllers) + ["learned"]

    def controller(name):
        """(controller, control period); a learned policy runs at its training period."""
        if name == "learned":
            from socialnav_core.controllers import LearnedPolicy
            pol = LearnedPolicy(policy_path=args.policy)
            return pol, float(pol.dt or 0.1)
        return make_controller(name), 0.05
    os.makedirs(args.out, exist_ok=True)
    if args.save_logs:
        os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)

    rows, examples = [], {}
    t0 = time.time()
    for scn in args.scenarios:
        for ctrl in args.controllers:
            for seed in range(args.seeds):
                sc = make_scenario(scn, seed)
                c, dt = controller(ctrl)
                log = Simulator(sc, c, dt=dt, robot_visible=args.robot_visible).run()
                log.controller = ctrl
                m = episode_metrics(log)
                rows.append({"scenario": scn, "controller": ctrl, "seed": seed, **m})
                if seed == 0:
                    examples[(scn, ctrl)] = (sc, log)
                if args.save_logs:
                    log.save(os.path.join(args.out, "logs", f"{scn}_{ctrl}_{seed:03d}.npz"))
        print(f"  {scn:13s} done ({time.time() - t0:5.1f} s)")

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # summary table
    lines = [f"# Stage 1 benchmark ({args.seeds} seeds per cell, pedestrians "
             f"{'react to' if args.robot_visible else 'ignore'} the robot)", "",
             "| Scenario | Controller | Success % | Collision % | Timeout % | Time (s)* | "
             "Min dist (m) | Exposure % |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    agg = {}
    for scn in args.scenarios:
        for ctrl in args.controllers:
            r = [x for x in rows if x["scenario"] == scn and x["controller"] == ctrl]
            s = [x for x in r if x["outcome"] == "success"]
            agg[(scn, ctrl)] = [100 * np.mean([x[k] for x in r]) for k in
                                ("success", "collision", "timeout")]
            ttg = f"{np.mean([x['time'] for x in s]):.1f}" if s else "–"
            lines.append(f"| {scn} | {ctrl} | {agg[(scn, ctrl)][0]:.0f} | "
                         f"{agg[(scn, ctrl)][1]:.0f} | {agg[(scn, ctrl)][2]:.0f} | {ttg} | "
                         f"{np.mean([x['min_dist'] for x in r]):.2f} | "
                         f"{np.mean([x['exposure_pct'] for x in r]):.1f} |")
    lines += ["", "*Time to goal over successful episodes only. Distances are centre-to-centre; "
              "exposure = share of time with a pedestrian closer than 1.0 m."]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # outcome bar chart
    fig, axs = plt.subplots(1, len(args.scenarios), figsize=(3.2 * len(args.scenarios), 3.4),
                            sharey=True)
    axs = np.atleast_1d(axs)
    colors = ["#16a34a", "#dc2626", "#f59e0b"]
    for ax, scn in zip(axs, args.scenarios):
        bottom = np.zeros(len(args.controllers))
        for k, (lab, c) in enumerate(zip(["success", "collision", "timeout"], colors)):
            vals = np.array([agg[(scn, ctrl)][k] for ctrl in args.controllers])
            ax.bar(range(len(args.controllers)), vals, bottom=bottom, color=c, label=lab)
            bottom += vals
        ax.set_xticks(range(len(args.controllers)))
        ax.set_xticklabels([c.replace("_", "\n") for c in args.controllers], fontsize=8)
        ax.set_title(scn, fontsize=10)
    axs[0].set_ylabel("episodes (%)")
    axs[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "summary.png"), dpi=150)
    plt.close(fig)

    # example trajectories per scenario
    for scn in args.scenarios:
        fig, axs = plt.subplots(1, len(args.controllers),
                                figsize=(5.2 * len(args.controllers), 4.2))
        for ax, ctrl in zip(np.atleast_1d(axs), args.controllers):
            sc, log = examples[(scn, ctrl)]
            viz.plot_episode(log, sc, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"trajectories_{scn}.png"), dpi=130)
        plt.close(fig)
    print(f"\nwrote results to {args.out}")


if __name__ == "__main__":
    main()
