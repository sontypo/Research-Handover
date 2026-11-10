#!/usr/bin/env python3
"""Animate one episode of the 2-D simulator as a GIF.

Usage:
  python3 animate.py --scenario corridor --controller predictive_sf --seed 0
  python3 animate.py --all            # one GIF per scenario for every baseline
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))

from socialnav_core import SCENARIOS, make_controller, make_scenario  # noqa
from socialnav_core.sim import Simulator  # noqa
from socialnav_core.viz import animate_episode  # noqa


def one(scn, ctrl, seed, out, visible):
    sc = make_scenario(scn, seed)
    log = Simulator(sc, make_controller(ctrl), robot_visible=visible).run()
    path = os.path.join(out, f"{scn}_{ctrl}_seed{seed}.gif")
    animate_episode(log, sc, path)
    print(f"{path}  ({log.outcome}, {log.t[-1]:.1f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenario", default="corridor", choices=sorted(SCENARIOS))
    ap.add_argument("--controller", default="predictive_sf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--robot-visible", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "gifs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.all:
        for scn in sorted(SCENARIOS):
            for ctrl in ("social_force", "predictive_sf"):
                one(scn, ctrl, a.seed, a.out, a.robot_visible)
    else:
        one(a.scenario, a.controller, a.seed, a.out, a.robot_visible)


if __name__ == "__main__":
    main()
