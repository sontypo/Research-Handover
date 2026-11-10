"""Benchmark scenarios.

Every scenario uses the same convention: the robot starts at the origin facing +x
and must reach a goal 11 m ahead. Pedestrians are described by a start, a goal,
a preferred speed, a group id, and whether they walk back and forth (``loop``)
or stand still (``speed == 0``). Seeds jitter positions, speeds and start delays,
so that every seed gives a different but comparable crowd.

The same scenario objects are used by the 2-D simulator and by the Gazebo world
generator, so both stages show the same crowds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class Wall:
    """A wall segment from ``a`` to ``b`` (metres)."""
    a: Tuple[float, float]
    b: Tuple[float, float]


@dataclass
class PedSpec:
    """One pedestrian. ``speed == 0`` means the pedestrian stands still."""
    start: Tuple[float, float]
    goal: Tuple[float, float]
    speed: float = 1.2
    group: int = -1          # -1: walks alone
    delay: float = 0.0       # seconds before the pedestrian starts walking
    loop: bool = False       # walk back and forth between start and goal
    via: Tuple[Tuple[float, float], ...] = ()   # intermediate waypoints (e.g. a doorway)

    def route(self):
        return [tuple(self.start), *[tuple(v) for v in self.via], tuple(self.goal)]


@dataclass
class Scenario:
    name: str
    robot_start: Tuple[float, float, float]
    robot_goal: Tuple[float, float]
    peds: List[PedSpec]
    walls: List[Wall] = field(default_factory=list)
    bounds: Tuple[float, float, float, float] = (-3.0, 15.0, -6.0, 6.0)
    time_limit: float = 30.0
    description: str = ""


GOAL = (11.0, 0.0)
START = (0.0, 0.0, 0.0)


class _Jitter:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def xy(self, x, y, s=0.25):
        return (float(x + self.rng.uniform(-s, s)), float(y + self.rng.uniform(-s, s)))

    def speed(self, mean=1.2):
        return float(np.clip(self.rng.normal(mean, 0.15), 0.8, 1.5))

    def delay(self, hi=2.0):
        return float(self.rng.uniform(0.0, hi))


def _group(j: _Jitter, centre, goal, n, gid, spread=0.6, delay=0.0, loop=False, axis="y",
           via=()):
    """A walking group of ``n`` people side by side, sharing one speed.

    Members walking through ``via`` points go in single file (offset along the
    walking direction) so that a group can pass a narrow doorway.
    """
    v = j.speed()
    out = []
    for k in range(n):
        off = (k - (n - 1) / 2) * spread
        if via:
            d = np.subtract(goal, centre)
            d = d / np.linalg.norm(d)
            s = (centre[0] - d[0] * off * 1.4, centre[1] - d[1] * off * 1.4)
            g = goal
        elif axis == "y":
            s, g = (centre[0], centre[1] + off), (goal[0], goal[1] + off)
        else:
            s, g = (centre[0] + off, centre[1]), (goal[0] + off, goal[1])
        out.append(PedSpec(j.xy(*s, s=0.08), j.xy(*g, s=0.08), v, gid, delay, loop,
                           tuple(via)))
    return out


def corridor(seed: int = 0) -> Scenario:
    """5 m wide corridor with two-way traffic in loose lanes."""
    j = _Jitter(seed)
    peds = []
    peds += _group(j, (2.5, -0.9), (17.0, -0.9), 2, 0, delay=0.0)               # with the flow
    peds += _group(j, (5.0, -1.2), (17.0, -1.2), 3, 1, spread=0.55, delay=0.5)  # with the flow
    peds += _group(j, (12.5, 1.2), (-4.0, 1.2), 2, 2, delay=j.delay(1.0))       # head-on pair
    peds += _group(j, (16.0, 0.9), (-4.0, 0.9), 2, 3, delay=j.delay(2.0))       # head-on pair
    peds.append(PedSpec(j.xy(14.0, 0.2), j.xy(-4.0, 0.4), j.speed(), -1, j.delay(4.0)))
    walls = [Wall((-3.0, 2.5), (17.0, 2.5)), Wall((-3.0, -2.5), (17.0, -2.5))]
    return Scenario("corridor", START, GOAL, peds, walls, (-3.0, 17.0, -3.2, 3.2), 30.0,
                    "5 m wide corridor with two-way traffic")


def crossing(seed: int = 0) -> Scenario:
    j = _Jitter(seed)
    peds = []
    peds += _group(j, (3.5, -6.0), (3.5, 7.0), 3, 0, axis="x", delay=0.5)
    peds += _group(j, (5.5, -7.0), (5.5, 7.0), 2, 1, axis="x", delay=j.delay(2.0))
    peds += _group(j, (7.0, 6.5), (7.0, -7.0), 2, 2, axis="x", delay=j.delay(1.0))
    peds += _group(j, (8.8, 7.5), (8.8, -7.0), 3, 3, axis="x", delay=j.delay(2.0))
    return Scenario("crossing", START, GOAL, peds, [], (-3.0, 14.0, -7.0, 7.0), 30.0,
                    "groups crossing the robot's path at right angles")


def chokepoint(seed: int = 0) -> Scenario:
    j = _Jitter(seed)
    gap = 1.4
    walls = [Wall((5.5, gap / 2), (5.5, 5.0)), Wall((5.5, -gap / 2), (5.5, -5.0)),
             Wall((-3.0, 5.0), (14.0, 5.0)), Wall((-3.0, -5.0), (14.0, -5.0))]
    door = ((6.2, 0.0), (4.8, 0.0))          # approach and exit points, east -> west
    peds = []
    peds += _group(j, (10.0, 1.8), (0.5, 2.0), 2, 0, delay=0.0, via=door)
    peds += _group(j, (12.5, -1.8), (0.0, -1.5), 2, 1, delay=j.delay(3.0), via=door)
    peds.append(PedSpec(j.xy(9.0, 0.0), j.xy(-1.0, 0.8), j.speed(), -1, j.delay(6.0),
                        via=door))
    peds += _group(j, (1.5, 2.5), (13.0, 2.5), 2, 2, delay=j.delay(1.0), via=door[::-1])
    return Scenario("chokepoint", START, GOAL, peds, walls, (-3.0, 14.0, -5.5, 5.5), 35.0,
                    "a 1.4 m doorway that people cross in both directions")


def dense_social(seed: int = 0) -> Scenario:
    j = _Jitter(seed)
    peds = []

    def circle(c, n, gid, r=0.55):
        for k in range(n):
            a = 2 * np.pi * k / n + j.rng.uniform(-0.2, 0.2)
            p = j.xy(c[0] + r * np.cos(a), c[1] + r * np.sin(a), s=0.05)
            peds.append(PedSpec(p, p, 0.0, gid))

    circle((3.5, 0.6), 3, 0)
    circle((6.2, -1.0), 3, 1)
    circle((8.6, 1.1), 2, 2, r=0.45)
    peds += _group(j, (4.8, 5.0), (4.8, -5.0), 2, 3, axis="x", delay=j.delay(2.0), loop=True)
    peds[-1].speed = peds[-2].speed = 0.8   # a slow stroll
    return Scenario("dense_social", START, GOAL, peds, [], (-3.0, 14.0, -5.5, 5.5), 30.0,
                    "standing conversation groups and one strolling pair")


SCENARIOS: Dict[str, Callable[[int], Scenario]] = {
    "corridor": corridor,
    "crossing": crossing,
    "chokepoint": chokepoint,
    "dense_social": dense_social,
}


def random_crowd(seed: int = 0, n_min: int = 2, n_max: int = 12) -> Scenario:
    """Training scenario: a random number of people walking in random directions.

    Used for reinforcement learning so that a policy does not memorise the four
    benchmark layouts (it is not part of the benchmark).
    """
    j = _Jitter(seed)
    n = int(j.rng.integers(n_min, n_max + 1))
    peds = []
    for _ in range(n):
        while True:
            s = (j.rng.uniform(1.5, 12.5), j.rng.uniform(-4.5, 4.5))
            g = (j.rng.uniform(-2.0, 14.0), j.rng.uniform(-4.5, 4.5))
            if np.hypot(s[0], s[1]) > 1.5 and np.hypot(g[0] - s[0], g[1] - s[1]) > 4.0:
                break
        peds.append(PedSpec(s, g, j.speed(), -1, j.delay(3.0), loop=bool(j.rng.random() < 0.5)))
    return Scenario("random_crowd", START, GOAL, peds, [], (-3.0, 15.0, -6.0, 6.0), 30.0,
                    f"{n} people walking in random directions (training only)")


TRAINING_SCENARIOS: Dict[str, Callable[[int], Scenario]] = {
    **SCENARIOS, "random_crowd": random_crowd}


def make_scenario(name: str, seed: int = 0) -> Scenario:
    if name not in TRAINING_SCENARIOS:
        raise KeyError(f"unknown scenario '{name}', choose from {sorted(TRAINING_SCENARIOS)}")
    return TRAINING_SCENARIOS[name](seed)
