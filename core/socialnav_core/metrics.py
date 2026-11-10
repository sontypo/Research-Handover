"""Episode metrics.

Distances are centre-to-centre between the robot and a pedestrian. The comfort
threshold is a convention, not a calibrated value: report it with your results.
"""
from __future__ import annotations

import numpy as np

COMFORT_DIST = 1.0   # m, centre distance below which a pedestrian counts as "too close"


def episode_metrics(log, comfort_dist: float = COMFORT_DIST) -> dict:
    xy = log.robot[:, :2]
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path = float(steps.sum())
    if log.peds.shape[1]:
        d = np.linalg.norm(log.peds - xy[:, None, :], axis=2)       # (T, N)
        dmin_t = d.min(axis=1)
        d_min = float(dmin_t.min())
        exposure = float((dmin_t < comfort_dist).mean() * 100)
    else:
        d_min, exposure = float("inf"), 0.0
    straight = float(np.linalg.norm(log.goal - xy[0]))
    v = log.robot[:, 3]
    return {
        "outcome": log.outcome,
        "success": float(log.outcome == "success"),
        "collision": float(log.outcome == "collision"),
        "timeout": float(log.outcome == "timeout"),
        "time": float(log.t[-1]),
        "path_length": path,
        "path_ratio": path / max(straight, 1e-9),
        "mean_speed": float(v.mean()),
        "min_dist": d_min,
        "exposure_pct": exposure,
    }
