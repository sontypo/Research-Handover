"""Pedestrian motion predictors.

Only a constant-velocity (CV) predictor is provided. CV is a surprisingly strong
baseline for short horizons and is the reference any learned forecaster must
beat. To plug in your own forecaster (for example a graph or sequence model
trained on ETH/UCY), subclass ``Predictor`` and return trajectories with the same
shape; ``PredictiveSocialForce`` and the ROS nodes accept any ``Predictor``.
"""
from __future__ import annotations

import numpy as np


class Predictor:
    """Interface: future positions of N pedestrians over a horizon.

    ``predict`` returns an array of shape (K, T, N, 2): K hypotheses (modes),
    T future steps of length ``dt``, N pedestrians. ``weights`` (K,) sum to one.
    Deterministic predictors return K = 1.
    """

    def predict(self, pos: np.ndarray, vel: np.ndarray, horizon: float, dt: float):
        raise NotImplementedError

    def weights(self) -> np.ndarray:
        return np.ones(1)


class ConstantVelocityPredictor(Predictor):
    def predict(self, pos, vel, horizon, dt):
        pos = np.asarray(pos, float).reshape(-1, 2)
        vel = np.asarray(vel, float).reshape(-1, 2)
        steps = np.arange(1, int(round(horizon / dt)) + 1) * dt          # (T,)
        traj = pos[None, :, :] + steps[:, None, None] * vel[None, :, :]   # (T, N, 2)
        return traj[None]                                                 # (1, T, N, 2)


def ade_fde(pred: np.ndarray, truth: np.ndarray):
    """Best-of-K average and final displacement error.

    pred: (K, T, N, 2), truth: (T, N, 2). Returns (minADE, minFDE) in metres.
    """
    err = np.linalg.norm(pred - truth[None], axis=-1)       # (K, T, N)
    ade = err.mean(axis=(1, 2))
    fde = err[:, -1].mean(axis=1)
    return float(ade.min()), float(fde.min())
