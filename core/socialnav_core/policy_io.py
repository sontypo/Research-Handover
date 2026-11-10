"""Load a trained policy as a plain function ``obs_vector -> action in [-1, 1]^2``.

Three formats are supported, chosen by the file extension:

  .pt    a TorchScript actor saved by ``export_policy.py``; needs ``torch`` only
         (not stable-baselines3).
  .npz   weights of a feed-forward MLP, run with numpy only (no torch needed).
  .zip   a stable-baselines3 model (PPO or SAC); needs stable-baselines3 + torch.

Every exported policy comes with ``policy_meta.json`` (same folder): the control
period, the observation settings and the action scaling used in training. A
deployment must reproduce them, so ``load_policy`` returns them too.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Tuple

import numpy as np

ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0),
    "identity": lambda x: x,
}


def numpy_mlp(path: str) -> Callable:
    """Rebuild the actor saved by ``export_policy.py``."""
    z = np.load(path, allow_pickle=False)
    n = int(z["n_layers"])
    layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
    act = ACTIVATIONS[str(z["activation"])]
    squash = bool(z["squash"])          # SAC-style tanh on the output

    def f(x):
        h = np.asarray(x, np.float64).reshape(-1)
        for i, (W, b) in enumerate(layers):
            h = W @ h + b
            if i < n - 1:
                h = act(h)
        return np.tanh(h) if squash else h
    return f


def torch_policy(path: str) -> Callable:
    import torch
    net = torch.jit.load(path, map_location="cpu").eval()

    def f(x):
        with torch.no_grad():
            return net(torch.as_tensor(np.asarray(x, np.float32))[None])[0].numpy()
    return f


def sb3_policy(path: str) -> Callable:
    from stable_baselines3 import PPO, SAC
    for cls in (PPO, SAC):
        try:
            model = cls.load(path, device="cpu")
            break
        except Exception:  # noqa: BLE001 - try the next algorithm
            model = None
    if model is None:
        raise ValueError(f"{path} is not a PPO or SAC model")
    return lambda x: model.predict(np.asarray(x, np.float32), deterministic=True)[0]


def find_meta(path: str) -> dict:
    meta = os.path.join(os.path.dirname(os.path.abspath(path)), "policy_meta.json")
    if os.path.isfile(meta):
        with open(meta) as f:
            return json.load(f)
    return {}


def load_policy(path: str) -> Tuple[Callable, dict]:
    ext = os.path.splitext(path)[1].lower()
    loader = {".pt": torch_policy, ".npz": numpy_mlp, ".zip": sb3_policy}.get(ext)
    if loader is None:
        raise ValueError(f"unsupported policy file '{path}' (use .pt, .npz or .zip)")
    return loader(path), find_meta(path)
