#!/usr/bin/env python3
"""Export a stable-baselines3 policy for deployment (Gazebo, MuJoCo, real robot).

Writes next to the model (or to --out):
  policy.pt           the actor as a TorchScript model (torch only, no stable-baselines3)
  policy.npz          the same actor's weights, runs with numpy only
  policy_meta.json    control period, observation layout, action scaling

The exported actor is the deterministic policy: the Gaussian mean for PPO, the
tanh-squashed mean for SAC. The script checks that all three formats give the
same actions as stable-baselines3 on observations from real rollouts.

Usage:
  python3 export_policy.py runs/ppo_mixed/best_model.zip
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "core"))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402

from socialnav_core.env import CrowdNavEnv, ObsConfig  # noqa: E402
from socialnav_core.policy_io import numpy_mlp, torch_policy  # noqa: E402


def load_model(path):
    for cls in (PPO, SAC):
        try:
            return cls.load(path, device="cpu"), cls.__name__
        except Exception:  # noqa: BLE001 - try the next algorithm
            continue
    raise ValueError(f"{path} is not a PPO or SAC model")


def actor_layers(model, algo):
    """Linear layers of the deterministic actor, its activation, and output squashing."""
    if algo == "PPO":
        body = list(model.policy.mlp_extractor.policy_net)
        head, squash = model.policy.action_net, False
    else:  # SAC
        body = list(model.policy.actor.latent_pi)
        head, squash = model.policy.actor.mu, True
    acts = {type(m).__name__ for m in body if not isinstance(m, nn.Linear)}
    act = {"Tanh": "tanh", "ReLU": "relu"}.get(next(iter(acts), "Identity"), "identity")
    linears = [m for m in body if isinstance(m, nn.Linear)] + [head]
    return linears, act, squash


def export(model_path, out_dir=None, check=True):
    out_dir = out_dir or os.path.dirname(os.path.abspath(model_path))
    os.makedirs(out_dir, exist_ok=True)
    model, algo = load_model(model_path)
    linears, act, squash = actor_layers(model, algo)

    # numpy weights
    arrays = {"n_layers": len(linears), "activation": act, "squash": squash}
    for i, lin in enumerate(linears):
        arrays[f"W{i}"] = lin.weight.detach().cpu().numpy().astype(np.float64)
        arrays[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float64)
    np.savez(os.path.join(out_dir, "policy.npz"), **arrays)

    # TorchScript model of the same actor
    mods = []
    for i, lin in enumerate(linears):
        mods.append(nn.Linear(lin.in_features, lin.out_features))
        mods[-1].load_state_dict(lin.state_dict())
        if i < len(linears) - 1:
            mods.append(nn.Tanh() if act == "tanh" else nn.ReLU() if act == "relu" else nn.Identity())
    if squash:
        mods.append(nn.Tanh())
    net = nn.Sequential(*mods).eval()
    dim = linears[0].in_features
    torch.jit.trace(net, torch.zeros(1, dim)).save(os.path.join(out_dir, "policy.pt"))
    formats = ["policy.pt", "policy.npz"]

    # metadata: copy the training configuration saved by train_sb3.py
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "train_config.json")
    meta = json.load(open(cfg_path)) if os.path.isfile(cfg_path) else CrowdNavEnv().config()
    meta.update({"algo": algo, "obs_dim": dim, "source_model": os.path.basename(model_path),
                 "formats": formats})
    with open(os.path.join(out_dir, "policy_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if check:
        o = meta.get("obs", {})
        env = CrowdNavEnv(scenario="mixed", seed=123, obs_cfg=ObsConfig(**o) if o else None)
        xs, x = [], env.reset()
        for _ in range(300):
            xs.append(x)
            x, _, done, _ = env.step(env.sample_action())
            if done:
                x = env.reset()
        xs = np.array(xs, np.float32)
        ref = np.clip(model.predict(xs, deterministic=True)[0], -1, 1)
        f_np = numpy_mlp(os.path.join(out_dir, "policy.npz"))
        err_np = np.max(np.abs(np.clip(np.array([f_np(v) for v in xs]), -1, 1) - ref))
        msg = f"max |npz - sb3| = {err_np:.2e}"
        f_pt = torch_policy(os.path.join(out_dir, "policy.pt"))
        err_pt = np.max(np.abs(np.clip(np.array([f_pt(v) for v in xs]), -1, 1) - ref))
        msg += f", max |pt - sb3| = {err_pt:.2e}"
        assert err_pt < 1e-4, "TorchScript export does not match the model"
        assert err_np < 1e-4, "numpy export does not match the model"
        print("export check passed:", msg)
    print(f"exported {algo} actor ({dim} inputs) to {out_dir}: {', '.join(formats)}, "
          f"policy_meta.json")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model", help="stable-baselines3 .zip")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    export(a.model, a.out, not a.no_check)
