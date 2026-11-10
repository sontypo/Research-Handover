"""2-D visualisation of episodes (matplotlib only)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import numpy as np  # noqa: E402

ROBOT_C = "#E4572E"
PED_C = "#3B6EA8"
WALL_C = "#334155"
GROUP_CMAP = plt.get_cmap("tab10")


def _ped_color(group):
    return PED_C if group < 0 else GROUP_CMAP(group % 10)


def draw_scene(ax, scenario):
    for w in scenario.walls:
        ax.plot([w.a[0], w.b[0]], [w.a[1], w.b[1]], color=WALL_C, lw=3, solid_capstyle="round")
    x0, x1, y0, y1 = scenario.bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.grid(color="#e5e7eb", lw=0.6)
    sx, sy, _ = scenario.robot_start
    ax.plot(sx, sy, "o", mfc="white", mec=ROBOT_C, ms=8, mew=2, zorder=5)
    ax.plot(*scenario.robot_goal, "*", color=ROBOT_C, ms=16, zorder=5)


def plot_episode(log, scenario, ax=None, title=None):
    """Static figure: pedestrian trails, robot path, closest approach."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    draw_scene(ax, scenario)
    groups = [p.group for p in scenario.peds]
    for i in range(log.peds.shape[1]):
        ax.plot(log.peds[:, i, 0], log.peds[:, i, 1], color=_ped_color(groups[i]), lw=1,
                alpha=0.55)
        ax.add_patch(Circle(log.peds[-1, i], 0.25, color=_ped_color(groups[i]), alpha=0.8))
    xy = log.robot[:, :2]
    ax.plot(xy[:, 0], xy[:, 1], color=ROBOT_C, lw=2.5, zorder=4)
    ax.add_patch(Circle(xy[-1], 0.35, fc="white", ec=ROBOT_C, lw=2, zorder=6))
    if log.peds.shape[1]:
        d = np.linalg.norm(log.peds - xy[:, None, :], axis=2)
        t, i = np.unravel_index(np.argmin(d), d.shape)
        ax.plot([xy[t, 0], log.peds[t, i, 0]], [xy[t, 1], log.peds[t, i, 1]], "--",
                color="#b91c1c", lw=1.2, zorder=7)
    ax.set_title(title or f"{log.scenario} · {log.controller} · {log.outcome} "
                          f"({log.t[-1]:.1f} s)", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return ax


def animate_episode(log, scenario, path, fps=20, stride=2, figsize=(8, 5)):
    """Write a GIF of the episode."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_scene(ax, scenario)
    groups = [p.group for p in scenario.peds]
    ped_patches = [ax.add_patch(Circle((0, 0), 0.25, color=_ped_color(g), alpha=0.85))
                   for g in groups]
    robot_patch = ax.add_patch(Circle((0, 0), 0.35, fc="white", ec=ROBOT_C, lw=2, zorder=6))
    (heading,) = ax.plot([], [], color=ROBOT_C, lw=2, zorder=7)
    (trail,) = ax.plot([], [], color=ROBOT_C, lw=2, alpha=0.7)
    label = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", fontsize=9)
    frames = range(0, len(log.t), stride)

    def update(k):
        for i, p in enumerate(ped_patches):
            p.center = log.peds[k, i]
        x, y, th = log.robot[k, :3]
        robot_patch.center = (x, y)
        heading.set_data([x, x + 0.5 * np.cos(th)], [y, y + 0.5 * np.sin(th)])
        trail.set_data(log.robot[:k + 1, 0], log.robot[:k + 1, 1])
        label.set_text(f"{log.scenario} · {log.controller}   t = {log.t[k]:.1f} s   "
                       f"v = {log.robot[k, 3]:.2f} m/s")
        return ped_patches + [robot_patch, heading, trail, label]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
