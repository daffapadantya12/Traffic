"""Animate a vehicle_states_<id>.csv with matplotlib.
Run: python3 -m traffic_twin.viewer data/dev10/vehicle_states_0.csv [--edges data/dev10/network_edges.csv] [--save out.mp4]

Traffic-engineering idea: a time-space picture of the network (roads drawn
from network_edges.csv, one dot per vehicle coloured by speed) is the fastest
way to see queues, spillback and red commits with your own eyes.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("states", help="vehicle_states_<episode>.csv")
    p.add_argument("--edges", default=None, help="network_edges.csv (default: next to the states file)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--start", type=float, default=0.0, help="first second to show")
    p.add_argument("--end", type=float, default=None, help="last second to show")
    p.add_argument("--save", default=None, help="write an mp4/gif instead of opening a window")
    args = p.parse_args(argv)

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    states = pd.read_csv(args.states)
    edges_path = args.edges or os.path.join(os.path.dirname(args.states), "network_edges.csv")
    edges = pd.read_csv(edges_path)
    t_values = np.sort(states.t.unique())
    t_values = t_values[t_values >= args.start]
    if args.end is not None:
        t_values = t_values[t_values <= args.end]
    frames = {t: g for t, g in states.groupby("t")}

    fig, ax = plt.subplots(figsize=(7, 7))
    half = float(edges.n_lanes.max()) * 3.5           # road half-width = lanes x w_lane
    for _, e in edges.iterrows():                       # road axes (grey) and stop lines (red ticks)
        ax.plot([e.x_start, e.x_end], [e.y_start, e.y_end], color="lightgrey", lw=6, zorder=0)
        if e.junction_id_at_dest >= 0:
            sx, sy = e.x_end - e.hx * 2.0, e.y_end - e.hy * 2.0
            ax.plot([sx - e.hy * half, sx + e.hy * half], [sy + e.hx * half, sy - e.hx * half], color="tomato", lw=1, zorder=1)
    scat = ax.scatter([], [], s=14, c=[], cmap="viridis", vmin=0, vmax=15, zorder=2)
    title = ax.set_title("")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (east)")
    ax.set_ylabel("y [m] (north)")
    fig.colorbar(scat, ax=ax, label="speed [m/s]")

    def update(t):
        g = frames.get(t)
        if g is None or len(g) == 0:
            scat.set_offsets(np.empty((0, 2)))
            scat.set_array(np.array([]))
        else:
            scat.set_offsets(g[["x_m", "y_m"]].values)
            scat.set_array(g.v_mps.values)
        title.set_text(f"t = {t:.0f} s   vehicles = {0 if g is None else len(g)}")
        return scat, title

    anim = FuncAnimation(fig, update, frames=t_values, interval=1000 / args.fps, blit=False)
    if args.save:
        anim.save(args.save, fps=args.fps)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
