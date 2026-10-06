"""Figures for the Assignment 1 proposal.

Run from the project root:  python3 reports/proposal/make_proposal_figures.py
Writes reports/proposal/figures/P1_twin_overview.png and P2_pilot_evidence.png.
Figure 2 reads reports/validation_summary.json and data/<run>/episodes.csv.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
RUN = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/dev10")

# palette (validated categorical slots 1-2, text tokens, surface)
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#dddcd8", "#ffffff"
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False})


# ------------------------------------------------------------------ Figure 1: the twin and the pipeline
def fig_overview():
    fig = plt.figure(figsize=(6.0, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], width_ratios=[0.12, 0.76, 0.12], hspace=0.10)

    # --- top: the "#" grid with an ambulance route (centred, narrower slot)
    ax = fig.add_subplot(gs[0, 1]); ax.set_aspect("equal"); ax.axis("off")
    L, S, w = 200.0, 150.0, 14.0
    lo, hi = -S, L + S
    for x in (0, L):
        ax.add_patch(Rectangle((x - w / 2, lo), w, hi - lo, color="#dcdcdc", zorder=1))
    for y in (0, L):
        ax.add_patch(Rectangle((lo, y - w / 2), hi - lo, w, color="#dcdcdc", zorder=1))
    for x in (0, L):
        for y in (0, L):
            ax.add_patch(Rectangle((x - w / 2, y - w / 2), w, w, color="#b5b5b5", zorder=3))
    # centre lines
    for x in (0, L):
        ax.plot([x, x], [lo, hi], color="#9a9a9a", lw=0.6, zorder=2)
    for y in (0, L):
        ax.plot([lo, hi], [y, y], color="#9a9a9a", lw=0.6, zorder=2)
    # entries
    ends = [(0, lo), (L, lo), (0, hi), (L, hi), (lo, 0), (lo, L), (hi, 0), (hi, L)]
    for (x, y) in ends:
        ax.plot(x, y, "o", color=BLUE, ms=5, zorder=5)
    # ambulance route: enters at the south-west entry (left-hand traffic: west lane northbound),
    # through J(0,0), turns east at J(0,1)? keep it simple: north through two junctions, exits north
    xr = -3.5
    ax.add_patch(FancyArrowPatch((xr, lo + 8), (xr, hi - 8), arrowstyle="-|>", color=RED, lw=2.2,
                                 mutation_scale=14, zorder=6))
    ax.plot(xr, -95, marker="s", ms=9, color=RED, mec=SURF, mew=1.5, zorder=7)
    ax.text(xr - 14, -95, "ambulance", color=INK2, fontsize=8, ha="right", va="center")
    # signal dots at the two junctions on the route (green for its approach)
    for y in (0, L):
        ax.plot(-w / 2 - 4, y - w / 2 - 4, "o", color="#1baf7a", ms=6, mec=SURF, mew=1.2, zorder=7)
        ax.text(-w / 2 - 12, y - w / 2 - 22, "green on request", color=INK2, fontsize=8, ha="right", va="top")
    ax.text(100, lo - 40, "$2\\times2$ grid, 200 m blocks, 2 lanes per direction, left-hand traffic",
            ha="center", fontsize=8, color=INK2)
    ax.annotate("8 entries = sources and sinks", xy=(L + 4, hi - 6), xytext=(L - 60, L + 95), fontsize=8, color=INK2,
                ha="center", va="center", arrowprops=dict(arrowstyle="-", color="#9a9a96", lw=0.8))
    ax.set_xlim(lo - 70, hi + 20); ax.set_ylim(lo - 60, hi + 25)
    ax.set_title("(a) The twin's arena", fontsize=9.5, loc="left")

    # --- bottom: the pipeline, full width
    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    steps = [
        ("Simulate", "2×2 grid; IDM drivers with lane changing;\nPoisson arrivals; cars yield to ambulances;\nred/green signals with all-red clearance"),
        ("Log the\ntables (CSV)", "episodes · vehicles · signal ticks (one row\nper junction per decision) · 1 Hz junction\naggregates; data dictionary and data card"),
        ("Train", "gradient-boosted model: per candidate\nphase, the general queue 30 s ahead\nand the ambulance delay"),
        ("Control behind\nthe safety shield", "the model only *requests* a phase; the\nsignal plant applies it when safe (min/max\ngreen, all-red until the box is empty)"),
        ("Evaluate on\npaired seeds", "vs fixed-time, max-pressure, naive\npreemption: ambulance travel time (primary);\ngeneral delay, recovery time, fairness"),
    ]
    n = len(steps); top, bot = 0.96, 0.02; h = (top - bot) / n
    for i, (title, sub) in enumerate(steps):
        y1 = top - (i + 1) * h + 0.012; hh = h - 0.03
        col = "#eef3fb" if i in (0, 1) else ("#fdf0ea" if i == 3 else "#f4f4f2")
        ax.add_patch(FancyBboxPatch((0.03, y1), 0.83, hh, boxstyle="round,pad=0.008", fc=col, ec="#8a8a86", lw=0.8,
                                    transform=ax.transAxes))
        ax.text(0.05, y1 + hh * 0.5, f"{i + 1}. {title}", fontsize=9, weight="bold", va="center", transform=ax.transAxes)
        ax.text(0.33, y1 + hh * 0.5, sub.replace("*requests*", "requests"), fontsize=7.8, va="center", color=INK2, transform=ax.transAxes)
        if i < n - 1:
            ax.annotate("", xy=(0.20, y1 - 0.004), xytext=(0.20, y1 + 0.012 - 0.03 + 0.03 - 0.014),
                        xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#555"))
    # feedback arrow: evaluation -> simulate (closed loop)
    ax.annotate("", xy=(0.885, top - 0.03), xytext=(0.885, bot + 0.05), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=BLUE, connectionstyle="arc3,rad=0.22"))
    ax.text(0.97, 0.5, "closed loop: the trained\nmodel drives the twin", rotation=90, fontsize=8, color=BLUE,
            ha="center", va="center", transform=ax.transAxes)
    ax.set_title("(b) From simulation to a shielded controller", fontsize=9.5, loc="left")
    fig.savefig(OUT / "P1_twin_overview.png", dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 2: pilot evidence
def fig_evidence():
    val = json.load(open(ROOT / "reports/validation_summary.json"))
    sw = pd.DataFrame(val["demand_sweep"])
    ep = pd.read_csv(RUN / "episodes.csv")
    e = ep[(ep["ev_rate"] == "default") & ep["car_mean_delay_penalty_s"].notna()]
    g = e.groupby("controller")[["ev_mean_delay_s", "car_mean_delay_penalty_s"]].mean()
    fig, axs = plt.subplots(2, 1, figsize=(6.0, 7.4), gridspec_kw=dict(hspace=0.42))
    # (a) validation: delay vs load
    ax = axs[0]
    ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    ax.axvline(1.0, color=GRID, lw=1, zorder=0)
    ax.plot(sw["vc"], sw["mean_delay_s"], color=BLUE, lw=2, marker="o", ms=5, mec=SURF, mew=1.5, zorder=3,
            label="simulated, one junction, fixed-time")
    wb = sw[sw["webster_d1_s"].notna()]
    ax.plot(wb["vc"], wb["webster_d1_s"], color=ORANGE, lw=2, marker="o", ms=5, mec=SURF, mew=1.5, zorder=3,
            label="Webster (1958) uniform delay")
    ax.text(1.012, sw["mean_delay_s"].max() * 0.38, "capacity", color=INK2, fontsize=8, ha="left", va="center", rotation=90)
    last = sw.iloc[-1]
    ax.annotate(f"{last['mean_delay_s']:.0f} s", (last["vc"], last["mean_delay_s"]), xytext=(-6, 4),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    ax.set_xlabel("load on the busiest approach, volume / capacity")
    ax.set_ylabel("mean delay per vehicle (s)")
    ax.set_title("(a) Validation: delay rises smoothly, then blows up at capacity", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    # (b) trade-off per controller: closed-loop paired-seed evaluation when available, else the episode table
    ax = axs[1]
    ax.grid(color=GRID, lw=1, zorder=0)
    cl_path = ROOT / "reports/ml/closed_loop.csv"
    names = {"fixed_time": "fixed-time", "max_pressure": "max-pressure", "max_pressure_eps": "max-pressure, ε-greedy",
             "random_dwell": "random dwell", "preempt_naive": "naive preemption", "priority_mp": "priority-aware teacher",
             "ml_sensor": "learned controller (sensor features)", "ml_truth": "learned controller (truth features)"}
    if cl_path.exists():
        cl = pd.read_csv(cl_path, low_memory=False)
        if "sensor_noise_level" in cl.columns:
            cl = cl[cl["sensor_noise_level"].fillna(0) == 0]
        if "ev_rate" in cl.columns:
            cl = cl[cl["ev_rate"].astype(str) != "0"]
        xcol = "ev_mean_travel_time_s" if "ev_mean_travel_time_s" in cl.columns else "ev_mean_delay_s"
        d = cl[cl["demand_level"] == "design"]
        g = d.groupby("controller")[[xcol, "mean_total_delay_s"]].agg(["mean", "std", "count"])
        n = int(g[(xcol, "count")].min())
        tcrit = {8: 2.365, 7: 2.447, 6: 2.571, 5: 2.776, 4: 3.182, 3: 4.303}.get(n, 2.0)
        xs, ys = g[(xcol, "mean")], g[("mean_total_delay_s", "mean")]
        xe = tcrit * g[(xcol, "std")] / np.sqrt(g[(xcol, "count")])
        ye = tcrit * g[("mean_total_delay_s", "std")] / np.sqrt(g[("mean_total_delay_s", "count")])
        ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="none", ecolor="#9a9a96", elinewidth=1, capsize=2, zorder=2)
        ax.scatter(xs, ys, s=70, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3)
        order = xs.sort_values().index.tolist()
        key = []
        for k, ctl in enumerate(order, start=1):
            ax.annotate(str(k), (xs[ctl], ys[ctl]), xytext=(7, 6), textcoords="offset points", fontsize=8.5,
                        color=INK, ha="left", va="center", weight="bold")
            key.append(f"{k}  {names.get(ctl, ctl)}")
        ax.text(0.97, 0.05, "\n".join(key), transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom", ha="right", multialignment="left",
                bbox=dict(boxstyle="round,pad=0.35", fc=SURF, ec=GRID, lw=0.8))
        ax.set_xlabel("ambulance mean travel time (s), lower is better" if xcol == "ev_mean_travel_time_s" else "ambulance mean delay (s), lower is better")
        ax.set_ylabel("mean delay of all cars (s)")
        ax.set_xlim(0, xs.max() * 1.25); ax.set_ylim(0, ys.max() * 1.35)
        ax.set_title(f"(b) Closed loop at design demand, {n} paired seeds (95% t-intervals)", fontsize=9, loc="left")
    else:
        g = e.groupby("controller")[["ev_mean_delay_s", "car_mean_delay_penalty_s"]].mean()
        ax.scatter(g["ev_mean_delay_s"], g["car_mean_delay_penalty_s"], s=70, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3)
        order = g["ev_mean_delay_s"].sort_values().index.tolist()
        xmax = max(560, g["ev_mean_delay_s"].max() * 1.15)
        key = []
        for k, ctl in enumerate(order, start=1):
            r = g.loc[ctl]
            ax.annotate(str(k), (r["ev_mean_delay_s"], r["car_mean_delay_penalty_s"]), xytext=(7, 5),
                        textcoords="offset points", fontsize=8.5, color=INK, ha="left", va="center", weight="bold")
            key.append(f"{k}  {names.get(ctl, ctl)}")
        ax.text(0.03, 0.97, "\n".join(key), transform=ax.transAxes, fontsize=8.3, color=INK, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc=SURF, ec=GRID, lw=0.8))
        ax.set_xlabel("ambulance mean delay (s), lower is better")
        ax.set_ylabel("extra delay per car caused\nby ambulances (s)")
        ax.set_xlim(0, xmax); ax.set_ylim(-6, max(50, g["car_mean_delay_penalty_s"].max() * 1.2))
        ax.set_title(f"(b) {'Pilot' if len(e) < 20 else 'Release'} ({len(e)} episodes): each controller trades ambulance delay against cars", fontsize=9.5, loc="left")
    fig.savefig(OUT / "P2_pilot_evidence.png", dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    fig_overview(); print("wrote P1_twin_overview.png")
    fig_evidence(); print("wrote P2_pilot_evidence.png")
