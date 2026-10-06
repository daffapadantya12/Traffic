"""Static diagrams for the methodology document (no simulator needed).

Run:  python3 reports/methodology/make_static_figures.py
Writes PNGs into reports/methodology/figures/.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch, Polygon
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

# ---------------------------------------------------------------- 1. arena
def fig_arena():
    L_block, L_stub, w_lane, lanes = 200.0, 150.0, 3.5, 2
    road_w = 2 * lanes * w_lane          # 14 m
    half = road_w / 2
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    xs = [0, L_block]; ys = [0, L_block]
    lo, hi = -L_stub, L_block + L_stub
    # roads
    for x in xs:
        ax.add_patch(Rectangle((x - half, lo), road_w, hi - lo, color="#d9d9d9", zorder=1))
    for y in ys:
        ax.add_patch(Rectangle((lo, y - half), hi - lo, road_w, color="#d9d9d9", zorder=1))
    # lane lines: axis (solid) and lane divider (dashed), skipping boxes
    def seg_v(x, y0, y1, style):
        ax.plot([x, x], [y0, y1], **style, zorder=2)
    def seg_h(y, x0, x1, style):
        ax.plot([x0, x1], [y, y], **style, zorder=2)
    axis = dict(color="#444", lw=1.0)
    dash = dict(color="white", lw=0.8, ls=(0, (4, 4)))
    cuts_v = [(lo, -half), (half, L_block - half), (L_block + half, hi)]
    for x in xs:
        for y0, y1 in cuts_v:
            seg_v(x, y0, y1, axis)
            seg_v(x - w_lane, y0, y1, dash); seg_v(x + w_lane, y0, y1, dash)
    for y in ys:
        for x0, x1 in cuts_v:
            seg_h(y, x0, x1, axis)
            seg_h(y - w_lane, x0, x1, dash); seg_h(y + w_lane, x0, x1, dash)
    # junction boxes + stop lines (2 m upstream of box edge)
    names = {(0, 0): "J(0,0)", (1, 0): "J(1,0)", (0, 1): "J(0,1)", (1, 1): "J(1,1)"}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ax.add_patch(Rectangle((x - half, y - half), road_w, road_w, color="#b8b8b8", zorder=3))
            ax.text(x + 12, y + 12, names[(i, j)], ha="left", va="bottom", fontsize=8, zorder=5, color="#222")
            s = half + 2.0
            # left-hand traffic: approach lanes are on the LEFT of the axis seen from the driver
            ax.plot([x - half, x], [y - s, y - s], color="k", lw=1.5, zorder=4)      # northbound stop line (west half)
            ax.plot([x, x + half], [y + s, y + s], color="k", lw=1.5, zorder=4)      # southbound (east half)
            ax.plot([x - s, x - s], [y, y + half], color="k", lw=1.5, zorder=4)      # eastbound (north half)
            ax.plot([x + s, x + s], [y - half, y], color="k", lw=1.5, zorder=4)      # westbound (south half)
    # boundary nodes
    bn = {"S0": (0, lo), "S1": (L_block, lo), "N0": (0, hi), "N1": (L_block, hi),
          "W0": (lo, 0), "W1": (lo, L_block), "E0": (hi, 0), "E1": (hi, L_block)}
    for k, (x, y) in bn.items():
        ax.plot(x, y, "o", color="#1f77b4", ms=6, zorder=5)
        dx = 12 if x <= 0 else -12 if x >= L_block else 0
        dy = 12 if y <= 0 else -12 if y >= L_block else 0
        ax.text(x + (dx if dx else 14), y + (dy if dy else 0), k, fontsize=9, color="#1f77b4",
                ha="center", va="center", zorder=6)
    # direction arrows (left-hand traffic)
    arr = dict(arrowstyle="-|>", color="#c0392b", lw=1.2, mutation_scale=10)
    ax.add_patch(FancyArrowPatch((-w_lane * 1.5, -120), (-w_lane * 1.5, -90), **arr))   # northbound west lanes
    ax.add_patch(FancyArrowPatch((w_lane * 1.5, -90), (w_lane * 1.5, -120), **arr))     # southbound east lanes
    ax.add_patch(FancyArrowPatch((-120, w_lane * 1.5), (-90, w_lane * 1.5), **arr))     # eastbound north lanes
    ax.add_patch(FancyArrowPatch((-90, -w_lane * 1.5), (-120, -w_lane * 1.5), **arr))   # westbound south lanes
    ax.text(-125, -140, "left-hand traffic", color="#c0392b", fontsize=8, ha="left")
    # dimension lines
    ax.annotate("", xy=(0, 260), xytext=(200, 260), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(100, 268, r"$L_{block}=200$ m", ha="center", fontsize=9)
    ax.annotate("", xy=(-150, 260), xytext=(0, 260), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(-75, 268, r"$L_{stub}=150$ m", ha="center", fontsize=9)
    ax.annotate("", xy=(100, -7), xytext=(100, 7), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(108, -18, r"road width $=14$ m (4 lanes)", va="center", fontsize=8)
    ax.set_xlim(lo - 30, hi + 30); ax.set_ylim(lo - 30, hi + 30)
    ax.set_aspect("equal"); ax.set_xlabel("x (m, east)"); ax.set_ylabel("y (m, north)")
    ax.set_title("Figure A1. The '#' arena: 4 junctions, 8 boundary nodes, 2 lanes per direction")
    fig.tight_layout(); fig.savefig(OUT / "A1_arena.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 2. pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off")
    boxes = [("Configuration", "all numbers in one configuration;\nsaved as JSON + hash with every dataset", "#eef"),
             ("Network builder", "nodes, directed edges, lanes,\nconnectors, storage per lane", "#eef"),
             ("Demand", "Poisson spawning per entry,\nOD matrix, Dijkstra routes", "#efe"),
             ("Vehicles", "cars and ambulances: IDM,\nballistic update, stop-or-commit", "#efe"),
             ("Signal plant", "GREEN / ALLRED, min & max green,\nclearance gate = safety shield", "#fee"),
             ("Controller", "fixed-time, max-pressure, preemption,\npriority teacher, learned model", "#fee"),
             ("Logger", "1 Hz junction aggregates,\none row per decision instant", "#ffe"),
             ("CSV tables", "episodes, vehicles,\nsignal_ticks, junction_1hz", "#ffe")]
    cols, rows = 4, 2; w, h = 0.21, 0.30; gx, gy = 0.04, 0.16
    pos = {}
    for i, (title, sub, col) in enumerate(boxes):
        r, c = divmod(i, cols)
        if r == 1: c = cols - 1 - c           # second row runs right-to-left (snake)
        x = 0.02 + c * (w + gx); y = 0.62 - r * (h + gy)
        pos[i] = (x, y)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=col, ec="#555", transform=ax.transAxes))
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=10, weight="bold", transform=ax.transAxes)
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", fontsize=7.2, transform=ax.transAxes)
    arr = dict(arrowstyle="-|>", lw=1.3, color="#333")
    for i in range(7):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if i == 3:   # row break: down arrow
            ax.annotate("", xy=(x1 + w / 2, y1 + h), xytext=(x0 + w / 2, y0), xycoords="axes fraction", arrowprops=arr)
        elif i < 3:
            ax.annotate("", xy=(x1, y1 + h / 2), xytext=(x0 + w, y0 + h / 2), xycoords="axes fraction", arrowprops=arr)
        else:
            ax.annotate("", xy=(x1 + w, y1 + h / 2), xytext=(x0, y0 + h / 2), xycoords="axes fraction", arrowprops=arr)
    # ML feedback: CSV -> Controller (both in row 2)
    (xc, yc), (xl, yl) = pos[5], pos[7]
    ax.annotate("", xy=(xc + w * 0.5, yc), xytext=(xl + w * 0.5, yl), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#1f77b4", connectionstyle="arc3,rad=-0.45"))
    ax.text((xc + xl + w) / 2, 0.02, "offline: train the gradient-boosted model on the tables → deploy it as a priority-aware controller behind the plant (closed loop)",
            fontsize=8, color="#1f77b4", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.995, "Figure A2. Generation pipeline: the dataset is the log of the simulation", ha="center", va="top", fontsize=10, transform=ax.transAxes)
    fig.savefig(OUT / "A2_pipeline.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 3. signal state machine
def fig_fsm():
    fig, ax = plt.subplots(figsize=(9.5, 4.6)); ax.axis("off")
    def node(x, y, w, h, txt, col):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.015", fc=col, ec="#333", transform=ax.transAxes))
        ax.text(x, y, txt, ha="center", va="center", fontsize=9, transform=ax.transAxes)
    node(0.22, 0.55, 0.30, 0.30, "GREEN (phase k)\none approach green,\nall three of its movements", "#dfd")
    node(0.78, 0.55, 0.30, 0.30, "ALLRED\nevery approach red,\npending phase j", "#fdd")
    # forward transition (top)
    ax.annotate("", xy=(0.625, 0.62), xytext=(0.375, 0.62), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=1.3))
    ax.text(0.50, 0.76, "controller requests j ≠ k and time_in_phase ≥ 5 s;  or forced, overriding the request:\nan occupied approach has had no green for 180 s → longest-waiting such approach (forced = 4),\nelse time_in_phase ≥ 60 s with vehicles waiting elsewhere → the other approach with most vehicles (forced = 2)", ha="center", va="bottom", fontsize=7.6, transform=ax.transAxes)
    # back transition (bottom)
    ax.annotate("", xy=(0.375, 0.48), xytext=(0.625, 0.48), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=1.3))
    ax.text(0.50, 0.34, "time_in_allred ≥ 4 s  and  junction box empty  and  no vehicle latched COMMITTED\nat the last red onset still on an approach edge (clearance gate)  →  phase k := j, time_in_phase := 0", ha="center", va="top", fontsize=7.6, transform=ax.transAxes)
    # self loop on the far left
    ax.annotate("", xy=(0.07, 0.66), xytext=(0.07, 0.44), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.1, connectionstyle="arc3,rad=-1.3"))
    ax.text(0.02, 0.06, "request j = k: green extended\n(decision instants every 5 s during GREEN)", fontsize=8, transform=ax.transAxes)
    ax.text(0.98, 0.06, "no decision instants during ALLRED:\nthe controller is not consulted until the next green", fontsize=8, ha="right", transform=ax.transAxes)
    ax.text(0.5, 0.99, "Figure A3. Signal plant per junction: red/green display, two internal states", ha="center", va="top", fontsize=10, transform=ax.transAxes)
    fig.savefig(OUT / "A3_signal_fsm.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 4. IDM
def idm(v, gap, v_lead, v0=13.9, a=1.5, b=2.0, T=1.2, s0=2.0):
    s_star = s0 + np.maximum(0.0, v * T + v * (v - v_lead) / (2 * np.sqrt(a * b)))
    return a * (1 - (v / v0) ** 4 - (s_star / gap) ** 2)

def fig_idm():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    gap = np.linspace(2.5, 80, 400)
    for v in [3, 8, 13.9]:
        axs[0].plot(gap, idm(v, gap, v), label=f"v = v_lead = {v} m/s")
    axs[0].axhline(0, color="k", lw=0.6); axs[0].axhline(-3, color="#c0392b", lw=0.8, ls="--")
    axs[0].text(60, -2.8, "hard-brake log threshold", color="#c0392b", fontsize=8, va="bottom")
    axs[0].set_ylim(-6, 2); axs[0].set_xlabel("gap to leader (m)"); axs[0].set_ylabel("IDM acceleration (m/s²)")
    axs[0].set_title("IDM acceleration vs gap (same speed as leader)"); axs[0].legend(fontsize=8)
    v = np.linspace(0, 16, 200)
    for dv in [-3, 0, 3]:
        s_star = 2.0 + np.maximum(0, v * 1.2 + v * dv / (2 * np.sqrt(1.5 * 2.0)))
        axs[1].plot(v, s_star, label=f"v - v_lead = {dv:+d} m/s")
    axs[1].set_xlabel("own speed v (m/s)"); axs[1].set_ylabel("desired gap s* (m)"); axs[1].set_title("Desired gap s*(v, Δv)")
    axs[1].legend(fontsize=8)
    fig.suptitle("Figure A4. Intelligent Driver Model (a=1.5, b=2.0, T=1.2 s, s0=2 m, v0=13.9 m/s)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "A4_idm.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 5. stop-or-commit
def fig_stop_commit():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    v = np.linspace(0, 16, 200); b_stop = 3.0
    d = v ** 2 / (2 * b_stop)
    ax.fill_between(v, d, 60, color="#dfd", alpha=0.8, label="STOPPING: d ≥ v²/(2b_stop)")
    ax.fill_between(v, 0, d, color="#fdd", alpha=0.8, label="COMMITTED: cannot stop at 3 m/s²")
    ax.plot(v, d, color="k", lw=1.2)
    ax.set_xlabel("speed at red onset v (m/s)"); ax.set_ylabel("distance to stop line d (m)")
    ax.set_ylim(0, 60); ax.set_xlim(0, 16); ax.legend(loc="upper left", fontsize=8)
    ax.text(13.9, 13.9 ** 2 / 6 + 2, "v_limit 13.9 m/s → 32 m", fontsize=8, ha="right")
    ax.set_title("Figure A5. Stop-or-commit rule evaluated once per vehicle at red onset")
    fig.tight_layout(); fig.savefig(OUT / "A5_stop_commit.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 6. parameter distributions
def fig_params():
    rng = np.random.default_rng(7); n = 20000; v_limit = 13.9
    alpha = rng.beta(2, 2, n)
    LN = lambda s: rng.lognormal(0, s, n)
    v0 = np.clip(v_limit * (1 + 0.20 * (alpha - 0.5)) * LN(0.08), 0.85 * v_limit, 1.15 * v_limit)
    a_max = np.clip(1.5 * (1 + 0.50 * (alpha - 0.5)) * LN(0.10), 0.8, 2.5)
    b = np.clip(2.0 * (1 + 0.30 * (alpha - 0.5)) * LN(0.10), 1.2, 3.0)
    T = np.clip(1.2 * (1 - 0.40 * (alpha - 0.5)) * LN(0.10), 0.8, 1.8)
    fig, axs = plt.subplots(1, 5, figsize=(13, 2.8))
    for ax, x, name in zip(axs, [alpha, v0, a_max, b, T],
                           ["alpha ~ Beta(2,2)", "v0 (m/s)", "a_max (m/s²)", "b (m/s²)", "T (s)"]):
        ax.hist(x, bins=40, color="#1f77b4", alpha=0.85); ax.set_title(name, fontsize=9); ax.set_yticks([])
    fig.suptitle("Figure A6. Per-driver parameters: one latent aggressiveness alpha shifts every IDM parameter (20 000 draws)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "A6_param_dists.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 7. lane use at the stop line (left-hand)
def fig_lanes():
    fig, ax = plt.subplots(figsize=(5.2, 5.2)); ax.set_aspect("equal"); ax.axis("off")
    half, wl = 7.0, 3.5
    ax.add_patch(Rectangle((-half, -40), 14, 80, color="#d9d9d9")); ax.add_patch(Rectangle((-40, -half), 80, 14, color="#d9d9d9"))
    ax.add_patch(Rectangle((-half, -half), 14, 14, color="#b8b8b8"))
    ax.plot([0, 0], [-40, -half], color="#444"); ax.plot([0, 0], [half, 40], color="#444")
    ax.plot([-40, -half], [0, 0], color="#444"); ax.plot([half, 40], [0, 0], color="#444")
    for s in (-wl, wl):
        ax.plot([s, s], [-40, -half], color="w", ls="--", lw=0.8); ax.plot([s, s], [half, 40], color="w", ls="--", lw=0.8)
        ax.plot([-40, -half], [s, s], color="w", ls="--", lw=0.8); ax.plot([half, 40], [s, s], color="w", ls="--", lw=0.8)
    ax.plot([-half, 0], [-9, -9], color="k", lw=2)   # northbound stop line
    # northbound approach from the south: kerb lane 0 (x=-5.25), median lane 1 (x=-1.75)
    arr = dict(arrowstyle="-|>", lw=1.4, mutation_scale=12)
    ax.add_patch(FancyArrowPatch((-5.25, -30), (-5.25, 20), color="#2a7", connectionstyle="arc3,rad=0", **arr))           # T from lane 0
    ax.add_patch(FancyArrowPatch((-5.25, -30), (-30, 5.25), color="#2a7", connectionstyle="arc3,rad=0.5", **arr))         # NS (left turn) from lane 0 -> west arm
    ax.add_patch(FancyArrowPatch((-1.75, -30), (-1.75, 20), color="#27b", connectionstyle="arc3,rad=0", **arr))           # T from lane 1
    ax.add_patch(FancyArrowPatch((-1.75, -30), (30, -1.75), color="#27b", connectionstyle="arc3,rad=-0.5", **arr))       # FS (right turn) from lane 1 -> east arm
    ax.text(-16, -36, "lane 0 (kerb):\nnear-side + through", ha="right", va="top", fontsize=8, color="#2a7")
    ax.plot([-14.5, -5.25], [-38, -32], color="#2a7", lw=0.6)
    ax.text(16, -36, "lane 1 (median):\nfar-side + through", ha="left", va="top", fontsize=8, color="#27b")
    ax.plot([14.5, -1.75], [-38, -32], color="#27b", lw=0.6)
    ax.text(-30, 10, "near-side\n(left) exit", fontsize=8, ha="center", color="#2a7")
    ax.text(30, -10, "far-side\n(right) exit", fontsize=8, ha="center", color="#27b")
    ax.text(0, 36, "through", fontsize=8, ha="center")
    ax.set_xlim(-42, 42); ax.set_ylim(-48, 42)
    ax.set_title("Figure A7. Lane use for a northbound approach (left-hand traffic)", fontsize=9)
    fig.savefig(OUT / "A7_lane_use.png", dpi=170, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    for f in (fig_arena, fig_pipeline, fig_fsm, fig_idm, fig_stop_commit, fig_params, fig_lanes):
        f(); print("wrote", f.__name__)
    print(sorted(p.name for p in OUT.iterdir()))
