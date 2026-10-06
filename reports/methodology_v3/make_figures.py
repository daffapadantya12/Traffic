"""Figures for the v3 proposal and methodology (Perempatan Fatmawati digital twin).

Run from the project root:  python3 reports/methodology_v3/make_figures.py [names...]
Reads the OpenStreetMap extract and the SUMO network of the study area (read-only) and writes
PNG figures into reports/methodology_v3/figures/ and reports/proposal_v3/figures/.
Map data (c) OpenStreetMap contributors, ODbL.
"""
from __future__ import annotations

import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "data", "fatmawati")
OUT_M = os.path.join(ROOT, "reports", "methodology_v3", "figures")
OUT_P = os.path.join(ROOT, "reports", "proposal_v3", "figures")
os.makedirs(OUT_M, exist_ok=True)
os.makedirs(OUT_P, exist_ok=True)

LAT0, LON0 = -6.29250, 106.79522          # junction box centre
HOSP_LON, HOSP_LAT = 106.79627, -6.29525   # RSUP Fatmawati (OSM amenity=hospital)
KX = 111320.0 * math.cos(math.radians(abs(LAT0)))
KY = 110574.0

# palette: validated categorical slots + text tokens
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                            "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#dddcd8", "#ffffff"
SIG_G, SIG_A, SIG_R = "#1f9d55", "#eda100", "#d64545"       # signal semantics (labelled in every chart)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": SURF})


def xy(lon, lat):
    return (lon - LON0) * KX, (lat - LAT0) * KY


def save(fig, name, *dirs):
    for d in dirs:
        fig.savefig(os.path.join(d, name), dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------- OSM data
ROAD_STYLE = {   # class: (colour, width, z)
    "motorway": (ORANGE, 2.4, 6), "motorway_link": (ORANGE, 1.0, 5),
    "trunk": ("#2f2f2d", 2.0, 5), "trunk_link": ("#2f2f2d", 0.9, 4),
    "primary": ("#4d4c49", 1.7, 4), "primary_link": ("#4d4c49", 0.8, 3),
    "secondary": ("#7c7b76", 1.2, 3), "secondary_link": ("#7c7b76", 0.7, 2),
    "tertiary": ("#a9a8a2", 0.8, 2), "tertiary_link": ("#a9a8a2", 0.5, 1),
}


def load_osm_roads():
    tree = ET.parse(os.path.join(SRC, "osm", "fatmawati_roads.osm.xml"))
    root = tree.getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    roads = []
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
        hw = tags.get("highway")
        if hw not in ROAD_STYLE:
            continue
        pts = [nodes[nd.get("ref")] for nd in w.iter("nd") if nd.get("ref") in nodes]
        if len(pts) >= 2:
            roads.append((hw, tags.get("name", ""), np.array([xy(*p) for p in pts])))
    return roads


def load_buildings():
    with open(os.path.join(SRC, "osm", "buildings_1600m.json")) as f:
        js = json.load(f)
    polys = []
    for el in js.get("elements", []):
        g = el.get("geometry")
        if el.get("type") == "way" and g and len(g) >= 3:
            polys.append(np.array([xy(p["lon"], p["lat"]) for p in g]))
    return polys


def bbox_m():
    with open(os.path.join(SRC, "osm", "bbox.txt")) as f:
        s, w, n, e = (float(v) for v in f.read().strip().split(","))
    (x0, y0), (x1, y1) = xy(w, s), xy(e, n)
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------------------- figure: study area
def fig_study_area(out_dirs=(OUT_M, OUT_P), name="F1_study_area.png", inset=True):
    import sumolib
    roads = load_osm_roads()
    blds = load_buildings()
    bx0, by0, bx1, by1 = bbox_m()
    fig = plt.figure(figsize=(6.3, 3.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.86], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_collection(PolyCollection(blds, facecolor="#e7e6e1", edgecolor="none", zorder=0))
    for cls in ROAD_STYLE:
        col, lw, z = ROAD_STYLE[cls]
        segs = [r[2] for r in roads if r[0] == cls]
        ax.add_collection(LineCollection(segs, colors=col, linewidths=lw * 0.62, zorder=z, capstyle="round"))
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ls=(0, (4, 3)), lw=0.8, ec=MUTED, zorder=7))
    hx, hy = xy(HOSP_LON, HOSP_LAT)
    ax.plot(0, 0, marker="o", ms=7, mfc=RED, mec=SURF, mew=1.5, zorder=9)
    ax.plot(hx, hy, marker="P", ms=8, mfc=AQUA, mec=SURF, mew=1.2, zorder=9)
    ax.annotate("Perempatan\nFatmawati", (0, 0), xytext=(-1500, 900), fontsize=7.6, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), zorder=10)
    ax.annotate("RSUP Fatmawati\n(national hospital)", (hx, hy), xytext=(1250, -1350), fontsize=7.6, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), zorder=10)
    ax.text(380, 150, "JORR toll road\n(viaduct)", fontsize=6.6, color=ORANGE, ha="left", va="bottom", zorder=10,
            bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.85))
    ax.text(1180, -1650, "Depok–Antasari\ntoll road", fontsize=6.6, color=ORANGE, ha="left", va="center", zorder=10)
    ax.text(bx0 + 150, by0 + 650, "study area 5.4 km × 5.4 km", fontsize=6.6, color=INK2, ha="left", va="bottom", zorder=10)
    # scale bar and north arrow
    sx, sy = bx0 + 150, by0 + 180
    ax.plot([sx, sx + 1000], [sy, sy], color=INK, lw=1.6, zorder=10)
    ax.text(sx + 500, sy + 90, "1 km", fontsize=7, color=INK, ha="center", va="bottom", zorder=10)
    ax.annotate("N", xy=(bx0 + 200, by1 - 160), xytext=(bx0 + 200, by1 - 620), fontsize=8, ha="center", color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0), zorder=10)
    # legend (road classes)
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=ORANGE, lw=1.6), Line2D([0], [0], color="#2f2f2d", lw=1.3),
         Line2D([0], [0], color="#4d4c49", lw=1.1), Line2D([0], [0], color="#7c7b76", lw=0.8), Line2D([0], [0], color="#a9a8a2", lw=0.6)]
    ax.legend(h, ["toll road", "trunk", "primary", "secondary", "tertiary"], loc="upper right", fontsize=6.6,
              frameon=True, facecolor=SURF, edgecolor=GRID, framealpha=0.95, handlelength=1.6, borderpad=0.4)
    ax.set_xlim(bx0 - 40, bx1 + 40); ax.set_ylim(by0 - 40, by1 + 40)
    ax.set_title("(a) Study area: main roads around the junction", fontsize=8.6, loc="left")

    if inset:
        net = sumolib.net.readNet(os.path.join(SRC, "net", "fatmawati.net.xml"), withInternal=True)
        jx, jy = net.convertLonLat2XY(LON0, LAT0)
        cs = net.getNode("cluster_353175558_356234398").getCoord()
        cn = net.getNode("cluster_356228989_356228993").getCoord()
        cx, cy = (cs[0] + cn[0]) / 2.0, (cs[1] + cn[1]) / 2.0          # view centre between the two crossings
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_visible(False)
        R = 78.0
        ax2.set_xlim(-R, R); ax2.set_ylim(-R, R)
        fig.canvas.draw()
        bb = ax2.get_window_extent()
        pt_per_m = (bb.width * 72.0 / fig.dpi) / (2 * R)               # line widths in points that equal metres
        groups = {"road": [], "internal": [], "toll": []}
        for e in net.getEdges(withInternal=True):
            internal = e.getFunction() == "internal"
            t = "internal" if internal else ("toll" if e.getType().startswith("highway.motorway") else "road")
            for ln in e.getLanes():
                sh = np.array(ln.getShape()) - [cx, cy]
                if np.min(np.hypot(sh[:, 0], sh[:, 1])) > R * 1.5:
                    continue
                groups[t].append((sh, ln.getWidth()))
        for t, col, z, a in (("road", "#c9c8c2", 1, 1.0), ("internal", "#dedcd6", 2, 1.0), ("toll", ORANGE, 6, 0.28)):
            segs = [g[0] for g in groups[t]]
            lws = [g[1] * pt_per_m for g in groups[t]]
            ax2.add_collection(LineCollection(segs, colors=col, linewidths=lws, zorder=z, alpha=a, capstyle="butt"))
        APP = {"N": "953628421#0", "E": "1069005557#0", "S": "1069005556#0", "W": "1069005554#0"}
        for arm, eid in APP.items():
            try:
                e = net.getEdge(eid)
            except KeyError:
                continue
            for ln in e.getLanes():
                sh = np.array(ln.getShape()) - [cx, cy]
                # walk back 8 m along the lane polyline from the stop line
                pts, need = [sh[-1]], 8.0
                for i in range(len(sh) - 1, 0, -1):
                    a, b = sh[i - 1], sh[i]
                    L = float(np.hypot(*(b - a)))
                    if L >= need:
                        pts.append(b - (b - a) / L * need)
                        need = 0.0
                        break
                    pts.append(a)
                    need -= L
                pts = np.array(pts[::-1])
                ax2.plot(pts[:, 0], pts[:, 1], color=GREEN, lw=ln.getWidth() * pt_per_m * 0.92, solid_capstyle="butt",
                         zorder=4, alpha=0.85)
                d = sh[-1] - sh[-2]
                u = d / max(float(np.hypot(*d)), 1e-9)
                nrm = np.array([-u[1], u[0]]) * ln.getWidth() / 2
                for q, lw_, c_ in ((sh[-1], 1.1, INK), (pts[0], 0.8, INK2)):
                    ax2.plot([q[0] + nrm[0], q[0] - nrm[0]], [q[1] + nrm[1], q[1] - nrm[1]], color=c_, lw=lw_, zorder=5)
        for (nx_, ny_), txt, ty in ((cn, "north crossing\n(eastbound side)", 62), (cs, "south crossing\n(westbound side)", -48)):
            px, py = nx_ - cx, ny_ - cy
            ax2.annotate(txt, (px, py), xytext=(-R + 4, ty), fontsize=6.4, color=INK, ha="left", va="center",
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6), zorder=9,
                         bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.9))
        lab = {"N": (24, R - 4, "from the north\n3 lanes", "top"), "S": (8, -R + 2, "from the south\n2 lanes", "bottom"),
               "E": (R - 2, -48, "from the east\n4 lanes", "right"), "W": (-R + 2, 26, "from the west\n2 lanes", "left")}
        for arm, (lx, ly, txt, side) in lab.items():
            ha = {"top": "center", "bottom": "center", "right": "right", "left": "left"}[side]
            va = {"top": "top", "bottom": "bottom", "right": "center", "left": "center"}[side]
            ax2.text(lx, ly, txt, fontsize=6.4, ha=ha, va=va, color=INK2, zorder=9,
                     bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.9))
        from matplotlib.lines import Line2D
        hh = [Line2D([0], [0], color=GREEN, lw=4, alpha=0.85), Line2D([0], [0], color=ORANGE, lw=4, alpha=0.3),
              Line2D([0], [0], color=INK, lw=1.1)]
        ax2.legend(hh, ["motorcycle box", "toll viaduct", "stop line"], loc="upper center",
                   bbox_to_anchor=(0.55, -0.01), ncol=3, fontsize=6.2, frameon=False, handlelength=1.5, columnspacing=1.0)
        ax2.set_title("(b) The junction as simulated (to scale)", fontsize=8.6, loc="left")
    fig.text(0.13, 0.06, "Map data © OpenStreetMap contributors (ODbL).", fontsize=6.4, color=MUTED, ha="left")
    save(fig, name, *out_dirs)



# ---------------------------------------------------------------------------------------- figure: system architecture
def _box(ax, x, y, w, h, title, sub, fc, ec="#8a8984", tsize=8.2, ssize=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=0.8,
                                transform=ax.transAxes, zorder=2, clip_on=False))
    ax.text(x + 0.012, y + h - 0.02, title, fontsize=tsize, weight="bold", va="top", ha="left", color=INK, transform=ax.transAxes, zorder=3)
    ax.text(x + 0.012, y + h - 0.075, sub, fontsize=ssize, va="top", ha="left", color=INK2, transform=ax.transAxes, zorder=3,
            linespacing=1.25)


def _arrow(ax, p0, p1, color="#555", rad=0.0, lw=1.0, ls="-"):
    ax.annotate("", xy=p1, xytext=p0, xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, ls=ls, connectionstyle=f"arc3,rad={rad}",
                                shrinkA=0, shrinkB=0), zorder=4)


def fig_architecture(out_dirs=(OUT_M, OUT_P), name="F2_architecture.png"):
    fig, ax = plt.subplots(figsize=(6.4, 4.6)); ax.axis("off")
    w, h, gap = 0.226, 0.19, 0.026
    xs = [i * (w + gap) for i in range(4)]
    rows = {"gen": 0.74, "learn": 0.41, "live": 0.08}
    heads = {"gen": "1  Generate the open dataset (offline, reproducible)",
             "learn": "2  Learn and evaluate (closed loop, right to left)",
             "live": "3  Live view: the same SUMO model beside live TomTom speeds (display only)"}
    for key, label in heads.items():
        ax.text(0.0, rows[key] + h + 0.03, label, fontsize=7.8, color=INK, weight="bold", transform=ax.transAxes)
    gen = [("Open sources", "OSM roads (ODbL),\nIndonesian guidelines,\nopen speed data", "#eef3fb"),
           ("Fatmawati twin", "SUMO network, 70%\nmotorcycles, boxes,\nsignal plant", "#eef3fb"),
           ("Virtual sensors", "loops, queue cameras,\ncounting gates and an\nambulance detector", "#eef3fb"),
           ("Open dataset", "sensor readings as\nCSV, dictionary\nand data card", "#f6f6f3")]
    learn = [("Decision support", "ambulance seconds\nsaved vs delay to\neveryone else", "#f6f6f3"),
             ("Evaluation grid", "every ambulance case,\nsame for every method,\nflattened per direction", "#fdf0ea"),
             ("Behind the shield", "model only requests\na phase; the plant\napplies it safely", "#fdf0ea"),
             ("Train, validate", "several model families\nvs stillness and\nrandom baselines", "#fdf0ea")]
    live = [("TomTom speeds", "read only while the\nviewer is open;\nnever stored", "#f2f8f3"),
            ("Side by side", "TomTom and twin\ncamera speeds per\napproach, \u00a9TomTom", "#f2f8f3"),
            ("Feeds nothing", "no demand sync, no\ncontroller input, no\nambulance decision", "#f2f8f3"),
            ("3D viewer", "web view: buildings,\ncars, motorcycles,\nambulances", "#f6f6f3")]
    for r, items in (("gen", gen), ("learn", learn), ("live", live)):
        for i, (t, sub, fc) in enumerate(items):
            _box(ax, xs[i], rows[r], w, h, t, sub, fc, tsize=7.4, ssize=6.5)
    for i in range(3):   # row 1 and row 3 left to right, row 2 right to left
        _arrow(ax, (xs[i] + w, rows["gen"] + h / 2), (xs[i + 1], rows["gen"] + h / 2))
        _arrow(ax, (xs[i] + w, rows["live"] + h / 2), (xs[i + 1], rows["live"] + h / 2))
        _arrow(ax, (xs[i + 1], rows["learn"] + h / 2), (xs[i] + w, rows["learn"] + h / 2))
    _arrow(ax, (xs[3] + w / 2, rows["gen"]), (xs[3] + w / 2, rows["learn"] + h), color=BLUE, lw=1.1)
    ax.text(xs[3] + w / 2 + 0.01, rows["learn"] + h + 0.045, "train", fontsize=6.6, color=BLUE, transform=ax.transAxes)
    _arrow(ax, (xs[2] + w / 2, rows["learn"] + h), (xs[2] + w / 2, rows["gen"]), color=BLUE, lw=1.1, ls="--")
    ax.text(xs[2] + w / 2 + 0.01, rows["learn"] + h + 0.045, "drives the twin", fontsize=6.6, color=BLUE, transform=ax.transAxes)
    ax.set_xlim(-0.012, 1.0); ax.set_ylim(0, 1)
    save(fig, name, *out_dirs)


# ---------------------------------------------------------------------------------------- figure: signal timeline
def fig_signal_timeline(out_dirs=(OUT_M,), name="M_signal_timeline.png"):
    """The city's static four-stage plan (Sumabrata et al., 2026, Table 3): greens 80/12/78/45 s, amber 2 s, all-red 2 s."""
    Y, AR = 2.0, 2.0
    C = 80 + 12 + 78 + 45 + 4 * (Y + AR)                    # 231 s
    rows = [  # label, list of (start, duration, colour)
        ("W through (S1)", [(0, 80, SIG_G), (80, Y, SIG_A), (82, C - 82, SIG_R)]),
        ("E through (S1, S2)", [(0, 96, SIG_G), (96, Y, SIG_A), (98, C - 98, SIG_R)]),
        ("E right (S2)", [(0, 84, SIG_R), (84, 12, SIG_G), (96, Y, SIG_A), (98, C - 98, SIG_R)]),
        ("N through, right (S3)", [(0, 100, SIG_R), (100, 78, SIG_G), (178, Y, SIG_A), (180, C - 180, SIG_R)]),
        ("S through, right (S4)", [(0, 182, SIG_R), (182, 45, SIG_G), (227, Y, SIG_A), (229, C - 229, SIG_R)]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    for i, (lab, segs) in enumerate(rows):
        for s0, d, c in segs:
            ax.barh(i, d, left=s0, height=0.7, color=c, edgecolor=SURF, linewidth=0.6)
    for x0, x1, st in ((0, 84, "S1"), (84, 100, "S2"), (100, 182, "S3"), (182, C, "S4")):
        ax.text((x0 + x1) / 2, -0.85, st, ha="center", va="center", fontsize=7.4, color=INK, weight="bold")
        ax.axvline(x0, color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylim(len(rows) - 0.4, -1.3)
    ax.set_xlim(0, C); ax.set_xlabel("time in the fixed-time cycle (s)")
    ax.set_xticks([0, 84, 100, 182, 231])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SIG_G, label="green"), Patch(color=SIG_A, label="amber"), Patch(color=SIG_R, label="red, incl. all-red")],
              loc="upper right", bbox_to_anchor=(1.0, 1.30), ncol=3, fontsize=7, frameon=False)
    ax.set_title("The city's four-stage plan: greens 80, 12, 78 and 45 s; cycle 231 s", fontsize=8.4, loc="left", pad=16)
    save(fig, name, *out_dirs)


# ---------------------------------------------------------------------------------------- figure: demand over the day
def fig_daily_demand(out_dirs=(OUT_M,), name="M_daily_demand.png"):
    """Demand relative to the calibrated peak: level factor k (0.75 / 1.0 / 1.15) times the daily profile scaled to peak 1."""
    sys.path.insert(0, ROOT)
    from traffic_twin.config import DAILY_PROFILE
    hours = np.arange(24)
    prof = np.array(DAILY_PROFILE) / max(DAILY_PROFILE)
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    for lvl, k, col in (("low", 0.75, "#9ec5f4"), ("design", 1.0, BLUE), ("high", 1.15, "#104281")):
        ax.step(np.append(hours, 24), np.append(k * prof, k * prof[-1]), where="post", color=col, lw=2, label=f"{lvl}: {k:.2f} × profile")
    for h, lab in ((7, "07:00"), (12, "12:00"), (17, "17:00")):
        ax.axvspan(h, h + 1, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.text(7.5, 1.27, "recorded hours", fontsize=6.8, color=INK2, ha="left")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_ylim(0, 1.35)
    ax.set_ylabel("demand relative to\nthe calibrated peak")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.set_title("Hourly demand: level factor times the daily profile (peak = 1)", fontsize=8.4, loc="left", pad=20)
    save(fig, name, *out_dirs)


# ---------------------------------------------------------------------------------------- figure: driver profiles
def fig_driver_profiles(out_dirs=(OUT_M,), name="M_driver_profiles.png"):
    a = np.array([0.1, 0.3, 0.5, 0.7, 0.9]); k = a - 0.5
    car = dict(accel=1.5 * (1 + 0.5 * k), tau=1.2 * (1 - 0.4 * k), v0=50 * (1 + 0.2 * k))
    moto = dict(accel=2.0 * (1 + 0.4 * k), tau=0.8 * (1 - 0.4 * k), v0=50 * 0.80 * (1 + 0.2 * k))
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.2), gridspec_kw=dict(wspace=0.42))
    for ax, key, lab in ((axs[0], "accel", "max acceleration (m/s²)"), (axs[1], "tau", "desired time headway (s)"),
                         (axs[2], "v0", "mean desired speed (km/h)")):
        ax.plot(a, car[key], color=BLUE, lw=2, marker="o", ms=4.5, mec=SURF, mew=1, label="car")
        ax.plot(a, moto[key], color=ORANGE, lw=2, marker="o", ms=4.5, mec=SURF, mew=1, label="motorcycle")
        ax.set_xlabel("aggressiveness α"); ax.set_title(lab, fontsize=7.8, loc="left")
        ax.set_xticks(a); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    axs[0].legend(fontsize=7, frameon=False)
    save(fig, name, *out_dirs)


# ---------------------------------------------------------------------------------------- figure: vehicle mix
def fig_vehicle_mix(out_dirs=(OUT_M,), name="M_vehicle_mix.png"):
    shares = [0.70, 0.80]
    other = {"car": 0.19, "bus": 0.005, "truck": 0.005}
    tot_o = sum(other.values())
    fig, ax = plt.subplots(figsize=(6.2, 1.6))
    cols = {"motorcycle": ORANGE, "car": BLUE, "bus": AQUA, "truck": VIOLET}
    for i, m in enumerate(shares):
        parts = [("motorcycle", m)] + [(kk, v / tot_o * (1 - m)) for kk, v in other.items()]
        left = 0
        for kk, v in parts:
            ax.barh(i, v * 100, left=left, color=cols[kk], edgecolor=SURF, linewidth=1.5, height=0.6)
            if v >= 0.08:
                ax.text(left + v * 50, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=7, color=SURF if kk != "car" else SURF)
            left += v * 100
    ax.set_yticks(range(2)); ax.set_yticklabels(["70% (evaluation\nsensitivity run)", "80% (base,\nsite counts)"], fontsize=7.2)
    ax.set_xlim(0, 100); ax.set_xlabel("share of vehicles (%)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in cols.items()], ncol=4, fontsize=7, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.42))
    ax.set_title("Vehicle mix: the site's 80% motorcycles and the 70% sensitivity run", fontsize=8.4, loc="left", pad=18)
    save(fig, name, *out_dirs)


# ---------------------------------------------------------------------------------------- figure: yielding at the stop line
def _veh(ax, x, y, L, W, fc, ec="#444", lw=0.5, z=5, label=None):
    ax.add_patch(FancyBboxPatch((x - L, y - W / 2), L, W, boxstyle="round,pad=0,rounding_size=0.35", fc=fc, ec=ec, lw=lw, zorder=z))
    if label:
        ax.text(x - L / 2, y, label, fontsize=5.6, ha="center", va="center", color=INK, zorder=z + 1)


def fig_yielding(out_dirs=(OUT_M, OUT_P), name="F3_yielding.png"):
    lw_, carL, carW, mL, mW, aL, aW = 3.25, 4.5, 1.8, 2.0, 0.8, 6.0, 2.1
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 2.75), gridspec_kw=dict(hspace=0.25))
    for ax in axs:
        ax.set_aspect("equal"); ax.axis("off")
    # common road: heading east (right); left-hand traffic -> kerb lane on the north (top) side
    def road(ax, x0, x1, stop=None, rhk=None):
        ax.add_patch(Rectangle((x0, -lw_), x1 - x0, 2 * lw_, fc="#d9d8d3", ec="none", zorder=1))
        ax.plot([x0, x1], [lw_, lw_], color="#8a8984", lw=1.2, zorder=2)               # kerb
        ax.plot([x0, x1], [-lw_, -lw_], color="#5a5955", lw=1.2, zorder=2)             # median edge
        ax.plot([x0, (stop - (rhk or 0)) if stop else x1], [0, 0], color=SURF, lw=0.9, ls=(0, (6, 5)), zorder=2)
        ax.text(x0 + 0.5, lw_ + 0.6, "kerb", fontsize=6, color=INK2, va="bottom")
        ax.text(x0 + 0.5, -lw_ - 0.6, "median side", fontsize=6, color=INK2, va="top")
        if stop is not None:
            ax.plot([stop, stop], [-lw_, lw_], color=INK, lw=1.6, zorder=3)
            if rhk:
                ax.add_patch(Rectangle((stop - rhk, -lw_), rhk, 2 * lw_, fc="#cfe8d5", ec="none", zorder=1.5))
                ax.plot([stop - rhk, stop - rhk], [-lw_, lw_], color=INK2, lw=0.9, zorder=3)
                ax.text(stop - rhk / 2, lw_ + 0.6, "motorcycle box\n8 m", fontsize=6, color=GREEN, ha="center", va="bottom")
    # (a) moving traffic
    ax = axs[0]
    road(ax, 0, 72)
    _veh(ax, 22, -lw_ / 2, aL, aW, YELLOW, label="ambulance")
    _veh(ax, 33, -lw_ / 2, carL, carW, "#9ec5f4")
    ax.annotate("", xy=(37.5, lw_ / 2 - 0.2), xytext=(33.5, -lw_ / 2 + 0.6),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2, connectionstyle="arc3,rad=-0.25"), zorder=7)
    for xx in (8, 47, 60):
        _veh(ax, xx, lw_ / 2, carL, carW, "#cfcfca")
    for xx, yy in ((40, -0.2), (55, -lw_ / 2 + 0.3), (66, lw_ / 2 + 0.5)):
        _veh(ax, xx, yy, mL, mW, "#f6b59a")
    ax.text(30.0, -lw_ - 0.7, "gap in the kerb lane: the car ahead moves over\nif SUMO judges the lane change safe",
            fontsize=6.4, color=BLUE, va="top")
    ax.set_xlim(-1, 73); ax.set_ylim(-6.5, 6.0)
    ax.set_title("(a) Moving traffic: the car directly ahead yields where there is room", fontsize=8.2, loc="left")
    # (b) standing queue at red
    ax = axs[1]
    stop, rhk = 66.0, 8.0
    road(ax, 0, 72, stop=stop, rhk=rhk)
    # motorcycles in the box and filtering between the lanes
    rng = np.random.default_rng(3)
    for i in range(9):
        _veh(ax, stop - 0.3 - (i % 3) * 2.5, -2.3 + (i // 3) * 2.3, mL, mW, "#f6b59a")
    x = stop - rhk - 0.6
    for k in range(5):
        _veh(ax, x - k * (carL + 2.0), lw_ / 2, carL, carW, "#cfcfca")
        _veh(ax, x - k * (carL + 2.0) - 0.8, -lw_ / 2, carL, carW, "#cfcfca")
    for xx in (50.5, 44, 37.5):
        _veh(ax, xx, 0.05, mL, mW, "#f6b59a")
    _veh(ax, x - 5 * (carL + 2.0) - 1.2, -lw_ / 2, aL, aW, YELLOW, label="ambulance")
    ax.text(stop + 0.8, 0, "red", fontsize=6.6, color=SIG_R, va="center", ha="left", weight="bold")
    ax.text(30.0, -lw_ - 0.7, "boxed in: no car can move aside, so only a green\nfor this approach clears the path in time",
            fontsize=6.4, color=RED, va="top")
    ax.set_xlim(-1, 73); ax.set_ylim(-6.5, 6.0)
    ax.set_title("(b) Standing queue at red: motorcycles fill the box and the gaps", fontsize=8.2, loc="left")
    save(fig, name, *out_dirs)


# ---------------------------------------------------------------------------------------- figure: evaluation design
def fig_evaluation(out_dirs=(OUT_M,), name="M_evaluation_design.png"):
    fig, ax = plt.subplots(figsize=(6.4, 2.9)); ax.axis("off")
    _box(ax, 0.012, 0.32, 0.19, 0.36, "Seed i", "identical cars,\nmotorcycles and\nambulance requests\nfor every controller", "#f6f6f3",
         tsize=7.6, ssize=6.5)
    ctrls = [("fixed-time", "current practice"), ("max-pressure", "reference, no priority"), ("naive preemption", "green when ETA ≤ 20 s"),
             ("priority teacher", "green when ETA ≤ t_clear"), ("learned controller", "imitates the teacher")]
    ys = np.linspace(0.84, 0.0, len(ctrls))
    for (c, sub), y in zip(ctrls, ys):
        ax.add_patch(FancyBboxPatch((0.30, y), 0.24, 0.13, boxstyle="round,pad=0.006,rounding_size=0.01", fc="#fdf0ea",
                                    ec="#8a8984", lw=0.7, transform=ax.transAxes))
        ax.text(0.31, y + 0.088, c, fontsize=7.0, weight="bold", color=INK, transform=ax.transAxes, va="center")
        ax.text(0.31, y + 0.035, sub, fontsize=6.3, color=INK2, transform=ax.transAxes, va="center")
        _arrow(ax, (0.2, 0.5), (0.30, y + 0.065), color="#8a8984", lw=0.8)
        _arrow(ax, (0.54, y + 0.065), (0.63, 0.5), color="#8a8984", lw=0.8)
    _box(ax, 0.63, 0.27, 0.36, 0.46, "Paired difference per seed",
         "Δᵢ = KPI(ctrl, i) − KPI(max-pressure, i)\nreport mean Δ ± t(n−1, 0.975)·s/√n\nn ≥ 8 seeds in every cell\n"
         "cells: 2 demand levels ×\n3 motorcycle shares", "#f6f6f3", tsize=7.4, ssize=6.4)
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(-0.02, 1.0)
    save(fig, name, *out_dirs)

# ---------------------------------------------------------------------------------------- figure: amber and dilemma zone
AMBER = dict(t_r=1.0, a=3.05, Y=2.0, red_clear=2.0)      # perception-reaction time, deceleration, provided amber, red after amber
CROSS_W = dict(through_E=29.6, through_W=27.5, through_S=21.4, first_N=14.9)   # stop line to far side, metres (measured on the network)


def fig_amber(out_dirs=(OUT_M,), name="M_amber.png"):
    t_r, a, Y = AMBER["t_r"], AMBER["a"], AMBER["Y"]
    kmh = np.linspace(20, 60, 200); v = kmh / 3.6
    y_req = t_r + v / (2 * a)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw=dict(wspace=0.32, width_ratios=[1, 1.15]))
    ax = axs[0]
    ax.plot(kmh, y_req, color=BLUE, lw=2)
    ax.axhline(Y, color=SIG_A, lw=1.6)
    ax.text(21, Y + 0.06, f"the city's amber: {Y:.0f} s", fontsize=7, color=INK, va="bottom")
    ax.axhline(3.0, color=MUTED, lw=1.0, ls="--")
    ax.text(21, 3.06, "usual amber: 3 s", fontsize=7, color=INK2, va="bottom")
    for k, off, ha in ((40, (6, -12), "left"), (50, (-6, 8), "right")):
        vv = k / 3.6; yy = t_r + vv / (2 * a)
        ax.plot(k, yy, "o", ms=5, color=BLUE, mec=SURF, mew=1.2)
        ax.annotate(f"{yy:.1f} s at {k} km/h", (k, yy), xytext=off, textcoords="offset points", ha=ha, fontsize=7, color=INK)
    ax.set_xlabel("approach speed (km/h)"); ax.set_ylabel("amber needed (s)")
    ax.set_title("(a) Kinematic amber requirement", fontsize=8.2, loc="left")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_ylim(1.6, 4.2)
    # (b) zones at 50 km/h
    ax = axs[1]
    v50 = 50 / 3.6
    x_stop = v50 * t_r + v50 ** 2 / (2 * a)      # needs at least this distance to stop comfortably
    x_go = v50 * Y                                # can reach the stop line before red from at most this distance
    ax.barh(0, x_go, color="#cde8d5", height=0.5, edgecolor=SURF)
    ax.barh(0, x_stop - x_go, left=x_go, color="#f8d3d3", height=0.5, edgecolor=SURF)
    ax.barh(0, 70 - x_stop, left=x_stop, color="#dbe7f8", height=0.5, edgecolor=SURF)
    ax.text(x_go / 2, 0, "can reach the line\nbefore red", ha="center", va="center", fontsize=6.8, color=INK)
    ax.text((x_go + x_stop) / 2, 0.42, f"dilemma\nzone\n{x_stop - x_go:.1f} m", ha="center", va="bottom", fontsize=6.6, color=RED)
    ax.text((x_stop + 70) / 2, 0, "can stop\ncomfortably", ha="center", va="center", fontsize=6.8, color=INK)
    ax.set_xlim(0, 70); ax.set_ylim(-0.6, 1.2); ax.set_yticks([])
    ax.set_xlabel("distance to the stop line when amber starts (m)")
    ax.set_title(f"(b) At 50 km/h with a {Y:.0f} s amber", fontsize=8.2, loc="left")
    ax.spines["left"].set_visible(False)
    save(fig, name, *out_dirs)


# ---------------------------------------------------------------------------------------- figure: clearing time of the queue ahead
# Teacher rule: t_clear = I + l_s + (3600 / S) * sum_k e_k n_k, with S = 600 * W_e pcu/h (PKJI 2023, protected
# approaches) and e = 0.15 motorcycle, 1.0 car, 1.3 bus or truck (PKJI 2023, Table 5-2).  l_s = 2 s is an assumption.
CLEAR = dict(I=4.0, l_s=2.0, widths={"S and W (2 lanes, 6.5 m)": 6.5, "N (3 lanes, 9.75 m)": 9.75, "E (4 lanes, 13.0 m)": 13.0})


def fig_clearing(out_dirs=(OUT_M,), name="M_clearing_time.png"):
    I, l_s = CLEAR["I"], CLEAR["l_s"]
    pcu = np.linspace(0, 30, 121)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    cols = [BLUE, AQUA, VIOLET]
    for (lab, w), col in zip(CLEAR["widths"].items(), cols):
        S = 600.0 * w
        t = I + l_s + 3600.0 / S * pcu
        ax.plot(pcu, t, color=col, lw=2, label=f"{lab}: S = {S:,.0f} pcu/h")
    ax.axhline(20, color=MUTED, lw=1, ls="--")
    ax.text(0.4, 19.6, "naive rule: ETA = 20 s", fontsize=6.8, color=INK2, va="top")
    ex = 20 * 0.15 + 5 * 1.0 + 1 * 1.3
    tW = I + l_s + 3600.0 / 3900.0 * ex
    ax.plot(ex, tW, "o", ms=6, color=BLUE, mec=SURF, mew=1.2, zorder=5)
    ax.annotate(f"20 motorcycles, 5 cars, 1 bus = {ex:.1f} pcu\non a 2-lane approach: stage needed {tW:.1f} s ahead",
                (ex, tW), xytext=(0.5, 22.6), textcoords="data", fontsize=6.8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_xlabel("queue ahead of the ambulance (pcu: motorcycle 0.15, car 1.0, bus or truck 1.3)")
    ax.set_ylabel("time to clear the path (s)")
    ax.set_xlim(0, 30); ax.set_ylim(5, 34); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(fontsize=6.8, frameon=False, loc="upper left")
    ax.set_title("When the teacher asks for its stage: intergreen 4 s + start-up loss 2 s + queue discharge", fontsize=8.2, loc="left")
    save(fig, name, *out_dirs)


# ---------------------------------------------------------------------------------------- figure: KPI design
def _pill(ax, x, y, w, h, text, fc, ec="#8a8984", size=6.5, bold=False, color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008", fc=fc, ec=ec, lw=0.7,
                                transform=ax.transAxes, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, fontsize=size, ha="center", va="center", color=color or INK, transform=ax.transAxes,
            zorder=3, weight="bold" if bold else "normal", linespacing=1.2)


def fig_kpi(out_dirs=(OUT_M,), name="M_kpi_design.png"):
    fig, ax = plt.subplots(figsize=(6.6, 5.0)); ax.axis("off")
    X = {"cases": 0.005, "runs": 0.345, "score": 0.685}; W = 0.305
    heads = {"cases": "1  Every ambulance case", "runs": "2  Reference runs per case", "score": "3  Flatten, gate, choose"}
    for k, t in heads.items():
        ax.text(X[k], 0.985, t, fontsize=7.8, weight="bold", color=INK, transform=ax.transAxes, va="top")
    cases = [("Movement", "N through, E free-left, the\nwest detour; all 13 in general", "#eef3fb", 2),
             ("Arrival moment", "8 equal slices of the\ncity's 231 s cycle", "#eef3fb", 2),
             ("Demand", "low, design, high", "#eef3fb", 1),
             ("Second ambulance", "none, or a conflicting one\nwithin 30 s (stress test)", "#eef3fb", 2),
             ("Seeds", "8, shared by every method", "#eef3fb", 1)]
    runs = [("Empty road", "ambulance alone, signal off:\nthe floor T_empty", "#f2f8f3", 2),
            ("No-ambulance copy", "same seed, no ambulance:\nextra waiting and recovery", "#f2f8f3", 2),
            ("Today", "the city's fixed four-stage\nplan, no priority", "#f6f6f3", 2),
            ("Methods", "random, still, longest-queue,\nmax-pressure, naive preemption,\nteacher, learned models", "#fdf0ea", 3),
            ("Hindsight ceiling", "best switch time, replayed\nfrom detection (one seed)", "#f6f6f3", 2)]
    score = [("Per movement", "D = T − T_empty, averaged\nover slices, levels, seeds", "#f6f6f3", 2),
             ("Flatten", "equal weight per movement;\nshow the worst one", "#f6f6f3", 2),
             ("Other traffic", "each approach's delay ÷ today's;\ngeometric mean and worst", "#f6f6f3", 2),
             ("Gates", "no collision; ≤ 10% over max-\npressure; no approach more\nthan 20% worse than today", "#fdf0ea", 3),
             ("Equilibrium", "Pareto front, then the largest\nNash product; bootstrap", "#fdf0ea", 2)]
    H = {1: 0.105, 2: 0.135, 3: 0.165}
    for key, items in (("cases", cases), ("runs", runs), ("score", score)):
        y = 0.905
        for t, sub, fc, n in items:
            _box(ax, X[key], y - H[n], W, H[n], t, sub, fc, tsize=7.0, ssize=6.3); y -= H[n] + 0.02
    for a_, b_ in (("cases", "runs"), ("runs", "score")):
        _arrow(ax, (X[a_] + W + 0.004, 0.47), (X[b_] - 0.004, 0.47), color=MUTED, lw=1.0)
    ax.text(X["cases"], 0.115, "Core grid: 3 × 8 × 3 × 8\n= 576 passages per method.\nEvery method meets the same\nambulances at the same moments.",
            fontsize=6.5, color=INK2, transform=ax.transAxes, va="top", linespacing=1.3)
    ax.set_xlim(-0.012, 1.0); ax.set_ylim(0.0, 1.0)
    save(fig, name, *out_dirs)


# ---------------------------------------------------------------------------------------- figure: supervised learning design
def fig_supervised(out_dirs=(OUT_M,), name="M_supervised.png"):
    fig, ax = plt.subplots(figsize=(6.8, 4.7)); ax.axis("off")
    xs = [0.005, 0.258, 0.511, 0.764]; W = 0.226
    heads = ["1  Ground truth", "2  Validators", "3  Models", "4  Closed loop"]
    for x, t in zip(xs, heads):
        ax.text(x, 0.985, t, fontsize=7.8, weight="bold", color=INK, transform=ax.transAxes, va="top")
    gt = [("A  Teacher's choice", "by construction: the\npriority rule on the\nsame sensor columns", "#eef3fb"),
          ("B  Queue in 30 s", "by measurement: the\ncamera reading 30 s\nlater", "#eef3fb"),
          ("C  Arrival time", "by measurement: the\ntracker's stop-line\ntime (planned)", "#eef3fb"),
          ("D  Best in hindsight", "by replay: the switch\nthat worked best\n(subset, planned)", "#f6f6f3")]
    val = [("Schema", "columns, units and\nranges; vehicles in\nmatch vehicles out", "#f6f6f3"),
           ("Recompute labels", "A from the logged\npressures, B from the\nreading 30 s on: 100%", "#f6f6f3"),
           ("Leakage", "no future columns;\nno controller request;\nshuffled labels fail", "#f6f6f3"),
           ("Grouped splits", "by episode and seed;\nhold out a window\nand high demand", "#f6f6f3")]
    mod = [("Baselines", "random, majority and\nstillness: keep the\nstage, queue stays", "#fdf0ea"),
           ("Interpretable", "logistic regression,\ndecision tree", "#fdf0ea"),
           ("Ensembles", "random forest,\ngradient boosting", "#fdf0ea"),
           ("Neural", "small multilayer\nperceptron", "#fdf0ea")]
    cl = [("Drive the twin", "each model behind\nthe shield on the\ncase grid", "#f2f8f3"),
          ("Normalised score", "0 = random,\n1 = teacher", "#f2f8f3"),
          ("Fix drift", "teacher labels the\nstates the model\nvisits; retrain", "#f2f8f3"),
          ("Report", "offline and closed-\nloop scores side\nby side", "#f6f6f3")]
    for x, items in zip(xs, (gt, val, mod, cl)):
        y = 0.905
        for t, sub, fc in items:
            _box(ax, x, y - 0.185, W, 0.185, t, sub, fc, tsize=6.9, ssize=6.3); y -= 0.207
    for i in range(3):
        _arrow(ax, (xs[i] + W + 0.003, 0.49), (xs[i + 1] - 0.003, 0.49), color=MUTED, lw=1.0)
    ax.set_xlim(-0.012, 1.0); ax.set_ylim(0.0, 1.0)
    save(fig, name, *out_dirs)


FIGS = {"study_area": fig_study_area, "architecture": fig_architecture, "signal_timeline": fig_signal_timeline,
        "daily_demand": fig_daily_demand, "driver_profiles": fig_driver_profiles, "vehicle_mix": fig_vehicle_mix,
        "yielding": fig_yielding, "evaluation": fig_evaluation, "amber": fig_amber, "clearing": fig_clearing,
        "kpi": fig_kpi, "supervised": fig_supervised}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for k in which:
        FIGS[k]()
        print("wrote", k)
