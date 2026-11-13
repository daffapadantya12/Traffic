"""Assemble the presentation into ONE self-contained HTML file: slides, charts, Three.js and the recorded 3D replays.

    python3 reports/a3/presentation/build_deck.py                 # src/ + clips/ + vendor/ -> Green_for_the_Ambulance.html

The replay clips in clips/ are cut from recordings (record_fatmawati.py, record_bundaran.py); to cut them again:

    python3 reports/a3/presentation/build_deck.py --fat-rec <fixed.jsonl.gz> <learned.jsonl.gz> --fat-ev ev1234
    python3 reports/a3/presentation/build_deck.py --bhi-rec <rec_bundaran_hi.jsonl.gz> --bhi-geo <bundaran_hi_geometry.json>

Chart numbers are read from the repository's own outputs (reports/a3/kpi*, reports/a3/learning/results.json,
data/fatmawati/calibration.json, data/fatmawati_v3), so the slides cannot drift from the report.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC, CLIPS, VENDOR = os.path.join(HERE, "src"), os.path.join(HERE, "clips"), os.path.join(HERE, "vendor")
OUT = os.path.join(HERE, "Green_for_the_Ambulance.html")
STAGES = {"S1": "west + east through", "S2": "east through + right", "S3": "north", "S4": "south"}
FROM = {"N": "from the north", "E": "from the east", "S": "from the south", "W": "from the west"}


# ---------------------------------------------------------------------------------------------- small helpers
def read_frames(path: str, t0: float = -1e18, t1: float = 1e18) -> list:
    out = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            f = json.loads(line)
            if f["t"] > t1:
                break
            if f["t"] >= t0:
                out.append(f)
    return out


def write_gz(name: str, obj) -> str:
    os.makedirs(CLIPS, exist_ok=True)
    path = os.path.join(CLIPS, name)
    raw = json.dumps(obj, separators=(",", ":")).encode()
    with open(path, "wb") as fh:
        fh.write(gzip.compress(raw, 9, mtime=0))
    print(f"  {name}: {len(raw) / 1e6:.2f} MB json -> {os.path.getsize(path) / 1e6:.2f} MB gz")
    return path


def b64_file(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def runs(points, radius: float):
    """Index ranges of a polyline inside the radius, each widened by one point so the road reaches the edge."""
    inside = [math.hypot(p[0], p[1]) <= radius for p in points]
    out, i, n = [], 0, len(points)
    while i < n:
        if inside[i]:
            j = i
            while j + 1 < n and inside[j + 1]:
                j += 1
            a, b = max(i - 1, 0), min(j + 1, n - 1)
            if b > a:
                out.append((a, b + 1))
            i = j + 1
        else:
            i += 1
    return out


def r1(p):
    return [round(p[0], 1), round(p[1], 1)] + ([round(p[2], 2)] if len(p) > 2 and p[2] else [])


def centroid_in(poly, radius: float) -> bool:
    return math.hypot(sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)) <= radius


# ---------------------------------------------------------------------------------------------- geometry
def crop_geometry(g: dict, radius: float) -> dict:
    """The viewer's static scene cut to a disc around the junction (the full file covers 6 x 7.5 km)."""
    lanes = []
    for ln in g["lanes"]:
        for k, (a, b) in enumerate(runs(ln["p"], radius)):
            lanes.append(dict(id=ln["id"], w=ln["w"], k=ln["k"], idx=ln["idx"], run=k, p=[r1(p) + ([0] if len(r1(p)) < 3 else []) for p in ln["p"][a:b]]))
    out = dict(lanes=lanes,
               junctions=[dict(z=j.get("z", 0), p=[r1(p) for p in j["p"]]) for j in g["junctions"] if any(math.hypot(p[0], p[1]) <= radius for p in j["p"])],
               stoplines=[s for s in g["stoplines"] if math.hypot(*s["a"]) <= radius],
               heads=[h for h in g["heads"] if math.hypot(h["x"], h["y"]) <= radius],
               buildings=[dict(h=b["h"], p=[r1(p) for p in b["p"]], **({"hosp": True} if b.get("hosp") else {}))
                          for b in g["buildings"] if len(b["p"]) >= 3 and centroid_in(b["p"], radius)])
    for key in ("hospital", "monument", "pool", "entries", "attribution"):
        if g.get(key) is not None:
            out[key] = g[key]
    F = g.get("flyover")
    if F:
        decks = [dict(hw=d["hw"], p=[[round(p[0], 1), round(p[1], 1), round(p[2], 2)] for p in d["p"][a:b]]) for d in F["decks"] for a, b in runs(d["p"], radius)]
        out["flyover"] = dict(depth=F["depth"], parapet=F["parapet"], parapet_w=F["parapet_w"], decks=decks,
                              piers=[p for p in F["piers"] if math.hypot(p[0], p[1]) <= radius],
                              lamps=[p for p in F["lamps"] if math.hypot(p[0], p[1]) <= radius])
    M = g.get("mrt")
    if M:
        decks = [dict(p=[r1(p) for p in d["p"][a:b]], side=d["side"][a:b], inner=d["inner"][a:b]) for d in M["decks"] for a, b in runs(d["p"], radius)]
        out["mrt"] = dict(rail=M["rail"], top=M["top"], under=M["under"], half=M["half"], decks=decks,
                          piers=[p for p in M["piers"] if math.hypot(p[0], p[1]) <= radius],
                          masts=[p for p in M["masts"] if math.hypot(p[0], p[1]) <= radius],
                          stations=[s for s in M["stations"] if any(math.hypot(p[0], p[1]) <= radius for p in s["p"])])
    return out


# ---------------------------------------------------------------------------------------------- clips
def pack(rows, idmap: dict, radius: float, keep: int | None) -> list:
    """[id, x*10, y*10, angle, code + 8*(z*10)] per vehicle, flat."""
    flat, r2 = [], radius * radius
    for r in rows:
        if r[1] * r[1] + r[2] * r[2] > r2 and r[0] != keep:
            continue
        flat += [idmap[r[0]], round(r[1] * 10), round(r[2] * 10), round(r[4]) % 360, int(r[5]) + 8 * max(0, round(r[3] * 10))]
    return flat


def maxima(V: list) -> dict:
    mx = {}
    for flat in V:
        n = {}
        for k in range(4, len(flat), 5):
            c = flat[k] & 7
            n[c] = n.get(c, 0) + 1
        for c, v in n.items():
            mx[c] = max(mx.get(c, 0), v)
    return {str(c): v + 8 for c, v in mx.items()}


def ambulance_track(path: str, num: int) -> dict:
    """When the ambulance came within 300 m of the junction, and when it was 30 m past the centre."""
    pts = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            f = json.loads(line)
            for i, x, y, sp in f["ev"]:
                if i == num:
                    pts.append((f["t"], x, y, sp))
    near = [k for k, p in enumerate(pts) if math.hypot(p[1], p[2]) <= 300.0]
    if not near:
        raise SystemExit(f"ambulance {num} never came within 300 m of the junction in {path}")
    k0 = near[0]
    ang = math.degrees(math.atan2(pts[k0][1], pts[k0][2])) % 360.0
    leg = "N" if ang < 45 or ang >= 315 else "E" if ang < 135 else "S" if ang < 225 else "W"
    travelled, cross = 0.0, None
    for k in range(k0 + 1, len(pts)):
        travelled += math.hypot(pts[k][1] - pts[k - 1][1], pts[k][2] - pts[k - 1][2])
        if travelled >= 330.0:
            cross = pts[k][0]
            break
    if cross is None:
        raise SystemExit(f"ambulance {num} did not pass the junction in {path}")
    return dict(leg=leg, detect=pts[k0][0], cross=cross, stopped=sum(0.5 for p in pts[k0:] if p[0] <= cross and p[3] < 0.5))


def cut_fatmawati(recs: list, ev: str, lead: float, tail: float, radius: float, geo_radius: float, step: int, start_hour: float, seed: int) -> None:
    num = 10_000_000 + int(ev[2:])
    labels = [("fixed", "Today's fixed plan (Dishub)", "Today's plan"), ("learned_gb", "Learned priority (gradient boosting)", "Learned priority")]
    tracks = [ambulance_track(p, num) for p in recs]
    t0 = min(t["detect"] for t in tracks) - lead
    t1 = max(t["cross"] for t in tracks) + tail
    frames = [read_frames(p, t0, t1)[::step] for p in recs]
    n = min(len(f) for f in frames)
    frames = [f[:n] for f in frames]
    ids = sorted({r[0] for fs in frames for f in fs for r in f["veh"]})
    idmap = {v: k + 1 for k, v in enumerate(ids)}
    dt = 0.5 * step
    print(f"Fatmawati clip: {ev} {FROM[tracks[0]['leg']]}, t {frames[0][0]['t']:.0f}-{frames[0][-1]['t']:.0f} s, {n} frames at {1 / dt:g} Hz")
    for (key, label, short), fs, tr in zip(labels, frames, tracks):
        tls, ph = [], []
        F, V = [], []
        for f in fs:
            if f["tls"] not in tls:
                tls.append(f["tls"])
            if f["phase"] not in ph:
                ph.append(f["phase"])
            cams = f.get("cams") or {}
            q = [int(cams.get(a, [f["q"].get(a, 0)])[0]) for a in "NESW"]
            F.append([tls.index(f["tls"]), ph.index(f["phase"])] + q)
            V.append(pack(f["veh"], idmap, radius, num))
        first = fs[0]["t"]
        clip = dict(key=key, label=label, short=short, dt=dt, clock0=start_hour * 3600.0 + first, tls=tls, ph=ph, stages=STAGES,
                    qLabel="Standing vehicles seen by the queue cameras", F=F, V=V, max=maxima(V), seed=seed, leg=tr["leg"],
                    ev=dict(id=idmap[num], **{"from": FROM[tr["leg"]]}, detect=round((tr["detect"] - first) / dt),
                            cross=round((tr["cross"] - first) / dt), seconds=round(tr["cross"] - tr["detect"]), stopped=round(tr["stopped"])))
        print(f"  {key}: 300 m out -> through the junction in {tr['cross'] - tr['detect']:.0f} s (standing {tr['stopped']:.0f} s)")
        write_gz(f"fatmawati_{key}.json.gz", clip)
    with open(os.path.join(ROOT, "data", "fatmawati", "viewer", "geometry.json")) as fh:
        write_gz("fatmawati_geometry.json.gz", crop_geometry(json.load(fh), geo_radius))


def cut_bundaran(rec: str, geo: str, start: float, seconds: float, radius: float, geo_radius: float, step: int, start_hour: float) -> None:
    fs = read_frames(rec)
    t0 = fs[0]["t"] + start
    fs = [f for f in fs if t0 <= f["t"] <= t0 + seconds][::step]
    ids = sorted({r[0] for f in fs for r in f["veh"]})
    idmap = {v: k + 1 for k, v in enumerate(ids)}
    tls, F, V = [], [], []
    for f in fs:
        s = "|".join(f["tls"])
        if s not in tls:
            tls.append(s)
        F.append([tls.index(s), 0] + [int(f["legs"].get(a, [0, 0])[1]) for a in "NES"])
        V.append(pack(f["veh"], idmap, radius, None))
    clip = dict(key="bhi", label="Uncalibrated demo: assumed demand and signals", short="Demo", dt=0.5 * step,
                clock0=start_hour * 3600.0 + fs[0]["t"], tls=tls, ph=[], stages={}, qLabel="Standing vehicles within 300 m of the ring",
                F=F, V=V, max=maxima(V), ev=None)
    print(f"Bundaran HI clip: {len(fs)} frames")
    write_gz("bundaran_hi.json.gz", clip)
    with open(geo) as fh:
        write_gz("bundaran_hi_geometry.json.gz", crop_geometry(json.load(fh), geo_radius))


# ---------------------------------------------------------------------------------------------- chart data
NAMES = {"fixed": "Fixed plan (today)", "longest_queue": "Longest queue", "max_pressure": "Max-pressure", "priority_mp": "Priority teacher",
         "preempt_naive": "Naive pre-emption", "random": "Random", "still": "Still",
         "decision_tree": "Learned: decision tree", "gradient_boosting": "Learned: gradient boosting", "logistic_regression": "Learned: logistic regression",
         "neural_network": "Learned: neural network", "random_forest": "Learned: random forest"}


def mkey(method: str) -> str:
    m = re.search(r"taskA_(\w+)\.joblib", method)
    return m.group(1) if m else method


def rows(path: str) -> list:
    with open(os.path.join(ROOT, path), newline="") as fh:
        return list(csv.DictReader(fh))


def num(v, nd=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return round(x, nd) if nd is not None else x


def chart_data() -> dict:
    with open(os.path.join(ROOT, "reports", "a3", "kpi", "equilibrium.json")) as fh:
        eq = json.load(fh)
    nash = {mkey(k): v for k, v in eq["point_estimate"]["nash"].items()}
    passing = {mkey(k) for k in eq["point_estimate"]["passing"]}
    low = []
    for r in rows("reports/a3/kpi/kpi_methods.csv"):
        k = mkey(r["method"])
        low.append(dict(key=k, name=NAMES[k], A=num(r["A"], 1), A_lo=num(r["A_lo"], 1), A_hi=num(r["A_hi"], 1), R=num(r["R"], 3),
                        R_lo=num(r["R_lo"], 3), R_hi=num(r["R_hi"], 3), worst=num(r["R_worst"], 3), worst_leg=r["R_worst_approach"],
                        red=max(num(r[f"longest_red_{a}"]) for a in "NESW"), coll=num(r["collisions_per_1000_veh_km_study"], 3),
                        gain=num(r["G_vs_base"], 3), rep=num(r["A_on_replayed"], 1), teacher=num(r["share_of_teacher_gain"], 3), safe=r["safe"] == "True",
                        passes=k in passing, nash=round(nash[k], 4) if k in nash else None))
    by = {"design": [], "high": []}
    for r in rows("reports/a3/kpi/kpi_by_level.csv"):
        if r["level"] in by:
            by[r["level"]].append(dict(key=mkey(r["method"]), name=NAMES[mkey(r["method"])], A=num(r["A"], 0), worst=num(r["R_worst"], 2), unf=num(r["unfinished_share"], 3)))
    moto = [dict(key=mkey(r["method"]), name=NAMES[mkey(r["method"])], A=num(r["A"], 1), worst=num(r["R_worst"], 2)) for r in rows("reports/a3/kpi_moto70/kpi_by_level.csv")]
    mp = next(m for m in low if m["key"] == "max_pressure")
    hind_row = next(r for r in rows("reports/a3/kpi/kpi_methods.csv") if r["method"] == "max_pressure")
    with open(os.path.join(ROOT, "reports", "a3", "learning", "results.json")) as fh:
        ta = json.load(fh)["task_a"]
    model_names = {"gradient_boosting": "Gradient boosting", "neural_network": "Neural network", "logistic_regression": "Logistic regression",
                   "random_forest": "Random forest", "decision_tree": "Decision tree"}
    base_names = {"stillness": "Keep current stage", "random_uniform": "Random stage", "majority": "Most common stage"}
    models = [dict(name=n, f1=round(ta["models"][k]["macro_f1"], 3), amb=round(ta["models"][k]["ambulance_accuracy"], 3), kind="model") for k, n in model_names.items()]
    models += [dict(name=n, f1=round(ta["baselines"][k]["macro_f1"], 3), amb=round(ta["baselines"][k]["ambulance_accuracy"], 3), kind="baseline") for k, n in base_names.items()]
    gen = {k: round(v["unseen"]["macro_f1"], 3) for k, v in ta["generalisation"].items()}
    with open(os.path.join(ROOT, "data", "fatmawati", "calibration.json")) as fh:
        cal = json.load(fh)
    ver = cal["verification"]
    calib = [dict(leg=a, target=cal["target_max_queue_m"][a], twin=ver["max_queue_m"][a], sd=ver["max_queue_sd_m"][a], err=round(100 * cal["rel_error"][a]),
                  share=round(100 * ver["volume_share_of_observed"][a]), limited=cal["capacity_limited"][a]) for a in "NESW"]
    # the dataset itself: vehicle mix at the entry gates, and one hour of queues (fixed plan, design level, 17:00, first seed)
    d = os.path.join(ROOT, "data", "fatmawati_v3")
    mix = {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0}
    with open(os.path.join(d, "loops_1min.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "entry_gate":
                for k in mix:
                    mix[k] += int(r[f"n_{k}"])
    total = sum(mix.values())
    with open(os.path.join(d, "episodes.csv"), newline="") as fh:
        ep = next(r["episode_id"] for r in csv.DictReader(fh) if r["controller"] == "fixed" and r["level"] == "design" and r["window"] == "pm_peak")
    queues = {a: [] for a in "NESW"}
    with open(os.path.join(d, "ticks_5s.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["episode_id"] == ep:
                for a in "NESW":
                    queues[a].append(int(float(r[f"q_veh_{a}"])))
    return dict(low=low, win={NAMES[mkey(k)] if mkey(k) in NAMES else "No method": round(100 * v, 1) for k, v in eq["bootstrap"]["win_share"].items()},
                **{"lambda": [round(eq["implied_lambda"]["towards_less_ambulance_priority"]["lambda_veh_s_per_amb_s"], 1),
                              round(eq["implied_lambda"]["towards_more_ambulance_priority"]["lambda_veh_s_per_amb_s"], 1)]},
                stress=by, moto70=moto, hind=dict(base=num(hind_row["A_on_replayed"], 1), ceiling=num(hind_row["A_ceiling"], 1)), cost_gate=round(1.10 * mp["R"], 3),
                models=models, gen=gen, shuffled=round(ta["shuffled_label_test"]["macro_f1"], 3), calib=calib,
                mix={k: round(100 * v / total, 1) for k, v in mix.items()}, mix_n=total, queues=queues)


# ---------------------------------------------------------------------------------------------- assemble
def assemble() -> None:
    def src(name):
        with open(os.path.join(SRC, name), encoding="utf-8") as fh:
            return fh.read()

    def gzb64(path):
        with open(path, "rb") as fh:
            text = re.sub(rb"\n//# sourceMappingURL=.*", b"", fh.read())
        return base64.b64encode(gzip.compress(text, 9, mtime=0)).decode()

    vendor = dict(core=gzb64(os.path.join(VENDOR, "three.core.min.js")), module=gzb64(os.path.join(VENDOR, "three.module.min.js")),
                  orbit=gzb64(os.path.join(VENDOR, "OrbitControls.js")), map=gzb64(os.path.join(VENDOR, "MapControls.js")),
                  bgu=gzb64(os.path.join(VENDOR, "BufferGeometryUtils.js")))
    clip = lambda n: b64_file(os.path.join(CLIPS, n))
    twin = dict(fat=dict(geo=clip("fatmawati_geometry.json.gz"), clips=[clip("fatmawati_fixed.json.gz"), clip("fatmawati_learned_gb.json.gz")]),
                bhi=dict(geo=clip("bundaran_hi_geometry.json.gz"), clips=[clip("bundaran_hi.json.gz")]))
    data = chart_data()
    safe = lambda o: json.dumps(o, separators=(",", ":")).replace("<", "\\u003c")
    html = src("shell.html")
    evs = []
    for name in ("fatmawati_fixed.json.gz", "fatmawati_learned_gb.json.gz"):
        with gzip.open(os.path.join(CLIPS, name), "rt") as fh:
            c = json.load(fh)
        evs.append((c["ev"], c["seed"], c["leg"]))
    (a, seed, leg), (b, _, _) = evs
    note = (f"One passage chosen to show the mechanism (low demand, seed {seed}): {a['seconds']} s under today's plan, {b['seconds']} s with the learned controller. "
            f"Average over 96 passages: 14% less delay; some get worse.")
    slides = src("slides.html").replace("@FAT_CAM@", {"N": "north", "W": "west"}.get(leg, "junction")).replace("@FAT_NOTE@", note).replace("@IMG_F1@", "data:image/png;base64," + b64_file(os.path.join(ROOT, "reports", "a3", "figures", "F1_study_area.png")))
    parts = {"/*@CSS*/": src("deck.css"), "<!--@SLIDES-->": slides, "/*@JS*/": "\n".join(src(n) for n in ("viewer.js", "charts.js", "engine.js")),
             '"@VENDOR"': safe(vendor), '"@TWIN"': safe(twin), '"@DATA"': safe(data)}
    for mark, text in parts.items():
        if mark not in html:
            raise SystemExit(f"shell.html has no {mark} marker")
        html = html.replace(mark, text, 1)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"{os.path.relpath(OUT, ROOT)}: {os.path.getsize(OUT) / 1e6:.2f} MB "
          f"(Three.js {sum(len(v) for v in vendor.values()) / 1e6:.2f}, replays {sum(len(c) for s in twin.values() for c in [s['geo']] + s['clips']) / 1e6:.2f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fat-rec", nargs=2, metavar=("FIXED", "LEARNED"), help="recordings of record_fatmawati.py, same seed")
    ap.add_argument("--fat-ev", help="the ambulance to feature, e.g. ev1234")
    ap.add_argument("--fat-seed", type=int, default=1, help="the seed of those recordings (shown on the slide)")
    ap.add_argument("--lead", type=float, default=40.0, help="seconds before the ambulance is 300 m out")
    ap.add_argument("--tail", type=float, default=30.0, help="seconds after it has passed the junction")
    ap.add_argument("--radius", type=float, default=520.0, help="vehicles kept within this distance of the junction")
    ap.add_argument("--geo-radius", type=float, default=680.0)
    ap.add_argument("--step", type=int, default=1, help="keep every n-th recorded frame (recordings are at 2 Hz)")
    ap.add_argument("--start-hour", type=float, default=7.0)
    ap.add_argument("--bhi-rec")
    ap.add_argument("--bhi-geo")
    ap.add_argument("--bhi-from", type=float, default=60.0)
    ap.add_argument("--bhi-seconds", type=float, default=240.0)
    ap.add_argument("--bhi-radius", type=float, default=460.0)
    args = ap.parse_args()
    if args.fat_rec:
        if not args.fat_ev:
            ap.error("--fat-rec needs --fat-ev")
        cut_fatmawati(args.fat_rec, args.fat_ev, args.lead, args.tail, args.radius, args.geo_radius, args.step, args.start_hour, args.fat_seed)
    if args.bhi_rec:
        if not args.bhi_geo:
            ap.error("--bhi-rec needs --bhi-geo")
        cut_bundaran(args.bhi_rec, args.bhi_geo, args.bhi_from, args.bhi_seconds, args.bhi_radius, 620.0, args.step, args.start_hour)
    assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())
