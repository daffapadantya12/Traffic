"""Record the Fatmawati twin's viewer frames to a file, for the replay baked into the presentation.

It runs the same world as the live 3D viewer (traffic_twin/fatmawati/webapp3d.py: same network, demand, JORR
through-traffic and MRT timetable) without the web server, and writes one JSON line per 0.5 s simulation step,
keeping only the vehicles within --radius metres of the junction.  Run it once per controller on the same seed:
the trips are identical, so the recordings differ only through the signal control.

    python3 reports/a3/presentation/record_fatmawati.py --prepare --work <dir>
    python3 reports/a3/presentation/record_fatmawati.py --controller fixed      --work <dir>
    python3 reports/a3/presentation/record_fatmawati.py --controller learned_gb --work <dir>

--scan writes no frames, only one line per ambulance (when it came within 300 m of the junction, when it crossed,
how long it stood): use it to pick a passage, then record that window with --from-min and --minutes.  The trip
files are reused when they exist, so keep --work and --seed and never shorten the first run of a seed.

Nothing under data/ is written: the trip files and the recordings go to --work.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LEARNED = {"learned_gb": os.path.join("reports", "a3", "learning", "models", "taskA_gradient_boosting.joblib")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="folder for the trip files and the recordings")
    ap.add_argument("--prepare", action="store_true", help="only write the trip files, then stop")
    ap.add_argument("--controller", default="fixed")
    ap.add_argument("--level", default="low", choices=["low", "design", "high"])
    ap.add_argument("--start-hour", type=float, default=7.0)
    ap.add_argument("--warmup-min", type=float, default=10.0)
    ap.add_argument("--minutes", type=float, default=50.0, help="simulated minutes after the warm-up")
    ap.add_argument("--from-min", type=float, default=0.0, help="start writing frames this many minutes after the warm-up")
    ap.add_argument("--scan", action="store_true", help="no frames: a summary of every ambulance passage")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--radius", type=float, default=700.0)
    ap.add_argument("--toll-vph", type=float, default=None)
    args = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from traffic_twin.fatmawati.demand import DATA, routes_xml
    from traffic_twin.fatmawati.jorr import PEAK_VPH, routes_xml as toll_routes_xml
    from traffic_twin.fatmawati.mrt import Trains

    os.makedirs(args.work, exist_ok=True)
    total_s = 60.0 * (args.warmup_min + args.minutes)
    tag = f"{args.level}_{int(args.start_hour):02d}h_s{args.seed}"
    routes = os.path.join(args.work, f"trips_{tag}.rou.xml")
    toll = os.path.join(args.work, f"jorr_{tag}.rou.xml")
    if args.prepare or not os.path.exists(routes):
        info = routes_xml(routes, args.level, start_hour=args.start_hour, duration_s=total_s, seed=args.seed)
        toll_routes_xml(toll, args.start_hour, total_s, peak_vph=PEAK_VPH if args.toll_vph is None else args.toll_vph)
        print(f"trips: {info['n_vehicles']} vehicles, {info['n_ambulances']} ambulances over {total_s / 60:g} min", flush=True)
        if args.prepare:
            return 0

    from traffic_twin.fatmawati.sumo_world import FatmawatiWorld
    with open(os.path.join(DATA, "viewer", "geometry.json")) as fh:
        line = json.load(fh).get("mrt")
    trains = Trains(line["tracks"]) if line else None
    controller = f"learned:{LEARNED[args.controller]}" if args.controller in LEARNED else args.controller
    w = FatmawatiWorld(routes + "," + toll, start_hour=args.start_hour, seed=args.seed, controller=controller,
                       sensors=True, store_rows=False)
    if args.scan:
        return scan(w, args, total_s, os.path.join(args.work, f"scan_{args.controller}_{tag}.json"))
    out = os.path.join(args.work, f"rec_{args.controller}_{tag}.jsonl.gz")
    r2 = args.radius ** 2
    wall0, seen_events, n = time.perf_counter(), set(), 0
    with gzip.open(out, "wt", compresslevel=5) as fh:
        while w.t < total_s:
            w.step()
            if w.t < 60.0 * (args.warmup_min + args.from_min):
                continue
            f = w.frame()
            rows = f["vehicles"] + (trains.rows(args.start_hour * 3600.0 + w.t) if trains else [])
            ev = [[r[0], r[1], r[2], r[6]] for r in rows if r[5] == 1]            # every ambulance, wherever it is
            near = [r for r in rows if r[1] * r[1] + r[2] * r[2] <= r2]
            new = [e for e in w.events if "ambulance" in e and e not in seen_events]   # each starts with its clock time
            seen_events.update(new)
            cams = {a: [c.get("q_veh", 0), c.get("moto_box", 0)] for a, c in f["cams"].items()}
            fh.write(json.dumps(dict(t=f["t"], clock=f["clock"], tls=f["tls"], green=f["green"], phase=f["phase_name"],
                                     q=f["kpi"]["queues"], cams=cams, ev=ev, veh=near, events=new,
                                     ev_arrived=f["kpi"]["ev_arrived"]), separators=(",", ":")) + "\n")
            n += 1
            if n % 600 == 0:
                print(f"{args.controller}: {f['clock']}  {len(near)} of {len(rows)} vehicles in range, "
                      f"{w.t / max(time.perf_counter() - wall0, 1e-6):.1f}x real time", flush=True)
    w.close()
    print(f"{args.controller}: wrote {n} frames to {out} ({os.path.getsize(out) / 1e6:.1f} MB)", flush=True)
    return 0


def scan(w, args, total_s: float, out: str) -> int:
    """One record per ambulance: its track near the junction, without writing frames."""
    ts, track = w.ts, {}
    while w.t < total_s:
        w.step()
        for vid in ts.vehicle.getIDList():
            if vid.startswith("ev"):
                x, y = ts.vehicle.getPosition(vid)
                track.setdefault(vid, []).append((w.t, x - w.x0, y - w.y0, ts.vehicle.getSpeed(vid)))
    w.close()
    rows = []
    for vid, tr in track.items():
        near = [p for p in tr if math.hypot(p[1], p[2]) <= 300.0]
        if not near:
            continue
        ang = math.degrees(math.atan2(near[0][1], near[0][2])) % 360.0
        leg = "N" if ang < 45 or ang >= 315 else "E" if ang < 135 else "S" if ang < 225 else "W"
        centre = min(near, key=lambda p: math.hypot(p[1], p[2]))
        before = [p for p in near if p[0] <= centre[0]]
        rows.append(dict(id=vid, leg=leg, depart=tr[0][0], in300=near[0][0], centre=centre[0], end=tr[-1][0],
                         to_centre_s=centre[0] - near[0][0], stopped_s=sum(0.5 for p in before if p[3] < 0.5),
                         finished=tr[-1][0] < total_s - 1.0))
    with open(out, "w") as fh:
        json.dump(dict(controller=args.controller, seed=args.seed, level=args.level, ambulances=rows), fh, indent=1)
    print(f"{args.controller} seed {args.seed}: {len(rows)} ambulance passages -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
