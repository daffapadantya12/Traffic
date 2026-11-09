"""Uncalibrated SUMO demo of Bundaran HI. NOT part of the traffic_twin study and NOT evidence about the real junction.

Real:     the road geometry (OpenStreetMap extract in data/bundaran_hi_demo/osm/, (c) OpenStreetMap contributors, ODbL).
Assumed:  every volume, turning share and the vehicle mix (see DEMAND below), and the signals: OSM tags no signal on the
          ring, so SUMO's automatic program is placed on the three ring entries.
Left out: the TransJakarta busway (highway=busway is not in the kept road classes) and the 16:00-21:00 movement ban
          (the demo stands for a morning hour).

    python -I scripts/bundaran_hi_demo.py [--scale 1.0] [--sim-s 1800] [--warm-s 900] [--replay-s 300] [--no-signals]

The study's own loader (traffic_twin.fatmawati.site) refuses this site until the survey fields of
data/sites/bundaran_hi.json are filled; this script deliberately does not touch that file or data/bundaran_hi/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict

import sumo
import sumolib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO = os.path.join(ROOT, "data", "bundaran_hi_demo")
OSM = os.path.join(DEMO, "osm", "bundaran_hi_roads.osm.xml")
TYPEMAP = os.path.join(ROOT, "data", "fatmawati", "patches", "jakarta.typ.xml")
VTYPES_SRC = os.path.join(ROOT, "data", "fatmawati", "demand", "vtypes.add.xml")
CENTRE_LON, CENTRE_LAT = 106.823, -6.195
ROAD_TYPES = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
              "secondary_link", "tertiary", "tertiary_link"]
LEG_OF_NAME = {"Jalan Mohammad Husni Thamrin": "N", "Jalan Imam Bonjol": "E", "Jalan Jenderal Sudirman": "S",
               "Jalan Haji Mohammad Saleh Ishak": "W"}

# DEMAND: ASSUMPTIONS, not observations.  Total entering flow = 98,000 veh/day (ANTARA 2022, two-way, period not stated,
# read here as a daily total) x 7 % peak-hour share (assumed) = 6,860 veh/h, split by entry lanes (N 3, S 3, E 2).
TOTAL_VPH = 98000 * 0.07
ENTRY_LANES = {"N": 3, "S": 3, "E": 2}
TURN_SHARE = {"N": {"S": 0.55, "E": 0.20, "W": 0.20, "N": 0.05},
              "S": {"N": 0.55, "E": 0.25, "W": 0.15, "S": 0.05},
              "E": {"N": 0.35, "S": 0.35, "W": 0.25, "E": 0.05}}
MIX = {"moto": 0.70, "car": 0.27, "bus": 0.015, "truck": 0.015}   # 0.70 motorcycle: Jakarta-wide fallback in the site spec
# netconvert lets only the outermost ring lane reach each exit, which turns the 4-lane ring into a 1-lane bottleneck.
# ASSUMED lane use (the real markings are unknown): this many ring lanes, counted from the outside, leave at each exit.
RING_LANES_LEAVING = {"N": 3, "S": 3, "E": 1, "W": 1}


def run(cmd: list) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, SUMO_HOME=sumo.SUMO_HOME))
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])


def write_exit_connections(net, ring_ids: set, exits: dict, path: str) -> None:
    """Ring lanes 0..k-1 leave at the exit, the others carry on, so the exit and through paths do not cross."""
    with open(path, "w") as f:
        f.write("<connections>\n")
        for leg, xid in exits.items():
            x = net.getEdge(xid)
            node = x.getFromNode()
            r = next(e for e in node.getIncoming() if e.getID() in ring_ids)
            on = next(e for e in node.getOutgoing() if e.getID() in ring_ids)
            taken = {c.getToLane().getIndex() for e in node.getIncoming() if e.getID() not in ring_ids
                     for c in e.getOutgoing().get(x, [])}      # exit lanes already fed by a slip from an entry
            free = [i for i in range(x.getLaneNumber()) if i not in taken]
            k = RING_LANES_LEAVING[leg]
            pairs = [(0, j) for j in free] if k == 1 else [(i, free[min(i * len(free) // k, len(free) - 1)]) for i in range(k)]
            f.write(f'  <delete from="{r.getID()}" to="{xid}"/>\n  <delete from="{r.getID()}" to="{on.getID()}"/>\n')
            for i, j in pairs:
                f.write(f'  <connection from="{r.getID()}" to="{xid}" fromLane="{i}" toLane="{j}"/>\n')
            for i in range(k, r.getLaneNumber()):
                f.write(f'  <connection from="{r.getID()}" to="{on.getID()}" fromLane="{i}" toLane="{i}"/>\n')
        f.write("</connections>\n")


def build_net(signals: bool, net_path: str) -> dict:
    raw = net_path + ".raw.xml"
    typemaps = ",".join([os.path.join(sumo.SUMO_HOME, "data", "typemap", "osmNetconvert.typ.xml"), TYPEMAP])
    run(["netconvert", "--osm-files", OSM, "-o", raw, "--type-files", typemaps, "--lefthand", "--check-lane-foes.all",
         "--geometry.remove", "--ramps.guess", "--junctions.join", "--junctions.corner-detail", "5", "--tls.guess-signals",
         "--tls.discard-simple", "--tls.join", "--tls.default-type", "static", "--tls.yellow.time", "3", "--tls.allred.time", "2",
         "--keep-edges.by-type", ",".join("highway." + t for t in ROAD_TYPES), "--remove-edges.isolated", "--osm.turn-lanes",
         "--osm.lane-access", "--osm.layer-elevation", "6", "--output.street-names", "--output.original-names", "--proj.utm",
         "--no-turnarounds", "--default.lanewidth", "3.25", "--ignore-widths"])
    net = sumolib.net.readNet(raw, withInternal=False)
    cx, cy = net.convertLonLat2XY(CENTRE_LON, CENTRE_LAT)
    ring_ways = {w.get("id") for w in ET.parse(OSM).getroot().findall("way")
                 if any(t.get("k") == "junction" and t.get("v") == "roundabout" for t in w.findall("tag"))}
    ring = [e for e in net.getEdges() if e.getID().lstrip("-").split("#")[0] in ring_ways
            and math.dist(e.getFromNode().getCoord(), (cx, cy)) < 130]
    ring_ids = {e.getID() for e in ring}
    entries, exits = {}, {}
    for e in ring:
        node = e.getFromNode()
        for i in node.getIncoming():
            if i.getID() not in ring_ids:
                entries[LEG_OF_NAME[i.getName()]] = i.getID()
        for o in node.getOutgoing():
            if o.getID() not in ring_ids:
                exits[LEG_OF_NAME[o.getName()]] = o.getID()
    entry_nodes = sorted({net.getEdge(e).getToNode().getID() for e in entries.values()})
    con = net_path + ".exits.con.xml"
    write_exit_connections(net, ring_ids, exits, con)
    args = ["netconvert", "--sumo-net-file", raw, "--connection-files", con, "-o", net_path, "--lefthand", "--check-lane-foes.all",
            "--tls.default-type", "static", "--tls.yellow.time", "3", "--tls.allred.time", "2", "--no-turnarounds"]
    if signals:
        args += ["--tls.set", ",".join(entry_nodes)]
    run(args)

    net = sumolib.net.readNet(net_path, withInternal=False)
    nodes = [n for e in ring for n in (e.getFromNode(),)]
    rx = sum(n.getCoord()[0] for n in nodes) / len(nodes)
    ry = sum(n.getCoord()[1] for n in nodes) / len(nodes)

    def heading(shape_a, shape_b):
        return math.atan2(shape_b[0] - shape_a[0], shape_b[1] - shape_a[1])

    def walk(start_id: str, forward: bool, limit_m: float = 1150.0) -> str:
        """Follow the street away from the ring: same name first, then the straightest continuation."""
        edge, seen = net.getEdge(start_id), {start_id}
        name = edge.getName()
        while True:
            node = edge.getToNode() if forward else edge.getFromNode()
            if math.dist(node.getCoord(), (rx, ry)) > limit_m:
                return edge.getID()
            options = [e for e in (node.getOutgoing() if forward else node.getIncoming())
                       if e.getID() not in seen and e.getID() not in ring_ids and e.allows("passenger")]
            if forward:
                options = [e for e in options if e.getToNode() != edge.getFromNode()]
            else:
                options = [e for e in options if e.getFromNode() != edge.getToNode()]
            s = edge.getShape()
            h0 = heading(s[-2], s[-1]) if forward else heading(s[0], s[1])

            def turn(e):
                t = e.getShape()
                h1 = heading(t[0], t[1]) if forward else heading(t[-2], t[-1])
                return abs((h1 - h0 + math.pi) % (2 * math.pi) - math.pi)

            def far_node(e):
                return e.getToNode() if forward else e.getFromNode()
            # never turn back toward the ring: the carriageways are separate edges, so a U-turn is not a reverse edge
            here = math.dist(node.getCoord(), (rx, ry))
            options = [e for e in options if turn(e) < math.radians(100) and math.dist(far_node(e).getCoord(), (rx, ry)) > here - 15]
            if not options:
                return edge.getID()
            edge = min(options, key=lambda e: (e.getName() != name, turn(e)))
            seen.add(edge.getID())

    sources = {leg: walk(eid, False) for leg, eid in entries.items()}
    sinks = {leg: walk(eid, True) for leg, eid in exits.items()}
    layout = dict(centre=[rx, ry], ring=sorted(ring_ids), entries=entries, exits=exits, sources=sources, sinks=sinks,
                  entry_nodes=entry_nodes, signals=signals)
    with open(net_path + ".layout.json", "w") as f:
        json.dump(layout, f, indent=1)
    return layout


def write_vtypes(path: str) -> None:
    """The study's vehicle types with their shares rescaled to MIX (the ambulance type is dropped: none in this demo)."""
    root = ET.parse(VTYPES_SRC).getroot()
    dist = root.find("vTypeDistribution")
    group = lambda v: "moto" if v.get("id").startswith("moto") else "car" if v.get("id").startswith("car") else v.get("id")
    total = defaultdict(float)
    for v in dist:
        total[group(v)] += float(v.get("probability"))
    for v in dist:
        v.set("probability", f"{MIX[group(v)] * float(v.get('probability')) / total[group(v)]:.5f}")
    for extra in root.findall("vType"):
        root.remove(extra)
    ET.ElementTree(root).write(path)


def write_flows(layout: dict, path: str, scale: float, sim_s: float) -> dict:
    lanes = sum(ENTRY_LANES.values())
    table = {}
    with open(path, "w") as f:
        f.write("<routes>\n")
        for a, shares in TURN_SHARE.items():
            inflow = TOTAL_VPH * scale * ENTRY_LANES[a] / lanes
            for b, share in shares.items():
                vph = inflow * share
                table[f"{a}>{b}"] = round(vph)
                via = f"{layout['entries'][a]} {layout['exits'][b]}"
                f.write(f'  <flow id="{a}_{b}" type="mix" begin="0" end="{sim_s:g}" from="{layout["sources"][a]}" via="{via}" '
                        f'to="{layout["sinks"][b]}" vehsPerHour="{vph:.1f}" departLane="best" departSpeed="max"/>\n')
        f.write("</routes>\n")
    return table


def write_additional(path: str, net_path: str, layout: dict, tls_out: str) -> None:
    net = sumolib.net.readNet(net_path, withInternal=False, withPrograms=True)
    ids = [t.getID() for t in net.getTrafficLights()]
    with open(path, "w") as f:
        f.write("<additional>\n")
        for tid in ids:
            f.write(f'  <timedEvent type="SaveTLSStates" source="{tid}" dest="{tls_out}"/>\n')
        f.write("</additional>\n")


def simulate(a, out: str, net_path: str, rou: str, vt: str, add: str) -> None:
    run(["sumo", "-n", net_path, "-r", rou, "-a", f"{vt},{add}", "--begin", "0", "--end", f"{a.sim_s:g}", "--step-length", "0.5",
         "--lateral-resolution", "0.8", "--collision.action", "warn", "--collision.check-junctions", "true",
         "--collision.mingap-factor", "0", "--time-to-teleport", "300", "--no-step-log", "true", "--no-warnings", "true",
         "--ignore-route-errors", "true", "--seed", str(a.seed),
         "--statistic-output", os.path.join(out, "statistics.xml"), "--summary-output", os.path.join(out, "summary.xml"),
         "--tripinfo-output", os.path.join(out, "tripinfo.xml"), "--tripinfo-output.write-unfinished", "true",
         "--fcd-output", os.path.join(out, "fcd.xml"), "--device.fcd.begin", f"{a.sim_s - a.replay_s:g}",
         "--device.fcd.period", "1", "--fcd-output.attributes", "x,y,angle,type,speed"])


def report(a, out: str, table: dict) -> dict:
    stats = ET.parse(os.path.join(out, "statistics.xml")).getroot()
    v, tp, sf = stats.find("vehicles"), stats.find("teleports"), stats.find("safety")
    per = defaultdict(list)
    unfinished = 0
    for t in ET.parse(os.path.join(out, "tripinfo.xml")).getroot().findall("tripinfo"):
        if float(t.get("depart")) < a.warm_s and float(t.get("depart")) >= 0:
            continue
        if float(t.get("arrival", -1)) < 0:
            unfinished += 1
            continue
        per[t.get("id").split(".")[0]].append((float(t.get("duration")), float(t.get("waitingTime")), float(t.get("timeLoss")), float(t.get("routeLength"))))
    rows = {k: dict(n=len(x), duration_s=round(sum(i[0] for i in x) / len(x), 1), wait_s=round(sum(i[1] for i in x) / len(x), 1),
                    time_loss_s=round(sum(i[2] for i in x) / len(x), 1), route_m=round(sum(i[3] for i in x) / len(x)))
            for k, x in sorted(per.items())}
    all_rows = [i for x in per.values() for i in x]
    sm = ET.parse(os.path.join(out, "summary.xml")).getroot().findall("step")
    res = dict(scale=a.scale, signals=not a.no_signals, flows_vph=table, total_vph=sum(table.values()),
               loaded=int(v.get("loaded")), inserted=int(v.get("inserted")), running_end=int(v.get("running")), waiting_insert_end=int(v.get("waiting")),
               teleports=int(tp.get("total")), teleport_jam=int(tp.get("jam")), teleport_yield=int(tp.get("yield")), teleport_wrong_lane=int(tp.get("wrongLane")),
               collisions=int(sf.get("collisions")), emergency_stops=int(sf.get("emergencyStops")),
               after_warm=dict(finished=len(all_rows), unfinished=unfinished,
                               mean_duration_s=round(sum(i[0] for i in all_rows) / max(1, len(all_rows)), 1),
                               mean_wait_s=round(sum(i[1] for i in all_rows) / max(1, len(all_rows)), 1),
                               mean_time_loss_s=round(sum(i[2] for i in all_rows) / max(1, len(all_rows)), 1)),
               running_by_5min=[int(s.get("running")) for s in sm if abs(float(s.get("time")) % 300) < 1e-6],
               halting_by_5min=[int(s.get("halting")) for s in sm if abs(float(s.get("time")) % 300) < 1e-6],
               per_movement=rows)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def write_replay(out: str, net_path: str, layout: dict, rep: dict, dest: str, radius: float = 430.0) -> None:
    """Self-contained HTML replay: lanes and junctions near the ring, 1 Hz vehicle positions, signal states."""
    cx, cy = layout["centre"]
    root = ET.parse(net_path).getroot()

    def pts(shape: str) -> list:
        return [tuple(map(float, p.split(",")))[:2] for p in shape.split()]   # elevated lanes carry a third coordinate

    def near(ps: list) -> bool:
        return any(math.hypot(x - cx, y - cy) < radius for x, y in ps)

    def flat(ps: list) -> list:
        return [round(v, 1) for x, y in ps for v in (x - cx, y - cy)]

    lanes, lane_of, junctions = [], {}, []
    for e in root.findall("edge"):
        if e.get("function") == "internal":
            continue
        for ln in e.findall("lane"):
            ps = pts(ln.get("shape"))
            lane_of[ln.get("id")] = (ps, float(ln.get("width", 3.2)))
            if near(ps):
                lanes.append([float(ln.get("width", 3.2)), flat(ps)])
    for j in root.findall("junction"):
        if j.get("type") != "internal" and j.get("shape"):
            ps = pts(j.get("shape"))
            if len(ps) >= 3 and near(ps):
                junctions.append(flat(ps))

    t0, t1 = rep["replay_t0"], rep["replay_t1"]
    changes = defaultdict(list)
    for s in ET.parse(os.path.join(out, "tls_states.xml")).getroot().findall("tlsState"):
        t, tid, st = float(s.get("time")), s.get("id"), s.get("state")
        rows = changes[tid]
        if t <= t0:
            rows[:] = [[t0, st]]
        elif t < t1 and rows and rows[-1][1] != st:
            rows.append([t, st])
    tids = sorted(changes)
    groups = defaultdict(list)
    for c in root.findall("connection"):
        if c.get("tl") in changes:
            groups[(c.get("tl"), f'{c.get("from")}_{c.get("fromLane")}')].append(int(c.get("linkIndex")))
    heads = []
    for (tid, lane), idx in groups.items():
        ps, w = lane_of[lane]
        x1, y1 = ps[-1]
        if not near([(x1, y1)]):
            continue
        x0, y0 = next(((x, y) for x, y in reversed(ps[:-1]) if math.hypot(x - x1, y - y1) > 0.05), (x1, y1 - 1))
        d = math.hypot(x1 - x0, y1 - y0)
        nx, ny = -(y1 - y0) / d, (x1 - x0) / d
        heads.append([tids.index(tid), idx] + [round(v, 1) for v in (x1 - cx + nx * w / 2, y1 - cy + ny * w / 2, x1 - cx - nx * w / 2, y1 - cy - ny * w / 2)])

    ids, frames = {}, []
    for _, el in ET.iterparse(os.path.join(out, "fcd.xml")):
        if el.tag != "timestep":
            continue
        row = []
        for v in el.findall("vehicle"):
            x, y = float(v.get("x")) - cx, float(v.get("y")) - cy
            if math.hypot(x, y) < radius + 40:
                t = v.get("type")
                row += [ids.setdefault(v.get("id"), len(ids)), round(x, 1), round(y, 1), round(float(v.get("angle"))),
                        0 if t.startswith("moto") else 1 if t.startswith("car") else 2 if t == "bus" else 3]
        frames.append(row)
        el.clear()

    labels = []
    for text, bearing in (("Jl. MH Thamrin", 8), ("Jl. Imam Bonjol", 122), ("Jl. Jend. Sudirman", 190), ("Jl. Saleh Ishak (keluar)", 287)):
        b = math.radians(bearing)
        labels.append([text, round(math.sin(b) * 104, 1), round(math.cos(b) * 104, 1)])
    fmt = lambda n: f"{n:,}".replace(",", ".")
    data = dict(radius=radius, t0=t0, t1=t1, lanes=lanes, junctions=junctions, tls=[changes[t] for t in tids], heads=heads, frames=frames, labels=labels,
                statsText=f"Volume asumsi {fmt(rep['total_vph'])} kend/jam ({rep['scale'] * 100:.0f}% dari {fmt(round(TOTAL_VPH))}) · "
                          f"lampu otomatis SUMO · teleport {rep['teleports']} · tabrakan {rep['collisions']}")
    with open(os.path.join(ROOT, "scripts", "bundaran_hi_replay.html"), encoding="utf-8") as f:
        html = f.read()
    with open(dest, "w", encoding="utf-8") as f:
        f.write(html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    print("replay:", dest, round(os.path.getsize(dest) / 1e6, 2), "MB;", len(frames), "frames;", len(heads), "signal heads;", len(lanes), "lanes")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", type=float, default=0.4,
                   help="multiplier on the assumed demand; 0.4 is the highest tested level that stays stable, 0.5 and above jam")
    p.add_argument("--sim-s", type=float, default=1800.0)
    p.add_argument("--warm-s", type=float, default=900.0)
    p.add_argument("--replay-s", type=float, default=300.0, help="last seconds written to fcd.xml for the replay")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-signals", action="store_true", help="keep the ring as OSM maps it (priority/yield, no signals)")
    p.add_argument("--tag", default="", help="suffix of the run folder")
    p.add_argument("--replay", action="store_true", help="write data/bundaran_hi_demo/replay.html from this run")
    p.add_argument("--skip-run", action="store_true", help="reuse the run folder's outputs instead of simulating again")
    a = p.parse_args()
    name = ("signals" if not a.no_signals else "nosignals") + (f"_{a.tag}" if a.tag else "")
    out = os.path.join(DEMO, "run", name)
    os.makedirs(out, exist_ok=True)
    net_path = os.path.join(DEMO, "net", f"{name}.net.xml")
    if a.skip_run:
        with open(net_path + ".layout.json") as f:
            layout = json.load(f)
        with open(os.path.join(out, "report.json")) as f:
            rep = json.load(f)
    else:
        layout = build_net(not a.no_signals, net_path)
        vt, rou, add = (os.path.join(out, n) for n in ("vtypes.add.xml", "flows.rou.xml", "tls_events.add.xml"))
        write_vtypes(vt)
        table = write_flows(layout, rou, a.scale, a.sim_s)
        write_additional(add, net_path, layout, os.path.join(out, "tls_states.xml"))
        print("layout:", json.dumps({k: layout[k] for k in ("entries", "exits", "sources", "sinks", "entry_nodes")}))
        simulate(a, out, net_path, rou, vt, add)
        rep = report(a, out, table)
        print(json.dumps(rep, indent=1))
    if a.replay:
        rep = dict(rep, replay_t0=a.sim_s - a.replay_s, replay_t1=a.sim_s)
        write_replay(out, net_path, layout, rep, os.path.join(DEMO, "replay.html"))


if __name__ == "__main__":
    main()
