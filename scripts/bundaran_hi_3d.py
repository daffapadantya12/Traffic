"""Live 3D view of the uncalibrated Bundaran HI demo, built like the Fatmawati twin (traffic_twin/fatmawati/webapp3d.py).

NOT part of the traffic_twin study and NOT evidence about the real junction; data/bundaran_hi_demo/README.md lists what is
real and what is assumed.  It runs the network and demand of scripts/bundaran_hi_demo.py in SUMO (libsumo) and streams every
frame to a Three.js page.  Run it with the demo's network already built:

    python -I scripts/bundaran_hi_3d.py [--port 8792] [--scale 0.4] [--speed 5] [--warmup-min 10]

  GET /          scripts/bundaran_hi_3d.html
  GET /geometry  data/bundaran_hi_demo/viewer/geometry.json (lanes, junctions, stop lines, signal heads, buildings, pool, monument;
                 (c) OpenStreetMap contributors, ODbL)
  GET /stream    Server-Sent Events, one frame per tick
  GET /control?speed=N&pause=0|1&signals=auto|off
  GET /health
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import sumolib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bundaran_hi_demo as demo  # noqa: E402  (the demand, vehicle types and paths of the batch demo)

NET = os.path.join(demo.DEMO, "net", "signals_s04.net.xml")
LAYOUT = NET + ".layout.json"
BUILDINGS = os.path.join(demo.DEMO, "osm", "buildings_1600m.json")
VIEWER = os.path.join(demo.DEMO, "viewer")
GEOMETRY = os.path.join(VIEWER, "geometry.json")
PAGE = os.path.join(HERE, "bundaran_hi_3d.html")
TICK_S = 0.1
MAX_SPEED = 200.0
STUDY_M = 355.0                  # the approach panel counts vehicles within 300 m of the ring (ring radius about 55 m)
TYPE_CODE = {"car": 0, "moto": 2, "bus": 3, "truck": 4}   # the codes of the Fatmawati viewer
LEG_NAME = {"N": "Jl. MH Thamrin", "E": "Jl. Imam Bonjol", "S": "Jl. Jend. Sudirman", "W": "Jl. Saleh Ishak"}
DEFAULT_HEIGHT = {"house": 7.0, "residential": 7.0, "detached": 7.0, "terrace": 8.0, "apartments": 36.0, "commercial": 14.0,
                  "retail": 10.0, "office": 30.0, "hotel": 30.0, "hospital": 22.0, "school": 12.0, "university": 16.0,
                  "mosque": 12.0, "church": 14.0, "industrial": 10.0, "warehouse": 9.0, "garage": 4.0, "roof": 5.0,
                  "yes": 8.0}          # the defaults of traffic_twin/fatmawati/geometry3d.py


def _r(v, k=2):
    return round(float(v), k)


def _height(tags: dict) -> float:
    try:
        return float(str(tags.get("height", "")).replace("m", "").strip())
    except ValueError:
        pass
    try:
        return 3.2 * float(tags["building:levels"]) + 1.0
    except (KeyError, ValueError):
        return DEFAULT_HEIGHT.get(tags.get("building", "yes"), 8.0)


def build_geometry(radius_m: float = 1600.0) -> str:
    """The viewer's static scene, in metres from the ring centre (x east, y north); every road at ground level."""
    net = sumolib.net.readNet(NET, withInternal=False, withPrograms=True)
    with open(LAYOUT) as f:
        layout = json.load(f)
    x0, y0 = layout["centre"]
    rel = lambda x, y: (_r(x - x0), _r(y - y0))
    lanes = [dict(id=ln.getID(), w=_r(ln.getWidth()), k=e.getType().replace("highway.", ""), idx=ln.getIndex(), n=e.getLaneNumber(),
                  p=[[*rel(p[0], p[1]), 0.0] for p in ln.getShape()])
             for e in net.getEdges() if e.getFunction() == "" for ln in e.getLanes()]
    junctions = [dict(id=n.getID(), z=0.0, p=[list(rel(p[0], p[1])) for p in n.getShape3D()]) for n in net.getNodes() if len(n.getShape3D()) >= 3]

    tls_ids = sorted(t.getID() for t in net.getTrafficLights())
    stoplines, heads, entries = [], [], {}
    leg_of_entry = {eid: leg for leg, eid in layout["entries"].items()}
    for t_index, tid in enumerate(tls_ids):
        tl = net.getTLS(tid)
        phases = next(iter(tl.getPrograms().values())).getPhases()
        always_green = lambda i: all(ph.state[i] in "Gg" for ph in phases)     # a free slip must not set the colour of its approach
        by_lane = {}
        for inl, _, idx in tl.getConnections():
            by_lane.setdefault(inl.getID(), []).append(idx)
        top_lane = {}
        for lane_id in by_lane:
            ln = net.getLane(lane_id)
            e = ln.getEdge().getID()
            if e not in top_lane or ln.getIndex() > net.getLane(top_lane[e]).getIndex():
                top_lane[e] = lane_id
        for lane_id, idxs in sorted(by_lane.items()):
            ln = net.getLane(lane_id)
            shp = ln.getShape()
            (xa, ya), (xb, yb) = shp[-2][:2], shp[-1][:2]
            L = math.hypot(xb - xa, yb - ya) or 1.0
            ux, uy = (xb - xa) / L, (yb - ya) / L
            nx, ny, hw = -uy, ux, ln.getWidth() / 2.0
            end = rel(xb, yb)
            stoplines.append(dict(lane=lane_id, a=[_r(end[0] + nx * hw), _r(end[1] + ny * hw)], b=[_r(end[0] - nx * hw), _r(end[1] - ny * hw)]))
            edge = ln.getEdge().getID()
            if top_lane[edge] == lane_id:                                      # one head per approach edge, on the median side
                links = sorted(i for l2 in ln.getEdge().getLanes() for i in by_lane.get(l2.getID(), []))
                shown = [i for i in links if not always_green(i)] or links
                heads.append(dict(x=_r(end[0] - nx * (hw + 0.8)), y=_r(end[1] - ny * (hw + 0.8)), t=t_index, links=shown))
                if edge in leg_of_entry:
                    entries[leg_of_entry[edge]] = dict(x=end[0], y=end[1], ux=_r(ux, 4), uy=_r(uy, 4), t=t_index, link=shown[0])

    with open(BUILDINGS) as f:
        elements = json.load(f)["elements"]
    buildings, pool, monument = [], None, dict(x=0.0, y=0.0, name="Monumen Selamat Datang")
    to_rel = lambda g: [list(rel(*net.convertLonLat2XY(p["lon"], p["lat"]))) for p in g]
    for el in elements:
        tags = el.get("tags", {})
        if tags.get("natural") == "water" and el.get("geometry"):
            pool = to_rel(el["geometry"])
            pool = pool[:-1] if pool[0] == pool[-1] else pool
        elif tags.get("historic") == "monument" and "lat" in el:
            x, y = rel(*net.convertLonLat2XY(el["lon"], el["lat"]))
            monument = dict(x=x, y=y, name=tags.get("name", monument["name"]))
        elif "building" in tags:
            rings = [el["geometry"]] if el.get("geometry") else [m["geometry"] for m in el.get("members", []) if m.get("role") == "outer"
                                                                 and m.get("geometry") and m["geometry"][0] == m["geometry"][-1]]
            for g in rings:
                if len(g) < 4:
                    continue
                pts = to_rel(g)
                if math.hypot(sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)) <= radius_m:
                    buildings.append(dict(h=_r(_height(tags), 1), p=pts[:-1] if pts[0] == pts[-1] else pts))
    data = dict(origin=dict(lat=demo.CENTRE_LAT, lon=demo.CENTRE_LON), lanes=lanes, junctions=junctions, stoplines=stoplines, heads=heads,
                tls_ids=tls_ids, entries=entries, buildings=buildings, pool=pool, monument=monument, ring_tls=layout["entry_nodes"],
                attribution="Road and building data (c) OpenStreetMap contributors, ODbL")
    os.makedirs(VIEWER, exist_ok=True)
    with open(GEOMETRY, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print("geometry:", GEOMETRY, os.path.getsize(GEOMETRY) // 1024, "KB;", {k: len(v) for k, v in data.items() if isinstance(v, list)})
    return GEOMETRY


def approach_edges(net, layout: dict, reach_m: float = 500.0) -> dict:
    """{edge id: leg} for the street upstream of each ring entry."""
    out = {}
    for leg, eid in layout["entries"].items():
        e, walked = net.getEdge(eid), 0.0
        while e is not None and walked < reach_m and e.getID() not in out:
            out[e.getID()] = leg
            walked += e.getLength()
            e = next((i for i in e.getFromNode().getIncoming() if i.getName() == e.getName() and i.getFromNode() != e.getToNode()), None)
    return out


class Sim(threading.Thread):
    """SUMO lives in this thread and is only touched here; HTTP threads read the latest frame."""

    def __init__(self, routes: str, vtypes: str, start_hour: float, speed: float, seed: int, warmup_s: float, geo: dict, edge_leg: dict):
        super().__init__(daemon=True)
        self.routes, self.vtypes, self.start_hour, self.speed, self.seed, self.warmup_s = routes, vtypes, start_hour, speed, seed, warmup_s
        self.tls_ids, self.ring_tls, self.entries, self.edge_leg = geo["tls_ids"], geo["ring_tls"], geo["entries"], edge_leg
        with open(LAYOUT) as f:
            self.x0, self.y0 = json.load(f)["centre"]
        self.legs = sorted(geo["entries"], key="NESW".index)
        self.paused, self.signals, self.applied = False, "auto", "auto"
        self.lock = threading.Lock()
        self.frame_json = json.dumps(dict(t=0, clock="--:--:--", vehicles=[], kpi={}, events=["starting SUMO..."]))
        self.frame_seq = 0
        self.stop_flag = False
        self.carry = 0.0
        self.ready = threading.Event()
        self.t, self.n_arrived, self.n_teleports, self.n_lane, self.n_junction = 0.0, 0, 0, 0, 0
        self.events, self.seen, self.ids = [], set(), {}

    def clock(self) -> str:
        h = (self.start_hour * 3600.0 + self.t) % 86400.0
        return f"{int(h // 3600):02d}:{int(h % 3600 // 60):02d}:{int(h % 60):02d}"

    def step(self, ts) -> None:
        ts.simulationStep()
        self.t = ts.simulation.getTime()
        self.n_arrived += ts.simulation.getArrivedNumber()
        tele = ts.simulation.getStartingTeleportNumber()
        if tele:
            self.n_teleports += tele
            self.events.append(f"{self.clock()} teleport: {tele} vehicle{'s' if tele > 1 else ''} stuck for 300 s")
        for c in ts.simulation.getCollisions():
            key = tuple(sorted((c.collider, c.victim)))
            if key not in self.seen:
                self.seen.add(key)
                if c.lane.startswith(":"):
                    self.n_junction += 1
                else:
                    self.n_lane += 1

    def frame(self, ts) -> dict:
        vs, moto, halt = [], 0, 0
        legs = {a: dict(n=0, q_veh=0, sum_speed=0.0) for a in self.legs}
        for vid in ts.vehicle.getIDList():
            x, y = ts.vehicle.getPosition(vid)
            sp = ts.vehicle.getSpeed(vid)
            type_id = ts.vehicle.getTypeID(vid)
            code = next((v for k, v in TYPE_CODE.items() if type_id.startswith(k)), 0)
            moto += code == 2
            halt += sp < 0.5
            num = self.ids.setdefault(vid, len(self.ids) + 1)
            vs.append([num, round(x - self.x0, 2), round(y - self.y0, 2), 0.0, round(ts.vehicle.getAngle(vid), 1), code, round(sp, 1)])
            leg = self.edge_leg.get(ts.vehicle.getRoadID(vid))
            if leg and math.hypot(x - self.x0, y - self.y0) <= STUDY_M:
                s = legs[leg]
                s["n"] += 1
                s["q_veh"] += sp < 0.5
                s["sum_speed"] += sp
        for s in legs.values():
            s["speed_kmh"] = round(3.6 * s.pop("sum_speed") / s["n"], 1) if s["n"] else None
        tls = [ts.trafficlight.getRedYellowGreenState(t) for t in self.tls_ids]
        return dict(t=round(self.t, 1), clock=self.clock(), vehicles=vs, tls=tls,
                    entries={a: tls[e["t"]][e["link"]] for a, e in self.entries.items()}, legs=legs,
                    kpi=dict(n_vehicles=len(vs), n_motorcycles=moto, n_halting=halt, n_arrived=self.n_arrived, n_teleports=self.n_teleports,
                             n_lane_collisions=self.n_lane, n_junction_overlaps=self.n_junction),
                    events=list(self.events[-10:]))

    def publish(self, f: dict) -> None:
        self.frame_json = json.dumps(f, separators=(",", ":"))
        self.frame_seq += 1

    def run(self) -> None:
        import libsumo as ts
        ts.start(["sumo", "-n", NET, "-r", self.routes, "-a", self.vtypes, "--begin", "0", "--step-length", "0.5", "--lateral-resolution", "0.8",
                  "--collision.action", "warn", "--collision.check-junctions", "true", "--collision.mingap-factor", "0", "--time-to-teleport", "300",
                  "--no-step-log", "true", "--no-warnings", "true", "--ignore-route-errors", "true", "--seed", str(self.seed)])   # as the batch demo
        shown = 0.0
        while self.t < self.warmup_s and not self.stop_flag:      # fill the network before the first live frame
            with self.lock:
                self.step(ts)
                if time.perf_counter() - shown > 0.5:
                    shown = time.perf_counter()
                    f = self.frame(ts)
                    f["vehicles"] = []
                    f["events"] = [f"warming up the network: {self.t / 60:.1f} of {self.warmup_s / 60:g} simulated minutes"]
                    self.publish(f)
        self.ready.set()
        next_tick = time.perf_counter()
        while not self.stop_flag:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(next_tick - now, TICK_S))
                continue
            next_tick = max(next_tick + TICK_S, now)
            t0 = time.perf_counter()
            stepped = False
            with self.lock:
                if self.signals != self.applied:                  # SUMO's own "off" program: right of way decides at the ring entries
                    for tid in self.ring_tls:
                        ts.trafficlight.setProgram(tid, "off" if self.signals == "off" else "0")
                    self.applied = self.signals
                    self.events.append(f"{self.clock()} ring signals {'off' if self.signals == 'off' else 'back on'}")
                if not self.paused:
                    self.carry += self.speed * TICK_S
                    while self.carry >= 0.5 - 1e-9:
                        self.step(ts)
                        stepped = True
                        self.carry -= 0.5
                        if time.perf_counter() - t0 > 0.09:
                            self.carry = min(self.carry, 0.5)
                            break
                if stepped or self.frame_seq == 0 or self.paused:
                    f = self.frame(ts)
                    f["ctl"] = dict(speed=self.speed, paused=self.paused, signals=self.signals)
                    self.publish(f)
        ts.close()

    def control(self, speed=None, pause=None, signals=None) -> dict:
        with self.lock:
            if speed is not None:
                self.speed = speed
            if pause is not None:
                self.paused = pause
            if signals in ("auto", "off"):
                self.signals = signals
            return dict(speed=self.speed, paused=self.paused, signals=self.signals)


def make_handler(sim: Sim):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body: bytes, ctype: str, cache: str = "no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                with open(PAGE, "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            elif url.path == "/geometry":
                with open(GEOMETRY, "rb") as fh:
                    self._send(200, fh.read(), "application/json", "no-cache")
            elif url.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                last = -1
                try:
                    while not sim.stop_flag:
                        if sim.frame_seq != last:
                            last = sim.frame_seq
                            self.wfile.write(b"data: " + sim.frame_json.encode() + b"\n\n")
                            self.wfile.flush()
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    self.close_connection = True
            elif url.path == "/control":
                q = parse_qs(url.query)
                speed = None
                if "speed" in q:
                    try:
                        speed = float(q["speed"][0])
                    except ValueError:
                        speed = math.nan
                    if not (math.isfinite(speed) and 0 < speed <= MAX_SPEED):
                        self._send(400, b'{"error":"speed must be in (0, 200]"}', "application/json")
                        return
                pause = (q["pause"][0] not in ("0", "false", "")) if "pause" in q else None
                signals = q["signals"][0] if "signals" in q else None
                self._send(200, json.dumps(sim.control(speed, pause, signals)).encode(), "application/json")
            elif url.path == "/health":
                self._send(200, json.dumps(dict(ok=sim.ready.is_set(), t=sim.t, speed=sim.speed, paused=sim.paused)).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8792)
    ap.add_argument("--scale", type=float, default=0.4, help="multiplier on the assumed demand (see scripts/bundaran_hi_demo.py)")
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--start-hour", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-min", type=float, default=10.0, help="simulated minutes run before the first live frame")
    ap.add_argument("--rebuild-geometry", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(NET):
        sys.exit(f"{NET} is missing: run  python -I scripts/bundaran_hi_demo.py --tag s04  first")
    if args.rebuild_geometry or not os.path.exists(GEOMETRY):
        build_geometry()
    with open(GEOMETRY) as f:
        geo = json.load(f)
    with open(LAYOUT) as f:
        layout = json.load(f)
    os.makedirs(VIEWER, exist_ok=True)
    vtypes, routes = os.path.join(VIEWER, "vtypes.add.xml"), os.path.join(VIEWER, "live_flows.rou.xml")
    demo.write_vtypes(vtypes)
    table = demo.write_flows(layout, routes, args.scale, 86400.0)        # constant demand all day
    edge_leg = approach_edges(sumolib.net.readNet(NET, withInternal=False), layout)
    sim = Sim(routes, vtypes, args.start_hour, args.speed, args.seed, 60.0 * args.warmup_min, geo, edge_leg)
    sim.start()
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(sim))
    httpd.daemon_threads = True
    print(f"Bundaran HI 3D demo on http://{args.host}:{args.port}/  ({sum(table.values())} veh/h assumed, uncalibrated)", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop_flag = True
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
