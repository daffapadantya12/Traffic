"""Live 3D web view of the Fatmawati twin (DECISIONS §13.5).

  GET /           -> traffic_twin/fatmawati/static/index.html (Three.js)
  GET /geometry   -> data/fatmawati/viewer/geometry.json (lanes, junctions, stop lines, motorcycle boxes,
                     signal heads, the toll flyover, buildings; (c) OpenStreetMap contributors, ODbL)
  GET /stream     -> Server-Sent Events, one frame per simulation tick; the first viewer also gets "tomtom"
                     events (live TomTom flow, display only, see tomtom.py)
  GET /control?speed=N&pause=0|1&controller=fixed|longest_queue
  GET /health

The SUMO simulation (libsumo) lives in one thread and is only touched there; HTTP threads read the
latest frame.  Run:  python3 -m traffic_twin.fatmawati.webapp3d --port 8770 --start-hour 7 --hours 3
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

from .demand import DATA, routes_xml, vtypes_xml
from .sumo_world import CONTROLLERS

# trained controllers the viewer can run (the closed-loop winner of the evaluation; reports/a3)
LEARNED = {"learned_gb": os.path.join("reports", "a3", "learning", "models", "taskA_gradient_boosting.joblib")}

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
GEOMETRY = os.path.join(DATA, "viewer", "geometry.json")
TICK_S = 0.1
MAX_SPEED = 200.0


class Sim(threading.Thread):
    def __init__(self, routes: str, start_hour: float, speed: float, controller: str, seed: int, warmup_s: float = 0.0):
        super().__init__(daemon=True)
        self.routes, self.start_hour, self.speed, self.controller, self.seed = routes, start_hour, speed, controller, seed
        self.warmup_s = warmup_s
        self.paused = False
        self.lock = threading.Lock()
        self.frame_json = json.dumps(dict(t=0, clock="--:--:--", vehicles=[], kpi={}, events=["starting SUMO..."]))
        self.frame_seq = 0
        self.stop_flag = False
        self.carry = 0.0
        self.ready = threading.Event()

    def run(self):
        from .sumo_world import FatmawatiWorld
        w = FatmawatiWorld(self.routes, start_hour=self.start_hour, seed=self.seed, controller=self.controller,
                           sensors=True, store_rows=False)
        self.world = w
        shown = 0.0
        while w.t < self.warmup_s and not self.stop_flag:       # fill the network before the first live frame
            with self.lock:
                w.step()
                if time.perf_counter() - shown > 0.5:          # progress only; vehicles appear once live
                    shown = time.perf_counter()
                    f = w.frame()
                    f["vehicles"] = []
                    f["events"] = f["events"] + [f"warming up the network: {w.t / 60:.1f} of {self.warmup_s / 60:g} simulated minutes"]
                    self.frame_json = json.dumps(f, separators=(",", ":"))
                    self.frame_seq += 1
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
                if w.controller != self.controller:          # a trained model is loaded when it is chosen
                    path = LEARNED.get(self.controller)
                    if path is not None:
                        from .learn import load_policy
                        w.policy = load_policy(path)
                    else:
                        w.policy = None
                    w.controller = self.controller
                if not self.paused:
                    self.carry += self.speed * TICK_S
                    while self.carry >= 0.5 - 1e-9:
                        w.step()
                        stepped = True
                        self.carry -= 0.5
                        if time.perf_counter() - t0 > 0.09:
                            self.carry = min(self.carry, 0.5)
                            break
                if stepped or self.frame_seq == 0:
                    f = w.frame()
                    f["ctl"] = dict(speed=self.speed, paused=self.paused, controller=self.controller)
                    self.frame_json = json.dumps(f, separators=(",", ":"))
                    self.frame_seq += 1
        w.close()

    def control(self, speed=None, pause=None, controller=None) -> dict:
        with self.lock:
            if speed is not None:
                self.speed = speed
            if pause is not None:
                self.paused = pause
            if controller in CONTROLLERS or (controller in LEARNED and os.path.exists(LEARNED[controller])):
                self.controller = controller
            return dict(speed=self.speed, paused=self.paused, controller=self.controller)


class Viewers:
    """Open /stream connections; live TomTom readings go to the first one only (TomTom T&C 11.4: no serving of
    cached results to several users)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.owner = None

    def claim(self, token) -> bool:
        with self.lock:
            if self.owner is None:
                self.owner = token
            return self.owner is token

    def release(self, token) -> None:
        with self.lock:
            if self.owner is token:
                self.owner = None

    def any(self) -> bool:
        return self.owner is not None


def make_handler(sim: Sim, viewers: Viewers | None = None, tomtom=None):
    viewers = viewers or Viewers()

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
                with open(os.path.join(STATIC, "index.html"), "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            elif url.path == "/geometry":
                with open(GEOMETRY, "rb") as fh:                  # no-cache: a rebuilt network shows on reload
                    self._send(200, fh.read(), "application/json", "no-cache")
            elif url.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                last, token, next_tt = -1, object(), 0.0
                owner = viewers.claim(token)
                try:
                    if tomtom is not None and not owner:
                        busy = dict(status="TomTom live speeds are shown in another viewer", caption="©TomTom", readings={})
                        self.wfile.write(b"event: tomtom\ndata: " + json.dumps(busy).encode() + b"\n\n")
                    while not sim.stop_flag:
                        if sim.frame_seq != last:
                            last = sim.frame_seq
                            self.wfile.write(b"data: " + sim.frame_json.encode() + b"\n\n")
                            self.wfile.flush()
                        if owner and tomtom is not None and time.monotonic() >= next_tt:
                            next_tt = time.monotonic() + 5.0
                            self.wfile.write(b"event: tomtom\ndata: " + json.dumps(tomtom.snapshot()).encode() + b"\n\n")
                            self.wfile.flush()
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    viewers.release(token)
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
                ctrl = q["controller"][0] if "controller" in q else None
                self._send(200, json.dumps(sim.control(speed, pause, ctrl)).encode(), "application/json")
            elif url.path == "/health":
                self._send(200, json.dumps(dict(ok=sim.ready.is_set(), speed=sim.speed, paused=sim.paused)).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--start-hour", type=float, default=7.0)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--level", default="design", choices=["low", "design", "high"])
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--controller", default="fixed", choices=list(CONTROLLERS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warmup-min", type=float, default=10.0, help="simulated minutes run before the first live frame")
    args = ap.parse_args(argv)
    if not os.path.exists(GEOMETRY):
        from .geometry3d import build
        build()
    vtypes_xml(os.path.join(DATA, "demand", "vtypes.add.xml"))
    routes = os.path.join(DATA, "demand", f"live_{args.level}_{int(args.start_hour):02d}h_{args.hours:g}h.rou.xml")
    info = routes_xml(routes, args.level, start_hour=args.start_hour, duration_s=args.hours * 3600.0, seed=args.seed)
    sim = Sim(routes, args.start_hour, args.speed, args.controller, args.seed, warmup_s=60.0 * args.warmup_min)
    sim.start()
    from .sumo_world import sensor_layout
    from .tomtom import TomTomLive, probes
    viewers = Viewers()
    tomtom = TomTomLive(probes(sensor_layout()), active=viewers.any)    # polls only while a viewer is open
    tomtom.start()
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(sim, viewers, tomtom))
    httpd.daemon_threads = True
    print(f"Fatmawati 3D twin on http://{args.host}:{args.port}/  ({info['n_vehicles']} vehicles, {info['n_ambulances']} ambulances "
          f"over {args.hours:g} h from {args.start_hour:g}:00, {args.level} demand)", flush=True)
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
