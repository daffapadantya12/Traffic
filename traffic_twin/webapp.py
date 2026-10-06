"""Live web demo of the digital twin: a standard-library HTTP server around `TwinWorld`.

  GET /          -> traffic_twin/static/index.html
  GET /stream    -> Server-Sent Events, one JSON frame every 100 ms wall time
                    (the fields index.html's render() / renderPanels() read)
  GET /control?speed=N&pause=0|1 -> simulated seconds per wall second (0 < N <= 2000) / pause
  GET /frame, GET /health          -> latest frame / server state (HEAD works too)
  GET /geometry                    -> static drawing geometry (box size, kerb radius, lane paths)

The simulation thread owes speed x 0.1 simulated seconds per 100 ms tick and pays them in
whole physics steps (a fractional carry, so 1x really is 1x); a tick stops early when the
machine cannot keep up and the backlog is dropped (the achieved factor is kpi.speed_achieved).
Every frame carries ctl = {speed, paused} so the page can show the server's state.

Run:  python3 -m traffic_twin.webapp --port 8765 --speed 60 \
          --model reports/ml/models/taskA_hgb_sensor.joblib --demand design --profile daily
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

from .twin import DEFAULT_MODEL, TwinWorld, load_base_config

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TICK_S = 0.1
MAX_SPEED = 2000.0


class Simulation(threading.Thread):
    """Owns the TwinWorld; advances it on a 100 ms wall clock and keeps the latest frame."""

    def __init__(self, twin: TwinWorld, speed: float = 60.0, max_tick_wall: float = 0.085):
        super().__init__(daemon=True)
        self.twin = twin
        self.speed = float(speed)
        self.paused = False
        self.max_tick_wall = max_tick_wall
        self.lock = threading.Lock()
        self.frame_json = json.dumps(twin.frame())
        self.geometry_json = json.dumps(twin.geometry())
        self.frame_seq = 0
        self.stop_flag = False
        self.achieved = 0.0
        self.carry = 0.0                        # simulated seconds owed but not yet stepped

    def run(self) -> None:
        dt = self.twin.cfg.dt
        next_tick = time.perf_counter()
        while not self.stop_flag:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(next_tick - now, TICK_S))
                continue
            next_tick = max(next_tick + TICK_S, now)
            t_start = time.perf_counter()
            advanced = 0.0
            with self.lock:
                if not self.paused:
                    self.carry += self.speed * TICK_S
                    while self.carry >= dt - 1e-9:
                        self.twin.step()
                        advanced += dt
                        self.carry -= dt
                        if time.perf_counter() - t_start > self.max_tick_wall:
                            self.carry = min(self.carry, dt)   # the machine cannot keep up: drop the backlog
                            break
                    self.twin.speed_achieved = self.achieved
                frame = self.twin.frame()
                frame["ctl"] = dict(speed=self.speed, paused=self.paused)
                self.frame_json = json.dumps(frame, allow_nan=False, default=_json_default)
                self.frame_seq += 1
            wall = max(time.perf_counter() - t_start, 1e-6)
            inst = advanced / max(wall, TICK_S) if not self.paused else 0.0
            self.achieved = inst if self.achieved == 0.0 else 0.8 * self.achieved + 0.2 * inst

    def control(self, speed: float | None, pause: bool | None) -> dict:
        """speed must already be validated (finite, 0 < speed <= MAX_SPEED)."""
        with self.lock:
            if speed is not None:
                self.speed = float(speed)
            if pause is not None:
                self.paused = bool(pause)
            return dict(speed=self.speed, paused=self.paused, t=self.twin.t)


def _json_default(o):
    if hasattr(o, "item"):                      # numpy scalar
        return o.item()
    raise TypeError(repr(o))


def parse_speed(text: str) -> float | None:
    """The /control speed, or None when it is not a finite number in (0, MAX_SPEED]."""
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) and 0.0 < v <= MAX_SPEED else None


def make_handler(sim: Simulation):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):      # quiet
            if os.environ.get("TWIN_WEBAPP_LOG"):
                super().log_message(fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, path: str):
            """(code, body, content type) of the side-effect-free routes, or None."""
            if path in ("/", "/index.html"):
                with open(os.path.join(STATIC_DIR, "index.html"), "rb") as fh:
                    return 200, fh.read(), "text/html; charset=utf-8"
            if path == "/frame":
                return 200, sim.frame_json.encode(), "application/json"
            if path == "/geometry":
                return 200, sim.geometry_json.encode(), "application/json"
            if path == "/health":
                return 200, json.dumps(dict(ok=True, t=sim.twin.t, speed=sim.speed, paused=sim.paused,
                                            achieved=round(sim.achieved, 1))).encode(), "application/json"
            return None

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/stream":
                self.stream()
            elif url.path == "/control":
                q = parse_qs(url.query)
                speed = None
                if "speed" in q:
                    speed = parse_speed(q["speed"][0])
                    if speed is None:
                        msg = dict(error=f"speed must be a number in (0, {MAX_SPEED:g}]")
                        self._send(400, json.dumps(msg).encode(), "application/json")
                        return
                pause = (q["pause"][0] not in ("0", "false", "")) if "pause" in q else None
                self._send(200, json.dumps(sim.control(speed, pause)).encode(), "application/json")
            else:
                self._send(*(self._static(url.path) or (404, b"not found", "text/plain")))

        def do_HEAD(self):
            res = self._static(urlparse(self.path).path)
            code, body, ctype = res or (404, b"not found", "text/plain")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            last_seq = -1
            try:
                while not sim.stop_flag:
                    seq, body = sim.frame_seq, sim.frame_json
                    if seq != last_seq:
                        self.wfile.write(b"data: " + body.encode() + b"\n\n")
                        self.wfile.flush()
                        last_seq = seq
                    time.sleep(TICK_S)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                self.close_connection = True        # an event stream never returns to keep-alive

    return Handler


def build_twin(args) -> TwinWorld:
    cfg = load_base_config(args.config)
    return TwinWorld(cfg, demand_level=args.demand, controller=args.controller, model_path=args.model, seed=args.seed,
                     profile=args.profile, start_hour=args.start_hour)


def serve(args):
    twin = build_twin(args)
    if args.freeze:
        twin.bank.freeze_camera(args.freeze, args.freeze_t0, args.freeze_dur)
    sim = Simulation(twin, speed=args.speed)
    sim.start()
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(sim))
    httpd.daemon_threads = True
    print(f"twin web app on http://{args.host}:{args.port}/  controller={twin.controller_label}  "
          f"demand={args.demand} x {args.profile} profile  speed={args.speed}x", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop_flag = True
        httpd.server_close()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="live web demo of the traffic-light digital twin")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--speed", type=float, default=60.0, help="simulated seconds per wall second")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--controller", default="ml", help="ml (default) | priority_mp | max_pressure | fixed_time ...")
    ap.add_argument("--config", default=None, help="config.json of the release (lambda_base per level)")
    ap.add_argument("--demand", default="design")
    ap.add_argument("--profile", default="daily", choices=["daily", "flat"])
    ap.add_argument("--start-hour", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--freeze", default=None, help="edge id whose camera is frozen (fault demo)")
    ap.add_argument("--freeze-t0", type=float, default=600.0)
    ap.add_argument("--freeze-dur", type=float, default=60.0)
    args = ap.parse_args(argv)
    return serve(args)


if __name__ == "__main__":
    sys.exit(main())
