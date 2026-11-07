"""MRT Jakarta North-South Line through the twin's area, for the live 3D view only (DECISIONS §13.5).

Track, stations and stop positions come from OpenStreetMap ((c) OpenStreetMap contributors, ODbL), downloaded
once into data/fatmawati/osm/mrt.json.  The line runs on a viaduct through the whole area and meets no road
traffic, so its trains are not simulated in SUMO: they run to MRT Jakarta's published service pattern.
- 05:00-24:00, a train every 5 min in the peaks (07-09, 17-19) and every 10 min otherwise.
- 6 cars of 20 m; up to 80 km/h, 0.9 m/s2 acceleration and braking; 30 s at intermediate stations.
- Northbound trains start at Lebak Bulus (the terminus) and leave the area north of Cipete Raya.
  Southbound trains enter from the north and end at Lebak Bulus.
The rail level (RAIL_Z) is an assumption: the viaduct crosses above the JORR deck (6 m) and its traffic.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import urllib.parse
import urllib.request

from .build_net import bbox
from .demand import DATA

OSM = os.path.join(DATA, "osm", "mrt.json")
RADIUS_KM = 2.7                          # the road network's box (build_net --radius-km)
RAIL_Z = 14.0                            # rail level above the street
CAR_M, N_CARS, COUPLING_M = 20.0, 6, 0.5
TRAIN_M = N_CARS * CAR_M + (N_CARS - 1) * COUPLING_M
V_MAX, ACC = 80.0 / 3.6, 0.9
DWELL_S, TERMINUS_S = 30.0, 120.0
SERVICE_H = (5.0, 24.0)
PEAKS_H = ((7.0, 9.0), (17.0, 19.0))
HEADWAY_S = {"peak": 300.0, "off": 600.0}
MRT_CODE = 5                             # vehicle type code in the viewer frames


def download(path: str = OSM, radius_km: float = RADIUS_KM, force: bool = False) -> str:
    """Subway tracks, stations, stop positions and station buildings in the box (Overpass API, cached)."""
    if os.path.exists(path) and not force:
        return path
    s, w, n, e = bbox(radius_km)
    b = f"{s:.5f},{w:.5f},{n:.5f},{e:.5f}"
    q = (f'[out:json][timeout:180];(way["railway"="subway"]({b});node["railway"~"^(station|stop)$"]({b});'
         f'way["building"="train_station"]({b}););out geom tags;')
    req = urllib.request.Request("https://overpass-api.de/api/interpreter", data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "traffic_twin/3.0 (ITI5145 student project)"})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read()
    json.loads(body)                                             # fail before caching a broken answer
    with open(path, "wb") as f:
        f.write(body)
    return path


def _heading(a, b) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _chains(polys: dict) -> list:
    """Ways joined head to tail into through tracks; at a fork the straightest continuation wins."""
    key = lambda p: (round(p[0], 1), round(p[1], 1))
    starts = {}
    for wid, p in polys.items():
        starts.setdefault(key(p[0]), []).append(wid)

    def after(wid):
        p, best, bd = polys[wid], None, math.radians(60)
        for o in starts.get(key(p[-1]), []):
            q = polys[o]
            d = abs((_heading(q[0], q[1]) - _heading(p[-2], p[-1]) + math.pi) % (2 * math.pi) - math.pi)
            if o != wid and d < bd:
                best, bd = o, d
        return best

    nxt = {w: after(w) for w in polys}
    chains = []
    for w in polys:
        if w in nxt.values():
            continue
        c = [w]
        while nxt[c[-1]] and nxt[c[-1]] not in c:
            c.append(nxt[c[-1]])
        pts = []
        for wid in c:
            pts += polys[wid][1:] if pts else polys[wid]
        chains.append(pts)
    return chains


def _clip(pts: list, box: tuple) -> list:
    """The longest run of the polyline inside the box (x0, y0, x1, y1), cut where it crosses the edge."""
    x0, y0, x1, y1 = box
    inside = lambda p: x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def cut(a, b):                                               # a inside, b outside
        f = 1.0
        for lo, hi, i in ((x0, x1, 0), (y0, y1, 1)):
            for lim in (lo, hi):
                if (b[i] - lim) * (a[i] - lim) < 0:
                    f = min(f, (lim - a[i]) / (b[i] - a[i]))
        return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))

    runs, cur = [], []
    for i, p in enumerate(pts):
        if inside(p):
            if not cur and i > 0:
                cur.append(cut(p, pts[i - 1]))
            cur.append(p)
        elif cur:
            cur.append(cut(pts[i - 1], p))
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return max(runs, key=lambda r: sum(math.dist(a, b) for a, b in zip(r, r[1:])), default=[])


def _along(pts: list) -> list:
    s = [0.0]
    for a, b in zip(pts, pts[1:]):
        s.append(s[-1] + math.dist(a, b))
    return s


def _project(pts: list, q) -> tuple:
    """(arc length, distance) of the point of the polyline nearest to q."""
    best, s = (math.inf, 0.0), 0.0
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / L2))
        d = math.hypot(q[0] - a[0] - t * dx, q[1] - a[1] - t * dy)
        if d < best[0]:
            best = (d, s + t * math.sqrt(L2))
        s += math.sqrt(L2)
    return best[1], best[0]


def lines(to_xy, box: tuple, path: str = OSM) -> dict:
    """The two main-line tracks in local metres (to_xy(lon, lat) -> (x, y)), clipped to the box, each with its
    stops (station name, arc length); and the station buildings."""
    with open(path) as f:
        els = json.load(f)["elements"]
    polys = {e["id"]: [to_xy(p["lon"], p["lat"]) for p in e["geometry"]] for e in els
             if e["type"] == "way" and e.get("tags", {}).get("railway") == "subway" and "service" not in e["tags"]}
    tracks = sorted((_clip(c, box) for c in _chains(polys)), key=lambda c: -sum(math.dist(a, b) for a, b in zip(c, c[1:])))[:2]
    stations = {e["tags"]["name"]: to_xy(e["lon"], e["lat"]) for e in els
                if e["type"] == "node" and e.get("tags", {}).get("railway") == "station" and "name" in e["tags"]}
    stops = [(e["tags"].get("name", ""), to_xy(e["lon"], e["lat"])) for e in els
             if e["type"] == "node" and e.get("tags", {}).get("railway") == "stop"]
    out = {}
    for pts in tracks:
        name = "northbound" if pts[-1][1] > pts[0][1] else "southbound"
        at = []
        for st, xy in stations.items():
            cands = [q for n, q in stops if n == st] or [xy]
            s, d = min((_project(pts, q) for q in cands), key=lambda r: r[1])
            if d < 30.0 and 0.0 < s < _along(pts)[-1]:
                at.append((st, round(s, 1)))
        out[name] = dict(p=[[round(x, 2), round(y, 2)] for x, y in pts], stops=sorted(at, key=lambda r: r[1]))
    buildings = [dict(name=e["tags"].get("name", ""), p=[list(map(lambda v: round(v, 2), to_xy(p["lon"], p["lat"])))
                                                          for p in e["geometry"][:-1]])
                 for e in els if e["type"] == "way" and e.get("tags", {}).get("building") == "train_station"]
    return dict(tracks=out, stations=buildings)


def _leg(D: float, v0: float, v1: float | None) -> list:
    """Motion segments (duration, start speed, acceleration) covering D metres from v0 to a stop (v1 = 0), or
    out of the area at line speed (v1 = None)."""
    if v1 is None:
        d_acc = (V_MAX * V_MAX - v0 * v0) / (2 * ACC)
        if D <= d_acc:
            return [((math.sqrt(v0 * v0 + 2 * ACC * D) - v0) / ACC, v0, ACC)]
        return [s for s in [((V_MAX - v0) / ACC, v0, ACC), ((D - d_acc) / V_MAX, V_MAX, 0.0)] if s[0] > 1e-6]
    vp = min(V_MAX, math.sqrt(max((2 * ACC * D + v0 * v0 + v1 * v1) / 2, 0.0)))
    d_acc, d_dec = (vp * vp - v0 * v0) / (2 * ACC), (vp * vp - v1 * v1) / (2 * ACC)
    segs = [((vp - v0) / ACC, v0, ACC)]
    if D - d_acc - d_dec > 1e-6:
        segs.append(((D - d_acc - d_dec) / vp, vp, 0.0))
    segs.append(((vp - v1) / ACC, vp, -ACC))
    return [s for s in segs if s[0] > 1e-6]


class Trains:
    """Positions of every car on both tracks at a time of day (seconds after midnight)."""

    def __init__(self, tracks: dict):
        self.dirs = []
        day = []
        h = SERVICE_H[0] * 3600.0
        while h < SERVICE_H[1] * 3600.0 - 1800.0:               # departures from the termini (last about 23:30)
            day.append(h)
            peak = any(a * 3600.0 <= h < b * 3600.0 for a, b in PEAKS_H)
            h += HEADWAY_S["peak" if peak else "off"]
        for k, (name, tr) in enumerate(sorted(tracks.items())):
            pts, s = [tuple(p) for p in tr["p"]], _along(tr["p"])
            stops = [st for _, st in tr["stops"]]
            if len(pts) < 2 or not stops:
                continue
            # a stopped train's middle is at the stop position; northbound trains start at the first stop (the terminus)
            starts_here = name == "northbound"
            front = [st + TRAIN_M / 2 for st in stops]
            segs, s0, v0 = [], 0.0, V_MAX
            if starts_here:
                s0, v0 = front[0], 0.0
                segs.append((TERMINUS_S, 0.0, 0.0))
                front = front[1:]
            for i, f in enumerate(front):
                segs += _leg(f - s0, v0, 0.0)
                last = not starts_here and i == len(front) - 1
                segs.append((TERMINUS_S if last else DWELL_S, 0.0, 0.0))
                s0, v0 = f, 0.0
            if starts_here:                                      # out of the area at line speed
                segs += _leg(s[-1] + TRAIN_M - s0, v0, None)
            t, prof = 0.0, []
            sp = stops[0] + TRAIN_M / 2 if starts_here else 0.0
            for dur, vstart, a in segs:                          # (start time, start position, speed, acceleration)
                prof.append((t, sp, vstart, a))
                sp += vstart * dur + a * dur * dur / 2
                t += dur
            offset = 0.0 if starts_here else HEADWAY_S["peak"] / 2    # southbound: as if from Bundaran HI
            self.dirs.append(dict(k=k, pts=pts, s=s, prof=prof, end=t,
                                  departs=[d + offset - (TERMINUS_S if starts_here else 0.0) for d in day]))

    @staticmethod
    def _state(prof: list, tau: float) -> tuple:
        i = bisect.bisect_right([p[0] for p in prof], tau) - 1
        t0, s0, v0, a = prof[max(i, 0)]
        dt = tau - t0
        return s0 + v0 * dt + a * dt * dt / 2, v0 + a * dt

    @staticmethod
    def _point(pts: list, s: list, at: float) -> tuple:
        i = min(max(bisect.bisect_right(s, at) - 1, 0), len(pts) - 2)
        f = (at - s[i]) / ((s[i + 1] - s[i]) or 1.0)
        return pts[i][0] + f * (pts[i + 1][0] - pts[i][0]), pts[i][1] + f * (pts[i + 1][1] - pts[i][1])

    def rows(self, clock_s: float) -> list:
        """Viewer rows [id, x, y, z, angle, MRT_CODE, speed] for every car fully inside the area."""
        c = clock_s % 86400.0
        out = []
        for d in self.dirs:
            L = d["s"][-1]
            for n, t0 in enumerate(d["departs"]):
                tau = c - t0
                if tau < 0 or tau > d["end"]:
                    continue
                head, v = self._state(d["prof"], tau)
                for i in range(N_CARS):
                    f = head - i * (CAR_M + COUPLING_M)
                    if f > L or f - CAR_M < 0:
                        continue
                    (fx, fy), (rx, ry) = self._point(d["pts"], d["s"], f), self._point(d["pts"], d["s"], f - CAR_M)
                    ang = math.degrees(math.atan2(fx - rx, fy - ry)) % 360.0
                    out.append([30_000_000 + d["k"] * 1_000_000 + n * 10 + i, round(fx, 2), round(fy, 2), RAIL_Z,
                                round(ang, 1), MRT_CODE, round(v, 1)])
        return out
