"""Virtual sensors at Perempatan Fatmawati (DECISIONS §13.6): the twin's dataset is what a roadside
installation would record, not the simulator's internal state.

- stop-line loops: one induction loop per approach lane, 1 m before the stop line (discharge counts by class,
  occupancy, speed at the loop);
- entry and exit gates: a counting line across every lane of each leg, outside the free-left slips and the median
  U-turn loops (cordon counts).  Matching a vehicle's entry and exit, as video or ANPR re-identification does, gives
  turning-movement counts and cordon travel times;
- queue cameras: the 300 m upstream of each stop line (standing vehicles by class, queue tail distance,
  motorcycles in the box, speed), and exit cameras over the first 150 m of each exit leg (spillback);
- emergency-vehicle detector: an ambulance inside an approach camera zone (siren or GPS priority request).

Sensors are exact (no detection errors).  Vehicle identities are used only to match entries with exits and are
not written out.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict

import numpy as np

from .build_net import APPROACHES, EXITS, JUNCTION_LAT, JUNCTION_LON, TLS_ID
from .demand import DATA, NET

LEGS = ("N", "E", "S", "W")
GATE_R = {"N": 150.0, "E": 150.0, "S": 150.0, "W": 185.0}     # W: outside the U-turn loop that leaves RA Kartini 169 m out
CAMERA_M = 300.0
EXIT_CAMERA_M = 150.0
LOOP_SETBACK_M = 1.0
CLASSES = ("motorcycle", "car", "bus", "truck", "emergency")
VCLASS = {"motorcycle": "motorcycle", "passenger": "car", "bus": "bus", "truck": "truck", "emergency": "emergency"}
PCU = {"motorcycle": 0.15, "car": 1.0, "bus": 1.3, "truck": 1.3, "emergency": 1.0}   # PKJI 2023 emp, protected phase
HALT_MS = 0.1                                                  # standing: slower than 0.1 m/s (SUMO's halting threshold)
QUEUE_MS, QUEUE_GAP_M = 5.0 / 3.6, 20.0                        # queue length as VISSIM's queue counter: < 5 km/h, gaps <= 20 m
EV_ETA_MIN_MS = 5.0                                            # ETA uses at least 5 m/s (a stopped ambulance still needs serving)
ADDITIONAL = os.path.join(DATA, "sensors", "sensors.add.xml")


# ---------------------------------------------------------------- layout (static, from the network)
def _same_street_pred(e, chain):
    c = [p for p in e.getIncoming() if p.getName() == e.getName() and p not in chain and p.getFromNode() is not e.getToNode()]
    return c[0] if len(c) == 1 else None


def _same_street_succ(e, chain):
    c = [s for s in e.getOutgoing() if s.getName() == e.getName() and s not in chain and s.getToNode() is not e.getFromNode()]
    return c[0] if len(c) == 1 else None


def chain_upstream(net, edge_id: str, length: float) -> list:
    """[(edge, distance from the end of edge to the stop line)] walking up the same street until `length`."""
    out, e, off = [], net.getEdge(edge_id), 0.0
    while e is not None and off < length:
        out.append((e, off))
        off += e.getLength()
        e = _same_street_pred(e, [x for x, _ in out])
    return out


def chain_downstream(net, edge_id: str, length: float) -> list:
    """[(edge, distance from the junction exit to the start of edge)] walking down the same street until `length`."""
    out, e, off = [], net.getEdge(edge_id), 0.0
    while e is not None and off < length:
        out.append((e, off))
        off += e.getLength()
        e = _same_street_succ(e, [x for x, _ in out])
    return out


def _crossing_point(shape, x0, y0, radius):
    """First position along a polyline whose distance to (x0, y0) reaches `radius` (walking from its start)."""
    acc = 0.0
    for (xa, ya), (xb, yb) in zip(shape, shape[1:]):
        seg = math.hypot(xb - xa, yb - ya)
        da, db = math.hypot(xa - x0, ya - y0), math.hypot(xb - x0, yb - y0)
        if (da - radius) * (db - radius) <= 0 and seg > 0:
            n = max(2, int(seg / 0.25))
            for i in range(n + 1):
                f = i / n
                if (math.hypot(xa + (xb - xa) * f - x0, ya + (yb - ya) * f - y0) - radius) * (da - radius) <= 0:
                    return acc + seg * f
        acc += seg
    return None


def _gate(net, chain, x0, y0, radius, upstream: bool):
    """(edge, position from the edge start) where the leg's carriageway crosses the circle of `radius`."""
    for e, _ in chain:
        shape = e.getLanes()[0].getShape()
        s = _crossing_point(shape[::-1] if upstream else shape, x0, y0, radius)
        if s is not None:
            L = e.getLanes()[0].getLength()
            return e, (L - s if upstream else s)
    raise ValueError(f"no gate at {radius} m on {[e.getID() for e, _ in chain]}")


def build_layout(net_path: str = NET) -> dict:
    """Sensor inventory: loops (point sensors on one lane) and cameras (zones over several lane segments)."""
    import sumolib
    net = sumolib.net.readNet(net_path, withInternal=False)
    x0, y0 = net.convertLonLat2XY(JUNCTION_LON, JUNCTION_LAT)

    def lonlat(lane, pos):
        x, y = sumolib.geomhelper.positionAtShapeOffset(lane.getShape(), pos)
        lon, lat = net.convertXY2LonLat(x, y)
        return round(lat, 6), round(lon, 6), round(math.hypot(x - x0, y - y0), 1)

    from .build_net import stage_link_sets
    green, _median, _n = stage_link_sets(net_path)
    tls = sumolib.net.readNet(net_path, withInternal=False, withPrograms=True).getTLS(TLS_ID)
    lane_stages = defaultdict(set)                             # approach lane id -> stages that give it green
    for inl, _outl, idx in tls.getConnections():
        lane_stages[inl.getID()] |= {st for st, links in green.items() if idx in links}
    loops, cameras = [], []
    for leg in LEGS:
        e = net.getEdge(APPROACHES[leg])
        app_groups = ["+".join(sorted(lane_stages[ln.getID()])) for ln in e.getLanes()]
        for ln in e.getLanes():                                  # stop-line loops
            pos = max(0.1, ln.getLength() - LOOP_SETBACK_M)
            lat, lon, d = lonlat(ln, pos)
            loops.append(dict(sensor_id=f"SL_{leg}{ln.getIndex()}", kind="stopline_loop", leg=leg, lane=ln.getIndex(),
                              lane_id=ln.getID(), pos=round(pos, 2), lat=lat, lon=lon, dist_centre_m=d,
                              street=e.getName()))
        up = chain_upstream(net, APPROACHES[leg], CAMERA_M + 400.0)
        ge, gpos = _gate(net, up, x0, y0, GATE_R[leg], upstream=True)
        for ln in ge.getLanes():                                 # entry gate
            pos = min(max(0.1, gpos), ln.getLength() - 0.1)
            lat, lon, d = lonlat(ln, pos)
            loops.append(dict(sensor_id=f"EN_{leg}{ln.getIndex()}", kind="entry_gate", leg=leg, lane=ln.getIndex(),
                              lane_id=ln.getID(), pos=round(pos, 2), lat=lat, lon=lon, dist_centre_m=d, street=ge.getName()))
        down = chain_downstream(net, EXITS[leg], 600.0)
        xe, xpos = _gate(net, down, x0, y0, GATE_R[leg], upstream=False)
        for ln in xe.getLanes():                                 # exit gate
            pos = min(max(0.1, xpos), ln.getLength() - 0.1)
            lat, lon, d = lonlat(ln, pos)
            loops.append(dict(sensor_id=f"EX_{leg}{ln.getIndex()}", kind="exit_gate", leg=leg, lane=ln.getIndex(),
                              lane_id=ln.getID(), pos=round(pos, 2), lat=lat, lon=lon, dist_centre_m=d, street=xe.getName()))
        segs, group_of = [], {}                                  # approach queue camera: 300 m up from the stop line
        for ce, off in chain_upstream(net, APPROACHES[leg], CAMERA_M):
            n = ce.getLaneNumber()
            for ln in ce.getLanes():
                L = ln.getLength()
                segs.append((ln.getID(), max(0.0, L - (CAMERA_M - off)), L, off))
                # lane group: the stop-line lane at the same relative position across the road
                k = min(len(app_groups) - 1, int((ln.getIndex() + 0.5) * len(app_groups) / n))
                group_of[ln.getID()] = app_groups[k]
        lat, lon, d = lonlat(e.getLanes()[0], e.getLanes()[0].getLength())
        width = sum(ln.getWidth() for ln in e.getLanes())
        cameras.append(dict(sensor_id=f"QC_{leg}", kind="queue_camera", leg=leg, segments=segs, length_m=CAMERA_M,
                            lat=lat, lon=lon, dist_centre_m=d, street=e.getName(), box_m=8.0, width_m=round(width, 2),
                            sat_flow_pcu_h=round(600.0 * width),        # PKJI 2023 (Bina Marga 09/P/BM/2023) base S0 = 600 W_e
                            group_of=group_of, groups=sorted(set(app_groups))))
        segs = []                                                # exit camera: first 150 m of the exit leg
        for ce, off in chain_downstream(net, EXITS[leg], EXIT_CAMERA_M):
            for ln in ce.getLanes():
                segs.append((ln.getID(), 0.0, min(ln.getLength(), EXIT_CAMERA_M - off), off))
        x_e = net.getEdge(EXITS[leg])
        lat, lon, d = lonlat(x_e.getLanes()[0], 0.0)
        cameras.append(dict(sensor_id=f"XC_{leg}", kind="exit_camera", leg=leg, segments=segs, length_m=EXIT_CAMERA_M,
                            lat=lat, lon=lon, dist_centre_m=d, street=x_e.getName(), box_m=0.0))
    return dict(loops=loops, cameras=cameras)


def write_additional(layout: dict, path: str = ADDITIONAL) -> str:
    """SUMO induction loops for the point sensors; their own XML output is discarded (readings go to CSV)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("<!-- Virtual sensors of the Fatmawati twin (traffic_twin/fatmawati/sensors.py); read through libsumo. -->\n<additional>\n")
        for s in layout["loops"]:
            f.write(f'    <inductionLoop id="{s["sensor_id"]}" lane="{s["lane_id"]}" pos="{s["pos"]}" period="86400" file="{os.devnull}"/>\n')
        f.write("</additional>\n")
    return path


def sensor_rows(layout: dict) -> list:
    """Rows of sensors.csv."""
    rows = []
    for s in layout["loops"]:
        rows.append(dict(sensor_id=s["sensor_id"], kind=s["kind"], leg=s["leg"], lane=s["lane"], street=s["street"],
                         lat=s["lat"], lon=s["lon"], dist_centre_m=s["dist_centre_m"], zone_length_m=0.0, n_lane_segments=1))
    for c in layout["cameras"]:
        rows.append(dict(sensor_id=c["sensor_id"], kind=c["kind"], leg=c["leg"], lane=-1, street=c["street"],
                         lat=c["lat"], lon=c["lon"], dist_centre_m=c["dist_centre_m"], zone_length_m=c["length_m"],
                         n_lane_segments=len(c["segments"])))
    return rows


# ---------------------------------------------------------------- readings (live, through libsumo)
class SensorBank:
    """Reads the virtual sensors every simulation step and aggregates them into the tables of the release.

    Loops are read every step (counts by class at the moment a vehicle enters the loop, occupancy, speed at the
    loop); cameras are sampled once per second; ticks are written every `tick_s`; nothing is recorded before
    `record_from` (warm-up)."""

    def __init__(self, ts, layout: dict, record_from: float = 0.0, tick_s: float = 5.0, store_rows: bool = True):
        self.ts, self.layout, self.record_from, self.tick_s = ts, layout, record_from, tick_s
        self.store_rows = store_rows                # False in the live viewer: current readings only, no tables
        self.loops = {s["sensor_id"]: s for s in layout["loops"]}
        self.cameras = layout["cameras"]
        self.cls_of_type = {}
        self.seen = {}                             # (kind, leg) -> {vehicle: last detection t}
        self.t_prev = 0.0
        self.minute = None
        self.acc_loop = {}
        self.acc_cam = {}
        self.rows_loops, self.rows_cams, self.rows_ticks = [], [], []
        self.recent = defaultdict(list)            # (kind, leg) -> [(t, class)] of recent loop entries
        self.cam_now = {}                          # camera id -> latest 1 Hz snapshot
        self.queue_series = defaultdict(list)      # leg -> [(t, standing vehicles)] at 1 Hz (main period)
        self.pending = {}                          # vehicle -> (entry leg, t, class): re-identification
        self.cordon = defaultdict(list)            # (15-min period, entry leg, exit leg, class) -> travel times
        self.n_unmatched_exit = 0
        self.ev = {}                               # ambulance -> AVL-style trip record
        self.ev_done = []
        self.turn_prior = np.ones((4, 4))          # turning counts so far (for the max-pressure teacher)

    # -- helpers
    def vclass(self, vid: str) -> str:
        return self._type_class(self.ts.vehicle.getTypeID(vid))

    def _type_class(self, type_id: str) -> str:
        c = self.cls_of_type.get(type_id)
        if c is None:
            c = self.cls_of_type[type_id] = VCLASS.get(self.ts.vehicletype.getVehicleClass(type_id), "car")
        return c

    # -- every step
    def step(self, t: float) -> None:
        ts, rec = self.ts, t > self.record_from + 1e-9
        minute = int(t // 60)
        if self.minute is None:
            self.minute = minute
        if minute != self.minute:
            self._flush_minute()
            self.minute = minute
        for sid, s in self.loops.items():
            data = ts.inductionloop.getVehicleData(sid)
            occ = ts.inductionloop.getLastStepOccupancy(sid)
            a = self.acc_loop.setdefault(sid, dict(n={c: 0 for c in CLASSES}, occ=0.0, steps=0, spd=0.0, spd_n=0))
            a["occ"] += occ
            a["steps"] += 1
            for vid, _length, t_in, _t_out, type_id in data:
                if t_in <= self.t_prev + 1e-9:
                    continue                                   # already counted when it entered
                seen = self.seen.setdefault((s["kind"], s["leg"]), {})
                if t - seen.get(vid, -1e9) < 30.0:
                    continue                                   # one vehicle on two loops of a gate (straddling lanes)
                seen[vid] = t
                c = self._type_class(type_id)
                a["n"][c] += 1
                v = ts.vehicle.getSpeed(vid)
                a["spd"] += v
                a["spd_n"] += 1
                self._on_loop(s, vid, c, t, rec)
        if int(t) != int(self.t_prev):                        # cameras at 1 Hz
            self._sample_cameras(t, rec)
        if self.store_rows and t >= self.record_from - 1e-9 and abs(t / self.tick_s - round(t / self.tick_s)) < 1e-6:
            self._tick_snapshot(t)
        self.t_prev = t

    def _on_loop(self, s: dict, vid: str, c: str, t: float, rec: bool) -> None:
        kind, leg = s["kind"], s["leg"]
        self.recent[(kind, leg)].append((t, c))
        if kind == "entry_gate":
            self.pending[vid] = (leg, t, c)
            if c == "emergency" and vid in self.ev:
                self.ev[vid].update(entry_leg=leg, t_entry_gate=t)
                self.ev[vid].setdefault("entries", []).append((leg, t))
        elif kind == "exit_gate":
            hit = self.pending.pop(vid, None)
            if hit is None:
                self.n_unmatched_exit += rec
            else:
                leg_in, t_in, c_in = hit
                self.turn_prior[LEGS.index(leg_in), LEGS.index(leg)] += 1
                if rec and self.store_rows and t_in > self.record_from:
                    self.cordon[(int(t // 900), leg_in, leg, c_in)].append(t - t_in)
            if c == "emergency" and vid in self.ev:
                self.ev[vid].update(exit_leg=leg, t_exit_gate=t)            # the last exit (a detour passes several)
                self.ev[vid].setdefault("exits", []).append((leg, t))
        elif kind == "stopline_loop":
            if c == "emergency" and vid in self.ev and "t_stopline" not in self.ev[vid]:
                self.ev[vid].update(stopline_leg=leg, t_stopline=t)

    def _sample_cameras(self, t: float, rec: bool) -> None:
        ts = self.ts
        for cam in self.cameras:
            q_veh = q_pcu = n_veh = moto_box = 0
            q_m, spd, ev_d, ev_v, ev_id = 0.0, 0.0, None, None, None
            standing = []                                          # (distance, PCU) of standing vehicles
            slow = []                                              # (distance, length) of vehicles below 5 km/h
            approach = cam["kind"] == "queue_camera"
            q_grp = {g: 0 for g in cam.get("groups", ())}
            q_grp_pcu = {g: 0.0 for g in cam.get("groups", ())}
            for lane, a, b, off in cam["segments"]:
                L = ts.lane.getLength(lane)
                for vid in ts.lane.getLastStepVehicleIDs(lane):
                    pos = ts.vehicle.getLanePosition(vid)
                    if pos < a or pos > b:
                        continue
                    c = self.vclass(vid)
                    v = ts.vehicle.getSpeed(vid)
                    d = (L - pos) + off if approach else off + pos      # from the stop line / the junction exit
                    n_veh += 1
                    spd += v
                    if approach and v < QUEUE_MS:
                        slow.append((d, ts.vehicle.getLength(vid)))
                    if v < HALT_MS:
                        q_veh += 1
                        q_pcu += PCU[c]
                        if not approach:
                            q_m = max(q_m, d)
                        standing.append((d, PCU[c]))
                        if approach:
                            g = cam["group_of"][lane]
                            q_grp[g] += 1
                            q_grp_pcu[g] += PCU[c]
                    if approach and c == "motorcycle" and d <= cam["box_m"]:
                        moto_box += 1
                    if approach and c == "emergency":
                        if ev_d is None or d < ev_d:
                            ev_d, ev_v, ev_id = d, v, vid
                        self._ev_seen(vid, cam["leg"], t, d)
            if approach:                                           # back of the queue that reaches the stop line
                for d, L in sorted(slow):
                    if d - q_m > QUEUE_GAP_M:
                        break
                    q_m = max(q_m, d + L)
            snap = dict(q_veh=q_veh, q_pcu=round(q_pcu, 2), q_m=round(q_m, 1), n_veh=n_veh,
                        speed_kmh=round(3.6 * spd / n_veh, 1) if n_veh else None, moto_box=moto_box, ev_d=ev_d)
            if approach:
                snap.update(q_groups=q_grp, q_groups_pcu={g: round(v, 2) for g, v in q_grp_pcu.items()})
            if ev_d is not None:                                   # what a GPS priority request would report
                route, k = ts.vehicle.getRoute(ev_id), ts.vehicle.getRouteIndex(ev_id)
                snap.update(ev_eta_s=round(ev_d / max(ev_v, EV_ETA_MIN_MS), 1),
                            ev_q_ahead_pcu=round(sum(e for d, e in standing if d < ev_d), 2),
                            ev_needs_signal=APPROACHES[cam["leg"]] in route[max(k, 0):])   # its route crosses the stop line
            self.cam_now[cam["sensor_id"]] = snap
            if rec:
                a = self.acc_cam.setdefault(cam["sensor_id"], dict(n=0, q_veh=0, q_veh_max=0, q_pcu=0.0, q_m=0.0, q_m_max=0.0,
                                                                   n_veh=0, spd=0.0, spd_n=0, moto_box=0, ev_s=0))
                a["n"] += 1
                a["q_veh"] += q_veh
                a["q_veh_max"] = max(a["q_veh_max"], q_veh)
                a["q_pcu"] += q_pcu
                a["q_m"] += q_m
                a["q_m_max"] = max(a["q_m_max"], q_m)
                a["n_veh"] += n_veh
                a["spd"] += spd
                a["spd_n"] += n_veh
                a["moto_box"] += moto_box
                a["ev_s"] += ev_d is not None
                if approach and self.store_rows:
                    self.queue_series[cam["leg"]].append((t, q_veh))
        if rec:
            self._ev_recovery(t)

    # -- ambulances (AVL-style log: the ambulance's own GPS plus the priority-request detector)
    def ev_departed(self, vid: str, t: float) -> None:
        self.ev[vid] = dict(t_depart=t, vid=vid)

    def ev_arrived(self, vid: str, t: float) -> None:
        r = self.ev.pop(vid, None)
        if r is not None:
            r["t_arrive"] = t
            self.ev_done.append(r)

    def _ev_seen(self, vid: str, leg: str, t: float, d: float) -> None:
        r = self.ev.get(vid)
        if r is not None and "t_detect" not in r:
            r.update(detect_leg=leg, t_detect=t, detect_dist_m=round(d, 1),
                     queue_at_detect=sum(self.cam_now.get(f"QC_{g}", {}).get("q_veh", 0) for g in LEGS))

    def _ev_recovery(self, t: float) -> None:
        total = sum(self.cam_now.get(f"QC_{g}", {}).get("q_veh", 0) for g in LEGS)
        for r in list(self.ev.values()) + self.ev_done:
            if "t_stopline" in r and "t_recovered" not in r and t >= r["t_stopline"]:
                q0 = r.get("queue_at_detect", 0)
                if total <= max(1.1 * q0, q0 + 1):
                    r["t_recovered"] = t

    # -- aggregation
    def _flush_minute(self) -> None:
        t0 = self.minute * 60.0
        if self.store_rows and t0 >= self.record_from - 1e-9:
            for sid, a in self.acc_loop.items():
                s = self.loops[sid]
                n = sum(a["n"].values())
                self.rows_loops.append(dict(t_start=t0, sensor_id=sid, kind=s["kind"], leg=s["leg"], lane=s["lane"],
                                            **{f"n_{c}": a["n"][c] for c in CLASSES}, n_total=n,
                                            pcu=round(sum(PCU[c] * a["n"][c] for c in CLASSES), 2),
                                            occupancy_pct=round(a["occ"] / max(a["steps"], 1), 2),
                                            speed_kmh=round(3.6 * a["spd"] / a["spd_n"], 1) if a["spd_n"] else None))
            for cam in self.cameras:
                a = self.acc_cam.get(cam["sensor_id"])
                if not a or not a["n"]:
                    continue
                n = a["n"]
                self.rows_cams.append(dict(t_start=t0, sensor_id=cam["sensor_id"], kind=cam["kind"], leg=cam["leg"],
                                           queue_veh_mean=round(a["q_veh"] / n, 2), queue_veh_max=a["q_veh_max"],
                                           queue_pcu_mean=round(a["q_pcu"] / n, 2), queue_m_mean=round(a["q_m"] / n, 1),
                                           queue_m_max=round(a["q_m_max"], 1), vehicles_mean=round(a["n_veh"] / n, 2),
                                           speed_kmh_mean=round(3.6 * a["spd"] / a["spd_n"], 1) if a["spd_n"] else None,
                                           moto_in_box_mean=round(a["moto_box"] / n, 2) if cam["kind"] == "queue_camera" else None,
                                           ambulance_present_s=a["ev_s"]))
        self.acc_loop, self.acc_cam = {}, {}
        for seen in self.seen.values():
            for vid in [v for v, tt in seen.items() if tt < t0 - 60.0]:
                del seen[vid]

    def counts_since(self, kind: str, leg: str, t: float, window: float) -> int:
        lst = self.recent[(kind, leg)]
        while lst and lst[0][0] <= t - 120.0:                 # keep two minutes
            lst.pop(0)
        return sum(1 for tt, _ in lst if tt > t - window)

    def pressures(self) -> dict:
        """Max-pressure on sensor readings: approach queue (PCU) minus the queue on the exits it feeds, weighted by
        the turning shares counted at the gates so far."""
        share = self.turn_prior / self.turn_prior.sum(axis=1, keepdims=True)
        q_in = {g: self.cam_now.get(f"QC_{g}", {}).get("q_pcu", 0.0) for g in LEGS}
        q_out = np.array([self.cam_now.get(f"XC_{g}", {}).get("q_pcu", 0.0) for g in LEGS])
        return {g: q_in[g] - float(share[i] @ q_out) for i, g in enumerate(LEGS)}

    def _tick_snapshot(self, t: float) -> None:
        self.rows_ticks.append(self.tick_row(t))

    def tick_row(self, t: float) -> dict:
        """The sensor columns of a ticks_5s row at time t (also the input of a learned controller)."""
        row = dict(t=t)
        for g in LEGS:
            c, x = self.cam_now.get(f"QC_{g}", {}), self.cam_now.get(f"XC_{g}", {})
            row.update({f"q_veh_{g}": c.get("q_veh", 0), f"q_pcu_{g}": c.get("q_pcu", 0.0), f"q_m_{g}": c.get("q_m", 0.0),
                        f"moto_box_{g}": c.get("moto_box", 0), f"cam_speed_kmh_{g}": c.get("speed_kmh"),
                        f"dis_5s_{g}": self.counts_since("stopline_loop", g, t, 5.0),
                        f"dis_60s_{g}": self.counts_since("stopline_loop", g, t, 60.0),
                        f"arr_60s_{g}": self.counts_since("entry_gate", g, t, 60.0),
                        f"exit_q_pcu_{g}": x.get("q_pcu", 0.0),
                        f"ev_{g}": int(c.get("ev_d") is not None), f"ev_dist_m_{g}": c.get("ev_d"),
                        f"ev_eta_s_{g}": c.get("ev_eta_s"), f"ev_q_ahead_pcu_{g}": c.get("ev_q_ahead_pcu"),
                        f"ev_needs_signal_{g}": (int(c["ev_needs_signal"]) if "ev_needs_signal" in c else None)})
            groups = c.get("q_groups", {})
            if len(groups) > 1:                                    # an approach split over stages (TB Simatupang)
                row.update({f"q_veh_{g}_{k.replace('+', '')}": v for k, v in groups.items()})
        p = self.pressures()
        row.update({f"pressure_{g}": round(p[g], 2) for g in LEGS})
        return row

    def finish(self, t_end: float) -> None:
        self._flush_minute()
        for r in self.ev.values():                              # still travelling at the end
            r["t_arrive"] = None
            self.ev_done.append(r)
        self.ev = {}
