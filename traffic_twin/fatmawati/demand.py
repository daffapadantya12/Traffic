"""Vehicle types and travel demand for the Fatmawati twin (DECISIONS §13.2).

Traffic-engineering idea: Jakarta traffic is motorcycle-dominated.  Classified counts on the north leg of
Perempatan Fatmawati give 77-82% motorcycles, under 1% buses and trucks (Yusuf & Isradi 2025, 2024 weekday,
16 hours), so the mix is 80% motorcycles, 19% cars, 0.5% buses, 0.5% trucks; 70% (Jakarta-wide on-road
counts, Dishub DKI 2019) is the sensitivity level.  Motorcycles are
small (2.0 x 0.8 m), quicker off the line (a 2.0 m/s^2, T 0.8 s) and filter through standing queues
to the motorcycle box in front of the cars (sublane model, SUMO stopOffset exceptions).  Drivers of
every class carry a latent aggressiveness alpha that shifts desired speed, acceleration, braking
and headway together (the same idea as the grid study).  Motorcycles may not use toll roads.

Demand at the junction is the 13 movements counted by CCTV in a weekday peak hour (Sumabrata, Haqueline &
Mauramdha 2026): each enters at its leg's upstream end and leaves at the exit leg's far end, scaled per
approach by the calibration (calibrate.py: volumes that reproduce the observed queues) and by the hour of
the daily profile relative to its peak.  Background trips between fringe points (where the 2.5 km cut
crosses a road), weighted by road class and lanes, keep the rest of the network alive but never come
within 200 m of Fatmawati.  Ambulances drive to RSUP Fatmawati, the national hospital 325 m south of
the junction.
"""

from __future__ import annotations

import math
import os

import numpy as np

from ..config import DAILY_PROFILE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data", "fatmawati")
NET = os.path.join(DATA, "net", "fatmawati.net.xml")

MIX = {"motorcycle": 0.80, "car": 0.19, "bus": 0.005, "truck": 0.005}
ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)              # driver profiles per class (latent aggressiveness quantiles)
# per lane; toll roads (motorway) mostly pass overhead on the viaduct, so they get a small share of the trips
CLASS_WEIGHT = {"motorway": 0.6, "motorway_link": 0.4, "trunk": 6.0, "trunk_link": 2.0, "primary": 4.0,
                "primary_link": 1.5, "secondary": 2.0, "secondary_link": 1.0, "tertiary": 1.0, "tertiary_link": 0.5}
HOSPITAL_LONLAT = (106.79627, -6.29525)          # RSUP Fatmawati (OSM amenity=hospital, emergency=yes)
# observed weekday peak-hour movement volumes, veh/h, all classes, (entry leg, exit leg); same leg = U-turn
# (Sumabrata, Haqueline & Mauramdha 2026, CCTV counts); there is no RA Kartini right turn
MOVEMENTS = {("N", "S"): 3135, ("N", "W"): 1254, ("N", "E"): 1067, ("S", "N"): 1495, ("S", "E"): 2174, ("S", "W"): 963,
             ("E", "W"): 5607, ("E", "N"): 858, ("E", "S"): 1226, ("E", "E"): 1764,
             ("W", "E"): 3271, ("W", "N"): 1215, ("W", "W"): 1848}
LEG_SOURCE = {"N": "778513710#0", "E": "698048527#0", "S": "648964736#6", "W": "692766250#0"}   # upstream end of each leg
LEG_SINK = {"N": "1304381185#0", "E": "31806415#0", "S": "1069005555#0", "W": "827224286#8"}    # far end of each exit
CALIBRATION = os.path.join(DATA, "calibration.json")
LEVEL_FACTOR = {"low": 0.75, "design": 1.0, "high": 1.15}      # x the calibrated peak
BASE_INFLOW_VPH = {k: 3000.0 * f for k, f in LEVEL_FACTOR.items()}   # background trips at the profile peak
CORDON_M = 200.0                                               # background trips stay this far from the junction
EV_PER_HOUR = 6.0
EV_LEGS = ("N", "E", "W")                                      # ambulance origins with a route to RSUP Fatmawati


def load_calibration() -> dict:
    """Per-approach volume scale fitted by calibrate.py (1.0 = the observed counts); 0.3 before calibration."""
    import json
    try:
        with open(CALIBRATION) as f:
            return {k: float(v) for k, v in json.load(f)["scale"].items()}
    except (OSError, KeyError, ValueError):
        return {g: 0.3 for g in "NESW"}


def _clip(v, lo, hi):
    return min(max(v, lo), hi)


def mix_with_moto_share(share: float) -> dict:
    """The vehicle mix with another motorcycle share; buses and trucks keep theirs, cars take the rest."""
    return {"motorcycle": share, "car": round(1.0 - share - MIX["bus"] - MIX["truck"], 4), "bus": MIX["bus"], "truck": MIX["truck"]}


def vtypes_xml(path: str, mix: dict | None = None) -> str:
    """Write the vType distribution (class x driver profile) and the ambulance type (mix: default MIX)."""
    MIX_ = mix or MIX
    lines = ['<!-- traffic_twin v3 vehicle types (DECISIONS §13.2). Motorcycle values are modelling assumptions within the ranges',
             '     reported for Southeast Asian mixed traffic (sources in DECISIONS §13.2): length 2.0 m, width 0.8 m, v0 ~ 11 m/s,',
             '     a ~ 2.0 m/s^2, T ~ 0.8 s; cars as the grid study (IDM-family drivers with latent aggressiveness). Motorcycles',
             '     filter (latAlignment arbitrary) and may use the motorcycle box (stopOffset exceptions); everyone else keeps to',
             '     the lane centre.  Motorcycle gaps 0.5 m (standstill and lateral): the VISSIM values of this junction',
             '     (Sumabrata et al. 2026: 0.2 m standstill, 0.3 m lateral) gave 15x more lane collisions in SUMO\'s sublane',
             '     model at calibrated demand (DECISIONS 13.2). -->',
             '<additional>', '    <vTypeDistribution id="mix">']
    n = len(ALPHAS)
    for a in ALPHAS:
        k = a - 0.5
        lines.append(
            f'        <vType id="car_a{int(a*10)}" vClass="passenger" probability="{MIX_["car"]/n:.4f}" length="4.5" width="1.8" minGap="2.0" '
            f'accel="{1.5*(1+0.5*k):.3f}" decel="{2.0*(1+0.3*k):.3f}" emergencyDecel="7.0" tau="{1.2*(1-0.4*k):.3f}" '
            f'speedFactor="normc({1+0.2*k:.3f},0.08,0.85,1.15)" carFollowModel="EIDM" latAlignment="center" minGapLat="0.6" '
            f'lcStrategic="2.0" lcCooperative="0.8" color="0.36,0.55,0.95"/>')
    for a in ALPHAS:
        k = a - 0.5
        lines.append(
            f'        <vType id="moto_a{int(a*10)}" vClass="motorcycle" probability="{MIX_["motorcycle"]/n:.4f}" length="2.0" width="0.8" '
            f'minGap="0.5" accel="{2.0*(1+0.4*k):.3f}" decel="3.0" emergencyDecel="6.5" tau="{0.8*(1-0.4*k):.3f}" '
            f'speedFactor="normc({0.80*(1+0.2*k):.3f},0.12,0.55,1.05)" carFollowModel="EIDM" latAlignment="arbitrary" minGapLat="0.5" '
            f'maxSpeedLat="1.5" lcSublane="2.0" lcPushy="0" lcAssertive="1.0" lcStrategic="2.0" color="0.25,0.75,0.35"/>')
    lines.append(f'        <vType id="bus" vClass="bus" probability="{MIX_["bus"]}" length="10.0" width="2.5" minGap="2.5" accel="1.0" decel="2.0" '
                 'emergencyDecel="6.0" tau="1.5" speedFactor="normc(0.9,0.05,0.8,1.0)" carFollowModel="EIDM" latAlignment="center" '
                 'minGapLat="0.6" color="0.85,0.5,0.15"/>')
    lines.append(f'        <vType id="truck" vClass="truck" probability="{MIX_["truck"]}" length="8.0" width="2.4" minGap="2.5" accel="0.9" decel="2.0" '
                 'emergencyDecel="6.0" tau="1.5" speedFactor="normc(0.85,0.05,0.75,0.95)" carFollowModel="EIDM" latAlignment="center" '
                 'minGapLat="0.6" color="0.6,0.6,0.6"/>')
    lines.append('    </vTypeDistribution>')
    lines.append('    <vType id="ambulance" vClass="emergency" length="6.0" width="2.1" minGap="2.0" accel="2.5" decel="3.0" emergencyDecel="7.0" '
                 'tau="0.9" speedFactor="1.25" carFollowModel="EIDM" latAlignment="center" minGapLat="0.6" guiShape="emergency" '
                 'color="1.0,0.85,0.0"/>')
    lines.append('</additional>')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def fringe(net):
    """(sources, sinks): edges whose start (end) node has no other incoming (outgoing) edge than its own reverse."""
    src, snk = [], []
    for e in net.getEdges():
        if e.getFunction() != "":
            continue
        fn, tn = e.getFromNode(), e.getToNode()
        if not [i for i in fn.getIncoming() if i.getFromNode() is not tn]:
            src.append(e)
        if not [o for o in tn.getOutgoing() if o.getToNode() is not fn]:
            snk.append(e)
    return src, snk


def hospital_edge(net):
    """(edge, arrival position) on Jl. RS Fatmawati nearest RSUP Fatmawati.  The campus lies east of the road,
    i.e. beside the southbound carriageway under left-hand traffic."""
    import sumolib
    x, y = net.convertLonLat2XY(*HOSPITAL_LONLAT)
    cands = [(d, e) for e, d in net.getNeighboringEdges(x, y, 250.0)
             if e.getName() == "Jalan RS Fatmawati" and e.allows("emergency")]
    d, e = min(cands, key=lambda t: t[0])
    pos = sumolib.geomhelper.polygonOffsetWithMinimumDistanceToPoint((x, y), e.getShape())
    return e, max(1.0, min(pos, e.getLength() - 1.0))


def reachable(net, src, vclasses=("passenger", "motorcycle")) -> dict:
    """source edge id -> set of edge ids every class in `vclasses` can reach from it (U-turns exist only
    where OSM maps them, so not every fringe pair is connected)."""
    out = {}
    for e in src:
        common = None
        for vc in vclasses:
            seen, todo = {e.getID()}, [e]
            while todo:
                cur = todo.pop()
                for nxt in cur.getAllowedOutgoing(vc):
                    if nxt.getID() not in seen and nxt.allows(vc):
                        seen.add(nxt.getID())
                        todo.append(nxt)
            common = seen if common is None else common & seen
        out[e.getID()] = common
    return out


def class_weight(e) -> float:
    return CLASS_WEIGHT.get(e.getType().replace("highway.", ""), 1.0) * e.getLaneNumber()


_BACKGROUND_PAIRS = {}


_GATES = None


def gate_edges() -> tuple:
    """({leg: edge of its entry gate}, {leg: edge of its exit gate}) from the sensor layout."""
    global _GATES
    if _GATES is None:
        from .sensors import build_layout
        loops = build_layout()["loops"]
        _GATES = tuple({s["leg"]: s["lane_id"].rsplit("_", 1)[0] for s in loops if s["kind"] == kind}
                       for kind in ("entry_gate", "exit_gate"))
    return _GATES


def background_pairs(net, src, snk) -> dict:
    """source id -> {sink id: route edges} for routes that stay CORDON_M away from the junction (written out as
    explicit routes, so SUMO's router cannot send them through it)."""
    key = (NET, os.path.getmtime(NET))
    if key in _BACKGROUND_PAIRS:
        return _BACKGROUND_PAIRS[key]
    from .build_net import JUNCTION_LAT, JUNCTION_LON
    x0, y0 = net.convertLonLat2XY(JUNCTION_LON, JUNCTION_LAT)

    def near(e):
        return any(math.hypot(x - x0, y - y0) < CORDON_M for x, y in e.getShape())
    blocked = {e.getID() for e in net.getEdges() if near(e)}
    out = {}
    for o in src:
        ok = {}
        for d in snk:
            if d.getFromNode() is o.getToNode() or d.getToNode() is o.getFromNode():
                continue
            path, _ = net.getShortestPath(o, d, vClass="passenger")
            if path and not blocked & {e.getID() for e in path}:
                ok[d.getID()] = " ".join(e.getID() for e in path)
        out[o.getID()] = ok
    _BACKGROUND_PAIRS[key] = out
    return out


def routes_xml(path: str, level: str = "design", seed: int = 0, start_hour: float = 0.0, duration_s: float = 3600.0,
               profile=DAILY_PROFILE, scale: dict | None = None, peak: bool = False, ev_per_hour: float = EV_PER_HOUR) -> dict:
    """Trips for `duration_s` simulated seconds from `start_hour`: the observed Fatmawati movements x the calibrated
    approach scale x the level x the hour's share of the profile peak (`peak`: always 1, for calibration);
    background trips that avoid the junction; ambulances to RSUP Fatmawati.  Poisson arrivals (thinned in time)."""
    import sumolib
    rng = np.random.default_rng(seed)
    net = sumolib.net.readNet(NET, withInternal=False)
    scale = scale or load_calibration()
    lf = LEVEL_FACTOR[level]
    peak_value = max(profile)

    def tod(t: float) -> float:
        return 1.0 if peak else profile[int((start_hour * 3600.0 + t) // 3600.0) % 24] / peak_value

    entry, exit_ = gate_edges()
    trips = []                                                   # (t, from, to, type, via or route)
    for (a, b), vph in MOVEMENTS.items():                       # the junction's movements, through their gates
        rate = vph * scale[a] * lf / 3600.0
        t = 0.0
        while rate > 0:
            t += rng.exponential(1.0 / rate)
            if t >= duration_s:
                break
            if rng.random() <= tod(t):
                trips.append((t, LEG_SOURCE[a], LEG_SINK[b], "mix", ("via", f"{entry[a]} {exit_[b]}")))
    n_junction = len(trips)
    src, snk = fringe(net)                                       # background, away from the junction
    pairs = background_pairs(net, src, snk)
    src = [e for e in src if pairs[e.getID()]]
    w_src = np.array([class_weight(e) for e in src])
    p_src = w_src / w_src.sum()
    snk_by_id = {e.getID(): e for e in snk}
    toll = {e.getID(): e.getType().startswith("highway.motorway") for e in list(src) + list(snk)}
    reach_mix = reachable(net, src)
    t, rate = 0.0, BASE_INFLOW_VPH[level] / 3600.0
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= duration_s:
            break
        if rng.random() > tod(t):
            continue
        o = src[rng.choice(len(src), p=p_src)]
        ox, oy = o.getFromNode().getCoord()
        cands = sorted(pairs[o.getID()])
        w = np.array([class_weight(snk_by_id[d]) * (1.0 + math.hypot(snk_by_id[d].getToNode().getCoord()[0] - ox,
                                                                       snk_by_id[d].getToNode().getCoord()[1] - oy) / 1000.0)
                      for d in cands])
        d = cands[rng.choice(len(cands), p=w / w.sum())]
        moto_ok = not (toll[o.getID()] or toll.get(d, False)) and d in reach_mix[o.getID()]
        trips.append((t, o.getID(), d, "mix" if moto_ok else "car_only", ("route", pairs[o.getID()][d])))
    hosp, hosp_pos = hospital_edge(net)                          # ambulances
    w_ev = np.array([sum(v for (a, _), v in MOVEMENTS.items() if a == g) for g in EV_LEGS], dtype=float)
    t_ev, n_ev = 0.0, 0
    while ev_per_hour > 0:
        t_ev += rng.exponential(3600.0 / ev_per_hour)
        if t_ev >= duration_s:
            break
        leg = EV_LEGS[rng.choice(len(EV_LEGS), p=w_ev / w_ev.sum())]
        trips.append((t_ev, LEG_SOURCE[leg], hosp.getID(), "ambulance", ("via", entry[leg])))
        n_ev += 1
    trips.sort()
    lines = ['<routes>', '    <vTypeDistribution id="car_only" vTypes="car_a1 car_a3 car_a5 car_a7 car_a9 bus truck" '
             'probabilities="0.198 0.198 0.198 0.198 0.198 0.005 0.005"/>']
    for i, (t, o, d, typ, (kind, edges)) in enumerate(trips):
        prefix = "ev" if typ == "ambulance" else "v"
        common = f'id="{prefix}{i}" type="{typ}" depart="{t:.2f}" departLane="best" departSpeed="max"'
        if kind == "route":
            lines.append(f'    <vehicle {common}><route edges="{edges}"/></vehicle>')
        else:
            lines.append(f'    <trip {common} from="{o}" to="{d}" via="{edges}"'
                         + (f' arrivalPos="{hosp_pos:.1f}"' if typ == "ambulance" else "") + '/>')
    lines.append('</routes>')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return dict(path=path, n_vehicles=len(trips) - n_ev, n_junction=n_junction, n_ambulances=n_ev,
                hospital_edge=hosp.getID(), hospital_pos=round(hosp_pos, 1), scale=scale, level=level)


if __name__ == "__main__":
    import json
    vtypes_xml(os.path.join(DATA, "demand", "vtypes.add.xml"))
    print(json.dumps(routes_xml(os.path.join(DATA, "demand", "design_0800.rou.xml"), "design", start_hour=8.0, duration_s=3600.0, seed=1)))
