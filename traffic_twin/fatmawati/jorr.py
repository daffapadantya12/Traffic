"""JORR toll through-traffic for the live 3D view only (DECISIONS §13.5).

The twin's demand (demand.py) covers the movements through Perempatan Fatmawati and background trips around
it, so the JORR mainline on the flyover carried only a few vehicles.  The live view adds through-traffic on the
toll road.  It stays on the mainline, so it crosses the junction on the viaduct and never meets the
junction's traffic.  It is not in the released dataset, the evaluation or the calibration.

No counts are published for this section, so the flow is an assumption: about half the base freeway capacity
at the daily peak (MKJI 1997 / PKJI 2023 jalan bebas hambatan: 2,300 pcu/h per lane, 3 lanes each way), shaped
by the twin's daily profile.  No motorcycles: the JORR has no motorcycle lanes.
"""

from __future__ import annotations

import heapq
import math
import os

from .demand import DAILY_PROFILE, NET

# boundary edge where the mainline enters the network -> edge where it leaves (both ends of the 2.5 km box)
ENDS = {"westbound": ("354001305", "1188795207#1"), "eastbound": ("511963837", "1188226881")}
PEAK_VPH = 3000.0                        # per direction at the profile peak
MIX = {"car": 0.82, "bus": 0.03, "truck": 0.15}
CAR_TYPES = ("car_a1", "car_a3", "car_a5", "car_a7", "car_a9")     # the driver profiles of vtypes.add.xml


def mainline(net, src: str, dst: str) -> list:
    """Shortest route from `src` to `dst` over motorway edges only (no ramps, no streets)."""
    best, prev, pq = {src: 0.0}, {}, [(0.0, src)]
    while pq:
        d, eid = heapq.heappop(pq)
        if eid == dst:
            break
        if d > best[eid]:
            continue
        for o in net.getEdge(eid).getOutgoing():
            if o.getType() == "highway.motorway" and d + o.getLength() < best.get(o.getID(), math.inf):
                best[o.getID()], prev[o.getID()] = d + o.getLength(), eid
                heapq.heappush(pq, (best[o.getID()], o.getID()))
    if dst not in best:
        raise ValueError(f"no motorway route from {src} to {dst}")
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


def routes_xml(path: str, start_hour: float, duration_s: float, peak_vph: float = PEAK_VPH,
               profile=DAILY_PROFILE) -> dict:
    """Hourly through-flows in both directions for `duration_s` simulated seconds from `start_hour`."""
    import sumolib
    net = sumolib.net.readNet(NET, withInternal=False)
    car = MIX["car"] / len(CAR_TYPES)
    lines = ['<routes>',
             f'    <vTypeDistribution id="toll_mix" vTypes="{" ".join(CAR_TYPES)} bus truck" '
             f'probabilities="{" ".join(f"{car:.4f}" for _ in CAR_TYPES)} {MIX["bus"]} {MIX["truck"]}"/>']
    routes = {}
    for name, (src, dst) in ENDS.items():
        routes[name] = mainline(net, src, dst)
        lines.append(f'    <route id="jorr_{name}" edges="{" ".join(routes[name])}"/>')
    peak, t0, flows = max(profile), start_hour * 3600.0, []
    h = math.floor(start_hour)
    while h * 3600.0 < t0 + duration_s:                      # one flow per clock hour, Bernoulli departures each second
        begin, end = max(h * 3600.0, t0) - t0, min((h + 1) * 3600.0, t0 + duration_s) - t0
        p = peak_vph * profile[h % 24] / peak / 3600.0
        if end > begin and p > 0:
            for name in ENDS:
                flows.append((begin, f'    <flow id="jorr_{name}_{h}" type="toll_mix" route="jorr_{name}" begin="{begin:.0f}" '
                                     f'end="{end:.0f}" probability="{min(p, 1.0):.4f}" departLane="random" departSpeed="max"/>'))
        h += 1
    lines += [f for _, f in sorted(flows, key=lambda x: x[0])]
    lines.append('</routes>')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return dict(path=path, peak_vph=peak_vph, routes={k: len(v) for k, v in routes.items()},
                expected_vehicles=round(sum(peak_vph * profile[int((t0 + s) // 3600) % 24] / peak / 3600.0
                                            for s in range(0, int(duration_s), 60)) * 60 * len(ENDS)))
