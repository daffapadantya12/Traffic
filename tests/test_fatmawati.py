"""traffic_twin v3 (DECISIONS §13): the Perempatan Fatmawati network, its demand and short-run safety rules.

Needs SUMO (pip: eclipse-sumo, libsumo, sumolib) and the built network
(python3 -m traffic_twin.fatmawati.build_net); skipped otherwise.
"""

import collections
import os
import xml.etree.ElementTree as ET

import pytest

sumolib = pytest.importorskip("sumolib")
pytest.importorskip("libsumo")

from traffic_twin.fatmawati.build_net import APPROACHES, RHK_DEPTH_M, TLS_ID  # noqa: E402
from traffic_twin.fatmawati.demand import DATA, NET, reachable, routes_xml, vtypes_xml  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(NET), reason="Fatmawati network not built")


@pytest.fixture(scope="module")
def net():
    return sumolib.net.readNet(NET, withPrograms=True)


@pytest.fixture(scope="module")
def routes(tmp_path_factory):
    vtypes_xml(os.path.join(DATA, "demand", "vtypes.add.xml"))
    path = str(tmp_path_factory.mktemp("fatmawati") / "trips.rou.xml")
    routes_xml(path, "design", start_hour=7.0, duration_s=900.0, seed=3)
    return path


def test_dishub_stage_program(net):
    from traffic_twin.fatmawati.build_net import stage_link_sets
    tl = net.getTLS(TLS_ID)
    phases = tl.getPrograms()["dishub"].getPhases()
    assert len(phases) == 4 + 12 * 2             # stage greens + amber and all-red for every ordered stage pair
    assert [round(p.duration) for p in phases[:12]] == [80, 2, 2, 12, 2, 2, 78, 2, 2, 45, 2, 2]   # the Dishub cycle
    assert [p.name for p in phases[:3]] == ["S1 green", "S1>S2 amber", "S1>S2 all-red"]
    green, median, n = stage_link_sets(NET)        # raises if a link is green in no stage
    assert n == len({idx for _, _, idx in tl.getConnections()})
    assert green["S1"] & green["S2"]               # TB Simatupang through keeps green from S1 into S2


def test_motorcycle_boxes_on_multilane_approaches():
    offsets = {}
    for _, el in ET.iterparse(NET):
        if el.tag == "edge" and el.get("id") in APPROACHES.values():
            so = el.find("stopOffset")
            offsets[el.get("id")] = None if so is None else (float(so.get("value")), so.get("exceptions"))
    assert set(offsets) == set(APPROACHES.values())
    for edge, so in offsets.items():
        assert so == (RHK_DEPTH_M, "motorcycle"), edge


def test_lanes_fit_a_car_and_uturns_only_where_mapped(net):
    widths = [ln.getWidth() for e in net.getEdges() for ln in e.getLanes()]
    assert min(widths) >= 3.0
    uturns = sum(1 for e in net.getEdges() for conns in e.getOutgoing().values() for c in conns if c.getDirection() == "t")
    uturns += sum(1 for e in net.getEdges() for conns in e.getOutgoing().values() for c in conns if c.getDirection() == "T")
    assert uturns < 20                            # netconvert's default would add one at every junction (~200)


def test_trips_only_between_connected_ends(net, routes):
    trips = [el for _, el in ET.iterparse(routes) if el.tag == "trip"]            # junction movements and ambulances
    assert len(trips) > 100
    origins = [net.getEdge(o) for o in sorted({t.get("from") for t in trips})]
    for _, v in ET.iterparse(routes):                                              # background: explicit routes avoid the junction
        if v.tag == "route":
            edges = v.get("edges").split()
            assert not set(edges) & set(APPROACHES.values())
            assert all(b.getID() in {o.getID() for o in net.getEdge(a).getOutgoing()} for a, b in
                       zip(edges, [net.getEdge(x) for x in edges[1:]]))
    reach = {"mix": reachable(net, origins), "car_only": reachable(net, origins, ("passenger",)),
             "ambulance": reachable(net, origins, ("emergency",))}
    for t in trips:
        assert t.get("to") in reach[t.get("type")][t.get("from")], t.get("id")


def test_junction_demand_follows_the_observed_movements(tmp_path):
    from traffic_twin.fatmawati.demand import LEG_SINK, LEG_SOURCE, MOVEMENTS
    path = str(tmp_path / "peak.rou.xml")
    routes_xml(path, "design", seed=5, duration_s=3600.0, peak=True, scale={g: 0.25 for g in "NESW"}, ev_per_hour=0)
    leg_of_src = {v: k for k, v in LEG_SOURCE.items()}
    leg_of_snk = {v: k for k, v in LEG_SINK.items()}
    counts = collections.Counter((leg_of_src[t.get("from")], leg_of_snk[t.get("to")]) for _, t in ET.iterparse(path)
                                 if t.tag == "trip" and t.get("type") == "mix")
    for m, vph in MOVEMENTS.items():                # Poisson: within 4 standard deviations of 0.25 x observed
        mean = 0.25 * vph
        assert abs(counts[m] - mean) < 4 * mean ** 0.5 + 1, (m, counts[m], mean)


def test_lateral_mode_is_checked_before_sumo_starts(routes):
    from traffic_twin.fatmawati.sumo_world import FatmawatiWorld
    with pytest.raises(ValueError):
        FatmawatiWorld(routes, lateral_mode="band")


def test_short_run_no_lane_collisions_and_cars_respect_the_box(routes):
    """10 simulated minutes: no collision on a lane; motorcycles use the box at red, other vehicles never
    cross the car stop line while their signal is red."""
    from traffic_twin.fatmawati.sumo_world import FatmawatiWorld
    w = FatmawatiWorld(routes, start_hour=7.0, seed=3, controller="longest_queue", sensors=True)
    ts = w.ts
    try:
        lane_links = collections.defaultdict(list)
        for i, links in enumerate(ts.trafficlight.getControlledLinks(TLS_ID)):
            for a, _, _ in links:
                lane_links[a].append(i)
        lanes = [f"{e}_{k}" for e in APPROACHES.values() for k in range(ts.edge.getLaneNumber(e))]
        behind, red_entries, moto_in_box = {}, 0, 0
        while w.t < 600.0:
            w.step()
            state = ts.trafficlight.getRedYellowGreenState(TLS_ID)
            for ln in lanes:
                red = all(state[i] in "rR" for i in lane_links[ln])
                end = ts.lane.getLength(ln)
                for v in ts.lane.getLastStepVehicleIDs(ln):
                    d = end - ts.vehicle.getLanePosition(v)
                    moto = ts.vehicle.getVehicleClass(v) == "motorcycle"
                    if moto:
                        moto_in_box += red and d < RHK_DEPTH_M
                        continue
                    was_behind = behind.get(v, True)
                    now_behind = d > RHK_DEPTH_M - 0.3
                    red_entries += red and was_behind and not now_behind
                    behind[v] = now_behind
        assert w.n_lane_collisions == 0
        assert red_entries == 0
        assert moto_in_box > 0
        assert w.motos                            # the junction rule tracks motorcycles
        # the viewer draws every vehicle on the road surface it draws: toll lanes at their height, streets flat
        from traffic_twin.fatmawati.geometry3d import raised_z
        from traffic_twin.fatmawati.sumo_world import display_raised
        raised_lanes, _ = display_raised()
        drawn = {row[0]: row[3] for row in w.frame()["vehicles"]}
        num = lambda v: int(v[2:]) + 10_000_000 if v.startswith("ev") else int(v[1:]) if v[1:].isdigit() else hash(v) % 10_000_000
        on_toll = 0
        for v in ts.vehicle.getIDList():
            lane = ts.vehicle.getLaneID(v)
            if lane.startswith(":"):
                continue
            want = raised_z(raised_lanes[lane], ts.vehicle.getLanePosition(v)) if lane in raised_lanes else 0.0
            assert drawn[num(v)] == pytest.approx(want, abs=0.02)
            on_toll += want > 5.0
        assert on_toll > 0                        # some traffic is up on the flyover
    finally:
        w.close()


def test_sensor_layout_matches_the_design(net):
    from traffic_twin.fatmawati.sensors import CAMERA_M, GATE_R, build_layout
    lay = build_layout()
    kinds = collections.Counter(s["kind"] for s in lay["loops"])
    assert kinds["stopline_loop"] == sum(net.getEdge(e).getLaneNumber() for e in APPROACHES.values())
    assert kinds["entry_gate"] > 0 and kinds["exit_gate"] > 0
    for s in lay["loops"]:
        if s["kind"] in ("entry_gate", "exit_gate"):
            assert abs(s["dist_centre_m"] - GATE_R[s["leg"]]) < 5.0, s["sensor_id"]
    for cam in lay["cameras"]:
        assert all(0.0 <= a < b for _, a, b, _ in cam["segments"])
        if cam["kind"] == "queue_camera":                    # reaches exactly 300 m up from the stop line
            reach = max(off + net.getLane(lane).getLength() - a for lane, a, _, off in cam["segments"])
            assert abs(reach - CAMERA_M) < 0.5, cam["sensor_id"]


def test_generator_writes_sensor_tables_only(tmp_path):
    """Tiny end-to-end release: every column is documented and no simulator internals are written."""
    import pandas as pd
    from traffic_twin.fatmawati.generate import generate
    out = str(tmp_path / "release")
    counts = generate(out, "smoke", workers=1)
    assert counts["episodes"] == 1 and counts["ticks_5s"] > 0 and counts["loops_1min"] > 0
    dictionary = pd.read_csv(os.path.join(out, "data_dictionary.csv"))
    documented = set(zip(dictionary["table"], dictionary["column"]))
    forbidden = ("vehicle_id", "veh_id", "lane_pos", "alpha", "tau", "accel", "lat_offset", "route")
    for table in counts:
        cols = pd.read_csv(os.path.join(out, f"{table}.csv"), nrows=1).columns
        for col in cols:
            assert (table, col) in documented, (table, col)
            assert not any(col == f or col.startswith(f + "_") for f in forbidden), (table, col)
    ticks = pd.read_csv(os.path.join(out, "ticks_5s.csv"))
    assert (ticks["t"].diff().dropna() == 5.0).all()
    qc = pd.read_csv(os.path.join(out, "simulation_qc.csv"))
    assert qc["lane_collisions"].sum() == 0 and qc["unmatched_exits"].sum() == 0
    sums = open(os.path.join(out, "SHA256SUMS.txt")).read()
    assert "ticks_5s.csv" in sums and "DATACARD.md" in sums


def test_fixed_controller_runs_the_dishub_greens(routes):
    from traffic_twin.fatmawati.sumo_world import FatmawatiWorld
    w = FatmawatiWorld(routes, start_hour=7.0, seed=3, controller="fixed")
    try:
        while w.t < 600.0:
            w.step()
        log = w.signal_log
    finally:
        w.close()
    greens = [(b["t"] - a["t"], a["stage"]) for a, b in zip(log, log[1:]) if a["state"] == "green"]
    assert {st: d for d, st in greens[1:5]} == {"S1": 80.0, "S2": 12.0, "S3": 78.0, "S4": 45.0}
    amber = [b["t"] - a["t"] for a, b in zip(log, log[1:]) if a["state"] == "amber"]
    assert set(amber) == {2.0}


def test_toll_flyover_spans_the_junction_on_piers_clear_of_the_streets(tmp_path):
    """The JORR toll road crosses above the junction on a deck with room for a bus below; streets stay on the
    ground (also where SUMO lifts a river bridge); no pier stands in a street lane."""
    import json
    import math
    from traffic_twin.fatmawati.geometry3d import DECK_DEPTH_M, PIER_HALF_M, _seg_dist, build
    g = json.load(open(build(str(tmp_path / "geometry.json"), buildings_path=str(tmp_path / "none.json"))))
    fly = g["flyover"]
    over = {d["edge"]: [p for p in d["p"] if math.hypot(p[0], p[1]) < 80] for d in fly["decks"]}
    over = {e: pts for e, pts in over.items() if pts}
    assert len(over) == 2                                         # both carriageways
    assert min(p[2] for pts in over.values() for p in pts) - DECK_DEPTH_M >= 4.0
    streets = [ln for ln in g["lanes"] if not ln["k"].startswith("motorway")]
    assert all(q[2] == 0.0 for ln in streets for q in ln["p"])
    near = [pr for pr in fly["piers"] if math.hypot(pr[0], pr[1]) < 400]
    assert len(near) >= 20
    for x, y, *_ in near:
        for ln in streets:
            pts = ln["p"]
            gap = min(_seg_dist(x, y, a[0], a[1], b[0], b[1]) for a, b in zip(pts, pts[1:]))
            assert gap >= ln["w"] / 2 + PIER_HALF_M, (x, y, ln["id"])
