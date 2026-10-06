"""traffic_twin v2.0 (DECISIONS §12): safe driving, lane changing, ambulance yielding, safety metrics."""

import math

import pandas as pd

from traffic_twin import Scenario, SimConfig, World
from traffic_twin.network import DIR_INDEX, shortest_route
from traffic_twin.safety import SafetyMonitor
from traffic_twin.vehicles import Vehicle, acc_accel, ambulance_params

SHORT = dict(warmup_s=60.0, main_s=240.0, cooldown_s=60.0)
CAR = dict(alpha=0.5, v0=13.0, a_max=1.5, b=2.0, T=1.2, l_veh=4.5, vehicle_type="car", politeness=0.25)


def _world(grid=2, **kw):
    cfg = SimConfig(grid_x=grid, grid_y=grid, warmup_s=0.0, main_s=300.0, cooldown_s=0.0, **kw)
    return World(cfg, Scenario(lambda_base_vph=0.0, controller="fixed_time"), seed=1)


def _place(w, origin, dest, edge, lane, s, v, params=None):
    edges, moves = shortest_route(w.net, origin, dest, w.cfg)
    veh = Vehicle(w.next_vid, dict(params or CAR), origin, dest, edges, moves, w.t)
    w.next_vid += 1
    w.n_spawn_requests += 1
    veh.edge, veh.lane, veh.s, veh.v, veh.v_prev, veh.enter_t, veh.status = edge, lane, s, v, v, w.t, "FREE"
    veh.lc_prev_lane = lane
    edge.lanes[lane].append(veh)
    edge.lanes[lane].sort(key=lambda x: (-x.s, x.id))
    w.active.append(veh)
    return veh


# ----------------------------------------------------------------- car following
def test_acc_brakes_for_a_sudden_red_without_over_reacting():
    """A STOPPING vehicle just outside the dilemma zone (d = v t_pr + v^2 / (2 b_stop)) stops behind the
    line at no more than about b_stop: no clipped emergency demand (v1.2's IDM demanded < -7 m/s^2)."""
    w = _world(grid=1)
    cfg = w.cfg
    w.plants[0].phase = DIR_INDEX["E"]                       # red for S
    edge = w.net.edges["S0_J00"]
    v = 13.9
    d = v * cfg.t_pr_s + v * v / (2 * cfg.b_stop)
    veh = _place(w, "S0", "N0", edge, 0, edge.stop_s - d, v)
    veh.latch = "STOPPING"
    worst = 0.0
    for _ in range(40):
        v_prev = veh.v
        w.step()
        worst = max(worst, (v_prev - veh.v) / cfg.dt)
    assert veh.v == 0.0 and veh.s <= edge.stop_s and worst <= cfg.b_stop + 0.5, worst
    assert w.n_brake_sat_virtual == 0 and w.invariant_counts()["red_violations"] == 0


def test_acc_answers_a_cut_in_gently():
    """IDM demands -5.6 m/s^2 for a car that cuts in 8 m ahead at equal speed; the ACC model about -2."""
    a = acc_accel(12.0, 13.9, 8.0, 12.0, 0.0, 1.5, 2.0, 1.2, 2.0)
    assert -2.5 <= a <= 0.0


def test_turning_vehicles_reach_the_arc_at_its_curvature_limit():
    """A vehicle at 13.9 m/s heading for a kerb-lane near-side turn slows on its approach and enters the
    arc at its curvature limit: lateral acceleration v^2 / r stays near a_lat_max = 3 m/s^2."""
    w = _world(grid=1)
    w.plants[0].phase = DIR_INDEX["S"]
    edge = w.net.edges["S0_J00"]
    veh = _place(w, "S0", "W0", edge, 0, 20.0, 13.9, dict(CAR, v0=15.0))
    assert veh.route_moves[0][1] == "NS"
    lat = 0.0
    for _ in range(60):
        w.step()
        if veh.conn is not None and veh.conn.is_turn:
            p = veh.conn.path
            if p.lead_in <= veh.s <= p.length - p.lead_out:
                lat = max(lat, veh.v * veh.v / p.radius)
    assert 0.0 < lat <= w.cfg.a_lat_max + 0.15, lat
    assert w.n_harsh_brake_events == 0


# ----------------------------------------------------------------- lane changing
def test_lane_change_needs_gaps_and_never_cuts_in_front_of_an_ambulance():
    w = _world()
    cfg = w.cfg
    w._shadows = {}
    edge = w.net.edges["J00_J10"]
    me = _place(w, "W0", "E0", edge, 0, 80.0, 10.0)
    assert w.lc_safe(me, edge, 1)[0]                          # empty target lane: safe
    blocker = _place(w, "W0", "E0", edge, 1, 81.0, 10.0)      # abreast in the target lane
    assert not w.lc_safe(me, edge, 1)[0]
    edge.lanes[1].remove(blocker)
    w.active.remove(blocker)
    amb = _place(w, "W0", "E0", edge, 1, 30.0, 12.0, ambulance_params(cfg))
    assert not w.lc_safe(me, edge, 1)[0]                      # ambulance 50 m behind in the target lane
    amb.s = 80.0 - cfg.lc_ev_clear_m - 5.0
    edge.lanes[1].sort(key=lambda x: (-x.s, x.id))
    assert w.lc_safe(me, edge, 1)[0]


def test_cars_move_aside_for_an_ambulance_only_where_there_is_room():
    """A car in the median lane with an ambulance close behind moves to the kerb lane when that lane
    has room; a car boxed in next to a standing queue stays put (and the attempt is counted)."""
    w = _world()
    edge = w.net.edges["J00_J10"]
    car = _place(w, "W0", "E0", edge, 1, 100.0, 8.0)
    amb = _place(w, "W0", "E0", edge, 1, 60.0, 12.0, ambulance_params(w.cfg))
    for _ in range(6):
        w.step()
    assert car.lane == 0 and w.n_lc_yield == 1
    w2 = _world()
    e2 = w2.net.edges["J00_J10"]
    car2 = _place(w2, "W0", "E0", e2, 1, 100.0, 0.0)
    for k in range(4):                                        # a standing queue beside it in the kerb lane
        _place(w2, "W0", "E0", e2, 0, 106.0 - k * 6.5, 0.0)
    _place(w2, "W0", "E0", e2, 1, 60.0, 0.0, ambulance_params(w2.cfg))
    w2.plants[1].phase = DIR_INDEX["N"]                       # keep the queue standing (red for W at J10)
    w2.change_lanes()
    assert car2.lane == 1 and w2.n_yield_blocked == 1


def test_vehicles_are_conserved_and_lists_stay_ordered_with_lane_changes():
    cfg = SimConfig(**SHORT)
    w = World(cfg, Scenario(lambda_base_vph=400.0, controller="max_pressure", lambda_ev_vph=20.0, ev_rate="default"), seed=7)
    for _ in range(int(cfg.episode_length_s / cfg.dt)):
        w.step()
        for e in w.net.edges.values():
            for lane in e.lanes:
                assert all(lane[i].s >= lane[i + 1].s for i in range(len(lane) - 1))
    assert w.n_lane_changes > 0
    assert w.invariant_counts()["conservation_error"] == 0 and w.invariant_counts()["collisions"] == 0


# ----------------------------------------------------------------- safety metrics
class _V:
    def __init__(self, vid, v):
        self.id, self.v, self.vehicle_type, self.edge, self.conn = vid, v, "car", None, None
        self.n_conflicts = self.n_near_misses = self.n_collisions = 0
        self.min_ttc, self.max_drac = math.inf, 0.0


def test_ttc_drac_classification():
    w = _world()
    mon = SafetyMonitor(w)
    cases = [((10.0, 10.0, 5.0), "none"), ((2.0, 1.5, 0.0), "conflict"), ((8.0, 12.0, 2.0), "serious"), ((6.0, 12.0, 2.0), "critical")]
    for k, ((gap, v, vl), expect) in enumerate(cases):
        a, b = _V(2 * k, v), _V(2 * k + 1, vl)
        mon.follow(a, b, gap, vl)
    mon.finish()
    rows = {r["veh_a"]: r for r in mon.rows}
    assert 0 not in rows                                       # TTC 2.0 s: no conflict
    assert not rows[2]["near_miss"] and math.isclose(rows[2]["max_drac_mps2"], 1.5 ** 2 / 4)      # slow creep
    assert rows[4]["near_miss"] and rows[4]["serious"] and not rows[4]["critical"]               # TTC 0.8, DRAC 6.25
    assert rows[6]["critical"] and mon.n_critical == 1                                           # DRAC 8.3 > b_emerg


def test_determinism_includes_conflicts_and_lane_changes():
    cfg = SimConfig(**SHORT)
    sc = Scenario(lambda_base_vph=380.0, controller="max_pressure", lambda_ev_vph=20.0, ev_rate="default")
    a, b = World(cfg, sc, seed=5).run(), World(cfg, sc, seed=5).run()
    pd.testing.assert_frame_equal(pd.DataFrame(a.vehicle_rows()), pd.DataFrame(b.vehicle_rows()))
    assert a.conflict_rows() == b.conflict_rows() and a.n_lane_changes == b.n_lane_changes


def test_saturation_flow_is_measured_with_lane_changes_off():
    """Saturation flow is per lane (HCM): with lane changes on, the test queue would spill into the
    empty neighbouring lane and read about twice the real value."""
    from traffic_twin.calibrate import measure_saturation_flow
    m = measure_saturation_flow(SimConfig())
    assert 1600.0 <= m["sat_flow_vphpl"] <= 1900.0, m["sat_flow_vphpl"]
