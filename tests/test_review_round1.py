"""Regression tests for the round-1 review findings (DECISIONS §11 amendments).

safety-physics-01  a vehicle leaving a connector is integrated exactly once per step
safety-physics-02  a COMMITTED vehicle still upstream of the box holds the clearance gate;
                   a stalled commit is re-latched STOPPING; crossing under another green counts
safety-physics-03  plant starvation guard (forced = 4) and the max_approach_red_s KPI
safety-physics-04  (v2.0: lane discipline) two abreast through vehicles keep their lanes through the box
traffic-validation-05  per-junction gridlock detector is reachable
data-schema-001    data_dictionary.csv covers every column of every CSV
"""

import math

import pandas as pd
import pytest

from traffic_twin import Scenario, SimConfig, World
from traffic_twin.network import DIR_INDEX, shortest_route
from traffic_twin.signals import ALLRED, FORCED_STARVATION, GREEN
from traffic_twin.vehicles import COMMITTED, STOPPING, Vehicle, idm_accel, sample_params

SHORT = dict(warmup_s=60.0, main_s=240.0, cooldown_s=60.0)


def _empty_world(grid=1, **kw):
    cfg = SimConfig(grid_x=grid, grid_y=grid, warmup_s=0.0, main_s=200.0, cooldown_s=0.0, **kw)
    return World(cfg, Scenario(lambda_base_vph=0.0, controller="longest_queue"), seed=1)


def _place(world, origin, destination, edge, lane, s, v, params=None):
    """Put a vehicle with the given route on `edge` at position s with speed v."""
    cfg = world.cfg
    edges, moves = shortest_route(world.net, origin, destination, cfg)
    params = params or sample_params(world.rng_vehicles, cfg)
    veh = Vehicle(world.next_vid, params, origin, destination, edges, moves, world.t)
    world.next_vid += 1
    world.n_spawn_requests += 1
    veh.edge, veh.lane, veh.s, veh.v, veh.enter_t, veh.status = edge, lane, s, v, world.t, "FREE"
    edge.lanes[lane].append(veh)
    edge.lanes[lane].sort(key=lambda x: -x.s)
    world.active.append(veh)
    return veh


def _route_pos(world, veh):
    pos = sum(world.net.edges[veh.route_edges[i]].length + world.cfg.w_box for i in range(veh.edge_idx))
    if veh.edge is not None:
        return pos + veh.s
    return pos + world.net.edges[veh.route_edges[veh.edge_idx]].length + veh.s


# ----------------------------------------------------------------- safety-physics-01
def test_box_exit_is_integrated_once_per_step():
    """A lone vehicle at constant speed advances exactly v*dt per step, also in the step
    in which it leaves the connector (it used to be stepped twice: v*dt extra)."""
    w = _empty_world()
    for p in w.plants:
        p.phase = DIR_INDEX["S"]                      # green for the S approach
    edge = w.net.edges["S0_J00"]
    params = dict(alpha=0.5, v0=10.0, a_max=1.5, b=2.0, T=1.2, l_veh=4.5, vehicle_type="car")
    veh = _place(w, "S0", "N0", edge, 0, edge.length - 20.0, 10.0, params)
    seen_exit = False
    for _ in range(14):
        before, on_conn = _route_pos(w, veh), veh.conn is not None
        w.step()
        disp = _route_pos(w, veh) - before
        assert math.isclose(disp, 10.0 * w.cfg.dt, abs_tol=1e-9), disp
        if on_conn and veh.conn is None:
            seen_exit = True
    assert seen_exit and veh.edge.id == "J00_N0"


# ----------------------------------------------------------------- safety-physics-02
def test_committed_vehicle_upstream_holds_the_clearance_gate():
    """A COMMITTED vehicle that is still crawling toward the line keeps the all-red going
    (box empty is not enough); it crosses during ALLRED, never under another green."""
    w = _empty_world()
    plant = w.plants[0]
    edge = w.net.edges["N0_J00"]
    # crawling at 4 m/s it needs ~8 s to clear the 26 m box: well inside allred_min + allred_stall_s, so the gate
    # (not the v2.0 stall re-pick, which would serve the crawler's own approach) is what holds the all-red here
    params = dict(alpha=0.5, v0=4.0, a_max=0.8, b=2.0, T=1.2, l_veh=4.5, vehicle_type="car")
    veh = _place(w, "N0", "S0", edge, 0, edge.stop_s - 3.0, 4.0, params)      # 3 m before the line at 4 m/s
    veh.latch = COMMITTED
    plant.state, plant.phase, plant.pending_phase = ALLRED, DIR_INDEX["N"], DIR_INDEX["E"]
    plant.allred_start_t = w.t - w.cfg.allred_min                              # all-red already 4 s old
    onset_t = None
    for _ in range(40):
        w.step()
        if onset_t is None and plant.state == GREEN:
            onset_t = w.t
            assert veh.edge is not None and veh.edge.id != edge.id, "green came before the vehicle left the box"
    assert onset_t is not None and plant.phase == DIR_INDEX["E"]
    assert math.isclose(veh.last_stop_cross_t, veh.last_stop_cross_t) and veh.last_stop_cross_t < onset_t
    assert w.invariant_counts()["red_violations"] == 0
    assert w.invariant_counts()["committed_at_onset_violations"] == 0
    assert plant.allred_extensions[-1] > 1.0                                   # the gate really extended the all-red


def test_stalled_commit_is_relatched_stopping_and_waits():
    """A COMMITTED vehicle that came to rest before the line is re-latched STOPPING: the
    gate releases and the vehicle must not enter during the other approach's green."""
    w = _empty_world()
    plant = w.plants[0]
    edge = w.net.edges["N0_J00"]
    veh = _place(w, "N0", "S0", edge, 0, edge.stop_s - 0.5, 0.0)
    veh.latch = COMMITTED
    plant.state, plant.phase, plant.pending_phase = ALLRED, DIR_INDEX["N"], DIR_INDEX["E"]
    plant.allred_start_t = w.t - w.cfg.allred_min
    w.step()
    assert veh.latch == STOPPING and w.n_commit_revoked == 1
    assert plant.state == GREEN and plant.phase == DIR_INDEX["E"]
    for _ in range(40):
        w.step()
    assert veh.edge is edge and veh.s < edge.stop_s                            # still waiting before the line
    assert w.invariant_counts()["red_violations"] == 0


def test_crossing_under_another_green_is_a_violation_even_when_committed():
    w = _empty_world()
    plant = w.plants[0]
    edge = w.net.edges["N0_J00"]
    veh = _place(w, "N0", "S0", edge, 0, edge.stop_s - 1.0, 5.0)
    veh.latch = COMMITTED
    plant.state, plant.phase = GREEN, DIR_INDEX["E"]                           # another approach is green
    w.record_crossing(veh, edge, w.t)
    assert w.red_violations == 1
    plant.state, plant.phase, plant.pending_phase = ALLRED, DIR_INDEX["N"], DIR_INDEX["E"]
    w.record_crossing(veh, edge, w.t)                                          # during its own all-red: allowed
    assert w.red_violations == 1


# ----------------------------------------------------------------- safety-physics-03
def test_plant_starvation_guard_forces_service():
    cfg = SimConfig()
    w = _empty_world()
    plant = w.plants[0]
    plant.time_in_phase = 10.0
    applied, forced, switched, _ = plant.request(0, [5, 0, 0, 3], cfg.starvation_s - 10.0)   # W waited 170 s: not yet
    assert not switched and forced == 0
    applied, forced, switched, red = plant.request(0, [5, 0, 0, 3], cfg.starvation_s)        # W waited 180 s: forced
    assert switched and red and forced == FORCED_STARVATION and applied == 3
    assert plant.last_green_end[0] == cfg.starvation_s
    # ties between starved approaches go in rotation order from the current phase
    w2 = _empty_world()
    p2 = w2.plants[0]
    p2.time_in_phase = cfg.max_green                       # max-green would pick the largest n (W); starvation wins
    applied, forced, switched, _ = p2.request(0, [5, 2, 0, 9], cfg.starvation_s + 5.0)
    assert switched and forced == FORCED_STARVATION and applied == 1
    # an approach that was green recently is not starved
    w3 = _empty_world()
    p3 = w3.plants[0]
    p3.last_green_end[3] = 150.0
    p3.time_in_phase = 10.0
    applied, forced, switched, _ = p3.request(0, [5, 0, 0, 3], 200.0)
    assert not switched and forced == 0


def test_no_approach_with_vehicles_waits_longer_than_the_starvation_bound():
    """Closed loop at stress demand: the longest red an occupied approach sees is bounded by
    starvation_s plus one decision interval and the all-red (used to be > 1000 s)."""
    cfg = SimConfig(warmup_s=60.0, main_s=600.0, cooldown_s=60.0)
    w = World(cfg, Scenario(lambda_base_vph=600.0, controller="max_pressure"), seed=100006).run()
    r = w.episode_row()
    assert r["max_approach_red_s"] <= cfg.starvation_s + cfg.decision_interval + cfg.allred_min + 20.0
    assert r["n_forced_starvation"] >= 0 and r["n_invariant_violations"] == 0
    # the KPI is measured, not computed from the guard: it is positive and below the bound
    assert 0.0 < r["max_approach_red_s"]


# ----------------------------------------------------------------- traffic-validation-05
def test_gridlock_is_detected_per_junction():
    """Ten vehicles waiting at a junction whose box never clears -> gridlock at 120 s,
    the episode ends there and the junction is recorded."""
    w = _empty_world()
    plant = w.plants[0]
    edge = w.net.edges["N0_J00"]
    s = edge.stop_s - w.cfg.s0
    for _ in range(10):
        _place(w, "N0", "S0", edge, 0, s, 0.0)
        s -= w.cfg.l_veh + w.cfg.s0
    plant.state, plant.phase, plant.pending_phase = ALLRED, DIR_INDEX["E"], DIR_INDEX["N"]
    plant.allred_start_t = 0.0
    w.conflicts_clear = lambda J, phase_idx: False         # a vehicle stuck in a conflict zone forever
    w.run()
    assert w.gridlock and math.isclose(w.gridlock_t, w.cfg.gridlock_window_s, abs_tol=w.cfg.dt + 1e-9)
    assert w.gridlock_junction == 0
    assert w.t < w.cfg.episode_length_s                    # ended early
    assert w.episode_row()["gridlock"] is True


# ----------------------------------------------------------------- safety-physics-04
def test_two_abreast_vehicles_keep_their_lanes_through_the_box():
    """v2.0 lane discipline (DECISIONS §12.1) replaces the v1 zipper merge: two standing through
    vehicles side by side at the S stop line of J00 cross the box on their own straight paths and
    come out in their own lanes - never overlapping, no hard braking, no red violation."""
    w = _empty_world(grid=2)
    w.plants[0].phase = DIR_INDEX["S"]
    edge = w.net.edges["S0_J00"]
    a = _place(w, "S0", "N0", edge, 0, edge.stop_s - w.cfg.s0_line, 0.0)
    b = _place(w, "S0", "N0", edge, 1, edge.stop_s - w.cfg.s0_line, 0.0)
    for _ in range(40):
        w.step()
        for veh in (a, b):
            if veh.conn is not None:
                assert veh.conn.in_lane == veh.conn.out_lane and 0.0 <= veh.s <= veh.conn.length + 1e-9
    out = w.net.edges["J00_J01"]
    assert a in out.lanes[0] and b in out.lanes[1]
    assert w.n_harsh_brake_events == 0 and w.n_overlap_fixed == 0
    assert w.invariant_counts()["red_violations"] == 0 and w.invariant_counts()["collisions"] == 0


def test_idm_rejects_a_non_positive_gap():
    with pytest.raises(ValueError):
        idm_accel(5.0, 13.9, 0.0, 0.0, 1.5, 2.0, 1.2, 2.0)
    with pytest.raises(ValueError):
        idm_accel(5.0, 13.9, -1.0, 0.0, 1.5, 2.0, 1.2, 2.0)


def test_no_negative_connector_positions_in_a_stress_episode():
    cfg = SimConfig(warmup_s=30.0, main_s=300.0, cooldown_s=30.0)
    w = World(cfg, Scenario(lambda_base_vph=600.0, controller="random_dwell"), seed=100003)
    orig = w.step
    worst = [0.0]

    def step():
        orig()
        for J in w.net.junctions:
            for c in J.connectors.values():
                for v in c.vehicles:
                    worst[0] = min(worst[0], v.s)
    w.step = step
    w.run()
    assert worst[0] >= 0.0
    assert w.episode_row()["n_invariant_violations"] == 0


# ----------------------------------------------------------------- data-schema-001
def test_data_dictionary_covers_every_csv_column(tmp_path):
    from traffic_twin.generate import generate
    cfg = SimConfig(**SHORT)
    generate(cfg, 2, str(tmp_path), verbose=False, paired=False)
    dd = pd.read_csv(tmp_path / "data_dictionary.csv")
    for table in ("episodes", "vehicles", "signal_ticks", "junction_1hz", "network_edges"):
        cols = set(pd.read_csv(tmp_path / f"{table}.csv").columns)
        documented = set(dd[dd.table == table].column)
        assert cols == documented, (table, cols ^ documented)
    eps = pd.read_csv(tmp_path / "episodes.csv")
    assert {"gridlock_junction", "max_approach_red_s", "n_forced_starvation"} <= set(eps.columns)
