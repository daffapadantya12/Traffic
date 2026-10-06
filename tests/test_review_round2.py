"""Regression tests for the recheck findings (DECISIONS §11.7-§11.9, traffic_twin v1.2).

NF-02  (v2.0) the zipper merge is gone (lane discipline); its successor checks: the follower of a
       COMMITTED vehicle brakes for the stop line in time, and stress episodes have no collisions,
       brake saturation or near misses
NF-03  lane-level liveness KPIs: max_head_wait_s and the per-junction spillback lock
§11.9  calibration measures the realised capacity and derives lambda_base from it
"""

import json
import math

from traffic_twin import Scenario, SimConfig, World
from traffic_twin.calibrate import (approach_capacity_vph, busiest_approach_flow, calibrate_demand,
                                    lambda_for_vc, measure_capacity)
from traffic_twin.config import load_config
from traffic_twin.network import DIR_INDEX, shortest_route
from traffic_twin.signals import ALLRED
from traffic_twin.vehicles import Vehicle, sample_params


def _empty_world(grid=1, **kw):
    cfg = SimConfig(grid_x=grid, grid_y=grid, warmup_s=0.0, main_s=200.0, cooldown_s=0.0, **kw)
    return World(cfg, Scenario(lambda_base_vph=0.0, controller="longest_queue"), seed=1)


def _place(world, origin, destination, edge, lane, s, v, params=None):
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


# ----------------------------------------------------------------- NF-02 (v2.0: no zipper merge any more)
def test_follower_of_a_committed_vehicle_brakes_for_the_stop_line_in_time():
    """Red onset with a COMMITTED leader close to the line and a STOPPING follower: the follower must
    obey the most demanding obstacle (the stop line), not the nearest one (its leader, which is about
    to cross).  v1.2 picked the nearest by gap and braked at -7 m/s^2 once the leader had crossed."""
    w = _empty_world()
    cfg = w.cfg
    plant = w.plants[0]
    plant.phase = DIR_INDEX["S"]
    edge = w.net.edges["S0_J00"]
    p = dict(alpha=0.5, v0=12.5, a_max=1.5, b=2.0, T=1.2, l_veh=4.5, vehicle_type="car")
    lead = _place(w, "S0", "N0", edge, 0, edge.stop_s - 12.0, 12.4, dict(p))
    fol = _place(w, "S0", "N0", edge, 0, edge.stop_s - 47.0, 12.4, dict(p))     # leader 30.5 m ahead, line 47 m
    w.stop_or_commit(w.net.junctions[0], plant.phase)
    plant.state, plant.pending_phase, plant.allred_start_t = ALLRED, DIR_INDEX["E"], w.t
    assert lead.latch == "COMMITTED" and fol.latch == "STOPPING"
    worst = 0.0
    for _ in range(40):
        v_prev = fol.v
        w.step()
        worst = max(worst, (v_prev - fol.v) / cfg.dt)
    assert fol.edge is edge and fol.s <= edge.stop_s + 1e-9          # stopped before the line
    assert worst <= 3.4 + 1e-6, worst                                   # no harsh braking
    assert w.n_brake_sat_virtual == 0 and w.n_brake_sat_vehicle == 0
    assert w.invariant_counts()["red_violations"] == 0


def test_no_collisions_or_brake_saturation_in_stress_episodes():
    cfg = SimConfig(warmup_s=60.0, main_s=600.0, cooldown_s=60.0)
    for ctrl, seed in (("random_dwell", 100003), ("fixed_time", 100004), ("max_pressure", 100005)):
        w = World(cfg, Scenario(lambda_base_vph=480.0, controller=ctrl, lambda_ev_vph=8.0, ev_rate="default"), seed=seed).run()
        r = w.episode_row()
        assert r["n_collisions"] == 0 and r["n_invariant_violations"] == 0, (ctrl, w.invariant_counts())
        assert r["n_brake_sat_vehicle"] == 0 and r["n_brake_sat_virtual"] == 0, ctrl
        assert r["n_near_misses"] == 0, ctrl


# ----------------------------------------------------------------- NF-03
def test_max_head_wait_measures_a_standing_lane_head():
    """One vehicle at a stop line that never turns green stands for the whole 200 s episode;
    with one vehicle there is no gridlock and no spillback lock (the approach never had green)."""
    w = _empty_world()
    plant = w.plants[0]
    edge = w.net.edges["N0_J00"]
    _place(w, "N0", "S0", edge, 0, edge.stop_s - w.cfg.s0, 0.0)
    plant.state, plant.phase, plant.pending_phase = ALLRED, DIR_INDEX["E"], DIR_INDEX["E"]
    plant.allred_start_t = 0.0
    w.conflicts_clear = lambda J, phase_idx: False         # the all-red never ends
    w.run()
    r = w.episode_row()
    assert not r["gridlock"] and r["max_head_wait_s"] >= 199.0
    assert r["n_spillback_locks"] == 0 and r["spillback_lock"] is False and r["spillback_lock_junction"] == -1


def test_spillback_lock_flags_a_blocked_exit_under_green():
    """A lane head whose exit lanes are blocked stands through a permanent green: after
    gridlock_window_s with >= 2 x fixed_time_green of green the lane is flagged as spillback
    locked; the episode continues and gridlock (>= 10 vehicles) is untouched."""
    w = _empty_world()
    cfg = w.cfg
    assert w.plants[0].phase == DIR_INDEX["N"]
    parked = dict(alpha=0.5, v0=10.0, a_max=1e-6, b=2.0, T=1.2, l_veh=4.5, vehicle_type="car")
    out = w.net.edges["J00_S0"]
    for k in range(out.n_lanes):                          # a stopped vehicle leaves < l_veh+s0+1 m free
        _place(w, "N0", "S0", out, k, 5.0, 0.0, dict(parked))
    edge = w.net.edges["N0_J00"]
    head = _place(w, "N0", "S0", edge, 0, edge.stop_s - cfg.s0, 0.0)
    while w.t < 119.0:
        w.step()
    assert w.n_spillback_locks == 0 and head.edge is edge and head.v < cfg.stop_speed
    w.run()
    r = w.episode_row()
    assert r["spillback_lock"] is True and r["n_spillback_locks"] == 1 and r["spillback_lock_junction"] == 0
    assert math.isclose(r["spillback_lock_t"], cfg.gridlock_window_s, abs_tol=1.0 + 1e-9)
    assert not r["gridlock"] and r["max_head_wait_s"] >= cfg.gridlock_window_s
    assert w.green_time_in_window(w.net.junctions[0], w.plants[0], "N", w.t - cfg.gridlock_window_s) >= 2 * cfg.fixed_time_green


def test_liveness_kpis_in_a_closed_loop_episode_and_pressure_untouched():
    from traffic_twin.controllers import pressure
    cfg = SimConfig(warmup_s=60.0, main_s=300.0, cooldown_s=60.0)
    w = World(cfg, Scenario(lambda_base_vph=300.0, controller="max_pressure"), seed=100007).run()
    r = w.episode_row()
    assert 0.0 < r["max_head_wait_s"] <= cfg.episode_length_s and r["n_spillback_locks"] >= 0
    row = {f"n_{a}": 3 for a in "NESW"}
    row.update({f"down_n_{d}": 1 for d in "NESW"})
    row.update({f"down_blocked_{d}": False for d in "NESW"})
    assert pressure(row, "N") == 2.0                      # 3 - mean(1, 1, 1): the §6 function unchanged


# ----------------------------------------------------------------- §11.9 calibration
def test_calibration_measures_realised_capacity_and_derives_lambda(tmp_path):
    cfg = SimConfig()
    sat = 1631.0
    probe = measure_capacity(cfg, sat, main_s=600.0)
    formula = approach_capacity_vph(cfg, sat)
    assert math.isclose(probe["capacity_formula_vph"], formula)
    assert 0.7 <= probe["capacity_realisation"] <= 1.0 and probe["capacity_vph"] < formula
    assert math.isclose(probe["capacity_realisation"], probe["capacity_vph"] / formula)
    assert not probe["probe_gridlock"] and probe["probe_n_invariant_violations"] == 0
    # lambda_base per level comes from the MEASURED capacity (given here to skip the probe)
    c = calibrate_demand(cfg, sat_flow_vphpl=sat, capacity_vph=500.0)
    _, k = busiest_approach_flow(cfg)
    assert cfg.capacity_vph == 500.0 and math.isclose(cfg.capacity_formula_vph, round(formula, 2))
    assert math.isclose(cfg.capacity_realisation, round(500.0 / formula, 4))
    for level, x in cfg.vc_targets.items():
        assert math.isclose(cfg.lambda_base_vph[level], round(x * 500.0 / k, 3))
        assert math.isclose(c["lambda_base_vph"][level], cfg.lambda_base_vph[level])
    cfg1 = cfg.copy(grid_x=1, grid_y=1)
    _, k1 = busiest_approach_flow(cfg1)
    assert math.isclose(lambda_for_vc(cfg1, 1.0, 500.0), 500.0 / k1)
    # the three capacity fields survive the config.json round trip
    p = tmp_path / "config.json"
    p.write_text(cfg.to_json())
    back = load_config(str(p))
    assert back.capacity_vph == 500.0 and back.capacity_realisation == cfg.capacity_realisation
    assert back.capacity_formula_vph == cfg.capacity_formula_vph and back.config_hash() == cfg.config_hash()
    from traffic_twin.config import GENERATOR_VERSION
    assert json.loads(cfg.to_json())["generator_version"] == GENERATOR_VERSION
