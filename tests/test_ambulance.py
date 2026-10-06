"""Ambulance extension (DECISIONS §10): spawning, ev_* features, teachers, CRN for cars."""

import math

import pandas as pd

from traffic_twin import Scenario, SimConfig, World
from traffic_twin.controllers import label_request, max_pressure_request, preempt_naive_request, priority_mp_request
from traffic_twin.features import EV_COLUMNS, SIGNAL_TICK_COLUMNS

SHORT = dict(warmup_s=60.0, main_s=300.0, cooldown_s=60.0)


def _row(phase=0, tip=10.0):
    row = {"phase": phase, "time_in_phase_s": tip}
    for a in "NESW":
        row[f"n_{a}"] = 0
        row[f"ev_present_{a}"] = False
        row[f"ev_dist_{a}"] = -1.0
        row[f"ev_eta_{a}"] = -1.0
        row[f"ev_q_ahead_{a}"] = -1
    for d in "NESW":
        row[f"down_n_{d}"] = 0
        row[f"down_blocked_{d}"] = False
    return row


def test_teachers_on_hand_made_rows():
    cfg = SimConfig()
    cfg.measured_sat_flow_vphpl = 1800.0          # sat headway 2.0 s
    cfg.startup_lost_time_s = 2.0
    row = _row(phase=0)
    row["n_E"] = 5                                 # max-pressure alone would pick E
    assert max_pressure_request(row, cfg) == 1
    assert priority_mp_request(row, cfg) == 1 and preempt_naive_request(row, cfg) == 1
    # ambulance on S, 15 s away, 3 queued vehicles ahead: t_clear = 4 + 2 + 3*2 = 12 s < 15 -> keep max-pressure
    row.update(ev_present_S=True, ev_dist_S=150.0, ev_eta_S=15.0, ev_q_ahead_S=3)
    assert priority_mp_request(row, cfg) == 1
    assert preempt_naive_request(row, cfg) == 2   # naive: 15 s <= 20 s -> force S
    row["ev_eta_S"] = 11.0                         # now inside the clearing time -> S
    assert priority_mp_request(row, cfg) == 2
    assert label_request(row, cfg) == 2
    row["ev_eta_S"] = 25.0                         # far away: neither teacher reacts
    assert priority_mp_request(row, cfg) == 1 and preempt_naive_request(row, cfg) == 1
    # two ambulances: the smaller ETA wins
    row.update(ev_eta_S=5.0, ev_present_W=True, ev_dist_W=20.0, ev_eta_W=3.0, ev_q_ahead_W=0)
    assert preempt_naive_request(row, cfg) == 3


def test_ambulances_spawn_and_are_observed():
    cfg = SimConfig(**SHORT)
    sc = Scenario(lambda_base_vph=200.0, controller="priority_mp", lambda_ev_vph=60.0, ev_rate="default")
    w = World(cfg, sc, seed=100011).run()
    assert w.n_ambulances > 5
    v = pd.DataFrame(w.vehicle_rows())
    amb = v[v.vehicle_type == "ambulance"]
    assert len(amb) == w.n_ambulances and (amb.v0_mps > cfg.v_limit * 1.2).all()
    ticks = pd.DataFrame(w.signal_tick_rows())
    assert list(ticks.columns) == SIGNAL_TICK_COLUMNS and all(c in ticks.columns for c in EV_COLUMNS)
    present = ticks[[f"ev_present_{a}" for a in "NESW"]].any(axis=1)
    assert present.any()
    seen = ticks[ticks.ev_present_N]
    if len(seen):
        assert (seen.ev_dist_N >= 0).all() and (seen.ev_eta_N >= 0).all() and (seen.ev_q_ahead_N >= 0).all()
    assert (ticks[~ticks.ev_present_N].ev_dist_N == -1).all()
    for _, row in ticks.iterrows():
        assert label_request(row.to_dict(), cfg) == row.y_mp_action
    r = w.episode_row()
    assert r["n_ambulances"] == w.n_ambulances and not math.isnan(r["ev_mean_delay_s"])
    assert r["n_invariant_violations"] == 0 and w.invariant_counts()["red_violations"] == 0


def test_cars_keep_common_random_numbers_with_and_without_ambulances():
    cfg = SimConfig(**SHORT)
    base = dict(lambda_base_vph=220.0, controller="max_pressure")
    a = World(cfg, Scenario(**base), seed=100012).run()
    b = World(cfg, Scenario(**base, lambda_ev_vph=30.0, ev_rate="default"), seed=100012).run()
    # the drawn quantities are identical; realised routes may differ because cars make room for ambulances
    cols = ["spawn_request_t", "origin", "destination", "alpha", "v0_mps", "T_headway_s"]
    cars_a = pd.DataFrame(a.vehicle_rows()).query("vehicle_type == 'car'")[cols].sort_values(cols)
    cars_b = pd.DataFrame(b.vehicle_rows()).query("vehicle_type == 'car'")[cols].sort_values(cols)
    pd.testing.assert_frame_equal(cars_a.reset_index(drop=True), cars_b.reset_index(drop=True))
    assert b.n_ambulances > 0 and a.n_ambulances == 0
