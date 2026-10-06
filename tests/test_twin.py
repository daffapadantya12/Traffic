"""Digital-twin tests: estimator convergence, the controller sees twin fields only, the web
frame carries every key index.html reads, a short twin run keeps every invariant, and the
daily demand profile integrates to the expected vehicle count."""

import json
import math
import os
import re

import numpy as np
import pytest

from traffic_twin import Scenario, SimConfig, World
from traffic_twin.config import DAILY_PROFILE, PROFILE_FIELDS
from traffic_twin.controllers import Controller
from traffic_twin.features import FEATURE_COLUMNS, build_features
from traffic_twin.network import DIRS, build_grid
from traffic_twin.twin import (DetectorBank, QueueEstimator, SensorFrame, TwinRow, TwinWorld, TWIN_ROW_KEYS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_HTML = os.path.join(ROOT, "traffic_twin", "static", "index.html")


# ---------------------------------------------------------------- 1. estimator on a synthetic stream
def _synthetic_stream(rng, n_s=900, storage=56):
    """A link with Poisson arrivals, a 40 s red / 30 s green cycle; returns per-second
    (entries, exits, n_true, q_true).  Vehicles take 10 s to reach the queue."""
    n, q, moving = 0, 0, []
    out = []
    for t in range(n_s):
        arr = rng.poisson(0.35)
        moving.append([t, arr])
        n += arr
        joined = sum(c for tt, c in moving if t - tt >= 10)
        moving = [m for m in moving if t - m[0] < 10]
        q += joined
        green = (t % 70) >= 40
        exits = min(q, 1) if green else 0          # saturation discharge 1 veh/s, 30 s green > 0.35 x 70 arrivals
        q -= exits
        n -= exits
        q = min(q, storage)
        out.append((arr, exits, n, q))
    return out


def test_estimator_converges_on_synthetic_stream():
    cfg = SimConfig()
    net = build_grid(cfg)
    edge = net.edges["W0_J00"]
    rng = np.random.default_rng(3)
    stream = _synthetic_stream(rng)
    # noiseless channel: the estimate must track the queue exactly once the EMA has settled
    est = QueueEstimator(net, cfg)
    errs = []
    for t, (arr, ex, n, q) in enumerate(stream):
        est.update(SensorFrame(float(t), float(t), edge.id, arr, ex, q, True))
        if t > 60:
            errs.append(abs(est.edges[edge.id].queue_est - q))
    assert np.mean(errs) < 1.0 and np.percentile(errs, 90) <= 2.0, (np.mean(errs), np.percentile(errs, 90))
    # noisy channel (misses, doubles, camera N(0,1), drops): still within 1.5 veh on average
    bank = DetectorBank(net, np.random.default_rng(5))
    est2 = QueueEstimator(net, cfg)
    errs2, nhat_err = [], []
    for t, (arr, ex, n, q) in enumerate(stream):
        est2.update(bank.corrupt(float(t), edge, arr, ex, q))
        if t > 60:
            errs2.append(abs(est2.edges[edge.id].queue_est - q))
            nhat_err.append(abs(est2.edges[edge.id].n_hat - n))
    assert np.mean(errs2) < 1.5, np.mean(errs2)
    assert np.mean(nhat_err) < 4.0, np.mean(nhat_err)        # conservation + re-anchoring bounds the drift
    assert est2.n_stuck_events == 0


def test_frozen_camera_is_detected_and_recovers():
    cfg = SimConfig()
    net = build_grid(cfg)
    edge = net.edges["W0_J00"]
    rng = np.random.default_rng(11)
    stream = _synthetic_stream(rng)
    bank = DetectorBank(net, np.random.default_rng(7))
    bank.freeze_camera(edge.id, 300.0, 60.0)
    est = QueueEstimator(net, cfg)
    weights = {}
    for t, (arr, ex, n, q) in enumerate(stream):
        est.update(bank.corrupt(float(t), edge, arr, ex, q))
        weights[t] = est.edges[edge.id].weight
    stuck = [e for e in est.log if e["event"] == "camera_stuck"]
    recov = [e for e in est.log if e["event"] == "camera_recovered"]
    assert stuck and 300 <= stuck[0]["t"] <= 330, stuck            # fires within 15-30 frames of the freeze
    assert recov and recov[0]["t"] <= 360 + 20, recov              # trusted again within 20 s of recovery
    assert weights[340] == 0.0 and weights[420] == 1.0


# ---------------------------------------------------------------- 2. the controller sees twin fields only
class SpyController(Controller):
    name = "spy"

    def __init__(self, cfg, rng=None):
        super().__init__(cfg, rng)
        self.rows = []

    def act(self, row):
        self.rows.append(dict(row))
        self.seen_types = getattr(self, "seen_types", set()) | {type(row)}
        return int(row["phase"]) if row["time_in_phase_s"] < 20 else (int(row["phase"]) + 1) % 4


def _twin(controller="priority_mp", seconds=120, **kw):
    cfg = SimConfig(lambda_base_vph={"design": 325.678})
    tw = TwinWorld(cfg, demand_level="design", controller=controller, profile="daily", start_hour=8.0, seed=4, **kw)
    return tw


def test_controller_receives_only_twin_fields():
    tw = _twin()
    spy = SpyController(tw.cfg)
    tw.bridge.inner = spy
    truth_rows = []
    orig_act = tw.bridge.act

    def act_and_record(physical_row):          # what World hands the bridge, for comparison only
        truth_rows.append(dict(physical_row))
        return orig_act(physical_row)

    tw.world.controller.act = act_and_record
    tw.advance(180)
    assert len(spy.rows) > 20
    assert spy.seen_types == {TwinRow}
    for row in spy.rows:
        assert row["source"] == "twin"
        assert set(row) == set(TWIN_ROW_KEYS), set(row) ^ set(TWIN_ROW_KEYS)
        assert set(FEATURE_COLUMNS) <= set(row)
    # the physical row never reaches the controller: its queue columns are estimates, and they differ
    # from the truth on a substantial share of decisions (1 s delay, noise, EMA)
    diffs = [any(abs(float(tr[f"q_{a}"]) - float(sr[f"q_{a}"])) > 1e-9 for a in DIRS)
             for tr, sr in zip(truth_rows, spy.rows)]
    assert np.mean(diffs) > 0.1, np.mean(diffs)
    # ... but they are close (the twin works): mean |q_est - q_true| over decisions < 1.5 veh
    mae = np.mean([abs(float(tr[f"q_{a}"]) - float(sr[f"q_{a}"])) for tr, sr in zip(truth_rows, spy.rows) for a in DIRS])
    assert mae < 1.5, mae


def test_twin_row_has_build_features_schema_and_types():
    tw = _twin()
    tw.advance(30)
    row = tw.row(0)
    truth = build_features(tw.world, tw.net.junctions[0])
    assert set(truth) == set(row) - {"source"}
    for k in truth:
        if isinstance(truth[k], bool):
            assert isinstance(row[k], (bool, np.bool_)), k
        else:
            assert isinstance(row[k], (int, float, np.integer, np.floating)), k


# ---------------------------------------------------------------- 3. frame schema vs index.html
def _keys_read_by_index_html():
    src = open(INDEX_HTML, encoding="utf-8").read()
    body = src.split("function renderPanels")[1].split("function drawProfile")[0]
    top = set(re.findall(r"\bf\.([a-zA-Z_]+)", body)) | set(re.findall(r"\bf\.([a-zA-Z_]+)", src.split("function render(")[1][:200]))
    kpi = set(re.findall(r"\bf\.kpi\.([a-zA-Z_]+)", body))
    junction = set(re.findall(r"\bj\.([a-zA-Z_]+)", body))
    signal = set(re.findall(r"\bsig\.([a-zA-Z_]+)", src.split("function drawArena")[1].split("function line")[0]))
    return top - {"kpi", "twin"}, kpi, junction, signal


def test_frame_contains_every_key_index_html_reads():
    tw = _twin()
    tw.advance(20)
    f = tw.frame()
    json.dumps(f, allow_nan=False)                     # strictly serialisable
    top, kpi, junction, signal = _keys_read_by_index_html()
    assert {"clock", "day", "hour", "demand_mult", "demand_label", "controller", "signals", "vehicles", "profile", "events"} <= top
    assert "t" in f
    assert top <= set(f), top - set(f)
    assert kpi <= set(f["kpi"]), kpi - set(f["kpi"])
    assert {"n_vehicles", "car_delay_s", "ev_delay_s", "throughput_vph", "switches_per_min", "ambulances_today"} <= set(f["kpi"])
    assert "speed_achieved" in f["kpi"]
    assert len(f["signals"]) == 4 and all(signal <= set(s) for s in f["signals"])
    assert all(s["state"] in ("GREEN", "ALLRED") and 0 <= s["phase"] <= 3 for s in f["signals"])
    assert len(f["twin"]["junctions"]) == 4
    for j in f["twin"]["junctions"]:
        assert junction <= set(j), junction - set(j)
        assert len(j["q_true"]) == 4 and len(j["q_est"]) == 4
    assert [j["id"] for j in f["twin"]["junctions"]] == [0, 1, 2, 3]
    assert len(f["profile"]) == 24 and f["profile"] == list(DAILY_PROFILE)
    assert f["vehicles"] and all(len(v) == 5 and v[3] in (0, 1) for v in f["vehicles"])
    assert re.fullmatch(r"\d\d:\d\d:\d\d", f["clock"]) and f["day"] == 1 and 8.0 <= f["hour"] < 8.1


# ---------------------------------------------------------------- 4. a 2-minute twin run keeps every invariant
@pytest.mark.parametrize("controller", ["priority_mp", "ml"])
def test_two_minute_twin_run_has_zero_invariant_violations(controller):
    model = os.path.join(ROOT, "reports", "ml", "models", "taskA_hgb_sensor.joblib")
    if controller == "ml" and not os.path.exists(model):
        pytest.skip("trained model not present")
    tw = _twin(controller=controller, model_path=model)
    tw.advance(120)
    assert tw.t == 120.0
    assert tw.invariant_violations() == 0
    inv = tw.world.invariant_counts()
    assert all(v == 0 for k, v in inv.items() if k != "conservation_error"), inv
    assert tw.bridge.n_decisions > 20
    s = tw.summary()
    assert s["mean_queue_mae_truth"] < 2.0 and s["mean_data_age_s"] <= 1.5
    if controller == "ml":
        assert tw.controller_label.startswith("ML (")
    f = tw.frame()
    assert f["kpi"]["n_vehicles"] == tw.world.n_in_network()


def test_missing_model_falls_back_visibly():
    tw = _twin(controller="ml", model_path="/nonexistent/model.joblib")
    assert tw.controller_label == "priority_mp (model not found)"
    tw.advance(10)
    assert tw.frame()["controller"] == "priority_mp (model not found)"


# ---------------------------------------------------------------- 5. the daily profile
def test_flat_profile_keeps_config_hash_and_daily_changes_it():
    flat = SimConfig()
    daily = SimConfig(demand_profile="daily")
    assert not any(k in json.loads(flat.to_json()) for k in PROFILE_FIELDS)
    assert all(k in json.loads(daily.to_json()) for k in PROFILE_FIELDS)
    assert flat.config_hash() != daily.config_hash()
    assert flat.demand_multiplier(12 * 3600) == 1.0
    assert daily.demand_multiplier(8 * 3600 + 1) == DAILY_PROFILE[8] == 1.35
    assert daily.demand_multiplier(86400 + 2 * 3600) == DAILY_PROFILE[2]
    assert SimConfig(demand_profile="daily", demand_profile_t0_s=7 * 3600).demand_multiplier(3600) == DAILY_PROFILE[8]


def test_daily_profile_integrates_to_expected_vehicle_count():
    """A compressed day (2400 s = 100 s per 'hour') in the real simulator: spawn requests
    must match lambda x sum(profile) within 10%; the flat profile is bit-identical to v1.2."""
    day_s = 2400.0
    lam = 400.0
    cfg = SimConfig(grid_x=1, grid_y=1, warmup_s=0.0, main_s=day_s, cooldown_s=0.0,
                    demand_profile="daily", demand_profile_day_s=day_s)
    sc = Scenario(lambda_base_vph=lam, controller="max_pressure")
    w = World(cfg, sc, seed=100042).run()
    n_nodes = len(w.net.boundary_nodes)
    expected = n_nodes * lam / 3600.0 * (day_s / 24.0) * sum(DAILY_PROFILE)
    assert abs(w.n_spawn_requests - expected) / expected < 0.10, (w.n_spawn_requests, expected)
    # flat: the profile code path leaves the draw sequence untouched (same seed -> same spawns)
    cfg_flat = SimConfig(grid_x=1, grid_y=1, warmup_s=0.0, main_s=600.0, cooldown_s=0.0)
    a = World(cfg_flat, sc, seed=100043).run()
    b = World(cfg_flat.copy(demand_profile="flat", demand_profile_t0_s=5 * 3600.0), sc, seed=100043).run()
    assert a.n_spawn_requests == b.n_spawn_requests
    assert [v.spawn_request_t for v in a.finished] == [v.spawn_request_t for v in b.finished]


def test_twin_day_counter_and_hourly_aggregation():
    cfg = SimConfig(lambda_base_vph={"design": 325.678}, demand_profile_day_s=86400.0)
    tw = TwinWorld(cfg, demand_level="design", controller="priority_mp", profile="daily", start_hour=23.0 + 3590 / 3600, seed=9)
    tw.advance(40)                                    # crosses midnight at t = 10 s
    assert tw.day == 2 and tw.clock().startswith("00:00:")
    assert len(tw.hour_stats) == 1 and tw.hour_stats[0]["hour"] == 23 and tw.hour_stats[0]["day"] == 1
    assert tw.hour_stats[0]["invariant_violations"] == 0
