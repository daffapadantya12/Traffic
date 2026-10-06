"""ML pipeline (traffic_twin/ml.py) and closed-loop evaluator (traffic_twin/evaluate.py):
feature-set leakage guards, split integrity, the MLController contract, the acceptance bound,
the fairness ratio, the ambulance-off pairing and a smoke run of the protocol."""

import math
import os

import numpy as np
import pandas as pd
import pytest

from traffic_twin import SimConfig
from traffic_twin import evaluate, ml
from traffic_twin.controllers import CONTROLLERS, make_controller
from traffic_twin.features import ACTION_COLUMNS, FEATURE_COLUMNS, KEY_COLUMNS, LABEL_COLUMNS, SIGNAL_TICK_COLUMNS
from traffic_twin.generate import data_dictionary_rows

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "traffic_v1")
SHORT = dict(warmup_s=60.0, main_s=240.0, cooldown_s=60.0)


@pytest.fixture(scope="module")
def dd():
    d = pd.DataFrame(data_dictionary_rows())
    return d[d.table == "signal_ticks"].reset_index(drop=True)


@pytest.fixture(scope="module")
def tiny_bundle(dd):
    """A real HGB classifier fitted on random sensor rows: enough to exercise the controller contract."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    cols = ml.feature_sets(dd)["sensor"]
    rng = np.random.default_rng(0)
    X = rng.integers(0, 10, size=(400, len(cols))).astype(np.float32)
    y = rng.integers(0, 4, size=400)
    model = HistGradientBoostingClassifier(max_iter=5, max_depth=2).fit(X, y)
    return {"model": model, "features": cols, "classes": [0, 1, 2, 3]}


def _row(phase=2, tip=10.0):
    """A full feature row like build_features() produces, with no ambulance anywhere (ev columns = -1)."""
    row = {c: 0 for c in FEATURE_COLUMNS}
    row.update(episode_id=0, junction_id=0, t=100.0, tick_idx=3, is_warmup=False, is_cooldown=False, is_tail=False,
               phase=phase, time_in_phase_s=tip, max_green_remaining_s=60.0 - tip)
    for a in "NESW":
        row[f"ev_present_{a}"] = False
        row[f"ev_dist_{a}"] = -1.0
        row[f"ev_eta_{a}"] = -1.0
        row[f"ev_q_ahead_{a}"] = -1
        row[f"down_blocked_{a}"] = False
        row[f"nb_{a}_exists"] = a in "NE"
        row[f"nb_{a}_phase"] = 1 if a in "NE" else -1
        row[f"nb_{a}_green_toward_me"] = False
    return row


# ---------------------------------------------------------------- feature sets
def test_feature_sets_exclude_leaky_columns(dd):
    fs = ml.feature_sets(dd)
    leaky = set(KEY_COLUMNS) | set(ACTION_COLUMNS) | set(LABEL_COLUMNS)
    for name, cols in fs.items():
        assert set(cols) <= set(SIGNAL_TICK_COLUMNS)
        allowed = {"action_applied"} if name == "regression" else set()
        assert not ((set(cols) & leaky) - allowed), name
        assert not any(c.startswith("y_") for c in cols)
        assert "junction_id" not in cols and "t" not in cols and "tick_idx" not in cols
        assert len(cols) == len(set(cols))
    assert "action_applied" in fs["regression"] and "action_applied" not in fs["truth"]
    sensor = fs["sensor"]
    assert all(f"obs_q_{a}" in sensor for a in "NESW")
    assert not any(c.startswith(("n_", "arr_", "q_")) or c.endswith(("_l0", "_l1")) for c in sensor)
    assert all(f"q_{a}" in fs["truth"] and f"n_{a}" in fs["truth"] for a in "NESW")
    assert (len(fs["truth"]), len(fs["sensor"]), len(fs["regression"])) == (63, 47, 64)


def test_feature_sets_match_shipped_dictionary():
    """The dictionary written into data/traffic_v1 (v1.2 schema) selects the same sets as the code's own rows."""
    path = os.path.join(DATA, "data_dictionary.csv")
    if not os.path.exists(path):
        pytest.skip("data/traffic_v1 not generated")
    fs_file = ml.feature_sets(ml.load_dictionary(DATA))
    fs_code = ml.feature_sets(pd.DataFrame(data_dictionary_rows()).query("table == 'signal_ticks'").reset_index(drop=True))
    assert fs_file == fs_code


def test_assert_feature_set_clean_catches_leaks(dd):
    for bad in (["phase", "y_mp_action"], ["phase", "t"], ["phase", "tick_idx"], ["phase", "action_requested"],
                ["phase", "switched"], ["phase", "junction_id"], ["phase", "is_tail"], ["phase", "action_applied"]):
        with pytest.raises(AssertionError):
            ml.assert_feature_set_clean(bad, "truth", dd)
    with pytest.raises(AssertionError):
        ml.assert_feature_set_clean(["phase", "q_N"], "sensor", dd)          # truth queue in the sensor set
    with pytest.raises(AssertionError):
        ml.assert_feature_set_clean(["phase", "not_a_column"], "truth", dd)
    ml.assert_feature_set_clean(["phase", "obs_q_N", "down_n_E"], "sensor", dd)   # fine


# ---------------------------------------------------------------- splits
def _episodes_table():
    path = os.path.join(DATA, "episodes.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    rows = []
    for i in range(40):
        level = ["low", "design", "high", "stress"][i % 4]
        ctrl = ["max_pressure", "fixed_time", "random_dwell", "max_pressure_eps"][(i // 4) % 4]
        split = "test_ood" if level == "stress" else ("train" if ctrl == "random_dwell" else ["train", "val", "test"][i % 3])
        rows.append(dict(episode_id=i, demand_level=level, controller=ctrl, split=split))
    return pd.DataFrame(rows)


def test_split_integrity():
    ep = _episodes_table()
    counts = ml.check_splits(ep)
    assert ep.episode_id.is_unique
    assert set(counts) == {"train", "val", "test", "test_ood"} and sum(counts.values()) == len(ep)
    assert set(ep.loc[ep.demand_level == "stress", "split"]) == {"test_ood"}
    assert set(ep.loc[ep.split == "test_ood", "demand_level"]) == {"stress"}
    assert set(ep.loc[ep.controller == "random_dwell", "split"]) <= {"train"}
    # the guard really fires
    bad = ep.copy()
    bad.loc[bad.demand_level == "stress", "split"] = "test"
    with pytest.raises(AssertionError):
        ml.check_splits(bad)
    bad = pd.concat([ep, ep.iloc[[0]].assign(split="val")])
    with pytest.raises(AssertionError):
        ml.check_splits(bad)


# ---------------------------------------------------------------- the controller
def test_ml_controller_valid_phase_on_row_without_ambulances(tiny_bundle):
    cfg = SimConfig()
    c = evaluate.MLController(cfg, bundle=tiny_bundle)
    for phase in range(4):
        for tip in (5.0, 10.0, 60.0):
            out = c.act(_row(phase=phase, tip=tip))
            assert isinstance(out, int) and out in (0, 1, 2, 3)
    assert c.stats["n_main"] == 12
    row = _row()
    row["is_warmup"] = True
    c.act(row)
    assert c.stats["n_main"] == 12                      # warm-up rows are not counted


def test_ml_controller_hysteresis_rule(tiny_bundle):
    c = evaluate.MLController(SimConfig(), bundle=tiny_bundle, margin=0.10)
    c.probabilities = lambda row: np.array([0.50, 0.30, 0.20, 0.0])
    assert c.act(_row(phase=1)) == 0                    # 0.50 > 0.30 + 0.10 -> switch to the argmax
    c.probabilities = lambda row: np.array([0.38, 0.30, 0.32, 0.0])
    assert c.act(_row(phase=1)) == 1                    # 0.38 <= 0.40 -> hold
    c.probabilities = lambda row: np.array([0.45, 0.45, 0.10, 0.0])
    assert c.act(_row(phase=1)) == 1                    # tie -> lowest index is 0 but no margin -> hold
    c.probabilities = lambda row: np.array([0.0, 0.0, 0.0, 1.0])
    assert c.act(_row(phase=3)) == 3


def test_ml_controller_handles_model_missing_a_class(dd):
    from sklearn.linear_model import LogisticRegression
    cols = ml.feature_sets(dd)["sensor"]
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, len(cols)))
    y = rng.integers(0, 3, size=200)                    # class 3 never seen
    model = LogisticRegression(max_iter=200).fit(X, y)
    c = evaluate.MLController(SimConfig(), bundle={"model": model, "features": cols})
    p = c.probabilities(_row(phase=3))
    assert p.shape == (4,) and p[3] == 0.0 and abs(p.sum() - 1) < 1e-6
    assert c.act(_row(phase=3)) in (0, 1, 2, 3)


def test_registration_makes_world_find_the_controller_and_its_alias(tiny_bundle):
    evaluate.register_ml_controller(tiny_bundle, margin=0.1)
    try:
        c = make_controller("ml_sensor", SimConfig(), None)
        assert isinstance(c, evaluate.MLController) and c.margin == 0.1
        alias = make_controller("ml_v1", SimConfig(), None)            # DECISIONS §10 name
        assert isinstance(alias, evaluate.MLController) and alias.features == c.features
        assert "ml_v1" in evaluate.ML_NAMES and "ml_sensor" in evaluate.ML_NAMES
    finally:
        evaluate.unregister_ml_controller()
    assert "ml_sensor" not in CONTROLLERS and "ml_v1" not in CONTROLLERS


def test_thread_env_is_pinned_for_serial_inference():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ.get(var) is not None


# ---------------------------------------------------------------- the acceptance bound and the secondary metrics
def _episode_table(ml_delays, mp_delays, demand="design", noise=0.0, ev_rate="default", start_seed=1):
    rows = []
    for i, (a, b) in enumerate(zip(ml_delays, mp_delays)):
        seed = start_seed + i
        for ctrl, d in (("ml_sensor", a), ("max_pressure", b)):
            rows.append(dict(controller=ctrl, demand_level=demand, seed=seed, sensor_noise_level=noise, ev_rate=ev_rate,
                             mean_total_delay_s=d))
    return pd.DataFrame(rows)


def test_acceptance_bound_pass_rate_and_t_interval():
    mp = [100.0] * 8
    ml_d = [105.0, 108.0, 109.0, 112.0, 95.0, 110.0, 101.0, 99.0]          # 7 of 8 within 1.10 x (112 fails)
    df = _episode_table(ml_d, mp)
    b = evaluate.acceptance_bound(df)
    assert len(b) == 1
    r = b.iloc[0]
    ratios = np.array(ml_d) / 100.0
    assert r.n_pairs == 8 and r.n_pass == 7 and abs(r.pass_rate - 7 / 8) < 1e-12
    assert abs(r.mean_ratio - ratios.mean()) < 1e-12
    half = 2.365 * ratios.std(ddof=1) / math.sqrt(8)                     # t_{7,0.975}
    assert abs(r.ci95_half - half) < 1e-9 and abs(r.ci95_low - (ratios.mean() - half)) < 1e-9
    assert abs(r.max_ratio - 1.12) < 1e-12 and abs(r.min_ratio - 0.95) < 1e-12
    assert not r.bound_met_all_seeds and bool(r.bound_met_ci) == (r.ci95_high <= 1.10)
    assert abs(r.mean_ml - np.mean(ml_d)) < 1e-9 and abs(r.mean_reference - 100.0) < 1e-12
    # methodology form: CI upper end of the paired difference against 10% of the reference mean
    d = np.array(ml_d) - 100.0
    assert abs(r.diff_mean_s - d.mean()) < 1e-9 and abs(r.margin_s - 10.0) < 1e-9
    assert abs(r.diff_ci95_high_s - (d.mean() + 2.365 * d.std(ddof=1) / math.sqrt(8))) < 1e-9
    assert bool(r.bound_met_methodology) == (r.diff_ci95_high_s <= 10.0)
    # every seed passes -> the flags say so; the factor is a parameter
    b2 = evaluate.acceptance_bound(_episode_table([105.0] * 8, mp))
    assert bool(b2.iloc[0].bound_met_all_seeds) and b2.iloc[0].pass_rate == 1.0 and b2.iloc[0].sd_ratio == 0.0
    b3 = evaluate.acceptance_bound(_episode_table([105.0] * 8, mp), factor=1.02)
    assert b3.iloc[0].n_pass == 0 and b3.iloc[0].pass_rate == 0.0


def test_acceptance_bound_is_paired_by_seed_and_grouped_by_demand_and_noise():
    design = _episode_table([110.0] * 4, [100.0] * 4, demand="design")
    high = _episode_table([130.0] * 4, [100.0] * 4, demand="high")
    noisy = _episode_table([120.0] * 4, [100.0] * 4, demand="design", noise=0.1)
    off = _episode_table([999.0] * 4, [1.0] * 4, demand="design", ev_rate="0")   # ambulance-off rows must be ignored
    # shuffle the rows: pairing must come from the seed, not the row order
    df = pd.concat([design, high, noisy, off]).sample(frac=1.0, random_state=3).reset_index(drop=True)
    b = evaluate.acceptance_bound(df).set_index(["sensor_noise_level", "demand_level"])
    assert abs(b.loc[(0.0, "design")].mean_ratio - 1.10) < 1e-12 and b.loc[(0.0, "design")].n_pass == 4
    assert abs(b.loc[(0.0, "high")].mean_ratio - 1.30) < 1e-12 and b.loc[(0.0, "high")].n_pass == 0
    assert abs(b.loc[(0.0, "all")].mean_ratio - 1.20) < 1e-12 and b.loc[(0.0, "all")].n_pairs == 8 and b.loc[(0.0, "all")].pass_rate == 0.5
    assert abs(b.loc[(0.1, "design")].mean_ratio - 1.20) < 1e-12 and (0.1, "all") not in b.index
    # a seed without a reference partner is dropped, not mismatched
    df2 = pd.concat([design, _episode_table([200.0], [100.0], start_seed=50).query("controller == 'ml_sensor'")])
    assert evaluate.acceptance_bound(df2).iloc[0].n_pairs == 4


def test_attach_ev_off_penalty_pairs_same_controller_demand_seed():
    on = _episode_table([120.0, 130.0], [100.0, 110.0])
    off = _episode_table([115.0, 100.0], [90.0, 111.0], ev_rate="0")
    df = evaluate.attach_ev_off_penalty(pd.concat([off, on]).reset_index(drop=True))
    on_rows = df[df.ev_rate == "default"].set_index(["controller", "seed"]).car_mean_delay_penalty_s
    assert on_rows[("ml_sensor", 1)] == 5.0 and on_rows[("ml_sensor", 2)] == 30.0
    assert on_rows[("max_pressure", 1)] == 10.0 and on_rows[("max_pressure", 2)] == -1.0
    assert df[df.ev_rate == "0"].car_mean_delay_penalty_s.isna().all()
    lonely = evaluate.attach_ev_off_penalty(on)                           # no pair -> NaN, no crash
    assert lonely.car_mean_delay_penalty_s.isna().all()


def test_fairness_ratio_over_entry_nodes():
    cfg = SimConfig(**SHORT)
    t_main = cfg.warmup_s + 10.0
    rows = []
    for origin, delays in (("A", [10.0, 20.0]), ("B", [30.0]), ("C", [60.0, 60.0])):
        for d in delays:
            rows.append(dict(vehicle_type="car", spawn_request_t=t_main, origin=origin, total_delay_s=d))
    rows.append(dict(vehicle_type="ambulance", spawn_request_t=t_main, origin="C", total_delay_s=1000.0))   # ignored
    rows.append(dict(vehicle_type="car", spawn_request_t=1.0, origin="D", total_delay_s=0.0))              # warm-up, ignored
    ratio, n, worst, best = evaluate.fairness_ratio(rows, cfg)
    assert abs(ratio - 60.0 / 15.0) < 1e-12 and n == 3 and worst == "C" and best == "A"
    assert math.isnan(evaluate.fairness_ratio(rows[:2], cfg)[0])          # one origin only


def test_t_interval_and_t_critical():
    assert evaluate.t_critical(8) == 2.365
    assert abs(evaluate.t_critical(16) - 2.131) < 0.002
    ti = evaluate.t_interval([1.0, 2.0, 3.0, 4.0, float("nan")])
    assert ti["n"] == 4 and ti["mean"] == 2.5 and abs(ti["ci95_half"] - 3.182 * np.std([1, 2, 3, 4], ddof=1) / 2) < 1e-3


def test_simulator_matches_shipped_dataset_hash():
    if not os.path.exists(os.path.join(DATA, "episodes.csv")):
        pytest.skip("data/traffic_v1 not generated")
    cfg = evaluate.load_cfg(DATA)
    from traffic_twin.config import GENERATOR_VERSION
    shipped = pd.read_csv(os.path.join(DATA, "episodes.csv"), nrows=1)["generator_version"].iloc[0]
    if shipped != GENERATOR_VERSION:
        pytest.skip(f"{DATA} was generated by {shipped}, the simulator is {GENERATOR_VERSION}: regenerate to re-enable")
    ident = evaluate.check_simulator_matches_dataset(cfg, DATA, strict=True)
    assert ident["hash_match"] and ident["simulator_config_hash"] == ident["dataset_config_hash"]
    with pytest.raises(AssertionError):
        evaluate.check_simulator_matches_dataset(cfg.copy(min_green=6.0), DATA, strict=True)


# ---------------------------------------------------------------- smoke run of the protocol
def test_closed_loop_smoke_two_episodes(tiny_bundle, tmp_path):
    cfg = SimConfig(**SHORT, lambda_base_vph={"design": 300.0}, lambda_ev_vph=90.0)
    controllers = ("max_pressure", "ml_sensor", "priority_mp")
    df = evaluate.run_protocol(cfg, bundle=tiny_bundle, controllers=controllers, demands=("design",), seeds=(1, 2),
                               workers=1, paired_ev_off=True, log=lambda *a, **k: None)
    assert len(df) == 12 and set(df.controller) == set(controllers) and set(df.seed) == {1, 2}
    assert set(df.ev_rate.astype(str)) == {"default", "0"}
    on = df[df.ev_rate.astype(str) == "default"]
    assert len(on) == 6 and (df.n_invariant_violations == 0).all()
    assert on.n_ambulances.gt(0).all() and (df[df.ev_rate.astype(str) == "0"].n_ambulances == 0).all()
    assert on.ev_mean_delay_s.notna().all() and on.car_mean_delay_penalty_s.notna().all()
    assert on.fairness_ratio.notna().all() and (on.fairness_ratio >= 1).all() and (on.fairness_n_origins == 8).all()
    assert on.agreement_with_teacher.between(0, 1).all() and on.request_switch_rate.between(0, 1).all()
    assert (on.loc[on.controller == "priority_mp", "agreement_with_teacher"] == 1.0).all()
    # common random numbers: the same seed gives the same number of car spawn requests under every controller and ev_rate
    cars = df.n_spawned - df.n_ambulances
    assert cars.groupby(df.seed).nunique().eq(1).all()
    # the penalty is exactly on - off for the same (controller, seed)
    off = df[df.ev_rate.astype(str) == "0"].set_index(["controller", "seed"]).mean_total_delay_s
    for _, r in on.iterrows():
        assert abs(r.car_mean_delay_penalty_s - (r.mean_total_delay_s - off[(r.controller, r.seed)])) < 1e-6
    diffs = evaluate.paired_differences(df)
    assert set(diffs.controller) == {"ml_sensor", "priority_mp"}
    assert (diffs.n_pairs == 2).all() and diffs[diffs.metric == "mean_total_delay_s"].mean_diff.notna().all()
    assert set(evaluate.SECONDARY) <= set(diffs.metric)
    bound = evaluate.acceptance_bound(df)
    assert len(bound) == 1 and bound.iloc[0].n_pairs == 2
    evaluate.plot_closed_loop(df, diffs, str(tmp_path / "closed_loop.png"), controllers)
    evaluate.plot_acceptance_bound(df, bound, str(tmp_path / "closed_loop_bound.png"))
    meta = dict(n_seeds=2, seeds=[1, 2], lambda_ev_vph=90.0, sensor_noise_level=0.0, wall_s=1.0, workers=1,
                model_path="tiny.joblib", margin=0.1)
    section = evaluate.closed_loop_markdown(df, diffs, meta, controllers)
    for needle in ("## 4. Closed loop", "acceptance bound", "ml_sensor", "fairness_ratio", "## 5. Verdict", "## 6. Reproducibility"):
        assert needle in section, needle
    evaluate.update_results_md(str(tmp_path / "RESULTS.md"), section)
    evaluate.update_results_md(str(tmp_path / "RESULTS.md"), section)         # idempotent replace
    text = (tmp_path / "RESULTS.md").read_text()
    assert text.count("## 4. Closed loop") == 1 and text.count("## 5. Verdict") == 1
    summary = evaluate.summary_table(df, controllers)
    assert len(summary) == 3 and set(summary.controller) == set(controllers)
    evaluate.unregister_ml_controller()
