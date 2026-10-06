"""Supervised learning on the Fatmawati sensor release (methodology Section 13).

Three tasks, each against its baselines, on splits that keep whole episodes and seeds together:
  A  imitate the priority-aware teacher: label y_teacher_stage (S1..S4), weighted multinomial log-loss
  B  forecast each approach's standing vehicles 30 s ahead: labels y_q_veh_30s_<leg>, Poisson loss
  C  ambulance arrival: seconds until a detected ambulance crosses the stop line (from ambulance_trips),
     quantile loss with tau < 0.5 so that the model errs early
Validators run first: schema and spacing, conservation at the gates, label recomputation (forecast label and
teacher rule), and a leakage audit; the shuffled-label test runs with Task A.

load_policy(path) turns a saved Task A model into a controller: policy(row) -> "S1".."S4", where row is the
dict of ticks_5s columns that the twin builds at a decision instant.

  python3 -m traffic_twin.fatmawati.learn validate --data data/fatmawati_pilot
  python3 -m traffic_twin.fatmawati.learn train --data data/fatmawati_v3 --out reports/a3/learning --jobs 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

LEGS = ("N", "E", "S", "W")
STAGES = ("S1", "S2", "S3", "S4")
STAGE_OF_APPROACH = {"W": "S1", "E": "S2", "N": "S3", "S": "S4"}
INTERGREEN_S, START_LOSS_S, DECISION_S, MP_MARGIN_PCU = 4.0, 2.0, 5.0, 1.0
SAT_FLOW_DEFAULT = {"N": 5850.0, "E": 7800.0, "S": 3900.0, "W": 3900.0}     # PKJI 2023: S0 = 600 W_e pcu/h
NO_AMB_ETA_S, NO_AMB_DIST_M = 600.0, 400.0                                  # stand-ins when no ambulance is seen
EXCLUDE = {"episode_id", "t", "clock", "controller", "controller_request"}
CATEGORICAL = ("stage", "phase_state", "transition_to", "green_arm", "phase_arm")


# ---------------------------------------------------------------------------------------------------- data
def _sat_flow() -> dict:
    try:
        from .sensors import sensor_layout
        return {c["leg"]: float(c["sat_flow_pcu_h"]) for c in sensor_layout()["cameras"] if c["kind"] == "queue_camera"}
    except Exception:                                                        # stand-alone use
        return dict(SAT_FLOW_DEFAULT)


def load_release(data: str):
    dd = pd.read_csv(os.path.join(data, "data_dictionary.csv"))
    ticks = pd.read_csv(os.path.join(data, "ticks_5s.csv"), low_memory=False)
    eps = pd.read_csv(os.path.join(data, "episodes.csv"))
    keep = [c for c in ("episode_id", "seed", "window", "level") if c in eps.columns]
    ticks = ticks.merge(eps[keep], on="episode_id", how="left", suffixes=("", "_ep"))
    return dd, ticks, eps


def label_column(ticks: pd.DataFrame) -> str:
    return "y_teacher_stage" if "y_teacher_stage" in ticks.columns else "y_teacher_arm"


def feature_columns(dd: pd.DataFrame, ticks: pd.DataFrame) -> tuple[list, list]:
    feats = dd[(dd["table"] == "ticks_5s") & (dd["role"] == "feature")]["column"].tolist()
    feats = [c for c in feats if c in ticks.columns and c not in EXCLUDE and not c.startswith("y_")]
    cats = [c for c in feats if c in CATEGORICAL or ticks[c].dtype == object]
    return feats, cats


def _truthy(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(float)
    return s.map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else
                 float(str(v).strip().lower() in ("1", "1.0", "true", "yes"))).astype(float)


def prepare_X(df: pd.DataFrame, feats: list, cats: list) -> pd.DataFrame:
    """Same preparation in training and in the live policy: no-ambulance stand-ins, numeric types, categories."""
    X = df.reindex(columns=feats).copy()
    for c in feats:
        if c in cats:
            X[c] = X[c].astype(object).where(X[c].notna(), "none").astype(str)
            continue
        if c.startswith("ev_needs_signal_") or (c.startswith("ev_") and c[3:] in LEGS):
            X[c] = _truthy(X[c]).fillna(0.0)
        elif c.startswith("ev_eta_s_"):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(NO_AMB_ETA_S)
        elif c.startswith("ev_dist_m_"):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(NO_AMB_DIST_M)
        elif c.startswith("ev_q_ahead_pcu_"):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def ambulance_mask(df: pd.DataFrame) -> np.ndarray:
    m = np.zeros(len(df), dtype=bool)
    for g in LEGS:
        if f"ev_{g}" in df.columns:
            on = _truthy(df[f"ev_{g}"]).fillna(0.0).values > 0
            if f"ev_needs_signal_{g}" in df.columns:
                on &= _truthy(df[f"ev_needs_signal_{g}"]).fillna(0.0).values > 0
            m |= on
    return m


def split(ticks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """Test = the highest seed (seed 3 in the release); with a single seed, the last episode (pilot smoke runs)."""
    if "seed" in ticks.columns and ticks["seed"].nunique() >= 2:
        s = ticks["seed"].max()
        return (ticks["seed"] != s).values, (ticks["seed"] == s).values, f"train seeds != {s}, test seed {s}"
    last = ticks["episode_id"].max()
    return (ticks["episode_id"] != last).values, (ticks["episode_id"] == last).values, f"test episode {last}"


# ---------------------------------------------------------------------------------------------------- validators
def teacher_stage(row: pd.Series, sat: dict) -> str:
    best = None
    for g in LEGS:
        eta = row.get(f"ev_eta_s_{g}")
        if eta is None or pd.isna(eta) or not (row.get(f"ev_needs_signal_{g}") in (1, 1.0, True, "True", "true", "1")):
            continue
        q = row.get(f"ev_q_ahead_pcu_{g}")
        q = 0.0 if q is None or pd.isna(q) else float(q)
        need = INTERGREEN_S + START_LOSS_S + 3600.0 / sat[g] * q + DECISION_S
        if float(eta) <= need and (best is None or float(eta) < best[0]):
            best = (float(eta), STAGE_OF_APPROACH[g])
    if best:
        return best[1]
    p = {st: float(row[f"stage_pressure_{st}"]) for st in STAGES}
    cur = row["stage"]
    top = max(STAGES, key=lambda st: (p[st], -STAGES.index(st)))
    return top if p[top] > p[cur] + MP_MARGIN_PCU else cur


def validate(data: str) -> dict:
    dd, ticks, eps = load_release(data)
    rep = dict(data=data, rows=int(len(ticks)), episodes=int(ticks["episode_id"].nunique()))
    # schema: every dictionary column exists in its table, ticks are 5 s apart
    missing = {}
    for tab, g in dd.groupby("table"):
        path = os.path.join(data, f"{tab}.csv")
        if os.path.exists(path):
            cols = set(pd.read_csv(path, nrows=1).columns)
            miss = sorted(set(g["column"]) - cols)
            if miss:
                missing[tab] = miss
    rep["schema_missing_columns"] = missing
    gaps = ticks.sort_values(["episode_id", "t"]).groupby("episode_id")["t"].diff().dropna()
    rep["ticks_5s_apart"] = bool((gaps.round(6) == 5.0).all())
    # conservation at the gates and the engine's own check
    loops = pd.read_csv(os.path.join(data, "loops_1min.csv"))
    gate = loops[loops["kind"].isin(["entry_gate", "exit_gate"])].groupby(["episode_id", "kind"])["n_total"].sum().unstack()
    exits_total = None
    if {"entry_gate", "exit_gate"} <= set(gate.columns):
        gate["imbalance"] = (gate["entry_gate"] - gate["exit_gate"]) / gate["entry_gate"].clip(lower=1)
        rep["gate_imbalance_max"] = round(float(gate["imbalance"].abs().max()), 4)
        exits_total = int(gate["exit_gate"].sum())
    qc_path = os.path.join(data, "simulation_qc.csv")
    if os.path.exists(qc_path):
        qc = pd.read_csv(qc_path)
        rep["unmatched_exits_total"] = int(qc.get("unmatched_exits", pd.Series([0])).sum())
        # an exit with no matching entry: a vehicle teleported across a gate; judged against all exits counted
        rep["unmatched_exit_share"] = round(rep["unmatched_exits_total"] / exits_total, 6) if exits_total else None
        rep["lane_collisions_total"] = int(qc.get("lane_collisions", pd.Series([0])).sum())
    # forecast label: the camera reading six ticks later in the same episode
    bad = checked = 0
    for _, g in ticks.sort_values(["episode_id", "t"]).groupby("episode_id"):
        for a in LEGS:
            fut, lab = g[f"q_veh_{a}"].shift(-6), g[f"y_q_veh_30s_{a}"]
            m = fut.notna() & lab.notna()
            checked += int(m.sum())
            bad += int((lab[m] != fut[m]).sum())
    rep["forecast_label_checked"], rep["forecast_label_mismatch"] = checked, bad
    # teacher label: recompute the rule from the logged columns
    lab = label_column(ticks)
    if lab == "y_teacher_stage":
        sat = _sat_flow()
        rec = ticks.apply(lambda r: teacher_stage(r, sat), axis=1)
        agree = rec == ticks[lab]
        rep["teacher_label_agreement"] = round(float(agree.mean()), 5)
        rep["teacher_label_mismatches"] = int((~agree).sum())
        amb = ambulance_mask(ticks)
        rep["teacher_label_mismatches_with_ambulance"] = int((~agree & amb).sum())
        # the simulator decides on unrounded pressures but logs them to 0.01 pcu, so a tie or an exact 1-pcu margin in
        # the logged values can go either way; such boundary ticks are reported, and only the rest count as errors
        tol = 0.0101                                     # two values rounded to 0.01 differ from the truth by <= 0.01

        def boundary(r) -> bool:
            p = {st: float(r[f"stage_pressure_{st}"]) for st in STAGES}
            cur, got, want = r["stage"], rec[r.name], r[lab]
            if want == cur:                              # the simulator kept the stage where the recomputed rule switches
                return p[got] - p[cur] - MP_MARGIN_PCU <= tol
            return p[want] >= max(p.values()) - tol and p[want] - p[cur] - MP_MARGIN_PCU >= -tol
        bnd = ticks.loc[~agree].apply(boundary, axis=1) if (~agree).any() else pd.Series(dtype=bool)
        rep["teacher_label_boundary_ties"] = int(bnd.sum())
        rep["teacher_label_unexplained"] = int((~bnd).sum())
        if (~agree).any():
            cols = ["episode_id", "t", "stage", lab] + [f"stage_pressure_{s}" for s in STAGES]
            rep["teacher_label_examples"] = ticks.loc[~agree, cols].head(5).astype(str).to_dict("records")
    # leakage: no feature from the future, the running controller's request excluded
    feats, cats = feature_columns(dd, ticks)
    rep["features"] = len(feats)
    rep["label_like_features"] = [c for c in feats if c.startswith("y_") or "request" in c]
    if "controller_request" in ticks.columns:
        rep["controller_request_equals_label"] = {c: round(float((g["controller_request"] == g[lab]).mean()), 3)
                                                  for c, g in ticks.groupby("controller")}
    # lane collisions are a simulator artefact judged by the KPI's relative safety gate, so they are reported, not failed here
    rep["passed"] = bool(not missing and rep["ticks_5s_apart"] and bad == 0
                         and rep.get("teacher_label_unexplained", 0) == 0 and not rep["label_like_features"]
                         and (rep.get("unmatched_exit_share") or 0.0) <= 0.005)
    return rep


# ---------------------------------------------------------------------------------------------------- models
def _pre(feats: list, cats: list, scale: bool, ordinal: bool = False):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    nums = [c for c in feats if c not in cats]
    if ordinal:
        return ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
                                  ("num", "passthrough", nums)])
    num = [("imp", SimpleImputer(strategy="median", add_indicator=True))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats), ("num", Pipeline(num), nums)])


def classifiers(feats: list, cats: list, jobs: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    catmask = [True] * len(cats) + [False] * (len(feats) - len(cats))
    return {
        "logistic_regression": (Pipeline([("pre", _pre(feats, cats, True)), ("m", LogisticRegression(max_iter=3000))]),
                                {"m__C": [0.1, 1.0]}),
        "decision_tree": (Pipeline([("pre", _pre(feats, cats, False)), ("m", DecisionTreeClassifier(min_samples_leaf=20, random_state=0))]),
                          {"m__max_depth": [4, 6, 8]}),
        "random_forest": (Pipeline([("pre", _pre(feats, cats, False)),
                                    ("m", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=jobs, random_state=0))]),
                          {"m__max_depth": [None, 12]}),
        "gradient_boosting": (Pipeline([("pre", _pre(feats, cats, False, ordinal=True)),
                                        ("m", HistGradientBoostingClassifier(max_iter=300, l2_regularization=1.0, early_stopping=False,
                                                                             categorical_features=catmask, random_state=0))]),
                              {"m__learning_rate": [0.05, 0.1], "m__max_leaf_nodes": [15, 31]}),
        "neural_network": (Pipeline([("pre", _pre(feats, cats, True)),
                                     ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=0))]),
                           {"m__alpha": [1e-4, 1e-3]}),
    }


def class_weights(y: np.ndarray, amb: np.ndarray, w_amb: float) -> np.ndarray:
    cls, cnt = np.unique(y, return_counts=True)
    per = {c: len(y) / (len(cls) * n) for c, n in zip(cls, cnt)}
    return np.array([per[v] for v in y]) * np.where(amb, w_amb, 1.0)


def _fit(model, X, y, w):
    try:
        model.fit(X, y, m__sample_weight=w)
    except TypeError:                                                       # an estimator without sample weights
        model.fit(X, y)
    return model


def task_a_scores(y, pred, cur, amb) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    sw = y != cur
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average="macro", zero_division=0),
                switch_accuracy=float((pred[sw] == y[sw]).mean()) if sw.any() else np.nan,
                ambulance_accuracy=float((pred[amb] == y[amb]).mean()) if amb.any() else np.nan,
                n=int(len(y)), n_switch=int(sw.sum()), n_ambulance=int(amb.sum()))


def task_a(ticks, feats, cats, out, jobs, rng) -> dict:
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import GroupKFold, ParameterGrid
    import joblib
    lab = label_column(ticks)
    tr, te, how = split(ticks)
    X = prepare_X(ticks, feats, cats)
    y = ticks[lab].astype(str).values
    cur = ticks["stage"].astype(str).values if "stage" in ticks.columns else ticks["green_arm"].astype(str).values
    amb = ambulance_mask(ticks)
    groups = ticks["episode_id"].values
    res = dict(split=how, label=lab, n_train=int(tr.sum()), n_test=int(te.sum()), models={}, baselines={})
    # baselines on the test split
    classes = np.array(sorted(set(y[tr])))
    prior = pd.Series(y[tr]).value_counts(normalize=True).reindex(classes).values
    res["baselines"]["random_uniform"] = task_a_scores(y[te], rng.choice(classes, te.sum()), cur[te], amb[te])
    res["baselines"]["random_by_frequency"] = task_a_scores(y[te], rng.choice(classes, te.sum(), p=prior), cur[te], amb[te])
    res["baselines"]["majority"] = task_a_scores(y[te], np.full(te.sum(), classes[np.argmax(prior)]), cur[te], amb[te])
    res["baselines"]["stillness"] = task_a_scores(y[te], cur[te], cur[te], amb[te])
    # grouped cross-validation within the training split
    n_groups = len(np.unique(groups[tr]))
    cv = GroupKFold(n_splits=min(3, n_groups)) if n_groups >= 2 else None
    models = classifiers(feats, cats, jobs)

    def cv_score(est, params, w_amb):
        if cv is None:
            return np.nan
        s = []
        for a, b in cv.split(X[tr], y[tr], groups[tr]):
            m = clone(est).set_params(**params)
            _fit(m, X[tr].iloc[a], y[tr][a], class_weights(y[tr][a], amb[tr][a], w_amb))
            s.append(f1_score(y[tr][b], m.predict(X[tr].iloc[b]), average="macro", zero_division=0))
        return float(np.mean(s))

    # the extra weight on ambulance ticks is tuned once, on gradient boosting
    est, grid = models["gradient_boosting"]
    p0 = list(ParameterGrid(grid))[0]
    w_scores = {w: cv_score(est, p0, w) for w in (1.0, 5.0, 20.0)}
    w_amb = max(w_scores, key=lambda w: (np.nan_to_num(w_scores[w], nan=-1), -w)) if cv is not None else 5.0
    res["ambulance_weight"], res["ambulance_weight_cv"] = w_amb, {str(k): v for k, v in w_scores.items()}
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    for name, (est, grid) in models.items():
        t0 = time.perf_counter()
        scored = [(cv_score(est, p, w_amb), p) for p in ParameterGrid(grid)] if cv is not None else [(np.nan, list(ParameterGrid(grid))[0])]
        best_cv, best_p = max(scored, key=lambda sp: np.nan_to_num(sp[0], nan=-1))
        m = _fit(clone(est).set_params(**best_p), X[tr], y[tr], class_weights(y[tr], amb[tr], w_amb))
        pred = m.predict(X[te])
        sc = task_a_scores(y[te], pred, cur[te], amb[te])
        sc.update(cv_macro_f1=best_cv, params={k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in best_p.items()},
                  fit_s=round(time.perf_counter() - t0, 1))
        sc["by_controller"] = {}
        if "controller" in ticks.columns:
            for c in sorted(ticks.loc[te, "controller"].unique()):
                mk = (ticks.loc[te, "controller"] == c).values
                sc["by_controller"][c] = task_a_scores(y[te][mk], pred[mk], cur[te][mk], amb[te][mk])
        res["models"][name] = sc
        labs = sorted(set(y))
        cm = confusion_matrix(y[te], pred, labels=labs)
        pd.DataFrame(cm, index=[f"true_{s}" for s in labs], columns=[f"pred_{s}" for s in labs]).to_csv(
            os.path.join(out, f"taskA_confusion_{name}.csv"))
        joblib.dump(dict(model=m, feats=feats, cats=cats, label=lab, params=best_p, ambulance_weight=w_amb,
                         trained_on=how), os.path.join(out, "models", f"taskA_{name}.joblib"))
        print(f"  task A {name}: macro-F1 {sc['macro_f1']:.3f}, ambulance acc {sc['ambulance_accuracy']:.3f} ({sc['fit_s']} s)", flush=True)
    # shuffled-label test on the gradient-boosting settings
    est, _ = models["gradient_boosting"]
    gb = res["models"]["gradient_boosting"]["params"]
    ysh = rng.permutation(y[tr])
    msh = _fit(clone(est).set_params(**gb), X[tr], ysh, class_weights(ysh, amb[tr], w_amb))
    res["shuffled_label_test"] = task_a_scores(y[te], msh.predict(X[te]), cur[te], amb[te])
    m = joblib.load(os.path.join(out, "models", "taskA_gradient_boosting.joblib"))["model"]
    # two harder hold-outs on the same settings: a time window and a demand level never seen in training, against the
    # main model (which saw them on the training seeds) on the same test rows
    res["generalisation"] = {}
    for col, val in (("window", "pm_peak"), ("level", "high")):
        if col not in ticks.columns:
            continue
        held = (ticks[col] == val).values
        trg, teg = tr & ~held, te & held
        if trg.sum() == 0 or teg.sum() == 0 or len(set(y[trg])) < 2:
            continue
        mg = _fit(clone(est).set_params(**gb), X[trg], y[trg], class_weights(y[trg], amb[trg], w_amb))
        res["generalisation"][f"{col}={val}"] = dict(unseen=task_a_scores(y[teg], mg.predict(X[teg]), cur[teg], amb[teg]),
                                                     seen=task_a_scores(y[teg], m.predict(X[teg]), cur[teg], amb[teg]))
    # what the gradient-boosting model relies on
    sub = np.where(te)[0]
    if len(sub) > 4000:
        sub = rng.choice(sub, 4000, replace=False)
    pi = permutation_importance(m, X.iloc[sub], y[sub], scoring="f1_macro", n_repeats=3, random_state=0, n_jobs=jobs)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    imp.to_csv(os.path.join(out, "taskA_importance.csv"), header=["macro_f1_drop"])
    res["top_features"] = {k: round(float(v), 4) for k, v in imp.head(10).items()}
    return res


def task_b(ticks, feats, cats, out, jobs) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeRegressor
    tr, te, how = split(ticks)
    X = prepare_X(ticks, feats, cats)
    catmask = [True] * len(cats) + [False] * (len(feats) - len(cats))
    fam = {
        "ridge": lambda: Pipeline([("pre", _pre(feats, cats, True)), ("m", Ridge(alpha=1.0))]),
        "decision_tree": lambda: Pipeline([("pre", _pre(feats, cats, False)), ("m", DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=0))]),
        "random_forest": lambda: Pipeline([("pre", _pre(feats, cats, False)), ("m", RandomForestRegressor(n_estimators=150, min_samples_leaf=5, n_jobs=jobs, random_state=0))]),
        "gradient_boosting": lambda: Pipeline([("pre", _pre(feats, cats, False, ordinal=True)),
                                               ("m", HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.1,
                                                                                   categorical_features=catmask, random_state=0))]),
        "neural_network": lambda: Pipeline([("pre", _pre(feats, cats, True)), ("m", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=0))]),
    }
    res = dict(split=how, legs={})
    for g in LEGS:
        lab = f"y_q_veh_30s_{g}"
        ok = ticks[lab].notna().values
        a, b = tr & ok, te & ok
        y = ticks[lab].values.astype(float)
        now = ticks[f"q_veh_{g}"].values.astype(float)
        rows = dict(persistence=float(np.mean(np.abs(y[b] - now[b]))), mean=float(np.mean(np.abs(y[b] - y[a].mean()))))
        rng = np.random.default_rng(0)
        rows["random"] = float(np.mean(np.abs(y[b] - rng.choice(y[a], b.sum()))))
        best_base = min(rows["persistence"], rows["mean"])
        models = {}
        for name, mk in fam.items():
            m = mk().fit(X[a], y[a])
            mae = float(np.mean(np.abs(y[b] - np.clip(m.predict(X[b]), 0, None))))
            models[name] = dict(mae=mae, skill=1 - mae / best_base)
        res["legs"][g] = dict(baselines=rows, best_baseline=best_base, models=models)
        print(f"  task B {g}: persistence {rows['persistence']:.2f}, mean {rows['mean']:.2f}, "
              + ", ".join(f"{k} {v['mae']:.2f}" for k, v in models.items()), flush=True)
    return res


def task_c(data, ticks, feats, cats, out) -> dict:
    """Seconds until a detected ambulance whose route needs the signal crosses that approach's stop line."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    trips = pd.read_csv(os.path.join(data, "ambulance_trips.csv"))
    trips = trips[trips["t_stopline"].notna() & trips["t_detect"].notna()]
    rows = []
    for _, tp in trips.iterrows():
        g = tp.get("stopline_leg")
        if g not in LEGS:
            continue
        sel = ticks[(ticks["episode_id"] == tp["episode_id"]) & (ticks["t"] >= tp["t_detect"]) & (ticks["t"] < tp["t_stopline"])]
        if f"ev_{g}" in sel.columns:
            sel = sel[_truthy(sel[f"ev_{g}"]).fillna(0).values > 0]
        if len(sel):
            s = sel.copy()
            s["leg"] = g
            s["y_arrival_s"] = tp["t_stopline"] - s["t"]
            s["own_eta_s"] = pd.to_numeric(s[f"ev_eta_s_{g}"], errors="coerce")
            rows.append(s)
    if not rows:
        return dict(skipped="no ambulance ticks with a stop-line crossing")
    d = pd.concat(rows, ignore_index=True)
    if len(d) < 200:
        return dict(skipped=f"only {len(d)} rows")
    fc, cc = feats + ["leg"], cats + ["leg"]
    tr, te, how = split(d)
    X, y = prepare_X(d, fc, cc), d["y_arrival_s"].values
    catmask = [True] * len(cc) + [False] * (len(fc) - len(cc))
    tau = 0.2
    def pinball(yt, yp, q=tau):
        r = yt - yp
        return float(np.mean(np.maximum(q * r, (q - 1) * r)))
    eta = d["own_eta_s"].values
    res = dict(split=how, n_train=int(tr.sum()), n_test=int(te.sum()), tau=tau,
               baselines=dict(teacher_eta=dict(mae=float(np.nanmean(np.abs(y[te] - eta[te]))), late_share=float(np.nanmean(eta[te] > y[te])),
                                               pinball=pinball(y[te], np.nan_to_num(eta[te], nan=float(np.nanmean(eta[tr]))))),
                              mean=dict(mae=float(np.mean(np.abs(y[te] - y[tr].mean()))))))
    for name, loss in (("gradient_boosting_quantile", "quantile"), ("gradient_boosting_absolute", "absolute_error")):
        kw = dict(quantile=tau) if loss == "quantile" else {}
        m = Pipeline([("pre", _pre(fc, cc, False, ordinal=True)),
                      ("m", HistGradientBoostingRegressor(loss=loss, max_iter=300, learning_rate=0.05, categorical_features=catmask,
                                                          random_state=0, **kw))]).fit(X[tr], y[tr])
        p = m.predict(X[te])
        res[name] = dict(mae=float(np.mean(np.abs(y[te] - p))), late_share=float(np.mean(p > y[te])), pinball=pinball(y[te], p))
    print(f"  task C: teacher ETA MAE {res['baselines']['teacher_eta']['mae']:.2f} s, "
          f"quantile model MAE {res['gradient_boosting_quantile']['mae']:.2f} s, late {res['gradient_boosting_quantile']['late_share']:.2f}", flush=True)
    return res


def train(data: str, out: str, jobs: int = 2) -> dict:
    warnings.filterwarnings("ignore")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261003)
    dd, ticks, eps = load_release(data)
    feats, cats = feature_columns(dd, ticks)
    res = dict(data=data, rows=int(len(ticks)), episodes=int(ticks["episode_id"].nunique()), features=feats, categorical=cats)
    print(f"training on {res['rows']} ticks from {res['episodes']} episodes, {len(feats)} features", flush=True)
    res["validation"] = validate(data)
    import sklearn.ensemble, sklearn.neural_network      # noqa: F401  load OpenMP and BLAS first: a limit binds loaded libraries only
    from threadpoolctl import threadpool_limits
    # one OpenMP/BLAS thread per estimator: beside a busy simulator, two threads made gradient boosting 30x slower
    # (each tree's parallel regions wait on a descheduled thread); the random forest still uses `jobs` workers
    with threadpool_limits(limits=1):
        res["task_a"] = task_a(ticks, feats, cats, out, jobs, rng)
        res["task_b"] = task_b(ticks, feats, cats, out, jobs)
        res["task_c"] = task_c(data, ticks, feats, cats, out)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return res


# ---------------------------------------------------------------------------------------------------- closed loop
def load_policy(path: str):
    """A saved Task A model as a controller: policy(row) -> stage, row being the ticks_5s columns at a decision."""
    import joblib
    b = joblib.load(path)
    model, feats, cats = b["model"], b["feats"], b["cats"]
    final = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(final, "n_jobs"):
        final.n_jobs = 1                             # a forest's worker threads cost more than one row's prediction

    from threadpoolctl import threadpool_limits

    def policy(row: dict) -> str:
        X = prepare_X(pd.DataFrame([{c: row.get(c) for c in feats}]), feats, cats)
        with threadpool_limits(limits=1):            # one-row predictions: thread start-up costs more than it saves
            return str(model.predict(X)[0])

    policy.feats, policy.label = feats, b.get("label")
    return policy


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("--data", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", required=True); t.add_argument("--out", required=True)
    t.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        print(json.dumps(validate(a.data), indent=1, default=str))
    else:
        train(a.data, a.out, a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
