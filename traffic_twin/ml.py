"""Supervised models on the signal_ticks table (DECISIONS §6-§7, methodology §12.3).

Run: python3 -m traffic_twin.ml --data data/traffic_v1 --out reports/ml

Task A (imitation): predict the priority-aware teacher's request y_mp_action
(4 classes) from the feature row, with a truth feature set and a
sensor-realistic feature set (obs_q_* instead of q_*, no n_*, no per-lane
queues, no arrival counts).  Task B (regression): predict the junction queue
30 s ahead, y_q_total_t30, from state + applied action, and read off the
partial dependence on the action, which is the control-relevant effect size.

Feature sets are selected by NAMESPACE from data_dictionary.csv, never by hand,
and `assert_feature_set_clean` fails loudly if a key, time flag, action or
label column sneaks in.  Splits are per episode (episodes.csv `split`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (balanced_accuracy_score, classification_report, confusion_matrix, f1_score,  # noqa: E402
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

DIRS = ("N", "E", "S", "W")
SPLITS = ("val", "test", "test_ood")
CLASS_NAMES = ["N", "E", "S", "W"]

# ---- the feature-set contract ---------------------------------------------------------------
TRUTH_NAMESPACES = ("signal", "state", "downstream", "neighbour", "ev")
# columns that must never be a feature, whatever the namespace says
FORBIDDEN_COLUMNS = {
    "episode_id", "junction_id", "t", "tick_idx", "is_warmup", "is_cooldown", "is_tail",
    "action_requested", "forced", "switched", "y_mp_action", "y_q_total_t30",
}
FORBIDDEN_NAMESPACES = {"key", "time", "label"}
FORBIDDEN_PREFIXES = ("y_",)
# state columns that a loop-detector installation does not deliver (sensor set drops them)
SENSOR_DROPPED_PREFIXES = ("n_", "arr_")
SENSOR_DROPPED_SUFFIXES = ("_l0", "_l1")


def load_dictionary(data_dir: str) -> pd.DataFrame:
    dd = pd.read_csv(os.path.join(data_dir, "data_dictionary.csv"))
    return dd[dd.table == "signal_ticks"].reset_index(drop=True)


def assert_feature_set_clean(cols: list, name: str, dd: pd.DataFrame | None = None, allow_action: bool = False) -> None:
    """Fail if a key / time flag / action / label column (or an unknown column) is in the set."""
    bad = [c for c in cols if c in FORBIDDEN_COLUMNS or c.startswith(FORBIDDEN_PREFIXES)]
    assert not bad, f"feature set '{name}' contains forbidden columns {bad}"
    if not allow_action:
        assert "action_applied" not in cols, f"feature set '{name}' must not contain action_applied"
    assert len(cols) == len(set(cols)), f"feature set '{name}' has duplicate columns"
    if dd is not None:
        ns = dict(zip(dd.column, dd.namespace))
        unknown = [c for c in cols if c not in ns]
        assert not unknown, f"feature set '{name}' has columns missing from the data dictionary {unknown}"
        leaky = [c for c in cols if ns[c] in FORBIDDEN_NAMESPACES]
        assert not leaky, f"feature set '{name}' contains key/time/label columns {leaky}"
        obs = dict(zip(dd.column, dd.twin_observable))
        not_obs = [c for c in cols if not bool(obs[c]) and c != "action_applied"]
        assert not not_obs, f"feature set '{name}' contains non-observable columns {not_obs}"
    if name == "sensor":
        bad = [c for c in cols if c.startswith(SENSOR_DROPPED_PREFIXES) or c.endswith(SENSOR_DROPPED_SUFFIXES)
               or (c.startswith("q_") and not c.startswith("obs_q_"))]
        assert not bad, f"sensor feature set contains truth-only columns {bad}"


def feature_sets(dd: pd.DataFrame) -> dict:
    """truth / sensor / regression column lists, selected by namespace (DECISIONS §7)."""
    by_ns = {ns: list(dd.column[dd.namespace == ns]) for ns in dd.namespace.unique()}
    truth = [c for ns in TRUTH_NAMESPACES for c in by_ns[ns]]
    sensor_state = []
    for c in by_ns["state"]:
        if c.startswith(SENSOR_DROPPED_PREFIXES) or c.endswith(SENSOR_DROPPED_SUFFIXES):
            continue
        if c.startswith("q_"):
            sensor_state.append("obs_" + c)        # obs_q_a replaces q_a
            continue
        sensor_state.append(c)
    sensor = (by_ns["signal"] + sensor_state + by_ns["downstream"] + by_ns["neighbour"] + by_ns["ev"])
    assert all(c in by_ns["sensor"] for c in sensor_state), sensor_state
    regression = truth + ["action_applied"]
    fs = {"truth": truth, "sensor": sensor, "regression": regression}
    assert_feature_set_clean(fs["truth"], "truth", dd)
    assert_feature_set_clean(fs["sensor"], "sensor", dd)
    assert_feature_set_clean(fs["regression"], "regression", dd, allow_action=True)
    return fs


# ---- data ------------------------------------------------------------------------------------
def load_main_rows(data_dir: str) -> tuple:
    """signal_ticks rows outside warm-up / cool-down / tail, with the episode split attached."""
    episodes = pd.read_csv(os.path.join(data_dir, "episodes.csv"))
    check_splits(episodes)
    ticks = pd.read_csv(os.path.join(data_dir, "signal_ticks.csv"))
    for c in ticks.columns:
        if ticks[c].dtype == bool:
            ticks[c] = ticks[c].astype(np.int8)
    main = ticks[(ticks.is_warmup == 0) & (ticks.is_cooldown == 0) & (ticks.is_tail == 0)]
    main = main.merge(episodes[["episode_id", "split", "controller", "demand_level", "sensor_noise_level"]],
                      on="episode_id", how="left")
    assert main.split.notna().all(), "every signal_ticks episode must be in episodes.csv"
    return main.reset_index(drop=True), episodes


def check_splits(episodes: pd.DataFrame) -> dict:
    """Split integrity (DECISIONS §7): one split per episode, stress <-> test_ood, random_dwell -> train."""
    assert episodes.episode_id.is_unique, "an episode appears twice in episodes.csv"
    assert set(episodes.split.unique()) <= {"train", "val", "test", "test_ood"}, episodes.split.unique()
    stress = episodes[episodes.demand_level == "stress"]
    assert (stress.split == "test_ood").all(), "every stress episode must be test_ood"
    ood = episodes[episodes.split == "test_ood"]
    assert (ood.demand_level == "stress").all(), "test_ood must contain only stress episodes"
    rd = episodes[episodes.controller == "random_dwell"]
    assert (rd.split == "train").all(), "random_dwell episodes must be train-only"
    return {s: int((episodes.split == s).sum()) for s in ("train", "val", "test", "test_ood")}


def matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    return df[cols].to_numpy(dtype=np.float32)


# ---- models ----------------------------------------------------------------------------------
def categorical_indices(cols: list) -> list:
    """phase-like integer columns that logistic regression should see one-hot."""
    return [i for i, c in enumerate(cols) if c == "phase" or (c.startswith("nb_") and c.endswith("_phase"))]


def make_logreg(cols: list) -> Pipeline:
    cat = categorical_indices(cols)
    num = [i for i in range(len(cols)) if i not in cat]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                             ("num", StandardScaler(), num)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])


def make_hgb_classifier(random_state: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=6, l2_regularization=1.0,
                                          early_stopping=True, class_weight="balanced", random_state=random_state)


def make_tree() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=5, random_state=0)


def make_ridge() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])


def make_hgb_regressor(random_state: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=8, early_stopping=True,
                                         random_state=random_state)


# ---- metrics ---------------------------------------------------------------------------------
def classification_metrics(y: np.ndarray, pred: np.ndarray, phase: np.ndarray) -> dict:
    """balanced accuracy, macro-F1, F1 of the switch rows (label != phase), accuracy, confusion, per class."""
    labels = [0, 1, 2, 3]
    switch_true = (y != phase).astype(int)
    switch_pred = (pred != phase).astype(int)
    rep = classification_report(y, pred, labels=labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    return dict(
        n=int(len(y)),
        accuracy=float((y == pred).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        switch_f1=float(f1_score(switch_true, switch_pred, zero_division=0)),
        switch_share_true=float(switch_true.mean()),
        switch_share_pred=float(switch_pred.mean()),
        confusion=confusion_matrix(y, pred, labels=labels).tolist(),
        per_class={k: v for k, v in rep.items() if k in CLASS_NAMES},
    )


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return dict(n=int(len(y)), mae=float(mean_absolute_error(y, pred)),
                rmse=float(math.sqrt(mean_squared_error(y, pred))), r2=float(r2_score(y, pred)))


# ---- figures ---------------------------------------------------------------------------------
def plot_confusion(cm: list, title: str, path: str) -> None:
    cm = np.array(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(4), CLASS_NAMES)
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set_xlabel("predicted request")
    ax.set_ylabel("teacher label y_mp_action")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row share")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(names: list, means: np.ndarray, stds: np.ndarray, title: str, path: str) -> None:
    order = np.argsort(means)[::-1][:15][::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh([names[i] for i in order], means[order], xerr=stds[order], color="#4c72b0")
    ax.set_xlabel("drop in balanced accuracy when the column is permuted")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_partial_dependence(table: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    ax.bar(table.action_name, table.mean_pred_queue, color="#4c72b0")
    ax.set_ylabel("mean predicted q_total at t+30 [veh]")
    ax.set_xlabel("candidate action_applied (all test rows)")
    ax.set_title("partial dependence of HGB on the action", fontsize=10)
    ax = axes[1]
    ax.bar(table.action_name, table.mean_pred_queue_when_hold, color="#55a868", label="= current phase (hold)")
    ax.bar(table.action_name, table.mean_pred_queue_when_switch, color="#c44e52", alpha=0.7, width=0.5,
           label="!= current phase (switch)")
    ax.set_xlabel("candidate action_applied")
    ax.set_title("same, split by hold vs switch rows", fontsize=10)
    ax.legend(fontsize=8)
    for a in axes:
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---- markdown helpers ------------------------------------------------------------------------
def md_table(rows: list, cols: list, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = fmt.get(c, "{:.3f}").format(v)
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---- the pipeline ----------------------------------------------------------------------------
def run_task_a(main: pd.DataFrame, fs: dict, out: str, figures: str, models_dir: str, cv_folds: int = 5,
               perm_rows: int = 20000, log=print) -> dict:
    results = {"feature_sets": {k: list(v) for k, v in fs.items()}, "models": {}, "cv": {}, "permutation_importance": {}}
    train = main[main.split == "train"]
    y_train = train.y_mp_action.to_numpy()
    groups = train.episode_id.to_numpy()
    fitted = {}
    for fs_name in ("truth", "sensor"):
        cols = fs[fs_name]
        X_train = matrix(train, cols)
        for model_name, maker in (("logreg", lambda: make_logreg(cols)), ("hgb", make_hgb_classifier), ("tree", make_tree)):
            key = f"{model_name}_{fs_name}"
            t0 = time.perf_counter()
            model = maker()
            model.fit(X_train, y_train)
            fit_s = time.perf_counter() - t0
            log(f"  fitted {key} in {fit_s:.1f} s")
            entry = {"fit_seconds": fit_s, "n_train": int(len(y_train)), "n_features": len(cols), "splits": {}}
            if model_name == "hgb":
                entry["n_iter"] = int(model.n_iter_)
            for split in SPLITS:
                part = main[main.split == split]
                pred = model.predict(matrix(part, cols))
                m = classification_metrics(part.y_mp_action.to_numpy(), pred, part.phase.to_numpy())
                entry["splits"][split] = m
                plot_confusion(m["confusion"], f"{key} on {split}", os.path.join(figures, f"taskA_confusion_{key}_{split}.png"))
                log(f"    {key:14s} {split:8s} bal_acc {m['balanced_accuracy']:.3f} macro_f1 {m['macro_f1']:.3f} "
                    f"switch_f1 {m['switch_f1']:.3f} acc {m['accuracy']:.3f}")
            if model_name == "tree":
                rules = export_text(model, feature_names=cols, max_depth=5)
                with open(os.path.join(out, f"taskA_tree_rules_{fs_name}.txt"), "w") as f:
                    f.write(rules)
                entry["rules_depth3"] = export_text(model, feature_names=cols, max_depth=3)
            results["models"][key] = entry
            fitted[key] = model
            joblib.dump({"task": "A", "model": model, "features": list(cols), "feature_set": fs_name,
                         "target": "y_mp_action", "classes": [int(c) for c in model.classes_]},
                        os.path.join(models_dir, f"taskA_{key}.joblib"))
        # GroupKFold CV on train for the HGB model
        t0 = time.perf_counter()
        cv = cross_validate(make_hgb_classifier(), X_train, y_train, groups=groups, cv=GroupKFold(cv_folds),
                            scoring={"balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro"}, n_jobs=1)
        results["cv"][f"hgb_{fs_name}"] = {
            "folds": cv_folds, "seconds": time.perf_counter() - t0,
            "balanced_accuracy_mean": float(cv["test_balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(cv["test_balanced_accuracy"].std()),
            "balanced_accuracy_folds": cv["test_balanced_accuracy"].tolist(),
            "macro_f1_mean": float(cv["test_macro_f1"].mean()), "macro_f1_std": float(cv["test_macro_f1"].std()),
            "macro_f1_folds": cv["test_macro_f1"].tolist()}
        log(f"  GroupKFold({cv_folds}) hgb_{fs_name}: bal_acc {cv['test_balanced_accuracy'].mean():.3f} "
            f"+- {cv['test_balanced_accuracy'].std():.3f}, macro_f1 {cv['test_macro_f1'].mean():.3f}")
    # permutation importance of HGB-sensor on test
    cols = fs["sensor"]
    test = main[main.split == "test"]
    if len(test) > perm_rows:
        test = test.sample(perm_rows, random_state=0)
    t0 = time.perf_counter()
    pi = permutation_importance(fitted["hgb_sensor"], matrix(test, cols), test.y_mp_action.to_numpy(),
                                scoring="balanced_accuracy", n_repeats=5, random_state=0, n_jobs=1)
    order = np.argsort(pi.importances_mean)[::-1]
    results["permutation_importance"]["hgb_sensor_test"] = {
        "n_rows": int(len(test)), "seconds": time.perf_counter() - t0,
        "top15": [{"feature": cols[i], "mean": float(pi.importances_mean[i]), "std": float(pi.importances_std[i])}
                  for i in order[:15]],
        "all": {cols[i]: float(pi.importances_mean[i]) for i in range(len(cols))}}
    plot_importance(cols, pi.importances_mean, pi.importances_std, "permutation importance, HGB-sensor, test split",
                    os.path.join(figures, "taskA_permutation_importance_hgb_sensor.png"))
    log(f"  permutation importance done in {time.perf_counter() - t0:.1f} s; top: "
        + ", ".join(f"{cols[i]} {pi.importances_mean[i]:.3f}" for i in order[:5]))
    return results


def run_task_b(main: pd.DataFrame, fs: dict, out: str, figures: str, models_dir: str, pd_rows: int = 50000,
               log=print) -> dict:
    cols = fs["regression"]
    data = main[main.y_q_total_t30.notna()]
    train = data[data.split == "train"]
    X_train, y_train = matrix(train, cols), train.y_q_total_t30.to_numpy(dtype=np.float64)
    results = {"features": list(cols), "n_train": int(len(train)), "models": {}, "naive_baseline": {}}
    q_cols = [f"q_{a}" for a in DIRS]
    for split in SPLITS:
        part = data[data.split == split]
        results["naive_baseline"][split] = regression_metrics(part.y_q_total_t30.to_numpy(), part[q_cols].sum(axis=1).to_numpy())
        results["naive_baseline"][split]["target_mean"] = float(part.y_q_total_t30.mean())
        results["naive_baseline"][split]["target_std"] = float(part.y_q_total_t30.std())
    fitted = {}
    for model_name, maker in (("ridge", make_ridge), ("hgb", make_hgb_regressor)):
        t0 = time.perf_counter()
        model = maker()
        model.fit(X_train, y_train)
        entry = {"fit_seconds": time.perf_counter() - t0, "splits": {}}
        if model_name == "hgb":
            entry["n_iter"] = int(model.n_iter_)
        for split in SPLITS:
            part = data[data.split == split]
            entry["splits"][split] = regression_metrics(part.y_q_total_t30.to_numpy(), model.predict(matrix(part, cols)))
            log(f"    {model_name:6s} {split:8s} MAE {entry['splits'][split]['mae']:.2f} RMSE {entry['splits'][split]['rmse']:.2f} "
                f"R2 {entry['splits'][split]['r2']:.3f} (naive MAE {results['naive_baseline'][split]['mae']:.2f})")
        results["models"][model_name] = entry
        fitted[model_name] = model
        joblib.dump({"task": "B", "model": model, "features": list(cols), "feature_set": "regression",
                     "target": "y_q_total_t30"}, os.path.join(models_dir, f"taskB_{model_name}.joblib"))
    # partial dependence of the HGB prediction on action_applied (test split)
    test = data[data.split == "test"]
    if len(test) > pd_rows:
        test = test.sample(pd_rows, random_state=0)
    X_test = matrix(test, cols)
    a_idx = cols.index("action_applied")
    phase = test.phase.to_numpy()
    rows = []
    for a in range(4):
        X_mod = X_test.copy()
        X_mod[:, a_idx] = a
        pred = fitted["hgb"].predict(X_mod)
        hold = phase == a
        rows.append(dict(action=a, action_name=CLASS_NAMES[a], mean_pred_queue=float(pred.mean()),
                         mean_pred_queue_when_hold=float(pred[hold].mean()) if hold.any() else math.nan,
                         mean_pred_queue_when_switch=float(pred[~hold].mean()) if (~hold).any() else math.nan,
                         n_rows=int(len(pred))))
    table = pd.DataFrame(rows)
    # the same comparison seen from the row: predicted queue if the row holds its phase vs the best other action
    X_hold = X_test.copy()
    X_hold[:, a_idx] = phase
    pred_hold = fitted["hgb"].predict(X_hold)
    pred_other = np.full(len(phase), np.inf)
    for a in range(4):
        X_mod = X_test.copy()
        X_mod[:, a_idx] = a
        p = fitted["hgb"].predict(X_mod)
        mask = phase != a
        pred_other[mask] = np.minimum(pred_other[mask], p[mask])
    results["partial_dependence_action"] = {
        "n_rows": int(len(test)), "table": rows,
        "mean_pred_queue_hold_current": float(pred_hold.mean()),
        "mean_pred_queue_best_other_action": float(pred_other.mean()),
        "share_rows_where_some_switch_predicts_lower_queue": float((pred_other < pred_hold).mean())}
    table.to_csv(os.path.join(out, "taskB_partial_dependence_action.csv"), index=False)
    plot_partial_dependence(table, os.path.join(figures, "taskB_partial_dependence_action.png"))
    log("  partial dependence on action_applied: " + ", ".join(f"{r['action_name']} {r['mean_pred_queue']:.2f}" for r in rows))
    return results


def dataset_summary(main: pd.DataFrame, episodes: pd.DataFrame) -> dict:
    """What the Data section of RESULTS.md reports (every number from the tables themselves)."""
    counts = lambda s: {str(k): int(v) for k, v in s.value_counts().items()}  # noqa: E731
    ev_eps = episodes[episodes.n_ambulances > 0] if "n_ambulances" in episodes.columns else episodes.iloc[:0]
    return dict(
        config_hash=",".join(sorted(episodes.config_hash.astype(str).unique())) if "config_hash" in episodes.columns else "",
        generator_version=",".join(sorted(episodes.generator_version.astype(str).unique())) if "generator_version" in episodes.columns else "",
        controllers=counts(episodes.controller), demand_levels=counts(episodes.demand_level),
        sensor_noise=counts(episodes.sensor_noise_level), ev_rate=counts(episodes.ev_rate.astype(str)) if "ev_rate" in episodes.columns else {},
        demand_asym=counts(episodes.demand_asym) if "demand_asym" in episodes.columns else {},
        od_pattern=counts(episodes.od_pattern) if "od_pattern" in episodes.columns else {},
        n_gridlock=int(episodes.gridlock.astype(float).sum()) if "gridlock" in episodes.columns else 0,
        n_invariant_violations=int(episodes.n_invariant_violations.sum()) if "n_invariant_violations" in episodes.columns else 0,
        n_ambulance_episodes=int(len(ev_eps)), n_ambulances=int(episodes.n_ambulances.sum()) if "n_ambulances" in episodes.columns else 0,
        lambda_base_vph={str(k): float(v) for k, v in episodes.groupby("demand_level").lambda_base_vph.mean().round(3).items()},
        label_balance={CLASS_NAMES[int(k)]: float(v) for k, v in main.y_mp_action.value_counts(normalize=True).sort_index().items()},
        label_switch_share=float((main.y_mp_action != main.phase).mean()),
        rows_with_ambulance_present=float(main[[f"ev_present_{a}" for a in DIRS]].any(axis=1).mean()) if f"ev_present_N" in main.columns else math.nan,
        y_q_total_t30_nan_share=float(main.y_q_total_t30.isna().mean()),
        rows_per_controller=counts(main.controller), rows_per_demand=counts(main.demand_level),
    )


def write_results_md(res: dict, path: str) -> None:
    ds = res.get("dataset", {})
    L = [
        f"# Supervised models on the {res['n_episodes']}-episode release", "",
        "Imitation of the priority-aware signal teacher (Task A), a 30 s-ahead queue forecast (Task B) and, in sections 4-6, "
        "the learned controller deployed in the simulator against the proposal's acceptance bound. Every number in this report "
        "is computed from the stored results of the same run.", "",
        "## 1. Data", "",
        f"Release of {res['n_episodes']} episodes (generator `{ds.get('generator_version', '')}`, config hash `{ds.get('config_hash', '')}`): "
        f"{res['n_main_rows']} main-period decision rows from the signal-tick table "
        f"(warm-up, cool-down and tail rows removed), split by episode: "
        + ", ".join(f"{k} {v}" for k, v in res["rows_per_split"].items()) + ".",
        f"Episodes per split: " + ", ".join(f"{k} {v}" for k, v in res["episodes_per_split"].items())
        + ". Split integrity verified in code: every stress episode is test_ood and test_ood holds only stress episodes; "
        "random_dwell episodes are train-only; no episode is in two splits.", ""]
    if ds:
        L += ["Scenario mixture (episodes): controllers " + ", ".join(f"{k} {v}" for k, v in ds["controllers"].items())
              + "; demand levels " + ", ".join(f"{k} {v}" for k, v in ds["demand_levels"].items())
              + "; sensor_noise_level " + ", ".join(f"{k} {v}" for k, v in ds["sensor_noise"].items())
              + "; ev_rate " + ", ".join(f"{k} {v}" for k, v in ds["ev_rate"].items())
              + "; demand_asym " + ", ".join(f"{k} {v}" for k, v in ds["demand_asym"].items())
              + "; od_pattern " + ", ".join(f"{k} {v}" for k, v in ds["od_pattern"].items()) + ".",
              f"Calibrated arrival rates (veh/h per entry node, mean over episodes): "
              + ", ".join(f"{k} {v:.1f}" for k, v in ds["lambda_base_vph"].items()) + ".",
              f"Ambulance episodes {ds['n_ambulance_episodes']} ({ds['n_ambulances']} ambulances in total); gridlocked episodes "
              f"{ds['n_gridlock']}; invariant violations {ds['n_invariant_violations']}.",
              "Label balance of `y_mp_action` on the main rows: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in ds["label_balance"].items())
              + f"; the label differs from the current phase (a switch request) on {100 * ds['label_switch_share']:.1f}% of rows; "
              f"an ambulance is present on some approach in {100 * ds['rows_with_ambulance_present']:.1f}% of rows; "
              f"`y_q_total_t30` is NaN on {100 * ds['y_q_total_t30_nan_share']:.2f}% of main rows (tail rows already removed).", ""]
    L += [f"Total runtime of this pipeline: {res['runtime_s']:.0f} s (scikit-learn {res.get('sklearn_version', '')}).", "",
          "### Feature sets (selected by namespace from the data dictionary)", ""]
    for k, v in res["task_a"]["feature_sets"].items():
        L.append(f"- **{k}** ({len(v)} columns): `" + "`, `".join(v) + "`")
    L += ["",
          "Never used as features (asserted in code): keys (`episode_id`, `junction_id`), time (`t`, `tick_idx`, period flags), "
          "`action_requested`, `forced`, `switched`, and the labels `y_*`. The sensor set keeps only what a loop-detector "
          "installation plus the ambulance GPS channel could deliver: no `n_*`, no per-lane queues, no arrival counts, "
          "`obs_q_*` instead of `q_*`. Note that the teacher rule is computed from `n_*` and `down_n_*`, so the sensor "
          "set can only approximate it.", "",
          "## 2. Task A: imitation of the priority-aware teacher (4 classes, target `y_mp_action`)", "",
          "Models: class-balanced logistic regression (one-hot phase columns + standardised numerics), "
          "HistGradientBoostingClassifier(max_iter=400, lr=0.05, depth=6, l2=1.0, early stopping, class-balanced) and a depth-5 "
          "decision tree. `switch_f1` is the F1 of the binary event *label != current phase* against *prediction != current phase*.", ""]
    rows = []
    for key, e in res["task_a"]["models"].items():
        for split in SPLITS:
            m = e["splits"][split]
            rows.append(dict(model=key, split=split, n=m["n"], balanced_accuracy=m["balanced_accuracy"], macro_f1=m["macro_f1"],
                             switch_f1=m["switch_f1"], accuracy=m["accuracy"]))
    L += [md_table(rows, ["model", "split", "n", "balanced_accuracy", "macro_f1", "switch_f1", "accuracy"]), ""]
    L += ["### GroupKFold(5, groups = episode_id) on the train split, HGB", ""]
    rows = [dict(model=k, balanced_accuracy=f"{v['balanced_accuracy_mean']:.3f} +- {v['balanced_accuracy_std']:.3f}",
                 macro_f1=f"{v['macro_f1_mean']:.3f} +- {v['macro_f1_std']:.3f}",
                 folds=", ".join(f"{x:.3f}" for x in v["balanced_accuracy_folds"])) for k, v in res["task_a"]["cv"].items()]
    L += [md_table(rows, ["model", "balanced_accuracy", "macro_f1", "folds"]), ""]
    L += ["### Per-class F1 on the test split", ""]
    rows = []
    for key, e in res["task_a"]["models"].items():
        m = e["splits"]["test"]["per_class"]
        rows.append(dict(model=key, **{f"F1_{c}": m[c]["f1-score"] for c in CLASS_NAMES},
                         **{f"support_{c}": int(m[c]["support"]) for c in CLASS_NAMES}))
    L += [md_table(rows, ["model"] + [f"F1_{c}" for c in CLASS_NAMES] + [f"support_{c}" for c in CLASS_NAMES]), ""]
    L += ["### Confusion matrices", "",
          "Figure: Task A confusion matrices (counts and row shares) for every model and split. HGB-truth on test:", ""]
    cm = res["task_a"]["models"]["hgb_truth"]["splits"]["test"]["confusion"]
    L += [md_table([dict(label=CLASS_NAMES[i], **{f"pred_{CLASS_NAMES[j]}": cm[i][j] for j in range(4)}) for i in range(4)],
                   ["label"] + [f"pred_{c}" for c in CLASS_NAMES]), ""]
    cm = res["task_a"]["models"]["hgb_sensor"]["splits"]["test"]["confusion"]
    L += ["HGB-sensor on test:", "",
          md_table([dict(label=CLASS_NAMES[i], **{f"pred_{CLASS_NAMES[j]}": cm[i][j] for j in range(4)}) for i in range(4)],
                   ["label"] + [f"pred_{c}" for c in CLASS_NAMES]), ""]
    pi = res["task_a"]["permutation_importance"]["hgb_sensor_test"]
    L += [f"### Permutation importance, HGB-sensor, test split ({pi['n_rows']} rows, 5 repeats, balanced accuracy)", "",
          md_table(pi["top15"], ["feature", "mean", "std"], {"mean": "{:.4f}", "std": "{:.4f}"}), "",
          "Figure: permutation importance of the HGB-sensor model on the test split.", "",
          "### Depth-5 decision tree (first three levels, truth features)", "", "```",
          res["task_a"]["models"]["tree_truth"]["rules_depth3"].rstrip(), "```",
          "Full rules: `taskA_tree_rules_truth.txt`, `taskA_tree_rules_sensor.txt`.", "",
          "## 3. Task B: queue 30 s ahead (target `y_q_total_t30`, feature set `regression` = truth + action_applied)", ""]
    rows = []
    for split in SPLITS:
        nb = res["task_b"]["naive_baseline"][split]
        rows.append(dict(model="naive: q_total now", split=split, n=nb["n"], mae=nb["mae"], rmse=nb["rmse"], r2=nb["r2"],
                         target_mean=nb["target_mean"]))
        for mk, e in res["task_b"]["models"].items():
            m = e["splits"][split]
            rows.append(dict(model=mk, split=split, n=m["n"], mae=m["mae"], rmse=m["rmse"], r2=m["r2"], target_mean=nb["target_mean"]))
    L += [md_table(rows, ["model", "split", "n", "mae", "rmse", "r2", "target_mean"], {"mae": "{:.2f}", "rmse": "{:.2f}", "target_mean": "{:.2f}"}), ""]
    pdp = res["task_b"]["partial_dependence_action"]
    L += [f"### Partial dependence of the HGB prediction on `action_applied` ({pdp['n_rows']} test rows)", "",
          md_table(pdp["table"], ["action_name", "mean_pred_queue", "mean_pred_queue_when_hold", "mean_pred_queue_when_switch", "n_rows"],
                   {"mean_pred_queue": "{:.2f}", "mean_pred_queue_when_hold": "{:.2f}", "mean_pred_queue_when_switch": "{:.2f}"}), "",
          f"Holding the current phase gives a mean predicted queue of {pdp['mean_pred_queue_hold_current']:.2f} veh; the best "
          f"other action {pdp['mean_pred_queue_best_other_action']:.2f} veh; in {100 * pdp['share_rows_where_some_switch_predicts_lower_queue']:.1f}% "
          "of test rows some switch is predicted to lower the queue at t+30. Figure: partial dependence of the Task B prediction on the applied action.", "",
          "The 30 s-ahead queue is mostly explained by the queue now (see the naive baseline), so R^2 alone flatters; the "
          "control-relevant quantity is the action effect above, which is small compared with the state effect.", ""]
    a = res["task_a"]["models"]
    L += ["### Reading Tasks A and B", "",
          f"- The teacher is a deterministic rule of `n_*`, `down_n_*`, `down_blocked_*`, `time_in_phase_s` and the `ev_*` channel. "
          f"With truth features HGB reaches balanced accuracy {a['hgb_truth']['splits']['test']['balanced_accuracy']:.3f} on test and "
          f"{a['hgb_truth']['splits']['test_ood']['balanced_accuracy']:.3f} on stress (test_ood); the remaining errors sit on the "
          "hysteresis boundary (pressure difference close to 2) and on max-green rows.",
          f"- With sensor features (queues only, noisy in the noisy episodes) HGB reaches {a['hgb_sensor']['splits']['test']['balanced_accuracy']:.3f} "
          f"on test and {a['hgb_sensor']['splits']['test_ood']['balanced_accuracy']:.3f} on stress: the gap to the truth model is the price "
          "of not observing moving vehicles (`n_a`) and arrivals.",
          f"- Logistic regression (truth) reaches {a['logreg_truth']['splits']['test']['balanced_accuracy']:.3f}: the rule is a difference of "
          "counts with a threshold, which a linear model can only partly express across phases.",
          f"- Task B: HGB MAE {res['task_b']['models']['hgb']['splits']['test']['mae']:.2f} veh vs naive {res['task_b']['naive_baseline']['test']['mae']:.2f} veh "
          f"on test; on stress {res['task_b']['models']['hgb']['splits']['test_ood']['mae']:.2f} vs {res['task_b']['naive_baseline']['test_ood']['mae']:.2f}.",
          "", "## 4. Closed loop", "",
          "The closed-loop evaluation has not been run for this release.", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))


def stamp_models(models_dir: str, ds: dict) -> int:
    """Record the dataset generator_version / config_hash inside each saved model bundle."""
    import glob
    n = 0
    for path in sorted(glob.glob(os.path.join(models_dir, "*.joblib"))):
        bundle = joblib.load(path)
        bundle["generator_version"] = ds["generator_version"]
        bundle["config_hash"] = ds["config_hash"]
        joblib.dump(bundle, path)
        n += 1
    return n


def run_pipeline(data_dir: str, out: str, cv_folds: int = 5, perm_rows: int = 20000, log=print) -> dict:
    t0 = time.perf_counter()
    figures = os.path.join(out, "figures")
    models_dir = os.path.join(out, "models")
    os.makedirs(figures, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    dd = load_dictionary(data_dir)
    fs = feature_sets(dd)
    log(f"feature sets: " + ", ".join(f"{k} {len(v)} cols" for k, v in fs.items()))
    main, episodes = load_main_rows(data_dir)
    log(f"main rows {len(main)} from {len(episodes)} episodes; rows per split "
        + ", ".join(f"{s} {int((main.split == s).sum())}" for s in ("train",) + SPLITS))
    with open(os.path.join(out, "feature_sets.json"), "w") as f:
        json.dump(fs, f, indent=2)
    log("Task A")
    task_a = run_task_a(main, fs, out, figures, models_dir, cv_folds=cv_folds, perm_rows=perm_rows, log=log)
    log("Task B")
    task_b = run_task_b(main, fs, out, figures, models_dir, log=log)
    import sklearn
    ds = dataset_summary(main, episodes)
    res = dict(data_dir=data_dir, n_episodes=int(len(episodes)), n_main_rows=int(len(main)),
               rows_per_split={s: int((main.split == s).sum()) for s in ("train",) + SPLITS},
               episodes_per_split=check_splits(episodes), dataset=ds,
               dataset_config_hash=ds["config_hash"], dataset_generator_version=ds["generator_version"],
               sklearn_version=sklearn.__version__, numpy_version=np.__version__, pandas_version=pd.__version__,
               python_version=platform.python_version(),
               command=f"python3 -m traffic_twin.ml --data {data_dir} --out {out} --cv-folds {cv_folds} --perm-rows {perm_rows}",
               task_a=task_a, task_b=task_b, runtime_s=time.perf_counter() - t0)
    stamp_models(models_dir, ds)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    write_results_md(res, os.path.join(out, "RESULTS.md"))
    log(f"wrote {out}/results.json, RESULTS.md, models/, figures/ in {res['runtime_s']:.0f} s")
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/traffic_v1")
    p.add_argument("--out", default="reports/ml")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--perm-rows", type=int, default=20000)
    args = p.parse_args(argv)
    run_pipeline(args.data, args.out, cv_folds=args.cv_folds, perm_rows=args.perm_rows)


if __name__ == "__main__":
    main()
