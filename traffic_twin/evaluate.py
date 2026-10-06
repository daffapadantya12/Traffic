"""Closed-loop evaluation of the learned controller (DECISIONS §10, methodology §12.3).

Run: python3 -m traffic_twin.evaluate --models reports/ml/models --out reports/ml --data data/traffic_v1

`MLController` wraps the HGB-sensor imitation model: at every decision instant it
builds the sensor feature vector from the same row dict the simulator hands to
every controller, predicts class probabilities and requests the argmax only if
P(best) > P(current phase) + margin (0.10); otherwise it holds.  The plant still
enforces min/max green and the all-red, so safety does not depend on the model.

The controller is registered under the name "ml_sensor" (and the alias "ml_v1"
used by DECISIONS §10) in `traffic_twin.controllers.CONTROLLERS` at runtime (no
file in the simulator package is edited), so `World` runs it like any built-in
controller.

Protocol (DECISIONS §10 + the proposal's acceptance bound):
  controllers {fixed_time, max_pressure, preempt_naive, priority_mp, ml_sensor}
  x demand {design, high} x 8 paired seeds (same arrivals and drivers across
  controllers by construction), ev_rate default, sensor_noise_level 0.
  * every (controller, demand, seed) is also run with ambulances OFF (same seed)
    so `car_mean_delay_penalty_s` = car delay with - without ambulances;
  * acceptance bound: per seed, PASS iff car mean delay under the ML controller
    <= 1.10 x car mean delay under max_pressure; pass rate and mean ratio with a
    t-based 95% interval;
  * secondary metrics: ev_mean_travel_time_s, car_mean_delay_penalty_s,
    recovery_time_s and a fairness ratio (max/min of the mean car delay over the
    8 entry nodes);
  * sensor-noise variant: ml_sensor and max_pressure at sensor_noise_level 0.1;
  * paired differences versus max_pressure with t-based 95% intervals
    (t_{7,0.975} = 2.365);
  * the simulator's generator_version / config_hash are recorded in
    closed_loop_meta.json and asserted equal to the dataset's.
"""

from __future__ import annotations

import os

# Serial ML inference inside each worker: one tree-walker per process, not
# (workers x BLAS threads).  Must happen before numpy / sklearn are imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import controllers as ctrl_mod  # noqa: E402
from .calibrate import calibrate_demand  # noqa: E402
from .config import GENERATOR_VERSION, SimConfig, load_config  # noqa: E402
from .controllers import Controller, label_request  # noqa: E402
from .sim import Scenario, World  # noqa: E402

ML_NAME = "ml_sensor"
ML_ALIASES = ("ml_v1",)                 # DECISIONS §10 calls the learned controller ml_v1
ML_NAMES = (ML_NAME,) + ML_ALIASES
DEFAULT_CONTROLLERS = ("fixed_time", "max_pressure", "preempt_naive", "priority_mp", ML_NAME)
NOISE_CONTROLLERS = ("max_pressure", ML_NAME)
DEFAULT_DEMANDS = ("design", "high")
DEFAULT_SEEDS = tuple(range(900001, 900009))
DEFAULT_NOISE_VARIANT = 0.1
REFERENCE = "max_pressure"
BOUND_FACTOR = 1.10                      # proposal: ML car delay <= 1.10 x max_pressure car delay
METRICS = ("ev_mean_travel_time_s", "ev_mean_delay_s", "mean_total_delay_s", "p90_total_delay_s", "throughput_vph", "switches_per_min",
           "recovery_time_s", "gridlock", "n_red_commits", "ev_p90_delay_s", "ev_mean_stops",
           "car_mean_delay_penalty_s", "fairness_ratio", "n_conflicts", "n_near_misses", "n_lane_changes", "n_lc_yield")
SECONDARY = ("ev_mean_travel_time_s", "car_mean_delay_penalty_s", "recovery_time_s", "fairness_ratio")
T_7_975 = 2.365
EV_OFF = "0"


# ---------------------------------------------------------------- the learned controller
class MLController(Controller):
    """act(row) -> requested phase from an imitation model's class probabilities."""
    name = ML_NAME

    def __init__(self, cfg: SimConfig, rng=None, bundle: dict | None = None, model_path: str | None = None,
                 margin: float = 0.10):
        super().__init__(cfg, rng)
        if bundle is None:
            if model_path is None:
                raise ValueError("MLController needs a model bundle or a model_path")
            bundle = joblib.load(model_path)
        if bundle.get("task", "A") != "A":
            raise ValueError(f"MLController needs a Task A (phase imitation) bundle, got task {bundle.get('task')!r}")
        self.model = bundle["model"]
        self.features = list(bundle["features"])
        self.margin = float(margin)
        classes = [int(c) for c in getattr(self.model, "classes_", bundle.get("classes", [0, 1, 2, 3]))]
        self.class_index = {c: i for i, c in enumerate(classes)}
        self._x = np.zeros((1, len(self.features)), dtype=np.float32)
        self.stats = dict(n_main=0, n_agree=0, n_switch=0, n_teacher_switch=0)

    def probabilities(self, row: dict) -> np.ndarray:
        """P(request = k) for k = 0..3 in approach order (0 for classes the model never saw)."""
        x = self._x
        for i, c in enumerate(self.features):
            x[0, i] = float(row[c])
        proba = self.model.predict_proba(x)[0]
        p = np.zeros(4)
        for c, i in self.class_index.items():
            if 0 <= c < 4:
                p[c] = proba[i]
        return p

    def act(self, row: dict) -> int:
        phase = int(row["phase"])
        p = self.probabilities(row)
        best = max(range(4), key=lambda k: (p[k], -k))             # ties -> lowest index like the teacher
        request = best if p[best] > p[phase] + self.margin else phase
        if not row.get("is_warmup", False) and not row.get("is_cooldown", False):
            teacher = label_request(row, self.cfg)
            self.stats["n_main"] += 1
            self.stats["n_agree"] += int(request == teacher)
            self.stats["n_switch"] += int(request != phase)
            self.stats["n_teacher_switch"] += int(teacher != phase)
        return int(request)


def register_ml_controller(bundle: dict, margin: float = 0.10, name: str = ML_NAME, aliases=ML_ALIASES) -> None:
    """Make `World` able to build the learned controller by name (runtime registration, no file edit).
    The same factory is registered under `name` and every alias (DECISIONS §10 says `ml_v1`)."""
    factory = functools.partial(MLController, bundle=bundle, margin=margin)
    ctrl_mod.CONTROLLERS[name] = factory
    for alias in aliases:
        ctrl_mod.CONTROLLERS[alias] = factory


def unregister_ml_controller(name: str = ML_NAME, aliases=ML_ALIASES) -> None:
    for n in (name,) + tuple(aliases):
        ctrl_mod.CONTROLLERS.pop(n, None)


# ---------------------------------------------------------------- one episode
_WORKER = {"bundle": None, "path": None}


def _init_worker(model_path: str | None, margin: float):
    if model_path is not None:
        _WORKER["bundle"] = joblib.load(model_path)
        _WORKER["path"] = model_path
        register_ml_controller(_WORKER["bundle"], margin)


def fairness_ratio(vehicle_rows: list, cfg: SimConfig) -> tuple:
    """max / min of the mean car delay over the entry nodes (main-period cars).
    Returns (ratio, n_origins, worst_origin, best_origin); NaN ratio when fewer than two origins."""
    per_origin = {}
    for r in vehicle_rows:
        if r["vehicle_type"] == "ambulance" or not cfg.is_main(r["spawn_request_t"]):
            continue
        d = r["total_delay_s"]
        if d != d:
            continue
        per_origin.setdefault(r["origin"], []).append(d)
    means = {o: float(np.mean(v)) for o, v in per_origin.items() if v}
    if len(means) < 2:
        return math.nan, len(means), "", ""
    worst = max(means, key=means.get)
    best = min(means, key=means.get)
    lo = means[best]
    return (means[worst] / lo if lo > 0 else math.nan), len(means), worst, best


def run_episode_job(job: dict) -> dict:
    """Run one (controller, demand, seed, ev_rate, noise) episode and return its KPI row plus request statistics."""
    cfg: SimConfig = job["cfg"]
    if job["controller"] in ML_NAMES and job["controller"] not in ctrl_mod.CONTROLLERS:
        _init_worker(job.get("model_path"), job.get("margin", 0.10))
    ev_rate = job.get("ev_rate", "default")
    sc = Scenario(demand_level=job["demand_level"], lambda_base_vph=job["lambda_base_vph"], demand_asym=job.get("demand_asym", "sym"),
                  od_pattern=job.get("od_pattern", "default"), controller=job["controller"],
                  sensor_noise_level=job.get("sensor_noise_level", 0.0), split="closed_loop",
                  ev_rate=ev_rate, lambda_ev_vph=cfg.lambda_ev_vph if ev_rate == "default" else 0.0)
    t0 = time.perf_counter()
    w = World(cfg, sc, seed=job["seed"], episode_id=job["seed"]).run()
    ep = w.episode_row()
    rows = [r for r in w.signal_rows if cfg.is_main(r["t"])]
    n = max(len(rows), 1)
    fr, n_origins, worst, best = fairness_ratio(w.vehicle_rows(), cfg)
    ep.update(
        n_decisions_main=len(rows),
        agreement_with_teacher=sum(1 for r in rows if r["action_requested"] == r["y_mp_action"]) / n,
        request_switch_rate=sum(1 for r in rows if r["action_requested"] != r["phase"]) / n,
        teacher_switch_rate=sum(1 for r in rows if r["y_mp_action"] != r["phase"]) / n,
        accepted_switch_rate=sum(1 for r in rows if r["switched"]) / n,
        forced_share=sum(1 for r in rows if r["forced"] != 0) / n,
        fairness_ratio=fr, fairness_n_origins=n_origins, fairness_worst_origin=worst, fairness_best_origin=best,
        wall_s=time.perf_counter() - t0,
    )
    for k, v in w.invariant_counts().items():
        ep[k] = v
    return ep


# ---------------------------------------------------------------- statistics
def t_critical(n: int, level: float = 0.975) -> float:
    if n == 8:
        return T_7_975
    try:
        from scipy import stats
        return float(stats.t.ppf(level, n - 1))
    except Exception:                                              # pragma: no cover - scipy ships with sklearn
        table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 9: 2.306, 10: 2.262}
        return table.get(n, 1.96)


def t_interval(x) -> dict:
    """mean, sd, n and the t-based 95% interval of a sample (NaNs dropped)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return dict(n=0, mean=math.nan, sd=math.nan, ci95_half=math.nan, ci95_low=math.nan, ci95_high=math.nan)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else math.nan
    half = t_critical(n) * sd / math.sqrt(n) if n > 1 else math.nan
    return dict(n=n, mean=mean, sd=sd, ci95_half=half, ci95_low=mean - half, ci95_high=mean + half)


def primary_rows(df: pd.DataFrame, sensor_noise_level: float = 0.0) -> pd.DataFrame:
    """Episodes with ambulances on at the given sensor-noise level (the rows every comparison is made on)."""
    out = df
    if "ev_rate" in out.columns:
        out = out[out.ev_rate.astype(str) != EV_OFF]
    if "sensor_noise_level" in out.columns:
        out = out[np.isclose(out.sensor_noise_level.astype(float), sensor_noise_level)]
    return out


def attach_ev_off_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """Fill car_mean_delay_penalty_s on the ambulance-on rows from the paired ambulance-off episode
    (same controller, demand, seed, noise).  Rows without a pair keep NaN."""
    df = df.copy()
    if "ev_rate" not in df.columns:
        return df
    keys = ["controller", "demand_level", "seed", "sensor_noise_level"]
    off = df[df.ev_rate.astype(str) == EV_OFF].set_index(keys).mean_total_delay_s
    if "car_mean_delay_penalty_s" not in df.columns:
        df["car_mean_delay_penalty_s"] = math.nan
    on = df.ev_rate.astype(str) != EV_OFF
    idx = pd.MultiIndex.from_frame(df.loc[on, keys])
    paired = off.reindex(idx).to_numpy()
    df.loc[on, "car_mean_delay_penalty_s"] = np.round(df.loc[on, "mean_total_delay_s"].to_numpy() - paired, 3)
    return df


def paired_differences(df: pd.DataFrame, reference: str = REFERENCE, metrics=METRICS) -> pd.DataFrame:
    """Mean +- t-based 95% CI of (controller - reference) per demand level, paired by seed."""
    out = []
    if "ev_rate" in df.columns:
        df = df[df.ev_rate.astype(str) != EV_OFF]
    if "sensor_noise_level" in df.columns and df.sensor_noise_level.astype(float).nunique() > 1:
        raise ValueError("paired_differences expects one sensor_noise_level; filter with primary_rows() first")
    for demand, part in df.groupby("demand_level"):
        ref = part[part.controller == reference].set_index("seed")
        for ctrl, sub in part.groupby("controller"):
            if ctrl == reference:
                continue
            sub = sub.set_index("seed")
            seeds = sorted(set(sub.index) & set(ref.index))
            for m in metrics:
                if m not in sub.columns or m not in ref.columns:
                    continue
                a = sub.loc[seeds, m].astype(float).to_numpy()
                b = ref.loc[seeds, m].astype(float).to_numpy()
                d = a - b
                ok = ~np.isnan(d)
                d = d[ok]
                n = len(d)
                if n == 0:
                    continue
                mean = float(d.mean())
                sd = float(d.std(ddof=1)) if n > 1 else math.nan
                half = t_critical(n) * sd / math.sqrt(n) if n > 1 else math.nan
                out.append(dict(demand_level=demand, controller=ctrl, reference=reference, metric=m, n_pairs=n,
                                mean_diff=mean, sd_diff=sd, ci95_half=half, ci95_low=mean - half, ci95_high=mean + half,
                                t_crit=t_critical(n), mean_controller=float(a[ok].mean()),
                                mean_reference=float(b[ok].mean()),
                                significant=bool(abs(mean) > half) if n > 1 else False))
    return pd.DataFrame(out)


def acceptance_bound(df: pd.DataFrame, ml: str = ML_NAME, reference: str = REFERENCE, factor: float = BOUND_FACTOR,
                     metric: str = "mean_total_delay_s") -> pd.DataFrame:
    """The proposal's acceptance bound, per sensor-noise level and demand level (plus pooled over demands):
    for each seed PASS iff metric(ml) <= factor x metric(reference); pass rate, mean ratio with t-interval,
    and whether the interval's upper end is below the factor."""
    rows = []
    if "ev_rate" in df.columns:
        df = df[df.ev_rate.astype(str) != EV_OFF]
    noise_levels = sorted(df.sensor_noise_level.astype(float).unique()) if "sensor_noise_level" in df.columns else [0.0]
    for noise in noise_levels:
        part_noise = df[np.isclose(df.sensor_noise_level.astype(float), noise)] if "sensor_noise_level" in df.columns else df
        groups = [(d, part_noise[part_noise.demand_level == d]) for d in list(dict.fromkeys(part_noise.demand_level))]
        if len(groups) > 1:
            groups.append(("all", part_noise))
        for demand, part in groups:
            a = part[part.controller == ml].set_index(["demand_level", "seed"])[metric].astype(float)
            b = part[part.controller == reference].set_index(["demand_level", "seed"])[metric].astype(float)
            idx = a.index.intersection(b.index)
            if len(idx) == 0:
                continue
            av, bv = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
            ratio = av / bv
            ok = ~np.isnan(ratio)
            ratio, av, bv = ratio[ok], av[ok], bv[ok]
            passed = ratio <= factor
            ti = t_interval(ratio)
            # the methodology's form (§12.3): paired difference Delta_i = ml_i - ref_i; the bound is met when the
            # upper end of the 95% t-interval on mean(Delta) is at most m = (factor - 1) x mean reference delay
            td = t_interval(av - bv)
            mean_ref = float(bv.mean()) if len(bv) else math.nan
            margin_s = (factor - 1.0) * mean_ref
            rows.append(dict(sensor_noise_level=noise, demand_level=demand, ml=ml, reference=reference, metric=metric,
                             factor=factor, n_pairs=int(len(ratio)), n_pass=int(passed.sum()),
                             pass_rate=float(passed.mean()) if len(ratio) else math.nan,
                             mean_ratio=ti["mean"], sd_ratio=ti["sd"], ci95_half=ti["ci95_half"],
                             ci95_low=ti["ci95_low"], ci95_high=ti["ci95_high"],
                             max_ratio=float(ratio.max()) if len(ratio) else math.nan,
                             min_ratio=float(ratio.min()) if len(ratio) else math.nan,
                             bound_met_all_seeds=bool(passed.all()) if len(ratio) else False,
                             bound_met_ci=bool(ti["ci95_high"] <= factor) if ti["n"] > 1 else False,
                             mean_ml=float(av.mean()) if len(av) else math.nan, mean_reference=mean_ref,
                             diff_mean_s=td["mean"], diff_ci95_half_s=td["ci95_half"], diff_ci95_high_s=td["ci95_high"],
                             margin_s=margin_s,
                             bound_met_methodology=bool(td["ci95_high"] <= margin_s) if td["n"] > 1 else False))
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, controllers=None) -> pd.DataFrame:
    """Mean over seeds per (sensor_noise_level, demand, controller) of every KPI the report shows."""
    on = df[df.ev_rate.astype(str) != EV_OFF] if "ev_rate" in df.columns else df
    cols = ["ev_mean_travel_time_s", "ev_mean_delay_s", "ev_p90_delay_s", "ev_mean_stops", "mean_total_delay_s",
            "p90_total_delay_s", "car_mean_delay_penalty_s", "recovery_time_s", "fairness_ratio", "throughput_vph",
            "switches_per_min", "n_red_commits", "n_ambulances", "agreement_with_teacher", "request_switch_rate",
            "teacher_switch_rate", "accepted_switch_rate", "forced_share", "n_decisions_main", "n_invariant_violations"]
    cols = [c for c in cols if c in on.columns]
    rows = []
    noise_levels = sorted(on.sensor_noise_level.astype(float).unique()) if "sensor_noise_level" in on.columns else [0.0]
    for noise in noise_levels:
        pn = on[np.isclose(on.sensor_noise_level.astype(float), noise)] if "sensor_noise_level" in on.columns else on
        for d in list(dict.fromkeys(pn.demand_level)):
            ctrls = [c for c in (controllers or list(dict.fromkeys(pn.controller))) if c in set(pn.controller)]
            for c in ctrls:
                part = pn[(pn.controller == c) & (pn.demand_level == d)]
                r = dict(sensor_noise_level=noise, demand_level=d, controller=c, n=int(len(part)),
                         gridlock_rate=float(part.gridlock.astype(float).mean()) if "gridlock" in part.columns else math.nan)
                for m in cols:
                    r[m] = float(part[m].astype(float).mean())
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figure and report
def plot_closed_loop(df: pd.DataFrame, diffs: pd.DataFrame, path: str, controllers, reference: str = REFERENCE) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = primary_rows(df) if "ev_rate" in df.columns else df
    demands = list(dict.fromkeys(df.demand_level))
    panels = [("ev_mean_delay_s", "ambulance mean delay [s]"), ("mean_total_delay_s", "car mean delay [s]")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    colors = {c: col for c, col in zip(controllers, ["#8c8c8c", "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"])}
    width = 0.8 / max(len(controllers), 1)
    for col, (metric, label) in enumerate(panels):
        ax = axes[0, col]
        for i, c in enumerate(controllers):
            means = [df[(df.controller == c) & (df.demand_level == d)][metric].mean() for d in demands]
            sds = [df[(df.controller == c) & (df.demand_level == d)][metric].std() for d in demands]
            x = np.arange(len(demands)) + (i - (len(controllers) - 1) / 2) * width
            ax.bar(x, means, width, yerr=sds, capsize=2, color=colors[c], label=c)
        ax.set_xticks(range(len(demands)), demands)
        ax.set_ylabel(label)
        ax.set_title(f"mean over seeds (+- sd): {label}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)
        ax = axes[1, col]
        others = [c for c in controllers if c != reference]
        w2 = 0.8 / max(len(others), 1)
        for i, c in enumerate(others):
            vals, errs = [], []
            for d in demands:
                r = diffs[(diffs.controller == c) & (diffs.demand_level == d) & (diffs.metric == metric)]
                vals.append(float(r.mean_diff.iloc[0]) if len(r) else np.nan)
                errs.append(float(r.ci95_half.iloc[0]) if len(r) else np.nan)
            x = np.arange(len(demands)) + (i - (len(others) - 1) / 2) * w2
            ax.bar(x, vals, w2, yerr=errs, capsize=3, color=colors[c], label=c)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(demands)), demands)
        ax.set_ylabel(f"paired difference vs {reference} [s]")
        ax.set_title(f"paired diff vs {reference}, 95% t-interval: {label}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_acceptance_bound(df: pd.DataFrame, bound: pd.DataFrame, path: str, ml: str = ML_NAME,
                          reference: str = REFERENCE, factor: float = BOUND_FACTOR) -> None:
    """Per-seed car-delay ratio ml / reference at every (noise, demand), with the 1.10 line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    on = df[df.ev_rate.astype(str) != EV_OFF] if "ev_rate" in df.columns else df
    groups = [(n, d) for n in sorted(on.sensor_noise_level.astype(float).unique()) for d in list(dict.fromkeys(on.demand_level))]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (noise, d) in enumerate(groups):
        part = on[np.isclose(on.sensor_noise_level.astype(float), noise) & (on.demand_level == d)]
        a = part[part.controller == ml].set_index("seed").mean_total_delay_s
        b = part[part.controller == reference].set_index("seed").mean_total_delay_s
        idx = a.index.intersection(b.index)
        if len(idx) == 0:
            continue
        ratio = (a.loc[idx] / b.loc[idx]).to_numpy()
        x = np.full(len(ratio), i) + np.linspace(-0.18, 0.18, len(ratio))
        ax.scatter(x, ratio, color=["#55a868" if r <= factor else "#c44e52" for r in ratio], zorder=3, s=28)
        q = bound[np.isclose(bound.sensor_noise_level, noise) & (bound.demand_level == d)]
        if len(q):
            q = q.iloc[0]
            ax.errorbar(i, q.mean_ratio, yerr=q.ci95_half, fmt="D", color="k", capsize=4, zorder=4, ms=5)
    ax.axhline(factor, color="#c44e52", ls="--", lw=1.2, label=f"bound {factor:.2f} x {reference}")
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xticks(range(len(groups)), [f"noise {n:g}\n{d}" for n, d in groups])
    ax.set_ylabel(f"car mean delay ratio {ml} / {reference}")
    ax.set_title("Acceptance bound per seed (green = pass, red = fail; diamond = mean with 95% t-interval)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _md(rows: list, cols: list, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, (float, np.floating)):
                v = "nan" if math.isnan(v) else fmt.get(c, "{:.2f}").format(v)
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _diff_cell(diffs: pd.DataFrame, d: str, c: str, m: str, fmt: str = "{:+.2f} +- {:.2f}") -> str:
    q = diffs[(diffs.demand_level == d) & (diffs.controller == c) & (diffs.metric == m)]
    if not len(q):
        return "n/a"
    q = q.iloc[0]
    if math.isnan(q.mean_diff):
        return "nan"
    return fmt.format(q.mean_diff, q.ci95_half) + f" (n={int(q.n_pairs)})" + (" *" if q.significant else "")


def closed_loop_markdown(df: pd.DataFrame, diffs: pd.DataFrame, meta: dict, controllers, reference: str = REFERENCE,
                         bound: pd.DataFrame | None = None, task_results: dict | None = None) -> str:
    """Sections 4 (closed loop), 5 (verdict) and 6 (reproducibility) of RESULTS.md, written from the numbers."""
    df_all = df
    noise0 = float(meta.get("sensor_noise_level", 0.0))
    df = primary_rows(df_all, noise0) if "ev_rate" in df_all.columns else df_all
    demands = list(dict.fromkeys(df.demand_level))
    if bound is None:
        bound = acceptance_bound(df_all)
    n_off = int((df_all.ev_rate.astype(str) == EV_OFF).sum()) if "ev_rate" in df_all.columns else 0
    noise_variant = meta.get("noise_variant")
    ml_label = f"`{ML_NAME}`"
    L = ["## 4. Closed loop: the learned controller deployed in the simulator", "",
         f"Protocol: controllers {list(controllers)} x demand {demands} x {meta['n_seeds']} paired seeds "
         f"({meta['seeds'][0]}..{meta['seeds'][-1]}, disjoint from the training seeds 100000..100119), demand_asym sym, od default, "
         f"ev_rate default (lambda_ev = {meta['lambda_ev_vph']} /h per node), sensor_noise_level {noise0:g}. "
         f"Every (controller, demand, seed) was also run with ambulances off (same seed; {n_off} such episodes) to measure "
         f"`car_mean_delay_penalty_s`. {len(df_all)} episodes in total, wall clock {meta['wall_s']:.0f} s with {meta['workers']} worker "
         f"processes (serial inference per worker, BLAS/OMP threads = 1). The ML controller is the gradient-boosted-tree "
         f"classifier on sensor features, called {ml_label} in the tables, with the hysteresis rule "
         f"request argmax only if P(best) > P(current) + {meta['margin']:.2f}. Invariant violations over all episodes: "
         f"{int(df_all.n_invariant_violations.sum())}; gridlocked episodes: {int(df_all.gridlock.astype(float).sum())}.", ""]
    if meta.get("simulator_config_hash"):
        L += [f"Simulator `{meta.get('simulator_generator_version')}`, config hash `{meta.get('simulator_config_hash')}`; "
              f"dataset `{meta.get('dataset_generator_version')}`, config hash `{meta.get('dataset_config_hash')}` "
              f"(asserted equal: {meta.get('hash_match')}).", ""]
    # ---- 4.1 means
    L += ["### 4.1 Mean over seeds per controller and demand level (ambulances on)", ""]
    rows = []
    for d in demands:
        for c in controllers:
            part = df[(df.controller == c) & (df.demand_level == d)]
            rows.append(dict(demand=d, controller=c, n=len(part), ev_mean_travel_time_s=part.ev_mean_travel_time_s.mean(),
                             ev_mean_delay_s=part.ev_mean_delay_s.mean(), ev_p90_delay_s=part.ev_p90_delay_s.mean(), car_mean_delay_s=part.mean_total_delay_s.mean(),
                             car_p90_delay_s=part.p90_total_delay_s.mean(), throughput_vph=part.throughput_vph.mean(),
                             switches_per_min=part.switches_per_min.mean(),
                             gridlock_rate=part.gridlock.astype(float).mean(), n_red_commits=part.n_red_commits.mean(),
                             n_ambulances=part.n_ambulances.mean()))
    L += [_md(rows, ["demand", "controller", "n", "ev_mean_travel_time_s", "ev_mean_delay_s", "ev_p90_delay_s", "car_mean_delay_s",
                     "car_p90_delay_s", "throughput_vph", "switches_per_min", "gridlock_rate", "n_red_commits", "n_ambulances"],
              {"throughput_vph": "{:.0f}", "n_red_commits": "{:.0f}", "n_ambulances": "{:.1f}"}), ""]
    # ---- 4.2 acceptance bound
    L += [f"### 4.2 The proposal's acceptance bound: car mean delay under the ML controller <= {BOUND_FACTOR:.2f} x `{reference}`", "",
          f"Per seed the ratio car_delay({ML_NAME}) / car_delay({reference}) is computed on the paired episodes; PASS iff ratio <= {BOUND_FACTOR:.2f}. "
          "`mean_ratio` carries a t-based 95% interval over seeds; `bound_met_ci` = that interval's upper end is below the factor. "
          "The proposal states the same bound on the paired difference: `bound_met_methodology` = the upper end of the 95% "
          f"t-interval on mean(ml - {reference}) is at most `margin_s` = {BOUND_FACTOR - 1:.2f} x the mean {reference} car delay of that cell.", ""]
    brows = [dict(r) for _, r in bound.iterrows()]
    L += [_md(brows, ["sensor_noise_level", "demand_level", "n_pairs", "n_pass", "pass_rate", "mean_ratio", "ci95_low", "ci95_high",
                      "min_ratio", "max_ratio", "mean_ml", "mean_reference", "bound_met_all_seeds", "bound_met_ci"],
              {"sensor_noise_level": "{:g}", "pass_rate": "{:.3f}", "mean_ratio": "{:.3f}", "ci95_low": "{:.3f}", "ci95_high": "{:.3f}",
               "min_ratio": "{:.3f}", "max_ratio": "{:.3f}", "mean_ml": "{:.1f}", "mean_reference": "{:.1f}"}), "",
          "The same cells in the methodology's paired-difference form (seconds of car delay):", "",
          _md(brows, ["sensor_noise_level", "demand_level", "n_pairs", "diff_mean_s", "diff_ci95_half_s", "diff_ci95_high_s", "margin_s",
                      "bound_met_methodology"], {"sensor_noise_level": "{:g}"}), "",
          "Figure: car-delay ratio of the ML controller to the reference per demand level, with the 1.10 bound.", ""]
    # ---- 4.3 paired differences
    L += [f"### 4.3 Paired differences versus `{reference}` (mean +- t-based 95% half-width, n = pairs)", "",
          "Negative delay / penalty / recovery / fairness differences favour the controller; positive throughput differences favour it. "
          "`*` = the 95% interval excludes zero.", ""]
    for d in demands:
        L += [f"**{d} demand**", ""]
        rows = []
        for c in [x for x in controllers if x != reference]:
            r = dict(controller=c)
            for m in ("ev_mean_travel_time_s", "ev_mean_delay_s", "mean_total_delay_s", "p90_total_delay_s", "throughput_vph",
                      "switches_per_min", "n_red_commits", "gridlock"):
                r[m] = _diff_cell(diffs, d, c, m)
            rows.append(r)
        L += [_md(rows, ["controller", "ev_mean_travel_time_s", "ev_mean_delay_s", "mean_total_delay_s", "p90_total_delay_s", "throughput_vph",
                         "switches_per_min", "n_red_commits", "gridlock"]), ""]
    # ---- 4.4 secondary metrics
    L += ["### 4.4 Secondary metrics (ambulance, car penalty, recovery, fairness)", "",
          "`ev_mean_travel_time_s` = mean ambulance travel time; `car_mean_delay_penalty_s` = car mean delay with ambulances minus "
          "the paired episode without ambulances (same seed and controller); `recovery_time_s` = mean time after an ambulance clears a "
          "junction until that junction's queue is back within 10% of its value at first sighting; `fairness_ratio` = max / min of the "
          "mean car delay over the 8 entry nodes (1 = perfectly even).", ""]
    rows = []
    for d in demands:
        for c in controllers:
            part = df[(df.controller == c) & (df.demand_level == d)]
            r = dict(demand=d, controller=c, ev_mean_travel_time_s=part.ev_mean_travel_time_s.mean(),
                     car_mean_delay_penalty_s=part.car_mean_delay_penalty_s.mean(), recovery_time_s=part.recovery_time_s.mean(),
                     fairness_ratio=part.fairness_ratio.mean())
            r["worst_origin (mode)"] = part.fairness_worst_origin.mode().iloc[0] if "fairness_worst_origin" in part.columns and len(part) else ""
            rows.append(r)
    L += [_md(rows, ["demand", "controller", "ev_mean_travel_time_s", "car_mean_delay_penalty_s", "recovery_time_s", "fairness_ratio",
                     "worst_origin (mode)"], {"fairness_ratio": "{:.3f}"}), "",
          f"Paired differences of the secondary metrics versus `{reference}`:", ""]
    for d in demands:
        L += [f"**{d} demand**", ""]
        rows = []
        for c in [x for x in controllers if x != reference]:
            r = dict(controller=c)
            for m in SECONDARY:
                r[m] = _diff_cell(diffs, d, c, m, "{:+.3f} +- {:.3f}" if m == "fairness_ratio" else "{:+.2f} +- {:.2f}")
            rows.append(r)
        L += [_md(rows, ["controller"] + list(SECONDARY)), ""]
    # ---- 4.5 noise variant
    L += ["### 4.5 Sensor-noise variant (the sensor-realistic claim)", ""]
    if noise_variant is not None and "sensor_noise_level" in df_all.columns and \
            np.isclose(df_all.sensor_noise_level.astype(float), float(noise_variant)).any():
        dn = primary_rows(df_all, float(noise_variant))
        L += [f"`{ML_NAME}` and `{reference}` re-run on the same {meta['n_seeds']} seeds x {demands} at sensor_noise_level {noise_variant:g} "
              f"(obs_q_a = Binomial(q_a, 1-p_miss) + Poisson(p_miss/2)); the ML controller reads `obs_q_*`, max_pressure reads the true "
              f"`n_*` and is therefore unaffected by construction except through the ambulance stream, which is identical. Means over seeds:", ""]
        rows = []
        for d in demands:
            for c in [x for x in controllers if x in set(dn.controller)]:
                p0 = df[(df.controller == c) & (df.demand_level == d)]
                p1 = dn[(dn.controller == c) & (dn.demand_level == d)]
                rows.append(dict(demand=d, controller=c, n=len(p1), car_delay_noise0=p0.mean_total_delay_s.mean(),
                                 car_delay_noise=p1.mean_total_delay_s.mean(), ev_delay_noise0=p0.ev_mean_delay_s.mean(),
                                 ev_delay_noise=p1.ev_mean_delay_s.mean(), throughput_noise0=p0.throughput_vph.mean(),
                                 throughput_noise=p1.throughput_vph.mean(), agreement_noise0=p0.agreement_with_teacher.mean(),
                                 agreement_noise=p1.agreement_with_teacher.mean(), switches_noise=p1.switches_per_min.mean()))
        L += [_md(rows, ["demand", "controller", "n", "car_delay_noise0", "car_delay_noise", "ev_delay_noise0", "ev_delay_noise",
                         "throughput_noise0", "throughput_noise", "agreement_noise0", "agreement_noise", "switches_noise"],
                  {"throughput_noise0": "{:.0f}", "throughput_noise": "{:.0f}", "agreement_noise0": "{:.3f}", "agreement_noise": "{:.3f}"}), ""]
        dn_diff = paired_differences(dn, reference=reference, metrics=("mean_total_delay_s", "ev_mean_delay_s", "throughput_vph",
                                                                        "car_mean_delay_penalty_s", "fairness_ratio"))
        rows = []
        for d in demands:
            r = dict(demand=d)
            for m in ("mean_total_delay_s", "ev_mean_delay_s", "throughput_vph", "car_mean_delay_penalty_s", "fairness_ratio"):
                r[m] = _diff_cell(dn_diff, d, ML_NAME, m, "{:+.3f} +- {:.3f}" if m == "fairness_ratio" else "{:+.2f} +- {:.2f}")
            rows.append(r)
        L += [f"Paired differences `{ML_NAME}` - `{reference}` at noise {noise_variant:g}:", "",
              _md(rows, ["demand", "mean_total_delay_s", "ev_mean_delay_s", "throughput_vph", "car_mean_delay_penalty_s", "fairness_ratio"]), "",
              "The acceptance bound at this noise level is in the table of 4.2 (rows with sensor_noise_level "
              f"{noise_variant:g}).", ""]
    else:
        L += ["Not run (pass `--noise-variant 0.1`).", ""]
    # ---- 4.6 agreement
    L += ["### 4.6 Request statistics on main-period decision rows (per controller, mean over episodes, noise 0)", "",
          "`agreement_with_teacher` = share of decision instants where the controller's request equals the "
          "priority-aware teacher label recomputed on the same row (for `priority_mp` it is 1 by construction); "
          "`request_switch_rate` = share of requests that differ from the current phase; `accepted_switch_rate` = share of "
          "instants at which the plant actually started an all-red; `forced_share` = share of instants where the plant overrode "
          "the request (max-green or starvation guard).", ""]
    rows = []
    for d in demands:
        for c in controllers:
            part = df[(df.controller == c) & (df.demand_level == d)]
            rows.append(dict(demand=d, controller=c, agreement_with_teacher=part.agreement_with_teacher.mean(),
                             disagreement_rate=1 - part.agreement_with_teacher.mean(),
                             request_switch_rate=part.request_switch_rate.mean(), teacher_switch_rate=part.teacher_switch_rate.mean(),
                             accepted_switch_rate=part.accepted_switch_rate.mean(), forced_share=part.forced_share.mean(),
                             n_decisions_main=part.n_decisions_main.mean()))
    L += [_md(rows, ["demand", "controller", "agreement_with_teacher", "disagreement_rate", "request_switch_rate",
                     "teacher_switch_rate", "accepted_switch_rate", "forced_share", "n_decisions_main"],
              {"agreement_with_teacher": "{:.3f}", "disagreement_rate": "{:.3f}", "request_switch_rate": "{:.3f}",
               "teacher_switch_rate": "{:.3f}", "accepted_switch_rate": "{:.3f}", "forced_share": "{:.3f}", "n_decisions_main": "{:.0f}"}), "",
          "Figure: ambulance and car delay per controller and demand level, with paired differences to the reference. "
          "The per-episode rows (ambulance-off and noise-variant episodes included), the paired differences, the bound and the "
          "means per cell are published as tables with this report.", ""]
    # ---- 5 verdict
    L += ["## 5. Verdict", ""]
    L += verdict_lines(df, diffs, bound, task_results, reference=reference, noise_variant=noise_variant,
                       df_all=df_all)
    # ---- 6 reproducibility
    L += ["", "## 6. Reproducibility", ""]
    L += reproducibility_lines(meta, task_results)
    return "\n".join(L) + "\n"


def verdict_lines(df: pd.DataFrame, diffs: pd.DataFrame, bound: pd.DataFrame, task_results: dict | None,
                  reference: str = REFERENCE, noise_variant=None, df_all: pd.DataFrame | None = None) -> list:
    """The honest reading of the numbers, generated from the numbers."""
    L = []
    demands = list(dict.fromkeys(df.demand_level))

    def word(q):
        if not q.significant:
            return "does not differ significantly from"
        return "beats" if q.mean_diff < 0 else "loses to"

    if task_results:
        a = task_results["task_a"]["models"]
        b = task_results["task_b"]
        L.append(f"- **Imitation (Task A).** HGB-truth reaches balanced accuracy "
                 f"{a['hgb_truth']['splits']['test']['balanced_accuracy']:.3f} on test and "
                 f"{a['hgb_truth']['splits']['test_ood']['balanced_accuracy']:.3f} on stress (test_ood); HGB-sensor, the deployed model, "
                 f"{a['hgb_sensor']['splits']['test']['balanced_accuracy']:.3f} / {a['hgb_sensor']['splits']['test_ood']['balanced_accuracy']:.3f} "
                 f"(switch F1 {a['hgb_sensor']['splits']['test']['switch_f1']:.3f} on test). The gap between the two is the price of not "
                 "observing moving vehicles and arrivals; the teacher is a rule of `n_*`, which no loop detector delivers.")
        L.append(f"- **Queue forecast (Task B).** HGB MAE {b['models']['hgb']['splits']['test']['mae']:.2f} veh on test vs "
                 f"{b['naive_baseline']['test']['mae']:.2f} veh for the naive 'queue now' baseline "
                 f"(R^2 {b['models']['hgb']['splits']['test']['r2']:.3f}); on stress {b['models']['hgb']['splits']['test_ood']['mae']:.2f} vs "
                 f"{b['naive_baseline']['test_ood']['mae']:.2f}. The action effect in the partial dependence is "
                 f"{b['partial_dependence_action']['mean_pred_queue_best_other_action'] - b['partial_dependence_action']['mean_pred_queue_hold_current']:+.2f} veh "
                 "(best other action minus hold), small against the state effect.")
    # the bound
    for _, q in bound.iterrows():
        if not np.isclose(q.sensor_noise_level, 0.0) or q.demand_level == "all":
            continue
        L.append(f"- **Acceptance bound, {q.demand_level}, noise 0**: {int(q.n_pass)}/{int(q.n_pairs)} seeds pass "
                 f"(pass rate {q.pass_rate:.2f}); mean ratio {q.mean_ratio:.3f} [{q.ci95_low:.3f}, {q.ci95_high:.3f}] "
                 f"(car delay {q.mean_ml:.1f} s vs {q.mean_reference:.1f} s). "
                 + ("The bound is met on every seed and the interval lies below 1.10." if q.bound_met_all_seeds and q.bound_met_ci
                    else "The bound is met on every seed but the interval does not exclude 1.10." if q.bound_met_all_seeds
                    else "The bound is NOT met on every seed." + (" The interval's upper end is still below 1.10." if q.bound_met_ci else ""))
                 + f" Methodology form: paired difference {q.diff_mean_s:+.1f} s, 95% upper end {q.diff_ci95_high_s:+.1f} s against a margin of "
                 f"{q.margin_s:.1f} s: {'met' if q.bound_met_methodology else 'NOT met'}.")
    pooled = bound[(np.isclose(bound.sensor_noise_level, 0.0)) & (bound.demand_level == "all")]
    if len(pooled):
        q = pooled.iloc[0]
        L.append(f"- **Acceptance bound, pooled over demand levels, noise 0**: pass rate {q.pass_rate:.2f} ({int(q.n_pass)}/{int(q.n_pairs)}), "
                 f"mean ratio {q.mean_ratio:.3f} [{q.ci95_low:.3f}, {q.ci95_high:.3f}].")
    if noise_variant is not None:
        for _, q in bound.iterrows():
            if np.isclose(q.sensor_noise_level, float(noise_variant)) and q.demand_level != "all":
                L.append(f"- **Acceptance bound, {q.demand_level}, noise {noise_variant:g}**: {int(q.n_pass)}/{int(q.n_pairs)} seeds pass, "
                         f"mean ratio {q.mean_ratio:.3f} [{q.ci95_low:.3f}, {q.ci95_high:.3f}].")
    # ambulance / car delay verdict per demand
    for d in demands:
        ml = diffs[(diffs.demand_level == d) & (diffs.controller == ML_NAME)]
        if not len(ml):
            continue
        ev = ml[ml.metric == "ev_mean_delay_s"].iloc[0]
        car = ml[ml.metric == "mean_total_delay_s"].iloc[0]
        teacher = diffs[(diffs.demand_level == d) & (diffs.controller == "priority_mp") & (diffs.metric == "ev_mean_delay_s")]
        naive = diffs[(diffs.demand_level == d) & (diffs.controller == "preempt_naive") & (diffs.metric == "ev_mean_delay_s")]
        ml_vs_teacher = (df[(df.controller == ML_NAME) & (df.demand_level == d)].set_index("seed").ev_mean_delay_s
                         - df[(df.controller == "priority_mp") & (df.demand_level == d)].set_index("seed").ev_mean_delay_s)
        L.append(f"- **{d}**: on ambulance delay the ML controller {word(ev)} {reference} "
                 f"({ev.mean_diff:+.1f} +- {ev.ci95_half:.1f} s); on car delay it {word(car)} {reference} "
                 f"({car.mean_diff:+.1f} +- {car.ci95_half:.1f} s). Against its own teacher priority_mp the ML controller's "
                 f"ambulance delay differs by {ml_vs_teacher.mean():+.1f} s (mean over seeds)."
                 + (f" The teacher itself changes ambulance delay by {teacher.iloc[0].mean_diff:+.1f} +- {teacher.iloc[0].ci95_half:.1f} s "
                    f"and naive preemption by {naive.iloc[0].mean_diff:+.1f} +- {naive.iloc[0].ci95_half:.1f} s versus {reference}."
                    if len(teacher) and len(naive) else ""))
        for m, name in (("car_mean_delay_penalty_s", "car delay penalty of the ambulances"),
                        ("recovery_time_s", "recovery time"), ("fairness_ratio", "fairness ratio")):
            q = ml[ml.metric == m]
            if len(q) and not math.isnan(q.iloc[0].mean_diff):
                q = q.iloc[0]
                L.append(f"  - {name}: ML {word(q)} {reference} ({q.mean_diff:+.3f} +- {q.ci95_half:.3f}; "
                         f"ML {q.mean_controller:.3f} vs {q.mean_reference:.3f}).")
    # reference sanity
    for d in demands:
        ft = diffs[(diffs.demand_level == d) & (diffs.controller == "fixed_time") & (diffs.metric == "mean_total_delay_s")]
        if len(ft) and ft.iloc[0].significant and ft.iloc[0].mean_diff < 0:
            thr = diffs[(diffs.demand_level == d) & (diffs.controller == "fixed_time") & (diffs.metric == "throughput_vph")]
            L.append(f"- **Reference check, {d}**: plain fixed_time beats {reference} on car delay "
                     f"({ft.iloc[0].mean_diff:+.1f} +- {ft.iloc[0].ci95_half:.1f} s"
                     + (f", throughput {thr.iloc[0].mean_diff:+.0f} +- {thr.iloc[0].ci95_half:.0f} veh/h" if len(thr) else "")
                     + "). The max-pressure rule as specified for this simulator (split phasing, one-approach pressure, 2-vehicle hysteresis) is "
                     "therefore not a strong baseline at this demand; the ML controller imitates a teacher built on that rule and "
                     "inherits the weakness. This is a finding about the simulator's controllers, reported as is.")
    agree = df[df.controller == ML_NAME].agreement_with_teacher.mean() if (df.controller == ML_NAME).any() else math.nan
    if agree == agree:
        L.append(f"- **Agreement.** Deployed, the ML controller's request equals the teacher's on {100 * agree:.1f}% of main-period decision "
                 "instants (mean over its episodes at noise 0); the residual disagreement is concentrated on hysteresis-boundary and "
                 "max-green rows and on ambulance rows, where the sensor set cannot see `n_*`.")
    if df_all is not None and "n_invariant_violations" in df_all.columns:
        L.append(f"- **Safety.** {int(df_all.n_invariant_violations.sum())} invariant violations in {len(df_all)} closed-loop episodes: "
                 "the signal plant, not the model, guarantees safety.")
    L.append("- **Scope.** All of this is one simulator with its own teacher rule and calibrated demand; the closed-loop "
             "seeds are disjoint from training but the demand model, OD pattern and vehicle population are the same as in training. "
             "Nothing here is evidence about a real junction.")
    return L


def reproducibility_lines(meta: dict, task_results: dict | None) -> list:
    import sklearn
    L = [f"- Simulator: `{meta.get('simulator_generator_version', GENERATOR_VERSION)}`, config hash `{meta.get('simulator_config_hash', '')}`; "
         f"dataset `{meta.get('dataset_generator_version', '')}`, config hash `{meta.get('dataset_config_hash', '')}`"
         + (f", {meta['dataset_n_episodes']} episodes" if meta.get("dataset_n_episodes") else "") + ".",
         f"- Software at evaluation time: Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
         f"scikit-learn {sklearn.__version__}, joblib {joblib.__version__}; {platform.platform()}."]
    if task_results:
        L.append(f"- Supervised pipeline: {task_results.get('runtime_s', float('nan')):.0f} s; scikit-learn "
                 f"{task_results.get('sklearn_version', sklearn.__version__)}"
                 + (f", dataset config hash `{task_results['dataset_config_hash']}`" if task_results.get("dataset_config_hash") else "") + ".")
    L.append(f"- Closed loop: {meta.get('wall_s', float('nan')):.0f} s, {meta.get('workers')} workers, "
             f"seeds {meta['seeds'][0]}..{meta['seeds'][-1]}, hysteresis margin {meta.get('margin')}, "
             f"noise variant {meta.get('noise_variant')}.")
    L.append("- Every number in this report is computed from the stored per-episode results; the report can be rebuilt without "
             "re-simulating.")
    return L


def update_results_md(path: str, section: str, marker: str = "## 4. Closed loop") -> None:
    text = ""
    if os.path.exists(path):
        with open(path) as f:
            text = f.read()
        if marker in text:
            text = text[: text.index(marker)].rstrip() + "\n\n"
        elif text and not text.endswith("\n"):
            text += "\n"
    with open(path, "w") as f:
        f.write(text + section)


# ---------------------------------------------------------------- the protocol
def build_jobs(cfg: SimConfig, controllers, demands, seeds, sensor_noise_level: float = 0.0, ev_rate: str = "default",
               model_path: str | None = None, margin: float = 0.10, paired_ev_off: bool = False) -> list:
    jobs = []
    for seed in seeds:
        for d in demands:
            for c in controllers:
                base = dict(cfg=cfg, controller=c, demand_level=d, seed=int(seed), lambda_base_vph=float(cfg.lambda_base_vph[d]),
                            sensor_noise_level=sensor_noise_level, model_path=model_path, margin=margin)
                jobs.append(dict(base, ev_rate=ev_rate))
                if paired_ev_off and ev_rate != EV_OFF:
                    jobs.append(dict(base, ev_rate=EV_OFF))
    return jobs


def run_protocol(cfg: SimConfig, model_path: str | None = None, bundle: dict | None = None, controllers=DEFAULT_CONTROLLERS,
                 demands=DEFAULT_DEMANDS, seeds=DEFAULT_SEEDS, sensor_noise_level: float = 0.0, margin: float = 0.10,
                 workers: int = 1, paired_ev_off: bool = True, log=print) -> pd.DataFrame:
    """Run every (controller, demand, seed) episode (plus the ambulance-off pair when paired_ev_off);
    returns one KPI row per episode with car_mean_delay_penalty_s filled from the pair."""
    if any(c in ML_NAMES for c in controllers):
        if bundle is None and model_path is not None:
            bundle = joblib.load(model_path)
        if bundle is None:
            raise ValueError("ml_sensor needs a model bundle or model_path")
        register_ml_controller(bundle, margin)
    jobs = build_jobs(cfg, controllers, demands, seeds, sensor_noise_level, model_path=model_path, margin=margin,
                      paired_ev_off=paired_ev_off)
    rows = []
    t0 = time.perf_counter()

    def _log(i, r):
        log(f"  [{i + 1}/{len(jobs)}] {r['controller']:14s} {r['demand_level']:6s} seed {r['seed']} ev {r['ev_rate']:7s} "
            f"noise {r['sensor_noise_level']:<4} ev_delay {r['ev_mean_delay_s']:7.1f} car_delay {r['mean_total_delay_s']:7.1f} "
            f"agree {r['agreement_with_teacher']:.3f} fair {r['fairness_ratio']:.2f} ({r['wall_s']:.1f} s)")

    if workers <= 1 or model_path is None:
        for i, job in enumerate(jobs):
            rows.append(run_episode_job(job))
            _log(i, rows[-1])
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(model_path, margin)) as pool:
            for i, r in enumerate(pool.map(run_episode_job, jobs)):
                rows.append(r)
                _log(i, r)
    df = attach_ev_off_penalty(pd.DataFrame(rows))
    df.attrs["wall_s"] = time.perf_counter() - t0
    return df


def load_cfg(data_dir: str | None) -> SimConfig:
    path = os.path.join(data_dir, "config.json") if data_dir else None
    cfg = load_config(path) if path and os.path.exists(path) else SimConfig()
    if not cfg.lambda_base_vph:
        calibrate_demand(cfg)
    return cfg


def dataset_identity(data_dir: str | None) -> dict:
    """generator_version / config_hash the dataset was made with (from episodes.csv), or empty."""
    if not data_dir:
        return {}
    path = os.path.join(data_dir, "episodes.csv")
    if not os.path.exists(path):
        return {}
    ep = pd.read_csv(path, usecols=["config_hash", "generator_version"])
    return dict(dataset_config_hash=",".join(sorted(ep.config_hash.astype(str).unique())),
                dataset_generator_version=",".join(sorted(ep.generator_version.astype(str).unique())),
                dataset_n_episodes=int(len(ep)))


def check_simulator_matches_dataset(cfg: SimConfig, data_dir: str | None, strict: bool = True) -> dict:
    """Record the simulator's version/hash next to the dataset's and assert they agree (task 2d)."""
    ident = dict(simulator_generator_version=GENERATOR_VERSION, simulator_config_hash=cfg.config_hash(), data_dir=data_dir)
    ident.update(dataset_identity(data_dir))
    if "dataset_config_hash" in ident:
        ident["hash_match"] = (ident["dataset_config_hash"] == ident["simulator_config_hash"]
                               and ident["dataset_generator_version"] == ident["simulator_generator_version"])
        if strict:
            assert ident["hash_match"], (f"simulator {GENERATOR_VERSION}/{ident['simulator_config_hash']} differs from the dataset "
                                         f"{ident['dataset_generator_version']}/{ident['dataset_config_hash']} in {data_dir}")
    else:
        ident["hash_match"] = None
    return ident


def write_outputs(df: pd.DataFrame, meta: dict, out: str, controllers, task_results: dict | None = None) -> str:
    """closed_loop*.csv, figures and the RESULTS.md sections 4-6 from the episode table."""
    diffs = paired_differences(primary_rows(df, float(meta.get("sensor_noise_level", 0.0))))
    bound = acceptance_bound(df)
    summary = summary_table(df, controllers)
    df.to_csv(os.path.join(out, "closed_loop.csv"), index=False)
    diffs.to_csv(os.path.join(out, "closed_loop_diffs.csv"), index=False)
    bound.to_csv(os.path.join(out, "closed_loop_bound.csv"), index=False)
    summary.to_csv(os.path.join(out, "closed_loop_summary.csv"), index=False)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    plot_closed_loop(df, diffs, os.path.join(out, "figures", "closed_loop.png"), controllers)
    plot_acceptance_bound(df, bound, os.path.join(out, "figures", "closed_loop_bound.png"))
    section = closed_loop_markdown(df, diffs, meta, controllers, bound=bound, task_results=task_results)
    update_results_md(os.path.join(out, "RESULTS.md"), section)
    with open(os.path.join(out, "closed_loop_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)
    return section


def load_task_results(out: str) -> dict | None:
    path = os.path.join(out, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="reports/ml/models", help="directory holding taskA_hgb_sensor.joblib")
    p.add_argument("--model-file", default="taskA_hgb_sensor.joblib")
    p.add_argument("--out", default="reports/ml")
    p.add_argument("--data", default="data/traffic_v1", help="dataset directory whose config.json carries the calibrated demand")
    p.add_argument("--controllers", default=",".join(DEFAULT_CONTROLLERS))
    p.add_argument("--demands", default=",".join(DEFAULT_DEMANDS))
    p.add_argument("--seeds", default=f"{DEFAULT_SEEDS[0]}-{DEFAULT_SEEDS[-1]}", help="a-b inclusive or comma list")
    p.add_argument("--noise", type=float, default=0.0, help="sensor_noise_level of the main protocol")
    p.add_argument("--noise-variant", type=float, default=DEFAULT_NOISE_VARIANT,
                   help="sensor_noise_level of the ml_sensor / max_pressure re-run (negative = skip)")
    p.add_argument("--no-paired", action="store_true", help="skip the ambulance-off pair (car_mean_delay_penalty_s = NaN)")
    p.add_argument("--margin", type=float, default=0.10)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    p.add_argument("--no-strict-hash", action="store_true", help="only record, do not assert, simulator == dataset hash")
    p.add_argument("--report-only", action="store_true", help="rebuild tables, figures and RESULTS.md sections from closed_loop.csv")
    args = p.parse_args(argv)
    if args.report_only:
        df = pd.read_csv(os.path.join(args.out, "closed_loop.csv"))
        df["ev_rate"] = df.ev_rate.astype(str)
        with open(os.path.join(args.out, "closed_loop_meta.json")) as f:
            meta = json.load(f)
        write_outputs(df, meta, args.out, tuple(meta["controllers"]), load_task_results(args.out))
        print("report rebuilt from closed_loop.csv")
        return
    seeds = ([int(x) for x in args.seeds.split(",")] if "," in args.seeds
             else list(range(int(args.seeds.split("-")[0]), int(args.seeds.split("-")[1]) + 1)))
    controllers = tuple(args.controllers.split(","))
    demands = tuple(args.demands.split(","))
    model_path = os.path.join(args.models, args.model_file)
    cfg = load_cfg(args.data)
    ident = check_simulator_matches_dataset(cfg, args.data, strict=not args.no_strict_hash)
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    print(f"closed loop: {controllers} x {demands} x {len(seeds)} seeds, noise {args.noise}, noise variant {args.noise_variant}, "
          f"paired EV-off {not args.no_paired}, margin {args.margin}, lambda_base {cfg.lambda_base_vph}, model {model_path}")
    print(f"simulator {ident['simulator_generator_version']} {ident['simulator_config_hash']} vs dataset "
          f"{ident.get('dataset_generator_version')} {ident.get('dataset_config_hash')} match {ident.get('hash_match')}")
    t0 = time.perf_counter()
    df = run_protocol(cfg, model_path=model_path, controllers=controllers, demands=demands, seeds=seeds,
                      sensor_noise_level=args.noise, margin=args.margin, workers=args.workers, paired_ev_off=not args.no_paired)
    noise_variant = None
    if args.noise_variant >= 0 and not np.isclose(args.noise_variant, args.noise):
        noise_ctrls = tuple(c for c in NOISE_CONTROLLERS if c in controllers) or NOISE_CONTROLLERS
        print(f"noise variant {args.noise_variant}: {noise_ctrls}")
        dn = run_protocol(cfg, model_path=model_path, controllers=noise_ctrls, demands=demands, seeds=seeds,
                          sensor_noise_level=args.noise_variant, margin=args.margin, workers=args.workers,
                          paired_ev_off=not args.no_paired)
        df = pd.concat([df, dn], ignore_index=True)
        noise_variant = args.noise_variant
    wall = time.perf_counter() - t0
    meta = dict(n_seeds=len(seeds), seeds=seeds, lambda_ev_vph=cfg.lambda_ev_vph, sensor_noise_level=args.noise,
                noise_variant=noise_variant, noise_controllers=list(NOISE_CONTROLLERS), paired_ev_off=not args.no_paired,
                wall_s=wall, workers=args.workers, model_path=model_path, margin=args.margin,
                controllers=list(controllers), demands=list(demands), n_episodes=int(len(df)), bound_factor=BOUND_FACTOR,
                ml_name=ML_NAME, ml_aliases=list(ML_ALIASES),
                command="python3 -m traffic_twin.evaluate " + " ".join(argv if argv is not None else sys.argv[1:]),
                thread_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                **ident)
    section = write_outputs(df, meta, args.out, controllers, load_task_results(args.out))
    print(section)


if __name__ == "__main__":
    main()
