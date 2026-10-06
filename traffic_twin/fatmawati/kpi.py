"""KPIs per method from the evaluation grid, the Pareto front and the Nash equilibrium (methodology Section 8).

Inputs, from evaluate.py (data/fatmawati_eval):
  passages.csv         one row per ambulance passage: method, level, seed, leg (movement), slice, delay_s, ...
  episodes_eval.csv    one row per run: stopped_delay_s_<leg>, served_<leg>, longest_red_s_<leg>, end_s
  evaluation_qc.csv    one row per run: lane_collisions, junction_overlaps, teleports
  passages_hindsight.csv (optional) hindsight_delay_s per passage, for the share of the possible gain

Ambulance KPI   A_m = mean over movements k of D_k,m (equal weight), D_k,m = mean delay over slices, levels, seeds
General KPI     R_a,m = stopped delay on approach a / the fixed plan's on the same level and seed (geometric
                means over cells); R_m = geometric mean over the four approaches; worst_m = max_a R_a,m
Gates           a method fails if it significantly raises lane collisions per vehicle-km above the fixed plan's on the
                same level and seed (lower 95% bound of the paired ratio > 1; the sublane model produces some
                collisions at any density, so the gate judges what a method adds); upper 95% bound of
                R_m / R_max-pressure <= 1.10; upper bound of every R_a,m <= 1.20.  With few seeds the bounds can fail
                every method; the result then falls back to point estimates for the cost and approach gates, the
                safety gate staying a significance test.  The bootstrap uses that fallback in each resample, with
                each method's safety verdict from the full sample.
Equilibrium     Nash bargaining over the methods that pass: players are the ambulance, general traffic and each
                approach; walk-away points are today's ambulance delay, 1.10 x max-pressure's R and 1.20; the winner
                maximises the product of the gains.  Bootstrap over seeds gives each method's share of wins.

  python3 -m traffic_twin.fatmawati.kpi summary --eval data/fatmawati_eval --out reports/a3/kpi
  python3 -m traffic_twin.fatmawati.kpi hand-check --out reports/a3/kpi
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

LEGS = ("N", "E", "S", "W")
GEN_GATE, APPROACH_GATE = 1.10, 1.20
TODAY, MP, TEACHER, RANDOM = "fixed", "max_pressure", "priority_mp", "random"
RELEASES_PER_H = 6.0
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 23: 2.069, 30: 2.042, 47: 2.012, 60: 2.000}


def _t(df: int) -> float:
    if df <= 0:
        return float("nan")
    keys = sorted(T975)
    return T975[min(keys, key=lambda k: (abs(k - df), k))] if df not in T975 else T975[df]


def _lower_log(x) -> float:
    """Lower 95% bound of a ratio from paired log ratios."""
    x = np.log(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(np.exp(x.mean())) if len(x) else float("nan")
    return float(np.exp(x.mean() - _t(len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))))


def _upper_log(x: np.ndarray) -> float:
    """Upper 95% bound of a ratio from paired log ratios."""
    x = np.log(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(np.exp(x.mean())) if len(x) else float("nan")
    return float(np.exp(x.mean() + _t(len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))))


# ---------------------------------------------------------------------------------------------------- KPIs
def _flag(s: pd.Series) -> pd.Series:
    """True/False from booleans, 0/1, 0.0/1.0 or their strings (stacked CSVs mix them); NaN where empty."""
    m = {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    return s.astype(str).str.strip().str.lower().map(m)


def load(eval_dir: str) -> dict:
    """One evaluation folder, or several joined by commas (the rule-based grid, the closed loop, the hindsight
    replays): their tables are stacked and the first folder's configuration is kept."""
    dirs = [x for x in eval_dir.split(",") if x]
    out = {}
    for name in ("passages", "episodes_eval", "evaluation_qc", "passages_hindsight"):
        parts = [pd.read_csv(os.path.join(d, f"{name}.csv")) for d in dirs if os.path.exists(os.path.join(d, f"{name}.csv"))]
        out[name] = pd.concat(parts, ignore_index=True) if parts else None
    if out["passages"] is None:
        raise FileNotFoundError(f"no passages.csv in {eval_dir}")
    cfg = os.path.join(dirs[0], "evaluation_config.json")
    out["config"] = json.load(open(cfg)) if os.path.exists(cfg) else {}
    return out


EXTRA_S = 1800.0      # evaluate.py keeps a run going this long past its schedule while released ambulances travel


def censor(d: dict) -> dict:
    """delay_s for every passage: an ambulance that never reached the hospital exit before the run stopped counts at
    the lower bound of its delay (run end + EXTRA_S - release - empty-road time), flagged in `unfinished`; dropping
    such passages would flatter the methods that leave ambulances stuck."""
    p = d["passages"].copy()
    p["unfinished"] = p["delay_s"].isna()
    e = d.get("episodes_eval")
    if p["unfinished"].any() and e is not None and "end_s" in e.columns:
        ends = e[_flag(e["with_ev"]) == True].groupby(["level", "seed", "method"])["end_s"].max()
        stop = p.set_index(["level", "seed", "method"]).index.map(lambda k: ends.get(k, np.nan) + EXTRA_S)
        lb = np.asarray(stop, dtype=float) - p["t_release"].astype(float) - p["t_empty_s"].astype(float)
        p.loc[p["unfinished"], "delay_s"] = lb[p["unfinished"].values]
    return {**d, "passages": p}


def restrict(d: dict, levels: list | None) -> dict:
    """The tables limited to some demand levels (the stationary regime as the main result, the rest as stress tests)."""
    if not levels:
        return d
    return {k: (v[v["level"].isin(levels)].copy() if isinstance(v, pd.DataFrame) and "level" in v.columns else v)
            for k, v in d.items()}


def kpis(d: dict, seeds: list | None = None) -> pd.DataFrame:
    """One row per method; seeds restricts (and may repeat) the seeds used, for the bootstrap."""
    p = d["passages"]
    if seeds is not None:
        p = pd.concat([p[p["seed"] == s] for s in seeds], ignore_index=True)
    rows = {}
    cell = p.groupby(["method", "leg", "level", "seed"])["delay_s"].mean().reset_index()
    mov = cell.groupby(["method", "leg"])["delay_s"].mean().unstack()                  # D_k,m
    for m in mov.index:
        r = rows.setdefault(m, {})
        r["A"] = float(mov.loc[m].mean())
        r["worst_movement"] = str(mov.loc[m].idxmax())
        r["A_worst"] = float(mov.loc[m].max())
        for k, v in mov.loc[m].items():
            r[f"D_{k}"] = float(v)
    pm = p.groupby("method")
    for m, g in pm:
        r = rows.setdefault(m, {})
        r["passages"] = int(len(g))
        r["passages_unfinished"] = int(g["unfinished"].sum()) if "unfinished" in g else int(g["delay_s"].isna().sum())
        r["unfinished_share"] = r["passages_unfinished"] / max(1, r["passages"])
        r["delay_median_s"] = float(g["delay_s"].median())
        r["travel_time_mean_s"] = float(g["travel_time_s"].mean()) if "travel_time_s" in g else float("nan")
        # methods run without reference runs (--no-reference) leave recovery and extra wait empty: not run, not zero
        cen = _flag(g["recovery_censored"]) if "recovery_censored" in g else pd.Series(np.nan, index=g.index)
        ovl = _flag(g["recovery_overlaps_next_release"]) if "recovery_overlaps_next_release" in g else pd.Series(False, index=g.index)
        rec = g[(cen == False) & (ovl != True)]
        r["recovery_median_s"] = float(pd.to_numeric(rec["recovery_s"], errors="coerce").median()) if len(rec) else float("nan")
        r["recovery_censored_share"] = float((cen[cen.notna()] == True).mean()) if cen.notna().any() else float("nan")
        for a in LEGS:
            c = f"extra_wait_s_{a}"
            if c in g:                                   # noisy per passage (the runs diverge): mean and median
                r[f"extra_wait_{a}"] = float(g[c].mean())
                r[f"extra_wait_med_{a}"] = float(g[c].median())
    e = d.get("episodes_eval")
    if e is not None:
        e = e[_flag(e["with_ev"]) == True]
        if seeds is not None:
            e = pd.concat([e[e["seed"] == s] for s in seeds], ignore_index=True)
        base = e[e["method"] == TODAY].set_index(["level", "seed"]).sort_index()
        for m, g in e.groupby("method"):
            r = rows.setdefault(m, {})
            logs = {a: [] for a in LEGS}
            for _, x in g.iterrows():
                key = (x["level"], x["seed"])
                if key not in base.index:
                    continue
                b = base.loc[key]
                b = b.iloc[0] if isinstance(b, pd.DataFrame) else b
                for a in LEGS:
                    num, den = x.get(f"stopped_delay_s_{a}"), b.get(f"stopped_delay_s_{a}")
                    if pd.notna(num) and pd.notna(den) and den > 0 and num > 0:
                        logs[a].append(math.log(num / den))
            for a in LEGS:
                r[f"R_{a}"] = float(math.exp(np.mean(logs[a]))) if logs[a] else float("nan")
                r[f"R_{a}_upper"] = _upper_log(np.exp(logs[a])) if len(logs[a]) >= 2 else r[f"R_{a}"]
            vals = [r[f"R_{a}"] for a in LEGS if not math.isnan(r[f"R_{a}"])]
            r["R"] = float(math.exp(np.mean(np.log(vals)))) if vals else float("nan")
            r["R_worst"] = max(vals) if vals else float("nan")
            r["R_worst_approach"] = max(LEGS, key=lambda a: -1 if math.isnan(r[f"R_{a}"]) else r[f"R_{a}"])
            hours = ((g["end_s"] - g.get("warm_s", 900.0)) / 3600.0) if "end_s" in g else None
            if hours is not None and all(f"served_{a}" in g for a in LEGS):
                veh_s = sum(g[f"stopped_delay_s_{a}"].fillna(0) * g[f"served_{a}"].fillna(0) for a in LEGS)
                r["general_veh_s_per_h"] = float((veh_s / hours).mean())
            for a in LEGS:
                c = f"longest_red_s_{a}"
                if c in g:
                    r[f"longest_red_{a}"] = float(g[c].max())
        # paired ratio of each method's general index to max-pressure's, cell by cell
        cells = {}
        for _, x in e.iterrows():
            key = (x["level"], x["seed"])
            if key in base.index:
                b = base.loc[key]
                b = b.iloc[0] if isinstance(b, pd.DataFrame) else b
                rr = [x[f"stopped_delay_s_{a}"] / b[f"stopped_delay_s_{a}"] for a in LEGS
                      if pd.notna(x.get(f"stopped_delay_s_{a}")) and b.get(f"stopped_delay_s_{a}", 0) > 0 and x[f"stopped_delay_s_{a}"] > 0]
                if rr:
                    cells[(x["method"],) + key] = float(np.exp(np.mean(np.log(rr))))
        for m in rows:
            ratios = [cells[(m,) + k[1:]] / cells[(MP,) + k[1:]] for k in cells if k[0] == m and (MP,) + k[1:] in cells]
            rows[m]["R_vs_mp"] = float(np.exp(np.mean(np.log(ratios)))) if ratios else float("nan")
            rows[m]["R_vs_mp_upper"] = _upper_log(ratios) if len(ratios) >= 2 else rows[m]["R_vs_mp"]
    qc = d.get("evaluation_qc")
    if qc is not None:
        if seeds is not None:
            qc = pd.concat([qc[qc["seed"] == s] for s in seeds], ignore_index=True)
        for m, g in qc[_flag(qc["with_ev"]) == True].groupby("method"):     # the runs every method has: with ambulances
            rows.setdefault(m, {})["lane_collisions"] = int(g["lane_collisions"].sum())
            if "lane_collisions_study" in g:
                rows[m]["lane_collisions_study"] = int(g["lane_collisions_study"].sum())
                if "veh_km_study" in g:
                    rows[m]["collisions_per_1000_veh_km_study"] = float(1000 * g["lane_collisions_study"].sum() / g["veh_km_study"].sum())
            rows[m]["teleports"] = int(g["teleports"].sum()) if "teleports" in g else 0
        # collisions a method adds: rate per vehicle-km (or per run) against the fixed plan, cell by cell
        q = qc[_flag(qc["with_ev"]) == True].copy()
        # the junction's own area (within 300 m) is where signal decisions act; fall back to network-wide counts
        coll = q["lane_collisions_study"] if "lane_collisions_study" in q else q["lane_collisions"]
        expo = q["veh_km_study"] if "veh_km_study" in q else (q["veh_km"] if "veh_km" in q else pd.Series(1.0, index=q.index))
        q["rate"] = (coll + 1.0) / expo.clip(lower=1e-9)
        base = q[q["method"] == TODAY].groupby(["level", "seed"])["rate"].mean()
        for m, g in q.groupby("method"):
            ratios = [r["rate"] / base[(r["level"], r["seed"])] for _, r in g.iterrows() if (r["level"], r["seed"]) in base.index]
            rows[m]["collision_ratio"] = float(np.exp(np.mean(np.log(ratios)))) if ratios else float("nan")
            rows[m]["collision_ratio_upper"] = _upper_log(ratios) if len(ratios) >= 2 else rows[m]["collision_ratio"]
            rows[m]["collision_ratio_lower"] = _lower_log(ratios) if len(ratios) >= 2 else rows[m]["collision_ratio"]
    h = d.get("passages_hindsight")
    if h is not None and {"hindsight_delay_s", "base_delay_s"} <= set(h.columns):
        # the replays fork from their own base run, which need not match the grid's run draw for draw, so the ceiling is
        # the grid's delay under the replays' base method on the same passages minus the gain measured inside the replays
        keys = ["leg", "level", "seed", "slice"]
        hh = h.dropna(subset=["hindsight_delay_s", "base_delay_s"]).copy()
        hh["gain"] = hh["base_delay_s"].astype(float) - hh["hindsight_delay_s"].astype(float)
        base_m = str(hh["base"].iloc[0]) if "base" in hh.columns and len(hh) else MP
        sub = p.merge(hh[keys].drop_duplicates(), on=keys)
        dm = sub.groupby(["method", "leg"])["delay_s"].mean().unstack()
        gain = hh.groupby("leg")["gain"].mean()
        if base_m in dm.index and TODAY in dm.index:
            ceil = {k: float(dm.loc[base_m, k] - gain.get(k, np.nan)) for k in dm.columns}
            a_today, a_ceil = float(dm.loc[TODAY].mean()), float(np.nanmean(list(ceil.values())))
            for m in dm.index:
                for k in dm.columns:
                    gap = float(dm.loc[TODAY, k]) - ceil[k]
                    rows[m][f"G_{k}"] = float((dm.loc[TODAY, k] - dm.loc[m, k]) / gap) if gap > 3.0 else float("nan")
                    rows[m][f"ceiling_D_{k}"] = ceil[k]
                rows[m]["A_on_replayed"] = float(dm.loc[m].mean())
                rows[m]["A_ceiling"] = a_ceil
                rows[m]["G"] = float((a_today - dm.loc[m].mean()) / (a_today - a_ceil)) if a_today - a_ceil > 3.0 else float("nan")
                # the share of the gain that perfect switch timing adds to the base method which method m captures:
                # defined even where the base method is slower than today on the replayed passages
                g_all = float(np.nanmean([gain.get(k, np.nan) for k in dm.columns]))
                rows[m]["G_vs_base"] = float((dm.loc[base_m].mean() - dm.loc[m].mean()) / g_all) if g_all > 3.0 else float("nan")
                for k in dm.columns:
                    gk = float(gain.get(k, np.nan))
                    rows[m][f"G_vs_base_{k}"] = float((dm.loc[base_m, k] - dm.loc[m, k]) / gk) if gk > 3.0 else float("nan")
                rows[m]["hindsight_gain_s"] = g_all
    df = pd.DataFrame(rows).T
    df.index.name = "method"
    return df


def gates(k: pd.DataFrame, point: bool = False, safe: pd.Series | None = None) -> pd.DataFrame:
    """point=False: the cost and approach gates on the upper 95% bounds (main result); point=True: on point estimates
    (the fallback when nothing passes at 95%, and the bootstrap). The safety gate is a significance test in both: a
    method fails only if its collisions rise significantly above the fixed plan's (rare events make a point ratio
    noise). `safe` overrides it with a verdict already reached on the full sample (bootstrap resamples)."""
    def col(name, default):
        return pd.to_numeric(k[name], errors="coerce") if name in k.columns else pd.Series(default, index=k.index, dtype=float)
    g = pd.DataFrame(index=k.index)
    g["safe"] = (safe.reindex(k.index).fillna(True).astype(bool) if safe is not None
                 else col("collision_ratio_lower", 1.0).fillna(1.0) <= 1.0)
    if point:
        g["cost_ok"] = col("R_vs_mp", np.nan) <= GEN_GATE
        vals = [col(f"R_{a}", np.nan) for a in LEGS]
    else:
        g["cost_ok"] = col("R_vs_mp_upper", np.nan) <= GEN_GATE
        vals = [col(f"R_{a}_upper", np.nan) for a in LEGS]
    g["approaches_ok"] = np.all([v <= APPROACH_GATE for v in vals], axis=0)
    g["passes"] = g["safe"] & g["cost_ok"] & g["approaches_ok"]
    return g


def pareto(k: pd.DataFrame, methods: list) -> list:
    cols = ["A", "R"] + [f"R_{a}" for a in LEGS]
    cols = [c for c in cols if c in k.columns]
    v = k.loc[methods, cols].astype(float)
    front = []
    for m in methods:
        dom = ((v.drop(index=m) <= v.loc[m]).all(axis=1) & (v.drop(index=m) < v.loc[m]).any(axis=1)).any()
        if not dom:
            front.append(m)
    return front


def nash(k: pd.DataFrame, candidates: list) -> pd.DataFrame:
    a_today = float(k.loc[TODAY, "A"]) if TODAY in k.index else float("nan")
    r_mp = float(k.loc[MP, "R"]) if MP in k.index and "R" in k.columns else float("nan")
    rows = {}
    for m in candidates:
        gains = dict(ambulance=a_today - float(k.loc[m, "A"]))
        if "R" in k.columns and not math.isnan(r_mp):
            gains["general"] = GEN_GATE * r_mp - float(k.loc[m, "R"])
        for a in LEGS:
            if f"R_{a}" in k.columns and not math.isnan(float(k.loc[m, f"R_{a}"])):
                gains[f"approach_{a}"] = APPROACH_GATE - float(k.loc[m, f"R_{a}"])
        ok = all(v > 0 for v in gains.values())
        rows[m] = dict(**{f"gain_{g}": v for g, v in gains.items()}, feasible=ok,
                       nash_product=float(np.prod(list(gains.values()))) if ok else float("nan"),
                       log_nash=float(sum(math.log(v) for v in gains.values())) if ok else float("nan"))
    return pd.DataFrame(rows).T


def implied_lambda(k: pd.DataFrame, candidates: list, winner: str) -> dict:
    """Road-user vehicle-seconds per hour the choice trades for each ambulance second: the slopes to the winner's
    neighbours on the two-measure front (ambulance delay per hour, general delay per hour) of the candidates."""
    if "general_veh_s_per_h" not in k.columns or winner not in candidates:
        return {}
    pts = k.loc[candidates, ["A", "general_veh_s_per_h"]].astype(float).dropna()
    pts["amb_s_per_h"] = pts["A"] * RELEASES_PER_H
    front2 = [m for m in pts.index if not ((pts["amb_s_per_h"] <= pts.loc[m, "amb_s_per_h"]) &
                                           (pts["general_veh_s_per_h"] <= pts.loc[m, "general_veh_s_per_h"]) &
                                           ((pts["amb_s_per_h"] < pts.loc[m, "amb_s_per_h"]) |
                                            (pts["general_veh_s_per_h"] < pts.loc[m, "general_veh_s_per_h"]))).any()]
    if winner not in front2:
        return dict(note="the equilibrium is not on the two-measure front, so no single price describes it")
    f = pts.loc[front2].sort_values("amb_s_per_h")
    i = list(f.index).index(winner)
    out = {}
    for side, j in (("towards_more_ambulance_priority", i - 1), ("towards_less_ambulance_priority", i + 1)):
        if 0 <= j < len(f):
            n = f.index[j]
            d_amb = abs(f.loc[n, "amb_s_per_h"] - f.loc[winner, "amb_s_per_h"])
            d_gen = abs(f.loc[n, "general_veh_s_per_h"] - f.loc[winner, "general_veh_s_per_h"])
            out[side] = dict(neighbour=n, lambda_veh_s_per_amb_s=float(d_gen / d_amb) if d_amb else float("nan"))
    if not out:
        out["note"] = "the equilibrium dominates every other candidate on ambulance and general delay, so no trade-off price applies"
    return out


def summary(eval_dir: str, out: str, boot: int = 1000, seed: int = 20261003, levels: list | None = None) -> dict:
    os.makedirs(out, exist_ok=True)
    d_all = censor(load(eval_dir))
    d = restrict(d_all, levels)
    k = kpis(d)
    g = gates(k)
    k = k.join(g)
    passing = [m for m in k.index if bool(g.loc[m, "passes"])]
    front = pareto(k, passing) if passing else []
    nb = nash(k, front) if front else pd.DataFrame()
    winner = str(nb["nash_product"].astype(float).idxmax()) if len(nb) and nb["feasible"].any() else None
    if TEACHER in k.index and RANDOM in k.index:
        span = float(k.loc[RANDOM, "A"] - k.loc[TEACHER, "A"])
        k["normalised_score"] = (k.loc[RANDOM, "A"] - k["A"].astype(float)) / span if span else np.nan
    if TEACHER in k.index and TODAY in k.index:           # the share of the teacher's gain over today's plan a method keeps
        span = float(k.loc[TODAY, "A"] - k.loc[TEACHER, "A"])
        k["share_of_teacher_gain"] = (k.loc[TODAY, "A"] - k["A"].astype(float)) / span if span else np.nan
    # bootstrap over seeds: the share of resamples each method wins
    seeds = sorted(d["passages"]["seed"].unique())
    rng = np.random.default_rng(seed)
    wins, draws = {}, []
    spread = ["A", "A_worst", "R", "R_worst"] + [f"R_{a}" for a in LEGS] + [f"extra_wait_{a}" for a in LEGS]
    for _ in range(boot if len(seeds) >= 2 else 0):
        s = list(rng.choice(seeds, len(seeds), replace=True))
        kb = kpis(d, s)
        draws.append(kb[[c for c in spread if c in kb.columns]].astype(float))
        kb = kb.join(gates(kb, point=True, safe=g["safe"]))
        pb = [m for m in kb.index if bool(kb.loc[m, "passes"])]
        fb = pareto(kb, pb) if pb else []
        nbb = nash(kb, fb) if fb else pd.DataFrame()
        w = str(nbb["nash_product"].astype(float).idxmax()) if len(nbb) and nbb["feasible"].any() else "none"
        wins[w] = wins.get(w, 0) + 1                    # shares over all resamples; "none" when nothing passes
    total = sum(wins.values())
    if draws:                                            # 95% intervals over seed resamples, for the tables
        allb = pd.concat(draws, keys=range(len(draws)), names=["draw", "method"])
        for c in allb.columns:
            q = allb[c].groupby(level="method").quantile([0.025, 0.975]).unstack()
            k[f"{c}_lo"], k[f"{c}_hi"] = q[0.025].reindex(k.index), q[0.975].reindex(k.index)
    breakdown(d_all, out)
    gp = gates(k.drop(columns=[c for c in ("safe", "cost_ok", "approaches_ok", "passes") if c in k.columns]), point=True)
    passing_pt = [m for m in k.index if bool(gp.loc[m, "passes"])]
    front_pt = pareto(k, passing_pt) if passing_pt else []
    nb_pt = nash(k, front_pt) if front_pt else pd.DataFrame()
    winner_pt = str(nb_pt["nash_product"].astype(float).idxmax()) if len(nb_pt) and nb_pt["feasible"].any() else None
    res = dict(eval_dir=eval_dir, levels=levels or sorted(d_all["passages"]["level"].unique()), methods=list(k.index),
               passing=passing, pareto_front=front, winner=winner,
               point_estimate=dict(passing=passing_pt, pareto_front=front_pt, winner=winner_pt,
                                   nash={m: float(v) for m, v in nb_pt["nash_product"].items()} if len(nb_pt) else {}),
               bootstrap=dict(resamples=boot if len(seeds) >= 2 else 0, seeds=[int(s) for s in seeds],
                              win_share={m: wins[m] / total for m in sorted(wins)} if total else {}),
               implied_lambda=implied_lambda(k, passing or passing_pt, winner or winner_pt) if (winner or winner_pt) else {})
    k.to_csv(os.path.join(out, "kpi_methods.csv"))
    if len(nb):
        nb.to_csv(os.path.join(out, "nash.csv"))
    with open(os.path.join(out, "equilibrium.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    figures(k, front or front_pt, winner or winner_pt, out)
    return res


def breakdown(d: dict, out: str) -> None:
    """kpi_by_level.csv: the main measures within each demand level (where a method breaks down);
    kpi_by_signal.csv: mean ambulance delay by the signal its approach showed at detection, per movement."""
    rows = []
    for lv in sorted(d["passages"]["level"].unique()):
        sub = {k: (v[v["level"] == lv] if isinstance(v, pd.DataFrame) and "level" in v.columns else v) for k, v in d.items()}
        kl = kpis(sub)
        cols = [c for c in ["A", "A_worst", "delay_median_s", "unfinished_share", "R", "R_worst", "R_worst_approach"]
                + [f"D_{m}" for m in ("N", "E", "W")] + [f"R_{a}" for a in LEGS]
                + [f"longest_red_{a}" for a in LEGS] + ["collision_ratio"] if c in kl.columns]
        rows.append(kl[cols].assign(level=lv))
    pd.concat(rows).reset_index().rename(columns={"index": "method"}).to_csv(os.path.join(out, "kpi_by_level.csv"), index=False)
    p = d["passages"]
    if "detect_signal" in p.columns:
        t = p.groupby(["method", "leg", "detect_signal"])["delay_s"].agg(["mean", "count"]).reset_index()
        t.to_csv(os.path.join(out, "kpi_by_signal.csv"), index=False)


NAMES = {"fixed": "Fixed plan (today)", "longest_queue": "Longest queue", "max_pressure": "Max-pressure",
         "priority_mp": "Priority teacher", "preempt_naive": "Naive preemption", "random": "Random", "still": "Still"}


def pretty(m: str) -> str:
    if m.startswith("learned:"):
        return "Learned: " + os.path.basename(m).replace("taskA_", "").replace(".joblib", "").replace("_", " ")
    return NAMES.get(m, m.replace("_", " "))


def _place_labels(ax, pts: list, names: list, fontsize: float = 7.5) -> list:
    """Greedy label placement: for each point, the first of eight offsets whose box overlaps no placed label or point."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    taken, texts = [], []
    disp = [ax.transData.transform(p) for p in pts]
    marks = [(x - 5, y - 5, x + 5, y + 5) for x, y in disp]
    offsets = [(8, 2), (8, -10), (-8, 2), (-8, -10), (8, 12), (-8, 12), (8, -20), (-8, -20), (8, 22), (-8, 22)]
    order = sorted(range(len(pts)), key=lambda i: disp[i][0])
    for i in order:
        best = None
        for dx, dy in offsets:
            t = ax.annotate(names[i], pts[i], xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                            ha="left" if dx > 0 else "right", va="bottom", color="#0b0b0b")
            bb = t.get_window_extent(renderer=rend).expanded(1.04, 1.1)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            hit = sum(1 for b in taken + [m for j, m in enumerate(marks) if j != i]
                      if not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]))
            if best is None or hit < best[0]:
                if best is not None:
                    best[1].remove()
                best = (hit, t, box)
            else:
                t.remove()
            if hit == 0:
                break
        taken.append(best[2])
        texts.append(best[1])
    return texts


def figures(k: pd.DataFrame, front: list, winner: str | None, out: str) -> None:
    """tradeoff.png: flattened ambulance delay against the general-delay index, with 95% bootstrap whiskers, the Pareto
    front, the equilibrium, today's ambulance delay and the 10% cost gate. Controllers far off the chart (the random and
    still baselines, any method above 5x the median ambulance delay) are listed under it instead of plotted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "R" not in k.columns:
        return
    kk = k[pd.to_numeric(k["A"], errors="coerce").notna() & pd.to_numeric(k["R"], errors="coerce").notna()].copy()
    med = float(kk["A"].astype(float).median())
    off = [m for m in kk.index if m in (RANDOM, "still") or float(kk.loc[m, "A"]) > 5 * med]
    on = kk.drop(index=off)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for m, r in on.iterrows():
        x, y = float(r["R"]), float(r["A"])
        if {"R_lo", "R_hi", "A_lo", "A_hi"} <= set(r.index) and pd.notna(r["R_lo"]):
            ax.plot([float(r["R_lo"]), float(r["R_hi"])], [y, y], color="#d6d4cf", lw=0.8, zorder=1)
            ax.plot([x, x], [float(r["A_lo"]), float(r["A_hi"])], color="#d6d4cf", lw=0.8, zorder=1)
        inf = m in front
        ax.plot(x, y, "o", ms=9 if m == winner else 7, color="#2a78d6" if inf else "#8a8984", mec="#0b0b0b" if m == winner else "white",
                mew=1.6 if m == winner else 1.0, zorder=3)
    if len(front) >= 2:
        pts = on.loc[[m for m in front if m in on.index], ["R", "A"]].astype(float).sort_values("R")
        ax.plot(pts["R"], pts["A"], "-", color="#2a78d6", lw=1.2, zorder=2)
    if MP in kk.index:
        gx = GEN_GATE * float(kk.loc[MP, "R"])
        ax.axvline(gx, color="#d64545", lw=1, ls="--", zorder=0)
        ax.annotate("cost gate: 10% above max-pressure", (gx, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -10),
                    textcoords="offset points", fontsize=7, color="#d64545", va="top")
    if TODAY in kk.index:
        ay = float(kk.loc[TODAY, "A"])
        ax.axhline(ay, color="#52514e", lw=0.8, ls=":", zorder=0)
        ax.annotate("today's ambulance delay", (0.0, ay), xycoords=("axes fraction", "data"), xytext=(4, 3),
                    textcoords="offset points", fontsize=7, color="#52514e")
    ax.set_xlabel("general delay index (1 = today's fixed plan)")
    ax.set_ylabel("flattened ambulance delay (s)")
    ax.grid(color="#eceae6", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.margins(x=0.12, y=0.12)
    names = [pretty(m) + (" (equilibrium)" if m == winner else "") for m in on.index]
    _place_labels(ax, [(float(on.loc[m, "R"]), float(on.loc[m, "A"])) for m in on.index], names)
    note = ("General delay index: geometric mean over the four approaches of stopped delay relative to the fixed plan.\n"
            "Blue: the Pareto front of the methods that pass the gates. Whiskers: 95% intervals over seed resamples.")
    if off:
        note += "\nOff the chart: " + "; ".join(f"{pretty(m)} {float(kk.loc[m, 'A']):,.0f} s at {float(kk.loc[m, 'R']):.2f}" for m in off) + "."
    fig.text(0.0, -0.02, note, fontsize=7, color="#52514e", va="top")
    fig.savefig(os.path.join(out, "tradeoff.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- hand check
def hand_check(out: str, n: int = 48, tol: float = 5.0) -> dict:
    """An ambulance alone under the city's fixed plan: mean standing time before its stop line vs r^2 / (2C).
    Arrivals are spread over n equal phases of the cycle (the evaluation grid's 8 slices sample it too coarsely
    for this check); one ambulance per movement every 600 s, so passages never interact."""
    import tempfile
    import libsumo as ts
    from .build_net import CROSSINGS, DISHUB_GREEN_S, STAGES, YELLOW_S, ALLRED_S
    from .evaluate import release_points, schedule, CYCLE_S
    from .sumo_world import FatmawatiWorld, NET
    from .demand import hospital_edge
    import sumolib
    os.makedirs(out, exist_ok=True)
    net = sumolib.net.readNet(NET, withInternal=False)
    hosp, pos_h = hospital_edge(net)
    rel = release_points(net)
    with tempfile.NamedTemporaryFile("w", suffix=".rou.xml", delete=False) as f:
        f.write("<routes/>\n")
    w = FatmawatiWorld(f.name, start_hour=7.0, seed=1, controller="fixed", sensors=False)
    sched = []
    for j in range(n):
        base = 900.0 + j * 600.0
        t = base + (j * CYCLE_S / n - base) % CYCLE_S
        sched += [(t, leg, j) for leg in ("N", "E", "W")]
    meta, pending, crossing = {}, list(sched), tuple(f":{c}_" for c in CROSSINGS)
    end = sched[-1][0] + 900.0
    while w.t < end:
        while pending and w.t >= pending[0][0] - 1e-9:
            t_rel, leg, k = pending.pop(0)
            edge, pos, _ = rel[leg]
            vid = f"ev_{len(meta):02d}"
            ts.route.add(f"r_{vid}", ts.simulation.findRoute(edge, hosp.getID(), vType="ambulance").edges)
            ts.vehicle.add(vid, f"r_{vid}", typeID="ambulance", depart="now", departPos=str(pos), departLane="best",
                           departSpeed="max", arrivalPos=str(pos_h))
            meta[vid] = dict(leg=leg, slice=k, t_release=t_rel, standing_s=0.0, done=False, t_stop=None, t_go=None)
        w.step()
        live = set(ts.vehicle.getIDList())
        for vid, m in meta.items():
            if m["done"] or vid not in live:
                m["done"] = m["done"] or (vid not in live and w.t > m["t_release"] + 5)
                continue
            if ts.vehicle.getRoadID(vid).startswith(crossing):
                m["done"] = True                                     # reached the junction: stop counting
            elif ts.vehicle.getSpeed(vid) < 0.1:
                m["standing_s"] += 0.5
                if m["t_stop"] is None:
                    m["t_stop"] = w.t
            elif m["t_stop"] is not None and m["t_go"] is None:
                m["t_go"] = w.t
    w.close() if hasattr(w, "close") else ts.close()
    os.unlink(f.name)
    df = pd.DataFrame(meta.values())
    stage_of = {"N": "S3", "W": "S1"}
    # exact check: a stopped ambulance moves off just after its stage's green starts in the fixed plan
    start, t0 = {}, 0.0
    for st in STAGES:
        start[st] = t0
        t0 += DISHUB_GREEN_S[st] + YELLOW_S + ALLRED_S
    def lag(r):
        if pd.isna(r["t_stop"]) or pd.isna(r["t_go"]) or r["leg"] not in stage_of:
            return np.nan
        g0 = start[stage_of[r["leg"]]]
        k = math.ceil((r["t_stop"] - g0) / CYCLE_S)
        return r["t_go"] - (g0 + k * CYCLE_S)
    df["go_lag_s"] = df.apply(lag, axis=1)
    rows = {}
    for leg, g in df.groupby("leg"):
        if leg in stage_of:
            green = DISHUB_GREEN_S[stage_of[leg]]
            r = CYCLE_S - green - YELLOW_S
            expected = r * r / (2 * CYCLE_S)
        else:
            expected = 0.0                                           # the free-left slip has no signal
        meas = float(g["standing_s"].mean())
        lagv = g["go_lag_s"].dropna()
        brake = 62.5 / 3.6 / (2 * 3.0)                               # seconds lost braking from 1.25 x 50 km/h at 3 m/s2
        corrected = (r - brake) ** 2 / (2 * CYCLE_S) if leg in stage_of else 0.0
        rows[leg] = dict(measured_mean_wait_s=round(meas, 1), expected_s=round(expected, 1),
                         expected_after_braking_s=round(corrected, 1), n=int(len(g)), stops=int(g["t_stop"].notna().sum()),
                         go_lag_mean_s=round(float(lagv.mean()), 2) if len(lagv) else None,
                         go_lag_max_s=round(float(lagv.max()), 2) if len(lagv) else None,
                         passes=bool(abs(meas - expected) <= tol and (lagv.abs().max() <= 3.0 if len(lagv) else True)))
    df.to_csv(os.path.join(out, "hand_check_passages.csv"), index=False)
    res = dict(cycle_s=CYCLE_S, arrival_phases=n, tolerance_s=tol, by_movement=rows,
               passes=all(v["passes"] for v in rows.values()))
    with open(os.path.join(out, "hand_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary"); s.add_argument("--eval", required=True); s.add_argument("--out", required=True)
    s.add_argument("--boot", type=int, default=1000)
    s.add_argument("--levels", default="", help="comma-separated demand levels for the main result (default: all)")
    h = sub.add_parser("hand-check"); h.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = (summary(a.eval, a.out, a.boot, levels=[x for x in a.levels.split(",") if x]) if a.cmd == "summary"
           else hand_check(a.out))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
