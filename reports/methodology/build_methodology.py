"""Fill the methodology template with measured numbers and render it.

Run from the project root:
    python3 reports/methodology/build_methodology.py [--run data/dev10]

Reads  data/<run>/{config.json, episodes.csv, signal_ticks.csv, vehicles.csv, junction_1hz.csv,
       data_dictionary.csv} and reports/validation_summary.json,
writes reports/methodology/Methodology_Dataset_Generation.{md,docx,html,pdf}.
Rendering uses RStudio's bundled pandoc and headless Google Chrome (no LaTeX needed).
"""
import argparse, datetime, json, math, os, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PANDOC = "/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64/pandoc"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:,.{nd}f}" if nd else f"{x:,.0f}"


def human_size(nbytes: float) -> str:
    return f"{nbytes/1e6:.1f} MB" if nbytes >= 1e5 else f"{nbytes/1e3:.0f} kB"


def md_table(df: pd.DataFrame, floatfmt=1) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("n/a" if math.isnan(v) else f"{v:,.{floatfmt}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/dev10")
    ap.add_argument("--download-url", default="*(download link to be inserted at release)*",
                    help="public link to the dataset release, quoted in Section 12.4")
    args = ap.parse_args()
    run = ROOT / args.run
    cfg = json.load(open(run / "config.json"))
    val = json.load(open(ROOT / "reports/validation_summary.json"))
    ep = pd.read_csv(run / "episodes.csv")
    st = pd.read_csv(run / "signal_ticks.csv", low_memory=False)
    vh = pd.read_csv(run / "vehicles.csv", low_memory=False)
    dd = pd.read_csv(run / "data_dictionary.csv")
    n1 = sum(1 for _ in open(run / "junction_1hz.csv")) - 1

    d = val["discharge"]; cal = val["calibration"]; des = val["design_episode"]
    lam = cal["lambda_base_vph"]
    V = {}
    V["GENERATOR_VERSION"] = cfg.get("generator_version", "")
    V["CONFIG_HASH"] = str(ep["config_hash"].iloc[0])
    V["RUN_NAME"] = args.run
    V["RUN_LABEL"] = f"{len(ep)}-episode run"
    V["DOWNLOAD_URL"] = args.download_url
    V["TODAY"] = datetime.date.today().isoformat()
    V["SAT_FLOW"] = fmt(d["sat_flow_vphpl"], 0)
    V["H_SAT"] = fmt(d["h_sat_s"], 2)
    V["LOST_TIME"] = fmt(d["startup_lost_time_s"], 2)
    V["CAPACITY"] = fmt(cal["capacity_vph"], 0)
    V["CAPACITY_M"] = f"{cal['capacity_vph']:.0f}"
    V["CAPACITY_FORMULA_M"] = f"{cal['capacity_formula_vph']:.0f}" if cal.get("capacity_formula_vph") else V["CAPACITY_M"]
    V["SAT_FLOW_M"] = f"{d['sat_flow_vphpl']:.0f}"
    V["F_MAX"] = fmt(cal["flow_per_unit_lambda"], 2)
    V["BUSIEST_EDGE"] = cal["busiest_edge"]
    V["LAMBDA_TABLE"] = ", ".join(f"{k} {fmt(lam[k], 0)}" for k in ["low", "design", "high", "stress"] if k in lam)
    V["LAMBDA_DESIGN"] = fmt(lam["design"], 0)
    pe = {k: lam[k] * cfg.get("asym_ns_factor", 1.4) * (1 + cfg.get("demand_jitter", 0.1)) / 3600 * cfg.get("dt", 0.5) for k in ["low", "design", "high", "stress"] if k in lam}
    over = [k for k in ["low", "design", "high", "stress"] if k in pe and pe[k] > 0.1]
    if over:
        V["PE_SENTENCE"] = ("That limit is exceeded only on the north–south entries of `ns_heavy` episodes at " + " and ".join(over) +
                            " demand ($p_e$ up to " + f"{max(pe[k] for k in over):.3f}" + " after the 1.4 factor and the 1.1 jitter), where arrivals are slightly more regular than Poisson; the effect is confined to the hardest scenario" + ("s" if len(over) > 1 else "") + ".")
    else:
        V["PE_SENTENCE"] = f"Even the north–south entries of `ns_heavy` episodes at stress demand stay below it ($p_e$ at most {max(pe.values()):.3f} after the 1.4 factor and the 1.1 jitter)."
    cl_path = ROOT / "reports/ml/closed_loop.csv"
    V["PILOT_PROVENANCE"] = ""
    V["CL_BOUND_CHECK"] = "The closed-loop check of this bound is reported in Section 12.5 once the modelling stage has run on the release."
    if cl_path.exists():
        cl = pd.read_csv(cl_path, low_memory=False)
        if "sensor_noise_level" in cl.columns:
            cl = cl[cl["sensor_noise_level"].fillna(0) == 0]
        if "ev_rate" in cl.columns:
            cl = cl[cl["ev_rate"].astype(str) != "0"]
        cl_ver = str(cl["generator_version"].iloc[0]) if "generator_version" in cl.columns else ""
        cl_hash = str(cl["config_hash"].iloc[0]) if "config_hash" in cl.columns else ""
        same = bool(cl_ver) and cl_ver == V["GENERATOR_VERSION"] and cl_hash == V["CONFIG_HASH"]
        if cl_ver and not same:
            lv = cl.groupby("demand_level")["lambda_base_vph"].first().to_dict() if "lambda_base_vph" in cl.columns else {}
            kf = cal["flow_per_unit_lambda"]; cc = cal["capacity_vph"]
            ratios = ", ".join(f"{d} {lv[d] * kf / cc:.2f}" for d in ["design", "high"] if d in lv)
            V["PILOT_PROVENANCE"] = (f"The pilot results of Section 12.5 were produced with generator version {cl_ver} (configuration hash {cl_hash}), "
                                     f"an earlier build of the dataset than the one this document describes; its demand levels were set before the capacity calibration of Section 9 "
                                     f"(realised volume-to-capacity ratios against the measured capacity: {ratios}), so its rows labelled design and high are not the levels Sections 9 and 10 define "
                                     f"and are not directly comparable with Experiment 5. The modelling stage is re-run on the released dataset and replaces those tables.")
        else:
            V["PILOT_PROVENANCE"] = "The pilot results of Section 12.5 were computed on the dataset this document describes (same generator version and configuration hash)."
        ml_name = next((n for n in ["ml_sensor", "ml_truth", "ml_v1"] if n in set(cl["controller"])), None)
        if ml_name and "max_pressure" in set(cl["controller"]) and "mean_total_delay_s" in cl.columns:
            parts = []
            tcrit = {8: 2.365, 7: 2.447, 6: 2.571, 5: 2.776, 4: 3.182, 3: 4.303}
            for lvl in ["design", "high"]:
                a = cl[(cl.demand_level == lvl) & (cl.controller == ml_name)].set_index("seed")["mean_total_delay_s"]
                m = cl[(cl.demand_level == lvl) & (cl.controller == "max_pressure")].set_index("seed")["mean_total_delay_s"]
                idx = a.index.intersection(m.index)
                if len(idx) < 3:
                    continue
                d = (a.loc[idx] - m.loc[idx]); n = len(idx)
                hw = tcrit.get(n, 2.0) * d.std(ddof=1) / math.sqrt(n)
                margin = 0.10 * m.loc[idx].mean()
                met = (d.mean() + hw) <= margin
                parts.append(f"at {lvl} demand the learned controller's car delay differs from max-pressure by {d.mean():+.1f} ± {hw:.1f} s over {n} paired seeds against a margin of {margin:.1f} s, so the bound is {'met' if met else 'not met'}")
            bpath = ROOT / "reports/ml/closed_loop_bound.csv"
            if bpath.exists():
                bd = pd.read_csv(bpath)
                bd = bd[(bd["sensor_noise_level"].fillna(0) == 0) & (bd["demand_level"].isin(["design", "high"]))]
                if len(bd):
                    parts = [f"at {r.demand_level} demand the learned controller's mean car delay is {r.mean_ml:.1f} s against {r.mean_reference:.1f} s for max-pressure "
                             f"(paired difference {r.diff_mean_s:+.1f} s, 95% upper end {r.diff_ci95_high_s:+.1f} s, against a margin of {r.margin_s:.1f} s; ratio {r.mean_ratio:.2f}, "
                             f"95% interval {r.ci95_low:.2f} to {r.ci95_high:.2f}), so the bound is {'met' if r.bound_met_methodology else 'not met'}"
                             for r in bd.itertuples()]
            if parts:
                prov = "same dataset" if same else f"generator {cl_ver}, before the recalibration of Section 9"
                V["CL_BOUND_CHECK"] = (f"In the pilot closed loop of Section 12.5 ({prov}), " + "; ".join(parts) +
                                       ". Where the bound is not met, the levers for the modelling stage are the truth feature set, the hold threshold and a calmer teacher (Section 6), and the report states whichever outcome holds.")
    V["DESIGN_TOTAL_VPH"] = fmt(8 * lam["design"], 0)
    V["DESIGN_DELAY"] = fmt(des.get("mean_delay_s"), 1)
    V["DESIGN_P90"] = fmt(des.get("p90_delay_s"), 1)
    V["DESIGN_THROUGHPUT"] = fmt(des.get("throughput_vph"), 0)
    V["DESIGN_ALLRED"] = fmt(des.get("allred_extension_mean_s"), 2)
    inv = des.get("invariants")
    if isinstance(inv, dict):
        V["DESIGN_VIOL"] = str(sum(int(v) for v in inv.values() if isinstance(v, (int, float)) and not isinstance(v, bool)))
    else:
        V["DESIGN_VIOL"] = str(des.get("n_invariant_violations", "n/a"))
    V["DESIGN_RUNTIME"] = fmt(des.get("runtime_s"), 1)
    V["DESIGN_SWITCH_RATE"] = fmt(des.get("switches_per_min"), 1)
    V["DESIGN_HB_PER_ONSET"] = fmt(des.get("hard_brakes_per_red_onset"), 2)
    V["DESIGN_EMERG_SHARE"] = f"{des['emergency_brake_share']*100:.0f}%" if des.get("emergency_brake_share") is not None else "n/a"
    V["DESIGN_MAX_RED"] = fmt(des.get("max_approach_red_s"), 0)
    V["DESIGN_N_STARV"] = str(des.get("n_forced_starvation", "n/a"))
    V["DESIGN_N_MERGE"] = str(des.get("n_merge_holds", "n/a"))
    V["DESIGN_N_REVOKED"] = str(des.get("n_commit_revoked", "n/a"))
    capx = val.get("demand_sweep_capacity") or {}
    ratio = cal.get("capacity_realisation") or capx.get("capacity_realisation") or capx.get("capacity_ratio")
    V["CAPACITY_RATIO"] = f"{ratio*100:.0f}%" if ratio else "n/a"
    V["DESIGN_MERGE_HARD"] = str(des.get("merge_hard_stops", "n/a"))
    V["DESIGN_HEAD_WAIT"] = fmt(des.get("max_head_wait_s"), 0)
    V["DESIGN_SPILL"] = str(des.get("n_spillback_locks", "n/a"))
    rv = capx.get("vc_levels_probe_vs_sweep") or capx.get("realised_vc_per_level") or {}
    # release-level liveness, gridlock and grid-load sentences (computed from the run's own episodes)
    n_ep = len(ep)
    if "gridlock" in ep.columns:
        g = ep[ep["gridlock"].astype(bool)]
        if len(g):
            combos = ", ".join(f"{c} at {d} demand" for (d, c), _ in g.groupby(["demand_level", "controller"]))
            V["GRIDLOCK_SENTENCE"] = f"In the {V['RUN_LABEL']}, {len(g)} of {n_ep} episodes end in gridlock ({combos}); none under stress."  if not (g["demand_level"] == "stress").any() else f"In the {V['RUN_LABEL']}, {len(g)} of {n_ep} episodes end in gridlock ({combos})."
        else:
            V["GRIDLOCK_SENTENCE"] = f"No episode of the {V['RUN_LABEL']} ends in gridlock."
    else:
        V["GRIDLOCK_SENTENCE"] = ""
    liv = []
    if "max_approach_red_s" in ep.columns:
        liv.append(f"the longest red faced by an occupied approach is {ep['max_approach_red_s'].max():.0f} s")
    if "max_head_wait_s" in ep.columns:
        liv.append(f"the longest time a lane head stood still is {ep['max_head_wait_s'].max():.0f} s")
    if "n_spillback_locks" in ep.columns:
        liv.append(f"{int(ep['n_spillback_locks'].sum())} spillback locks were recorded in {int((ep['n_spillback_locks'] > 0).sum())} episodes")
    V["LIVENESS_SENTENCE"] = (f"Over the {V['RUN_LABEL']}, " + "; ".join(liv) + ".") if liv else ""
    nh = int(ep["n_merge_holds"].sum()) if "n_merge_holds" in ep.columns else None
    ns = int(ep["merge_hard_stops"].sum()) if "merge_hard_stops" in ep.columns else None
    V["N_MERGE_HOLDS"] = "an unknown number of merge holds" if nh is None else ("one merge hold" if nh == 1 else f"{nh} merge holds")
    V["N_MERGE_HARD"] = "an unknown number of hard stops" if ns is None else ("no hard stop" if ns == 0 else ("one hard stop" if ns == 1 else f"{ns} hard stops"))
    V["GRID_LOAD_SENTENCE"] = ""
    if "n_in_network_end" in ep.columns:
        h = ep[ep["demand_level"] == "high"]
        mp = h[h["controller"].isin(["max_pressure", "max_pressure_eps"])]["n_in_network_end"]
        ft = h[h["controller"] == "fixed_time"]["n_in_network_end"]
        if len(mp) and len(ft):
            thr = 380
            V["GRID_LOAD_SENTENCE"] = (f"The demand calibration is single-junction, and on the 2×2 grid the nominal high level is over capacity for the max-pressure family: "
                                       f"{int((mp > thr).sum())} of {len(mp)} high-demand max-pressure and ε-greedy episodes end with more than {thr} vehicles in the network "
                                       f"(median {mp.median():.0f}, maximum {mp.max():.0f}), against a mean of {ft.mean():.0f} (maximum {ft.max():.0f}) under fixed time at the same level. "
                                       f"The learned controller, trained on that teacher, inherits the property, so its high-demand results describe an over-saturated grid.")
    tw = ROOT / "reports/twin/day_kpis_summary.json"
    tw0 = ROOT / "reports/twin/level0_drift.json"
    V["TWIN_SENTENCE"] = ""
    if tw.exists():
        d = json.load(open(tw))
        s0 = ""
        if tw0.exists():
            d0 = json.load(open(tw0))
            s0 = (f" An offline replay of a logged showcase episode through the same estimator tracks the true queues with a mean absolute error of "
                  f"{d0.get('mean_queue_mae', float('nan')):.2f} vehicles per edge.")
        V["TWIN_SENTENCE"] = ("After the release, a first lock-step twin was built on top of the dataset as an extension: a hidden 'physical' run of the simulator feeds "
                              "noisy, delayed detectors (loops, a queue camera per edge with misses, dropped frames and a 1 s delivery delay); a state estimator rebuilds "
                              "the same feature row as the dataset from those detectors alone; the trained sensor-feature model controls the lights from that row; and a "
                              "drift monitor compares estimate and detectors. Over a 24-hour simulated day with the daily demand profile at design level it handled "
                              f"{d.get('n_spawned', 0):,} vehicles with {d.get('gridlocks', 0)} gridlocks and {d.get('invariant_violations', 0)} invariant violations, "
                              f"a mean queue-estimation error of {d.get('mean_queue_mae_truth', float('nan')):.2f} vehicles, and {d.get('stuck_sensor_events', 0)} stuck-camera "
                              f"events detected and recovered." + s0 + " The ambulance channel is read without noise in the twin, as in the dataset.")
    gl = val.get("grid_levels_2x2") or {}
    if gl:
        rows = ["| Level | Controller | Mean delay (s) | Throughput (veh/h) | In network at 900 s | In network at end | Queues grow |", "|---|---|---|---|---|---|---|"]
        for key, r in gl.items():
            lvl, ctl = key.split("/", 1)
            rows.append(f"| {lvl} | {ctl} | {r.get('mean_delay_s', float('nan')):.1f} | {r.get('throughput_vph', float('nan')):.0f} | {r.get('vehicles_in_network_t900', float('nan')):.0f} | {r.get('vehicles_in_network_end', float('nan')):.0f} | {'yes' if r.get('queues_grow') else 'no'} |")
        V["GRID_LEVELS_TABLE"] = "\n".join(rows)
    else:
        V["GRID_LEVELS_TABLE"] = "*(grid load per level not present in this validation run)*"
    V["REALISED_VC"] = ", ".join(f"{k} {v:.2f}" for k, v in rv.items()) if rv else "n/a"
    pm = val.get("paired_mp_vs_ft") or {}
    if pm:
        rows = ["| Demand level | Seeds | Max-pressure mean delay (s) | Fixed-time mean delay (s) | Paired difference, max-pressure minus fixed time (s) | Throughput difference (veh/h) | Max-pressure switches/min |",
                "|---|---|---|---|---|---|---|"]
        for lvl, r in pm.items():
            rows.append(f"| {lvl} | {r.get('n_seeds','')} | {r.get('mp_mean_delay_s', float('nan')):.1f} | {r.get('ft_mean_delay_s', float('nan')):.1f} | "
                        f"{r.get('paired_delay_diff_mp_minus_ft_s', float('nan')):+.1f} ± {r.get('paired_delay_diff_ci95_halfwidth', float('nan')):.1f} | "
                        f"{r.get('paired_throughput_diff_vph', float('nan')):+.0f} | {r.get('mp_switches_per_min', float('nan')):.1f} |")
        V["PAIRED_TABLE"] = "\n".join(rows)
    else:
        V["PAIRED_TABLE"] = "*(paired comparison not present in this validation run)*"
    sw = pd.DataFrame(val["demand_sweep"])[["vc", "lambda_base_vph", "mean_delay_s", "p90_delay_s", "throughput_vph", "max_queue", "webster_d1_s"]].copy()
    sw["vc"] = sw["vc"].map(lambda x: f"{x:.2f}")
    sw["lambda_base_vph"] = sw["lambda_base_vph"].map(lambda x: f"{x:.0f}")
    sw["throughput_vph"] = sw["throughput_vph"].map(lambda x: f"{x:.0f}")
    sw["max_queue"] = sw["max_queue"].map(lambda x: f"{x:.0f}")
    sw.columns = ["x = v/c", "λ per entry (veh/h)", "mean delay (s)", "p90 delay (s)", "throughput (veh/h)", "max queue", "Webster d1 (s)"]
    V["SWEEP_TABLE"] = md_table(sw, 1)
    row07 = [r for r in val["demand_sweep"] if abs(r["vc"] - 0.7) < 1e-6]
    V["SWEEP_07_DELAY"] = fmt(row07[0]["mean_delay_s"], 1) if row07 else "n/a"

    V["N_EPISODES"] = str(len(ep)); V["N_VEHICLES"] = f"{len(vh):,}"; V["N_TICKS"] = f"{len(st):,}"; V["N_1HZ"] = f"{n1:,}"
    V["N_TICK_COLS"] = str(st.shape[1])
    V["TOTAL_VIOL"] = str(int(ep["n_invariant_violations"].sum()))
    V["OVERLAP_RATE"] = f"{ep['n_overlap_fixed'].sum() / max(ep['vehicle_steps'].sum(), 1):.2e}"
    V["RED_COMMITS_RANGE"] = f"{int(ep['n_red_commits'].min())}–{int(ep['n_red_commits'].max())}"
    V["N_GRIDLOCK"] = str(int(ep["gridlock"].sum()))
    main_rows = st[(~st["is_warmup"].astype(bool)) & (~st["is_cooldown"].astype(bool)) & (~st["is_tail"].astype(bool))]
    bal = main_rows["y_mp_action"].value_counts(normalize=True).sort_index()
    V["LABEL_BALANCE"] = ", ".join(f"{'NESW'[int(k)]} {v*100:.0f}%" for k, v in bal.items())
    V["LABEL_SWITCH_RATE"] = f"{(main_rows['y_mp_action'] != main_rows['phase']).mean()*100:.0f}%"
    V["RUNTIME_RANGE"] = f"{ep['runtime_s'].min():.1f}–{ep['runtime_s'].max():.1f}"
    V["RUNTIME_MEAN"] = fmt(ep["runtime_s"].mean(), 1)
    per_ep = ep["runtime_s"].mean()
    V["FULL_RUN_ESTIMATE"] = f"{120 * per_ep * 1.7 / 60:.0f} minutes"
    V["EPISODES_PER_HOUR"] = f"{round(3600 / (per_ep * 1.7), -1):,.0f}"      # paired no-ambulance runs are not logged
    V["SPLIT_COUNTS"] = ", ".join(f"{k} {v}" for k, v in ep["split"].value_counts().sort_index().items())
    sizes = {f: os.path.getsize(run / f) / 1e6 for f in ["episodes.csv", "vehicles.csv", "signal_ticks.csv", "junction_1hz.csv"]}
    V["SIZE_TABLE"] = ", ".join(f"{k} {human_size(v*1e6)}" for k, v in sizes.items())
    V["TOTAL_MB"] = fmt(sum(sizes.values()), 1)
    V["SCALED_TICKS"] = f"{round(len(st) / len(ep) * 120, -3):,.0f}"
    V["SCALED_MB"] = fmt(sum(sizes.values()) / len(ep) * 120, 0)
    if "n_ambulances" in ep.columns:
        ev_eps = ep[ep["n_ambulances"] > 0]
        V["EV_PER_EPISODE"] = fmt(ev_eps["n_ambulances"].mean(), 0) if len(ev_eps) else "0"
    else:
        V["EV_PER_EPISODE"] = "n/a (pilot generated before the ambulance extension)"

    cols = ["episode_id", "demand_level", "demand_asym", "od_pattern", "controller", "split", "n_spawned", "mean_total_delay_s", "p90_total_delay_s", "throughput_vph", "switches_per_min", "n_red_commits", "gridlock"]
    if "n_ambulances" in ep.columns:
        cols += ["n_ambulances", "ev_mean_delay_s", "car_mean_delay_penalty_s"]
    cols = [c for c in cols if c in ep.columns]
    if len(ep) <= 20:
        V["EPISODES_TABLE"] = md_table(ep[cols], 1)
    else:   # a release: summarise instead of listing every episode
        mix = ep.groupby(["demand_level", "controller"]).size().reset_index(name="episodes")
        kpi_cols = [c for c in ["mean_total_delay_s", "p90_total_delay_s", "throughput_vph", "switches_per_min", "n_red_commits", "ev_mean_delay_s", "car_mean_delay_penalty_s", "recovery_time_s"] if c in ep.columns]
        by_level = ep.groupby("demand_level")[kpi_cols].mean().reindex(["low", "design", "high", "stress"]).dropna(how="all").reset_index()
        by_ctl = ep.groupby("controller")[kpi_cols].mean().reset_index()
        splits = ep.groupby(["split", "demand_level"]).size().unstack(fill_value=0).reset_index()
        V["EPISODES_TABLE"] = ("Episode mix (count by demand level and controller):\n\n" + md_table(mix, 0) +
                               "\n\nSplit by demand level (episodes):\n\n" + md_table(splits, 0) +
                               "\n\nMean KPIs by demand level:\n\n" + md_table(by_level, 1) +
                               "\n\nMean KPIs by controller:\n\n" + md_table(by_ctl, 1))

    import hashlib
    sums = []
    for f in ["episodes.csv", "vehicles.csv", "signal_ticks.csv", "junction_1hz.csv", "network_edges.csv", "data_dictionary.csv", "config.json"]:
        fp = run / f
        if fp.exists():
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            sums.append((f, h, fp.stat().st_size))
    (run / "SHA256SUMS.txt").write_text("".join(f"{h}  {f}\n" for f, h, _ in sums))
    V["SHA256_TABLE"] = "\n".join(["| File | Size | SHA-256 (first 16) |", "|---|---|---|"] +
                                  [f"| `{f}` | {human_size(sz)} | `{h[:16]}…` |" for f, h, sz in sums])

    tick_dd = dd[dd["table"] == "signal_ticks"][["column", "dtype", "unit", "namespace", "description"]].copy()
    tick_dd["description"] = tick_dd["description"].str.replace("|", "/", regex=False)
    tick_dd["description"] = tick_dd["description"].str.replace(r"\s*\(DECISIONS §[^)]*\)", "", regex=True)   # repository pointer, unresolvable from this document
    V["TICK_DICT_TABLE"] = md_table(tick_dd, 0)

    # ML results: embed reports/ml/RESULTS.md if the training pipeline has run
    res_md = ROOT / "reports/ml/RESULTS.md"
    if res_md.exists():
        body = res_md.read_text()
        body = re.sub(r"^(#+)", lambda m: "#" * min(len(m.group(1)) + 2, 6), body, flags=re.M)   # demote headings under 12.5
        body = body.replace("](figures/", "](../ml/figures/").replace("](reports/ml/figures/", "](../ml/figures/")
        V["ML_RESULTS"] = body
    else:
        V["ML_RESULTS"] = "*Results of the pilot models are inserted here once the modelling stage's results are available.*"

    tpl = open(HERE / "Methodology_template.md").read()
    missing = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", tpl)) - set(V))
    if missing:
        print("UNFILLED placeholders:", missing)
    out_md = HERE / "Methodology_Dataset_Generation.md"
    text = tpl
    for k, v in V.items():
        text = text.replace("{{" + k + "}}", str(v))
    out_md.write_text(text)
    print("wrote", out_md)

    css = ROOT / "reports/_build/style.css"
    base = ["--from", "markdown+smart+tex_math_dollars", "--resource-path", f"{HERE}:{ROOT}/reports", "--toc", "--toc-depth=2"]
    subprocess.run([PANDOC, str(out_md), "-o", str(HERE / "Methodology_Dataset_Generation.docx")] + base, check=True, cwd=HERE)
    subprocess.run([PANDOC, str(out_md), "-o", str(HERE / "Methodology_Dataset_Generation.html"), "--standalone", "--embed-resources",
                    "--section-divs", "--css", str(css), "--css", str(ROOT / "reports/_build/methodology.css"), "--mathml"] + base, check=True, cwd=HERE)
    html = HERE / "Methodology_Dataset_Generation.html"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=5000",
                    f"--print-to-pdf={HERE / 'Methodology_Dataset_Generation.pdf'}", f"file://{html}"],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in ["Methodology_Dataset_Generation.docx", "Methodology_Dataset_Generation.pdf"]:
        p = HERE / f
        print(f, f"{p.stat().st_size/1e6:.1f} MB" if p.exists() else "MISSING")


if __name__ == "__main__":
    main()
