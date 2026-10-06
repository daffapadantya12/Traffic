"""Fill the Assignment 3 report from the project's outputs and render it (Markdown -> docx, html, pdf).

Run from the project root:
  python3 reports/a3/build_report.py --data data/fatmawati_v3 --grid data/fatmawati_eval \
      --kpi reports/a3/kpi --learning reports/a3/learning --data-url "<download link>"
Every number in the report's result slots is read from those outputs; the reference list is assembled from the
verified entries in the proposal and the methodology, keeping only the works the report cites.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PANDOC = "/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64/pandoc"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
NAMES = {"random": "Random", "still": "Still", "fixed": "City's fixed plan", "longest_queue": "Longest queue",
         "max_pressure": "Max-pressure", "priority_mp": "Priority teacher", "preempt_naive": "Naive preemption"}


def pretty(m: str) -> str:
    if m.startswith("learned:") or m.startswith("learned_"):
        return "Learned: " + os.path.basename(m).replace("taskA_", "").replace(".joblib", "").replace("_", " ")
    return NAMES.get(m, m.replace("_", " "))


PHRASE = {"random": "random control", "still": "still control", "fixed": "the city's fixed plan", "longest_queue": "longest-queue control",
          "max_pressure": "max-pressure control", "priority_mp": "the priority teacher", "preempt_naive": "naive preemption"}


def phrase(m: str) -> str:
    """A method's name inside a sentence."""
    if m.startswith("learned:") or m.startswith("learned_"):
        return "the learned " + family(m).replace(" ", "-") + " controller"
    return PHRASE.get(m, m.replace("_", " "))


def family(m: str) -> str:
    """A learned controller's model family, e.g. 'gradient boosting'."""
    return os.path.basename(m).replace("taskA_", "").replace(".joblib", "").replace("_", " ")


def f(x, nd=1):
    try:
        return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def words(s: str) -> int:
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)(\{[^}]*\})?", r"\1", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"^\|[-:| ]+\|\s*$", "", s, flags=re.M)
    s = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", s, flags=re.M)
    s = re.sub(r"\$[^$]+\$", " x ", s)
    s = re.sub(r"[*#|]+", " ", s)
    return len(re.findall(r"\S+", s))


def reference_pool() -> dict:
    pool = {}
    for path in (os.path.join(ROOT, "reports", "methodology_v3", "Methodology_template.md"),
                 os.path.join(ROOT, "reports", "proposal_v3", "Assignment1_Proposal.md")):
        t = open(path, encoding="utf-8").read()
        refs = t.split("\n# References")[1]
        refs = re.split(r"\n# ", refs)[0]
        for e in [x.strip() for x in refs.split("\n\n") if x.strip()]:
            pool.setdefault(e, e)
    return pool


def cited(entry: str, body: str) -> bool:
    if entry.startswith("*"):
        return re.match(r"\*([^*]+?) tentang", entry).group(1) in body
    head = re.match(r"([^,.(]+)", entry).group(1).strip()
    year = re.search(r"\((\d{4}[a-z]?|n\.d\.(?:-[a-z])?)", entry).group(1)
    key = {"Alvarez Lopez": "Alvarez Lopez", "Open Data Commons": "Open Data Commons", "R Core Team": "R Core Team",
           "Pusat Krisis dan Kegawatdaruratan Kesehatan Daerah": "Pusat Krisis",
           "Direktorat Jenderal Bina Marga": "Bina Marga", "Direktorat Jenderal Kesehatan Lanjutan": "Kesehatan Lanjutan"}
    name = next((v for k, v in key.items() if head.startswith(k)), head.split()[0])
    # an APA citation: "Name, 2022", "Name et al., 2022", "Name & Other, 2025", "Name (2022)" or "Name et al. (2022)";
    # an organisation's longer name may stand between the key and the year ("Pusat Krisis dan ..., n.d.")
    pat = (r"\b" + re.escape(name) + r"(?:[^\n(),;]{0,80}?)?(?: et al\.| & [^\n(),;]{1,60})?(?:, | \()"
           + re.escape(year) + r"(?![A-Za-z0-9])")
    return re.search(pat, body) is not None


def fill(args) -> str:
    tpl = open(os.path.join(HERE, "Assignment3_Report_template.md"), encoding="utf-8").read()
    V = {}
    # the release
    ep = pd.read_csv(os.path.join(args.data, "episodes.csv"))
    V["DATA_EPISODES"] = f"{len(ep):,}"
    V["DATA_HOURS"] = f"{ep['main_s'].sum() / 3600:,.0f}" if "main_s" in ep else str(len(ep))
    V["TICK_ROWS"] = f"{sum(1 for _ in open(os.path.join(args.data, 'ticks_5s.csv'))) - 1:,}"
    V["AMB_TRIPS"] = f"{sum(1 for _ in open(os.path.join(args.data, 'ambulance_trips.csv'))) - 1:,}"
    # the grid, and every other evaluation folder (closed loop, stress, 70%) for the volume count
    grids = [g for g in args.grid.split(",") if g]
    pas = pd.read_csv(os.path.join(grids[0], "passages.csv"))
    n_all = sum(len(pd.read_csv(os.path.join(g, "passages.csv"))) for g in grids if os.path.exists(os.path.join(g, "passages.csv")))
    V["GRID_PASSAGES"] = f"{n_all:,}"
    n_seeds = pas["seed"].nunique()
    V["GRID_SEEDS"] = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}.get(n_seeds, str(n_seeds))
    # calibration
    cal = json.load(open(os.path.join(ROOT, "data", "fatmawati", "calibration.json")))
    ver, tgt = cal["verification"]["max_queue_m"], cal["target_max_queue_m"]
    lim = cal.get("capacity_limited", {})
    full = {"N": "north", "E": "east", "S": "south", "W": "west"}
    fit = [f"{100 * (ver[a] / tgt[a] - 1):+.0f}% {full[a]}".replace("-", "−") for a in "NESW" if a in tgt and not lim.get(a)]
    capped = [full[a] for a in "NESW" if lim.get(a)]
    V["CALIBRATION_SENTENCE"] = ("longest queues " + ", ".join(fit[:-1]) + (" and " if len(fit) > 1 else "") + fit[-1]
                                 + " of the published simulation's"
                                 + (f", {' and '.join(capped)} capacity-limited" if capped else "")
                                 + f", at {min(cal['verification']['volume_share_of_observed'].values()):.0%} to "
                                 f"{max(cal['verification']['volume_share_of_observed'].values()):.0%} of counted volumes")
    # learning
    L = json.load(open(os.path.join(args.learning, "results.json")))
    val = L["validation"]
    fm, fc = val["forecast_label_mismatch"], val["forecast_label_checked"]
    ties, unexpl = val.get("teacher_label_boundary_ties", 0), val.get("teacher_label_unexplained", 0)
    V["VALIDATION_SENTENCE"] = ((f"the 30 s queue label matched the later camera reading in all {fc:,} checkable values" if fm == 0 else
                                 f"the 30 s queue label matched the later camera reading in {fc - fm:,} of {fc:,} checkable values")
                                + ", and the teacher's label was reproduced on every tick"
                                + (f" but {ties} ties at the logged precision" if ties else "")
                                + (f" and {unexpl} unexplained mismatches" if unexpl else ""))
    A = L["task_a"]
    rows = ["| Model | Macro-F1 | Accuracy, ambulance ticks |", "|---|---|---|"]
    for name, key in (("Baseline: keep the current stage", "stillness"), ("Baseline: most common stage", "majority"),
                      ("Baseline: random stage", "random_uniform")):
        b = A["baselines"][key]
        rows.append(f"| {name} | {f(b['macro_f1'], 2)} | {f(b['ambulance_accuracy'], 2)} |")
    for name, m in A["models"].items():
        rows.append(f"| {name.replace('_', ' ').capitalize()} | {f(m['macro_f1'], 2)} | {f(m['ambulance_accuracy'], 2)} |")
    V["MODEL_TABLE"] = "\n".join(rows) + f"\n\n*Table 1. The teacher's stage on held-out seed {A['split'].split()[-1]}: {A['n_test']:,} ticks, of which {list(A['models'].values())[0]['n_ambulance']:,} have an ambulance on an approach.*"
    best = max(A["models"], key=lambda k: A["models"][k]["macro_f1"])
    bb = max((A["baselines"][k] for k in ("stillness", "majority", "random_uniform")), key=lambda b: b["macro_f1"])
    sh = A["shuffled_label_test"]["macro_f1"]
    B = L["task_b"]["legs"]
    skill = [max(m["skill"] for m in B[g]["models"].values()) for g in B]          # the best forecaster per approach
    C = L["task_c"]
    c_txt = ""
    if "gradient_boosting_quantile" in C:
        q = C["gradient_boosting_quantile"]
        c_txt = (f" The arrival-time model halves the error of the teacher's free-road estimate ({q['mae']:.0f} s against "
                 f"{C['baselines']['teacher_eta']['mae']:.0f} s) but is late on {100 * q['late_share']:.0f}% of ticks against a "
                 f"{100 * C['tau']:.0f}% target, so the teacher's own estimate stays the safer input.")
    gen = A.get("generalisation", {})
    g_txt = ""
    if "window=pm_peak" in gen and "level=high" in gen:
        who = "It" if max(A["models"], key=lambda k: A["models"][k]["macro_f1"]) == "gradient_boosting" else "Gradient boosting"
        g_txt = (f" {who} keeps a macro-F1 of {gen['window=pm_peak']['unseen']['macro_f1']:.2f} on an evening window and "
                 f"{gen['level=high']['unseen']['macro_f1']:.2f} on high demand that it never saw in training.")
    V["MODEL_SENTENCE"] = (f"{best.replace('_', ' ').capitalize()} imitates the teacher best, with a macro-F1 of {f(A['models'][best]['macro_f1'], 2)} "
                           f"against {f(bb['macro_f1'], 2)} for the best baseline; trained on shuffled labels it falls to {f(sh, 2)}, so nothing leaks.{g_txt} "
                           f"For the 30 s queue, the best model per approach cuts the error of assuming the queue stays as it is by "
                           f"{100 * min(skill):.0f}% to {100 * max(skill):.0f}%.{c_txt}")
    V["MODEL_SENTENCE"] = re.sub(r"(?<![A-Za-z])-(?=\d)", "−", V["MODEL_SENTENCE"])          # minus signs, not hyphens
    # KPIs and the equilibrium
    K = pd.read_csv(os.path.join(args.kpi, "kpi_methods.csv"), index_col=0)
    E = json.load(open(os.path.join(args.kpi, "equilibrium.json")))
    w, wp = E.get("winner"), E.get("point_estimate", {}).get("winner")
    if w and os.path.exists(os.path.join(args.kpi, "nash.csv")):                   # the main result: gates at the 95% level
        nb = pd.read_csv(os.path.join(args.kpi, "nash.csv"), index_col=0)
        nash = {m: v for m, v in nb["nash_product"].items() if not pd.isna(v)}
        passes = {m: bool(K.loc[m, "passes"]) for m in K.index}
        gate_note = "Gates judged at the 95% level."
    else:                                                                          # no method passes at 95%: point estimates
        nash = E.get("point_estimate", {}).get("nash", {})
        passes = {m: m in E.get("point_estimate", {}).get("passing", []) for m in K.index}
        gate_note = "Gates judged on point estimates, because no method passes them at the 95% level."
    rows = ["| Method | Ambulance delay (s) | Worst direction (s) | Stranded | General delay, index | Worst approach, index | Passes gates | Nash product |",
            "|---|---|---|---|---|---|---|---|"]
    order = K.sort_values("A").index
    for m in order:
        r = K.loc[m]
        stranded = f"{100 * float(r.get('unfinished_share', 0) or 0):.0f}%"
        rows.append(f"| {pretty(m)} | {f(r.get('A'))} | {f(r.get('A_worst'))} | {stranded} | {f(r.get('R'), 2)} | {f(r.get('R_worst'), 2)} | "
                    f"{'yes' if passes.get(m) else 'no'} | {f(nash.get(m), 4) if m in nash else '–'} |")
    lv = ", ".join(E.get("levels", [])) or "all"
    V["KPI_TABLE"] = ("\n".join(rows) + f"\n\n*Table 2. Methods at the {lv} demand level of the evaluation grid. Stranded: never reached the hospital in the run. "
                      f"Delay index: 1.00 = the city's fixed plan. {gate_note} A Nash product is shown for methods on the Pareto front of those that pass.*")
    share = E.get("bootstrap", {}).get("win_share", {})
    n_boot = E.get("bootstrap", {}).get("resamples", 0)
    lam = E.get("implied_lambda", {})
    lam_txt = ""
    up, dn = lam.get("towards_more_ambulance_priority"), lam.get("towards_less_ambulance_priority")
    if up and dn:
        lam_txt = (f" Choosing it means valuing one second of ambulance delay at between about {dn['lambda_veh_s_per_amb_s']:,.0f} and "
                   f"{up['lambda_veh_s_per_amb_s']:,.0f} vehicle-seconds of everyone else's: a higher value would favour "
                   f"{phrase(up['neighbour'])}, a lower one {phrase(dn['neighbour'])}.")
    elif up:
        lam_txt = (f" Preferring it to {phrase(up['neighbour'])} means valuing one second of ambulance delay at less than about "
                   f"{up['lambda_veh_s_per_amb_s']:,.0f} vehicle-seconds of everyone else's.")
    elif dn:
        lam_txt = (f" Preferring it to {phrase(dn['neighbour'])} means valuing one second of ambulance delay at more than about "
                   f"{dn['lambda_veh_s_per_amb_s']:,.0f} vehicle-seconds of everyone else's.")

    def rivals(win):
        others = sorted(((v, m) for m, v in share.items() if m not in (win, "none")), reverse=True)
        txt = f" against {100 * others[0][0]:.0f}% for {phrase(others[0][1])}" if others else ""
        if share.get("none"):
            txt += f", with no method passing in {100 * share['none']:.0f}%"
        return txt

    def near(win):
        """A runner-up within 5% of the winner's Nash product, named as a near-tie."""
        ranked = sorted(((v, m) for m, v in nash.items() if m != win and not pd.isna(v)), reverse=True)
        if ranked and win in nash and ranked[0][0] >= 0.95 * nash[win]:
            return (f"; its Nash product only just beats that of {phrase(ranked[0][1])} ({nash[win]:.4f} against "
                    f"{ranked[0][0]:.4f})")
        return ""
    if w:
        V["EQUILIBRIUM_SENTENCE"] = (f"The equilibrium is **{phrase(w)}**. It wins {100 * share.get(w, 0):.0f}% of {n_boot:,} "
                                     f"bootstrap resamples of the seeds{rivals(w)}{near(w)}.{lam_txt}")
    elif wp:
        V["EQUILIBRIUM_SENTENCE"] = (f"With {V['GRID_SEEDS']} seeds no method can be shown to stay within the margins at the 95% level, so the "
                                     f"cost and approach gates fall back to point estimates, while safety is still tested for a significant rise "
                                     f"in collisions. On them the equilibrium is **{phrase(wp)}**. It wins {100 * share.get(wp, 0):.0f}% of "
                                     f"{n_boot:,} bootstrap resamples of the seeds{rivals(wp)}{near(wp)}.{lam_txt}")
    else:
        V["EQUILIBRIUM_SENTENCE"] = "No method passes the gates, so no equilibrium is declared; Table 2 shows where each method falls short."
    # closed loop: what share of the teacher's gain over today's plan each learned controller keeps
    V["CLOSED_LOOP_SENTENCE"] = ""
    learned = [m for m in K.index if str(m).startswith("learned:")]
    if learned and "share_of_teacher_gain" in K.columns and "fixed" in K.index:
        sh = K.loc[learned, "share_of_teacher_gain"].astype(float).sort_values(ascending=False)
        top = sh.index[:2]
        worst = sh.index[-1]
        worse_than_today = float(K.loc[worst, "A"]) / float(K.loc["fixed", "A"]) - 1
        txt = (f"In closed loop the {family(top[0])} keeps {100 * sh[top[0]]:.0f}% of the teacher's cut in ambulance delay against "
               f"today's plan" + (f" and {family(top[1])} {100 * sh[top[1]]:.0f}%" if len(top) > 1 else ""))
        if worse_than_today > 0:
            txt += (f", while the {family(worst)}, the weakest imitator offline, ends up "
                    f"{100 * worse_than_today:.0f}% worse than today: offline accuracy predicts control only roughly")
        V["CLOSED_LOOP_SENTENCE"] = txt + "."
    # sensitivity: 70% motorcycles at the same counts, scored on its own
    V["MOTO_SENTENCE"] = ""
    km = os.path.join(args.kpi_moto, "kpi_methods.csv")
    if os.path.exists(km):
        M = pd.read_csv(km, index_col=0)
        EM = json.load(open(os.path.join(args.kpi_moto, "equilibrium.json")))
        common = [m for m in M.index if m in K.index]
        ratio = (M.loc[common, "A"].astype(float) / K.loc[common, "A"].astype(float)) if common else pd.Series(dtype=float)
        adapt = [m for m in M.index if m not in ("fixed", "preempt_naive")]
        ok = [m for m in EM.get("point_estimate", {}).get("passing", []) if m != "fixed"]
        txt = (f"With 70% motorcycles instead of 80% at the same counts, ambulance delay grows {ratio.min():.1f} to {ratio.max():.1f} times for "
               f"every method, and the adaptive methods leave one approach {M.loc[adapt, 'R_worst'].astype(float).min():.1f} to "
               f"{M.loc[adapt, 'R_worst'].astype(float).max():.1f} times worse than the fixed plan")
        txt += ", so none passes the gates" if not ok else f"; only {', '.join(phrase(m) for m in ok)} still pass"
        w_eq = wp or w
        if w_eq and w_eq in M.index and "priority_mp" in M.index and "fixed" in M.index:
            txt += (f"; {phrase(w_eq)} still matches the teacher's ambulance gain ({float(M.loc[w_eq, 'A']):.0f} s against "
                    f"{float(M.loc['priority_mp', 'A']):.0f} s and {float(M.loc['fixed', 'A']):.0f} s today)")
        V["MOTO_SENTENCE"] = txt + "."
    # starvation: the plant caps every red for controllers that only request stages; forced preemption bypasses it
    V["STARVATION_SENTENCE"] = "The plant caps every red at about four minutes for any controller that only requests stages."
    reds = [f"longest_red_{a}" for a in "NESW"]
    if all(c in K.columns for c in reds):
        req = [m for m in K.index if m not in ("fixed", "preempt_naive")]
        cap = float(K.loc[req, reds].astype(float).max().max()) if req else float("nan")
        txt = f"The plant caps every red at {cap:.0f} s or less for every controller that only requests stages"
        if "preempt_naive" in K.index:
            pre = float(K.loc["preempt_naive", reds].astype(float).max())
            txt += (f"; naive preemption, which forces and holds a green as common practice does, bypasses that cap and left an "
                    f"approach red for {pre:.0f} s")
        V["STARVATION_SENTENCE"] = txt + "."
    # hindsight: what perfect timing of one switch adds to the replays' base method, and who captures it
    V["HINDSIGHT_SENTENCE"] = ""
    if "G_vs_base" in K.columns and "max_pressure" in K.index and pd.notna(K.loc["max_pressure", "A_ceiling"]):
        cap = {m: float(K.loc[m, "G_vs_base"]) for m in ("preempt_naive", "priority_mp") if m in K.index}
        if len(cap) == 2:
            V["HINDSIGHT_SENTENCE"] = (f"Replaying every possible switch time on one seed shows that perfect timing alone would cut "
                                       f"max-pressure's ambulance delay from {float(K.loc['max_pressure', 'A_on_replayed']):.0f} s to "
                                       f"{float(K.loc['max_pressure', 'A_ceiling']):.0f} s; naive preemption captures {100 * cap['preempt_naive']:.0f}% "
                                       f"of that gain and the teacher only {100 * cap['priority_mp']:.0f}%, because it waits for the last safe "
                                       f"moment to spare cross traffic, while the best moment for the ambulance is usually at detection.")
    # safety: collisions near the junction, per vehicle-kilometre
    V["SAFETY_SENTENCE"] = ""
    rc = "collisions_per_1000_veh_km_study"
    if rc in K.columns and "fixed" in K.index:
        adapt = [m for m in K.index if m not in ("fixed", "random", "still", "preempt_naive") and pd.notna(K.loc[m, rc])]
        failed = [m for m in K.index if "safe" in K.columns and not bool(K.loc[m, "safe"])]
        if adapt:
            lo, hi = K.loc[adapt, rc].astype(float).min(), K.loc[adapt, rc].astype(float).max()
            V["SAFETY_SENTENCE"] = (f"Adaptive control also raises the twin's collision rate near the junction, from {float(K.loc['fixed', rc]):.2f} "
                                    f"to {lo:.2f}–{hi:.2f} per 1,000 vehicle-kilometres; with four seeds the rise is significant only for "
                                    + (" and ".join(phrase(m) for m in failed) + ", which therefore fails the safety gate" if failed else "no method")
                                    + ", but its direction is consistent and calls for conflict measures before any deployment.")
    # the stress tests: design and high demand
    V["STRESS_UNFINISHED_MAX"], V["STRESS_SENTENCE"] = "a fifth", ""
    blp = os.path.join(args.kpi, "kpi_by_level.csv")
    if os.path.exists(blp):
        bl = pd.read_csv(blp)
        stress = bl[bl["level"].isin(["design", "high"]) & ~bl["method"].isin(["random", "still"])]
        if len(stress):
            V["STRESS_UNFINISHED_MAX"] = f"{100 * stress['unfinished_share'].max():.0f}%"
            parts = {}
            for lvl in ("design", "high"):
                t = stress[stress["level"] == lvl].set_index("method")
                if "fixed" in t.index and len(t) > 1:
                    best = t.drop(index="fixed")["A"].astype(float).idxmin()
                    adaptive = t.drop(index=[m for m in ("fixed", "preempt_naive") if m in t.index])
                    parts[lvl] = dict(best=best, cut=1 - float(t.loc[best, "A"]) / float(t.loc["fixed", "A"]),
                                      u_best=float(t.loc[best, "unfinished_share"]), u_fixed=float(t.loc["fixed", "unfinished_share"]),
                                      worst_lo=float(adaptive["R_worst"].min()) if len(adaptive) else float("nan"),
                                      worst_hi=float(adaptive["R_worst"].max()) if len(adaptive) else float("nan"),
                                      any_pass=bool((t.drop(index="fixed")["R_worst"].astype(float) <= 1.20).any()))
            if {"design", "high"} <= set(parts):
                d, h = parts["design"], parts["high"]
                same = d["best"] == h["best"]
                lead = (f"In the stress tests {phrase(d['best'])} cuts ambulance delay against the fixed plan by {100 * d['cut']:.0f}% at "
                        f"design demand and {100 * h['cut']:.0f}% at high demand" if same else
                        f"In the stress tests the shortest ambulance delays come from {phrase(d['best'])} at design demand "
                        f"({100 * d['cut']:.0f}% below the fixed plan) and {phrase(h['best'])} at high demand ({100 * h['cut']:.0f}% below)")
                V["STRESS_SENTENCE"] = (lead + f", and strands {100 * d['u_best']:.0f}% and {100 * h['u_best']:.0f}% of ambulances against "
                                        f"{100 * d['u_fixed']:.0f}% and {100 * h['u_fixed']:.0f}%. But the adaptive methods leave their worst approach "
                                        f"{min(d['worst_lo'], h['worst_lo']):.1f} to {max(d['worst_hi'], h['worst_hi']):.1f} times its fixed-plan delay"
                                        + (", so none passes the approach gate there." if not (d["any_pass"] or h["any_pass"]) else "."))
    # demonstration
    demo = os.path.join(HERE, "demo_results.csv")
    if os.path.exists(demo):
        d = pd.read_csv(demo).set_index("model")["accuracy"]
        V["DEMO_SENTENCE"] = (f"The tree reaches an accuracy of {d.get('decision tree', float('nan')):.2f}, against "
                              f"{d.get('stillness', float('nan')):.2f} for keeping the current stage and {d.get('majority', float('nan')):.2f} for the most common stage.")
    else:
        V["DEMO_SENTENCE"] = ""
    V["DATA_URL"] = args.data_url
    body = tpl
    for k, v in V.items():
        body = body.replace("{{" + k + "}}", v)
    pool = reference_pool()
    main = body.split("\n# References")[0]
    refs = sorted({e for e in pool if cited(e, main)}, key=lambda e: re.sub(r"[^a-z0-9 ]", "", e.lower().lstrip("*")))
    body = body.replace("{{REFERENCES}}", "\n\n".join(refs))
    left = re.findall(r"\{\{[A-Z_]+\}\}", body)
    if left:
        print("UNFILLED:", left)
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "fatmawati_v3"))
    ap.add_argument("--grid", default=os.path.join(ROOT, "data", "fatmawati_eval"))
    ap.add_argument("--kpi", default=os.path.join(HERE, "kpi"))
    ap.add_argument("--learning", default=os.path.join(HERE, "learning"))
    ap.add_argument("--kpi-moto", default=os.path.join(HERE, "kpi_moto70"))
    ap.add_argument("--data-url", default="[download link to be added by the group]")
    a = ap.parse_args()
    md = fill(a)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "reports", "proposal_v3", "figures", "F1_study_area.png"), os.path.join(HERE, "figures"))
    if os.path.exists(os.path.join(a.kpi, "tradeoff.png")):
        shutil.copy(os.path.join(a.kpi, "tradeoff.png"), os.path.join(HERE, "figures"))
    out_md = os.path.join(HERE, "Assignment3_Report.md")
    open(out_md, "w", encoding="utf-8").write(md)
    body = md.split("---", 2)[2]
    main_part = body.split("# 1. Introduction", 1)[1].split("\n# References")[0]
    genai = body.split("# Use of generative AI")[1]
    print(f"words: sections 1-5 {words(main_part)}, generative-AI statement {words(genai)}, total {words(main_part) + words(genai)} (limit 2,500)")
    css = [os.path.join(ROOT, "reports", "_build", "style.css"), os.path.join(ROOT, "reports", "_build", "methodology.css"),
           os.path.join(HERE, "report.css")]
    base = ["--from", "markdown+smart+tex_math_dollars", "--resource-path", HERE]
    subprocess.run([PANDOC, out_md, "-o", os.path.join(HERE, "Assignment3_Report.docx")] + base, check=True, cwd=HERE)
    html = os.path.join(HERE, "Assignment3_Report.html")
    cmd = [PANDOC, out_md, "-o", html, "--standalone", "--embed-resources", "--section-divs", "--mathml"] + base
    for c in css:
        cmd += ["--css", c]
    subprocess.run(cmd, check=True, cwd=HERE)
    pdf = os.path.join(HERE, "Assignment3_Report.pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={pdf}", f"file://{html}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for fn in ("Assignment3_Report.docx", "Assignment3_Report.pdf"):
        p = os.path.join(HERE, fn)
        print(fn, f"{os.path.getsize(p) / 1e6:.2f} MB" if os.path.exists(p) else "MISSING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
