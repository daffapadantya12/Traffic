"""M0 validation experiments (DECISIONS §8).  Run: python3 -m traffic_twin.validate

1. queue discharge -> saturation flow and start-up lost time
2. single junction, fixed time, demand sweep -> delay rises and blows up near capacity
3. 2x2 grid, design demand, max_pressure -> invariants, delay, all-red extensions
4. ring road fundamental diagram (optional, IDM only)
5. max_pressure versus fixed_time on paired seeds (design, high)
6. 2x2 grid at design / high / stress under fixed_time and max_pressure: network load, queue growth

Traffic-engineering idea: before a simulator produces data it must reproduce
textbook behaviour: a saturation flow of 1600-1900 veh/h/lane, delay that
explodes when demand reaches capacity, and a roughly triangular flow-density curve.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys
import time

import numpy as np

from .calibrate import calibrate_demand, lambda_for_vc, measure_saturation_flow
from .config import GENERATOR_VERSION, SimConfig
from .sim import Scenario, World
from .vehicles import acc_accel, ballistic_step, sample_params

FIG_DIR = os.path.join("reports", "figures")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def webster_uniform_delay(x: float, C: float, g: float) -> float:
    """Webster's uniform delay term d1 = C(1 - g/C)^2 / (2(1 - x g/C))."""
    return C * (1 - g / C) ** 2 / (2 * (1 - x * g / C))


# --------------------------------------------------------------------- 1. discharge
def run_discharge(cfg: SimConfig, fig_dir: str) -> dict:
    m = measure_saturation_flow(cfg)
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(np.arange(1, m["n"] + 1), m["t_cross"] - m["t_green"], "o-")
    ax[0].set_xlabel("vehicle number in queue")
    ax[0].set_ylabel("stop-line crossing time after green [s]")
    ax[0].set_title("Queue discharge")
    ax[1].bar(np.arange(1, m["n"] + 1), m["headways"])
    ax[1].axhline(m["h_sat"], color="k", ls="--", label=f"h_sat = {m['h_sat']:.2f} s")
    ax[1].set_xlabel("vehicle number")
    ax[1].set_ylabel("headway [s]")
    ax[1].set_title(f"saturation flow {m['sat_flow_vphpl']:.0f} veh/h/lane, lost time {m['startup_lost_time_s']:.2f} s")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "01_queue_discharge.png"), dpi=120)
    plt.close(fig)
    lo, hi = cfg.sat_flow_expected_range
    return dict(sat_flow_vphpl=m["sat_flow_vphpl"], h_sat_s=m["h_sat"],
                startup_lost_time_s=m["startup_lost_time_s"],
                in_expected_range=bool(lo <= m["sat_flow_vphpl"] <= hi))


# --------------------------------------------------------------------- 2. demand sweep
def run_demand_sweep(cfg: SimConfig, capacity_vph: float, fig_dir: str,
                     vcs=(0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.1), seed: int = 11) -> list:
    """1x1 fixed-time sweep; v/c is against the MEASURED capacity per approach (§11.9)."""
    cfg1 = cfg.copy(grid_x=1, grid_y=1)
    rows = []
    for vc in vcs:
        lam = lambda_for_vc(cfg1, vc, capacity_vph)
        sc = Scenario(demand_level=f"vc{vc}", lambda_base_vph=lam, controller="fixed_time")
        w = World(cfg1, sc, seed=seed).run()
        r = w.episode_row()
        rows.append(dict(vc=vc, lambda_base_vph=lam, mean_delay_s=r["mean_total_delay_s"],
                         p90_delay_s=r["p90_total_delay_s"], throughput_vph=r["throughput_vph"],
                         max_queue=r["max_queue_veh"], gridlock=r["gridlock"], runtime_s=r["runtime_s"],
                         webster_d1_s=webster_uniform_delay(vc, cfg.fixed_time_cycle_s, cfg.effective_green_s) if vc <= 0.7 else math.nan,
                         n_invariant_violations=r["n_invariant_violations"]))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["vc"] for r in rows], [r["mean_delay_s"] for r in rows], "o-", label="simulated mean total delay")
    wx = [r["vc"] for r in rows if not math.isnan(r["webster_d1_s"])]
    ax.plot(wx, [r["webster_d1_s"] for r in rows if not math.isnan(r["webster_d1_s"])], "s--", label="Webster uniform delay d1 (x <= 0.7)")
    ax.set_xlabel("v/c on the busiest approach (fixed time, 1x1; measured capacity)")
    ax.set_ylabel("mean total delay [s/veh]")
    ax.set_title("Delay vs demand")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "02_delay_vs_demand.png"), dpi=120)
    plt.close(fig)
    return rows


# --------------------------------------------------------------------- 3. 2x2 design episode
def run_design_episode(cfg: SimConfig, fig_dir: str, seed: int = 21) -> dict:
    sc = Scenario(demand_level="design", lambda_base_vph=cfg.lambda_base_vph["design"], controller="max_pressure")
    w = World(cfg, sc, seed=seed).run()
    r = w.episode_row()
    exts = [e for p in w.plants for e in p.allred_extensions]
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(exts, bins=20)
    ax[0].set_xlabel("all-red extension beyond 4 s [s]")
    ax[0].set_ylabel("count")
    ax[0].set_title(f"All-red extensions (mean {np.mean(exts):.2f} s)")
    q = {}
    for row in w.junction_1hz_rows:
        q.setdefault(row["junction_id"], []).append((row["t"], row["q_total"]))
    for jid, pts in q.items():
        ax[1].plot([p[0] for p in pts], [p[1] for p in pts], lw=0.7, label=f"J{jid}")
    ax[1].axvspan(0, cfg.warmup_s, color="grey", alpha=0.2)
    ax[1].axvspan(cfg.spawn_end_s, cfg.episode_length_s, color="grey", alpha=0.2)
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("q_total (vehicles with v < 2 m/s)")
    ax[1].set_title("2x2 design demand, max_pressure")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "03_design_episode.png"), dpi=120)
    plt.close(fig)
    inv = w.invariant_counts()
    n_red_onsets = sum(p.n_switches for p in w.plants)
    lo, hi = cfg.switch_rate_band
    return dict(
        mean_delay_s=r["mean_total_delay_s"], p90_delay_s=r["p90_total_delay_s"],
        throughput_vph=r["throughput_vph"], n_spawned=r["n_spawned"], switches_per_min=r["switches_per_min"],
        n_red_commits=r["n_red_commits"], n_hard_brakes=r["n_hard_brakes"], n_box_blocked=r["n_box_blocked"],
        n_overlap_fixed=r["n_overlap_fixed"], vehicle_steps=r["vehicle_steps"],
        allred_extension_mean_s=r["allred_extension_mean_s"], max_queue_veh=r["max_queue_veh"],
        gridlock=r["gridlock"], runtime_s=r["runtime_s"], invariants=inv,
        delay_plausible=bool(30.0 <= r["mean_total_delay_s"] <= 90.0),
        # v2.0 safety (§12.4): collisions are an invariant; near misses, PET, braking runs are KPIs
        collisions_ok=bool(r["n_collisions"] == 0),
        n_collisions=r["n_collisions"], n_conflicts=r["n_conflicts"], n_near_misses=r["n_near_misses"],
        near_miss_per_1000_veh=r["near_miss_per_1000_veh"], min_ttc_s=r["min_ttc_s"], min_pet_s=r["min_pet_s"],
        n_pet_serious=r["n_pet_serious"], min_gap_moving_m=r["min_gap_moving_m"], max_lat_accel_mps2=r["max_lat_accel_mps2"],
        n_harsh_brake_events=r["n_harsh_brake_events"], n_brake_sat_vehicle=r["n_brake_sat_vehicle"],
        n_brake_sat_virtual=r["n_brake_sat_virtual"], n_lane_changes=r["n_lane_changes"],
        lane_changes_per_veh=r["lane_changes_per_veh"], n_lc_yield=r["n_lc_yield"], n_yield_blocked=r["n_yield_blocked"],
        n_reroutes=r["n_reroutes"], n_destination_changes=r["n_destination_changes"],
        # review round 1 diagnostics: flicker band (synthesis: < 0.7 never switches, > 3.5 flicker),
        # hard brakes per red onset and the share of them clipped at -b_emerg (traffic-validation-03/04)
        switch_rate_ok=bool(lo <= r["switches_per_min"] <= hi),
        n_red_onsets=n_red_onsets,
        harsh_brakes_per_red_onset=round(r["n_harsh_brake_events"] / max(n_red_onsets, 1), 3),
        max_approach_red_s=r["max_approach_red_s"], n_forced_starvation=r["n_forced_starvation"],
        n_commit_revoked=r["n_commit_revoked"],
        # lane-level liveness KPIs (§11.8)
        max_head_wait_s=r["max_head_wait_s"],
        n_spillback_locks=r["n_spillback_locks"], spillback_lock_junction=r["spillback_lock_junction"],
        n_emergency_brakes=r["n_emergency_brakes"],
    )


# --------------------------------------------------------------------- 4. ring road
def run_ring(cfg: SimConfig, fig_dir: str, n_values=range(10, 151, 10), T_sim: float = 300.0, seed: int = 0) -> list:
    """Fundamental diagram on a 1 km ring with only IDM + ballistic update."""
    L = cfg.ring_length_m
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_values:
        params = [sample_params(rng, cfg) for _ in range(n)]
        s = np.linspace(0, L, n, endpoint=False)
        v = np.full(n, min(cfg.v_limit, (L / n - cfg.l_veh - cfg.s0) / cfg.T_mean))
        v = np.clip(v, 0.0, cfg.v_limit)
        n_steps = int(T_sim / cfg.dt)
        speeds = []
        a_prev = np.zeros(n)
        for k in range(n_steps):
            a = np.empty(n)
            for i in range(n):
                j = (i + 1) % n
                gap = (s[j] - s[i]) % L - cfg.l_veh
                # the simulator's own car-following law (ACC, §12.2), leader acceleration from the last step
                a[i] = max(acc_accel(v[i], params[i]["v0"], gap, v[j], a_prev[j], params[i]["a_max"], params[i]["b"],
                                     params[i]["T"], cfg.s0, cfg.acc_coolness, cfg.idm_delta), -cfg.b_emerg)
            a_prev = a.copy()
            for i in range(n):
                s[i], v[i] = ballistic_step(s[i], v[i], a[i], cfg.dt)
            if k * cfg.dt >= T_sim - 100:
                speeds.append(v.mean())
        density = n / L * 1000.0                       # veh/km
        flow = density * float(np.mean(speeds)) * 3.6  # veh/h
        rows.append(dict(n=n, density_veh_per_km=density, mean_speed_mps=float(np.mean(speeds)), flow_vph=flow))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["density_veh_per_km"] for r in rows], [r["flow_vph"] for r in rows], "o-")
    ax.axvline(1000 / (cfg.l_veh + cfg.s0), color="k", ls=":", label="jam density 1000/(l_veh+s0)")
    ax.set_xlabel("density [veh/km/lane]")
    ax.set_ylabel("flow [veh/h/lane]")
    ax.set_title("Ring road fundamental diagram (IDM)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "04_fundamental_diagram.png"), dpi=120)
    plt.close(fig)
    return rows


# --------------------------------------------------------------------- 5. paired teacher check
def sweep_capacity(cfg: SimConfig, sweep: list) -> dict:
    """Realised capacity per approach from the saturated end of the 1x1 fixed-time sweep
    (throughput of the most loaded point / 4 approaches) against the calibrated capacity
    (cfg.capacity_vph, measured by calibrate.measure_capacity) and the DECISIONS §7 formula
    value.  vc_levels_probe_vs_sweep restates each nominal level as lambda_base x flow per unit
    lambda / sweep capacity; because calibration used the same flow-per-lambda factor this is a
    consistency check between two 1x1 fixed-time experiments (probe vs sweep), NOT a measurement
    of the load on the 2x2 grid (see run_grid_levels for that)."""
    from .calibrate import busiest_approach_flow
    realised = max(r["throughput_vph"] for r in sweep) / 4.0
    _, k = busiest_approach_flow(cfg)
    return dict(capacity_formula_vph_per_approach=cfg.capacity_formula_vph,
                calibrated_capacity_vph_per_approach=cfg.capacity_vph,
                capacity_realisation=cfg.capacity_realisation,
                sweep_capacity_vph_per_approach=realised,
                sweep_vs_calibrated_ratio=realised / cfg.capacity_vph,
                sweep_vs_formula_ratio=realised / cfg.capacity_formula_vph,
                vc_levels_probe_vs_sweep={lvl: round(cfg.lambda_base_vph[lvl] * k / realised, 3) for lvl in cfg.vc_targets},
                probe_vs_sweep_within_5pct=all(abs(cfg.lambda_base_vph[lvl] * k / realised - x) <= 0.05 * x
                                            for lvl, x in cfg.vc_targets.items()))


def run_grid_levels(cfg: SimConfig, levels=("design", "high", "stress"), controllers=("fixed_time", "max_pressure"),
                    seeds=(41,)) -> dict:
    """Direct check on the 2x2 grid of what each nominal demand level does under a fixed-time and a
    max-pressure controller: mean delay, throughput and the number of vehicles in the network 900 s into
    the main period and at its end.  A level is flagged `queues_grow` when the network fills up
    (end count > 1.5 x the 900 s count and > 60 vehicles more), i.e. the level is over capacity for that
    controller.  This replaces the 1x1 probe-vs-sweep ratio as the evidence for the demand levels."""
    out = {}
    t_mid = cfg.warmup_s + 900.0
    t_end = cfg.warmup_s + cfg.main_s - 1.0
    for level in levels:
        for ctrl in controllers:
            rows = []
            for seed in seeds:
                sc = Scenario(demand_level=level, lambda_base_vph=cfg.lambda_base_vph[level], controller=ctrl)
                w = World(cfg, sc, seed=seed).run()
                r = w.episode_row()
                n_at = {}
                for rec in w.junction_1hz_rows:
                    if rec["t"] in (t_mid, t_end):
                        n_at[rec["t"]] = n_at.get(rec["t"], 0) + rec["n_total"]
                n_mid, n_end = n_at.get(t_mid, 0), n_at.get(t_end, 0)
                rows.append(dict(delay=r["mean_total_delay_s"], thr=r["throughput_vph"], n_mid=n_mid, n_end=n_end,
                                 grow=bool(n_end > 1.5 * n_mid and n_end - n_mid > 60), gridlock=bool(r["gridlock"])))
            out[f"{level}/{ctrl}"] = dict(
                n_seeds=len(seeds), mean_delay_s=float(np.mean([x["delay"] for x in rows])),
                throughput_vph=float(np.mean([x["thr"] for x in rows])),
                vehicles_in_network_t900=float(np.mean([x["n_mid"] for x in rows])),
                vehicles_in_network_end=float(np.mean([x["n_end"] for x in rows])),
                queues_grow=bool(any(x["grow"] for x in rows)), gridlock=bool(any(x["gridlock"] for x in rows)))
    return out


def run_paired_mp_vs_ft(cfg: SimConfig, levels=("design", "high"), seeds=(31, 32, 33)) -> dict:
    """max_pressure vs fixed_time on paired seeds (common random numbers).  The teacher is
    specified exactly (DECISIONS §6); this reports where it beats fixed time and where it
    does not, with t-based 95% intervals on the paired delay difference (traffic-validation-01)."""
    out = {}
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 8: 2.365, 10: 2.262}.get(len(seeds), 2.0)
    for level in levels:
        rows = {"max_pressure": [], "fixed_time": []}
        for seed in seeds:
            for ctrl in rows:
                sc = Scenario(demand_level=level, lambda_base_vph=cfg.lambda_base_vph[level], controller=ctrl)
                r = World(cfg, sc, seed=seed).run().episode_row()
                rows[ctrl].append(dict(delay=r["mean_total_delay_s"], thr=r["throughput_vph"],
                                       spm=r["switches_per_min"], gridlock=r["gridlock"]))
        d_delay = np.array([a["delay"] - b["delay"] for a, b in zip(rows["max_pressure"], rows["fixed_time"])])
        d_thr = np.array([a["thr"] - b["thr"] for a, b in zip(rows["max_pressure"], rows["fixed_time"])])
        half = tcrit * d_delay.std(ddof=1) / math.sqrt(len(d_delay)) if len(d_delay) > 1 else math.nan
        out[level] = dict(
            n_seeds=len(seeds),
            mp_mean_delay_s=float(np.mean([r["delay"] for r in rows["max_pressure"]])),
            ft_mean_delay_s=float(np.mean([r["delay"] for r in rows["fixed_time"]])),
            mp_throughput_vph=float(np.mean([r["thr"] for r in rows["max_pressure"]])),
            ft_throughput_vph=float(np.mean([r["thr"] for r in rows["fixed_time"]])),
            mp_switches_per_min=float(np.mean([r["spm"] for r in rows["max_pressure"]])),
            paired_delay_diff_mp_minus_ft_s=float(d_delay.mean()), paired_delay_diff_ci95_halfwidth=float(half),
            paired_throughput_diff_vph=float(d_thr.mean()),
            # the primary metric is delay: MP "beats" FT when the paired-difference CI lies below 0
            mp_beats_ft=bool(d_delay.mean() + half < 0),
        )
    return out


# --------------------------------------------------------------------- main
def main(argv=None) -> dict:
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    os.makedirs(FIG_DIR, exist_ok=True)
    cfg = SimConfig()
    t0 = time.perf_counter()
    print("== 1. queue discharge")
    disc = run_discharge(cfg, FIG_DIR)
    print(f"   saturation headway {disc['h_sat_s']:.3f} s -> {disc['sat_flow_vphpl']:.0f} veh/h/lane "
          f"(expected {cfg.sat_flow_expected_range[0]:.0f}-{cfg.sat_flow_expected_range[1]:.0f}: "
          f"{'OK' if disc['in_expected_range'] else 'OUTSIDE'}), start-up lost time {disc['startup_lost_time_s']:.2f} s")
    cfg.startup_lost_time_s = round(disc["startup_lost_time_s"], 3)
    calib = calibrate_demand(cfg, disc["sat_flow_vphpl"])
    print(f"   capacity per approach: formula (sat_flow x lanes x 18/96) {calib['capacity_formula_vph']:.0f} veh/h, measured in a saturated "
          f"1x1 fixed-time probe (x = {cfg.capacity_probe_x}, {cfg.capacity_probe_main_s:.0f} s) {calib['capacity_vph']:.0f} veh/h, "
          f"realisation ratio {calib['capacity_realisation']:.3f}")
    print(f"   busiest internal approach {calib['busiest_edge']} carries {calib['flow_per_unit_lambda']:.2f} x lambda_base; "
          f"lambda_base per level (from the measured capacity): {calib['lambda_base_vph']}")

    print("== 2. single junction fixed-time demand sweep (v/c against the measured capacity)")
    sweep = run_demand_sweep(cfg, cfg.capacity_vph, FIG_DIR, vcs=(0.4, 0.7, 0.95, 1.1) if quick else (0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.1))
    print("   v/c   lambda   mean_delay  p90   throughput  max_q  webster_d1  gridlock  runtime")
    for r in sweep:
        print(f"   {r['vc']:<5} {r['lambda_base_vph']:7.1f} {r['mean_delay_s']:9.1f} {r['p90_delay_s']:6.1f} "
              f"{r['throughput_vph']:9.0f} {r['max_queue']:6d} {r['webster_d1_s']:9.1f} {str(r['gridlock']):>8} {r['runtime_s']:6.1f}s")

    cap = sweep_capacity(cfg, sweep)
    print(f"   sweep saturating throughput {cap['sweep_capacity_vph_per_approach']:.0f} veh/h/approach = {cap['sweep_vs_calibrated_ratio']:.3f} x calibrated "
          f"capacity ({cap['calibrated_capacity_vph_per_approach']:.0f}), {cap['sweep_vs_formula_ratio']:.3f} x formula ({cap['capacity_formula_vph_per_approach']:.0f}); "
          f"probe-vs-sweep v/c per nominal level {cap['vc_levels_probe_vs_sweep']} (within 5%: {cap['probe_vs_sweep_within_5pct']})")

    print("== 3. 2x2 grid, design demand, max_pressure, one episode")
    design = run_design_episode(cfg, FIG_DIR)
    print(f"   mean delay {design['mean_delay_s']:.1f} s/veh (plausible 30-90: {design['delay_plausible']}), p90 {design['p90_delay_s']:.1f} s, "
          f"throughput {design['throughput_vph']:.0f} veh/h, {design['n_spawned']} vehicles, runtime {design['runtime_s']:.1f} s")
    print(f"   invariants {design['invariants']}")
    print(f"   switches/min {design['switches_per_min']:.2f} (band {cfg.switch_rate_band}: {design['switch_rate_ok']}), red commits {design['n_red_commits']}, "
          f"harsh braking runs {design['n_harsh_brake_events']} = {design['harsh_brakes_per_red_onset']:.2f} per red onset, "
          f"brake saturation vehicle/virtual {design['n_brake_sat_vehicle']}/{design['n_brake_sat_virtual']}, "
          f"box blocked {design['n_box_blocked']}, all-red extension mean {design['allred_extension_mean_s']:.3f} s")
    print(f"   safety: collisions {design['n_collisions']} (must be 0: {design['collisions_ok']}), conflicts {design['n_conflicts']}, "
          f"near misses {design['n_near_misses']} ({design['near_miss_per_1000_veh']} per 1000 veh), min TTC {design['min_ttc_s']} s, "
          f"min PET {design['min_pet_s']} s ({design['n_pet_serious']} < 1 s), min moving gap {design['min_gap_moving_m']} m, "
          f"max lateral accel {design['max_lat_accel_mps2']} m/s^2")
    print(f"   lane changes {design['n_lane_changes']} ({design['lane_changes_per_veh']:.2f}/veh), moved aside for ambulances "
          f"{design['n_lc_yield']} (no room {design['n_yield_blocked']}), reroutes {design['n_reroutes']} "
          f"(destination changes {design['n_destination_changes']})")
    print(f"   max approach red {design['max_approach_red_s']:.0f} s (starvation guard {cfg.starvation_s:.0f} s, forced {design['n_forced_starvation']}x), "
          f"commits revoked {design['n_commit_revoked']}")
    print(f"   liveness: max lane-head standing time {design['max_head_wait_s']:.0f} s, spillback locks {design['n_spillback_locks']} "
          f"(first at junction {design['spillback_lock_junction']})")

    ring = []
    if not quick:
        print("== 4. ring road fundamental diagram (optional)")
        ring = run_ring(cfg, FIG_DIR)
        peak = max(ring, key=lambda r: r["flow_vph"])
        print(f"   capacity {peak['flow_vph']:.0f} veh/h/lane at {peak['density_veh_per_km']:.0f} veh/km; "
              f"jam density intercept 1000/(l_veh+s0) = {1000 / (cfg.l_veh + cfg.s0):.0f} veh/km")

    print("== 5. max_pressure vs fixed_time on paired seeds (design, high)")
    paired = run_paired_mp_vs_ft(cfg, seeds=(31, 32) if quick else (31, 32, 33))
    for level, p in paired.items():
        print(f"   {level:6s} MP delay {p['mp_mean_delay_s']:6.1f} s vs FT {p['ft_mean_delay_s']:6.1f} s (paired diff {p['paired_delay_diff_mp_minus_ft_s']:+.1f} "
              f"+/- {p['paired_delay_diff_ci95_halfwidth']:.1f}), throughput {p['mp_throughput_vph']:.0f} vs {p['ft_throughput_vph']:.0f} veh/h, "
              f"MP switches/min {p['mp_switches_per_min']:.2f}, MP beats FT: {p['mp_beats_ft']}")

    print("== 6. 2x2 grid: what the nominal demand levels do under fixed_time and max_pressure")
    grid = run_grid_levels(cfg, levels=("design", "high") if quick else ("design", "high", "stress"))
    for key, g in grid.items():
        print(f"   {key:24s} delay {g['mean_delay_s']:6.1f} s, throughput {g['throughput_vph']:6.0f} veh/h, vehicles in network "
              f"{g['vehicles_in_network_t900']:5.0f} at 900 s -> {g['vehicles_in_network_end']:5.0f} at end of main period, "
              f"queues grow: {g['queues_grow']}, gridlock: {g['gridlock']}")

    summary = dict(generator_version=GENERATOR_VERSION, config_hash=cfg.config_hash(),
                   created_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   discharge=disc, calibration=calib, demand_sweep=sweep, demand_sweep_capacity=cap, design_episode=design,
                   ring=ring, paired_mp_vs_ft=paired, grid_levels_2x2=grid, total_runtime_s=time.perf_counter() - t0)
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"figures in {FIG_DIR}/, summary in reports/validation_summary.json, total {summary['total_runtime_s']:.0f} s")
    return summary


if __name__ == "__main__":
    main()
