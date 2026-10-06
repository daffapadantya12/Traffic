"""Calibrate the Fatmawati demand to the observed congestion (DECISIONS §13.3).

Targets come from the published study of this junction (Sumabrata, Haqueline & Mauramdha 2026: weekday peak
hour, CCTV counts, Dishub static signal plan, VISSIM model validated with GEH <= 3.12).  The twin runs the
observed movements under the Dishub fixed plan at a common volume scale k (share of the observed counts) for a
grid of k.  For each approach, the scale at which the hour's maximum queue (approach camera, VISSIM queue
definition: below 5 km/h, gaps up to 20 m) equals the VISSIM maximum queue is interpolated.  A verification run
at the per-approach scales then reports the queues, the stopped delays (validation against the PKJI delays) and
how far the volumes stay below the real counts.  The result is data/fatmawati/calibration.json, which the demand
model reads.

  python3 -m traffic_twin.fatmawati.calibrate --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .demand import CALIBRATION, MIX, MOVEMENTS, routes_xml, vtypes_xml, DATA
from .sensors import LEGS

TARGET_MAX_QUEUE_M = {"N": 289.5, "S": 166.6, "E": 239.0, "W": 155.5}       # VISSIM maximum queue
PKJI_DELAY_S = {"N": 79.8, "S": 106.3, "E": 78.7, "W": 78.3}                  # PKJI 2023 delay per approach
PKJI_DS = {"N": 0.86, "S": 0.79, "E": 0.85, "W": 0.65}                        # degree of saturation
VISSIM_JUNCTION_DELAY_S = 67.4
OBSERVED_VEH_H = {g: sum(v for (a, _), v in MOVEMENTS.items() if a == g) for g in LEGS}
GRID = (0.15, 0.25, 0.35, 0.5, 0.7)
SEEDS = (11, 12)
WARM_S, MAIN_S = 900.0, 3600.0


def run(scale: dict, seed: int, tmp: str, vtypes: str | None = None) -> dict:
    """One peak hour under the Dishub plan at the given approach scales: what the sensors measured."""
    from .sumo_world import FatmawatiWorld
    path = os.path.join(tmp, f"cal_{seed}_{'_'.join(f'{scale[g]:.3f}' for g in LEGS)}.rou.xml")
    routes_xml(path, "design", seed=seed, duration_s=WARM_S + MAIN_S, peak=True, scale=scale)
    t0 = time.perf_counter()
    from .sumo_world import VTYPES
    w = FatmawatiWorld(path, start_hour=7.0, seed=seed, controller="fixed", sensors=True, record_from=WARM_S,
                       vtypes=vtypes or VTYPES)
    q_max = {g: 0.0 for g in LEGS}
    last = -1
    while w.t < WARM_S + MAIN_S - 1e-9:
        w.step()
        if w.t >= WARM_S and int(w.t) != last:
            last = int(w.t)
            for g in LEGS:
                q_max[g] = max(q_max[g], w.sensors.cam_now.get(f"QC_{g}", {}).get("q_m", 0.0))
    S = w.sensors
    S.finish(w.t)
    stop = {g: sum(r["n_total"] for r in S.rows_loops if r["kind"] == "stopline_loop" and r["leg"] == g) for g in LEGS}
    entry = {g: sum(r["n_total"] for r in S.rows_loops if r["kind"] == "entry_gate" and r["leg"] == g) for g in LEGS}
    standing = {g: sum(q for _, q in S.queue_series[g]) for g in LEGS}
    out = dict(scale=scale, seed=seed, max_queue_m={g: round(q_max[g], 1) for g in LEGS},
               stopped_delay_s={g: round(standing[g] / stop[g], 1) if stop[g] else None for g in LEGS},
               entry_veh_h={g: round(entry[g] * 3600.0 / MAIN_S) for g in LEGS},
               stopline_veh_h={g: round(stop[g] * 3600.0 / MAIN_S) for g in LEGS},
               waiting_to_enter=len(w.ts.simulation.getPendingVehicles()), lane_collisions=w.n_lane_collisions,
               junction_overlaps=w.n_junction_overlaps, teleports=w.n_teleports,
               lane_collisions_study=w.n_lane_collisions_study, junction_overlaps_study=w.n_junction_overlaps_study,
               veh_km=round(w.veh_km, 1), veh_km_study=round(w.veh_km_study, 1),
               speed_x=round((WARM_S + MAIN_S) / (time.perf_counter() - t0), 1))
    w.close()
    return out


def interpolate(grid_runs: list) -> dict:
    """Per approach: the scale where the mean (over seeds) maximum queue crosses the target."""
    scale = {}
    for g in LEGS:
        ks = sorted({r["scale"][g] for r in grid_runs})
        q = [float(np.mean([r["max_queue_m"][g] for r in grid_runs if r["scale"][g] == k])) for k in ks]
        target = TARGET_MAX_QUEUE_M[g]
        k_star = ks[-1]
        for (k0, q0), (k1, q1) in zip(zip(ks, q), zip(ks[1:], q[1:])):
            if q0 <= target <= q1 and q1 > q0:
                k_star = k0 + (k1 - k0) * (target - q0) / (q1 - q0)
                break
        else:
            if q[0] >= target:
                k_star = ks[0]
        scale[g] = round(k_star, 3)
    return scale


def _mean(runs: list, key: str) -> dict:
    return {g: float(np.mean([r[key][g] for r in runs if r[key][g] is not None])) for g in LEGS}


def calibrate(workers: int = 8, rounds: int = 3, tol: float = 0.10, damp: float = 0.7, start: dict | None = None,
              vtypes: str | None = None, grid: bool = False) -> dict:
    """Optional grid + interpolation for a start, then damped fixed-point rounds s <- s (target / achieved)^damp until
    every approach's maximum queue is within tol of its target.  An approach whose entry gates carry less than 90%
    of the demand sent to it is capacity-limited: its scale is set to the flow it served and it is flagged."""
    vtypes_xml(os.path.join(DATA, "demand", "vtypes.add.xml"))
    history, grid_runs = [], []
    with tempfile.TemporaryDirectory(prefix="fatmawati_cal_") as tmp:
        if grid or start is None:
            jobs = [({g: k for g in LEGS}, seed) for k in GRID for seed in SEEDS]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                grid_runs = list(pool.map(run, [j[0] for j in jobs], [j[1] for j in jobs], [tmp] * len(jobs), [vtypes] * len(jobs)))
            start = interpolate(grid_runs)
        scale = dict(start)
        limited = {g: False for g in LEGS}
        for k in range(rounds + 1):
            with ProcessPoolExecutor(max_workers=workers) as pool:
                runs = list(pool.map(run, [scale] * len(SEEDS), list(SEEDS), [tmp] * len(SEEDS), [vtypes] * len(SEEDS)))
            q, entry = _mean(runs, "max_queue_m"), _mean(runs, "entry_veh_h")
            served = {g: entry[g] / (scale[g] * OBSERVED_VEH_H[g]) for g in LEGS}
            err = {g: q[g] / TARGET_MAX_QUEUE_M[g] - 1.0 for g in LEGS}
            history.append(dict(round=k, scale=dict(scale), max_queue_m={g: round(q[g], 1) for g in LEGS},
                                rel_error={g: round(err[g], 3) for g in LEGS}, served_share={g: round(served[g], 3) for g in LEGS},
                                runs=runs))
            done = all(abs(err[g]) <= tol or limited[g] for g in LEGS)
            if done or k == rounds:
                break
            for g in LEGS:
                if limited[g] or abs(err[g]) <= tol:
                    continue
                if served[g] < 0.9 and err[g] < 0:                # demand not getting in: a capacity limit, not a scale
                    limited[g] = True
                    scale[g] = round(entry[g] / OBSERVED_VEH_H[g], 3)
                else:
                    scale[g] = round(min(1.0, max(0.05, scale[g] * (TARGET_MAX_QUEUE_M[g] / max(q[g], 1.0)) ** damp)), 3)
    last = history[-1]["runs"]
    mean = lambda key, g: round(float(np.mean([r[key][g] for r in last if r[key][g] is not None])), 1)
    result = dict(
        scale=history[-1]["scale"], capacity_limited=limited,
        converged={g: abs(history[-1]["rel_error"][g]) <= tol for g in LEGS},
        method=f"Dishub fixed plan, observed movements x per-approach scale; damped rounds s <- s (target/achieved)^{damp} until the "
               f"peak-hour maximum queue (camera, VISSIM queue definition) is within {tol:.0%} of the VISSIM maximum queue; "
               "approaches whose entry gates carry < 90% of the demand sent to them are capacity-limited; mean of seeds " + str(list(SEEDS)),
        sources=["Sumabrata, Haqueline & Mauramdha (2026), Microscopic Simulation Analysis of The Fatmawati Intersection "
                 "in Jakarta, Proc. QiR 2025, AER 303, 31-44, doi:10.2991/978-94-6239-717-0_4 (CC BY-NC 4.0)",
                 "Yusuf & Isradi (2025), ETJ 10(3), 4052-4060, doi:10.47191/etj/v10i03.15 (CC BY 4.0): vehicle mix"],
        vehicle_mix=MIX, target_max_queue_m=TARGET_MAX_QUEUE_M,
        verification=dict(max_queue_m={g: mean("max_queue_m", g) for g in LEGS},
                          stopped_delay_s={g: mean("stopped_delay_s", g) for g in LEGS}, pkji_delay_s=PKJI_DELAY_S,
                          entry_veh_h={g: mean("entry_veh_h", g) for g in LEGS}, observed_veh_h=OBSERVED_VEH_H,
                          volume_share_of_observed={g: round(mean("entry_veh_h", g) / OBSERVED_VEH_H[g], 3) for g in LEGS},
                          lane_collisions=sum(r["lane_collisions"] for r in last),
                          lane_collisions_study=sum(r["lane_collisions_study"] for r in last),
                          junction_overlaps=sum(r["junction_overlaps"] for r in last),
                          teleports=sum(r["teleports"] for r in last), veh_km=round(sum(r["veh_km"] for r in last), 1),
                          speed_x=round(float(np.mean([r["speed_x"] for r in last])), 1)),
        rounds=[{k: v for k, v in h.items() if k != "runs"} for h in history], grid_runs=grid_runs, verification_runs=last)
    with open(CALIBRATION, "w") as f:
        json.dump(result, f, indent=1)
    return result


def verify_and_save(scale: dict, seeds=(11, 12, 13, 14), capacity_limited: dict | None = None, note: str = "",
                    workers: int = 4) -> dict:
    """Final check of hand-chosen scales over several seeds (the hourly maximum queue is noisy near capacity),
    written to calibration.json together with the earlier rounds."""
    vtypes_xml(os.path.join(DATA, "demand", "vtypes.add.xml"))
    try:
        with open(CALIBRATION) as f:
            previous = json.load(f)
    except (OSError, ValueError):
        previous = {}
    with tempfile.TemporaryDirectory(prefix="fatmawati_cal_") as tmp:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(run, [scale] * len(seeds), list(seeds), [tmp] * len(seeds)))
    stat = lambda key, g: [r[key][g] for r in runs if r[key][g] is not None]
    q = {g: stat("max_queue_m", g) for g in LEGS}
    limited = capacity_limited or {g: False for g in LEGS}
    rel = {g: round(float(np.mean(q[g])) / TARGET_MAX_QUEUE_M[g] - 1.0, 3) for g in LEGS}
    result = dict(
        scale=scale, capacity_limited=limited, converged={g: abs(rel[g]) <= 0.10 for g in LEGS}, rel_error=rel,
        method="Dishub fixed plan, observed movements x per-approach scale.  Damped rounds s <- s (target/achieved)^0.7 "
               "overshot: near capacity an approach's hourly maximum queue tips from a short queue to spill-back past the "
               "300 m camera, and it varies a lot between seeds.  The scales were then bracketed and checked over "
               f"{len(seeds)} seeds; an approach whose entry gates carry < 90% of the demand sent to it is capacity-limited.  " + note,
        sources=previous.get("sources"), vehicle_mix=MIX, target_max_queue_m=TARGET_MAX_QUEUE_M,
        verification=dict(seeds=list(seeds),
                          max_queue_m={g: round(float(np.mean(q[g])), 1) for g in LEGS},
                          max_queue_sd_m={g: round(float(np.std(q[g], ddof=1)), 1) for g in LEGS},
                          stopped_delay_s={g: round(float(np.mean(stat("stopped_delay_s", g))), 1) for g in LEGS},
                          pkji_delay_s=PKJI_DELAY_S,
                          entry_veh_h={g: round(float(np.mean(stat("entry_veh_h", g))), 1) for g in LEGS},
                          observed_veh_h=OBSERVED_VEH_H,
                          volume_share_of_observed={g: round(float(np.mean(stat("entry_veh_h", g))) / OBSERVED_VEH_H[g], 3) for g in LEGS},
                          lane_collisions=sum(r["lane_collisions"] for r in runs),
                          lane_collisions_study=sum(r["lane_collisions_study"] for r in runs),
                          junction_overlaps=sum(r["junction_overlaps"] for r in runs),
                          junction_overlaps_study=sum(r["junction_overlaps_study"] for r in runs),
                          teleports=sum(r["teleports"] for r in runs), veh_km=round(sum(r["veh_km"] for r in runs), 1),
                          speed_x=round(float(np.mean([r["speed_x"] for r in runs])), 1)),
        rounds=previous.get("rounds", []), grid_runs=previous.get("grid_runs", []), verification_runs=runs)
    with open(CALIBRATION, "w") as f:
        json.dump(result, f, indent=1)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start", default=None, help='JSON scale to start the rounds from, e.g. {"N":0.24,"E":0.23,"S":0.26,"W":0.4}')
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--grid", action="store_true", help="run the scale grid first (default when no --start)")
    args = ap.parse_args(argv)
    r = calibrate(args.workers, rounds=args.rounds, start=json.loads(args.start) if args.start else None, grid=args.grid)
    print(json.dumps(dict(scale=r["scale"], capacity_limited=r["capacity_limited"], converged=r["converged"],
                     verification=r["verification"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
