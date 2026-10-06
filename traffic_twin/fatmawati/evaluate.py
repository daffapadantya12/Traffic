"""Evaluation grid: ambulance passages through Perempatan Fatmawati (DECISIONS §13.10).

Every method faces the same ambulances at the same times: each episode releases one ambulance every 10 min at
a fixed point on its approach (at least 400 m before the stop line, upstream of the longest queue), cycling
through the three hospital movements (N through; E free-left; W detour = W through, U-turn east of the
junction, E free-left) and through 8 equal slices of the 231 s Dishub cycle.  A paired run with the same
seed, demand and controller but no ambulance gives the reference queues; an empty-road run per movement
(the ambulance alone, junction signal off) gives the free time T_empty.

Per passage (passages.csv), from the sensors only:
  junction_time_s = t_exit_gate - t_release;  delay_s = junction_time_s - T_empty
  travel_time_s   = arrival at RSUP Fatmawati - t_release (the patient outcome)
  recovery_s      = first time after the exit gate from which, on all four approaches, the 231 s rolling mean of
                    camera standing vehicles with and without the ambulance differ by at most max(10%, 1 vehicle)
                    for a full 231 s (censored at the end of the episode; recovery_overlaps_next_release flags the
                    ones whose 231 s check ran past the next release)
  extra_wait_s_<leg> = sum over [t_release, t_release + 600 s] of (standing with - standing without) per second
                    / that leg's stop-line count in the run without the ambulance over the same window
  detect_signal   = state of the ambulance's approach when the priority-request detector first saw it

Per run (episodes_eval.csv): served_<leg> (stop-line count over [warm-up end, end_s)), stopped_delay_s_<leg> (standing
vehicle-seconds in the approach camera / served_<leg>), longest_red_s_<leg> (longest spell without green for the
approach after the warm-up; TB Simatupang keeps green through S1>S2), end_s.

  python3 -m traffic_twin.fatmawati.evaluate --methods fixed,priority_mp --levels design --seeds 1,2 --workers 8
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
import pandas as pd

from .build_net import ALLRED_S, APPROACHES, DISHUB_GREEN_S, STAGE_LINKS, STAGES, TLS_ID, YELLOW_S
from .demand import DATA, NET, routes_xml, vtypes_xml
from .sensors import LEGS, chain_upstream

EV_MOVES = ("N", "E", "W")               # release legs; every ambulance drives to RSUP Fatmawati
HOSPITAL_LEG = "S"                       # the exit gate on the hospital side ends a passage
EXTRA_S = 1800.0                         # keep running (unrecorded) until released ambulances have arrived
RELEASE_MIN_M = 400.0
SLICES = 8
CYCLE_S = sum(DISHUB_GREEN_S.values()) + len(STAGES) * (YELLOW_S + ALLRED_S)   # 231 s
RELEASE_EVERY_S = 600.0
WARM_S = 900.0
TAIL_S = 600.0
METHODS = ("random", "still", "fixed", "longest_queue", "max_pressure", "priority_mp", "preempt_naive")
REFERENCE_OF = {"priority_mp": "max_pressure", "preempt_naive": "fixed"}     # identical without ambulances


def release_points(net=None) -> dict:
    """leg -> (edge, position on it, distance to the stop line): RELEASE_MIN_M up the approach where the road
    allows, else just past the junction at the start of the farthest same-street edge."""
    import sumolib
    net = net or sumolib.net.readNet(NET, withInternal=False)
    out = {}
    for leg in EV_MOVES:
        chain = chain_upstream(net, APPROACHES[leg], RELEASE_MIN_M + 1000.0)
        for e, off in chain:
            if off + e.getLength() >= RELEASE_MIN_M:
                out[leg] = (e.getID(), round(e.getLength() - (RELEASE_MIN_M - off), 1), RELEASE_MIN_M)
                break
        else:
            e, off = chain[-1]
            out[leg] = (e.getID(), 1.0, round(off + e.getLength() - 1.0, 1))
    return out


def schedule(n: int | None = None) -> list:
    """[(t_release, leg, slice)]: one release every RELEASE_EVERY_S, slice k at offset k x CYCLE_S / SLICES of the
    fixed-time cycle (every method gets the same times); n releases (default all movements x slices)."""
    out = []
    for j in range(n if n is not None else len(EV_MOVES) * SLICES):
        leg, k = EV_MOVES[j % len(EV_MOVES)], j // len(EV_MOVES)
        base = WARM_S + RELEASE_EVERY_S * j
        t = base + (k * CYCLE_S / SLICES - base) % CYCLE_S
        out.append((round(t * 2) / 2, leg, k))
    return out


# stages that give an approach green (TB Simatupang: its through lanes, green in S1 and S2)
APPROACH_STAGES = {leg: {st for st, pairs in STAGE_LINKS.items() if any(f == APPROACHES[leg] for f, _ in pairs)} for leg in APPROACHES}


def longest_red(signal_log: list, leg: str, t0: float, t1: float) -> float:
    """Longest continuous spell in [t0, t1) in which the approach shows no green.  A change between two stages that
    both serve it (S1>S2 for TB Simatupang) keeps its green."""
    served = APPROACH_STAGES[leg]

    def green(row) -> bool:
        if row["state"] == "green":
            return row["stage"] in served
        a, _, b = row["stage"].partition(">")
        return a in served and b in served
    spells, start = [], None
    state_at_t0 = next((r for r in reversed(signal_log) if r["t"] <= t0), None)
    if state_at_t0 is not None and not green(state_at_t0):
        start = t0
    for r in signal_log:
        if r["t"] <= t0 or r["t"] >= t1:
            continue
        if green(r):
            if start is not None:
                spells.append(r["t"] - start)
                start = None
        elif start is None:
            start = r["t"]
    if start is not None:
        spells.append(t1 - start)
    return round(max(spells), 1) if spells else 0.0


def _approach_state(ts, link_idx: list) -> str:
    st = ts.trafficlight.getRedYellowGreenState(TLS_ID)
    s = {st[i] for i in link_idx}
    return "green" if s & {"G", "g"} else "amber" if s & {"y", "Y"} else "red"


def run_episode(level: str, seed: int, method: str, with_ev: bool, tmp: str, n_releases: int | None = None,
                vtypes: str | None = None) -> dict:
    """One episode; with_ev=False is the paired reference run (same demand, no ambulances)."""
    from .sumo_world import FatmawatiWorld
    sched = schedule(n_releases)
    end = sched[-1][0] + TAIL_S
    path = os.path.join(tmp, f"eval_{level}_s{seed}.rou.xml")
    if not os.path.exists(path):
        routes_xml(path, level, seed=seed, start_hour=7.0, duration_s=end, peak=True, ev_per_hour=0.0)
    wall0 = time.perf_counter()
    from .sumo_world import VTYPES
    w = FatmawatiWorld(path, start_hour=7.0, seed=seed, controller=method, sensors=True, record_from=WARM_S,
                       vtypes=vtypes or VTYPES)
    ts = w.ts
    links = {}
    for i, lks in enumerate(ts.trafficlight.getControlledLinks(TLS_ID)):
        for a, _, _ in lks:
            for leg, e in APPROACHES.items():
                if a.startswith(e + "_"):
                    links.setdefault(leg, []).append(i)
    rel = release_points()
    hosp = None
    pending = list(sched) if with_ev else []
    meta = {}
    def travelling() -> bool:
        return any(vid in w.sensors.ev for vid in meta)

    while w.t < end - 1e-9 or (with_ev and travelling() and w.t < end + EXTRA_S):
        while pending and w.t >= pending[0][0] - 1e-9:
            t_rel, leg, k = pending.pop(0)
            if hosp is None:
                from .demand import hospital_edge
                import sumolib
                hosp = hospital_edge(sumolib.net.readNet(NET, withInternal=False))
            edge, pos, dist = rel[leg]
            vid = f"ev_{len(meta):02d}"
            route = ts.simulation.findRoute(edge, hosp[0].getID(), vType="ambulance").edges
            ts.route.add(f"r_{vid}", route)
            ts.vehicle.add(vid, f"r_{vid}", typeID="ambulance", depart="now", departPos=str(pos), departLane="best",
                           departSpeed="max", arrivalPos=str(hosp[1]))
            meta[vid] = dict(passage=len(meta), leg=leg, slice=k, t_release=t_rel, release_edge=edge, release_dist_m=dist)
        w.step()
        for vid, m in meta.items():                             # signal state at the first detection
            r = w.sensors.ev.get(vid)
            if r is not None and "t_detect" in r and "detect_signal" not in m:
                m["detect_signal"] = "all-red" if w.phase_names[w._phase].endswith("all-red") else _approach_state(ts, links[r["detect_leg"]])
                m["detect_stage"] = w.green_stage
    S = w.sensors
    S.finish(w.t)
    q = {g: dict(S.queue_series[g]) for g in LEGS}               # t -> standing vehicles (1 Hz)
    stop = {g: [(r["t_start"], r["n_total"]) for r in S.rows_loops if r["kind"] == "stopline_loop" and r["leg"] == g]
            for g in LEGS}
    trips = {r["vid"]: r for r in S.ev_done}
    passages = []
    for vid, m in meta.items():
        r = trips.get(vid)
        r = r or {}
        hosp_exit = [t for leg, t in r.get("exits", []) if leg == HOSPITAL_LEG]
        passages.append(dict(**m, **{k: r.get(k) for k in ("t_detect", "detect_leg", "queue_at_detect", "t_stopline",
                                                           "stopline_leg", "t_arrive")},
                             t_entry_gate=next((t for leg, t in r.get("entries", []) if leg == m["leg"]), None),
                             t_exit_gate=hosp_exit[-1] if hosp_exit else None,
                             n_gate_passes=len(r.get("entries", [])) + len(r.get("exits", []))))
    run = dict(level=level, seed=seed, method=method, with_ev=with_ev, end_s=end)     # sensor outcomes of the run
    for g in LEGS:
        served = sum(n for t0, n in stop[g] if WARM_S - 1e-9 <= t0 < end)
        standing = sum(v for t, v in q[g].items() if WARM_S <= t < end)
        run[f"served_{g}"] = served
        run[f"stopped_delay_s_{g}"] = round(standing / served, 1) if served else None
        run[f"longest_red_s_{g}"] = longest_red(w.signal_log, g, WARM_S, end)
    out = dict(level=level, seed=seed, method=method, with_ev=with_ev, q=q, stop=stop, passages=passages, run=run,
               qc=dict(level=level, seed=seed, method=method, with_ev=with_ev, lane_collisions=w.n_lane_collisions,
                       junction_overlaps=w.n_junction_overlaps, lane_collisions_study=w.n_lane_collisions_study,
                       junction_overlaps_study=w.n_junction_overlaps_study, teleports=w.n_teleports,
                       veh_km=round(w.veh_km, 1), veh_km_study=round(w.veh_km_study, 1), entered=w.n_departed,
                       waiting_to_enter=len(ts.simulation.getPendingVehicles()),
                       wall_s=round(time.perf_counter() - wall0, 1), speed_x=round(end / (time.perf_counter() - wall0), 1)))
    w.close()
    return out


def empty_road_times() -> dict:
    """leg -> seconds from release to the S exit gate for an ambulance alone with the junction signal off."""
    import libsumo as ts
    import sumolib
    from .demand import hospital_edge
    from .sensors import SensorBank, write_additional
    from .sumo_world import VTYPES, sensor_layout
    out = {}
    net = sumolib.net.readNet(NET, withInternal=False)
    hosp, pos_h = hospital_edge(net)
    rel = release_points(net)
    lay = sensor_layout()
    for leg in EV_MOVES:
        with tempfile.NamedTemporaryFile("w", suffix=".rou.xml", delete=False) as f:
            f.write("<routes/>\n")
        ts.start(["sumo", "-n", NET, "-r", f.name, "-a", VTYPES + "," + write_additional(lay), "--step-length", "0.5",
                  "--lateral-resolution", "0.8", "--no-step-log", "true", "--no-warnings", "true"])
        ts.trafficlight.setProgram(TLS_ID, "off")
        bank = SensorBank(ts, lay)
        edge, p, _ = rel[leg]
        ts.route.add("r", ts.simulation.findRoute(edge, hosp.getID(), vType="ambulance").edges)
        ts.vehicle.add("ev_0", "r", typeID="ambulance", depart="now", departPos=str(p), departLane="best",
                       departSpeed="max", arrivalPos=str(pos_h))
        bank.ev_departed("ev_0", 0.0)
        t = 0.0
        while t < 1800.0 and not any(g == HOSPITAL_LEG for g, _ in bank.ev.get("ev_0", {}).get("exits", [])):
            ts.simulationStep()
            t = ts.simulation.getTime()
            bank.step(t)
        out[leg] = round(max(tt for g, tt in bank.ev["ev_0"]["exits"] if g == HOSPITAL_LEG), 1)
        ts.close()
        os.unlink(f.name)
    return out


def _series(q: dict, end: float) -> np.ndarray:
    """Standing vehicles per whole second 0..end (1 Hz camera samples; 0 where nothing was recorded)."""
    arr = np.zeros(int(end) + 2)
    for t, v in q.items():
        if 0 <= int(t) < len(arr):
            arr[int(t)] = v
    return arr


def _rolling(arr: np.ndarray, window: int) -> np.ndarray:
    """Mean over the last `window` seconds (fewer at the start)."""
    c = np.concatenate([[0.0], np.cumsum(arr)])
    idx = np.arange(len(arr))
    lo = np.maximum(0, idx + 1 - window)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)


def passage_rows(with_run: dict, without_run: dict | None, t_empty: dict) -> list:
    """without_run None (no reference run): recovery and extra wait are left empty."""
    end = max(max(q) for run in (with_run, without_run) if run is not None for q in run["q"].values() if q)
    W = int(round(CYCLE_S))
    qa = {g: _series(with_run["q"][g], end) for g in LEGS}
    if without_run is not None:
        qb = {g: _series(without_run["q"][g], end) for g in LEGS}
        ra, rb = {g: _rolling(qa[g], W) for g in LEGS}, {g: _rolling(qb[g], W) for g in LEGS}
        ok = np.all([np.abs(ra[g] - rb[g]) <= np.maximum(0.1 * rb[g], 1.0) for g in LEGS], axis=0)
    rows = []
    for p in with_run["passages"]:
        r = dict(level=with_run["level"], seed=with_run["seed"], method=with_run["method"], **p)
        tx = p.get("t_exit_gate")
        r["junction_time_s"] = round(tx - p["t_release"], 1) if tx is not None else None
        r["t_empty_s"] = t_empty.get(p["leg"])
        r["delay_s"] = round(r["junction_time_s"] - t_empty[p["leg"]], 1) if tx is not None else None
        r["travel_time_s"] = round(p["t_arrive"] - p["t_release"], 1) if p.get("t_arrive") is not None else None
        r["recovery_s"], r["recovery_censored"], r["recovery_overlaps_next_release"] = None, True, None
        if without_run is None:                                  # no reference run: recovery and extra wait not measured
            r["recovery_censored"] = None
            r.update({f"extra_wait_s_{g}": None for g in LEGS})
            rows.append(r)
            continue
        if tx is not None:                                       # queues back to the no-ambulance run for a full cycle
            run_start = None
            for t in range(int(tx) + 1, int(end) + 1):
                if ok[t]:
                    run_start = t if run_start is None else run_start
                    if t - run_start >= W:
                        r["recovery_s"], r["recovery_censored"] = round(run_start - tx, 1), False
                        # the next ambulance (released RELEASE_EVERY_S after this one) may have disturbed the queues
                        r["recovery_overlaps_next_release"] = bool(t > p["t_release"] + RELEASE_EVERY_S)
                        break
                else:
                    run_start = None
        t0, t1 = int(p["t_release"]), int(min(p["t_release"] + RELEASE_EVERY_S, end))
        for g in LEGS:                                           # extra standing time per vehicle served
            served = sum(n for ts_, n in without_run["stop"][g] if t0 <= ts_ < t1)
            r[f"extra_wait_s_{g}"] = round(float(np.sum(qa[g][t0:t1] - qb[g][t0:t1])) / served, 2) if served else None
        rows.append(r)
    return rows


def evaluate(out: str, methods: list, levels: list, seeds: list, workers: int = 8, n_releases: int | None = None,
             moto_share: float | None = None, references: bool = True) -> dict:
    os.makedirs(out, exist_ok=True)
    vtypes_xml(os.path.join(DATA, "demand", "vtypes.add.xml"))
    vt = None
    if moto_share is not None:                                   # evaluation-only sensitivity of the vehicle mix
        from .demand import mix_with_moto_share
        vt = os.path.join(out, f"vtypes_moto{moto_share:g}.add.xml")
        vtypes_xml(vt, mix_with_moto_share(moto_share))
    t_empty = empty_road_times()
    # reference runs without ambulances: priority_mp and preempt_naive decide exactly as max_pressure and fixed do
    # when no ambulance is around, so they reuse those runs
    ref = {m: REFERENCE_OF.get(m, m) for m in methods}
    jobs = [(lv, sd, m, True) for lv in levels for sd in seeds for m in methods]
    if references:
        jobs += [(lv, sd, m, False) for lv in levels for sd in seeds for m in sorted(set(ref.values()))]
    with tempfile.TemporaryDirectory(prefix="fatmawati_eval_") as tmp:
        for lv in levels:                                        # trips first, shared by every method
            for sd in seeds:
                routes_xml(os.path.join(tmp, f"eval_{lv}_s{sd}.rou.xml"), lv, seed=sd, start_hour=7.0,
                           duration_s=schedule(n_releases)[-1][0] + TAIL_S, peak=True, ev_per_hour=0.0)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_episode, *zip(*[(lv, sd, m, ev, tmp, n_releases, vt) for lv, sd, m, ev in jobs])))
    by = {(r["level"], r["seed"], r["method"], r["with_ev"]): r for r in results}
    rows = []
    for lv in levels:
        for sd in seeds:
            for m in methods:
                rows += passage_rows(by[(lv, sd, m, True)], by.get((lv, sd, ref[m], False)) if references else None, t_empty)
    pd.DataFrame(rows).to_csv(os.path.join(out, "passages.csv"), index=False)
    pd.DataFrame([r["run"] for r in results]).to_csv(os.path.join(out, "episodes_eval.csv"), index=False)
    pd.DataFrame([r["qc"] for r in results]).to_csv(os.path.join(out, "evaluation_qc.csv"), index=False)
    with open(os.path.join(out, "evaluation_config.json"), "w") as f:
        from .demand import MIX, mix_with_moto_share
        json.dump(dict(methods=methods, levels=levels, seeds=seeds, t_empty_s=t_empty, release_points=release_points(),
                       reference_runs=references,
                       vehicle_mix=mix_with_moto_share(moto_share) if moto_share is not None else MIX,
                       schedule=schedule(n_releases), cycle_s=CYCLE_S, release_every_s=RELEASE_EVERY_S, warm_s=WARM_S), f, indent=1)
    return dict(passages=len(rows), runs=len(results), t_empty_s=t_empty)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("data", "fatmawati_eval"))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--levels", default="low,design,high")
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--releases", type=int, default=None, help="releases per episode (default 24 = 3 movements x 8 slices)")
    ap.add_argument("--moto-share", type=float, default=None, help="motorcycle share for a sensitivity run (base 0.80)")
    ap.add_argument("--no-reference", action="store_true", help="skip the runs without ambulances (recovery and extra wait empty)")
    ap.add_argument("--hindsight", action="store_true", help="best switch time per passage (first seed only, forked replays)")
    ap.add_argument("--children", type=int, default=4, help="hindsight: forked replays at a time per episode")
    ap.add_argument("--stride", type=float, default=5.0, help="hindsight: candidate switch times every STRIDE s")
    a = ap.parse_args(argv)
    if a.hindsight:
        print(json.dumps(hindsight(a.out, a.levels.split(","), int(a.seeds.split(",")[0]), a.workers, max_children=a.children,
                                   n_releases=a.releases, stride_s=a.stride)))
        return 0
    print(json.dumps(evaluate(a.out, a.methods.split(","), a.levels.split(","), [int(x) for x in a.seeds.split(",")], a.workers,
                              a.releases, a.moto_share, references=not a.no_reference)))
    return 0



# ---------------------------------------------------------------- hindsight (best switch time per passage)
def _fork_runs(w, jobs: list, run_child, max_children: int, timeout_s: float = 3600.0) -> list:
    """Run run_child(w, job) in forked copies of this process (libsumo state is copied exactly; SUMO's own
    save/load-state is not exact for this network) and return their JSON results in job order."""
    import signal
    results = [None] * len(jobs)
    k = 0
    while k < len(jobs):
        batch = []
        for idx in range(k, min(k + max_children, len(jobs))):
            rfd, wfd = os.pipe()
            pid = os.fork()
            if pid == 0:                                         # child
                os.close(rfd)
                try:
                    out = run_child(w, jobs[idx])
                except Exception as e:                           # report, never fall back into the parent's loop
                    out = dict(error=repr(e))
                os.write(wfd, json.dumps(out).encode())
                os.close(wfd)
                os._exit(0)
            os.close(wfd)
            batch.append((idx, pid, rfd, time.monotonic()))
        for idx, pid, rfd, t0 in batch:
            data = b""
            while True:
                chunk = os.read(rfd, 1 << 16)
                if not chunk:
                    break
                data += chunk
            os.close(rfd)
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            results[idx] = json.loads(data) if data else dict(error="no result")
        k += len(batch)
    return results


def _child_passage(w, job: dict) -> dict:
    """In a forked copy: run on (forcing the ambulance's stage from job['t_from'] if given) until the ambulance
    passes the hospital-side exit gate; report its gate times."""
    vid, stage, t_from = job["vid"], job["stage"], job.get("t_from")
    if t_from is not None:
        w.force = (stage, t_from, vid)
    t_cap = w.t + 1800.0
    while w.t < t_cap:
        w.step()
        r = w.sensors.ev.get(vid) or next((x for x in w.sensors.ev_done if x.get("vid") == vid), None)
        if r is not None and any(g == HOSPITAL_LEG for g, _ in r.get("exits", [])):
            return dict(t_exit=max(t for g, t in r["exits"] if g == HOSPITAL_LEG), t_stopline=r.get("t_stopline"))
    return dict(t_exit=None)


def hindsight_episode(level: str, seed: int, tmp: str, base: str = "max_pressure", n_releases: int | None = None,
                      stride_s: float = 5.0, max_children: int = 4, vtypes: str | None = None) -> list:
    """Base run with `base`; at each ambulance's first detection, fork one replay that keeps `base` (base delay and
    crossing time) and one per candidate switch time from detection to that crossing, every stride_s, that requests
    the ambulance's stage from then until it has crossed.  The lowest delay is the hindsight ceiling."""
    from .build_net import STAGE_OF_APPROACH
    from .demand import hospital_edge
    from .sumo_world import FatmawatiWorld, VTYPES
    import sumolib
    sched = schedule(n_releases)
    end = sched[-1][0] + TAIL_S
    path = os.path.join(tmp, f"eval_{level}_s{seed}.rou.xml")
    if not os.path.exists(path):
        routes_xml(path, level, seed=seed, start_hour=7.0, duration_s=end, peak=True, ev_per_hour=0.0)
    # all trips are loaded at start: SUMO otherwise reads the trip file as it goes, and forked replays share the
    # parent's file position ("unexpected end of input")
    w = FatmawatiWorld(path, start_hour=7.0, seed=seed, controller=base, sensors=True, record_from=WARM_S,
                       store_rows=False, vtypes=vtypes or VTYPES, extra_args=("--route-steps", "0"))
    ts = w.ts
    hosp = hospital_edge(sumolib.net.readNet(NET, withInternal=False))
    rel = release_points()
    pending, meta, done = list(sched), {}, set()
    rows = []
    while (w.t < end - 1e-9 or any(v in w.sensors.ev for v in meta)) and w.t < end + EXTRA_S:
        while pending and w.t >= pending[0][0] - 1e-9:
            t_rel, leg, k = pending.pop(0)
            edge, pos, dist = rel[leg]
            vid = f"ev_{len(meta):02d}"
            ts.route.add(f"r_{vid}", ts.simulation.findRoute(edge, hosp[0].getID(), vType="ambulance").edges)
            ts.vehicle.add(vid, f"r_{vid}", typeID="ambulance", depart="now", departPos=str(pos), departLane="best",
                           departSpeed="max", arrivalPos=str(hosp[1]))
            meta[vid] = dict(passage=len(meta), leg=leg, slice=k, t_release=t_rel)
        w.step()
        for vid, m in meta.items():
            r = w.sensors.ev.get(vid)
            if vid in done or r is None or "t_detect" not in r:
                continue
            done.add(vid)
            stage = STAGE_OF_APPROACH[m["leg"]]
            (base_run,) = _fork_runs(w, [dict(vid=vid, stage=stage)], _child_passage, 1)
            t_cross = base_run.get("t_stopline") or base_run.get("t_exit") or (w.t + 600.0)
            cands = list(np.arange(w.t, t_cross + stride_s, stride_s))
            runs = _fork_runs(w, [dict(vid=vid, stage=stage, t_from=float(d)) for d in cands], _child_passage, max_children)
            best = min(((r_["t_exit"], d) for r_, d in zip(runs, cands) if r_.get("t_exit") is not None), default=(None, None))
            rows.append(dict(level=level, seed=seed, base=base, **m, t_detect=r["t_detect"], stage=stage,
                             base_t_exit=base_run.get("t_exit"), best_t_exit=best[0], best_switch_t=best[1],
                             n_candidates=len(cands)))
    w.close()
    return rows


def hindsight(out: str, levels: list, seed: int, workers: int = 8, max_children: int = 4, base: str = "max_pressure",
              n_releases: int | None = None, stride_s: float = 5.0) -> dict:
    """One seed per cell: every level's episode is a base run plus forked replays (passages_hindsight.csv)."""
    os.makedirs(out, exist_ok=True)
    vtypes_xml(os.path.join(DATA, "demand", "vtypes.add.xml"))
    t_empty = empty_road_times()
    with tempfile.TemporaryDirectory(prefix="fatmawati_hind_") as tmp:
        for lv in levels:
            routes_xml(os.path.join(tmp, f"eval_{lv}_s{seed}.rou.xml"), lv, seed=seed, start_hour=7.0,
                       duration_s=schedule(n_releases)[-1][0] + TAIL_S, peak=True, ev_per_hour=0.0)
        with ProcessPoolExecutor(max_workers=max(1, workers // max_children)) as pool:
            results = list(pool.map(hindsight_episode, levels, [seed] * len(levels), [tmp] * len(levels), [base] * len(levels),
                                    [n_releases] * len(levels), [stride_s] * len(levels), [max_children] * len(levels)))
    rows = []
    for r in (x for res in results for x in res):
        te = t_empty[r["leg"]]
        r["t_empty_s"] = te
        r["base_delay_s"] = round(r["base_t_exit"] - r["t_release"] - te, 1) if r["base_t_exit"] is not None else None
        r["hindsight_delay_s"] = round(r["best_t_exit"] - r["t_release"] - te, 1) if r["best_t_exit"] is not None else None
        r["best_switch_after_detect_s"] = round(r["best_switch_t"] - r["t_detect"], 1) if r["best_switch_t"] is not None else None
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out, "passages_hindsight.csv"), index=False)
    return dict(passages=len(rows), t_empty_s=t_empty)


if __name__ == "__main__":
    sys.exit(main())
