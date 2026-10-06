"""Generate the Fatmawati sensor release (DECISIONS §13.7): CSV tables of what the virtual sensors recorded.

  python3 -m traffic_twin.fatmawati.generate --plan pilot --out data/fatmawati_pilot --workers 2

Tables (CSV only; no vehicle trajectories, internal model states or driver parameters):
  sensors.csv               sensor inventory: stop-line loops, entry/exit gates, queue and exit cameras
  episodes.csv              one row per episode: scenario, and outcomes computed from the sensor tables
  loops_1min.csv            point sensors per minute: counts by class, PCU, occupancy, speed at the loop
  cameras_1min.csv          cameras per minute: standing vehicles, queue length, speed, motorcycles in the box
  ticks_5s.csv              one row per 5 s decision instant: per-approach readings, signal state, labels
  signal_log.csv            controller log: one row per phase change
  turning_counts_15min.csv  entry leg -> exit leg counts by class, and cordon travel times (re-identification)
  ambulance_trips.csv       one row per ambulance trip: AVL log, priority-request detector, stop line, recovery
  simulation_qc.csv         simulation diagnostics per episode (not sensor readings)
  data_dictionary.csv, DATACARD.md, config.json, SHA256SUMS.txt

Every controller sees the same demand (same trips and seed) within a window, level and seed, so controller
comparisons are paired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .demand import CALIBRATION, DATA, LEVEL_FACTOR, MIX, NET, load_calibration, routes_xml, vtypes_xml
from .sensors import CAMERA_M, CLASSES, EXIT_CAMERA_M, GATE_R, LEGS, PCU, sensor_rows

GENERATOR_VERSION = "traffic_twin_fatmawati_v3.0"
WINDOWS = {"am_peak": 7.0, "midday": 12.0, "pm_peak": 17.0}
PLANS = {
    "pilot": dict(windows=["am_peak"], levels=["design"], controllers=["fixed", "priority_mp"], seeds=[1],
                  warmup_min=15, main_min=30),
    "smoke": dict(windows=["am_peak"], levels=["design"], controllers=["priority_mp"], seeds=[1], warmup_min=2, main_min=3),
    "release": dict(windows=list(WINDOWS), levels=["low", "design", "high"],
                    controllers=["fixed", "longest_queue", "max_pressure", "priority_mp"], seeds=[1, 2, 3],
                    warmup_min=15, main_min=60),                    # 108 episodes, 108 recorded hours
}
LABEL_HORIZON_S = 30.0


def episode_specs(plan: dict) -> list:
    specs = []
    for window in plan["windows"]:
        for level in plan["levels"]:
            for seed in plan["seeds"]:
                for controller in plan["controllers"]:
                    specs.append(dict(episode_id=len(specs), window=window, start_hour=WINDOWS[window], level=level,
                                      seed=seed, controller=controller, warmup_s=60.0 * plan["warmup_min"],
                                      main_s=60.0 * plan["main_min"]))
    return specs


def routes_path(tmp: str, spec: dict) -> str:
    return os.path.join(tmp, f"trips_{spec['window']}_{spec['level']}_s{spec['seed']}.rou.xml")


def _hms(start_hour: float, t: float) -> str:
    h = (start_hour * 3600.0 + t) % 86400.0
    return f"{int(h // 3600):02d}:{int(h % 3600 // 60):02d}:{int(h % 60):02d}"


# ---------------------------------------------------------------- one episode (runs in a worker process)
def run_episode(spec: dict, tmp: str) -> dict:
    from .sumo_world import FatmawatiWorld
    eid, warm, main = spec["episode_id"], spec["warmup_s"], spec["main_s"]
    t0_h = spec["start_hour"] - warm / 3600.0                     # clock time of simulation t = 0
    wall0 = time.perf_counter()
    w = FatmawatiWorld(routes_path(tmp, spec), start_hour=t0_h, seed=spec["seed"], controller=spec["controller"],
                       sensors=True, record_from=warm)
    ts = w.ts
    base = None
    end = warm + main

    def ambulance_on_way() -> bool:                    # departed in the recorded period, not arrived yet
        return any(warm - 1e-9 <= r["t_depart"] < end for r in w.sensors.ev.values())

    while w.t < end + LABEL_HORIZON_S - 1e-9 or (ambulance_on_way() and w.t < end + LABEL_HORIZON_S + 900.0):
        w.step()
        if base is None and w.t >= warm:
            base = dict(lane=w.n_lane_collisions, junction=w.n_junction_overlaps, tele=w.n_teleports,
                        dep=w.n_departed, arr=w.n_arrived, lane_s=w.n_lane_collisions_study,
                        junction_s=w.n_junction_overlaps_study, vkm=w.veh_km, vkm_s=w.veh_km_study)
    S = w.sensors
    S.finish(w.t)
    key = dict(episode_id=eid)

    def clocked(rows, tcol):
        return [{**key, **r, "clock": _hms(t0_h, r[tcol])} for r in rows if warm - 1e-9 <= r[tcol] < end - 1e-9]

    loops = clocked(S.rows_loops, "t_start")
    cams = clocked(S.rows_cams, "t_start")
    ticks = pd.DataFrame([{**key, **r} for r in S.rows_ticks if r["t"] >= warm - 1e-9])
    if len(ticks):
        lag = int(round(LABEL_HORIZON_S / S.tick_s))
        for g in LEGS:
            ticks[f"y_q_veh_30s_{g}"] = ticks[f"q_veh_{g}"].shift(-lag)
        ticks = ticks[ticks["t"] < end - 1e-9]
    signal = [{**key, **r, "clock": _hms(t0_h, r["t"])} for r in w.signal_log if warm - 1e-9 <= r["t"] < end]
    turning = []
    for (period, a, b, c), tts in sorted(S.cordon.items()):
        if period * 900.0 >= end:
            continue
        turning.append(dict(episode_id=eid, t_start=period * 900.0, clock=_hms(t0_h, period * 900.0), entry_leg=a,
                            exit_leg=b, vclass=c, n=len(tts), tt_mean_s=round(float(np.mean(tts)), 1),
                            tt_p50_s=round(float(np.median(tts)), 1), tt_p85_s=round(float(np.percentile(tts, 85)), 1)))
    amb = []
    recorded = [r for r in sorted(S.ev_done, key=lambda r: r["t_depart"]) if warm - 1e-9 <= r["t_depart"] < end]
    for k, r in enumerate(recorded):
        row = dict(episode_id=eid, trip=f"A{k + 1:02d}", t_depart=r["t_depart"], clock_depart=_hms(t0_h, r["t_depart"]))
        for f in ("detect_leg", "t_detect", "detect_dist_m", "queue_at_detect", "entry_leg", "t_entry_gate", "stopline_leg",
                  "t_stopline", "exit_leg", "t_exit_gate", "t_recovered", "t_arrive"):
            row[f] = r.get(f)
        row["arrived"] = r.get("t_arrive") is not None
        row["travel_time_s"] = round(r["t_arrive"] - r["t_depart"], 1) if row["arrived"] else None
        row["cordon_time_s"] = (round(r["t_exit_gate"] - r["t_entry_gate"], 1)
                                if r.get("t_exit_gate") is not None and r.get("t_entry_gate") is not None else None)
        row["recovery_s"] = (round(r["t_recovered"] - r["t_stopline"], 1)
                             if r.get("t_recovered") is not None and r.get("t_stopline") is not None else None)
        amb.append(row)
    # outcomes from the sensor tables only
    out = dict(spec)
    delays = {}
    stopline = {g: sum(r["n_total"] for r in loops if r["kind"] == "stopline_loop" and r["leg"] == g) for g in LEGS}
    for g in LEGS:
        qs = [q for t, q in S.queue_series[g] if t < end]
        n = stopline[g]
        out[f"stopline_veh_{g}"] = n
        out[f"queue_veh_mean_{g}"] = round(float(np.mean(qs)), 2) if qs else None
        delays[g] = sum(qs) / n if n else None               # 1 Hz samples: vehicle-seconds standing per vehicle served
        out[f"stopped_delay_s_{g}"] = round(delays[g], 1) if delays[g] is not None else None
    total_q = sum(sum(q for t, q in S.queue_series[g] if t < end) for g in LEGS)
    total_n = sum(stopline.values())
    out["stopped_delay_s"] = round(total_q / total_n, 1) if total_n else None
    d = [v for v in delays.values() if v is not None]
    out["fairness_ratio"] = round(max(d) / min(d), 2) if len(d) == 4 and min(d) > 0 else None
    entries = [r for r in loops if r["kind"] == "entry_gate"]
    n_in = sum(r["n_total"] for r in entries)
    out["entry_veh_h"] = round(n_in * 3600.0 / main, 0)
    out["motorcycle_share_entry"] = round(sum(r["n_motorcycle"] for r in entries) / n_in, 3) if n_in else None
    tts = [x for (p, a, b, c), v in S.cordon.items() if p * 900.0 < end for x in v]
    out["cordon_tt_mean_s"] = round(float(np.mean(tts)), 1) if tts else None
    out["cordon_tt_p85_s"] = round(float(np.percentile(tts, 85)), 1) if tts else None
    arrived = [r for r in amb if r["arrived"]]
    out["ambulance_trips"] = len(amb)
    out["ambulance_tt_mean_s"] = round(float(np.mean([r["travel_time_s"] for r in arrived])), 1) if arrived else None
    rec = [r["recovery_s"] for r in amb if r["recovery_s"] is not None]
    out["ambulance_recovery_mean_s"] = round(float(np.mean(rec)), 1) if rec else None
    qc = dict(episode_id=eid, lane_collisions=w.n_lane_collisions - base["lane"],
              junction_overlaps=w.n_junction_overlaps - base["junction"],
              lane_collisions_study=w.n_lane_collisions_study - base["lane_s"],
              junction_overlaps_study=w.n_junction_overlaps_study - base["junction_s"], teleports=w.n_teleports - base["tele"],
              veh_km=round(w.veh_km - base["vkm"], 1), veh_km_study=round(w.veh_km_study - base["vkm_s"], 1),
              departed=w.n_departed - base["dep"], arrived=w.n_arrived - base["arr"],
              waiting_to_enter=len(ts.simulation.getPendingVehicles()), unmatched_exits=S.n_unmatched_exit,
              wall_s=round(time.perf_counter() - wall0, 1), sumo_version=ts.getVersion()[1])
    w.close()
    return dict(episode=out, loops=loops, cams=cams, ticks=ticks, signal=signal, turning=turning, amb=amb, qc=qc)


# ---------------------------------------------------------------- documentation
def dictionary_rows() -> list:
    rows = []

    def add(table, column, unit, role, desc):
        rows.append(dict(table=table, column=column, unit=unit, role=role, description=desc))

    add("sensors", "sensor_id", "", "key", "SL_<leg><lane> stop-line loop, EN_/EX_<leg><lane> entry/exit gate loop, QC_<leg> queue camera, XC_<leg> exit camera")
    add("sensors", "kind", "", "metadata", "stopline_loop, entry_gate, exit_gate, queue_camera or exit_camera")
    add("sensors", "leg", "", "metadata", "N/S = Jl. RS Fatmawati, E = Jl. TB Simatupang, W = Jl. RA Kartini")
    add("sensors", "lane", "", "metadata", "lane index from the kerb (0) outwards; -1 for cameras (all lanes)")
    add("sensors", "street", "", "metadata", "street name (OpenStreetMap)")
    add("sensors", "lat", "deg", "metadata", "WGS84 latitude of the loop, or of the camera's stop line / exit")
    add("sensors", "lon", "deg", "metadata", "WGS84 longitude")
    add("sensors", "dist_centre_m", "m", "metadata", "straight-line distance from the junction centre (-6.29250, 106.79522)")
    add("sensors", "zone_length_m", "m", "metadata", f"camera zone length along the road ({CAMERA_M:g} m approach, {EXIT_CAMERA_M:g} m exit); 0 for loops")
    add("sensors", "n_lane_segments", "", "metadata", "number of lane segments the camera zone covers")
    add("episodes", "episode_id", "", "key", "episode number")
    add("episodes", "window", "", "scenario", "am_peak (07:00), midday (12:00) or pm_peak (17:00): start of the recorded hour")
    add("episodes", "start_hour", "h", "scenario", "clock time at which recording starts")
    add("episodes", "level", "", "scenario", "demand level: low / design / high = " + " / ".join(f"{LEVEL_FACTOR[k]:g}" for k in ("low", "design", "high")) + " x the calibrated peak-hour volumes (calibration.json), times the hour's share of the daily-profile peak")
    add("episodes", "seed", "", "scenario", "random seed of the demand and the simulation; the same seed gives the same trips for every controller")
    add("episodes", "controller", "", "scenario", "fixed (Dishub greens S1 80 / S2 12 / S3 78 / S4 45 s in order), longest_queue (stage with the most standing vehicles), max_pressure (stage with the largest pressure) or priority_mp (max_pressure that serves an ambulance in time: the teacher)")
    add("episodes", "warmup_s", "s", "scenario", "simulated time before recording starts (network filling); nothing is recorded in it")
    add("episodes", "main_s", "s", "scenario", "recorded duration")
    for g in LEGS:
        add("episodes", f"stopline_veh_{g}", "veh", "outcome", f"vehicles counted by the {g} stop-line loops in the recorded period")
        add("episodes", f"queue_veh_mean_{g}", "veh", "outcome", f"mean standing vehicles seen by camera QC_{g} (1 Hz)")
        add("episodes", f"stopped_delay_s_{g}", "s/veh", "outcome", f"stopped delay on {g}: sum over seconds of standing vehicles in QC_{g} divided by stop-line count")
    add("episodes", "stopped_delay_s", "s/veh", "outcome", "junction stopped delay: all approaches' standing vehicle-seconds / all stop-line counts")
    add("episodes", "fairness_ratio", "", "outcome", "largest / smallest approach stopped delay (empty if an approach served nobody)")
    add("episodes", "entry_veh_h", "veh/h", "outcome", "vehicles entering through the four entry gates per hour")
    add("episodes", "motorcycle_share_entry", "", "outcome", "motorcycle share of the entry-gate counts")
    add("episodes", "cordon_tt_mean_s", "s", "outcome", "mean entry-gate to exit-gate travel time (re-identified vehicles)")
    add("episodes", "cordon_tt_p85_s", "s", "outcome", "85th percentile of the cordon travel time")
    add("episodes", "ambulance_trips", "", "outcome", "ambulance trips that departed in the recorded period")
    add("episodes", "ambulance_tt_mean_s", "s", "outcome", "mean ambulance travel time from departure to RSUP Fatmawati (AVL)")
    add("episodes", "ambulance_recovery_mean_s", "s", "outcome", "mean queue recovery time after an ambulance crossed the stop line (see ambulance_trips)")
    for tab, kind in (("loops_1min", "loop"), ("cameras_1min", "camera")):
        add(tab, "episode_id", "", "key", "episode number")
        add(tab, "t_start", "s", "key", "start of the minute, simulated seconds since the episode started (including warm-up)")
        add(tab, "clock", "hh:mm:ss", "key", "clock time of t_start")
        add(tab, "sensor_id", "", "key", "sensor (see sensors.csv)")
        add(tab, "kind", "", "metadata", "sensor kind")
        add(tab, "leg", "", "metadata", "leg of the sensor")
    add("loops_1min", "lane", "", "metadata", "lane index of the loop")
    for c in CLASSES:
        add("loops_1min", f"n_{c}", "veh", "reading", f"{c}s that entered the loop in the minute")
    add("loops_1min", "n_total", "veh", "reading", "all vehicles that entered the loop in the minute")
    add("loops_1min", "pcu", "pcu", "reading", "count in passenger-car units, PKJI 2023 (Bina Marga 09/P/BM/2023) emp for protected phases: " + ", ".join(f"{c} {PCU[c]}" for c in CLASSES))
    add("loops_1min", "occupancy_pct", "%", "reading", "share of the minute the loop was occupied")
    add("loops_1min", "speed_kmh", "km/h", "reading", "mean speed of the vehicles as they entered the loop (empty if none)")
    add("cameras_1min", "queue_veh_mean", "veh", "reading", "mean standing vehicles (speed < 0.1 m/s) in the zone (1 Hz samples)")
    add("cameras_1min", "queue_veh_max", "veh", "reading", "maximum standing vehicles in the minute")
    add("cameras_1min", "queue_pcu_mean", "pcu", "reading", "mean standing vehicles in PCU")
    add("cameras_1min", "queue_m_mean", "m", "reading", "mean distance from the stop line (exit cameras: from the junction exit) to the back of the farthest standing vehicle")
    add("cameras_1min", "queue_m_max", "m", "reading", "maximum of that distance in the minute")
    add("cameras_1min", "vehicles_mean", "veh", "reading", "mean vehicles in the zone, moving or not")
    add("cameras_1min", "speed_kmh_mean", "km/h", "reading", "space-mean speed of the vehicles in the zone")
    add("cameras_1min", "moto_in_box_mean", "veh", "reading", "mean motorcycles within 8 m of the stop line (the motorcycle box); approach cameras only")
    add("cameras_1min", "ambulance_present_s", "s", "reading", "seconds in the minute with an ambulance in the zone")
    add("ticks_5s", "episode_id", "", "key", "episode number")
    add("ticks_5s", "t", "s", "key", "simulated seconds since the episode started (multiple of 5)")
    add("ticks_5s", "clock", "hh:mm:ss", "key", "clock time")
    for g in LEGS:
        add("ticks_5s", f"q_veh_{g}", "veh", "feature", f"standing vehicles in camera QC_{g} now")
        add("ticks_5s", f"q_pcu_{g}", "pcu", "feature", f"standing vehicles in QC_{g} in PCU")
        add("ticks_5s", f"q_m_{g}", "m", "feature", f"queue length on {g} (stop line to back of the farthest standing vehicle)")
        add("ticks_5s", f"moto_box_{g}", "veh", "feature", f"motorcycles in the {g} motorcycle box")
        add("ticks_5s", f"cam_speed_kmh_{g}", "km/h", "feature", f"space-mean speed in QC_{g} (empty if the zone is empty)")
        add("ticks_5s", f"dis_5s_{g}", "veh", "feature", f"vehicles over the {g} stop-line loops in the last 5 s")
        add("ticks_5s", f"dis_60s_{g}", "veh", "feature", f"vehicles over the {g} stop-line loops in the last 60 s")
        add("ticks_5s", f"arr_60s_{g}", "veh", "feature", f"vehicles through the {g} entry gate in the last 60 s")
        add("ticks_5s", f"exit_q_pcu_{g}", "pcu", "feature", f"standing vehicles on the {g} exit (camera XC_{g}), PCU")
        add("ticks_5s", f"ev_{g}", "", "feature", f"1 if the priority-request detector sees an ambulance on {g} (inside QC_{g})")
        add("ticks_5s", f"ev_dist_m_{g}", "m", "feature", f"distance of the nearest ambulance on {g} to the stop line (empty if none)")
        add("ticks_5s", f"ev_eta_s_{g}", "s", "feature", f"its arrival time at the stop line: ev_dist_m_{g} / max(its speed, 5 m/s)")
        add("ticks_5s", f"ev_q_ahead_pcu_{g}", "pcu", "feature", f"standing vehicles ahead of it in QC_{g}, PCU")
        add("ticks_5s", f"ev_needs_signal_{g}", "", "feature", f"1 if its priority request says its route crosses the {g} stop line (0: free-left slip or U-turn loop)")
        add("ticks_5s", f"pressure_{g}", "pcu", "feature", f"max-pressure of {g}: q_pcu_{g} minus the exit queues weighted by the turning shares counted so far")
    add("ticks_5s", "q_veh_E_S1S2", "veh", "feature", "standing vehicles in QC_E on the lanes that lead to the through lanes (green in S1 and S2)")
    add("ticks_5s", "q_veh_E_S2", "veh", "feature", "standing vehicles in QC_E on the lanes that lead to the right-turn lanes (green in S2 only)")
    for st in ("S1", "S2", "S3", "S4"):
        add("ticks_5s", f"stage_q_{st}", "veh", "feature", f"standing vehicles stage {st} would serve (camera lane groups it gives green)")
        add("ticks_5s", f"stage_pressure_{st}", "pcu", "feature", f"max-pressure of stage {st}: approach pressures shared over lane groups by standing PCU")
    add("ticks_5s", "controller", "", "metadata", "controller running the signal in this episode")
    add("ticks_5s", "stage", "", "feature", "stage that has (or, during a transition, last had) green: S1 RA Kartini + TB Simatupang through, S2 TB Simatupang through + right, S3 north, S4 south")
    add("ticks_5s", "phase_state", "", "feature", "green, amber or all-red")
    add("ticks_5s", "transition_to", "", "feature", "stage the signal is changing to (empty when green)")
    add("ticks_5s", "t_in_phase_s", "s", "feature", "time since the current phase started")
    add("ticks_5s", "t_since_green_s", "s", "feature", "time since the stage got green")
    add("ticks_5s", "controller_request", "", "metadata", "stage the running controller requested at its last decision instant; equals the label under priority_mp, so exclude it from features")
    add("ticks_5s", "y_teacher_stage", "", "label", "stage the priority-aware teacher requests now: the stage of an ambulance whose route crosses the signal when its ETA <= intergreen 4 s + start-up loss 2 s + (3600/S) x PCU queue ahead of it + the 5 s decision interval (S = PKJI 2023 base 600 x approach width, PCU/h), else the largest stage pressure with a 1 PCU switching margin")
    for g in LEGS:
        add("ticks_5s", f"y_q_veh_30s_{g}", "veh", "label", f"q_veh_{g} 30 s later (forecast target)")
    add("signal_log", "episode_id", "", "key", "episode number")
    add("signal_log", "t", "s", "key", "simulated seconds when the phase started")
    add("signal_log", "clock", "hh:mm:ss", "key", "clock time")
    add("signal_log", "phase", "", "reading", "phase index of program dishub")
    add("signal_log", "stage", "", "reading", "stage (S1-S4) when green, or the change 'S1>S2' during amber and all-red")
    add("signal_log", "state", "", "reading", "green, amber or all-red")
    add("signal_log", "controller", "", "metadata", "controller running the signal")
    add("turning_counts_15min", "episode_id", "", "key", "episode number")
    add("turning_counts_15min", "t_start", "s", "key", "start of the 15-minute period (by exit time)")
    add("turning_counts_15min", "clock", "hh:mm:ss", "key", "clock time of t_start")
    add("turning_counts_15min", "entry_leg", "", "key", "leg whose entry gate the vehicle passed")
    add("turning_counts_15min", "exit_leg", "", "key", "leg whose exit gate it passed (same leg = U-turn)")
    add("turning_counts_15min", "vclass", "", "key", "motorcycle, car, bus, truck or emergency")
    add("turning_counts_15min", "n", "veh", "reading", "vehicles of the movement and class that exited in the period")
    add("turning_counts_15min", "tt_mean_s", "s", "reading", "mean entry-to-exit travel time")
    add("turning_counts_15min", "tt_p50_s", "s", "reading", "median entry-to-exit travel time")
    add("turning_counts_15min", "tt_p85_s", "s", "reading", "85th percentile entry-to-exit travel time")
    for col, unit, desc in (("episode_id", "", "episode number"), ("trip", "", "ambulance trip in the episode (A01, A02, ...)"),
                            ("t_depart", "s", "departure from the network edge (AVL)"), ("clock_depart", "hh:mm:ss", "clock time of departure"),
                            ("detect_leg", "", "approach where the priority-request detector first saw it"),
                            ("t_detect", "s", "first detection inside an approach camera zone (300 m)"),
                            ("detect_dist_m", "m", "distance to the stop line at detection"),
                            ("queue_at_detect", "veh", "standing vehicles on all four approach cameras at detection"),
                            ("entry_leg", "", "leg of the entry gate passed"), ("t_entry_gate", "s", "entry gate time"),
                            ("stopline_leg", "", "approach whose stop-line loop it crossed"), ("t_stopline", "s", "stop-line crossing time"),
                            ("exit_leg", "", "leg of the exit gate passed"), ("t_exit_gate", "s", "exit gate time"),
                            ("t_recovered", "s", "first time after the stop-line crossing that the four approach queues are back within 10% (or 1 vehicle) of queue_at_detect"),
                            ("t_arrive", "s", "arrival at RSUP Fatmawati (AVL); empty if still travelling at the end"),
                            ("arrived", "", "reached the hospital within the episode"),
                            ("travel_time_s", "s", "t_arrive - t_depart"), ("cordon_time_s", "s", "t_exit_gate - t_entry_gate"),
                            ("recovery_s", "s", "t_recovered - t_stopline")):
        add("ambulance_trips", col, unit, "key" if col in ("episode_id", "trip") else "reading", desc)
    for col, desc in (("episode_id", "episode number"), ("lane_collisions", "overlapping vehicles on a lane anywhere in the 2.5 km network in the recorded period (rare, between motorcycles at walking pace: a limitation of SUMO's sublane model)"),
                      ("junction_overlaps", "overlapping vehicle shapes inside junction areas anywhere in the network (SUMO sublane limitation)"),
                      ("lane_collisions_study", "lane collisions within 300 m of the Fatmawati junction centre, where the sensors are"),
                      ("junction_overlaps_study", "junction overlaps within 300 m of the Fatmawati junction centre"),
                      ("veh_km", "vehicle-km travelled in the whole network in the recorded period (sampled every 10 s)"),
                      ("veh_km_study", "vehicle-km travelled on roads within 300 m of the junction centre"),
                      ("teleports", "vehicles removed after being stuck for 300 s"), ("departed", "vehicles inserted at the network edge"),
                      ("arrived", "vehicles that reached their destination"),
                      ("waiting_to_enter", "vehicles waiting at the network edge at the end (inflow blocked)"),
                      ("unmatched_exits", "exit-gate detections without an entry (should be 0)"),
                      ("wall_s", "computing time"), ("sumo_version", "SUMO version")):
        add("simulation_qc", col, "", "qc", desc + " (simulation diagnostic, not a sensor reading)")
    return rows


def _calibration_text() -> str:
    try:
        with open(CALIBRATION) as f:
            c = json.load(f)
    except (OSError, ValueError):
        return "Demand is NOT calibrated (no calibration.json): volumes are 0.3 x the observed counts, so queues are indicative."
    v = c["verification"]
    legs = ("N", "E", "S", "W")
    sd = v.get("max_queue_sd_m", {})

    def status(g):
        if c.get("capacity_limited", {}).get(g):
            return "capacity-limited (its demand backs up upstream before the camera queue reaches the target)"
        err = c.get("rel_error", {}).get(g)
        return f"{err:+.0%} of target" if err is not None else ""
    return ("Demand is the 13 movements counted at the junction in a weekday peak hour (Sumabrata, Haqueline & Mauramdha 2026), "
            "scaled per approach so that, under the Dishub fixed plan, the peak-hour maximum queue approaches that study's "
            "VISSIM maximum queue. Scales " + ", ".join(f"{g} {c['scale'][g]:.3f}" for g in legs) + ". Check over "
            f"{len(v.get('seeds', [])) or 2} seeds, maximum queue: "
            + "; ".join(f"{g} {v['max_queue_m'][g]:.0f} m" + (f" (SD {sd[g]:.0f})" if g in sd else "")
                        + f" vs {c['target_max_queue_m'][g]:.0f} m, {status(g)}" for g in legs)
            + ". Near capacity an approach's hourly maximum queue tips from a short queue to spill-back past the 300 m camera and "
            "varies strongly between seeds, so a closer fit was not possible. Stopped delay "
            + ", ".join(f"{g} {v['stopped_delay_s'][g]:.0f} s (PKJI control delay {v['pkji_delay_s'][g]:.0f} s)" for g in legs)
            + ". The twin carries " + ", ".join(f"{g} {100 * v['volume_share_of_observed'][g]:.0f}%" for g in legs)
            + " of the observed volumes at these queues: SUMO's motorcycles discharge far less densely than Jakarta's, so the "
            "twin reproduces the congestion, not the absolute counts. Vehicle mix 80% motorcycles (Yusuf & Isradi 2025).")


def datacard(out: str, plan_name: str, plan: dict, tables: dict, episodes: pd.DataFrame, qc: pd.DataFrame) -> str:
    n = {k: len(v) for k, v in tables.items()}
    lines = [
        f"# Fatmawati twin sensor dataset ({GENERATOR_VERSION}, plan '{plan_name}')", "",
        "Virtual-sensor readings from a SUMO digital twin of Perempatan Fatmawati (Cilandak, South Jakarta): what a roadside "
        "installation of loops, counting gates and queue cameras would record. The tables hold sensor readings only; no "
        "vehicle trajectories, model states or driver parameters. Vehicle identities were used inside the simulator to match "
        "entries with exits (as video or ANPR re-identification does) and are not written out.", "",
        "## Calibration", "",
        _calibration_text(), "",
        "Sensors are exact (no detection errors).", "",
        "## Sensors", "",
        f"- Stop-line loops: one per approach lane, 1 m before the stop line ({sum(1 for _ in LEGS)} approaches).",
        "- Entry and exit gates: a loop on every lane of each leg, " + ", ".join(f"{g} {GATE_R[g]:g} m" for g in LEGS) +
        " from the junction centre, outside the free-left slips and the median U-turn loops.",
        f"- Queue cameras: {CAMERA_M:g} m upstream of each stop line; exit cameras: first {EXIT_CAMERA_M:g} m of each exit.",
        "- Priority-request detector: an ambulance inside an approach camera zone.", "",
        "## Episodes", "",
        f"{len(episodes)} episodes: windows {plan['windows']}, demand levels {plan['levels']}, controllers {plan['controllers']}, "
        f"seeds {plan['seeds']}; {plan['warmup_min']} min warm-up (not recorded) then {plan['main_min']} min recorded.", "",
        "## Tables", "",
    ]
    lines += [f"- {k}.csv: {v} rows" for k, v in n.items()]
    lines += ["", "## Simulation quality (simulation_qc.csv)", "",
              f"- whole 2.5 km network: lane collisions {int(qc['lane_collisions'].sum())}, junction overlaps "
              f"{int(qc['junction_overlaps'].sum())}, teleports {int(qc['teleports'].sum())}; within 300 m of the junction (the "
              f"sensors' area): lane collisions {int(qc['lane_collisions_study'].sum())}, junction overlaps "
              f"{int(qc['junction_overlaps_study'].sum())}; unmatched exits {int(qc['unmatched_exits'].sum())}. Lane collisions "
              "are rare overlaps between motorcycles at walking pace, a limitation of SUMO's sublane model with filtering "
              "motorcycles, and are reported rather than hidden.", "",
              "## Licence and attribution", "",
              "The road network and sensor positions derive from OpenStreetMap: © OpenStreetMap contributors, available under "
              "the Open Database License (ODbL 1.0); this dataset is shared under the ODbL. Simulation: Eclipse SUMO (EPL 2.0). "
              "No TomTom or Google data were stored or used to make it.", ""]
    path = os.path.join(out, "DATACARD.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- driver
def generate(out: str, plan_name: str = "pilot", workers: int = 2) -> dict:
    plan = PLANS[plan_name]
    specs = episode_specs(plan)
    os.makedirs(out, exist_ok=True)
    vtypes_xml(os.path.join(DATA, "demand", "vtypes.add.xml"))
    with tempfile.TemporaryDirectory(prefix="fatmawati_trips_") as tmp:
        for spec in specs:                                     # one trip file per window, level and seed
            path = routes_path(tmp, spec)
            if not os.path.exists(path):
                routes_xml(path, spec["level"], start_hour=spec["start_hour"] - spec["warmup_s"] / 3600.0,
                           duration_s=spec["warmup_s"] + spec["main_s"] + LABEL_HORIZON_S, seed=spec["seed"])
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(run_episode, spec, tmp) for spec in specs]
            results = []
            for k, f in enumerate(futures):
                results.append(f.result())
                print(f"episode {k + 1}/{len(specs)} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    from .sumo_world import sensor_layout
    tables = dict(
        sensors=pd.DataFrame(sensor_rows(sensor_layout())),
        episodes=pd.DataFrame([r["episode"] for r in results]),
        loops_1min=pd.DataFrame([x for r in results for x in r["loops"]]),
        cameras_1min=pd.DataFrame([x for r in results for x in r["cams"]]),
        ticks_5s=pd.concat([r["ticks"] for r in results if len(r["ticks"])], ignore_index=True),
        signal_log=pd.DataFrame([x for r in results for x in r["signal"]]),
        turning_counts_15min=pd.DataFrame([x for r in results for x in r["turning"]]),
        ambulance_trips=pd.DataFrame([x for r in results for x in r["amb"]]),
        simulation_qc=pd.DataFrame([r["qc"] for r in results]),
    )
    order = {"ticks_5s": ["episode_id", "t", "clock"], "loops_1min": ["episode_id", "t_start", "clock"],
             "cameras_1min": ["episode_id", "t_start", "clock"], "signal_log": ["episode_id", "t", "clock"]}
    documented = {}
    for r in dictionary_rows():
        documented.setdefault(r["table"], []).append(r["column"])
    for name, df in tables.items():
        if not len(df.columns):                                  # nothing happened (e.g. no ambulance): header only
            df = tables[name] = pd.DataFrame(columns=documented.get(name, []))
        if name in order and len(df):
            first = order[name]
            df = df[first + [c for c in df.columns if c not in first]]
            tables[name] = df
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    pd.DataFrame(dictionary_rows()).to_csv(os.path.join(out, "data_dictionary.csv"), index=False)
    datacard(out, plan_name, plan, tables, tables["episodes"], tables["simulation_qc"])
    cfg = dict(generator=GENERATOR_VERSION, plan=plan_name, plan_spec=plan, windows=WINDOWS, level_factor=LEVEL_FACTOR,
               calibration_scale=load_calibration(),
               vehicle_mix=MIX, pcu=PCU, gate_radius_m=GATE_R, camera_m=CAMERA_M, exit_camera_m=EXIT_CAMERA_M,
               label_horizon_s=LABEL_HORIZON_S, network_sha256=sha256(NET), python=sys.version.split()[0])
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out, "SHA256SUMS.txt"), "w") as f:
        for name in sorted(os.listdir(out)):
            if name != "SHA256SUMS.txt":
                f.write(f"{sha256(os.path.join(out, name))}  {name}\n")
    return {k: len(v) for k, v in tables.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="pilot", choices=sorted(PLANS))
    ap.add_argument("--out", default=os.path.join("data", "fatmawati_pilot"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    counts = generate(args.out, args.plan, args.workers)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
