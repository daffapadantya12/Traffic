"""Dataset generator.  Run: python3 -m traffic_twin.generate --episodes 10 --out data/dev10

Writes CSV only: episodes.csv, vehicles.csv, signal_ticks.csv, junction_1hz.csv,
network_edges.csv, data_dictionary.csv, config.json, DATACARD.md and, for the
episode ids given with --trajectories, vehicle_states_<id>.csv.

Traffic-engineering idea: one deterministic policy would give a dataset with no
counterfactual actions, so episodes mix controllers, demand levels, OD patterns
and sensor noise (DECISIONS §7); the label is always the max-pressure request.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

from .calibrate import calibrate_demand
from .config import GENERATOR_VERSION, SimConfig, load_config
from .controllers import controller_eps
from .features import (ACTION_COLUMNS, APPROACH_COLUMNS, DOWNSTREAM_COLUMNS, KEY_COLUMNS, LABEL_COLUMNS,
                       NEIGHBOUR_COLUMNS, SENSOR_COLUMNS, SIGNAL_COLUMNS, SIGNAL_TICK_COLUMNS)
from .network import build_grid, network_edges_table
from .sim import Scenario, World


# ----------------------------------------------------------------- scenario mixture
def _stratified_labels(labels, probs, n: int, rng) -> list:
    """n labels in the given proportions (largest remainder), shuffled."""
    raw = np.array(probs, dtype=float) * n
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: n - counts.sum()]:
        counts[i] += 1
    out = [lab for lab, c in zip(labels, counts) for _ in range(c)]
    rng.shuffle(out)
    return out


def sample_scenarios(cfg: SimConfig, n: int, start_id: int = 0) -> list:
    """Stratified scenario mixture of DECISIONS §7 plus the split assignment."""
    rng = np.random.default_rng(cfg.seed_base + start_id)
    levels = _stratified_labels(cfg.demand_levels, cfg.demand_level_probs, n, rng)
    asyms = _stratified_labels(cfg.demand_asyms, cfg.demand_asym_probs, n, rng)
    ods = _stratified_labels(cfg.od_patterns, cfg.od_pattern_probs, n, rng)
    ctrls = _stratified_labels(cfg.controllers, cfg.controller_probs, n, rng)
    noises = _stratified_labels(cfg.sensor_noise_levels, cfg.sensor_noise_probs, n, rng)
    ev_rates = _stratified_labels(cfg.ev_rates, cfg.ev_rate_probs, n, rng)
    scenarios = []
    for i in range(n):
        jitter = rng.uniform(1 - cfg.demand_jitter, 1 + cfg.demand_jitter)
        scenarios.append(Scenario(
            demand_level=levels[i], lambda_base_vph=cfg.lambda_base_vph[levels[i]] * jitter,
            demand_asym=asyms[i], od_pattern=ods[i], controller=ctrls[i],
            controller_eps=controller_eps(ctrls[i], cfg), sensor_noise_level=float(noises[i]),
            ev_rate=ev_rates[i], lambda_ev_vph=cfg.lambda_ev_vph if ev_rates[i] == "default" else 0.0))
    assign_splits(cfg, scenarios, rng)
    return scenarios


def assign_splits(cfg: SimConfig, scenarios: list, rng) -> None:
    """70/15/15 train/val/test stratified by (demand_level, controller); every
    stress episode -> test_ood; random_dwell episodes -> train only."""
    groups = {}
    for i, sc in enumerate(scenarios):
        groups.setdefault((sc.demand_level, sc.controller), []).append(i)
    f_train, f_val, _ = cfg.split_fractions
    for members in groups.values():
        members = list(members)
        rng.shuffle(members)
        for pos, i in enumerate(members):
            f = (pos + 0.5) / len(members)
            scenarios[i].split = "train" if f < f_train else ("val" if f < f_train + f_val else "test")
    for sc in scenarios:
        if sc.demand_level == "stress":
            sc.split = "test_ood"
        elif sc.controller == "random_dwell":
            sc.split = "train"


# ----------------------------------------------------------------- data dictionary
def data_dictionary_rows() -> list:
    rows = []

    def add(table, column, dtype, unit, desc, ns, twin=True, leaky="", null="never"):
        rows.append(dict(table=table, column=column, dtype=dtype, unit=unit, description=desc,
                         namespace=ns, twin_observable=twin, leaky_for=leaky, null_policy=null))

    T = "signal_ticks"
    add(T, "episode_id", "int32", "", "episode key (= seed - 100000)", "key", False)
    add(T, "junction_id", "int8", "", "junction key j*grid_x + i; NOT a feature", "key", False)
    add(T, "t", "float32", "s", "decision instant; t = 0 is the start of warm-up", "time", False, "all")
    add(T, "tick_idx", "int32", "", "running index of decision rows of this junction", "time", False, "all")
    add(T, "is_warmup", "bool", "", "t < warmup_s; exclude from training and KPIs", "time", False)
    add(T, "is_cooldown", "bool", "", "t >= warmup_s + main_s (spawning off)", "time", False)
    add(T, "is_tail", "bool", "", "last 30 s of the main period (regression label NaN)", "time", False)
    add(T, "phase", "int8", "", "approach that is green: 0 N, 1 E, 2 S, 3 W", "signal")
    add(T, "time_in_phase_s", "float32", "s", "time since the current green began", "signal")
    add(T, "max_green_remaining_s", "float32", "s", "max_green - time_in_phase_s", "signal")
    for a in "NESW":
        add(T, f"q_{a}", "int16", "veh", f"vehicles with v < 2 m/s on the approach edge from arm {a} (all lanes)", "state")
        add(T, f"n_{a}", "int16", "veh", f"all vehicles on the approach edge from arm {a}", "state")
        add(T, f"q_{a}_l0", "int16", "veh", "queue on lane 0 (kerb lane); -1 when lanes_per_dir = 1", "state")
        add(T, f"q_{a}_l1", "int16", "veh", "queue on lane 1 (median lane); -1 when lanes_per_dir = 1", "state")
        add(T, f"arr_{a}_30s", "int16", "veh", "vehicles that entered the approach edge in (t-30, t]", "state")
    for d in "NESW":
        add(T, f"down_n_{d}", "int16", "veh", f"vehicles on the outgoing edge toward {d} (stubs: real count)", "downstream")
        add(T, f"down_occ_{d}", "float32", "", "down_n / storage of that edge (storage = per-lane storage x lanes)", "downstream")
        add(T, f"down_blocked_{d}", "bool", "", "a queued vehicle leaves < l_veh+s0+1 m free at the start of some lane of that edge", "downstream")
    for d in "NESW":
        add(T, f"nb_{d}_exists", "bool", "", f"there is a neighbour junction in direction {d}", "neighbour")
        add(T, f"nb_{d}_phase", "int8", "", "neighbour's phase (green or most recently green); -1 if none", "neighbour")
        add(T, f"nb_{d}_green_toward_me", "bool", "", "neighbour is GREEN for an approach that feeds the link toward this junction", "neighbour")
    for a in "NESW":
        add(T, f"obs_q_{a}", "int16", "veh", "Binomial(q_a, 1-p_miss) + Poisson(p_miss/2), p_miss = sensor_noise_level; = q_a at 0", "sensor")
    for a in "NESW":
        add(T, f"ev_present_{a}", "bool", "", f"an ambulance is on the approach edge from arm {a} within ev_horizon_m of the stop line (GPS channel, no noise)", "ev")
        add(T, f"ev_dist_{a}", "float32", "m", "distance of the nearest ambulance to the stop line; -1 if none", "ev")
        add(T, f"ev_eta_{a}", "float32", "s", "ev_dist / max(v_ambulance, 3 m/s); -1 if none", "ev")
        add(T, f"ev_q_ahead_{a}", "int16", "veh", "queued vehicles in the ambulance's lane ahead of it; -1 if none (input of the priority_mp teacher)", "ev")
    add(T, "action_requested", "int8", "", "phase requested by the driving controller", "action", False, "imitation")
    add(T, "action_applied", "int8", "", "phase in force after the plant's filter (the pending phase when a switch was accepted)", "action", False, "imitation")
    add(T, "forced", "int8", "", "0 none, 1 min_green, 2 max_green, 3 all_red, 4 starvation guard (an occupied approach had no green for starvation_s)", "action", False, "imitation")
    add(T, "switched", "bool", "", "a switch (all-red) started at this instant", "action", False, "imitation")
    add(T, "y_mp_action", "int8", "", "request of the priority-aware teacher priority_mp recomputed from this row (label for imitation): "
        "the ambulance's approach when ev_eta_a <= allred_min + startup lost time + ev_q_ahead_a x saturation headway, otherwise the "
        "max-pressure request; identical to the max-pressure request on every row without an ambulance", "label", False, "all")
    add(T, "y_q_total_t30", "float32", "veh", "q_total of this junction at the first whole second >= t+30 (junction_1hz)", "label", False, "all", "NaN on tail/cooldown rows or when the episode ended early")

    E = "episodes"
    for c, d, u in [("episode_id", "int32", ""), ("seed", "int64", ""), ("config_hash", "str", ""), ("generator_version", "str", ""),
                    ("grid_x", "int8", ""), ("grid_y", "int8", ""), ("lanes_per_dir", "int8", ""), ("drive_side", "str", ""),
                    ("L_block_m", "float32", "m"), ("v_limit_mps", "float32", "m/s"), ("dt_s", "float32", "s"),
                    ("warmup_s", "float32", "s"), ("main_s", "float32", "s"), ("cooldown_s", "float32", "s"),
                    ("demand_level", "str", ""), ("lambda_base_vph", "float32", "veh/h per entry node"), ("demand_asym", "str", ""),
                    ("od_pattern", "str", ""), ("controller", "str", ""), ("controller_eps", "float32", ""),
                    ("sensor_noise_level", "float32", ""), ("split", "str", ""), ("ev_rate", "str", ""),
                    ("lambda_ev_vph", "float32", "veh/h per entry node")]:
        add(E, c, d, u, "episode setting", "key")
    for c, d, u, desc in [
        ("n_spawned", "int32", "veh", "spawn requests in the whole episode"),
        ("n_arrived", "int32", "veh", "vehicles that reached their destination"),
        ("n_in_network_end", "int32", "veh", "vehicles still on the network at the end"),
        ("n_entry_queued_end", "int32", "veh", "vehicles still waiting in entry queues at the end (conservation: spawned = arrived + in_network + queued)"),
        ("n_in_network_main_end", "int32", "veh", "vehicles in the network when the main period ends (spawning stops): where queues kept growing"),
        ("n_entry_queued_main_end", "int32", "veh", "vehicles waiting in entry queues when the main period ends"),
        ("throughput_vph", "float32", "veh/h", "vehicles that exited during the main period, per hour"),
        ("mean_total_delay_s", "float32", "s", "mean over general-traffic vehicles (every vehicle_type except ambulance) spawned in the main period (unfinished: delay so far)"),
        ("p90_total_delay_s", "float32", "s", "90th percentile of the same"),
        ("mean_stops", "float32", "", "mean number of stops per vehicle"),
        ("max_queue_veh", "int16", "veh", "max q_total of any junction (1 Hz, main period)"),
        ("switches_per_min", "float32", "1/min", "accepted switches per junction per minute of the main period"),
        ("n_red_commits", "int32", "", "vehicles latched COMMITTED at a red onset"),
        ("n_hard_brakes", "int32", "", "vehicle-steps with car-following acceleration < -3 m/s^2 while moving (step count, kept from v1; "
         "n_harsh_brake_events counts braking runs)"),
        ("n_overlap_fixed", "int32", "", "followers whose front ended a step inside the vehicle ahead and were put back behind it; every such case is "
         "also a 1-D collision (n_collisions), so it must be 0"),
        ("n_box_blocked", "int32", "", "vehicles that came to a stop inside a junction box"),
        ("allred_extension_mean_s", "float32", "s", "mean all-red duration beyond the 4 s minimum"),
        ("gridlock", "bool", "", "no stop-line crossing for gridlock_window_s (120 s) network-wide with >= 10 vehicles in the network, OR at one junction with >= 10 vehicles on its approach edges; the episode ended there (labelled failure)"),
        ("gridlock_t", "float32", "s", "time of gridlock detection (NaN if none)"),
        ("gridlock_junction", "int8", "", "junction that gridlocked; -1 = none or the network-wide rule fired"),
        ("max_approach_red_s", "float32", "s", "longest time (main period, 1 Hz) an approach held vehicles without being green; bounded by the plant's starvation guard (starvation_s + decision_interval + all-red)"),
        ("max_total_wait_s", "float32", "s", "largest total_wait_s of any car spawned in the main period"),
        ("n_forced_starvation", "int32", "", "main-period decision rows with forced = 4 (plant starvation guard)"),
        ("n_emergency_brakes", "int32", "", "vehicle-steps whose car-following demand was below -b_emerg while moving (clipped at -7 m/s^2); subset of n_hard_brakes"),
        ("n_commit_revoked", "int32", "", "COMMITTED vehicles that came to rest before the stop line and were re-latched STOPPING"),
        ("n_brake_sat_vehicle", "int32", "", "braking runs whose demand exceeded b_emerg behind another vehicle (target 0)"),
        ("n_brake_sat_virtual", "int32", "", "braking runs whose demand exceeded b_emerg behind a stop line (target 0)"),
        ("n_harsh_brake_events", "int32", "", "braking runs with applied deceleration >= 3.4 m/s^2 (AASHTO comfortable limit); mostly stopping for a red that "
         "appeared at short distance (no amber)"),
        ("n_turn_overspeed", "int32", "", "vehicles that reached the arc of a turning path more than 0.5 m/s above its curvature speed limit"),
        ("n_lane_changes", "int32", "", "lane changes on links (all kinds)"),
        ("n_lc_mandatory", "int32", "", "lane changes back into the lane the vehicle's next turn needs (only after it moved aside for an ambulance)"),
        ("n_lc_discretionary", "int32", "", "MOBIL lane changes of cars whose remaining route lets them choose a lane (faster lane, letting a faster car pass)"),
        ("n_lc_yield", "int32", "", "cars that moved out of the lane of an ambulance approaching from behind (siren range ev_siren_range_m)"),
        ("n_yield_blocked", "int32", "", "(car, ambulance) pairs where the car wanted to move aside but had no safe gap (boxed in, e.g. in a queue)"),
        ("n_ev_overtakes", "int32", "", "ambulance lane changes to pass slower traffic"),
        ("n_reroutes", "int32", "", "vehicles that reached the solid line in a lane that does not serve their movement and took another movement"),
        ("n_destination_changes", "int32", "", "rerouted vehicles whose lane only led off the network at another boundary node, where they left "
         "(their free-flow time is recomputed for the trip they made)"),
        ("n_pending_repicks", "int32", "", "all-red held allred_stall_s past its minimum by a vehicle stuck in a conflict zone of the pending approach, "
         "resolved by serving an approach whose zones are clear (liveness, never unsafe)"),
        ("lane_changes_per_veh", "float32", "", "n_lane_changes / vehicles in vehicles.csv of this episode"),
        ("n_conflicts", "int32", "", "car-following conflicts that started in the main period: time to collision < ttc_conflict_s (1.5 s, FHWA SSAM)"),
        ("n_near_misses", "int32", "", "conflicts whose deceleration-to-avoid (DRAC) reached drac_near_miss (3.4 m/s^2)"),
        ("n_serious", "int32", "", "near misses with time to collision < 1.0 s"),
        ("near_miss_per_1000_veh", "float32", "", "n_near_misses per 1000 general-traffic vehicles spawned in the main period"),
        ("min_ttc_s", "float32", "s", "smallest time to collision of any main-period conflict (NaN if none)"),
        ("max_drac_mps2", "float32", "m/s^2", "largest deceleration-to-avoid of any main-period conflict (0 if none)"),
        ("min_pet_s", "float32", "s", "smallest post-encroachment time at a box conflict zone in the main period (NaN if none <= 5 s)"),
        ("n_pet_events", "int32", "", "zone entries with a post-encroachment time <= 5 s after a vehicle of another approach (SSAM window)"),
        ("n_pet_serious", "int32", "", "of those, PET < pet_conflict_s (1.0 s)"),
        ("n_collisions", "int32", "", "bodies that overlapped: 1-D (a follower inside the vehicle ahead on its own path) plus box conflict zones occupied "
         "by two approaches at once; a safety invariant, must be 0"),
        ("min_gap_moving_m", "float32", "m", "smallest bumper gap to the vehicle ahead of any vehicle moving at >= 2 m/s"),
        ("max_lat_accel_mps2", "float32", "m/s^2", "largest lateral acceleration v^2/r on a turning arc (limit a_lat_max = 3 m/s^2; 0.5 s steps leave a small excess)"),
        ("ev_mean_blocked_s", "float32", "s", "mean time an ambulance (main period) stood behind other vehicles: the delay a queue ahead of it causes"),
        ("ev_mean_red_s", "float32", "s", "mean time an ambulance stood first in its lane at a red light"),
        ("max_head_wait_s", "float32", "s", "liveness: longest time any lane-head vehicle (first on its lane of an approach edge) stood with v < stop_speed, "
         "whole episode; bounded by the starvation guard for red waits, NOT for a blocked exit lane"),
        ("spillback_lock", "bool", "", "liveness: some approach lane was occupied without a stop-line crossing for gridlock_window_s although its approach was "
         "green for >= 2 x fixed_time_green s in that window (per junction, 1 Hz). Does not end the episode and does not change gridlock"),
        ("n_spillback_locks", "int32", "", "number of spillback-lock events (one per lane latch; the latch releases on a crossing or when the lane empties)"),
        ("spillback_lock_t", "float32", "s", "time of the first spillback lock (NaN if none)"),
        ("spillback_lock_junction", "int8", "", "junction of the first spillback lock; -1 if none"),
        ("measured_sat_flow_vphpl", "float32", "veh/h/lane", "saturation flow measured in the simulator under this config (calibration step)"),
        ("label_switch_rate", "float32", "", "share of main-period decision rows with y_mp_action != phase"),
        ("runtime_s", "float32", "s", "wall-clock time of the episode"),
        ("n_ambulances", "int32", "veh", "ambulance spawn requests in the whole episode"),
        ("ev_mean_travel_time_s", "float32", "s", "mean travel time of ambulances spawned in the main period (NaN if none)"),
        ("ev_mean_delay_s", "float32", "s", "mean total delay of those ambulances"),
        ("ev_p90_delay_s", "float32", "s", "90th percentile of ambulance delay"),
        ("ev_mean_stops", "float32", "", "mean stops per ambulance"),
        ("car_mean_delay_penalty_s", "float32", "s", "mean car delay minus the same in the paired run without ambulances (same seed, controller); NaN if no paired run"),
        ("recovery_time_s", "float32", "s", "mean time after an ambulance clears a junction until its q_total is back within 10% of the value at first sighting"),
        ("n_invariant_violations", "int32", "", "sum of the safety-invariant counters: two greens, a conflict zone of the new green still occupied at onset, "
         "COMMITTED vehicle still upstream at onset, red crossing, min green, short all-red, collisions, critical conflicts (DRAC > b_emerg), "
         "speed drop faster than b_emerg in one step, box entry from a lane that does not serve the movement, conservation (must be 0)"),
        ("vehicle_steps", "int64", "", "number of vehicle-steps simulated (denominator of the overlap rate)"),
    ]:
        add(E, c, d, u, desc, "kpi", False)

    V = "vehicles"
    for c, d, u, desc in [
        ("episode_id", "int32", "", "episode key"), ("veh_id", "int32", "", "vehicle key within the episode"),
        ("spawn_request_t", "float32", "s", "time the vehicle was drawn (parameters and route fixed here)"),
        ("enter_t", "float32", "s", "time it entered the network (NaN if it never did)"),
        ("exit_t", "float32", "s", "time its rear bumper passed the destination node (NaN if unfinished)"),
        ("completed", "bool", "", "reached the destination"), ("is_warmup", "bool", "", "spawn_request_t < warmup_s"),
        ("origin", "str", "", "boundary node"), ("destination", "str", "", "boundary node drawn from the OD row"),
        ("exit_node", "str", "", "boundary node it actually leaves at: the destination, unless it missed its lane after making room "
         "for an ambulance and its lane only led off the network elsewhere (episodes.n_destination_changes)"),
        ("route_movements", "str", "", "junction:movement per junction, e.g. J00:T|J10:NS"),
        ("route_edges", "str", "", "edge ids separated by | (after any reroute)"),
        ("route_len_m", "float32", "m", "edges + true lengths of the box paths (after any reroute)"),
        ("n_junctions", "int8", "", ""), ("n_turns", "int8", "", "NS + FS movements"),
        ("free_flow_time_s", "float32", "s", "sum(edge_len/v0) + sum(box path length / min(v0, curvature speed limit)) of the planned route"),
        ("travel_time_s", "float32", "s", "exit_t - enter_t (unfinished: so far)"),
        ("entry_queue_delay_s", "float32", "s", "enter_t - spawn_request_t"),
        ("total_delay_s", "float32", "s", "entry_queue_delay + travel_time - free_flow_time"),
        ("total_wait_s", "float32", "s", "time with v < 2 m/s"), ("n_stops", "int8", "", "transitions v > 2 -> v < 0.5 m/s"),
        ("n_red_commits", "int8", "", ""), ("n_hard_brakes", "int8", "", "vehicle-steps with a < -3 while moving"),
        ("accel_energy", "float32", "m^2/s^3", "sum(a^2 dt)"),
        ("n_harsh_brake_events", "int16", "", "braking runs with applied deceleration >= 3.4 m/s^2"),
        ("max_decel_mps2", "float32", "m/s^2", "largest one-step deceleration"),
        ("n_lane_changes", "int16", "", "lane changes of this vehicle"),
        ("n_lc_mandatory", "int16", "", "of them, back into the lane its next turn needs"),
        ("n_lc_yield", "int16", "", "of them, out of the lane of an ambulance behind it"),
        ("n_reroutes", "int8", "", "times it missed its lane at the solid line and took another movement"),
        ("n_conflicts", "int16", "", "car-following conflicts (TTC < 1.5 s) in which it was the follower"),
        ("n_near_misses", "int16", "", "of them, near misses (DRAC >= 3.4 m/s^2)"),
        ("n_collisions", "int16", "", "collisions it was involved in (must be 0)"),
        ("min_ttc_s", "float32", "s", "smallest time to collision it had as a follower below 1.5 s; NaN if never"),
        ("max_drac_mps2", "float32", "m/s^2", "largest deceleration-to-avoid in its conflicts (0 if none)"),
        ("ev_blocked_s", "float32", "s", "ambulances: time standing behind other vehicles; NaN for other vehicles"),
        ("ev_red_s", "float32", "s", "ambulances: time standing first in its lane at a red light; NaN for other vehicles"),
        ("n_overtakes", "int8", "", "ambulances: lane changes to pass slower traffic"),
        ("alpha", "float32", "", "hidden driver aggressiveness ~ Beta(2,2)"), ("v0_mps", "float32", "m/s", "desired speed"),
        ("a_max_mps2", "float32", "m/s^2", ""), ("b_comf_mps2", "float32", "m/s^2", ""), ("T_headway_s", "float32", "s", ""),
        ("vehicle_type", "str", "", "'car' or 'ambulance' (plus the types of the optional vehicle_mix when it is switched on)"),
        ("length_m", "float32", "m", "vehicle length"),
        ("politeness", "float32", "", "MOBIL politeness p = lc_politeness_max x (1 - alpha): aggressive drivers are less polite"),
    ]:
        add(V, c, d, u, desc, "vehicle", False)

    J = "junction_1hz"
    for c, d, u, desc in [("episode_id", "int32", "", ""), ("junction_id", "int8", "", ""), ("t", "int32", "s", "whole second"),
                          ("q_total", "int16", "veh", "sum of q_a over the four approaches"), ("n_total", "int16", "veh", "sum of n_a"),
                          ("wait_sum_total", "float32", "s", "sum of cumulative waiting time of the vehicles on the four approaches"),
                          ("crossings_cum", "int32", "veh", "cumulative stop-line crossings at this junction")]:
        add(J, c, d, u, desc, "aggregate", True)

    N = "network_edges"
    for c, d, u, desc in [
        ("edge_id", "str", "", "directed edge key '<from_node>_<to_node>'; vehicles.route_edges and vehicle_states.edge_id refer to it"),
        ("from_node", "str", "", "node the edge starts at (junction 'Jij' or boundary node W/E/S/N + road index)"),
        ("to_node", "str", "", "node the edge ends at"),
        ("heading_deg", "float32", "deg", "direction of travel, degrees counter-clockwise from east (0 E, 90 N, 180 W, 270 S)"),
        ("hx", "float32", "", "x component of the unit heading vector (east positive)"),
        ("hy", "float32", "", "y component of the unit heading vector (north positive)"),
        ("length_m", "float32", "m", "edge length box edge to box edge (internal) or boundary node to box edge (stub)"),
        ("n_lanes", "int8", "", "lanes in this direction (= lanes_per_dir)"),
        ("storage_veh_per_lane", "int16", "veh", "jam storage per lane = floor(length / (l_veh + s0))"),
        ("storage_veh", "int16", "veh", "storage_veh_per_lane x n_lanes; the normaliser of down_occ_* for this edge"),
        ("is_stub", "bool", "", "True when one end is a boundary node"),
        ("approach_dir_at_dest", "str", "", "arm (N/E/S/W) a vehicle on this edge arrives from at the destination junction; '' when the edge ends at a boundary node"),
        ("junction_id_at_dest", "int8", "", "junction id at the end of the edge; -1 when it ends at a boundary node"),
        ("junction_id_at_from", "int8", "", "junction id at the start of the edge; -1 when it starts at a boundary node"),
        ("x_start", "float32", "m", "x of the edge start on the road axis (origin = centre of the south-west junction, x east)"),
        ("y_start", "float32", "m", "y of the edge start on the road axis (y north)"),
        ("x_end", "float32", "m", "x of the edge end on the road axis"),
        ("y_end", "float32", "m", "y of the edge end on the road axis"),
    ]:
        add(N, c, d, u, desc, "static", True)

    P = "network_paths"
    for c, d, u, desc in [
        ("junction_id", "int8", "", "junction"), ("path_id", "str", "", "connector id '<junction>_<approach><movement><in_lane><out_lane>'"),
        ("approach", "str", "", "arm the vehicle arrives from"), ("movement", "str", "", "T through, NS near-side turn, FS far-side turn"),
        ("in_lane", "int8", "", "entry lane (0 kerb); T keeps its lane, NS leaves the kerb lane, FS the median lane"),
        ("out_lane", "int8", "", "exit lane (a turn ends in the lane its next turn needs, else the emptier one)"),
        ("in_edge", "str", "", "approach edge"), ("out_edge", "str", "", "exit edge"),
        ("radius_m", "float32", "m", "arc radius (NaN for a straight path)"),
        ("length_m", "float32", "m", "true path length = the 1-D connector length the simulator uses"),
        ("lead_in_m", "float32", "m", "straight part before the arc (a turn into the far exit lane)"),
        ("lead_out_m", "float32", "m", "straight part after the arc"),
        ("v_cap_mps", "float32", "m/s", "curvature speed limit sqrt(a_lat_max x r) (v_limit on straight paths)"),
        ("n_conflicting_paths", "int8", "", "paths of other approaches whose corridor touches this one (crossing or merging)"),
    ]:
        add(P, c, d, u, desc, "static", True)

    C = "conflicts"
    for c, d, u, desc in [
        ("episode_id", "int32", "", "episode key"), ("event_id", "int32", "", "event key within the episode"),
        ("type", "str", "", "rear_end, diverge (two paths leaving one entry lane), lane_change (follower of a car still sliding across), "
         "collision, or zone_pet (a vehicle entering a box conflict zone after one of another approach left it)"),
        ("junction_id", "int8", "", "junction of a zone_pet event; -1 otherwise"),
        ("edge_id", "str", "", "edge or box path of the first vehicle when the event started (junction node for zone_pet)"),
        ("t_start", "float32", "s", "event start"), ("t_end", "float32", "s", "last step the conflict criterion held (events merge across gaps <= 1 s)"),
        ("veh_a", "int32", "", "follower / entering vehicle"), ("veh_b", "int32", "", "leader / vehicle that used the zone before"),
        ("vtype_a", "str", "", ""), ("vtype_b", "str", "", ""),
        ("v_a_mps", "float32", "m/s", "speed of veh_a at the smallest TTC (zone_pet: at entry)"),
        ("v_b_mps", "float32", "m/s", "speed of veh_b at the smallest TTC (zone_pet: when the entry happened)"),
        ("min_ttc_s", "float32", "s", "smallest time to collision gap / (v_a - v_b) during the event (NaN for zone_pet)"),
        ("max_drac_mps2", "float32", "m/s^2", "largest deceleration rate to avoid the crash (v_a - v_b)^2 / (2 gap) (NaN for zone_pet)"),
        ("min_gap_m", "float32", "m", "smallest bumper gap during the event (NaN for zone_pet)"),
        ("pet_s", "float32", "s", "post-encroachment time (zone_pet only)"),
        ("near_miss", "bool", "", "max_drac_mps2 >= drac_near_miss (3.4 m/s^2)"),
        ("serious", "bool", "", "near miss with min_ttc_s < 1.0 s, or zone_pet with pet_s < 1.0 s"),
        ("critical", "bool", "", "max_drac_mps2 > b_emerg: unavoidable within the modelled braking limit (must not occur)"),
        ("collision", "bool", "", "the bodies overlapped (must not occur)"),
        ("is_main", "bool", "", "the event started in the main period (KPIs count only these)"),
    ]:
        add(C, c, d, u, desc, "safety", False)
    return rows


CONFLICT_COLUMNS = ["episode_id", "event_id", "type", "junction_id", "edge_id", "t_start", "t_end", "veh_a", "veh_b",
                    "vtype_a", "vtype_b", "v_a_mps", "v_b_mps", "min_ttc_s", "max_drac_mps2", "min_gap_m", "pet_s",
                    "near_miss", "serious", "critical", "collision", "is_main"]


def network_paths_table(net) -> list:
    """Rows for network_paths.csv: every lane-level box path of every junction (DECISIONS §12.1)."""
    rows = []
    for J in net.junctions:
        for c in J.connectors.values():
            n_conf = len({kb for (ka, kb) in net.conflicts if ka == c.key})
            rows.append(dict(junction_id=J.id, path_id=c.id, approach=c.approach, movement=c.movement, in_lane=c.in_lane,
                             out_lane=c.out_lane, in_edge=c.in_edge, out_edge=c.out_edge,
                             radius_m=round(c.path.radius, 3) if c.path.radius != math.inf else math.nan,
                             length_m=round(c.length, 3), lead_in_m=round(c.path.lead_in, 3),
                             lead_out_m=round(c.path.lead_out, 3), v_cap_mps=round(c.v_cap, 3), n_conflicting_paths=n_conf))
    return rows


# ----------------------------------------------------------------- data card
def _md(df: pd.DataFrame) -> str:
    """Tiny markdown table writer (avoids the optional `tabulate` dependency)."""
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


VALIDATION_SUMMARY = os.path.join("reports", "validation_summary.json")


def _validation_summary() -> dict:
    """reports/validation_summary.json written by validate.py, or {} when it is absent or was written
    by a different generator version (no number of a stale validation run is quoted in a data card)."""
    try:
        with open(VALIDATION_SUMMARY) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    if d.get("generator_version") != GENERATOR_VERSION:
        if getattr(_validation_summary, "_warned", False):
            return {}
        _validation_summary._warned = True
        print(f"warning: {VALIDATION_SUMMARY} was written by generator {d.get('generator_version')!r}, not {GENERATOR_VERSION!r}; "
              "its numbers are not quoted in the data card (re-run python3 -m traffic_twin.validate)")
        return {}
    return d


def _hard_brake_measured() -> str:
    """Measured harsh-braking numbers of the validation design episode for the data card; an
    empty string (no number claimed) when no current validation summary is available."""
    d = _validation_summary().get("design_episode", {})
    try:
        per_onset = float(d["harsh_brakes_per_red_onset"])
    except (KeyError, TypeError, ValueError):
        return ""
    return f" (measured {per_onset:.2f} per red onset in the generator's design-demand validation episode)"


def _paired_measured() -> str:
    """Measured max_pressure-vs-fixed_time paired difference at high demand (validation run)."""
    d = _validation_summary().get("paired_mp_vs_ft", {}).get("high", {})
    try:
        diff = float(d["paired_delay_diff_mp_minus_ft_s"])
        thr = float(d["mp_throughput_vph"]) / float(d["ft_throughput_vph"]) - 1.0
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return ""
    return f" (paired seeds at high demand in the validation run: {diff:+.0f} s/veh delay, {thr:+.0%} throughput)"


def _gridlock_text(episodes: pd.DataFrame) -> str:
    """Gridlock / spillback-lock counts measured on the episodes table being documented."""
    n = len(episodes)
    g = episodes[episodes["gridlock"].astype(bool)]
    if len(g):
        combos = g.groupby(["demand_level", "controller"]).size()
        where = ", ".join(f"{lvl}/{ctl} x{k}" for (lvl, ctl), k in combos.items())
        gl = f"In this release {len(g)} of {n} episodes ended in gridlock (demand level/controller: {where})."
    else:
        gl = f"In this release none of the {n} episodes ended in gridlock."
    sl = episodes[episodes["spillback_lock"].astype(bool)] if "spillback_lock" in episodes else episodes.iloc[0:0]
    if len(sl):
        combos = sl.groupby(["demand_level", "controller"]).size()
        where = ", ".join(f"{lvl}/{ctl} x{k}" for (lvl, ctl), k in combos.items())
        sp = f" A spillback lock (lane stuck behind a blocked exit although green) occurred in {len(sl)} episodes ({where})."
    else:
        sp = " No spillback lock occurred."
    return gl + sp


def _liveness_text(cfg: SimConfig, episodes: pd.DataFrame) -> str:
    bound = cfg.starvation_s + cfg.decision_interval + cfg.allred_min
    red = float(episodes["max_approach_red_s"].max()) if "max_approach_red_s" in episodes else math.nan
    head = float(episodes["max_head_wait_s"].max()) if "max_head_wait_s" in episodes else math.nan
    return (f"The guard bounds the red wait of an occupied approach at about {bound:.0f} s unless a blocked box extends the all-red; "
            f"the longest red wait in this release is {red:.0f} s (episodes.max_approach_red_s) and the longest standing time of a lane "
            f"head is {head:.0f} s (episodes.max_head_wait_s).")


def _load_table(episodes: pd.DataFrame) -> str:
    """Measured performance per demand level and controller (means over the episodes of this release)."""
    order = [x for x in ("low", "design", "high", "stress") if x in set(episodes["demand_level"])]
    g = episodes.groupby(["demand_level", "controller"]).agg(
        episodes=("episode_id", "count"), mean_total_delay_s=("mean_total_delay_s", "mean"), throughput_vph=("throughput_vph", "mean"),
        n_in_network_main_end=("n_in_network_main_end", "mean"), switches_per_min=("switches_per_min", "mean")).round(1)
    g = g.reindex(pd.MultiIndex.from_tuples([(l, c) for l in order for c in sorted(set(episodes["controller"])) if (l, c) in g.index],
                                            names=["demand_level", "controller"]))
    return _md(g)


def _safety_text(episodes: pd.DataFrame) -> str:
    """Safety numbers measured on the episodes table being documented."""
    if "n_near_misses" not in episodes:
        return ""
    n = len(episodes)
    nm = int(episodes["n_near_misses"].sum())
    conf = int(episodes["n_conflicts"].sum())
    coll = int(episodes["n_collisions"].sum())
    pet = episodes["min_pet_s"].min()
    pet_s = int(episodes["n_pet_serious"].sum())
    sat = int(episodes["n_brake_sat_vehicle"].sum() + episodes["n_brake_sat_virtual"].sum())
    lat = float(episodes["max_lat_accel_mps2"].max())
    return (f"Across the {n} episodes: {conf} car-following conflicts (TTC < 1.5 s), {nm} near misses (DRAC >= 3.4 m/s^2), "
            f"{coll} collisions, smallest post-encroachment time between approaches {pet:.1f} s ({pet_s} below 1.0 s), "
            f"{sat} braking demands beyond the physical limit, largest lateral acceleration on an arc {lat:.2f} m/s^2.")


def _lane_text(episodes: pd.DataFrame) -> str:
    if "n_lane_changes" not in episodes:
        return ""
    tot = episodes[["n_lane_changes", "n_lc_discretionary", "n_lc_yield", "n_lc_mandatory", "n_yield_blocked", "n_ev_overtakes",
                    "n_reroutes", "n_destination_changes"]].sum()
    return (f"In this release: {int(tot.n_lane_changes)} lane changes ({episodes['lane_changes_per_veh'].mean():.2f} per vehicle; "
            f"{int(tot.n_lc_discretionary)} discretionary, {int(tot.n_lc_yield)} cars moving aside for an ambulance, "
            f"{int(tot.n_lc_mandatory)} back into the lane a turn needs, {int(tot.n_ev_overtakes)} ambulance overtakes); "
            f"{int(tot.n_yield_blocked)} times a car had no room to move aside; {int(tot.n_reroutes)} reroutes, "
            f"{int(tot.n_destination_changes)} of them leaving the network at another boundary node.")


def write_datacard(path: str, cfg: SimConfig, episodes: pd.DataFrame, ticks: pd.DataFrame, vehicles: pd.DataFrame) -> None:
    main = ticks[~ticks.is_warmup & ~ticks.is_cooldown]
    mg = {m: (round(L, 2), round(v, 2)) for m, (L, v) in build_grid(cfg).movement_geometry.items()}
    lines = [
        "# Data card: traffic_twin v2 dataset", "",
        f"generator_version `{GENERATOR_VERSION}`, config_hash `{cfg.config_hash()}`, {len(episodes)} episodes, "
        f"{len(ticks)} signal_ticks rows, {len(vehicles)} vehicles.", "",
        "## Generating process",
        f"- Microscopic simulation of a {cfg.grid_x}x{cfg.grid_y} grid of signalised junctions, {cfg.lanes_per_dir} lane(s) per direction, "
        f"{cfg.drive_side}-hand traffic, blocks of {cfg.L_block:.0f} m, stubs of {cfg.L_stub:.0f} m, v_limit {cfg.v_limit} m/s.",
        f"- Drivers follow the ACC car-following model (improved IDM blended with the constant-acceleration heuristic, coolness "
        f"{cfg.acc_coolness}; Treiber & Kesting 2013), dt = {cfg.dt} s, ballistic update. It brakes as hard as a situation really needs - a red that "
        "appears at short distance, a car that changes lane in front - instead of the plain IDM's over-reaction, never accelerates harder than on a "
        "free road, and obeys the most demanding obstacle ahead (minimum acceleration over the vehicle ahead, the stop line, bodies on its box path). "
        "Parameters per vehicle: alpha ~ Beta(2,2), v0 = v_limit(1+0.2(alpha-0.5))LN(0,0.08) clipped [0.85,1.15]v_limit (speeding allowed), "
        "a_max, b and T likewise; lane-change politeness = 0.5(1 - alpha) (see config.json).",
        f"- Junction boxes have kerb returns of radius {cfg.corner_radius:.1f} m (box {cfg.w_box:.0f} m square). Through traffic keeps its lane; "
        "a near-side turn leaves the kerb lane and a far-side turn the median lane, each on a circular arc, into the exit lane its next turn needs "
        "(else the emptier one). Path length and curvature speed limit sqrt(a_lat r), a_lat = "
        f"{cfg.a_lat_max} m/s^2, per movement (same-lane path): {mg} (length m, v_cap m/s); every lane-level path is in network_paths.csv. "
        "Turning vehicles slow down so that they reach the arc at its limit.",
        f"- Lane changes on the links (MOBIL, Kesting, Treiber & Helbing 2007): a change needs gaps >= {cfg.lc_min_gap} m to the new leader and "
        f"follower, neither may have to brake harder than {cfg.lc_b_safe} m/s^2 or close in with TTC < {cfg.lc_min_ttc_s} s; no change within "
        f"{cfg.lc_no_change_dist:.0f} m of a stop line (solid line) or before the rear is {cfg.lc_after_box_dist:.0f} m into a link, at most one per "
        f"{cfg.lc_cooldown_s:.0f} s, and the old-lane follower keeps treating the changer as a leader for {cfg.lc_duration_s:.0f} s (the sideways "
        "move). Vehicles choose their lane by look-ahead (spawn and turn exits put them in the lane their remaining turns need), so only "
        "vehicles whose remaining route leaves them free change lane for speed. Cars within "
        f"{cfg.ev_siren_range_m:.0f} m ahead of an ambulance in its lane move aside toward the kerb if there is room (an ambulance that "
        "needs the kerb lane makes kerb-lane cars move out); a boxed-in car stays put, and nobody crosses a stop line to make room. "
        "Ambulances overtake through the other lane. A car that moved aside and cannot regain its lane before the solid line takes another "
        f"movement and is rerouted. {_lane_text(episodes)}",
        f"- Signals: split phasing (one approach green at a time), min green {cfg.min_green:.0f} s, max green {cfg.max_green:.0f} s, "
        f"all-red >= {cfg.allred_min:.0f} s extended until no vehicle of another approach is still inside a conflict zone of the pending approach "
        "(rear bumpers included; a static conflict matrix of the lane-level paths) AND no vehicle latched COMMITTED at the last red onset is still "
        f"upstream of the box, decisions every {cfg.decision_interval:.0f} s of green. At red onset a vehicle stops if d >= v t_pr + v^2/(2 b_stop) "
        f"(t_pr = {cfg.t_pr_s} s, b_stop = {cfg.b_stop} m/s^2, the ITE dilemma-zone rule), else it is COMMITTED and crosses during the all-red. "
        f"Plant starvation guard: an approach that holds vehicles and has had no green for {cfg.starvation_s:.0f} s is served next (forced = 4). "
        f"{_liveness_text(cfg, episodes)} The guard does not help a lane head whose exit lane is blocked by spillback: episodes.max_head_wait_s "
        "(longest standing lane head) and episodes.spillback_lock / n_spillback_locks (an occupied lane with no crossing for "
        f"{cfg.gridlock_window_s:.0f} s despite >= {2 * cfg.fixed_time_green:.0f} s of green) report that case.",
        f"- Gridlock: no stop-line crossing for {cfg.gridlock_window_s:.0f} s while >= {cfg.gridlock_min_vehicles} vehicles wait, checked "
        "network-wide and per junction (episodes.gridlock_junction); the episode ends there and is kept as a labelled failure. "
        f"{_gridlock_text(episodes)}",
        f"- Episodes: {cfg.warmup_s:.0f} s warm-up + {cfg.main_s:.0f} s main + {cfg.cooldown_s:.0f} s cool-down (spawning off); seed = 100000 + episode_id, "
        "five RNG streams (demand, vehicles, controller, sensor, ambulances); lane changing, yielding and rerouting draw no random numbers.",
        f"- Saturation flow measured in the simulator (per lane, lane changes off): {cfg.measured_sat_flow_vphpl:.0f} veh/h/lane (start-up lost time "
        f"{cfg.startup_lost_time_s:.2f} s). Capacity per approach: formula sat_flow x lanes x 18/96 = {cfg.capacity_formula_vph:.0f} veh/h; realised "
        f"fixed-time capacity measured in a saturated 1x1 probe episode (x = {cfg.capacity_probe_x} of the formula value, {cfg.capacity_probe_main_s:.0f} s) "
        f"= {cfg.capacity_vph:.0f} veh/h; capacity_realisation = {cfg.capacity_realisation:.3f}. Demand levels are v/c against the MEASURED capacity: "
        f"lambda_base per level (veh/h per entry node): {cfg.lambda_base_vph}.",
        "- Scenario mixture: demand_level {low 25%, design 40%, high 25%, stress 10%} (lambda jittered U(0.9,1.1)), demand_asym {sym, ns_heavy}, "
        "od_pattern {default 70%, uniform 30%}, controller {max_pressure 40%, max_pressure_eps 30%, fixed_time 20%, random_dwell 10%}, "
        "sensor_noise_level {0: 60%, 0.1: 40%}.",
        "- Splits by episode: 70/15/15 stratified by (demand_level, controller); all stress episodes -> test_ood; random_dwell -> train only.",
        f"- Ambulances (healthcare scenario): ev_rate {{0: 30%, default: 70%}}; at 'default' every entry node spawns ambulances at "
        f"{cfg.lambda_ev_vph} per hour from a separate RNG stream (cars keep common random numbers for their draws; their trajectories differ "
        f"because they make room). Ambulance: v0 = {cfg.ev_v0_factor} v_limit, a_max {cfg.ev_a_max}, b {cfg.ev_b}, T {cfg.ev_T} s, length "
        f"{cfg.ev_l_veh} m; same car following, same stop-or-commit, same signals (never runs a red). Controllers see them through "
        "ev_present/ev_dist/ev_eta/ev_q_ahead per approach (GPS-style, no sensor noise). episodes.ev_mean_blocked_s (standing behind other "
        "vehicles) and ev_mean_red_s (first in its lane at a red) split an ambulance's standing time.", "",
        "## Safety measures (surrogate safety, FHWA SSAM)",
        f"- conflicts.csv holds one row per event with raw values, so thresholds can be changed: car-following conflicts (time to collision "
        f"TTC = gap / (v - v_lead) < {cfg.ttc_conflict_s} s), near misses (deceleration to avoid the crash DRAC = (v - v_lead)^2 / (2 gap) >= "
        f"{cfg.drac_near_miss} m/s^2), serious (near miss with TTC < 1.0 s), critical (DRAC > b_emerg), collisions (bodies overlapping), and "
        "zone_pet events (post-encroachment time between vehicles of different approaches at a box conflict zone, <= 5 s; serious below "
        f"{cfg.pet_conflict_s} s). Collisions and critical events are safety invariants (counted in n_invariant_violations).",
        f"- {_safety_text(episodes)}",
        "- The drivers have no reaction time or perception errors in car following, so these counts measure geometry and signal-control "
        "conflicts, not human crash risk.", "",
        "## Label",
        "- `y_mp_action` = the priority-aware teacher `priority_mp` recomputed from the row's own columns: the max-pressure request "
        "(pressure_a = n_a - mean of down_n over a's unblocked exits, hysteresis 2 vehicles, min/max green rules) unless an ambulance on "
        "approach a has ev_eta_a <= allred_min + start-up lost time + ev_q_ahead_a x saturation headway, in which case a. "
        "It is the request, not the applied phase.",
        "- `y_q_total_t30` = q_total of the junction 30 s later from junction_1hz; NaN on tail/cooldown rows.", "",
        "## Episode mix and KPIs", "",
        _md(episodes.groupby(["demand_level", "controller"]).size().rename("episodes").to_frame()), "",
        _md(episodes.groupby("demand_level")[["mean_total_delay_s", "p90_total_delay_s", "throughput_vph", "n_red_commits", "n_lane_changes",
                                              "n_near_misses", "gridlock"]].mean(numeric_only=True).round(2)), "",
        "Ambulance KPIs (episodes with ev_rate = default):", "",
        _md(episodes[episodes.lambda_ev_vph > 0].groupby("controller")[["n_ambulances", "ev_mean_delay_s", "ev_mean_stops", "ev_mean_blocked_s",
            "ev_mean_red_s", "car_mean_delay_penalty_s", "recovery_time_s"]].mean(numeric_only=True).round(2)), "",
        "## Label balance (main-period rows)", "",
        _md(main.groupby("controller_name")["y_mp_action"].value_counts(normalize=True).unstack(fill_value=0).round(3)), "",
        "switch share (y_mp_action != phase) per controller:",
        _md(main.assign(switch=main.y_mp_action != main.phase).groupby("controller_name")["switch"].mean().round(3).to_frame()), "",
        "## Known limitations and modelling simplifications",
        "- Poisson (per-step Bernoulli) arrivals; ACC drivers without reaction time or acceleration noise in car following (the red-onset "
        "decision has a 1 s perception-reaction time); no pedestrians.",
        "- Signal displays show only red and green (a design decision): the internal all-red is the clearance interval, so vehicles that "
        "could not stop safely cross the stop line during the first seconds of red. That is counted in n_red_commits and is safe by "
        "construction (the next green waits for them), but it is red-light running in the strict sense; an amber display would label it.",
        "- Passenger cars and ambulances only in the release. A vehicle_mix option (vans, buses) exists but is off: count features would change "
        "meaning and a 12 m bus cannot follow the kerb-lane arc without swinging wide. No motorcycles: in Indonesia most vehicles are "
        "motorcycles, which filter between lanes and need a sub-lane model.",
        "- Lane changes are instantaneous in the lane lists; the 3 s sideways move is represented by the old-lane follower keeping the changer "
        "as a leader and by the viewer's slide. A turn may end in the far exit lane ('turn, then move over' compressed into the box); this "
        "is conflict-free under split phasing because only one approach is green.",
        "- max_pressure is implemented exactly as specified (pressure = n_a - mean(down_n over unblocked exits), hysteresis 2 vehicles, "
        "min green 5 s). It switches far more often than fixed_time (switches_per_min in the table below) and every switch costs about 5.5 s "
        f"of lost time, so it can do worse than fixed_time at high and stress demand{_paired_measured()}. "
        "The label y_mp_action is this teacher; treat it as 'a strong parameter-light heuristic', not as the optimum.",
        f"- Demand levels are v/c against the fixed-time capacity REALISED by the simulated junction ({cfg.capacity_vph:.0f} veh/h per approach, "
        f"measured in a saturated probe episode), not against the formula value sat_flow x lanes x 18/96 ({cfg.capacity_formula_vph:.0f} veh/h): "
        f"the junction realises {cfg.capacity_realisation:.0%} of the formula capacity (turning vehicles slow to the curvature limit of their arc, "
        "clearance waits for conflict zones). The levels are NOT guaranteed to be sub-capacity for every controller on the 2x2 grid. Measured "
        "means per level and controller (n_in_network_main_end = vehicles still in the network when the main period ends, i.e. where queues kept growing):",
        "", _load_table(episodes), "",
        "- `n_hard_brakes` counts vehicle-steps with acceleration below -3 m/s^2 while moving (kept from v1); n_harsh_brake_events counts braking "
        f"runs at >= 3.4 m/s^2{_hard_brake_measured()}. Most are vehicles stopping for a red that appeared at short distance.",
        "- Not included as features (clock proxies / leakage): throughput_cum, n_switches, t_frac, time of day, oracle_*; junction_id is a key only. "
        "No safety or lane-change column is in signal_ticks.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------- main
def generate(cfg: SimConfig, n_episodes: int, out: str, start_id: int = 0, trajectories=(), verbose: bool = True,
             paired: bool = True) -> dict:
    os.makedirs(out, exist_ok=True)
    if not cfg.lambda_base_vph:
        calib = calibrate_demand(cfg)
        if verbose:
            print(f"calibrated: sat flow {calib['measured_sat_flow_vphpl']:.0f} veh/h/lane, capacity per approach {calib['capacity_vph']:.0f} veh/h "
                  f"(formula {calib['capacity_formula_vph']:.0f}, realisation {calib['capacity_realisation']:.3f}), lambda_base {calib['lambda_base_vph']}")
    scenarios = sample_scenarios(cfg, n_episodes, start_id)
    ep_rows, veh_rows, tick_rows, j1_rows, conf_rows = [], [], [], [], []
    t0 = time.perf_counter()
    for k, sc in enumerate(scenarios):
        eid = start_id + k
        seed = cfg.seed_base + eid
        w = World(cfg, sc, seed=seed, episode_id=eid, log_trajectories=eid in set(trajectories)).run()
        ep = w.episode_row()
        if paired and sc.lambda_ev_vph > 0:
            # same seed and controller without ambulances: common random numbers for the cars
            sc0 = Scenario(**{**sc.__dict__, "lambda_ev_vph": 0.0, "ev_rate": "0"})
            ep0 = World(cfg, sc0, seed=seed, episode_id=eid).run().episode_row()
            ep["car_mean_delay_penalty_s"] = round(ep["mean_total_delay_s"] - ep0["mean_total_delay_s"], 3)
        ep_rows.append(ep)
        veh_rows += w.vehicle_rows()
        ticks = w.signal_tick_rows()
        tick_rows += ticks
        j1_rows += w.junction_1hz_rows
        conf_rows += w.conflict_rows()
        if w.log_trajectories:
            pd.DataFrame(w.state_rows).to_csv(os.path.join(out, f"vehicle_states_{eid}.csv"), index=False)
        if verbose:
            r = ep_rows[-1]
            print(f"episode {eid:4d} {sc.demand_level:6s} {sc.controller:16s} {sc.demand_asym:8s} {sc.od_pattern:7s} "
                  f"noise {sc.sensor_noise_level:<3} ev {sc.ev_rate:7s} split {sc.split:8s} delay {r['mean_total_delay_s']:6.1f} s "
                  f"ev_delay {r['ev_mean_delay_s']:6.1f} penalty {r['car_mean_delay_penalty_s']:6.1f} "
                  f"ticks {len(ticks)} veh {r['n_spawned']} gridlock {r['gridlock']} inv {r['n_invariant_violations']} "
                  f"runtime {r['runtime_s']:.1f} s")
    episodes = pd.DataFrame(ep_rows)
    vehicles = pd.DataFrame(veh_rows)
    ticks = pd.DataFrame(tick_rows, columns=SIGNAL_TICK_COLUMNS)
    j1 = pd.DataFrame(j1_rows)
    episodes.to_csv(os.path.join(out, "episodes.csv"), index=False)
    vehicles.to_csv(os.path.join(out, "vehicles.csv"), index=False)
    ticks.to_csv(os.path.join(out, "signal_ticks.csv"), index=False)
    j1.to_csv(os.path.join(out, "junction_1hz.csv"), index=False)
    conflicts = pd.DataFrame(conf_rows, columns=CONFLICT_COLUMNS)
    conflicts.to_csv(os.path.join(out, "conflicts.csv"), index=False)
    net = build_grid(cfg)
    pd.DataFrame(network_edges_table(net)).to_csv(os.path.join(out, "network_edges.csv"), index=False)
    pd.DataFrame(network_paths_table(net)).to_csv(os.path.join(out, "network_paths.csv"), index=False)
    pd.DataFrame(data_dictionary_rows()).to_csv(os.path.join(out, "data_dictionary.csv"), index=False)
    with open(os.path.join(out, "config.json"), "w") as f:
        f.write(cfg.to_json())
    ticks_card = ticks.merge(episodes[["episode_id", "controller"]].rename(columns={"controller": "controller_name"}), on="episode_id")
    write_datacard(os.path.join(out, "DATACARD.md"), cfg, episodes, ticks_card, vehicles)
    summary = dict(out=out, n_episodes=len(episodes), rows=dict(episodes=len(episodes), vehicles=len(vehicles),
                   signal_ticks=len(ticks), junction_1hz=len(j1), conflicts=len(conflicts)), columns=dict(signal_ticks=len(ticks.columns),
                   vehicles=len(vehicles.columns), episodes=len(episodes.columns)),
                   total_runtime_s=time.perf_counter() - t0)
    if verbose:
        print(f"wrote {summary['rows']} rows to {out}/ in {summary['total_runtime_s']:.0f} s")
    return summary


def rewrite_docs(out: str) -> None:
    """Rewrite data_dictionary.csv and DATACARD.md of an existing release from its own CSV files and
    config.json, without simulating (text and computed sentences change, the data do not)."""
    cfg = load_config(os.path.join(out, "config.json"))
    episodes = pd.read_csv(os.path.join(out, "episodes.csv"))
    ticks = pd.read_csv(os.path.join(out, "signal_ticks.csv"),
                        usecols=["episode_id", "is_warmup", "is_cooldown", "y_mp_action", "phase"])
    n_veh = pd.read_csv(os.path.join(out, "vehicles.csv"), usecols=["episode_id"])
    ticks = ticks.merge(episodes[["episode_id", "controller"]].rename(columns={"controller": "controller_name"}), on="episode_id")
    pd.DataFrame(data_dictionary_rows()).to_csv(os.path.join(out, "data_dictionary.csv"), index=False)
    write_datacard(os.path.join(out, "DATACARD.md"), cfg, episodes, ticks, n_veh)
    print(f"rewrote data_dictionary.csv and DATACARD.md in {out}/ (generator {GENERATOR_VERSION}, config_hash {cfg.config_hash()})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--out", default="data/dev10")
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--trajectories", default="", help="comma-separated episode ids to log vehicle_states_<id>.csv")
    p.add_argument("--no-paired", action="store_true", help="skip the paired no-ambulance run (car_mean_delay_penalty_s = NaN)")
    p.add_argument("--rewrite-docs", metavar="DIR", help="rewrite data_dictionary.csv and DATACARD.md of the release in DIR without simulating")
    args = p.parse_args(argv)
    if args.rewrite_docs:
        rewrite_docs(args.rewrite_docs)
        return
    traj = [int(x) for x in args.trajectories.split(",") if x.strip()]
    generate(SimConfig(), args.episodes, args.out, args.start_id, traj, paired=not args.no_paired)


if __name__ == "__main__":
    main()
