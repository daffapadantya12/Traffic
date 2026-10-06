"""build_features(world, junction) -> one signal_ticks row (DECISIONS §7).

Traffic-engineering idea: a signal controller can only see counts on its own
approach edges, how full its exit links are, and what its neighbours show.
The same function feeds the controllers, the dataset logger and, later, the
digital twin, so train and serve rows always have the same columns.
"""

from __future__ import annotations

import math

from .network import DIRS, OPPOSITE, exit_lane_blocked
from .signals import GREEN

# ---- the column contract -------------------------------------------------------------
KEY_COLUMNS = ["episode_id", "junction_id", "t", "tick_idx", "is_warmup", "is_cooldown", "is_tail"]
SIGNAL_COLUMNS = ["phase", "time_in_phase_s", "max_green_remaining_s"]
APPROACH_COLUMNS = []
for _a in DIRS:
    APPROACH_COLUMNS += [f"q_{_a}", f"n_{_a}", f"q_{_a}_l0", f"q_{_a}_l1", f"arr_{_a}_30s"]
DOWNSTREAM_COLUMNS = []
for _d in DIRS:
    DOWNSTREAM_COLUMNS += [f"down_n_{_d}", f"down_occ_{_d}", f"down_blocked_{_d}"]
NEIGHBOUR_COLUMNS = []
for _d in DIRS:
    NEIGHBOUR_COLUMNS += [f"nb_{_d}_exists", f"nb_{_d}_phase", f"nb_{_d}_green_toward_me"]
SENSOR_COLUMNS = [f"obs_q_{_a}" for _a in DIRS]
EV_COLUMNS = []                                   # ambulance channel (DECISIONS §10, GPS-style, no obs noise)
for _a in DIRS:
    EV_COLUMNS += [f"ev_present_{_a}", f"ev_dist_{_a}", f"ev_eta_{_a}", f"ev_q_ahead_{_a}"]
ACTION_COLUMNS = ["action_requested", "action_applied", "forced", "switched"]
LABEL_COLUMNS = ["y_mp_action", "y_q_total_t30"]

FEATURE_COLUMNS = (KEY_COLUMNS + SIGNAL_COLUMNS + APPROACH_COLUMNS + DOWNSTREAM_COLUMNS + NEIGHBOUR_COLUMNS
                   + SENSOR_COLUMNS + EV_COLUMNS)
SIGNAL_TICK_COLUMNS = FEATURE_COLUMNS + ACTION_COLUMNS + LABEL_COLUMNS


def build_features(world, junction) -> dict:
    """Observable state of one junction at the world's current time.

    Counts come straight from the lane lists; nothing is hard-coded.  Queue
    columns are normalised by the storage of their own edge (down_occ_*).
    """
    cfg = world.cfg
    net = world.net
    plant = world.plants[junction.id]
    t = world.t
    row = {
        "episode_id": world.episode_id,
        "junction_id": junction.id,
        "t": t,
        "tick_idx": world.tick_counts[junction.id],
        "is_warmup": cfg.is_warmup(t),
        "is_cooldown": cfg.is_cooldown(t),
        "is_tail": cfg.is_tail(t),
        "phase": plant.phase,
        "time_in_phase_s": plant.time_in_phase,
        "max_green_remaining_s": plant.max_green_remaining(),
    }
    # per approach a: vehicles on the edge that arrives from arm a
    for a in DIRS:
        edge = net.edges[junction.approach_edges[a]]
        q_lanes = [sum(1 for v in lane if v.v < cfg.queue_speed) for lane in edge.lanes]
        row[f"q_{a}"] = sum(q_lanes)
        row[f"n_{a}"] = edge.n_vehicles()
        row[f"q_{a}_l0"] = q_lanes[0] if edge.n_lanes > 1 else -1
        row[f"q_{a}_l1"] = q_lanes[1] if edge.n_lanes > 1 else -1
        prune_entry_times(edge, t, cfg)
        row[f"arr_{a}_30s"] = sum(1 for te in edge.entry_times if t - cfg.arr_window_s < te <= t)
    # downstream per exit direction d
    for d in DIRS:
        edge = net.edges[junction.exit_edges[d]]
        n = edge.n_vehicles()
        row[f"down_n_{d}"] = n
        row[f"down_occ_{d}"] = n / edge.storage
        row[f"down_blocked_{d}"] = any(exit_lane_blocked(edge, k, cfg) for k in range(edge.n_lanes))
    # neighbours
    for d in DIRS:
        nb = junction.neighbours[d]
        if nb < 0:
            row[f"nb_{d}_exists"] = False
            row[f"nb_{d}_phase"] = -1
            row[f"nb_{d}_green_toward_me"] = False
        else:
            nb_plant = world.plants[nb]
            row[f"nb_{d}_exists"] = True
            row[f"nb_{d}_phase"] = nb_plant.phase
            # under split phasing every green approach except the one coming from me
            # has a movement onto the link toward me
            facing_me = DIRS.index(OPPOSITE[d])
            row[f"nb_{d}_green_toward_me"] = (nb_plant.state == GREEN and nb_plant.phase != facing_me)
    # ambulance channel: the nearest ambulance on the approach edge (within ev_horizon_m)
    for a in DIRS:
        edge = net.edges[junction.approach_edges[a]]
        best = None
        for lane in edge.lanes:
            for v in lane:
                if v.is_ambulance and edge.stop_s - v.s <= cfg.ev_horizon_m:
                    if best is None or v.s > best.s:
                        best = v
        if best is None:
            row[f"ev_present_{a}"] = False
            row[f"ev_dist_{a}"] = -1.0
            row[f"ev_eta_{a}"] = -1.0
            row[f"ev_q_ahead_{a}"] = -1
        else:
            dist = max(edge.stop_s - best.s, 0.0)
            row[f"ev_present_{a}"] = True
            row[f"ev_dist_{a}"] = round(dist, 2)
            row[f"ev_eta_{a}"] = round(dist / max(best.v, cfg.ev_eta_min_speed), 2)
            row[f"ev_q_ahead_{a}"] = sum(1 for v in edge.lanes[best.lane] if v.s > best.s and v.v < cfg.queue_speed)
    # sensor copy
    p_miss = world.sensor_noise_level
    for a in DIRS:
        q = row[f"q_{a}"]
        if p_miss <= 0.0:
            row[f"obs_q_{a}"] = q
        else:
            rng = world.rng_sensor
            row[f"obs_q_{a}"] = int(rng.binomial(q, 1.0 - p_miss) + rng.poisson(p_miss / 2.0))
    return row


def prune_entry_times(edge, t: float, cfg) -> None:
    """Forget edge entry timestamps older than the arrival window (keeps lists short)."""
    cutoff = t - cfg.arr_window_s
    while edge.entry_times and edge.entry_times[0] <= cutoff:
        edge.entry_times.pop(0)


def empty_labels() -> dict:
    return {"y_mp_action": -1, "y_q_total_t30": math.nan}
