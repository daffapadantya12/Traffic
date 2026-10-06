"""The World: one episode of the grid, vehicles and signals (DECISIONS §2-§7, §12).

Traffic-engineering idea: a microscopic simulation moves every vehicle every
dt with a car-following law, lets drivers change lane on the links, lets the
signal plants gate the junction boxes, and logs what a controller could observe.
Everything logged here is measured from the simulated state; nothing is hard-coded.

v2.0 (DECISIONS §12): ACC car following (no over-braking at red onset or cut-ins),
MOBIL + mandatory lane changes with cooperative gap opening and a reroute when a
vehicle misses its lane, cars making room for ambulances where there is space, lane
discipline on real turning arcs through a larger box, a geometric conflict-matrix
clearance gate, and a near-miss monitor (TTC, DRAC, PET, collisions).
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass

import numpy as np

from .config import GENERATOR_VERSION, SimConfig
from .controllers import controller_eps, label_request, make_controller
from .features import LABEL_COLUMNS, SIGNAL_TICK_COLUMNS, build_features, prune_entry_times
from .network import (DIRS, DIR_INDEX, MOVEMENTS, build_grid, choose_lane, lanes_for_movement, od_row,
                      route_from_edge, shortest_route)
from .safety import SafetyMonitor
from .signals import ALLRED, FORCED_STARVATION, GREEN, SignalPlant
from .vehicles import (COMMITTED, FOLLOWING, FREE, IN_BOX, STOPPING, Vehicle, acc_accel, ambulance_params,
                       ballistic_step, free_flow_time, sample_params)


@dataclass
class Scenario:
    """What varies between episodes (DECISIONS §7 scenario mixture)."""
    demand_level: str = "design"
    lambda_base_vph: float = 150.0      # arrivals per entry node per hour
    demand_asym: str = "sym"            # "sym" or "ns_heavy"
    od_pattern: str = "default"         # "default" or "uniform"
    controller: str = "max_pressure"
    controller_eps: float = 0.0
    sensor_noise_level: float = 0.0
    split: str = "train"
    lambda_ev_vph: float = 0.0          # ambulances per hour per entry node (0 = none)
    ev_rate: str = "0"                  # "0" or "default" (scenario factor of DECISIONS §10)


class World:
    """One episode.  Create, then call run(); results are in the *_rows lists."""

    def __init__(self, cfg: SimConfig, scenario: Scenario, seed: int, episode_id: int = 0,
                 log_trajectories: bool | None = None):
        self.cfg = cfg
        self.scenario = scenario
        self.seed = seed
        self.episode_id = episode_id
        self.log_trajectories = cfg.log_trajectories if log_trajectories is None else log_trajectories
        self.net = build_grid(cfg)
        # five independent random streams (DECISIONS §5 + §10): cars never touch the ev stream
        streams = np.random.SeedSequence(seed).spawn(5)
        self.rng_demand = np.random.default_rng(streams[0])
        self.rng_vehicles = np.random.default_rng(streams[1])
        self.rng_controller = np.random.default_rng(streams[2])
        self.rng_sensor = np.random.default_rng(streams[3])
        self.rng_ev = np.random.default_rng(streams[4])
        self.lambda_ev_per_s = scenario.lambda_ev_vph / 3600.0
        self.n_ambulances = 0
        self.ev_events = {}                    # (veh_id, junction_id) -> dict(t_first, q0, t_clear)
        self.sensor_noise_level = scenario.sensor_noise_level
        self.controller = make_controller(scenario.controller, cfg, self.rng_controller,
                                          eps=scenario.controller_eps if scenario.controller == "max_pressure_eps" else None)
        self.plants = [SignalPlant(cfg, j.id) for j in self.net.junctions]
        self.t = 0.0
        self.step_idx = 0
        self.n_steps = int(round(cfg.episode_length_s / cfg.dt))
        # demand
        self.lambda_per_s = {}
        for node in self.net.boundary_nodes:
            axis = self.net.nodes[node].road_axis
            asym = 1.0
            if scenario.demand_asym == "ns_heavy":
                asym = cfg.asym_ns_factor if axis == "NS" else cfg.asym_ew_factor
            self.lambda_per_s[node] = scenario.lambda_base_vph * asym / 3600.0
        self.od = {}
        for node in self.net.boundary_nodes:
            row = od_row(self.net, node, scenario.od_pattern, cfg)
            self.od[node] = (list(row.keys()), np.array(list(row.values())))
        self.entry_queues = {node: collections.deque() for node in self.net.boundary_nodes}
        # vehicles
        self.next_vid = 0
        self.active = []                       # vehicles on edges or connectors
        self.finished = []                     # vehicles that despawned
        self.n_spawn_requests = 0
        # counters (health metrics and invariants)
        self.vehicle_steps = 0
        self.n_overlap_fixed = 0
        self.n_hard_brakes = 0
        self.n_emergency_brakes = 0            # IDM demanded less than -b_emerg while moving (clipped)
        self.n_red_commits = 0
        self.n_commit_revoked = 0              # COMMITTED vehicles that came to rest before the line -> STOPPING
        self.n_box_blocked = 0
        self.n_turn_overspeed = 0              # entered a turning path faster than its curvature limit + 0.5 m/s
        # lane changing (§12.3)
        self.n_lane_changes = 0
        self.n_lc_mandatory = 0                # to reach the lane the next movement needs
        self.n_lc_discretionary = 0            # MOBIL: faster lane / letting others pass
        self.n_lc_yield = 0                    # cars moving out of an ambulance's lane
        self.n_yield_blocked = 0               # (car, ambulance) pairs where the car had no room to move aside
        self.n_ev_overtakes = 0                # ambulance lane changes to pass slower traffic
        self.n_reroutes = 0                    # missed its lane before the solid line: took another movement
        self.coop = {}                         # follower id -> vehicle it opens a gap for (this step)
        self.recent_lc = []                    # vehicles still sliding across after a lane change
        self._shadows = {}                     # (edge id, lane it left) -> those vehicles
        self._amb_lanes = {}
        self.n_brake_sat_vehicle = 0           # runs where the car-following demand exceeded b_emerg behind a vehicle ...
        self.n_brake_sat_virtual = 0           # ... or behind a stop line (both clipped; targets 0)
        self.n_harsh_brake_events = 0          # runs of applied deceleration >= 3.4 m/s^2
        self.lane_discipline_violations = 0    # entered a box path from a lane that does not serve it (invariant)
        self.n_destination_changes = 0         # rerouted vehicles whose lane only led off the network elsewhere
        self.n_pending_repicks = 0             # all-red stalls resolved by serving another (clear) approach
        self.n_in_network_main_end = None      # load when the main period ends (spawning stops)
        self.n_entry_queued_main_end = None
        self.decel_limit_violations = 0        # speed fell faster than b_emerg in one step (invariant)
        self.safety = SafetyMonitor(self)
        self.red_violations = 0
        self.box_clear_violations = 0
        self.committed_at_onset_violations = 0  # green onset while a COMMITTED vehicle of another approach is still upstream
        self.min_green_violations = 0
        self.allred_short_violations = 0
        self.two_green_violations = 0
        self.crossings_cum = {j.id: 0 for j in self.net.junctions}
        self.exits_main = 0
        self.last_crossing_t = 0.0
        self.last_crossing_t_j = {j.id: 0.0 for j in self.net.junctions}
        self.gridlock = False
        self.gridlock_t = math.nan
        self.gridlock_junction = -1
        # starvation bookkeeping: (junction, approach) -> time since when vehicles have waited at red
        self.red_wait_start = {(j.id, a): None for j in self.net.junctions for a in DIRS}
        self.max_approach_red_s = 0.0
        # lane-level liveness (amendment §11.8): longest standing lane head, and the per-junction
        # spillback lock (an occupied approach lane with no crossing for gridlock_window_s although
        # the approach was green for >= 2 full green periods' worth in that window)
        self.max_head_wait_s = 0.0
        lane_keys = [(e.id, k) for e in self.net.edges.values() if e.junction_at_dest >= 0 for k in range(e.n_lanes)]
        self.lane_last_cross_t = {key: 0.0 for key in lane_keys}
        self.lane_occupied_since = {key: None for key in lane_keys}
        self.lane_locked = {key: False for key in lane_keys}
        self.green_periods = {(j.id, a): collections.deque() for j in self.net.junctions for a in DIRS}  # completed (onset, end)
        self.n_spillback_locks = 0
        self.spillback_lock_t = math.nan
        self.spillback_lock_junction = -1
        # logs
        self.tick_counts = {j.id: 0 for j in self.net.junctions}
        self.signal_rows = []
        self.junction_1hz_rows = []
        self.q_total_lookup = {}               # (junction_id, t) -> q_total
        self.state_rows = []
        self.red_commit_log = []               # (t, junction, d, v)
        self.runtime_s = math.nan

    # ------------------------------------------------------------------ helpers
    def plant_for_edge(self, edge):
        return self.plants[edge.junction_at_dest]

    def approach_green(self, edge) -> bool:
        return self.plant_for_edge(edge).is_green(DIR_INDEX[edge.approach_dir])

    def n_in_network(self) -> int:
        return len(self.active)

    def n_entry_queued(self) -> int:
        return sum(len(q) for q in self.entry_queues.values())

    def compute_needs(self, veh: Vehicle) -> list:
        """Per route edge, the lane the vehicle must be in so that it can make every remaining
        movement without changing lanes (DECISIONS §12.3): NS -> kerb lane 0, FS -> the median
        lane, T -> whatever the next junction needs; None when only through movements to the
        boundary remain (then any lane will do)."""
        n = self.cfg.lanes_per_dir
        moves = veh.route_moves
        needs = []
        for i in range(len(veh.route_edges)):
            need = None
            if n > 1:
                for j in range(i, len(moves)):
                    m = moves[j][1]
                    if m == "NS":
                        need = 0
                        break
                    if m == "FS":
                        need = n - 1
                        break
            needs.append(need)
        return needs

    def needs_of(self, veh: Vehicle) -> list:
        """The vehicle's per-edge lane needs, computed on first use for vehicles placed by hand."""
        if len(veh.needs) != len(veh.route_edges):
            veh.needs = self.compute_needs(veh)
        return veh.needs

    def path_for(self, veh: Vehicle, edge, lane: int | None = None):
        """The box path the vehicle takes from `lane` (default: its lane) at the end of `edge`.
        Through traffic keeps its lane; a turn ends in the lane its next turn needs, else in the
        emptier exit lane, chosen once per edge.  A vehicle in a lane that does not serve its
        movement (it moved aside for an ambulance) looks ahead along that lane's through path."""
        lane = veh.lane if lane is None else lane
        J = self.net.junctions[edge.junction_at_dest]
        m = veh.next_movement()
        n = edge.n_lanes
        if m == "T" or lane not in lanes_for_movement(m, n):
            return J.connectors[(edge.approach_dir, "T", lane, lane)]
        needs = self.needs_of(veh)
        out = needs[veh.edge_idx + 1] if veh.edge_idx + 1 < len(needs) else None
        if out is None:
            if veh.plan_edge_idx != veh.edge_idx:
                c0 = J.connectors[(edge.approach_dir, m, lane, 0)]
                out_edge = self.net.edges[c0.out_edge]
                counts = [len(l) for l in out_edge.lanes]
                for c in J.connectors.values():
                    if c.out_edge == out_edge.id:
                        counts[c.out_lane] += len(c.vehicles)
                veh.plan_out_lane = min(range(n), key=lambda k: (counts[k], k))
                veh.plan_edge_idx = veh.edge_idx
            out = veh.plan_out_lane
        return J.connectors[(edge.approach_dir, m, lane, out)]

    # ------------------------------------------------------------------ episode loop
    def run(self):
        t_start = time.perf_counter()
        for _ in range(self.n_steps):
            self.step()
            if self.gridlock:
                break
        self.log_1hz()                               # final state
        if self.n_in_network_main_end is None:       # ended (gridlock) before the main period did
            self.n_in_network_main_end = self.n_in_network()
            self.n_entry_queued_main_end = self.n_entry_queued()
        self.finish_episode()
        self.runtime_s = time.perf_counter() - t_start
        return self

    def ambulance_behind_on(self, veh: Vehicle, edge) -> bool:
        """An ambulance is behind `veh` on this link, in any lane, within siren range."""
        r = self.cfg.ev_siren_range_m
        return any(a.is_ambulance and a.s < veh.s and veh.s - a.s <= r for lane in edge.lanes for a in lane)

    def step(self):
        t = self.t
        if self.n_in_network_main_end is None and t >= self.cfg.spawn_end_s - 1e-9:
            self.n_in_network_main_end = self.n_in_network()
            self.n_entry_queued_main_end = self.n_entry_queued()
        if abs(t - round(t)) < 1e-9:
            self.log_1hz()
        self.update_signals()
        self.spawn_vehicles()
        self.change_lanes()
        self.compute_accelerations()
        self.move_vehicles()
        self.overlap_guard()
        self.bookkeeping()
        for p in self.plants:
            p.advance(self.cfg.dt)
        self.t = round(t + self.cfg.dt, 6)
        self.step_idx += 1
        self.safety.end_step(self.t)
        self.check_gridlock()

    # ------------------------------------------------------------------ signals
    def update_signals(self):
        t = self.t
        for J in self.net.junctions:
            plant = self.plants[J.id]
            if plant.state == ALLRED:
                # clearance gate (DECISIONS §12.1, generalising §4 + §11.2): no vehicle of another approach
                # may still be inside a conflict zone of the pending approach (rear bumpers included),
                # and no vehicle latched COMMITTED at the last red onset may still be upstream of the box.
                # A committed vehicle that has come to rest before the line evidently can stop: it is
                # re-latched STOPPING so a stalled commit cannot freeze the junction.
                self.revoke_stopped_commits(J)
                if (t - plant.allred_start_t > self.cfg.allred_min + self.cfg.allred_stall_s
                        and not self.conflicts_clear(J, plant.pending_phase)):
                    # a vehicle stuck in a conflict zone of the pending approach (its exit lane is full) must not hold
                    # the whole junction in all-red: serve an approach whose zones are clear (its own is always clear)
                    n_app = [self.net.edges[J.approach_edges[a]].n_vehicles() for a in DIRS]
                    cands = [i for i in range(len(DIRS)) if self.conflicts_clear(J, i)]
                    if cands:
                        plant.pending_phase = max(cands, key=lambda i: (n_app[i], -i))
                        self.n_pending_repicks += 1
                if plant.try_green_onset(t, self.conflicts_clear(J, plant.pending_phase)):
                    # independent re-check of the invariants after the onset
                    if not self.zones_clear(J, plant.phase):
                        self.box_clear_violations += 1
                    if self.committed_on_approaches(J, exclude=DIRS[plant.phase]):
                        self.committed_at_onset_violations += 1
                    if plant.allred_durations[-1] + 1e-9 < self.cfg.allred_min:
                        self.allred_short_violations += 1
                    self.clear_latches(J, plant.phase)
            if plant.at_decision_instant(t):
                self.decide(J, plant)
        # invariant (1): never two approaches green at one junction is structural
        # (one phase per plant); count anything that breaks the structure anyway
        for J in self.net.junctions:
            greens = sum(1 for a in DIRS if self.plants[J.id].is_green(DIR_INDEX[a]))
            if greens > 1:
                self.two_green_violations += 1

    def conflicts_clear(self, J, phase_idx: int) -> bool:
        """May the pending approach go green?  (DECISIONS §12.1)"""
        return self.zones_clear(J, phase_idx) and not self.committed_on_approaches(J, exclude=DIRS[phase_idx])

    def zones_clear(self, J, phase_idx: int) -> bool:
        """No body of another approach is still in a zone that conflicts with a path of the
        approach: its rear must be past s_clear(path, approach) on its own path."""
        B = DIRS[phase_idx]
        s_clear = self.net.s_clear
        for veh, conn, s_front in self.box_bodies(J):
            if conn.approach == B:
                continue
            lim = s_clear.get((conn.key, B))
            if lim is not None and s_front - veh.l_veh < lim:
                return False
        return True

    def box_bodies(self, J):
        """(vehicle, path, front position on the path) for every body inside junction J's box:
        vehicles on its connectors and vehicles whose rear is still in it after leaving."""
        for c in J.connectors.values():
            for v in c.vehicles:
                yield v, c, v.s
        for d in DIRS:
            e = self.net.edges[J.exit_edges[d]]
            for lane in e.lanes:
                for v in reversed(lane):
                    if v.s >= v.l_veh:
                        break
                    if v.last_conn is not None and v.last_conn.junction_id == J.id:
                        yield v, v.last_conn, v.last_conn.length + v.s

    def decide(self, J, plant):
        """One decision instant: build the row, ask the controller, apply, log."""
        row = build_features(self, J)
        requested = int(self.controller.act(row))
        n_by_approach = [row[f"n_{a}"] for a in DIRS]
        tip_before = plant.time_in_phase
        applied, forced, switched, red_onset = plant.request(requested, n_by_approach, self.t)
        if switched and tip_before + 1e-9 < self.cfg.min_green:
            self.min_green_violations += 1
        if red_onset:
            self.stop_or_commit(J, plant.phase)
            self.green_periods[(J.id, DIRS[plant.phase])].append((plant.green_onset_t, self.t))
        row.update(action_requested=requested, action_applied=applied, forced=forced, switched=switched)
        for c in LABEL_COLUMNS:
            row[c] = math.nan
        self.signal_rows.append(row)
        self.tick_counts[J.id] += 1

    def clear_latches(self, J, phase_idx: int):
        """Green onset: vehicles on the newly green approach may cross again."""
        edge = self.net.edges[J.approach_edges[DIRS[phase_idx]]]
        for lane in edge.lanes:
            for v in lane:
                v.latch = None

    def committed_on_approaches(self, J, exclude: str | None = None) -> bool:
        """Is any vehicle latched COMMITTED still on an approach edge of J (i.e. not yet in the box)?"""
        for a in DIRS:
            if a == exclude:
                continue
            for lane in self.net.edges[J.approach_edges[a]].lanes:
                for v in lane:
                    if v.latch == COMMITTED:
                        return True
        return False

    def revoke_stopped_commits(self, J) -> None:
        """A COMMITTED vehicle that has come to rest (v < stop_speed) before the stop line
        can stop after all: re-latch it STOPPING so the stop line is its leader again."""
        cfg = self.cfg
        for a in DIRS:
            edge = self.net.edges[J.approach_edges[a]]
            for lane in edge.lanes:
                for v in lane:
                    if v.latch == COMMITTED and v.v < cfg.stop_speed and v.s < edge.stop_s:
                        v.latch = STOPPING
                        self.n_commit_revoked += 1

    def stop_or_commit(self, J, phase_idx: int):
        """Red onset (DECISIONS §5, §12.2): every vehicle on the approach edge is latched STOPPING
        if it can stop at b_stop after a perception-reaction time t_pr (ITE dilemma-zone rule,
        d >= v t_pr + v^2 / (2 b_stop)), else COMMITTED (it crosses during the all-red)."""
        cfg = self.cfg
        edge = self.net.edges[J.approach_edges[DIRS[phase_idx]]]
        for lane in edge.lanes:
            for v in lane:                       # lists are front-vehicle-first
                d = edge.stop_s - v.s
                if d >= v.v * cfg.t_pr_s + v.v * v.v / (2.0 * cfg.b_stop):
                    v.latch = STOPPING
                else:
                    v.latch = COMMITTED
                    v.n_red_commits += 1
                    self.n_red_commits += 1
                    self.red_commit_log.append((self.t, J.id, d, v.v))

    # ------------------------------------------------------------------ spawning
    def spawn_vehicles(self):
        cfg = self.cfg
        t = self.t
        spawning_on = t < cfg.spawn_end_s
        mult = cfg.demand_multiplier(t)                # daily profile (cars only); 1.0 = flat, bit-identical
        for node in self.net.boundary_nodes:           # fixed order -> common random numbers
            p_car = self.lambda_per_s[node] * cfg.dt
            if mult != 1.0:
                p_car *= mult
            if spawning_on and self.rng_demand.random() < p_car:
                self.entry_queues[node].append(self.new_request(node, t))
            if spawning_on and self.lambda_ev_per_s > 0 and self.rng_ev.random() < self.lambda_ev_per_s * cfg.dt:
                self.entry_queues[node].append(self.new_request(node, t, ambulance=True))
            q = self.entry_queues[node]
            if q and self.try_insert(q[0], node):
                q.popleft()

    def new_request(self, origin: str, t: float, ambulance: bool = False) -> Vehicle:
        """Draw destination, driver parameters and route at REQUEST time.
        Ambulances use the ev stream for everything so car draws are unchanged."""
        dests, probs = self.od[origin]
        if ambulance:
            destination = dests[self.rng_ev.choice(len(dests), p=probs)]
            params = ambulance_params(self.cfg)
            edges, moves = shortest_route(self.net, origin, destination, self.cfg, self.rng_ev)
            self.n_ambulances += 1
        else:
            destination = dests[self.rng_demand.choice(len(dests), p=probs)]
            params = sample_params(self.rng_vehicles, self.cfg)
            edges, moves = shortest_route(self.net, origin, destination, self.cfg, self.rng_demand)
        veh = Vehicle(self.next_vid, params, origin, destination, edges, moves, t)
        self.next_vid += 1
        self.n_spawn_requests += 1
        veh.needs = self.compute_needs(veh)
        veh.route_len = self.route_length(edges, moves)
        veh.free_flow_time = free_flow_time(edges, moves, self.net.edges, veh.v0, self.cfg, self.net.movement_geometry)
        return veh

    def route_length(self, edges: list, moves: list) -> float:
        return sum(self.net.edges[e].length for e in edges) + sum(self.net.movement_geometry[m][0] for _, m in moves)

    def try_insert(self, veh: Vehicle, node: str) -> bool:
        """Place the vehicle at s = 0 of its first edge, in the lane its turns need (else the
        emptier lane), if the lane start is free and it would not have to brake hard at once."""
        cfg = self.cfg
        edge = self.net.edges[veh.route_edges[0]]
        need = self.needs_of(veh)[0]
        lane = need if need is not None else choose_lane(edge, None, cfg)
        lst = edge.lanes[lane]
        if lst:
            last = lst[-1]
            gap = last.s - last.l_veh                        # rear bumper of the last vehicle
            if gap < cfg.spawn_free_dist:
                return False
            v_spawn = min(veh.v0, last.v) if last.s <= cfg.spawn_leader_dist else cfg.spawn_speed_factor * veh.v0
            if gap < cfg.s0 + v_spawn * veh.T:
                return False
            if self.accel(veh, veh.v0, gap, last.v, last.a_prev, v=v_spawn) < -veh.b:
                return False
        else:
            v_spawn = cfg.spawn_speed_factor * veh.v0
        veh.edge = edge
        veh.edge_idx = 0
        veh.lane = lane
        veh.lc_prev_lane = lane
        veh.s = 0.0
        veh.v = v_spawn
        veh.v_prev = v_spawn
        veh.enter_t = self.t
        veh.status = FREE
        veh.latch = None if (edge.junction_at_dest < 0 or self.approach_green(edge)) else STOPPING
        lst.append(veh)
        if edge.junction_at_dest >= 0:
            edge.entry_times.append(self.t)
        self.active.append(veh)
        self.note_ambulance_entry(veh, edge)
        return True

    def junction_q_total(self, J) -> int:
        return sum(1 for a in DIRS for lane in self.net.edges[J.approach_edges[a]].lanes for v in lane
                   if v.v < self.cfg.queue_speed)

    def note_ambulance_entry(self, veh: Vehicle, edge):
        """Recovery-time bookkeeping (DECISIONS §10): remember q_total of the junction
        ahead when an ambulance is first observed on one of its approaches."""
        if veh.is_ambulance and edge.junction_at_dest >= 0:
            J = self.net.junctions[edge.junction_at_dest]
            self.ev_events[(veh.id, J.id)] = dict(t_first=self.t, q0=self.junction_q_total(J), t_clear=math.nan)

    # ------------------------------------------------------------------ lane changing (§12.3)
    def change_lanes(self):
        """Lane decisions on the links, front vehicle first on every edge, one change per vehicle
        per step, applied at once so the vehicle behind sees it: (1) regain the lane its turns
        need (only after it moved aside for an ambulance), (2) move out of an ambulance's way
        where there is room, (3) MOBIL for vehicles whose turns leave them free to choose."""
        cfg = self.cfg
        self.coop = {}
        self._shadows = {}
        t = self.t
        for v in self.recent_lc:
            if v.edge is not None and t - v.lc_time < cfg.lc_duration_s - 1e-9:
                self._shadows.setdefault((v.edge.id, v.lc_prev_lane), []).append(v)
        self.recent_lc = [v for v in self.recent_lc if v.edge is not None and t - v.lc_time < cfg.lc_duration_s]
        if not cfg.lane_changes or cfg.lanes_per_dir < 2:
            return
        self._amb_lanes = {}
        for v in self.active:
            if v.is_ambulance and v.edge is not None:
                self._amb_lanes.setdefault((v.edge.id, v.lane), []).append(v)
        for edge in self.net.edges.values():
            if not any(edge.lanes):
                continue
            vehs = sorted((v for lane in edge.lanes for v in lane), key=lambda v: (-v.s, v.id))
            for veh in vehs:
                self.consider_lane_change(veh, edge, t)

    def consider_lane_change(self, veh: Vehicle, edge, t: float) -> None:
        cfg = self.cfg
        n = edge.n_lanes
        to_junction = edge.junction_at_dest >= 0
        d_stop = edge.stop_s - veh.s if to_junction else math.inf
        need = self.needs_of(veh)[veh.edge_idx]
        wrong = need is not None and veh.lane != need
        if to_junction and d_stop < cfg.lc_no_change_dist:
            if veh.lane not in lanes_for_movement(veh.next_movement(), n):
                self.reroute(veh, edge)                     # missed its lane at the solid line: take another movement
            return                                          # (a need further ahead is regained on the next link)
        if veh.s - veh.l_veh < cfg.lc_after_box_dist or veh.latch == COMMITTED:
            return
        if wrong:
            if not veh.is_ambulance and d_stop > cfg.lc_coop_dist / 2.0 and self.ambulance_behind_on(veh, edge):
                return                                      # stay aside until the ambulance has passed
            target = veh.lane + (1 if need > veh.lane else -1)
            if t - veh.lc_time >= cfg.lc_cooldown_s - 1e-9 and self.lc_safe(veh, edge, target, urgent=True)[0]:
                self.apply_lane_change(veh, edge, target, "mandatory")
            elif d_stop < cfg.lc_coop_dist:
                self.request_gap(veh, edge, target)
            return
        if t - veh.lc_time < cfg.lc_cooldown_s - 1e-9:
            return
        if not veh.is_ambulance:
            ambs = self._amb_lanes.get((edge.id, veh.lane))
            if ambs:
                amb = max((a for a in ambs if a.s < veh.s and veh.s - a.s <= cfg.ev_siren_range_m),
                          key=lambda a: a.s, default=None)
                if amb is not None:
                    target = self.yield_target(veh, edge, amb)
                    if target is not None:
                        if self.lc_safe(veh, edge, target, urgent=True)[0]:
                            self.apply_lane_change(veh, edge, target, "yield")
                        elif amb.id not in veh.yield_blocked_by:
                            veh.yield_blocked_by.add(amb.id)
                            self.n_yield_blocked += 1
                    return
        if need is not None:
            return                                          # in the lane its turns need (ambulances too: an overtake out of
                                                            # that lane may not be undoable before the solid line)
        lane = edge.lanes[veh.lane]
        i = lane.index(veh)
        ahead = lane[i - 1] if i > 0 else None
        behind = lane[i + 1] if i + 1 < len(lane) else None
        impeded = ahead is not None and ahead.s - ahead.l_veh - veh.s < 60.0 and ahead.v < veh.v0 - 0.5
        pressed = behind is not None and veh.s - veh.l_veh - behind.s < 25.0 and behind.v0 > veh.v0 + 0.5
        if not (impeded or pressed):
            return
        if not veh.is_ambulance and veh.v < cfg.queue_speed:
            return                                          # no lane hopping in a standing or crawling queue
        recent = veh.lc_edge_id == edge.id and t - veh.lc_time < cfg.lc_return_block_s
        best, best_gain = None, cfg.lc_threshold
        for target in (veh.lane - 1, veh.lane + 1):
            if recent and target == veh.lc_prev_lane:
                continue
            if 0 <= target < n:
                ok, gain = self.lc_safe(veh, edge, target, incentive=True)
                if ok and gain > best_gain:
                    best, best_gain = target, gain
        if best is not None:
            self.apply_lane_change(veh, edge, best, "overtake" if veh.is_ambulance else "discretionary")

    def yield_target(self, veh: Vehicle, edge, amb: Vehicle):
        """Where a car moves to clear an ambulance's lane: toward the kerb from any other lane;
        out of the kerb lane only when the ambulance itself needs the kerb lane (otherwise the
        ambulance passes in the other lane)."""
        if veh.lane > 0:
            return veh.lane - 1
        if self.needs_of(amb)[amb.edge_idx] == 0 and edge.n_lanes > 1:
            return 1
        return None

    def target_neighbours(self, veh: Vehicle, edge, target: int):
        """(leader, follower, follower position) of `veh` if it were in lane `target` now.  A
        vehicle still on a connector bound for that lane counts as a follower behind the edge start."""
        lead = fol = None
        for o in edge.lanes[target]:                         # front first
            if o.s > veh.s or (o.s == veh.s and o.id < veh.id):
                lead = o
            else:
                fol = o
                break
        if fol is not None:
            return lead, fol, fol.s
        if edge.junction_at_from >= 0:
            J = self.net.junctions[edge.junction_at_from]
            best, best_s = None, -math.inf
            for c in J.connectors.values():
                if c.out_edge == edge.id and c.out_lane == target and c.vehicles:
                    o = c.vehicles[0]
                    s_eq = o.s - c.length
                    if s_eq > best_s:
                        best, best_s = o, s_eq
            if best is not None:
                return lead, best, best_s
        return lead, None, -math.inf

    def lc_safe(self, veh: Vehicle, edge, target: int, incentive: bool = False, urgent: bool = False):
        """(safe, MOBIL gain) of moving `veh` into lane `target` (Kesting, Treiber & Helbing 2007,
        with hard gap and time-to-collision guards): gaps >= lc_min_gap to the new leader and
        follower, neither of them needs to brake harder than lc_b_safe, neither pair is closing
        with TTC < lc_min_ttc_s, and no vehicle still sliding out of that lane overlaps it.  The
        gain is a_c' - a_c + p (a_n' - a_n + a_o' - a_o)."""
        cfg = self.cfg
        b_safe = cfg.lc_b_safe_urgent if urgent else cfg.lc_b_safe
        lead, fol, s_fol = self.target_neighbours(veh, edge, target)
        if lead is not None and lead.s - lead.l_veh - veh.s < cfg.lc_min_gap:
            return False, 0.0
        if not veh.is_ambulance:
            for o in edge.lanes[target]:
                if o.is_ambulance and o.s < veh.s and veh.s - o.s <= max(cfg.lc_ev_clear_m, cfg.ev_siren_range_m):
                    return False, 0.0                       # never cut in front of an ambulance it would have to yield to
        g_r = veh.s - veh.l_veh - s_fol
        if fol is not None and g_r < cfg.lc_min_gap:
            return False, 0.0
        for o in self._shadows.get((edge.id, target), ()):
            if o is not veh and o.s - o.l_veh - cfg.lc_min_gap < veh.s and veh.s - veh.l_veh - cfg.lc_min_gap < o.s:
                return False, 0.0
        to_j = edge.junction_at_dest >= 0
        conn_t = self.path_for(veh, edge, target) if to_j else None
        v0_t = self.desired_speed(veh, edge, conn_t)
        obs_t = self.lane_obstacles(veh, edge, target, lead, conn_t)
        a_new, _ = self.cf_accel(veh, v0_t, obs_t)
        if a_new < -b_safe:
            return False, 0.0
        for gap, v_l, _, lobj, _ in obs_t:
            if lobj is not None and veh.v > v_l and gap / (veh.v - v_l) < cfg.lc_min_ttc_s:
                return False, 0.0
        a_fol_new = 0.0
        if fol is not None:
            v0_f = self.vehicle_v0_local(fol)
            a_fol_new = self.accel(fol, v0_f, g_r, veh.v, veh.a_prev)
            if a_fol_new < -b_safe:
                return False, 0.0
            if fol.v > veh.v and g_r / (fol.v - veh.v) < cfg.lc_min_ttc_s:
                return False, 0.0
        if not incentive:
            return True, math.inf
        cur = edge.lanes[veh.lane]
        i = cur.index(veh)
        c_lead = cur[i - 1] if i > 0 else None
        c_fol = cur[i + 1] if i + 1 < len(cur) else None
        conn_c = self.path_for(veh, edge) if to_j else None
        v0_c = self.desired_speed(veh, edge, conn_c)
        gain = a_new - self.cf_accel(veh, v0_c, self.lane_obstacles(veh, edge, veh.lane, c_lead, conn_c))[0]
        p = veh.politeness
        if p > 0.0:
            if fol is not None and fol.edge is edge:
                conn_f = self.path_for(fol, edge) if to_j else None
                a_fol_old = self.cf_accel(fol, self.vehicle_v0_local(fol), self.lane_obstacles(fol, edge, target, lead, conn_f))[0]
                gain += p * (a_fol_new - a_fol_old)
            if c_fol is not None:
                v0_o = self.vehicle_v0_local(c_fol)
                a_o = self.accel(c_fol, v0_o, veh.s - veh.l_veh - c_fol.s, veh.v, veh.a_prev)
                conn_o = self.path_for(c_fol, edge) if to_j else None
                a_o_new = self.cf_accel(c_fol, v0_o, self.lane_obstacles(c_fol, edge, veh.lane, c_lead, conn_o))[0]
                gain += p * (a_o_new - a_o)
        return True, gain

    def vehicle_v0_local(self, veh: Vehicle) -> float:
        if veh.conn is not None:
            return min(veh.v0, veh.conn.v_cap)
        edge = veh.edge
        return self.desired_speed(veh, edge, self.path_for(veh, edge) if edge.junction_at_dest >= 0 else None)

    def apply_lane_change(self, veh: Vehicle, edge, target: int, kind: str) -> None:
        old = veh.lane
        edge.lanes[old].remove(veh)
        lst = edge.lanes[target]
        i = 0
        while i < len(lst) and (lst[i].s > veh.s or (lst[i].s == veh.s and lst[i].id < veh.id)):
            i += 1
        lst.insert(i, veh)
        veh.lc_prev_lane, veh.lane, veh.lc_time, veh.lc_edge_id = old, target, self.t, edge.id
        veh.wait_since = math.nan
        veh.n_lane_changes += 1
        self.n_lane_changes += 1
        self.recent_lc.append(veh)
        self._shadows.setdefault((edge.id, old), []).append(veh)
        if kind == "mandatory":
            veh.n_lc_mandatory += 1
            self.n_lc_mandatory += 1
        elif kind == "yield":
            veh.n_lc_yield += 1
            self.n_lc_yield += 1
        elif kind == "overtake":
            veh.n_overtakes += 1
            self.n_ev_overtakes += 1
        else:
            self.n_lc_discretionary += 1

    def request_gap(self, veh: Vehicle, edge, target: int) -> None:
        """Cooperation: the follower in the target lane holds back behind `veh` (if that is
        comfortable for it, checked in compute_accelerations) so a gap opens."""
        _, fol, s_fol = self.target_neighbours(veh, edge, target)
        if fol is None or fol.edge is not edge or veh.s - veh.l_veh - s_fol <= 0.0:
            return
        cur = self.coop.get(fol.id)
        if cur is None or veh.s < cur.s:
            self.coop[fol.id] = veh

    def reroute(self, veh: Vehicle, edge) -> None:
        """The vehicle reached the solid line in a lane that does not serve its movement: it
        takes the movement with the shortest remaining route that its lane does serve
        (deterministic, no random draw; destination and free-flow time are kept, so the detour
        counts as delay).  If every movement its lane serves leaves the network at another
        boundary node, it leaves there (counted in n_destination_changes): nobody waits at the
        line, so two vehicles that need each other's lane can never lock each other."""
        cfg = self.cfg
        J = self.net.junctions[edge.junction_at_dest]
        penalty = {"T": cfg.turn_penalty_T, "NS": cfg.turn_penalty_NS, "FS": cfg.turn_penalty_FS}
        best = None
        for m in MOVEMENTS:
            if veh.lane not in lanes_for_movement(m, edge.n_lanes):
                continue
            c = next(c for c in J.lane_connectors[(edge.approach_dir, veh.lane)] if c.movement == m)
            r = route_from_edge(self.net, c.out_edge, veh.exit_node, cfg)
            if r is None:
                continue
            edges_r, moves_r, cost = r
            total = penalty[m] + self.net.edges[c.out_edge].length + cost
            if best is None or total < best[0] - 1e-9:
                best = (total, m, edges_r, moves_r)
        if best is None:
            ordered = sorted((c for c in J.lane_connectors[(edge.approach_dir, veh.lane)]
                              if veh.lane in lanes_for_movement(c.movement, edge.n_lanes)),
                             key=lambda c: (MOVEMENTS.index(c.movement), c.out_lane))
            c = ordered[0]
            out_edge = self.net.edges[c.out_edge]
            veh.exit_node = out_edge.to_node                # the drawn destination stays in vehicles.destination
            r0 = route_from_edge(self.net, veh.route_edges[0], veh.exit_node, cfg)
            if r0 is not None:                              # delay is measured against the trip it actually made
                veh.free_flow_time = free_flow_time(r0[0], r0[1], self.net.edges, veh.v0, cfg, self.net.movement_geometry)
            best = (0.0, c.movement, [out_edge.id], [])
            self.n_destination_changes += 1
        _, m, edges_r, moves_r = best
        veh.route_edges = veh.route_edges[:veh.edge_idx + 1] + edges_r
        veh.route_moves = veh.route_moves[:veh.edge_idx] + [(J.id, m)] + moves_r
        veh.needs = self.compute_needs(veh)
        veh.n_junctions = len(veh.route_moves)
        veh.n_turns = sum(1 for _, mm in veh.route_moves if mm != "T")
        veh.route_len = self.route_length(veh.route_edges, veh.route_moves)
        veh.plan_edge_idx = -1
        veh.n_reroutes += 1
        self.n_reroutes += 1

    # ------------------------------------------------------------------ car following
    def sees_stop_line(self, veh: Vehicle, edge, conn) -> bool:
        """Does this vehicle treat the stop line as a stationary leader?  Red (or latched
        STOPPING), or - anticipatory don't-block-the-box - its exit lane has no room and it can
        still stop (once it cannot, it enters and may stop inside the box)."""
        if edge.junction_at_dest < 0 or veh.latch == COMMITTED:
            return False
        if veh.latch == STOPPING or not self.approach_green(edge):
            return True
        return self.exit_blocked(veh, conn) and self.can_stop_at_line(veh, edge)

    def exit_blocked(self, veh: Vehicle, conn) -> bool:
        """The exit lane of `conn` is blocked for `veh`: its last vehicle is queued and the free
        space at its start is less than the vehicle's own length + s0 + margin plus the space
        claimed by every vehicle already on a box path to that lane (exit-space reservation)."""
        cfg = self.cfg
        lst = self.net.edges[conn.out_edge].lanes[conn.out_lane]
        if not lst:
            return False
        last = lst[-1]
        if last.v >= cfg.queue_speed:
            return False
        need = veh.l_veh + cfg.s0 + cfg.box_free_margin
        J = self.net.junctions[conn.junction_id]
        for c in J.connectors.values():
            if c.out_edge == conn.out_edge and c.out_lane == conn.out_lane:
                for o in c.vehicles:
                    if o is not veh:
                        need += o.l_veh + cfg.s0
        return (last.s - last.l_veh) < need

    def can_stop_at_line(self, veh: Vehicle, edge) -> bool:
        d = edge.stop_s - veh.s
        return d >= veh.v * veh.v / (2.0 * self.cfg.b_stop)

    def path_bodies(self, conn):
        """(vehicle, front position on the path, path) for every body on `conn` or on a path
        that shares its entry lane, including vehicles whose rear is still on it after leaving."""
        J = self.net.junctions[conn.junction_id]
        for c in J.lane_connectors[(conn.approach, conn.in_lane)]:
            for o in c.vehicles:
                yield o, o.s, c
            out = self.net.edges[c.out_edge].lanes[c.out_lane]
            for o in reversed(out):
                if o.s >= o.l_veh:
                    break
                if o.last_conn is c:
                    yield o, c.length + o.s, c

    def shadow_leader(self, veh: Vehicle, edge, lane: int):
        """A vehicle that left `lane` less than lc_duration_s ago is still sliding across: it
        stays a leader of the vehicles behind it in that lane."""
        best = None
        for o in self._shadows.get((edge.id, lane), ()):
            if o is not veh and o.edge is edge and o.s > veh.s:
                g = o.s - o.l_veh - veh.s
                if best is None or g < best[0]:
                    best = (g, o.v, o.a_prev, o, "lane_change")
        return best

    def lane_obstacles(self, veh: Vehicle, edge, lane: int, ahead, conn) -> list:
        """Every obstacle `veh` must respect if it is in `lane` of `edge` with `ahead` the vehicle in
        front of it in that lane (None = lane head) and `conn` the box path it would take:
        (gap, v_lead, a_lead, leader vehicle or None, kind).  The caller obeys the most demanding
        one (minimum acceleration), not the nearest: a stop line 20 m ahead can matter more than
        a car 17 m ahead that is about to cross it (DECISIONS §12.2)."""
        cfg = self.cfg
        out = []
        if ahead is not None:
            out.append((ahead.s - ahead.l_veh - veh.s, ahead.v, ahead.a_prev, ahead, "rear_end"))
        if self._shadows:
            sh = self.shadow_leader(veh, edge, lane)
            if sh is not None:
                out.append(sh)
        if edge.junction_at_dest < 0:
            return out
        d_stop = edge.stop_s - veh.s
        if d_stop > cfg.leader_horizon:
            return out
        if self.sees_stop_line(veh, edge, conn):
            out.append((d_stop, 0.0, 0.0, None, "stop_line"))
            if ahead is None:                               # a body that entered the box may still stick out behind it
                for o, s_front, c in self.path_bodies(conn):
                    rear = s_front - o.l_veh
                    if rear < 0.0 and (c is conn or rear < c.siblings[conn.id]):
                        out.append((edge.length - veh.s + rear, o.v, o.a_prev, o, "rear_end" if c is conn else "diverge"))
            return out
        if ahead is not None:
            return out
        remaining = edge.length - veh.s
        best = None
        for o, s_front, c in self.path_bodies(conn):
            rear = s_front - o.l_veh
            if c is not conn and rear >= c.siblings[conn.id]:
                continue                                    # it has turned away from this path
            g = remaining + rear
            if g <= cfg.leader_horizon and (best is None or g < best[0]):
                best = (g, o.v, o.a_prev, o, "rear_end" if c is conn else "diverge")
        if best is not None:
            out.append(best)
        lst = self.net.edges[conn.out_edge].lanes[conn.out_lane]
        if lst:
            last = lst[-1]
            g = remaining + conn.length + last.s - last.l_veh
            if g <= cfg.leader_horizon:
                out.append((g, last.v, last.a_prev, last, "rear_end"))
        return out

    def box_obstacles(self, veh: Vehicle, conn, idx: int) -> list:
        """Obstacles inside the box: the vehicle ahead on its own path, a body on a path that shares
        its entry lane and has not yet turned away, and the last vehicle on its exit lane."""
        out = []
        if idx > 0:
            ahead = conn.vehicles[idx - 1]
            out.append((ahead.s - ahead.l_veh - veh.s, ahead.v, ahead.a_prev, ahead, "rear_end"))
        for o, s_front, c in self.path_bodies(conn):
            if o is veh or s_front <= veh.s or o.conn is conn:
                continue
            rear = s_front - o.l_veh
            if c is not conn and rear >= c.siblings[conn.id]:
                continue
            out.append((rear - veh.s, o.v, o.a_prev, o, "rear_end" if c is conn else "diverge"))
        lst = self.net.edges[conn.out_edge].lanes[conn.out_lane]
        if lst:
            last = lst[-1]
            out.append(((conn.length - veh.s) + last.s - last.l_veh, last.v, last.a_prev, last, "rear_end"))
        return out

    def cf_accel(self, veh: Vehicle, v0_local: float, obstacles: list, v: float | None = None):
        """(acceleration, binding obstacle) = the minimum over the obstacles of the car-following
        law, or the free-road acceleration when there is none."""
        if not obstacles:
            return self.accel(veh, v0_local, None, 0.0, 0.0, v=v), None
        best_a, best_o = math.inf, None
        for o in obstacles:
            a = self.accel(veh, v0_local, o[0], o[1], o[2], v=v, s0=self.cfg.s0_line if o[4] == "stop_line" else None)
            if a < best_a:
                best_a, best_o = a, o
        return best_a, best_o

    def desired_speed(self, veh: Vehicle, edge, conn) -> float:
        """v0 on links (curves are a separate constraint, curve_limit)."""
        return veh.v0

    def curve_limit(self, veh: Vehicle, d: float, v_cap: float) -> float:
        """Largest acceleration that still lets the driver reach a curve d metres ahead at its
        curvature limit, braking at most at min(b, b_curve), in discrete time (exact for the
        ballistic step): a <= (sqrt(v_cap^2 + 2 b_c max(d - v dt, 0)) - v) / dt."""
        cfg = self.cfg
        b_c = min(veh.b, cfg.b_curve)
        return (math.sqrt(v_cap * v_cap + 2.0 * b_c * max(d - veh.v * cfg.dt, 0.0)) - veh.v) / cfg.dt

    def accel(self, veh: Vehicle, v0_local: float, gap, v_lead: float, a_lead: float, v: float | None = None,
              s0: float | None = None) -> float:
        """The car-following law (ACC model).  A non-positive gap is an overlap, which the model
        does not define: the vehicle brakes at the physical limit.  Behind a standing obstacle the
        driver brakes no harder than the kinematic need v^2 / (2 (gap - s0)) when that need
        exceeds b (it stops s0 short of it), instead of ACC's 'need + b'."""
        cfg = self.cfg
        if gap is not None and gap <= 0.0:
            return -cfg.b_emerg
        s0 = cfg.s0 if s0 is None else s0
        v = veh.v if v is None else v
        a = acc_accel(v, v0_local, gap, v_lead, a_lead, veh.a_max, veh.b, veh.T, s0, cfg.acc_coolness, cfg.idm_delta)
        if gap is not None and v_lead == 0.0 and a_lead == 0.0 and gap > s0 + 0.1:
            a = max(a, -max(v * v / (2.0 * (gap - s0)), veh.b))
        return a

    def compute_accelerations(self):
        cfg = self.cfg
        safety = self.safety
        for edge in self.net.edges.values():
            to_j = edge.junction_at_dest >= 0
            for lane in edge.lanes:
                for idx, veh in enumerate(lane):
                    conn = self.path_for(veh, edge) if to_j else None
                    obs = self.lane_obstacles(veh, edge, veh.lane, lane[idx - 1] if idx > 0 else None, conn)
                    for o in obs:
                        if o[3] is not None:
                            safety.follow(veh, o[3], o[0], o[1], o[4])
                    v0_local = self.desired_speed(veh, edge, conn)
                    a, binding = self.cf_accel(veh, v0_local, obs)
                    asker = self.coop.get(veh.id)
                    if asker is not None and asker.edge is edge:
                        g2 = asker.s - asker.l_veh - veh.s
                        if g2 > 0.0:
                            a2 = self.accel(veh, v0_local, g2, asker.v, asker.a_prev)
                            limit = cfg.ev_coop_b if asker.is_ambulance else cfg.lc_coop_b
                            if -limit <= a2 < a:
                                a = a2                      # holds back so the asker can merge in front
                    if conn is not None and conn.is_turn:
                        # reach the arc at its curvature limit (lateral acceleration <= a_lat_max)
                        a = min(a, self.curve_limit(veh, edge.length - veh.s + conn.path.lead_in, conn.v_cap))
                    self.set_accel(veh, a, binding)
                    if veh.latch is None:
                        veh.status = FREE if not obs else FOLLOWING
                    else:
                        veh.status = veh.latch
        for J in self.net.junctions:
            for conn in J.connectors.values():
                for idx, veh in enumerate(conn.vehicles):
                    obs = self.box_obstacles(veh, conn, idx)
                    for o in obs:
                        if o[3] is not None:
                            safety.follow(veh, o[3], o[0], o[1], o[4])
                    a, binding = self.cf_accel(veh, min(veh.v0, conn.v_cap), obs)
                    if conn.is_turn and veh.s < conn.path.lead_in:
                        a = min(a, self.curve_limit(veh, conn.path.lead_in - veh.s, conn.v_cap))
                    self.set_accel(veh, a, binding)
                    veh.status = IN_BOX
                    if conn.is_turn:
                        p = conn.path
                        if p.lead_in <= veh.s <= p.length - p.lead_out:
                            safety.max_lat_accel = max(safety.max_lat_accel, veh.v * veh.v / p.radius)

    def set_accel(self, veh: Vehicle, a: float, binding):
        """Apply the car-following acceleration: log the health metrics, clip at the physical limit."""
        cfg = self.cfg
        if veh.v > 0.0:
            if a < cfg.hard_brake_threshold:                 # v1.2 step count, kept for continuity
                veh.n_hard_brakes += 1
                self.n_hard_brakes += 1
            if a < -cfg.b_emerg:                            # demand beyond the physical limit (clipped)
                self.n_emergency_brakes += 1
                if not veh.sat_on:
                    veh.sat_on = True
                    if binding is not None and binding[3] is not None:
                        self.n_brake_sat_vehicle += 1
                    else:
                        self.n_brake_sat_virtual += 1
            else:
                veh.sat_on = False
        a = max(a, -cfg.b_emerg)
        if a <= -cfg.drac_near_miss and veh.v > 0.0:          # harsh braking, counted as runs
            if not veh.harsh_on:
                veh.harsh_on = True
                veh.n_harsh_brake_events += 1
                self.n_harsh_brake_events += 1
        else:
            veh.harsh_on = False
        veh.a = a
        self.vehicle_steps += 1

    # ------------------------------------------------------------------ movement
    def box_positions(self) -> dict:
        """Path position of every body that can touch a box path within this step."""
        out = {}
        for J in self.net.junctions:
            for v, c, s_front in self.box_bodies(J):
                out[v.id] = (v, c, J.id, s_front)
            for a in DIRS:
                edge = self.net.edges[J.approach_edges[a]]
                for lane in edge.lanes:
                    if lane:
                        v = lane[0]
                        if edge.length - v.s < 15.0:
                            out[v.id] = (v, self.path_for(v, edge), J.id, v.s - edge.length)
        return out

    def path_position(self, veh: Vehicle, conn):
        if veh.conn is conn:
            return veh.s
        if veh.edge is not None:
            if veh.last_conn is conn and veh.edge.id == conn.out_edge:
                return conn.length + veh.s
            if veh.edge.id == conn.in_edge:
                return veh.s - veh.edge.length
        return None

    def move_vehicles(self):
        cfg = self.cfg
        dt = cfg.dt
        t = self.t
        before = self.box_positions()
        # 1) connectors -> exit lanes, front vehicle first across the whole junction.
        #    A vehicle that leaves its connector is appended to an exit lane; it has
        #    already moved this step, so the edge loop below must skip it.
        just_exited = set()
        for J in self.net.junctions:
            box_vehicles = [(veh, conn) for conn in J.connectors.values() for veh in conn.vehicles]
            box_vehicles.sort(key=lambda vc: -vc[0].s)
            for veh, conn in box_vehicles:
                veh.v_prev = veh.v
                s_box_old = veh.s
                veh.s, veh.v = ballistic_step(veh.s, veh.v, veh.a, dt)
                if conn.is_turn and s_box_old < conn.path.lead_in <= veh.s and veh.v > conn.v_cap + 0.5:
                    self.n_turn_overspeed += 1
                if veh.v < 0.05 and not veh.box_blocked_logged:
                    veh.box_blocked_logged = True
                    self.n_box_blocked += 1
                if veh.s >= conn.length:
                    conn.vehicles.remove(veh)
                    out_edge = self.net.edges[conn.out_edge]
                    veh.conn = None
                    veh.edge = out_edge
                    veh.edge_idx += 1
                    veh.lane = conn.out_lane
                    veh.lc_prev_lane = conn.out_lane
                    veh.target_lane = conn.out_lane
                    veh.last_conn = conn
                    veh.s -= conn.length
                    veh.box_blocked_logged = False
                    veh.latch = None if (out_edge.junction_at_dest < 0 or self.approach_green(out_edge)) else STOPPING
                    out_edge.lanes[veh.lane].append(veh)
                    if out_edge.junction_at_dest >= 0:
                        out_edge.entry_times.append(t)
                    just_exited.add(veh)
                    if veh.is_ambulance:
                        self.ev_events[(veh.id, conn.junction_id)]["t_clear"] = t + dt
                        self.note_ambulance_entry(veh, out_edge)
        # 2) edges: move, stop-line crossings, box entry, despawn
        for edge in self.net.edges.values():
            for lane in edge.lanes:
                for veh in list(lane):
                    if veh in just_exited:
                        continue
                    s_old = veh.s
                    veh.v_prev = veh.v
                    veh.s, veh.v = ballistic_step(veh.s, veh.v, veh.a, dt)
                    if edge.junction_at_dest >= 0:
                        if s_old < edge.stop_s <= veh.s:
                            frac = (edge.stop_s - s_old) / max(veh.s - s_old, 1e-9)
                            self.record_crossing(veh, edge, t + frac * dt)
                        if veh.s >= edge.length:
                            self.enter_box(veh, edge, lane)
                    elif veh.s - veh.l_veh >= edge.length:
                        lane.remove(veh)
                        self.despawn(veh)
        for J in self.net.junctions:
            for conn in J.connectors.values():
                if len(conn.vehicles) > 1:
                    conn.vehicles.sort(key=lambda v: -v.s)
        # 3) conflict-zone entries and exits inside the step (PET, zone collisions)
        records = []
        for veh, conn, jid, s_old in before.values():
            s_new = self.path_position(veh, conn)
            if s_new is not None and s_new > s_old:
                records.append((veh, conn, jid, s_old, s_new))
        self.safety.zone_step(t, dt, records)

    def record_crossing(self, veh: Vehicle, edge, t_cross: float):
        """Stop-line crossing (time interpolated inside the step): invariant (3)
        check, junction throughput and the gridlock clock."""
        jid = edge.junction_at_dest
        plant = self.plant_for_edge(edge)
        if not plant.is_green(DIR_INDEX[edge.approach_dir]):
            # the COMMITTED exemption only holds during the all-red that follows its own
            # red onset; crossing while ANOTHER approach is green is a violation regardless
            if veh.latch != COMMITTED or plant.state == GREEN:
                self.red_violations += 1
        self.crossings_cum[jid] += 1
        self.last_crossing_t = self.t
        self.last_crossing_t_j[jid] = self.t
        self.lane_last_cross_t[(edge.id, veh.lane)] = self.t
        self.lane_locked[(edge.id, veh.lane)] = False
        veh.last_stop_cross_t = t_cross

    def enter_box(self, veh: Vehicle, edge, lane: list):
        if veh.lane not in lanes_for_movement(veh.next_movement(), edge.n_lanes):
            self.lane_discipline_violations += 1         # cannot happen: it is rerouted at the solid line
            self.reroute(veh, edge)
        conn = self.path_for(veh, edge)
        lane.remove(veh)
        veh.s -= edge.length
        veh.edge = None
        veh.conn = conn
        veh.target_lane = conn.out_lane
        veh.wait_since = math.nan
        veh.latch = None
        veh.status = IN_BOX
        conn.vehicles.append(veh)
        if conn.is_turn and veh.s >= conn.path.lead_in and veh.v > conn.v_cap + 0.5:
            self.n_turn_overspeed += 1                       # reached the arc above its limit within this step

    def despawn(self, veh: Vehicle):
        veh.exit_t = self.t + self.cfg.dt
        veh.edge = None
        self.active.remove(veh)
        self.finished.append(veh)
        self.safety.forget(veh)
        if self.cfg.is_main(veh.exit_t):
            self.exits_main += 1

    def overlap_guard(self):
        """Post-step collision check (DECISIONS §12.4): a follower whose front ended the step
        inside the body ahead on its own path is a collision (an invariant that must stay 0).
        The position is repaired only to keep the lane order consistent."""
        safety = self.safety
        for edge in self.net.edges.values():
            for lst in edge.lanes:
                for i in range(1, len(lst)):
                    leader, follower = lst[i - 1], lst[i]
                    limit = leader.s - leader.l_veh
                    if follower.s > limit:
                        safety.collision_1d(follower, leader)
                        follower.s = limit - 0.01
                        follower.v = min(follower.v, leader.v)
                        self.n_overlap_fixed += 1
        for J in self.net.junctions:
            for conn in J.connectors.values():
                vs = conn.vehicles
                for i in range(1, len(vs)):
                    leader, follower = vs[i - 1], vs[i]
                    if follower.s > leader.s - leader.l_veh:
                        safety.collision_1d(follower, leader)
                        follower.s = max(leader.s - leader.l_veh - 0.01, 0.0)
                        follower.v = min(follower.v, leader.v)
                        self.n_overlap_fixed += 1
                for veh in vs:
                    for o, s_front, c in self.path_bodies(conn):
                        if o is veh or o.conn is conn or s_front <= veh.s:
                            continue
                        rear = s_front - o.l_veh
                        if c is not conn and rear >= c.siblings[conn.id]:
                            continue
                        if veh.s > rear:
                            safety.collision_1d(veh, o)
        for edge in self.net.edges.values():                 # approach heads vs bodies still sticking out of a box
            if edge.junction_at_dest < 0:
                continue
            for lst in edge.lanes:
                if not lst:
                    continue
                head = lst[0]
                conn = self.path_for(head, edge)
                for o, s_front, c in self.path_bodies(conn):
                    rear = s_front - o.l_veh
                    if rear >= 0.0 or (c is not conn and rear >= c.siblings[conn.id]):
                        continue
                    if head.s - edge.length > rear:
                        safety.collision_1d(head, o)

    def bookkeeping(self):
        cfg = self.cfg
        dt = cfg.dt
        for veh in self.active:
            dec = (veh.v_prev - veh.v) / dt
            if dec > cfg.b_emerg + 1e-6:
                self.decel_limit_violations += 1
            if dec > veh.max_decel:
                veh.max_decel = dec
            veh.v_prev = veh.v
            veh.a_prev = veh.a
            if veh.is_ambulance and veh.v < cfg.queue_speed and veh.edge is not None:
                lane = veh.edge.lanes[veh.lane]
                if (lane and lane[0] is veh and veh.edge.junction_at_dest >= 0
                        and not self.approach_green(veh.edge)):
                    veh.ev_red_s += dt                      # first in its lane, waiting for green
                else:
                    veh.ev_blocked_s += dt                  # standing behind other vehicles
            veh.accel_energy += veh.a * veh.a * dt
            if veh.v < cfg.queue_speed:
                veh.total_wait += dt
            if veh.v > cfg.queue_speed:
                veh.was_moving = True
            elif veh.v < cfg.stop_speed and veh.was_moving:
                veh.n_stops += 1
                veh.was_moving = False
        # liveness KPI (§11.8): longest time any lane head has stood with v < stop_speed
        for edge in self.net.edges.values():
            for lane in edge.lanes:
                if not lane:
                    continue
                head = lane[0]
                if head.v < cfg.stop_speed:
                    if math.isnan(head.wait_since):
                        head.wait_since = self.t
                    self.max_head_wait_s = max(self.max_head_wait_s, self.t + dt - head.wait_since)
                else:
                    head.wait_since = math.nan

    def check_gridlock(self):
        """Gridlock (DECISIONS §5 + amendment §11.3): the network-wide rule of the spec
        (no stop-line crossing anywhere for gridlock_window_s with >= gridlock_min_vehicles
        in the network) AND the same rule per junction (no crossing at junction j for
        gridlock_window_s while >= gridlock_min_vehicles wait on j's approach edges), so a
        junction frozen by spillback is flagged even while stub-to-stub turns trickle
        through elsewhere.  The episode ends at the first detection (labelled failure)."""
        cfg = self.cfg
        if self.gridlock:
            return
        for J in self.net.junctions:                 # per junction first: it names the junction
            if self.t - self.last_crossing_t_j[J.id] < cfg.gridlock_window_s:
                continue
            n_wait = sum(self.net.edges[J.approach_edges[a]].n_vehicles() for a in DIRS)
            if n_wait >= cfg.gridlock_min_vehicles:
                self.gridlock, self.gridlock_t, self.gridlock_junction = True, self.t, J.id
                return
        if (self.t - self.last_crossing_t >= cfg.gridlock_window_s
                and self.n_in_network() >= cfg.gridlock_min_vehicles):
            self.gridlock, self.gridlock_t, self.gridlock_junction = True, self.t, -1

    # ------------------------------------------------------------------ logging
    def log_1hz(self):
        cfg = self.cfg
        t = round(self.t)
        for J in self.net.junctions:
            q_total = n_total = 0
            wait_sum = 0.0
            plant = self.plants[J.id]
            for a in DIRS:
                edge = self.net.edges[J.approach_edges[a]]
                prune_entry_times(edge, self.t, cfg)
                n_a = 0
                for lane in edge.lanes:
                    for v in lane:
                        n_total += 1
                        n_a += 1
                        wait_sum += v.total_wait
                        if v.v < cfg.queue_speed:
                            q_total += 1
                # starvation KPI: longest time an approach held vehicles without being green
                key = (J.id, a)
                if plant.is_green(DIR_INDEX[a]) or n_a == 0:
                    self.red_wait_start[key] = None
                else:
                    if self.red_wait_start[key] is None:
                        self.red_wait_start[key] = self.t
                    elif cfg.is_main(self.t):
                        self.max_approach_red_s = max(self.max_approach_red_s, self.t - self.red_wait_start[key])
            self.junction_1hz_rows.append(dict(
                episode_id=self.episode_id, junction_id=J.id, t=t, q_total=q_total,
                n_total=n_total, wait_sum_total=round(wait_sum, 3),
                crossings_cum=self.crossings_cum[J.id]))
            self.q_total_lookup[(J.id, t)] = q_total
            self.check_spillback_lock(J, plant)
        if self.log_trajectories:
            self.log_states(t)

    def green_time_in_window(self, J, plant, a: str, w0: float) -> float:
        """Seconds approach `a` of junction J was green in (w0, t]: completed green periods
        (pruned once they end before w0) plus the running one."""
        t = self.t
        periods = self.green_periods[(J.id, a)]
        while periods and periods[0][1] <= w0:
            periods.popleft()
        total = sum(max(0.0, min(end, t) - max(onset, w0)) for onset, end in periods)
        if plant.is_green(DIR_INDEX[a]):
            total += max(0.0, t - max(plant.green_onset_t, w0))
        return total

    def check_spillback_lock(self, J, plant) -> None:
        """Spillback lock (amendment §11.8, 1 Hz, never touches the pressure function): an
        approach lane that has been occupied, without a stop-line crossing, for
        gridlock_window_s although its approach was green for at least two full green
        periods' worth (2 x fixed_time_green) in that window.  Latched per lane until a
        crossing happens or the lane empties; counted once per latch."""
        cfg = self.cfg
        t = self.t
        w0 = t - cfg.gridlock_window_s
        for a in DIRS:
            edge = self.net.edges[J.approach_edges[a]]
            green_s = None
            for k, lane in enumerate(edge.lanes):
                key = (edge.id, k)
                if not lane:
                    self.lane_occupied_since[key] = None
                    self.lane_locked[key] = False
                    continue
                if self.lane_occupied_since[key] is None:
                    self.lane_occupied_since[key] = t
                if self.lane_locked[key]:
                    continue
                since = max(self.lane_occupied_since[key], self.lane_last_cross_t[key])
                if t - since < cfg.gridlock_window_s:
                    continue
                if green_s is None:
                    green_s = self.green_time_in_window(J, plant, a, w0)
                if green_s >= 2.0 * cfg.fixed_time_green:
                    self.lane_locked[key] = True
                    self.n_spillback_locks += 1
                    if self.spillback_lock_junction < 0:
                        self.spillback_lock_t, self.spillback_lock_junction = t, J.id

    def log_states(self, t):
        cfg = self.cfg
        for veh in self.active:
            if veh.edge is not None:
                x, y = veh.edge.lane_xy(veh.lane, veh.s, cfg)
                edge_id = veh.edge.id
                green = veh.edge.junction_at_dest >= 0 and self.approach_green(veh.edge)
                d_stop = veh.edge.stop_s - veh.s if veh.edge.junction_at_dest >= 0 else -1.0
                hd = math.atan2(veh.edge.heading[1], veh.edge.heading[0])
            else:
                conn = veh.conn
                x, y, hd = conn.path.pose(veh.s)
                edge_id = conn.id
                green = True
                d_stop = -1.0
            self.state_rows.append(dict(
                episode_id=self.episode_id, t=t, veh_id=veh.id, edge_id=edge_id, lane=veh.lane,
                s_m=round(veh.s, 2), x_m=round(x, 2), y_m=round(y, 2), heading_deg=round(math.degrees(hd) % 360.0, 1),
                v_mps=round(veh.v, 3),
                a_mps2=round(veh.a, 3), dist_to_stopline_m=round(d_stop, 2),
                signal_ahead="G" if green else "R", status=veh.status, cum_wait_s=round(veh.total_wait, 1)))

    # ------------------------------------------------------------------ end of episode
    def finish_episode(self):
        """Label post-pass: y_mp_action from the row itself, y_q_total_t30 from the
        1 Hz junction log (first whole second >= t + 30)."""
        cfg = self.cfg
        for row in self.signal_rows:
            row["y_mp_action"] = label_request(row, cfg)
            t_label = math.ceil(row["t"] + cfg.label_horizon_s - 1e-9)
            if row["is_tail"] or row["is_cooldown"]:
                row["y_q_total_t30"] = math.nan
            else:
                row["y_q_total_t30"] = self.q_total_lookup.get((row["junction_id"], t_label), math.nan)
        for veh in self.active:
            veh.exit_t = math.nan
        self.safety.finish()

    def conflict_rows(self) -> list:
        return self.safety.conflict_rows(self.episode_id)

    def vehicle_rows(self) -> list:
        cfg = self.cfg
        rows = []
        t_end = self.t
        queued = [v for q in self.entry_queues.values() for v in q]
        for veh in self.finished + self.active + queued:
            completed = not math.isnan(veh.exit_t)
            entered = not math.isnan(veh.enter_t)
            if entered:
                travel = (veh.exit_t if completed else t_end) - veh.enter_t
                entry_delay = veh.enter_t - veh.spawn_request_t
            else:
                travel = 0.0
                entry_delay = t_end - veh.spawn_request_t
            rows.append(dict(
                episode_id=self.episode_id, veh_id=veh.id,
                spawn_request_t=veh.spawn_request_t, enter_t=veh.enter_t, exit_t=veh.exit_t,
                completed=completed, is_warmup=cfg.is_warmup(veh.spawn_request_t),
                origin=veh.origin, destination=veh.destination, exit_node=veh.exit_node,
                route_movements="|".join(f"{self.net.junctions[j].node}:{m}" for j, m in veh.route_moves),
                route_edges="|".join(veh.route_edges), route_len_m=round(veh.route_len, 2),
                n_junctions=veh.n_junctions, n_turns=veh.n_turns,
                free_flow_time_s=round(veh.free_flow_time, 3), travel_time_s=round(travel, 3),
                entry_queue_delay_s=round(entry_delay, 3),
                total_delay_s=round(entry_delay + travel - veh.free_flow_time, 3) if entered else round(entry_delay, 3),
                total_wait_s=round(veh.total_wait, 2), n_stops=veh.n_stops,
                n_red_commits=veh.n_red_commits, n_hard_brakes=veh.n_hard_brakes,
                accel_energy=round(veh.accel_energy, 3),
                n_harsh_brake_events=veh.n_harsh_brake_events, max_decel_mps2=round(veh.max_decel, 3),
                n_lane_changes=veh.n_lane_changes, n_lc_mandatory=veh.n_lc_mandatory, n_lc_yield=veh.n_lc_yield,
                n_reroutes=veh.n_reroutes, n_conflicts=veh.n_conflicts, n_near_misses=veh.n_near_misses,
                n_collisions=veh.n_collisions,
                min_ttc_s=round(veh.min_ttc, 3) if veh.min_ttc < math.inf else math.nan,
                max_drac_mps2=round(veh.max_drac, 3),
                ev_blocked_s=round(veh.ev_blocked_s, 1) if veh.is_ambulance else math.nan,
                ev_red_s=round(veh.ev_red_s, 1) if veh.is_ambulance else math.nan,
                n_overtakes=veh.n_overtakes, **veh.params_row()))
        return rows

    def invariant_counts(self) -> dict:
        """Safety invariants (DECISIONS §6, §11.2, §12.4): every one must be exactly 0."""
        return dict(
            two_green_violations=self.two_green_violations,
            box_clear_violations=self.box_clear_violations,
            committed_at_onset_violations=self.committed_at_onset_violations,
            red_violations=self.red_violations,
            min_green_violations=self.min_green_violations,
            allred_short_violations=self.allred_short_violations,
            collisions=self.safety.n_collisions_1d + self.safety.n_collisions_zone,
            critical_conflicts=self.safety.n_critical,
            decel_limit_violations=self.decel_limit_violations,
            lane_discipline_violations=self.lane_discipline_violations,
            conservation_error=self.n_spawn_requests - (len(self.finished) + self.n_in_network() + self.n_entry_queued()),
        )

    def episode_row(self) -> dict:
        cfg, sc = self.cfg, self.scenario
        vrows = self.vehicle_rows()
        main = [r for r in vrows if cfg.is_main(r["spawn_request_t"]) and r["vehicle_type"] != "ambulance"]
        evs = [r for r in vrows if cfg.is_main(r["spawn_request_t"]) and r["vehicle_type"] == "ambulance"]
        delays = np.array([r["total_delay_s"] for r in main]) if main else np.array([math.nan])
        stops = np.array([r["n_stops"] for r in main]) if main else np.array([math.nan])
        ev_delays = np.array([r["total_delay_s"] for r in evs]) if evs else np.array([math.nan])
        ev_travel = np.array([r["travel_time_s"] for r in evs]) if evs else np.array([math.nan])
        ev_stops = np.array([r["n_stops"] for r in evs]) if evs else np.array([math.nan])
        main_1hz = [r for r in self.junction_1hz_rows if cfg.is_main(r["t"])]
        main_ticks = [r for r in self.signal_rows if cfg.is_main(r["t"])]
        n_switch_main = sum(1 for r in main_ticks if r["switched"])
        exts = [e for p in self.plants for e in p.allred_extensions]
        inv = self.invariant_counts()
        hours_main = cfg.main_s / 3600.0
        return dict(
            episode_id=self.episode_id, seed=self.seed, config_hash=cfg.config_hash(),
            generator_version=GENERATOR_VERSION, grid_x=cfg.grid_x, grid_y=cfg.grid_y,
            lanes_per_dir=cfg.lanes_per_dir, drive_side=cfg.drive_side, L_block_m=cfg.L_block,
            v_limit_mps=cfg.v_limit, dt_s=cfg.dt, warmup_s=cfg.warmup_s, main_s=cfg.main_s,
            cooldown_s=cfg.cooldown_s, demand_level=sc.demand_level,
            lambda_base_vph=round(sc.lambda_base_vph, 3), demand_asym=sc.demand_asym,
            od_pattern=sc.od_pattern, controller=sc.controller,
            controller_eps=sc.controller_eps if sc.controller == "max_pressure_eps" else controller_eps(sc.controller, cfg),
            sensor_noise_level=sc.sensor_noise_level, split=sc.split,
            ev_rate=sc.ev_rate, lambda_ev_vph=sc.lambda_ev_vph,
            n_spawned=self.n_spawn_requests, n_arrived=len(self.finished),
            n_in_network_end=self.n_in_network(), n_entry_queued_end=self.n_entry_queued(),
            n_in_network_main_end=self.n_in_network_main_end, n_entry_queued_main_end=self.n_entry_queued_main_end,
            throughput_vph=round(self.exits_main / hours_main, 2),
            mean_total_delay_s=round(float(np.mean(delays)), 3),
            p90_total_delay_s=round(float(np.percentile(delays, 90)), 3),
            mean_stops=round(float(np.mean(stops)), 4),
            max_queue_veh=max((r["q_total"] for r in main_1hz), default=0),
            switches_per_min=round(n_switch_main / len(self.plants) / (cfg.main_s / 60.0), 4),
            n_red_commits=self.n_red_commits, n_hard_brakes=self.n_hard_brakes,
            n_overlap_fixed=self.n_overlap_fixed, n_box_blocked=self.n_box_blocked,
            allred_extension_mean_s=round(float(np.mean(exts)), 3) if exts else math.nan,
            gridlock=self.gridlock, gridlock_t=self.gridlock_t, gridlock_junction=self.gridlock_junction,
            max_approach_red_s=round(self.max_approach_red_s, 1),
            max_total_wait_s=round(max((r["total_wait_s"] for r in main), default=math.nan), 2),
            n_forced_starvation=sum(1 for r in main_ticks if r["forced"] == FORCED_STARVATION),
            n_emergency_brakes=self.n_emergency_brakes, n_commit_revoked=self.n_commit_revoked,
            n_brake_sat_vehicle=self.n_brake_sat_vehicle, n_brake_sat_virtual=self.n_brake_sat_virtual,
            n_harsh_brake_events=self.n_harsh_brake_events, n_turn_overspeed=self.n_turn_overspeed,
            n_lane_changes=self.n_lane_changes, n_lc_mandatory=self.n_lc_mandatory,
            n_lc_discretionary=self.n_lc_discretionary, n_lc_yield=self.n_lc_yield,
            n_yield_blocked=self.n_yield_blocked, n_ev_overtakes=self.n_ev_overtakes, n_reroutes=self.n_reroutes,
            n_destination_changes=self.n_destination_changes, n_pending_repicks=self.n_pending_repicks,
            lane_changes_per_veh=round(self.n_lane_changes / max(len(vrows), 1), 4),
            **self.safety.kpis(len(main)),
            ev_mean_blocked_s=round(float(np.mean([r["ev_blocked_s"] for r in evs])), 2) if evs else math.nan,
            ev_mean_red_s=round(float(np.mean([r["ev_red_s"] for r in evs])), 2) if evs else math.nan,
            max_head_wait_s=round(self.max_head_wait_s, 1),
            spillback_lock=self.n_spillback_locks > 0, n_spillback_locks=self.n_spillback_locks,
            spillback_lock_t=self.spillback_lock_t, spillback_lock_junction=self.spillback_lock_junction,
            measured_sat_flow_vphpl=cfg.measured_sat_flow_vphpl,
            label_switch_rate=round(float(np.mean([r["y_mp_action"] != r["phase"] for r in main_ticks])), 4) if main_ticks else math.nan,
            runtime_s=round(self.runtime_s, 2),
            n_ambulances=self.n_ambulances,
            ev_mean_travel_time_s=round(float(np.mean(ev_travel)), 3),
            ev_mean_delay_s=round(float(np.mean(ev_delays)), 3),
            ev_p90_delay_s=round(float(np.percentile(ev_delays, 90)), 3),
            ev_mean_stops=round(float(np.mean(ev_stops)), 4),
            car_mean_delay_penalty_s=math.nan,          # filled by generate.py from the paired no-ambulance run
            recovery_time_s=self.recovery_time(),
            n_invariant_violations=sum(v for k, v in inv.items() if k != "conservation_error") + abs(inv["conservation_error"]),
            vehicle_steps=self.vehicle_steps,
        )

    def recovery_time(self) -> float:
        """Mean seconds after an ambulance clears a junction until that junction's
        q_total is back within recovery_tolerance of its value at first sighting."""
        cfg = self.cfg
        times = []
        t_end = round(self.t)
        for (vid, jid), ev in self.ev_events.items():
            if math.isnan(ev["t_clear"]):
                continue
            limit = (1.0 + cfg.recovery_tolerance) * ev["q0"]
            for t in range(math.ceil(ev["t_clear"]), t_end + 1):
                q = self.q_total_lookup.get((jid, t))
                if q is not None and q <= limit + 1e-9:
                    times.append(t - ev["t_clear"])
                    break
        return round(float(np.mean(times)), 3) if times else math.nan

    def signal_tick_rows(self) -> list:
        return [{c: row[c] for c in SIGNAL_TICK_COLUMNS} for row in self.signal_rows]


def run_episode(cfg: SimConfig, scenario: Scenario, seed: int, episode_id: int = 0,
                log_trajectories: bool | None = None) -> World:
    """Convenience wrapper: build the world, run it, return it."""
    return World(cfg, scenario, seed, episode_id, log_trajectories).run()
