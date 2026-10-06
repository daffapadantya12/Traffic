"""One dataclass that holds every number of the design (docs/DECISIONS.md).

Traffic-engineering idea: a simulation is only reproducible if every constant
lives in one place and is written next to the data it produced.  `SimConfig`
is that place; `to_json()` and `config_hash()` stamp it onto every dataset.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field


GENERATOR_VERSION = "traffic_twin_v2.0"     # v1.1/v1.2 = DECISIONS §11; v2.0 = §12 (safe driving, lane changing, real box geometry)

# Daily demand profile (digital-twin / web-demo extension): hourly multipliers of lambda_base
# for hours 0..23 of a simulated day.  Night 0.1-0.15, 05-06 ramp, 07-09 morning peak
# 1.2-1.35, midday 0.8-0.9, 17-19 evening peak 1.15-1.4, evening 0.3-0.6.
DAILY_PROFILE = (0.15, 0.12, 0.10, 0.10, 0.12, 0.30, 0.70, 1.20, 1.35, 1.25, 0.90, 0.85,
                 0.90, 0.85, 0.80, 0.90, 1.05, 1.30, 1.40, 1.15, 0.60, 0.45, 0.35, 0.30)
# These fields are left out of config.json / config_hash while the profile is "flat", so the
# flat configuration (every released dataset) keeps its hash bit-identical (DECISIONS §11.11).
PROFILE_FIELDS = ("demand_profile", "demand_profile_hourly", "demand_profile_day_s", "demand_profile_t0_s")
# Options that are left out of config.json / config_hash while they are off, so switching them on later
# does not change the hash of a release made without them (DECISIONS §11.11, §12.7).
OPTIONAL_OFF_FIELDS = ("vehicle_mix",)


@dataclass
class SimConfig:
    # ---- §1 student decisions -------------------------------------------
    grid_x: int = 2                     # junctions in the x (east) direction
    grid_y: int = 2                     # junctions in the y (north) direction
    lanes_per_dir: int = 2              # lanes per travel direction (1 must also work)
    drive_side: str = "left"            # "left" (Indonesia) or "right"

    # ---- §2 units, time -------------------------------------------------
    dt: float = 0.5                     # physics step [s]
    log_interval: float = 1.0           # per-junction aggregate log [s]
    warmup_s: float = 300.0             # t = 0 is the start of warm-up
    main_s: float = 3600.0
    cooldown_s: float = 300.0           # spawning off
    tail_s: float = 30.0                # is_tail = last 30 s of the main period

    # ---- §3 network -----------------------------------------------------
    L_block: float = 200.0              # junction centre to centre [m]
    L_stub: float = 150.0               # junction centre to boundary node [m]
    w_lane: float = 3.5                 # lane width [m]
    stop_line_setback: float = 2.0      # stop line upstream of the box edge [m]
    corner_radius: float = 6.0          # kerb return radius; box = road width + 2 x corner radius (§12.1)
    a_lat_max: float = 3.0              # lateral acceleration limit on a turning path -> v_cap = sqrt(a_lat r) (§12.1)
    turn_penalty_T: float = 0.0         # Dijkstra turn penalties [m]
    turn_penalty_NS: float = 10.0
    turn_penalty_FS: float = 25.0

    # ---- §4 lanes and box rules -----------------------------------------
    box_free_margin: float = 1.0        # free space needed at exit lane start = l_veh + s0 + 1.0 (per vehicle heading there)
    leader_horizon: float = 100.0       # leader lookup / virtual leader horizon [m]
    b_curve: float = 2.0                # drivers slow for a curve at most at min(b, b_curve): a <= (sqrt(v_cap^2 + 2 b_c max(d - v dt, 0)) - v) / dt
    conflict_margin: float = 0.9        # two box paths conflict where their centre lines are closer than the widest vehicle
                                        # + this (3.0 m: body width plus off-tracking on the arcs, §12.1)

    # ---- §12.2 car following (ACC model = IIDM + constant-acceleration heuristic) ------
    acc_coolness: float = 0.99          # Treiber & Kesting 2013, sec. 11.3.8
    idm_delta: float = 4.0              # acceleration exponent
    s0_line: float = 0.5                # jam distance to a line obstacle (stop line): drivers stop just behind the line

    # ---- §12.3 lane changing (MOBIL + mandatory changes + cooperation) ---------
    lane_changes: bool = True           # False only for the per-lane saturation-flow measurement (calibrate.py)
    lc_politeness_max: float = 0.5      # politeness p = lc_politeness_max * (1 - alpha): aggressive drivers are less polite
    lc_threshold: float = 0.1           # MOBIL switching threshold [m/s^2]
    lc_b_safe: float = 2.0              # MOBIL safety: neither the new follower nor the changer may need to brake harder [m/s^2]
                                        # (half the usual 4 m/s^2: near misses are what v2.0 removes)
    lc_b_safe_urgent: float = 3.0       # ... for moving aside for an ambulance or regaining the lane a turn needs
    lc_ev_clear_m: float = 100.0        # no car changes into a lane in front of an ambulance this close behind (= siren range)
    lc_return_block_s: float = 20.0     # no discretionary change back to the lane just left on the same link within this time
    allred_stall_s: float = 10.0        # all-red held this long past its minimum by a blocked conflict zone: serve a clear approach
    lc_min_gap: float = 2.0             # bumper gaps to the new leader and the new follower must be at least s0 [m]
    lc_min_ttc_s: float = 1.5           # ... and neither pair may be closing faster than this time to collision
    lc_cooldown_s: float = 4.0          # minimum time between two lane changes of one vehicle
    lc_duration_s: float = 3.0          # sideways move: the old-lane follower keeps treating the changer as a leader
    lc_no_change_dist: float = 30.0     # solid lane line: no lane change this close to the stop line [m]
    lc_after_box_dist: float = 10.0     # ... nor before the rear bumper is this far into the link [m]
    lc_coop_dist: float = 120.0         # a vehicle that must regain its lane asks for a gap within this distance of the line
    lc_coop_b: float = 2.0              # a car opens that gap only if it needs at most this deceleration [m/s^2]

    # ---- §12.4 surrogate safety measures (FHWA SSAM) ------------------------------
    ttc_conflict_s: float = 1.5         # conflict: time to collision below this
    drac_near_miss: float = 3.4         # near miss: a conflict whose deceleration-to-avoid exceeded this [m/s^2]
    pet_conflict_s: float = 1.0         # box conflict: post-encroachment time below this

    # ---- §5 vehicles ----------------------------------------------------
    v_limit: float = 13.9               # [m/s] 50 km/h
    l_veh: float = 4.5                  # passenger-car length [m] (storage and features are counted in cars)
    w_veh: float = 1.8                  # passenger-car width [m]
    # optional vehicle mix (§12.5), off in the release: ((type, share, length, width, a_max factor, T factor), ...)
    # e.g. (("car", 0.85, 4.5, 1.8, 1.0, 1.0), ("van", 0.15, 6.0, 2.0, 0.8, 1.0)); empty = cars only, no extra draw
    vehicle_mix: tuple = ()
    s0: float = 2.0                     # IDM minimum gap [m]
    b_emerg: float = 7.0                # physical deceleration limit [m/s^2]
    b_stop: float = 3.0                 # stop-or-commit deceleration [m/s^2]
    t_pr_s: float = 1.0                 # perception-reaction time in the red-onset decision: stop if d >= v t_pr + v^2/(2 b_stop) (ITE)
    hard_brake_threshold: float = -3.0  # log hard_brake below this [m/s^2]
    overlap_tolerance: float = 0.3      # post-step overlap guard [m]
    overlap_rate_limit: float = 1e-4    # health metric: < 1 per 10,000 vehicle-steps
    # parameter distributions: value = mean * (1 + slope*(alpha-0.5)) * LN(0, sigma), clipped
    alpha_beta_a: float = 2.0
    alpha_beta_b: float = 2.0
    v0_slope: float = 0.20
    v0_sigma: float = 0.08
    v0_clip: tuple = (0.85, 1.15)       # times v_limit
    a_max_mean: float = 1.5
    a_max_slope: float = 0.50
    a_max_sigma: float = 0.10
    a_max_clip: tuple = (0.8, 2.5)
    b_mean: float = 2.0
    b_slope: float = 0.30
    b_sigma: float = 0.10
    b_clip: tuple = (1.2, 3.0)
    T_mean: float = 1.2
    T_slope: float = -0.40              # T = 1.2*(1 - 0.40*(alpha-0.5))
    T_sigma: float = 0.10
    T_clip: tuple = (0.8, 1.8)
    queue_speed: float = 2.0            # a vehicle is "queued"/waiting when v < 2 m/s
    stop_speed: float = 0.5             # n_stops counts transitions v > 2 -> v < 0.5
    spawn_free_dist: float = 10.0       # first 10 m of the lane must be free
    spawn_leader_dist: float = 50.0     # leader within 50 m sets the spawn speed
    spawn_speed_factor: float = 0.8     # else spawn at 0.8 * v0
    od_own_road: float = 0.30           # default OD row: far end of own road
    od_parallel: float = 0.15           # each end of each parallel road
    od_crossing: float = 0.10           # each end of each crossing road
    asym_ns_factor: float = 1.4         # ns_heavy: NS entries x1.4, EW entries x0.7
    asym_ew_factor: float = 0.7
    gridlock_window_s: float = 120.0    # no stop-line crossing for 120 s ...
    gridlock_min_vehicles: int = 10     # ... while >= 10 vehicles are in the network
    seed_base: int = 100000             # seed = seed_base + episode_id

    # ---- §6 signal plant --------------------------------------------------
    min_green: float = 5.0
    max_green: float = 60.0
    allred_min: float = 4.0
    decision_interval: float = 5.0
    starvation_s: float = 180.0         # plant starvation guard (amendment §11.1): an approach with
                                        # vehicles and no green for 3 x max_green is served next (forced = 4)
    fixed_time_green: float = 20.0
    startup_lost_time_s: float = 2.0    # measured by validate.py; 2 s textbook default
    longest_queue_margin: int = 3       # longest_queue switches if n_best > n_current + 3
    mp_hysteresis: float = 2.0          # max_pressure switches if p_best > p_current + 2
    mp_eps: float = 0.3                 # max_pressure_eps random request probability
    random_dwell_min: float = 5.0       # random_dwell holds U(5, 40) s
    random_dwell_max: float = 40.0

    # ---- §7 dataset -------------------------------------------------------
    label_horizon_s: float = 30.0       # y_q_total_t30
    arr_window_s: float = 30.0          # arr_a_30s
    demand_levels: tuple = ("low", "design", "high", "stress")
    demand_level_probs: tuple = (0.25, 0.40, 0.25, 0.10)
    demand_asyms: tuple = ("sym", "ns_heavy")
    demand_asym_probs: tuple = (0.5, 0.5)
    od_patterns: tuple = ("default", "uniform")
    od_pattern_probs: tuple = (0.70, 0.30)
    controllers: tuple = ("max_pressure", "max_pressure_eps", "fixed_time", "random_dwell")
    controller_probs: tuple = (0.40, 0.30, 0.20, 0.10)
    sensor_noise_levels: tuple = (0.0, 0.1)
    sensor_noise_probs: tuple = (0.60, 0.40)
    split_fractions: tuple = (0.70, 0.15, 0.15)   # train / val / test
    demand_jitter: float = 0.10         # lambda_base jittered by U(1-0.1, 1+0.1) per episode
    vc_targets: dict = field(default_factory=lambda: {
        "low": 0.4, "design": 0.7, "high": 0.95, "stress": 1.15})
    fixed_time_cycle_s: float = 96.0    # 4 * (20 + 4)
    effective_green_s: float = 18.0     # 20 + 4 - 6 (DECISIONS §6)
    measured_sat_flow_vphpl: float = float("nan")  # filled by calibrate.py
    # capacity per approach (§11.9): the formula value sat_flow x lanes x 18/96, the realised
    # fixed-time capacity measured in a saturated 1x1 episode, and their ratio; lambda_base
    # per level is derived from the MEASURED capacity so realised v/c equals the target
    capacity_formula_vph: float = float("nan")
    capacity_vph: float = float("nan")
    capacity_realisation: float = float("nan")
    capacity_probe_x: float = 1.3       # nominal x (of the formula capacity) of the saturated probe episode
    capacity_probe_main_s: float = 1200.0
    lambda_base_vph: dict = field(default_factory=dict)  # per demand level, filled by calibrate.py
    log_trajectories: bool = False
    # ---- daily demand profile (twin extension; "flat" = bit-identical to the v1.2 generator) ----
    demand_profile: str = "flat"        # "flat" (lambda_base all day) or "daily" (hourly multipliers)
    demand_profile_hourly: tuple = DAILY_PROFILE   # multiplier per hour of the simulated day
    demand_profile_day_s: float = 86400.0          # simulated length of one day (tests compress it)
    demand_profile_t0_s: float = 0.0               # time of day at t = 0 [s]

    # ---- §10 ambulance extension (healthcare framing) --------------------------
    lambda_ev_vph: float = 4.0          # ambulances per hour per boundary node when ev_rate = "default"
    ev_v0_factor: float = 1.25          # ambulance v0 = 1.25 * v_limit
    ev_a_max: float = 2.5
    ev_b: float = 3.0
    ev_T: float = 0.9
    ev_l_veh: float = 6.0
    ev_w_veh: float = 2.1
    ev_siren_range_m: float = 100.0     # cars within this distance ahead of an ambulance in its lane move aside if there is room (§12.3)
    ev_coop_b: float = 3.0              # cars open a gap for an ambulance that must change lane at up to this deceleration
    ev_horizon_m: float = 300.0         # an ambulance is observable within this distance of the stop line
    ev_eta_min_speed: float = 3.0       # ev_eta = ev_dist / max(v_ev, 3.0)
    preempt_eta_s: float = 20.0         # preempt_naive requests the approach when eta <= 20 s
    ev_rates: tuple = ("0", "default")  # scenario factor: no ambulances (30%) / lambda_ev_vph (70%)
    ev_rate_probs: tuple = (0.30, 0.70)
    recovery_tolerance: float = 0.10    # recovered when q_total <= (1 + 0.10) * q_total at first sighting

    # ---- §8 validation ------------------------------------------------------
    discharge_queue_n: int = 20
    sat_headway_vehicles: tuple = (5, 15)   # headways of vehicles 5..15 define the saturation headway
    sat_flow_expected_range: tuple = (1600.0, 1900.0)
    switch_rate_band: tuple = (0.7, 3.5)    # switches/min/junction: < 0.7 never switches, > 3.5 flicker (synthesis)
    ring_length_m: float = 1000.0

    # ---- derived geometry (properties, never stored twice) -------------------
    @property
    def road_width(self) -> float:
        return 2 * self.lanes_per_dir * self.w_lane

    @property
    def w_box(self) -> float:
        """Junction box = road width + a kerb return of corner_radius on each side (26 m for
        2 lanes and 6 m): a car in the kerb lane turns on a 7.75 m arc, not a 1.75 m one (§12.1)."""
        return self.road_width + 2.0 * self.corner_radius

    @property
    def internal_edge_length(self) -> float:
        return self.L_block - self.w_box

    @property
    def stub_edge_length(self) -> float:
        return self.L_stub - self.w_box / 2

    @property
    def episode_length_s(self) -> float:
        return self.warmup_s + self.main_s + self.cooldown_s

    @property
    def spawn_end_s(self) -> float:
        return self.warmup_s + self.main_s

    @property
    def box_free_needed(self) -> float:
        return self.l_veh + self.s0 + self.box_free_margin

    @property
    def max_vehicle_width(self) -> float:
        return max([self.w_veh, self.ev_w_veh] + [m[3] for m in self.vehicle_mix])

    @property
    def sat_headway_s(self) -> float:
        """Saturation headway from the measured saturation flow (2.0 s until measured)."""
        f = self.measured_sat_flow_vphpl
        return 3600.0 / f if f == f and f > 0 else 2.0

    def storage_per_lane(self, length: float) -> int:
        """Jam storage of one lane = floor(L / (l_veh + s0))."""
        return int(length // (self.l_veh + self.s0))

    # ---- helpers -------------------------------------------------------------
    def copy(self, **changes) -> "SimConfig":
        return dataclasses.replace(self, **changes)

    def demand_multiplier(self, t: float) -> float:
        """Hourly multiplier of the daily demand profile at simulated time t (1.0 when flat)."""
        if self.demand_profile == "flat":
            return 1.0
        tod = (t + self.demand_profile_t0_s) % self.demand_profile_day_s
        hour = int(tod * 24.0 / self.demand_profile_day_s) % 24
        return float(self.demand_profile_hourly[hour])

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["generator_version"] = GENERATOR_VERSION
        if self.demand_profile == "flat":          # hash-neutral: released datasets keep their config_hash
            for k in PROFILE_FIELDS:
                d.pop(k, None)
        for k in OPTIONAL_OFF_FIELDS:
            if not d.get(k):
                d.pop(k, None)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def config_hash(self) -> str:
        """Short stable fingerprint of every field (so a dataset names its config)."""
        return hashlib.sha1(self.to_json(indent=None).encode()).hexdigest()[:12]

    def is_warmup(self, t: float) -> bool:
        return t < self.warmup_s

    def is_cooldown(self, t: float) -> bool:
        return t >= self.spawn_end_s

    def is_tail(self, t: float) -> bool:
        return (t > self.spawn_end_s - self.tail_s) and not self.is_cooldown(t)

    def is_main(self, t: float) -> bool:
        return (not self.is_warmup(t)) and (not self.is_cooldown(t))


def load_config(path: str) -> SimConfig:
    """Read a config.json written by `SimConfig.to_json()` back into a dataclass."""
    with open(path) as f:
        d = json.load(f)
    d.pop("generator_version", None)
    fields = {f.name for f in dataclasses.fields(SimConfig)}
    kwargs = {}
    for k, v in d.items():
        if k not in fields:
            continue
        if isinstance(v, list):
            v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
        kwargs[k] = v
    return SimConfig(**kwargs)
