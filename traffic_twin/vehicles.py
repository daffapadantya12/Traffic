"""Vehicles: parameter sampling, the Intelligent Driver Model and the ballistic update.

Traffic-engineering idea: each driver accelerates toward a desired speed and
brakes to keep a safe gap behind the leader.  v2.0 uses the ACC model (improved IDM
blended with the constant-acceleration heuristic, Treiber & Kesting 2013), which brakes
as hard as a situation really needs - a red light that appears at short distance, a car
that cuts in - instead of the plain IDM's over-reaction.  A latent "aggressiveness"
alpha shifts desired speed, acceleration, headway and lane-change politeness together.
"""

from __future__ import annotations

import math

from .config import SimConfig

# vehicle status values
FREE = "FREE"
FOLLOWING = "FOLLOWING"
STOPPING = "STOPPING"       # latched at red onset: stop line is the virtual leader
COMMITTED = "COMMITTED"     # latched at red onset: crosses during all-red
IN_BOX = "IN_BOX"
QUEUED_AT_ENTRY = "QUEUED_AT_ENTRY"


def _lognormal(rng, sigma: float) -> float:
    return math.exp(rng.normal(0.0, sigma))


def sample_params(rng, cfg: SimConfig) -> dict:
    """Draw one vehicle from the vehicles RNG stream (DECISIONS §5, §12.5).

    Vehicle type first (only when the optional vehicle_mix is on), then the driver:
    value = mean * (1 + slope*(alpha - 0.5)) * LN(0, sigma), clipped, times the type factor for
    a_max and T.  Politeness is derived from alpha (no draw).  Draw order is fixed so that
    common random numbers hold between runs.
    """
    vtype, length, width, fa, fT = "car", cfg.l_veh, cfg.w_veh, 1.0, 1.0
    if cfg.vehicle_mix:                       # optional mix: one extra draw, only when the option is on
        u, acc_p = rng.random(), 0.0
        for name, share, l_m, w_m, f_a, f_T in cfg.vehicle_mix:
            acc_p += share
            vtype, length, width, fa, fT = name, l_m, w_m, f_a, f_T
            if u < acc_p:
                break
    alpha = rng.beta(cfg.alpha_beta_a, cfg.alpha_beta_b)
    v0 = cfg.v_limit * (1 + cfg.v0_slope * (alpha - 0.5)) * _lognormal(rng, cfg.v0_sigma)
    v0 = min(max(v0, cfg.v0_clip[0] * cfg.v_limit), cfg.v0_clip[1] * cfg.v_limit)
    a_max = cfg.a_max_mean * (1 + cfg.a_max_slope * (alpha - 0.5)) * _lognormal(rng, cfg.a_max_sigma)
    a_max = min(max(a_max, cfg.a_max_clip[0]), cfg.a_max_clip[1])
    b = cfg.b_mean * (1 + cfg.b_slope * (alpha - 0.5)) * _lognormal(rng, cfg.b_sigma)
    b = min(max(b, cfg.b_clip[0]), cfg.b_clip[1])
    T = cfg.T_mean * (1 + cfg.T_slope * (alpha - 0.5)) * _lognormal(rng, cfg.T_sigma)
    T = min(max(T, cfg.T_clip[0]), cfg.T_clip[1])
    return dict(alpha=alpha, v0=v0, a_max=a_max * fa, b=b, T=T * fT, l_veh=length, w_veh=width, vehicle_type=vtype,
                politeness=cfg.lc_politeness_max * (1.0 - alpha))


def ambulance_params(cfg: SimConfig) -> dict:
    """Fixed ambulance parameters (DECISIONS §10): faster, keener, longer; no draw."""
    return dict(alpha=1.0, v0=cfg.ev_v0_factor * cfg.v_limit, a_max=cfg.ev_a_max, b=cfg.ev_b,
                T=cfg.ev_T, l_veh=cfg.ev_l_veh, w_veh=cfg.ev_w_veh, vehicle_type="ambulance", politeness=0.0)


def idm_accel(v: float, v0_local: float, gap: float | None, v_lead: float,
              a_max: float, b: float, T: float, s0: float) -> float:
    """IDM acceleration.  gap=None means free road (no leader within the horizon).

    a = a_max * (1 - (v/v0)^4 - (s*/gap)^2),
    s* = s0 + max(0, v*T + v*(v - v_lead) / (2*sqrt(a_max*b))).
    Not clamped here: the caller logs hard brakes and clips at -b_emerg.
    """
    free = 1.0 - (v / v0_local) ** 4
    if gap is None:
        return a_max * free
    if gap <= 0.0:
        # a non-positive gap is an overlap, which IDM does not define; the caller
        # (World.set_accel) resolves it explicitly instead of a hidden clamp
        raise ValueError(f"idm_accel needs a positive gap, got {gap}")
    s_star = s0 + max(0.0, v * T + v * (v - v_lead) / (2.0 * math.sqrt(a_max * b)))
    return a_max * (free - (s_star / gap) ** 2)


def iidm_free(v: float, v0: float, a_max: float, b: float, delta: float = 4.0) -> float:
    """Free-road term of the improved IDM (Treiber & Kesting 2013, eq. 11.23): above the
    desired speed it brakes at most at the comfortable b instead of a*(1-(v/v0)^4), which
    explodes when v0 drops (a turn's curvature cap)."""
    if v <= v0:
        return a_max * (1.0 - (v / v0) ** delta)
    return -b * (1.0 - (v0 / v) ** (a_max * delta / b))


def iidm_accel(v: float, v0: float, gap: float | None, v_lead: float, a_max: float, b: float,
               T: float, s0: float, delta: float = 4.0) -> float:
    """Improved IDM (Treiber & Kesting 2013, sec. 11.3.7): z = s*/s,
    v <= v0: a(1 - z^2) if z >= 1 else a_free(1 - z^(2a/a_free));
    v >  v0: a_free + a(1 - z^2) if z >= 1 else a_free.  gap=None = free road."""
    af = iidm_free(v, v0, a_max, b, delta)
    if gap is None:
        return af
    if gap <= 0.0:
        raise ValueError(f"iidm_accel needs a positive gap, got {gap}")
    s_star = s0 + max(0.0, v * T + v * (v - v_lead) / (2.0 * math.sqrt(a_max * b)))
    z = s_star / gap
    if v <= v0:
        if z >= 1.0:
            return a_max * (1.0 - z * z)
        if af <= 1e-12:
            return 0.0
        return af * (1.0 - z ** (2.0 * a_max / af))
    if z >= 1.0:
        return af + a_max * (1.0 - z * z)
    return af


def cah_accel(v: float, gap: float, v_lead: float, a_lead: float, a_max: float) -> float:
    """Constant-acceleration heuristic (Treiber & Kesting 2013, eq. 11.25): the acceleration
    that just avoids a collision if the leader keeps its acceleration a_lead (capped at a_max)."""
    al = min(a_lead, a_max)
    den = v_lead * v_lead - 2.0 * gap * al
    if v_lead * (v - v_lead) <= -2.0 * gap * al and den > 1e-9:
        return v * v * al / den
    dv = v - v_lead
    return al - (dv * dv if dv > 0.0 else 0.0) / (2.0 * gap)


def acc_accel(v: float, v0: float, gap: float | None, v_lead: float, a_lead: float, a_max: float, b: float,
              T: float, s0: float, coolness: float = 0.99, delta: float = 4.0) -> float:
    """ACC model (Treiber & Kesting 2013, eq. 11.26): IIDM blended with the CAH so that a
    suddenly small gap (red onset, cut-in, merge) is answered with the braking the situation
    really needs instead of IDM's over-reaction:
    a = a_IIDM if a_IIDM >= a_CAH else (1-c) a_IIDM + c [a_CAH + b tanh((a_IIDM - a_CAH)/b)],
    never more than the free-road acceleration: behind an accelerating leader the CAH is positive
    and the blend alone would push a driver who is above v0 (a curve's speed limit) further up."""
    ai = iidm_accel(v, v0, gap, v_lead, a_max, b, T, s0, delta)
    if gap is None:
        return ai
    ac = cah_accel(v, gap, v_lead, a_lead, a_max)
    if ai >= ac:
        return ai
    blend = (1.0 - coolness) * ai + coolness * (ac + b * math.tanh((ai - ac) / b))
    return min(blend, iidm_free(v, v0, a_max, b, delta))


def ballistic_step(s: float, v: float, a: float, dt: float):
    """Ballistic update with the stop-exactly branch (never drives backwards)."""
    if v + a * dt >= 0.0:
        return s + v * dt + 0.5 * a * dt * dt, v + a * dt
    return s + v * v / (2.0 * abs(a)), 0.0


class Vehicle:
    """Mutable per-vehicle state.  `s` is the front-bumper position along the
    current edge or connector; 2-D (x, y) is only derived for logging."""

    __slots__ = (
        "id", "alpha", "v0", "a_max", "b", "T", "l_veh", "vehicle_type", "origin", "destination",
        "route_edges", "route_moves", "edge_idx", "edge", "lane", "conn", "s", "v", "a",
        "status", "latch", "target_lane", "spawn_request_t", "enter_t", "exit_t",
        "total_wait", "n_stops", "n_red_commits", "n_hard_brakes", "accel_energy",
        "was_moving", "box_blocked_logged", "free_flow_time", "route_len",
        "n_junctions", "n_turns", "last_stop_cross_t", "wait_since", "w_veh", "politeness",
        "needs", "plan_edge_idx", "plan_out_lane", "exit_node", "lc_edge_id",
        "lc_time", "lc_prev_lane", "last_conn", "a_prev", "v_prev",
        "n_lane_changes", "n_lc_mandatory", "n_lc_yield", "n_reroutes", "yield_blocked_by",
        "n_conflicts", "n_near_misses", "n_collisions", "min_ttc", "max_drac", "max_decel",
        "n_harsh_brake_events", "harsh_on", "sat_on", "ev_blocked_s", "ev_red_s", "n_overtakes",
    )

    def __init__(self, vid: int, params: dict, origin: str, destination: str,
                 route_edges: list, route_moves: list, spawn_request_t: float):
        self.id = vid
        self.alpha = params["alpha"]
        self.v0 = params["v0"]
        self.a_max = params["a_max"]
        self.b = params["b"]
        self.T = params["T"]
        self.l_veh = params.get("l_veh", 4.5)
        self.w_veh = params.get("w_veh", 1.8)
        self.vehicle_type = params.get("vehicle_type", "car")
        self.politeness = params.get("politeness", 0.25)
        self.origin = origin
        self.destination = destination
        self.exit_node = destination              # boundary node it actually leaves at (differs only after a forced reroute)
        self.route_edges = route_edges
        self.route_moves = route_moves          # list of (junction_id, movement), one per junction
        self.edge_idx = 0
        self.edge = None                        # Edge object when on an edge
        self.lane = 0
        self.conn = None                        # Connector object when inside a box
        self.s = 0.0
        self.v = 0.0
        self.a = 0.0
        self.status = QUEUED_AT_ENTRY
        self.latch = None                       # None / STOPPING / COMMITTED
        self.target_lane = 0
        self.spawn_request_t = spawn_request_t
        self.enter_t = math.nan
        self.exit_t = math.nan
        self.total_wait = 0.0
        self.n_stops = 0
        self.n_red_commits = 0
        self.n_hard_brakes = 0
        self.accel_energy = 0.0
        self.was_moving = False
        self.box_blocked_logged = False
        self.free_flow_time = 0.0
        self.route_len = 0.0
        self.n_junctions = len(route_moves)
        self.n_turns = sum(1 for _, m in route_moves if m != "T")
        self.last_stop_cross_t = math.nan
        self.wait_since = math.nan                # since when it has stood (v < stop_speed) as the head of its lane
        # lane choice and lane changing (§12.3)
        self.needs = []                           # per route edge: the lane its remaining turns need (None = any)
        self.plan_edge_idx = -1                   # edge on which the exit lane of a turn was chosen ...
        self.plan_out_lane = 0                    # ... and that lane (kept until the box, so it never flips)
        self.lc_time = -math.inf                  # time of the last lane change (cool-down, drawing)
        self.lc_prev_lane = 0                     # lane before the last change (the viewer slides between the two)
        self.lc_edge_id = None                    # edge on which the last change happened
        self.last_conn = None                     # connector it last used (its rear may still be in that box)
        self.a_prev = 0.0                         # acceleration of the last step (leader term of the CAH)
        self.v_prev = 0.0                         # speed before the last move (deceleration-limit invariant)
        self.n_lane_changes = 0
        self.n_lc_mandatory = 0
        self.n_lc_yield = 0
        self.n_reroutes = 0
        self.yield_blocked_by = set()             # ambulances it wanted to make room for but could not
        self.n_overtakes = 0                      # ambulances: lane changes to pass slower traffic
        # surrogate safety measures (§12.4)
        self.n_conflicts = 0
        self.n_near_misses = 0
        self.n_collisions = 0
        self.min_ttc = math.inf
        self.max_drac = 0.0
        self.max_decel = 0.0
        self.n_harsh_brake_events = 0             # runs of applied deceleration >= 3.4 m/s^2
        self.harsh_on = False
        self.sat_on = False
        self.ev_blocked_s = 0.0                   # ambulances: time standing behind another vehicle
        self.ev_red_s = 0.0                       # ambulances: time standing at a red stop line as lane head

    # ---- route helpers -------------------------------------------------------
    def next_movement(self) -> str | None:
        """Movement at the junction at the end of the current edge (None if boundary)."""
        if self.edge_idx < len(self.route_moves):
            return self.route_moves[self.edge_idx][1]
        return None

    def following_movement(self) -> str | None:
        """Movement one junction further on (decides the exit lane)."""
        if self.edge_idx + 1 < len(self.route_moves):
            return self.route_moves[self.edge_idx + 1][1]
        return None

    def params_row(self) -> dict:
        return dict(alpha=self.alpha, v0_mps=self.v0, a_max_mps2=self.a_max,
                    b_comf_mps2=self.b, T_headway_s=self.T, vehicle_type=self.vehicle_type,
                    length_m=self.l_veh, politeness=round(self.politeness, 4))

    @property
    def is_ambulance(self) -> bool:
        return self.vehicle_type == "ambulance"


def free_flow_time(route_edges: list, route_moves: list, edges: dict, v0: float, cfg: SimConfig,
                   movement_geometry: dict) -> float:
    """Free-flow travel time (DECISIONS §5, §12.1): sum(edge_len / v0) + sum over junctions of
    path_len(movement) / min(v0, v_cap(movement)) (v_cap = v_limit for through paths, the
    curvature limit for turns).  The desired speed is NOT capped at v_limit on links."""
    t = sum(edges[e].length / v0 for e in route_edges)
    for _, m in route_moves:
        length, v_cap = movement_geometry[m]
        t += length / min(v0, v_cap)
    return t
