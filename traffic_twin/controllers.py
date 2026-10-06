"""Baseline controllers.  Every controller maps one signal_ticks feature row to
a requested approach (0:N 1:E 2:S 3:W); the plant enforces safety.

Traffic-engineering idea: max-pressure gives green to the approach whose
queue most exceeds the room on its exit links.  `max_pressure_request` is a
pure function of the row and is ALSO the dataset label `y_mp_action`.
"""

from __future__ import annotations

from .config import SimConfig
from .network import DIRS


def pressure(row: dict, approach: str) -> float:
    """pressure_a = n_a - mean(down_n over a's three exits); blocked exits are
    excluded from the mean.  If every exit is blocked the approach cannot
    discharge, so its pressure is -1e9."""
    exits = [d for d in DIRS if d != approach and not row[f"down_blocked_{d}"]]
    if not exits:
        return -1e9
    return row[f"n_{approach}"] - sum(row[f"down_n_{d}"] for d in exits) / len(exits)


def max_pressure_request(row: dict, cfg: SimConfig) -> int:
    """The max-pressure rule (DECISIONS §6) = the label function y_mp_action.

    keep during min green; otherwise switch to the highest-pressure approach
    if it beats the current one by the hysteresis (2 vehicles); at max green
    pick the highest-pressure other approach that has vehicles.
    """
    phase = int(row["phase"])
    tip = row["time_in_phase_s"]
    p = [pressure(row, a) for a in DIRS]
    if tip < cfg.min_green:
        return phase
    others_with_vehicles = [k for k in range(4) if k != phase and row[f"n_{DIRS[k]}"] > 0]
    if tip >= cfg.max_green and others_with_vehicles:
        return max(others_with_vehicles, key=lambda k: (p[k], -k))
    best = max(range(4), key=lambda k: (p[k], -k))      # ties -> lowest index (N,E,S,W order)
    if p[best] > p[phase] + cfg.mp_hysteresis:
        return best
    return phase


def ambulance_request(row: dict, cfg: SimConfig, mode: str) -> int | None:
    """Approach to give green for an ambulance, or None (DECISIONS §10).

    'naive': any ambulance with ev_eta <= 20 s.  'priority': only when the
    ambulance will arrive within the time needed to clear the queue ahead of it,
    t_clear = allred_min + start-up lost time + queued vehicles ahead x saturation headway.
    With several ambulances the one with the smallest ETA wins.
    """
    best = None
    for k, a in enumerate(DIRS):
        if not row[f"ev_present_{a}"]:
            continue
        eta = row[f"ev_eta_{a}"]
        if mode == "naive":
            limit = cfg.preempt_eta_s
        else:
            limit = cfg.allred_min + cfg.startup_lost_time_s + row[f"ev_q_ahead_{a}"] * cfg.sat_headway_s
        if eta <= limit and (best is None or eta < best[1]):
            best = (k, eta)
    return None if best is None else best[0]


def preempt_naive_request(row: dict, cfg: SimConfig) -> int:
    """Force green for an ambulance that is <= 20 s away; otherwise max-pressure."""
    k = ambulance_request(row, cfg, "naive")
    return max_pressure_request(row, cfg) if k is None else k


def priority_mp_request(row: dict, cfg: SimConfig) -> int:
    """Priority-aware teacher: max-pressure until the ambulance is close enough
    that the queue ahead of it needs the green now; back to max-pressure once it
    has crossed.  This is the label function y_mp_action (DECISIONS §10)."""
    k = ambulance_request(row, cfg, "priority")
    return max_pressure_request(row, cfg) if k is None else k


label_request = priority_mp_request       # y_mp_action on every row (= max_pressure without ambulances)


class Controller:
    """Base class: act(row) -> requested phase.  `rng` is the controller RNG stream."""
    name = "base"

    def __init__(self, cfg: SimConfig, rng=None):
        self.cfg = cfg
        self.rng = rng

    def act(self, row: dict) -> int:
        raise NotImplementedError


class FixedTime(Controller):
    """Rotation N,E,S,W with 20 s displayed green each (cycle 4*(20+4) = 96 s)."""
    name = "fixed_time"

    def act(self, row):
        phase = int(row["phase"])
        if row["time_in_phase_s"] >= self.cfg.fixed_time_green - 1e-9:
            return (phase + 1) % 4
        return phase


class LongestQueue(Controller):
    """Switch to argmax n_a if it beats the current approach by 3 vehicles."""
    name = "longest_queue"

    def act(self, row):
        phase = int(row["phase"])
        n = [row[f"n_{a}"] for a in DIRS]
        best = max(range(4), key=lambda k: (n[k], -k))
        if n[best] > n[phase] + self.cfg.longest_queue_margin:
            return best
        return phase


class MaxPressure(Controller):
    name = "max_pressure"

    def act(self, row):
        return max_pressure_request(row, self.cfg)


class MaxPressureEps(Controller):
    """Max-pressure, but with probability eps the request is a uniformly random approach."""
    name = "max_pressure_eps"

    def __init__(self, cfg, rng=None, eps: float | None = None):
        super().__init__(cfg, rng)
        self.eps = cfg.mp_eps if eps is None else eps

    def act(self, row):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, 4))
        return max_pressure_request(row, self.cfg)


class RandomDwell(Controller):
    """Hold the current green for U(5, 40) s, then request a random other approach.
    One dwell time per junction and green period (dataset coverage only)."""
    name = "random_dwell"

    def __init__(self, cfg, rng=None):
        super().__init__(cfg, rng)
        self.dwell = {}          # junction_id -> (phase, dwell_s)

    def act(self, row):
        jid = row["junction_id"]
        phase = int(row["phase"])
        if jid not in self.dwell or self.dwell[jid][0] != phase:     # new green period -> new dwell
            self.dwell[jid] = (phase, self.rng.uniform(self.cfg.random_dwell_min, self.cfg.random_dwell_max))
        if row["time_in_phase_s"] >= self.dwell[jid][1]:
            return int((phase + self.rng.integers(1, 4)) % 4)
        return phase


class PreemptNaive(Controller):
    name = "preempt_naive"

    def act(self, row):
        return preempt_naive_request(row, self.cfg)


class PriorityMP(Controller):
    name = "priority_mp"

    def act(self, row):
        return priority_mp_request(row, self.cfg)


CONTROLLERS = {
    "fixed_time": FixedTime,
    "longest_queue": LongestQueue,
    "max_pressure": MaxPressure,
    "max_pressure_eps": MaxPressureEps,
    "random_dwell": RandomDwell,
    "preempt_naive": PreemptNaive,
    "priority_mp": PriorityMP,
}


def make_controller(name: str, cfg: SimConfig, rng=None, eps: float | None = None) -> Controller:
    if name == "max_pressure_eps":
        return MaxPressureEps(cfg, rng, eps)
    return CONTROLLERS[name](cfg, rng)


def controller_eps(name: str, cfg: SimConfig) -> float:
    """The eps value stored in episodes.controller_eps (0 for deterministic controllers)."""
    return cfg.mp_eps if name == "max_pressure_eps" else 0.0

