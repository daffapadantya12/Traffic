"""Signal plant: one state machine per junction (DECISIONS §6).

Traffic-engineering idea: the controller only *requests* a green approach.
The plant guarantees safety on its own: minimum and maximum green, an all-red
of at least 4 s that is extended until the junction box is empty, and
split phasing (one approach green at a time, all three of its movements).
"""

from __future__ import annotations

from .config import SimConfig
from .network import DIRS

GREEN = "GREEN"
ALLRED = "ALLRED"

# forced codes logged on every decision row
FORCED_NONE = 0
FORCED_MIN_GREEN = 1
FORCED_MAX_GREEN = 2
FORCED_ALL_RED = 3
FORCED_STARVATION = 4     # an approach with vehicles has had no green for >= starvation_s (amendment §11.1)


class SignalPlant:
    """Split-phasing plant for one junction.  `phase` is the approach (0:N 1:E
    2:S 3:W) that is green or was most recently green."""

    def __init__(self, cfg: SimConfig, junction_id: int, initial_phase: int = 0, t0: float = 0.0):
        self.cfg = cfg
        self.junction_id = junction_id
        self.phase = initial_phase
        self.state = GREEN
        self.time_in_phase = 0.0
        self.time_in_allred = 0.0
        self.pending_phase = initial_phase
        self.n_switches = 0
        self.green_onset_t = t0
        self.allred_start_t = t0
        self.next_decision_t = t0 + cfg.min_green
        # health / invariant counters
        self.allred_extensions = []          # actual all-red minus allred_min, one per switch
        self.allred_durations = []
        self.min_green_violations = 0
        self.allred_short_violations = 0
        self.box_clear_violations = 0
        self.forced_counts = {FORCED_MIN_GREEN: 0, FORCED_MAX_GREEN: 0, FORCED_ALL_RED: 0, FORCED_STARVATION: 0}
        self.green_history = []              # (t_onset, phase) for the never-two-greens check
        self.last_green_end = [t0] * 4       # when each approach last lost (or never had) its green

    # ---- queries --------------------------------------------------------------
    def is_green(self, approach_idx: int) -> bool:
        return self.state == GREEN and self.phase == approach_idx

    def at_decision_instant(self, t: float) -> bool:
        """Decision instants: green_onset + min_green, then every decision_interval while GREEN."""
        return self.state == GREEN and abs(t - self.next_decision_t) < 1e-6

    def max_green_remaining(self) -> float:
        return max(0.0, self.cfg.max_green - self.time_in_phase)

    # ---- the switch procedure ----------------------------------------------------
    def request(self, requested: int, n_by_approach: list, t: float):
        """Apply a controller request at a decision instant.

        Returns (action_applied, forced, switched, red_onset).  `red_onset` is
        True when this call started an all-red, so the world must run the
        stop-or-commit evaluation on the approach that just lost its green.
        """
        cfg = self.cfg
        forced = FORCED_NONE
        if self.state == ALLRED:                       # cannot happen at a decision instant
            forced = FORCED_ALL_RED
            self.forced_counts[forced] += 1
            return self.pending_phase, forced, False, False
        others = [k for k in range(4) if k != self.phase and n_by_approach[k] > 0]
        rotation = [(self.phase + i) % 4 for i in range(1, 4)]
        # starvation guard (amendment §11.1): an approach that has held vehicles and had no
        # green for starvation_s is served now, longest wait first (ties in rotation order).
        # It pre-empts the max-green rule, which alone can starve a third approach.
        starved = [k for k in others if t - self.last_green_end[k] >= cfg.starvation_s - 1e-9]
        if starved and self.time_in_phase >= cfg.min_green - 1e-9:
            longest = max(t - self.last_green_end[k] for k in starved)
            requested = next(k for k in rotation if k in starved and t - self.last_green_end[k] >= longest - 1e-9)
            forced = FORCED_STARVATION
            self.forced_counts[forced] += 1
        # max-green rule: only forces when another approach has vehicles (DECISIONS §6)
        elif self.time_in_phase >= cfg.max_green and others:
            best = max(n_by_approach[k] for k in others)
            requested = next(k for k in rotation if k in others and n_by_approach[k] == best)
            forced = FORCED_MAX_GREEN
            self.forced_counts[forced] += 1
        if requested == self.phase:
            self._schedule_next_decision()
            return self.phase, forced, False, False
        if self.time_in_phase < cfg.min_green - 1e-9:
            forced = FORCED_MIN_GREEN                # request ignored, so no violation
            self.forced_counts[forced] += 1
            self._schedule_next_decision()
            return self.phase, forced, False, False
        # accepted: go to all-red, the pending approach gets green when the box is empty
        self.last_green_end[self.phase] = t
        self.state = ALLRED
        self.pending_phase = requested
        self.time_in_allred = 0.0
        self.allred_start_t = t
        self.n_switches += 1
        return requested, forced, True, True

    def _schedule_next_decision(self):
        self.next_decision_t += self.cfg.decision_interval

    # ---- time advance ---------------------------------------------------------------
    def try_green_onset(self, t: float, box_empty: bool) -> bool:
        """At the start of a step: end the all-red when it has lasted >= allred_min
        and the box is empty.  Returns True on a green onset."""
        if self.state != ALLRED:
            return False
        duration = t - self.allred_start_t
        if duration + 1e-9 < self.cfg.allred_min or not box_empty:
            return False
        # invariants (2) and (5) are enforced by the two lines above; the world
        # re-checks them independently and counts violations in its own counters
        self.allred_durations.append(duration)
        self.allred_extensions.append(duration - self.cfg.allred_min)
        self.state = GREEN
        self.phase = self.pending_phase
        self.time_in_phase = 0.0
        self.green_onset_t = t
        self.next_decision_t = t + self.cfg.min_green
        self.green_history.append((t, self.phase))
        return True

    def advance(self, dt: float):
        """Advance the timers after the vehicles have moved."""
        if self.state == GREEN:
            self.time_in_phase += dt
        else:
            self.time_in_allred += dt

    def approach_name(self) -> str:
        return DIRS[self.phase]
