"""Digital twin, level A (docs/design_panel_synthesis.md 'digital_twin', DECISIONS §10 hooks).

Five boxes, lock-step in one process, latency modelled in simulated time:

  P  the physical system: a `sim.World` whose state is hidden truth;
  S  `DetectorBank`: entry / stop-line loops and a queue camera per edge, every 1 s, with
     misses, doubles, Gaussian camera error, dropped frames, a 1 s delivery delay, and a
     `PhaseReport` per junction (what the lights really show); fault injection by
     `freeze_camera(edge, t0, duration)`;
  T  `QueueEstimator` + `TwinPhase`: per-edge link counts by conservation, queue estimate
     `min(EMA_0.5(camera), n_hat)`, stuck-camera rule, phase model corrected by reports;
     `TwinWorld.row()` builds the SAME signal_ticks feature row as `build_features()` but
     from the estimator (source = "twin");
  C  any controller (the trained HGB sensor model by default) called with the twin row only.
     The bridge that sits between World and the controller never hands the physical row on;
  M  `DriftMonitor`: queue_mae_truth (evaluation only), the operational drift
     |estimate - camera|, pred_err_30s and phase mismatch, with nudge / hard-resync / adopt
     actions and an event log.

The twin exists only if C never reads P's truth and T can be wrong: the ambulance channel
(`ev_*`) is the one deliberate exception (GPS/AVL, DECISIONS §10: no noise), and the
estimator is wrong by construction during a camera freeze or a lost actuation.

CLI:  python3 -m traffic_twin.twin --day --out reports/twin/day_kpis.csv [--model ...]
      python3 -m traffic_twin.twin --replay data/traffic_v1/vehicle_states_0.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import sys
import time

import numpy as np

from .config import DAILY_PROFILE, SimConfig, load_config
from .controllers import Controller, make_controller
from .features import FEATURE_COLUMNS
from .network import DIRS, DIR_INDEX, OPPOSITE, build_grid
from .signals import ALLRED, GREEN
from .sim import Scenario, World

DEFAULT_MODEL = os.path.join("reports", "ml", "models", "taskA_hgb_sensor.joblib")
DEFAULT_CONFIG = os.path.join("data", "traffic_v1", "config.json")
FALLBACK_LAMBDA = {"low": 186.102, "design": 325.678, "high": 441.992, "stress": 535.042}  # DECISIONS §11.9
SOURCE_TWIN = "twin"
DAY_S = 86400.0


# ====================================================================== S: sensing channel
class SensorFrame:
    """One second of detector readings for one edge, delivered `delay_s` after measurement."""
    __slots__ = ("t_meas", "t_deliver", "edge_id", "entry", "exit", "camera_q", "has_camera")

    def __init__(self, t_meas, t_deliver, edge_id, entry, exit_, camera_q, has_camera):
        self.t_meas, self.t_deliver, self.edge_id = t_meas, t_deliver, edge_id
        self.entry, self.exit, self.camera_q, self.has_camera = entry, exit_, camera_q, has_camera


class PhaseReport:
    __slots__ = ("t_meas", "t_deliver", "junction_id", "phase", "state", "time_in_phase")

    def __init__(self, t_meas, t_deliver, junction_id, phase, state, time_in_phase):
        self.t_meas, self.t_deliver, self.junction_id = t_meas, t_deliver, junction_id
        self.phase, self.state, self.time_in_phase = phase, state, time_in_phase


class DetectorBank:
    """Loops at both ends of every edge (entry loop, stop-line / exit loop) and a queue camera
    on every approach edge.  Counts are taken from the vehicle sets of consecutive seconds
    (a loop sees crossing events); every event is missed with p_miss and doubled with
    p_double.  Camera: max(0, round(q + N(0, 1))), frame dropped with p_drop, frozen on
    demand.  Everything is delivered `delay_s` later.  PhaseReports carry the plant's
    real display and are delivered after `delay_s` + U(0.2, 0.5) s."""

    def __init__(self, net, rng, p_miss=0.03, p_double=0.01, cam_sigma=1.0, p_drop=0.02,
                 delay_s=1.0, report_delay=(0.2, 0.5), queue_speed=2.0):
        self.net = net
        self.rng = rng
        self.p_miss, self.p_double, self.cam_sigma, self.p_drop = p_miss, p_double, cam_sigma, p_drop
        self.delay_s = delay_s
        self.report_delay = report_delay
        self.queue_speed = queue_speed
        self.prev_ids = {eid: set() for eid in net.edges}
        self.freezes = {}            # edge_id -> (t0, t1, frozen value or None)
        self.n_frames = 0
        self.n_dropped = 0

    # ---- faults -------------------------------------------------------------------
    def freeze_camera(self, edge_id: str, t0: float, duration: float) -> None:
        """From t0 the camera of `edge_id` repeats its last value for `duration` seconds."""
        self.freezes[edge_id] = [t0, t0 + duration, None]

    def _loop_count(self, n_events: int) -> int:
        if n_events == 0:
            return 0
        u = self.rng.random(n_events)
        return int(np.sum(u >= self.p_miss) + np.sum(u >= 1.0 - self.p_double))

    def corrupt(self, t: float, edge, entries: int, exits: int, q_true: int | None) -> SensorFrame:
        """Apply the sensor model to the true events / true queue of one edge."""
        entry = self._loop_count(entries)
        exit_ = self._loop_count(exits)
        has_camera = q_true is not None
        cam = None
        if has_camera:
            fr = self.freezes.get(edge.id)
            if fr is not None and fr[0] <= t < fr[1]:
                if fr[2] is None:
                    fr[2] = max(0, int(round(q_true + self.rng.normal(0.0, self.cam_sigma))))
                cam = fr[2]
            elif self.rng.random() < self.p_drop:
                cam = None
                self.n_dropped += 1
            else:
                cam = max(0, int(round(q_true + self.rng.normal(0.0, self.cam_sigma))))
        self.n_frames += 1
        return SensorFrame(t, t + self.delay_s, edge.id, entry, exit_, cam, has_camera)

    def measure(self, world: World, t: float):
        """Read the physical world at integer second t -> (frames, reports), both to be
        delivered later.  This is the ONLY place the sensing channel touches P."""
        frames = []
        for edge in self.net.edges.values():
            cur = set()
            q_true = 0
            for lane in edge.lanes:
                for v in lane:
                    cur.add(v.id)
                    if v.v < self.queue_speed:
                        q_true += 1
            prev = self.prev_ids[edge.id]
            entries = len(cur - prev)
            exits = len(prev - cur)
            self.prev_ids[edge.id] = cur
            frames.append(self.corrupt(t, edge, entries, exits, q_true if edge.junction_at_dest >= 0 else None))
        reports = []
        for plant in world.plants:
            lat = self.rng.uniform(*self.report_delay)
            reports.append(PhaseReport(t, t + self.delay_s + lat, plant.junction_id, plant.phase,
                                       plant.state, plant.time_in_phase))
        return frames, reports


# ====================================================================== T: state estimator
class EdgeEstimate:
    """Estimator state of one edge."""
    __slots__ = ("edge_id", "storage", "n_lanes", "t_travel", "has_camera", "n_hat", "entries", "last_anchor_t",
                 "cam_ema", "cam_last", "cam_t", "same_count", "loops_same", "innov_count", "weight",
                 "recover_count", "queue_est", "n_stuck_events", "stuck_since", "last_frame_t")

    def __init__(self, edge, v_limit: float, has_camera: bool):
        self.edge_id = edge.id
        self.storage = edge.storage
        self.n_lanes = edge.n_lanes
        self.t_travel = edge.length / v_limit          # free-flow traversal time of the link
        self.has_camera = has_camera
        self.n_hat = 0.0
        self.entries = collections.deque()             # (t_meas, entry count)
        self.last_anchor_t = 0.0
        self.cam_ema = None
        self.cam_last = None
        self.cam_t = -1.0
        self.same_count = 0
        self.loops_same = 0
        self.innov_count = 0
        self.weight = 1.0                              # camera weight: 1 trusted, 0 stuck (coast on loops)
        self.recover_count = 0
        self.queue_est = 0.0
        self.n_stuck_events = 0
        self.stuck_since = math.nan
        self.last_frame_t = -1.0

    def recent_entries(self, t: float, window: float) -> int:
        return sum(c for tm, c in self.entries if tm > t - window)

    def loop_queue(self, t: float) -> float:
        """Queue proxy from loops alone: vehicles that have overstayed the free-flow time."""
        return max(0.0, self.n_hat - self.recent_entries(t, self.t_travel))


class QueueEstimator:
    """Level-A estimator (synthesis 'digital_twin', T): per edge n_hat by conservation of
    loop counts, re-anchored to the camera when the edge is observed empty and at least
    every `anchor_s`; queue_est = min(EMA_0.5(camera_q), n_hat) >= 0; stuck-camera rule:
    value unchanged for `stuck_frames` frames while the loops count, or innovation
    |camera - predicted| >= `innov_veh` for `innov_frames` consecutive frames -> camera
    weight 0 (coast on loops), event logged; the camera is trusted again after
    `recover_frames` changing, plausible readings."""

    def __init__(self, net, cfg: SimConfig, ema=0.5, anchor_s=60.0, empty_quiet_s=10.0,
                 stuck_frames=15, innov_veh=3.0, innov_frames=5, recover_frames=3, log=None):
        self.cfg = cfg
        self.ema, self.anchor_s, self.empty_quiet_s = ema, anchor_s, empty_quiet_s
        self.stuck_frames, self.innov_veh, self.innov_frames, self.recover_frames = stuck_frames, innov_veh, innov_frames, recover_frames
        self.edges = {e.id: EdgeEstimate(e, cfg.v_limit, e.junction_at_dest >= 0) for e in net.edges.values()}
        self.log = log if log is not None else []
        self.n_stuck_events = 0
        self.n_recoveries = 0
        self.arr_window = cfg.arr_window_s

    def update(self, f: SensorFrame) -> None:
        st = self.edges[f.edge_id]
        t = f.t_meas
        st.last_frame_t = t
        st.n_hat = min(float(st.storage), max(0.0, st.n_hat + f.entry - f.exit))
        st.entries.append((t, f.entry))
        cutoff = t - max(self.arr_window, self.anchor_s)
        while st.entries and st.entries[0][0] <= cutoff:
            st.entries.popleft()
        if f.has_camera and f.camera_q is not None:
            cam = f.camera_q
            changed = st.cam_last is None or cam != st.cam_last
            if changed:
                st.same_count, st.loops_same = 0, 0
            else:
                st.same_count += 1
                st.loops_same += f.entry + f.exit
            innov = abs(cam - st.queue_est)
            st.innov_count = st.innov_count + 1 if innov >= self.innov_veh else 0
            if st.weight > 0.0:
                frozen = (st.same_count >= self.stuck_frames and st.loops_same > 0
                          and (cam > 0 or st.loop_queue(t) >= 2.0))
                if frozen or st.innov_count >= self.innov_frames:
                    st.weight = 0.0
                    st.recover_count = 0
                    st.stuck_since = t
                    st.n_stuck_events += 1
                    self.n_stuck_events += 1
                    self.log.append(dict(t=t, edge_id=st.edge_id, event="camera_stuck",
                                         detail="frozen value" if frozen else "innovation",
                                         value=cam, estimate=round(st.queue_est, 2)))
            else:
                plausible = cam <= st.n_hat + self.innov_veh
                st.recover_count = st.recover_count + 1 if (changed and plausible) else 0
                if st.recover_count >= self.recover_frames:
                    st.weight = 1.0
                    st.cam_ema = float(cam)
                    st.innov_count = 0
                    st.stuck_since = math.nan
                    self.n_recoveries += 1
                    self.log.append(dict(t=t, edge_id=st.edge_id, event="camera_recovered", detail="",
                                         value=cam, estimate=round(st.queue_est, 2)))
            st.cam_last, st.cam_t = cam, t
            if st.weight > 0.0:
                st.cam_ema = float(cam) if st.cam_ema is None else (1.0 - self.ema) * st.cam_ema + self.ema * cam
                # the queued vehicles the camera sees are on the link: a lower bound on n_hat
                st.n_hat = max(st.n_hat, st.cam_ema)
                # re-anchor n_hat: observed empty (camera 0 and quiet loops), or every anchor_s
                quiet = st.recent_entries(t, self.empty_quiet_s) == 0
                if cam == 0 and quiet and f.exit == 0:
                    st.n_hat = 0.0
                    st.last_anchor_t = t
                elif t - st.last_anchor_t >= self.anchor_s:
                    # camera count + recent entries bounds the link count from below; vehicles
                    # cannot linger unqueued for more than ~3 free-flow traversal times
                    lo = cam + st.recent_entries(t, st.t_travel)
                    hi = cam + st.recent_entries(t, 3.0 * st.t_travel)
                    st.n_hat = min(float(st.storage), min(max(st.n_hat, lo), max(hi, lo)))
                    st.last_anchor_t = t
        self._refresh(st, t)

    def _refresh(self, st: EdgeEstimate, t: float) -> None:
        if st.has_camera and st.weight > 0.0 and st.cam_ema is not None:
            st.queue_est = max(0.0, min(st.cam_ema, st.n_hat))
        else:
            st.queue_est = st.loop_queue(t)

    # ---- monitor actions ---------------------------------------------------------
    def nudge(self, edge_id: str, t: float) -> None:
        """Soft drift: re-anchor n_hat to the camera count (never below it)."""
        st = self.edges[edge_id]
        if st.cam_last is not None and st.weight > 0.0:
            st.n_hat = max(st.n_hat, float(st.cam_last))
            st.last_anchor_t = t
            self._refresh(st, t)

    def hard_resync(self, edge_id: str, t: float) -> None:
        """Hard drift: rebuild the edge estimate from the latest camera frame."""
        st = self.edges[edge_id]
        if st.cam_last is not None:
            st.cam_ema = float(st.cam_last)
            st.n_hat = min(float(st.storage), st.cam_last + st.recent_entries(t, st.t_travel))
            st.last_anchor_t = t
        self._refresh(st, t)

    def arrivals(self, edge_id: str, t: float) -> int:
        return self.edges[edge_id].recent_entries(t, self.arr_window)


class TwinPhase:
    """The twin's model of one plant.  It assumes its own command takes effect after the
    actuation delay (all-red of allred_min, then green on the requested approach) and is
    corrected by the delayed PhaseReports: a report older than the model's last transition
    is stale and ignored; an expected transition (the green the model is waiting for, or an
    all-red extended beyond allred_min) is taken over at once; any other disagreement
    (a lost command, a forced switch by the plant) must persist for more than `mismatch_s`
    before the model adopts the reported phase (logged as an actuation divergence)."""

    def __init__(self, cfg: SimConfig, junction_id: int, mismatch_s: float = 3.0):
        self.cfg = cfg
        self.junction_id = junction_id
        self.phase = 0
        self.state = GREEN
        self.time_in_phase = 0.0
        self.pending = None                 # (phase, t_apply)
        self.allred_start = math.nan
        self.pending_phase = 0
        self.mismatch = 0.0
        self.mismatch_s = mismatch_s
        self.n_adopted = 0
        self.n_plant_forced = 0             # switches the plant made on its own (max-green / starvation rules)
        self.last_report = None
        self.last_transition_t = -1.0
        self.target_known = True            # False while waiting for a green whose approach the plant chose
        self.last_green_end = [0.0] * 4     # the twin's copy of the plant's starvation clock

    def command(self, requested: int, t: float, delay: float) -> None:
        if requested != self.phase and self.state == GREEN and self.pending is None:
            self.pending = (requested, t + delay)

    def _to_allred(self, t: float, pending_phase: int, when: float | None = None, target_known: bool = True) -> None:
        self.pending_phase = pending_phase
        self.pending = None
        self.state = ALLRED
        self.allred_start = t
        self.last_transition_t = t if when is None else when
        self.target_known = target_known
        self.last_green_end[self.phase] = t

    def _to_green(self, t: float, phase: int, tip: float = 0.0, when: float | None = None) -> None:
        self.state = GREEN
        self.phase = phase
        self.pending = None
        self.pending_phase = phase
        self.time_in_phase = tip
        self.last_transition_t = t if when is None else when
        self.target_known = True

    def plant_rule_expected(self, t: float) -> bool:
        """Could the plant have switched on its own right now?  Max-green rule (time in phase
        at max_green) or the starvation guard (another approach without green for starvation_s)."""
        cfg = self.cfg
        if self.time_in_phase >= cfg.max_green - cfg.decision_interval - 1e-6:
            return True
        return any(t - self.last_green_end[k] >= cfg.starvation_s - cfg.decision_interval for k in range(4) if k != self.phase)

    def advance(self, t: float, dt: float) -> None:
        if self.pending is not None and t >= self.pending[1] - 1e-9 and self.state == GREEN:
            self._to_allred(t, self.pending[0])
        if self.state == GREEN:
            self.time_in_phase += dt
        elif self.target_known and t - self.allred_start >= self.cfg.allred_min - 1e-9:
            self._to_green(t, self.pending_phase)      # unknown target (plant-forced switch): wait for the report

    def on_report(self, rep: PhaseReport, t: float) -> bool:
        """Returns True when the model adopted an unexpected reported phase (divergence)."""
        self.last_report = rep
        if rep.t_meas < self.last_transition_t - 1e-9:
            return False                                   # stale: measured before the model moved
        age = t - rep.t_meas
        if rep.state == ALLRED:                            # the plant's phase during all-red = the one that lost green
            if self.state == ALLRED:                       # as expected; the model assumes green at allred_min
                self.mismatch = 0.0
                return False
            if self.pending is not None:                   # my command, applied before the model moved
                target, known, unexpected = self.pending[0], True, False
            elif rep.phase != self.phase:                  # the all-red of my own switch, extended beyond allred_min
                target, known, unexpected = self.phase, self.target_known, False
            else:                                          # the plant switched away from my phase on its own:
                target, known = self.phase, False          # max-green / starvation rule (expected, target unknown)
                unexpected = not self.plant_rule_expected(t)
                self.n_plant_forced += 1
            self.phase = rep.phase
            self._to_allred(t, target, when=rep.t_meas, target_known=known)
            self.mismatch = 0.0
            return self._count_adopt() if unexpected else False
        # rep.state == GREEN
        if self.state == GREEN and rep.phase == self.phase:
            self.mismatch = 0.0
            self.time_in_phase = rep.time_in_phase + age   # keep the clock honest (report age added)
            return False
        if self.state == ALLRED and (rep.phase == self.pending_phase or not self.target_known):
            self._to_green(t, rep.phase, rep.time_in_phase + age, when=rep.t_meas)   # the green I was waiting for
            self.mismatch = 0.0
            return False
        # model GREEN(X) or waiting for Y, report GREEN(Z): lost command / forced switch -> 3 s rule
        self.mismatch += 1.0
        if self.mismatch > self.mismatch_s:
            self._to_green(t, rep.phase, rep.time_in_phase + age, when=rep.t_meas)
            self.mismatch = 0.0
            return self._count_adopt()
        return False

    def _count_adopt(self) -> bool:
        self.n_adopted += 1
        return True

    def max_green_remaining(self) -> float:
        return max(0.0, self.cfg.max_green - self.time_in_phase)


class TwinRow(dict):
    """A signal_ticks feature row whose observable columns come from the estimator."""
    source = SOURCE_TWIN


TWIN_ROW_KEYS = tuple(FEATURE_COLUMNS) + ("source",)


# ====================================================================== M: drift monitor
class DriftMonitor:
    """Per junction per second: queue_mae_truth (evaluation only), the operational drift
    mean |estimate - latest camera| over trusted cameras, pred_err_30s (persistence
    prediction made 30 s ago vs the camera now) and phase_mismatch.  Soft threshold ->
    nudge; hard threshold sustained -> hard resync; the phase rule lives in TwinPhase."""

    def __init__(self, soft_veh=2.0, hard_veh=4.0, hard_sustain_s=30.0, horizon_s=30.0, keep_rows=1200):
        self.soft_veh, self.hard_veh, self.hard_sustain_s, self.horizon_s = soft_veh, hard_veh, hard_sustain_s, horizon_s
        self.hard_since = {}
        self.history = {}                    # jid -> deque of (t, [est per approach])
        self.rows = collections.deque(maxlen=keep_rows)
        self.n_nudges = 0
        self.n_hard_resyncs = 0
        self.n_phase_adoptions = 0
        self.sum_mae = 0.0
        self.sum_pred_err = 0.0
        self.n_pred = 0
        self.n_rows = 0
        self.sum_mismatch = 0

    def tick(self, twin: "TwinWorld", t: float) -> list:
        est = twin.estimator
        out = []
        for J in twin.net.junctions:
            jid = J.id
            ests, trues, cams, ws = [], [], [], []
            for a in DIRS:
                eid = J.approach_edges[a]
                st = est.edges[eid]
                ests.append(st.queue_est)
                trues.append(twin.true_queue(eid))
                cams.append(st.cam_last if st.cam_last is not None else st.queue_est)
                ws.append(st.weight)
            mae_truth = float(np.mean([abs(e - q) for e, q in zip(ests, trues)]))
            trusted = [(e, c) for e, c, w in zip(ests, cams, ws) if w > 0.0]
            obs_drift = float(np.mean([abs(e - c) for e, c in trusted])) if trusted else 0.0
            hist = self.history.setdefault(jid, collections.deque())
            hist.append((t, list(ests)))
            pred_err = math.nan
            while hist and hist[0][0] < t - self.horizon_s - 1e-9:
                hist.popleft()
            if hist and abs(hist[0][0] - (t - self.horizon_s)) < 1e-9:
                old = hist[0][1]
                pe = [abs(o - c) for o, c, w in zip(old, cams, ws) if w > 0.0]
                if pe:
                    pred_err = float(np.mean(pe))
                    self.sum_pred_err += pred_err
                    self.n_pred += 1
            plant = twin.world.plants[jid]
            tp = twin.phases[jid]
            mismatch = int(plant.phase != tp.phase or plant.state != tp.state)
            action = ""
            if obs_drift > self.hard_veh:
                since = self.hard_since.setdefault(jid, t)
                if t - since >= self.hard_sustain_s:
                    for a in DIRS:
                        est.hard_resync(J.approach_edges[a], t)
                    self.hard_since[jid] = t
                    self.n_hard_resyncs += 1
                    action = "hard_resync"
                    twin.log_event(t, f"J{jid} hard resync (drift {obs_drift:.1f} veh for {self.hard_sustain_s:.0f} s)")
            else:
                self.hard_since.pop(jid, None)
            if action == "" and obs_drift > self.soft_veh:
                for a in DIRS:
                    est.nudge(J.approach_edges[a], t)
                self.n_nudges += 1
                action = "nudge"
            self.sum_mae += mae_truth
            self.sum_mismatch += mismatch
            self.n_rows += 1
            row = dict(t=t, junction_id=jid, queue_mae_truth=round(mae_truth, 3), obs_drift=round(obs_drift, 3),
                       pred_err_30s=round(pred_err, 3) if not math.isnan(pred_err) else math.nan,
                       phase_mismatch=mismatch, action=action)
            self.rows.append(row)
            out.append(row)
        return out


# ====================================================================== C: the bridge
class TwinBridge(Controller):
    """Sits where `World` expects its controller.  `act(physical_row)` reads only the key
    `junction_id` from the physical row, builds the twin row and hands THAT to the inner
    controller.  The returned request passes through the actuator model: with p_fail the
    command is lost (the plant keeps its phase) while the twin believes it was applied."""
    name = "twin_bridge"

    def __init__(self, twin: "TwinWorld", inner: Controller, p_fail=0.005, act_delay=(0.2, 0.5)):
        super().__init__(twin.cfg, None)
        self.twin = twin
        self.inner = inner
        self.p_fail = p_fail
        self.act_delay = act_delay
        self.n_decisions = 0
        self.n_lost = 0

    def act(self, physical_row: dict) -> int:
        jid = int(physical_row["junction_id"])
        twin = self.twin
        trow = twin.row(jid)
        requested = int(self.inner.act(trow))
        self.n_decisions += 1
        twin.tick_counts[jid] += 1
        twin.data_age_sum += twin.t - twin.estimator.edges[twin.net.junctions[jid].approach_edges["N"]].last_frame_t
        # ---- actuator (physical side): delayed, occasionally lost ---------------------
        delay = float(twin.rng_twin.uniform(*self.act_delay))
        twin.phases[jid].command(requested, twin.t, delay)
        if requested != trow["phase"] and twin.rng_twin.random() < self.p_fail:
            self.n_lost += 1
            twin.log_event(twin.t, f"J{jid} actuation lost (command {DIRS[requested]} dropped)")
            return int(physical_row["phase"])            # the plant did not get the command
        return requested


# ====================================================================== the twin world
class TwinWorld:
    """Physical World + sensing + estimator + monitor + a controller that sees estimates only.
    Runs for any length (day counter, no cool-down) and exposes frame() for the web app."""

    def __init__(self, cfg: SimConfig | None = None, lambda_base_vph: float | None = None, demand_level: str = "design",
                 controller: str = "priority_mp", model_path: str | None = None, seed: int = 20261003,
                 lambda_ev_vph: float = 4.0, profile: str = "daily", start_hour: float = 0.0,
                 margin: float = 0.10, p_fail: float = 0.005, keep_drift_rows: int = 1200, drift_csv: str | None = None):
        base = cfg if cfg is not None else SimConfig()
        if lambda_base_vph is None:
            lambda_base_vph = (base.lambda_base_vph or FALLBACK_LAMBDA).get(demand_level, FALLBACK_LAMBDA[demand_level])
        self.cfg = base.copy(warmup_s=0.0, main_s=1e12, cooldown_s=0.0, log_trajectories=False,
                             demand_profile=profile, demand_profile_t0_s=start_hour * 3600.0)
        self.scenario = Scenario(demand_level=demand_level, lambda_base_vph=lambda_base_vph, controller="priority_mp",
                                 lambda_ev_vph=lambda_ev_vph, ev_rate="default" if lambda_ev_vph > 0 else "0")
        self.world = World(self.cfg, self.scenario, seed=seed, episode_id=0)
        self.net = self.world.net
        self.rng_twin = np.random.default_rng(np.random.SeedSequence(seed).spawn(7)[6])
        self.bank = DetectorBank(self.net, self.rng_twin, queue_speed=self.cfg.queue_speed)
        self.events = []                       # strings for the UI (bounded)
        self.estimator = QueueEstimator(self.net, self.cfg)
        self.phases = [TwinPhase(self.cfg, J.id) for J in self.net.junctions]
        self.monitor = DriftMonitor(keep_rows=keep_drift_rows)
        self.tick_counts = {J.id: 0 for J in self.net.junctions}
        self.data_age_sum = 0.0
        self.pending_frames = collections.deque()
        self.pending_reports = collections.deque()
        self.t = 0.0
        self.next_second = 0.0
        self.base_config_hash = base.config_hash()
        self.controller_label, inner = self._make_controller(controller, model_path, margin)
        self.bridge = TwinBridge(self, inner, p_fail=p_fail)
        self.world.controller = self.bridge     # the physical plant only ever talks to the bridge
        # KPIs
        self.window_s = 600.0
        self.completions = collections.deque()  # (exit_t, delay, is_amb)
        self.switch_times = collections.deque()
        self.lc_times = collections.deque()      # lane changes in the KPI window
        self.near_miss_times = collections.deque()
        self.last_lc_total = 0
        self._safety_idx = 0
        self.n_finished_total = 0
        self.last_switch_total = 0
        self.ambulances_today = 0
        self.amb_seen = set()
        self.day_start_t = 0.0
        self.day_amb_base = 0
        self.n_gridlocks = 0
        self.inv_prev = 0
        self.drift_csv = drift_csv
        self._drift_writer = None
        if drift_csv:
            os.makedirs(os.path.dirname(drift_csv) or ".", exist_ok=True)
            self._drift_fh = open(drift_csv, "w", newline="")
            self._drift_writer = csv.DictWriter(self._drift_fh, fieldnames=["t", "junction_id", "queue_mae_truth", "obs_drift",
                                                                              "pred_err_30s", "phase_mismatch", "action"])
            self._drift_writer.writeheader()
        self.hour_stats = []                    # per-hour aggregates (for the day run)
        self._hour_acc = self._new_hour_acc()
        self._last_frame_t = -1.0
        self.speed_achieved = 0.0

    # ---- construction helpers --------------------------------------------------------
    def _make_controller(self, name: str, model_path: str | None, margin: float):
        if name in ("ml", "ml_sensor", "ml_v1"):
            path = model_path or DEFAULT_MODEL
            if os.path.exists(path):
                import joblib
                from .config import GENERATOR_VERSION
                from .evaluate import MLController          # imports sklearn lazily
                bundle = joblib.load(path)
                if bundle.get("task") != "A" or bundle.get("feature_set") != "sensor":
                    raise ValueError(f"{path}: the twin feeds sensor-estimated rows, so it needs a Task A model on the "
                                     f"sensor feature set (got task={bundle.get('task')!r}, feature_set={bundle.get('feature_set')!r})")
                stamp = (bundle.get("generator_version"), bundle.get("config_hash"))
                if stamp != (GENERATOR_VERSION, self.base_config_hash):
                    msg = (f"model trained on {stamp[0]} / {stamp[1]}, simulator is {GENERATOR_VERSION} / "
                           f"{self.base_config_hash}: retrain before trusting its decisions")
                    print(f"warning: {msg}", file=sys.stderr, flush=True)
                    self.log_event(0.0, f"warning: {msg}")
                ctrl = MLController(self.cfg, None, bundle=bundle, margin=margin)
                return f"ML ({os.path.basename(path).replace('.joblib', '')}, twin rows)", ctrl
            return "priority_mp (model not found)", make_controller("priority_mp", self.cfg, self.world.rng_controller)
        return name, make_controller(name, self.cfg, self.world.rng_controller)

    def _new_hour_acc(self):
        return dict(spawned0=self.world.n_spawn_requests if hasattr(self, "world") else 0, car_delays=[], ev_delays=[],
                    completions=0, mae_sum=0.0, mae_n=0, resyncs0=0, stuck0=0, inv0=0, nudges0=0, wall0=time.perf_counter(),
                    t0=0.0, gridlocks0=0, lost0=0, adopted0=0, near_misses=0,
                    lc0=self.world.n_lane_changes if hasattr(self, "world") else 0)

    # ---- time of day -------------------------------------------------------------------
    @property
    def tod(self) -> float:
        return (self.t + self.cfg.demand_profile_t0_s) % DAY_S

    @property
    def day(self) -> int:
        return int((self.t + self.cfg.demand_profile_t0_s) // DAY_S) + 1

    @property
    def hour(self) -> float:
        return self.tod / 3600.0

    def clock(self, t: float | None = None) -> str:
        tod = self.tod if t is None else (t + self.cfg.demand_profile_t0_s) % DAY_S
        return f"{int(tod // 3600):02d}:{int(tod % 3600 // 60):02d}:{int(tod % 60):02d}"

    @staticmethod
    def demand_label(hour: float) -> str:
        return ("night" if hour < 5 else "early morning" if hour < 7 else "morning peak" if hour < 10
                else "midday" if hour < 16 else "evening peak" if hour < 20 else "evening")

    # ---- truth accessors (evaluation / sensing only; never used to build a row) --------------
    def true_queue(self, edge_id: str) -> int:
        edge = self.net.edges[edge_id]
        return sum(1 for lane in edge.lanes for v in lane if v.v < self.cfg.queue_speed)

    def log_event(self, t: float, text: str) -> None:
        self.events.append(f"{self.clock(t)} {text}")
        if len(self.events) > 60:
            del self.events[:-60]

    # ---- the twin row ---------------------------------------------------------------------
    def gps_channel(self, J) -> dict:
        """Ambulance channel (DECISIONS §10): GPS/AVL, no noise.  By the task definition the
        ev_* columns are truth: this is the one feed of the row that is not estimated, and it
        is the same computation as in build_features()."""
        cfg = self.cfg
        out = {}
        for a in DIRS:
            edge = self.net.edges[J.approach_edges[a]]
            best = None
            for lane in edge.lanes:
                for v in lane:
                    if v.is_ambulance and edge.stop_s - v.s <= cfg.ev_horizon_m and (best is None or v.s > best.s):
                        best = v
            if best is None:
                out[f"ev_present_{a}"], out[f"ev_dist_{a}"], out[f"ev_eta_{a}"], out[f"ev_q_ahead_{a}"] = False, -1.0, -1.0, -1
            else:
                dist = max(edge.stop_s - best.s, 0.0)
                out[f"ev_present_{a}"] = True
                out[f"ev_dist_{a}"] = round(dist, 2)
                out[f"ev_eta_{a}"] = round(dist / max(best.v, cfg.ev_eta_min_speed), 2)
                out[f"ev_q_ahead_{a}"] = sum(1 for v in edge.lanes[best.lane] if v.s > best.s and v.v < cfg.queue_speed)
        return out

    def row(self, jid: int) -> TwinRow:
        """The signal_ticks feature row of junction jid built from estimates only."""
        cfg = self.cfg
        J = self.net.junctions[jid]
        tp = self.phases[jid]
        est = self.estimator
        t = self.t
        row = TwinRow(episode_id=0, junction_id=jid, t=t, tick_idx=self.tick_counts[jid],
                      is_warmup=False, is_cooldown=False, is_tail=False,
                      phase=tp.phase, time_in_phase_s=round(tp.time_in_phase, 3),
                      max_green_remaining_s=round(tp.max_green_remaining(), 3))
        for a in DIRS:
            st = est.edges[J.approach_edges[a]]
            q = st.queue_est
            row[f"q_{a}"] = round(q, 3)
            row[f"n_{a}"] = round(max(st.n_hat, q), 3)
            if st.n_lanes > 1:                       # per-lane split is unobserved: even split
                row[f"q_{a}_l0"] = round(q / 2.0, 3)
                row[f"q_{a}_l1"] = round(q / 2.0, 3)
            else:
                row[f"q_{a}_l0"] = row[f"q_{a}_l1"] = -1
            row[f"arr_{a}_30s"] = est.arrivals(J.approach_edges[a], t)
        for d in DIRS:
            st = est.edges[J.exit_edges[d]]
            n = max(st.n_hat, st.queue_est)
            row[f"down_n_{d}"] = round(n, 3)
            row[f"down_occ_{d}"] = round(n / st.storage, 4)
            row[f"down_blocked_{d}"] = bool(n >= st.storage - st.n_lanes)
        for d in DIRS:
            nb = J.neighbours[d]
            if nb < 0:
                row[f"nb_{d}_exists"], row[f"nb_{d}_phase"], row[f"nb_{d}_green_toward_me"] = False, -1, False
            else:
                np_ = self.phases[nb]
                row[f"nb_{d}_exists"] = True
                row[f"nb_{d}_phase"] = np_.phase
                row[f"nb_{d}_green_toward_me"] = (np_.state == GREEN and np_.phase != DIRS.index(OPPOSITE[d]))
        row.update(self.gps_channel(J))
        for a in DIRS:
            row[f"obs_q_{a}"] = row[f"q_{a}"]
        row["source"] = SOURCE_TWIN
        return row

    # ---- stepping ---------------------------------------------------------------------------
    def advance(self, seconds: float) -> None:
        """Advance the physical world and the twin by `seconds` of simulated time."""
        n = max(1, int(round(seconds / self.cfg.dt)))
        for _ in range(n):
            self.step()

    def step(self) -> None:
        w = self.world
        dt = self.cfg.dt
        w.step()
        self.t = w.t
        for tp in self.phases:
            tp.advance(self.t, dt)
        self._deliver(self.t)
        if w.gridlock:                      # count, log and re-arm: the demo never stops
            self.n_gridlocks += 1
            self.log_event(self.t, f"gridlock detected at J{w.gridlock_junction}" if w.gridlock_junction >= 0 else "network gridlock detected")
            w.gridlock, w.gridlock_t = False, math.nan
            w.last_crossing_t = self.t
            for jid in w.last_crossing_t_j:
                w.last_crossing_t_j[jid] = self.t
        if abs(self.t - round(self.t)) < 1e-6:
            self._second(round(self.t))

    def _deliver(self, t: float) -> None:
        """Hand every frame / report whose delivery time has come to the estimator (each step)."""
        while self.pending_frames and self.pending_frames[0].t_deliver <= t + 1e-9:
            self.estimator.update(self.pending_frames.popleft())
        while self.pending_reports and self.pending_reports[0].t_deliver <= t + 1e-9:
            rep = self.pending_reports.popleft()
            if self.phases[rep.junction_id].on_report(rep, t):
                self.monitor.n_phase_adoptions += 1
                self.log_event(t, f"J{rep.junction_id} twin adopted reported phase {DIRS[rep.phase]}")

    def _second(self, t: float) -> None:
        # S: measure now; delivery happens `delay_s` later (see _deliver)
        frames, reports = self.bank.measure(self.world, t)
        self.pending_frames.extend(frames)
        self.pending_reports.extend(reports)
        self._deliver(t)
        for ev in self.estimator.log:
            self.log_event(ev["t"], f"{ev['edge_id']} camera {'stuck' if ev['event'] == 'camera_stuck' else 'recovered'}"
                           + (f" ({ev['detail']})" if ev["detail"] else ""))
        self.estimator.log.clear()
        # M
        rows = self.monitor.tick(self, t)
        if self._drift_writer is not None:
            self._drift_writer.writerows(rows)
        for r in rows:
            self._hour_acc["mae_sum"] += r["queue_mae_truth"]
            self._hour_acc["mae_n"] += 1
        # KPIs
        self._kpi_second(t)
        if int(self.tod // 3600) != int(((t - 1) + self.cfg.demand_profile_t0_s) % DAY_S // 3600) and t > 0:
            self._close_hour(t)
        if t > 0 and self.tod < 1e-6:
            self.ambulances_today = 0
            self.amb_seen.clear()
        self._trim_logs(t)

    def _kpi_second(self, t: float) -> None:
        w = self.world
        fin = w.finished
        while fin:                            # consume (and free) every vehicle that despawned
            veh = fin.pop(0)
            self.n_finished_total += 1
            delay = (veh.enter_t - veh.spawn_request_t) + (veh.exit_t - veh.enter_t) - veh.free_flow_time
            self.completions.append((veh.exit_t, delay, veh.is_ambulance))
            self._hour_acc["completions"] += 1
            (self._hour_acc["ev_delays"] if veh.is_ambulance else self._hour_acc["car_delays"]).append(delay)
            if veh.is_ambulance:
                self.log_event(veh.exit_t, f"ambulance #{veh.id} arrived at {veh.destination}, delay {delay:.0f} s")
        while self.completions and self.completions[0][0] < t - self.window_s:
            self.completions.popleft()
        total_sw = sum(p.n_switches for p in w.plants)
        for _ in range(total_sw - self.last_switch_total):
            self.switch_times.append(t)
        self.last_switch_total = total_sw
        while self.switch_times and self.switch_times[0] < t - self.window_s:
            self.switch_times.popleft()
        for _ in range(w.n_lane_changes - self.last_lc_total):
            self.lc_times.append(t)
        self.last_lc_total = w.n_lane_changes
        while self.lc_times and self.lc_times[0] < t - self.window_s:
            self.lc_times.popleft()
        rows = w.safety.rows
        for r in rows[self._safety_idx:]:
            if r["near_miss"] and not r["collision"]:
                self.near_miss_times.append(r["t_start"])
                self._hour_acc["near_misses"] += 1
                self.log_event(r["t_start"], f"near miss ({r['type'].replace('_', ' ')}): #{r['veh_a']} / #{r['veh_b']}, "
                                             f"TTC {r['min_ttc_s']:.1f} s")
        self._safety_idx = len(rows)
        if len(rows) > 2000:
            del rows[:-500]
            self._safety_idx = len(rows)
        while self.near_miss_times and self.near_miss_times[0] < t - self.window_s:
            self.near_miss_times.popleft()
        for veh in w.active:
            if veh.is_ambulance and veh.id not in self.amb_seen:
                self.amb_seen.add(veh.id)
                self.ambulances_today += 1
                self.log_event(t, f"ambulance #{veh.id} entered at {veh.origin} -> {veh.destination}")

    def kpis(self) -> dict:
        car = [d for _, d, amb in self.completions if not amb]
        ev = [d for _, d, amb in self.completions if amb]
        return dict(
            n_vehicles=self.world.n_in_network(),
            car_delay_s=round(float(np.mean(car)), 1) if car else None,
            ev_delay_s=round(float(np.mean(ev)), 1) if ev else None,
            throughput_vph=round(len(self.completions) * 3600.0 / self.window_s, 0),
            switches_per_min=round(len(self.switch_times) / len(self.world.plants) / (self.window_s / 60.0), 2),
            ambulances_today=self.ambulances_today,
            speed_achieved=round(self.speed_achieved, 1),
            lane_changes_per_min=round(len(self.lc_times) / (self.window_s / 60.0), 1),
            near_misses_recent=len(self.near_miss_times),
            collisions=self.world.safety.n_collisions_1d + self.world.safety.n_collisions_zone,
        )

    def invariant_violations(self) -> int:
        w = self.world
        inv = w.invariant_counts()
        conservation = w.n_spawn_requests - (self.n_finished_total + len(w.finished) + w.n_in_network() + w.n_entry_queued())
        return sum(v for k, v in inv.items() if k != "conservation_error") + abs(conservation)

    def _close_hour(self, t: float) -> None:
        acc = self._hour_acc
        w = self.world
        hour_idx = int(((t - 1) + self.cfg.demand_profile_t0_s) % DAY_S // 3600)
        inv = self.invariant_violations()
        self.hour_stats.append(dict(
            day=int(((t - 1) + self.cfg.demand_profile_t0_s) // DAY_S) + 1, hour=hour_idx,
            demand_mult=self.cfg.demand_multiplier(t - 1),
            vehicles_spawned=w.n_spawn_requests - acc["spawned0"],
            completions=acc["completions"],
            mean_car_delay_s=round(float(np.mean(acc["car_delays"])), 1) if acc["car_delays"] else math.nan,
            mean_ev_delay_s=round(float(np.mean(acc["ev_delays"])), 1) if acc["ev_delays"] else math.nan,
            n_ambulances=len(acc["ev_delays"]),
            throughput_vph=round(acc["completions"] * 3600.0 / max(t - acc["t0"], 1.0), 0),
            mean_drift_veh=round(acc["mae_sum"] / max(acc["mae_n"], 1), 3),
            hard_resyncs=self.monitor.n_hard_resyncs - acc["resyncs0"],
            nudges=self.monitor.n_nudges - acc["nudges0"],
            stuck_sensor_events=self.estimator.n_stuck_events - acc["stuck0"],
            phase_adoptions=self.monitor.n_phase_adoptions - acc["adopted0"],
            lost_commands=self.bridge.n_lost - acc["lost0"],
            gridlocks=self.n_gridlocks - acc["gridlocks0"],
            invariant_violations=inv - acc["inv0"],
            near_misses=acc["near_misses"],
            lane_changes=w.n_lane_changes - acc["lc0"],
            n_in_network_end=w.n_in_network(),
            wall_s=round(time.perf_counter() - acc["wall0"], 1),
        ))
        self._hour_acc = self._new_hour_acc()
        self._hour_acc.update(resyncs0=self.monitor.n_hard_resyncs, nudges0=self.monitor.n_nudges,
                              stuck0=self.estimator.n_stuck_events, inv0=inv, t0=t, gridlocks0=self.n_gridlocks,
                              lost0=self.bridge.n_lost, adopted0=self.monitor.n_phase_adoptions)

    def _trim_logs(self, t: float) -> None:
        """Keep the physical World's per-episode logs bounded for an unlimited run."""
        if int(t) % 60 != 0:
            return
        w = self.world
        if len(w.signal_rows) > 2000:
            del w.signal_rows[:-800]
        if len(w.junction_1hz_rows) > 2000:
            del w.junction_1hz_rows[:-800]
        if len(w.q_total_lookup) > 4000:
            cut = t - 300
            w.q_total_lookup = {k: v for k, v in w.q_total_lookup.items() if k[1] >= cut}
        if len(w.red_commit_log) > 2000:
            del w.red_commit_log[:-500]
        if len(w.safety.pet_values) > 2000:
            del w.safety.pet_values[:-500]
        if len(w.ev_events) > 500:
            w.ev_events = {k: v for k, v in w.ev_events.items() if math.isnan(v["t_clear"]) or v["t_clear"] > t - 600}
        if len(w.state_rows) > 0:
            w.state_rows.clear()
        for p in w.plants:
            if len(p.green_history) > 4000:
                del p.green_history[:-1000]
                del p.allred_durations[:-1000]
                del p.allred_extensions[:-1000]

    # ---- the web frame ---------------------------------------------------------------------
    def vehicle_list(self) -> list:
        """[x, y, heading_deg, type (0 car / 1 ambulance), speed] of every vehicle, front bumper
        centre.  Box vehicles sit on their real path (arc or straight); a vehicle that changed
        lane less than lc_duration_s ago slides across between the two lane centres (smoothstep),
        exactly the time the old-lane follower keeps treating it as a leader (DECISIONS §12.3)."""
        cfg = self.cfg
        out = []
        t = self.world.t
        tau = cfg.lc_duration_s
        for veh in self.world.active:
            if veh.edge is not None:
                e = veh.edge
                x, y = e.lane_xy(veh.lane, veh.s, cfg)
                hd = math.atan2(e.heading[1], e.heading[0])
                age = t - veh.lc_time
                if 0.0 <= age < tau and veh.lc_prev_lane != veh.lane:
                    u = age / tau
                    f = u * u * (3.0 - 2.0 * u)
                    x0, y0 = e.lane_xy(veh.lc_prev_lane, veh.s, cfg)
                    x1, y1 = e.lane_xy(veh.lane, veh.s, cfg)
                    lateral_rate = cfg.w_lane * 6.0 * u * (1.0 - u) / tau          # d(offset)/dt of the smoothstep
                    turn = math.atan2(lateral_rate, max(veh.v, 1.0))                # nose points where it is moving
                    left = e.heading[0] * (y1 - y0) - e.heading[1] * (x1 - x0) > 0
                    x, y = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
                    hd += turn if left else -turn
            else:
                x, y, hd = veh.conn.path.pose(veh.s)
            out.append([round(x, 2), round(y, 2), round(math.degrees(hd), 1), 1 if veh.is_ambulance else 0, round(veh.v, 1)])
        return out

    def geometry(self) -> dict:
        """Static drawing geometry for the web page (one source of truth with the simulator)."""
        cfg = self.cfg
        J = self.net.junctions[0]
        paths = []
        for c in J.connectors.values():
            pts = c.path.polyline(1.0)
            paths.append(dict(id=c.id.split("_", 1)[1], movement=c.movement,
                              pts=[[round(float(px - J.x), 2), round(float(py - J.y), 2)] for px, py in pts]))
        return dict(L_block=cfg.L_block, L_stub=cfg.L_stub, grid_x=cfg.grid_x, grid_y=cfg.grid_y,
                    lanes=cfg.lanes_per_dir, w_lane=cfg.w_lane, box_half=cfg.w_box / 2.0,
                    corner_radius=cfg.corner_radius, stop_line_setback=cfg.stop_line_setback,
                    drive_side=cfg.drive_side, veh_length=cfg.l_veh, veh_width=cfg.w_veh,
                    ev_length=cfg.ev_l_veh, ev_width=cfg.ev_w_veh, junction_paths=paths)

    def twin_panel(self) -> dict:
        js = []
        for J in self.net.junctions:
            tp = self.phases[J.id]
            q_true, q_est, stuck = [], [], False
            for a in DIRS:
                st = self.estimator.edges[J.approach_edges[a]]
                q_true.append(self.true_queue(st.edge_id))
                q_est.append(round(st.queue_est, 1))
                stuck = stuck or st.weight == 0.0
            drift = float(np.mean([abs(e - q) for e, q in zip(q_est, q_true)]))
            js.append(dict(id=J.id, phase=tp.phase, state=tp.state, q_true=q_true, q_est=q_est,
                           drift=round(drift, 2), sensor_stuck=stuck))
        return dict(junctions=js)

    def frame(self) -> dict:
        hour = self.hour
        return dict(
            t=round(self.t, 1), clock=self.clock(), day=self.day, hour=round(hour, 4),
            demand_mult=self.cfg.demand_multiplier(self.t), demand_label=self.demand_label(hour),
            controller=self.controller_label, kpi=self.kpis(),
            signals=[dict(phase=p.phase, state=p.state) for p in self.world.plants],
            vehicles=self.vehicle_list(), twin=self.twin_panel(),
            profile=list(self.cfg.demand_profile_hourly if self.cfg.demand_profile == "daily" else (1.0,) * 24),
            events=list(self.events[-12:]),
        )

    def close(self) -> None:
        if self._drift_writer is not None:
            self._drift_fh.close()
            self._drift_writer = None

    def summary(self) -> dict:
        m = self.monitor
        return dict(
            t_end=self.t, controller=self.controller_label, n_decisions=self.bridge.n_decisions,
            mean_queue_mae_truth=round(m.sum_mae / max(m.n_rows, 1), 3),
            mean_pred_err_30s=round(m.sum_pred_err / max(m.n_pred, 1), 3),
            phase_mismatch_frac=round(m.sum_mismatch / max(m.n_rows, 1), 4),
            mean_data_age_s=round(self.data_age_sum / max(self.bridge.n_decisions, 1), 3),
            nudges=m.n_nudges, hard_resyncs=m.n_hard_resyncs, phase_adoptions=m.n_phase_adoptions,
            plant_forced_switches=sum(p.n_plant_forced for p in self.phases),
            stuck_sensor_events=self.estimator.n_stuck_events, camera_recoveries=self.estimator.n_recoveries,
            lost_commands=self.bridge.n_lost, gridlocks=self.n_gridlocks,
            invariant_violations=self.invariant_violations(), n_spawned=self.world.n_spawn_requests,
            n_finished=self.n_finished_total + len(self.world.finished), frames_dropped=self.bank.n_dropped,
        )


# ====================================================================== level 0: offline replay
def level0_replay(vehicle_states_csv: str, out_png: str = os.path.join("reports", "twin", "level0_drift.png"),
                  cfg: SimConfig | None = None, seed: int = 0, freeze: tuple | None = ("W0_J00", 1800.0, 60.0)) -> dict:
    """Replay a logged showcase episode (vehicle_states_<id>.csv, 1 Hz) through the sensing
    channel and the estimator; compare with the logged truth; save a drift figure.
    Returns fidelity numbers.  No new simulation code is involved."""
    import pandas as pd

    cfg = cfg or SimConfig()
    net = build_grid(cfg)
    df = pd.read_csv(vehicle_states_csv, usecols=["t", "veh_id", "edge_id", "v_mps"])
    df = df[df.edge_id.isin(list(net.edges))]
    rng = np.random.default_rng(seed)
    bank = DetectorBank(net, rng, queue_speed=cfg.queue_speed)
    if freeze is not None:
        bank.freeze_camera(*freeze)
    est = QueueEstimator(net, cfg)
    approach_edges = [e for e in net.edges.values() if e.junction_at_dest >= 0]
    cells = {}                                        # t -> edge_id -> (ids, q_true)
    for (t, eid), sub in df.groupby(["t", "edge_id"]):
        cells.setdefault(int(t), {})[eid] = (set(sub.veh_id.tolist()), int((sub.v_mps < cfg.queue_speed).sum()))
    ts = sorted(cells)
    prev = {e.id: set() for e in net.edges.values()}
    pending = collections.deque()
    rec = {e.id: [] for e in approach_edges}          # (t, q_true, q_est, weight)
    stuck_events = []
    empty = (set(), 0)
    for t in ts:
        g = cells[t]
        truth = {}
        frames = []
        for e in net.edges.values():
            cur, q_true = g.get(e.id, empty)
            truth[e.id] = q_true
            frames.append(bank.corrupt(float(t), e, len(cur - prev[e.id]), len(prev[e.id] - cur),
                                       q_true if e.junction_at_dest >= 0 else None))
            prev[e.id] = cur
        pending.extend(frames)
        while pending and pending[0].t_deliver <= t + 1e-9:
            est.update(pending.popleft())
        for ev in est.log:
            stuck_events.append(ev)
        est.log.clear()
        for e in approach_edges:
            st = est.edges[e.id]
            rec[e.id].append((t, truth[e.id], st.queue_est, st.weight))
    # fidelity numbers: note the estimate at t compares with truth at t (1 s delivery delay included)
    per_edge = {}
    for eid, rows in rec.items():
        arr = np.array(rows, dtype=float)
        err = np.abs(arr[:, 2] - arr[:, 1])
        per_edge[eid] = dict(mae=round(float(err.mean()), 3), mae_last_half=round(float(err[len(err) // 2:].mean()), 3),
                             max_q=int(arr[:, 1].max()))
    overall = round(float(np.mean([v["mae"] for v in per_edge.values()])), 3)
    # figure: per junction the sum of the four queues (truth vs estimate), drift strip below
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(net.junctions) + 1, 1, figsize=(11, 2.2 * (len(net.junctions) + 1)), sharex=True)
        for J in net.junctions:
            ax = axes[J.id]
            tt = np.array([r[0] for r in rec[J.approach_edges["N"]]])
            tru = sum(np.array([r[1] for r in rec[J.approach_edges[a]]]) for a in DIRS)
            es = sum(np.array([r[2] for r in rec[J.approach_edges[a]]]) for a in DIRS)
            ax.plot(tt, tru, color="#3b6fd6", lw=1.0, label="true queue (sum of 4 approaches)")
            ax.plot(tt, es, color="#d93c3c", lw=1.0, alpha=0.8, label="twin estimate")
            ax.set_ylabel(f"J{J.id} veh")
            if J.id == 0:
                ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax = axes[-1]
        for eid, rows in rec.items():
            arr = np.array(rows, dtype=float)
            ax.plot(arr[:, 0], np.abs(arr[:, 2] - arr[:, 1]), lw=0.6, alpha=0.5)
        if freeze is not None:
            for a in axes:
                a.axvspan(freeze[1], freeze[1] + freeze[2], color="#f0c060", alpha=0.3)
        ax.set_ylabel("|est - true| per edge")
        ax.set_xlabel("t [s]")
        fig.suptitle(f"Level-0 replay of {os.path.basename(vehicle_states_csv)}: mean |est - true| = {overall} veh"
                     + (f"; camera {freeze[0]} frozen {freeze[1]:.0f}-{freeze[1] + freeze[2]:.0f} s" if freeze else ""))
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is a bonus
        print("level0 figure skipped:", exc)
    return dict(file=vehicle_states_csv, n_seconds=len(ts), mean_queue_mae=overall, per_edge=per_edge,
                stuck_events=[dict(t=e["t"], edge_id=e["edge_id"], event=e["event"], detail=e["detail"]) for e in stuck_events],
                figure=out_png)


# ====================================================================== CLI: a full day headless
def load_base_config(path: str | None) -> SimConfig:
    p = path or DEFAULT_CONFIG
    if os.path.exists(p):
        return load_config(p)
    return SimConfig(lambda_base_vph=dict(FALLBACK_LAMBDA))


def run_day(args) -> dict:
    cfg = load_base_config(args.config)
    tw = TwinWorld(cfg, demand_level=args.demand, controller=args.controller, model_path=args.model, seed=args.seed,
                   profile=args.profile, start_hour=args.start_hour, drift_csv=args.drift_csv)
    if args.freeze:
        tw.bank.freeze_camera(args.freeze, 1800.0, 60.0)
    hours = args.hours
    t0 = time.perf_counter()
    total = int(hours * 3600)
    report_every = 3600
    for sec in range(0, total, report_every):
        tw.advance(min(report_every, total - sec))
        if args.verbose:
            h = tw.hour_stats[-1] if tw.hour_stats else {}
            print(f"t={tw.t:7.0f} {tw.clock()} day {tw.day} veh {tw.world.n_in_network():3d} wall {time.perf_counter() - t0:6.1f}s "
                  f"hour-stats: {h}", flush=True)
    wall = time.perf_counter() - t0
    tw.close()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = list(tw.hour_stats[0].keys()) if tw.hour_stats else []
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(tw.hour_stats)
    summary = tw.summary()
    summary.update(wall_s=round(wall, 1), sim_s=tw.t, speed_factor=round(tw.t / max(wall, 1e-9), 1), out=args.out,
                   demand_level=args.demand, lambda_base_vph=tw.scenario.lambda_base_vph, profile=args.profile, seed=args.seed)
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="digital twin: headless day run or level-0 replay")
    ap.add_argument("--day", action="store_true", help="run a full simulated day through the twin")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--out", default=os.path.join("reports", "twin", "day_kpis.csv"))
    ap.add_argument("--drift-csv", default=None, help="optional per-second twin_drift log")
    ap.add_argument("--config", default=None, help="config.json of the release (lambda_base per level)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--controller", default="ml", help="ml | priority_mp | max_pressure | fixed_time | ...")
    ap.add_argument("--demand", default="design")
    ap.add_argument("--profile", default="daily", choices=["daily", "flat"])
    ap.add_argument("--start-hour", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--freeze", default=None, help="edge id whose camera is frozen at t = 1800 s for 60 s")
    ap.add_argument("--replay", default=None, help="vehicle_states_<id>.csv for the level-0 replay")
    ap.add_argument("--replay-out", default=os.path.join("reports", "twin", "level0_drift.png"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.replay:
        res = level0_replay(args.replay, args.replay_out)
        with open(os.path.splitext(args.replay_out)[0] + ".json", "w") as fh:
            json.dump(res, fh, indent=2)
        print(json.dumps({k: v for k, v in res.items() if k != "per_edge"}, indent=2))
    if args.day:
        run_day(args)
    if not args.replay and not args.day:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
