"""The Fatmawati simulation (SUMO via libsumo) behind the twin, the controllers and the 3D viewer (DECISIONS §13.4).

Traffic-engineering idea: SUMO moves the vehicles (EIDM drivers, sublane model so motorcycles filter
to the motorcycle box), our layer runs the study junction's signal plant and the ambulance rules:
- the Fatmawati controller runs Dishub's 4-stage plan (program "dishub", build_net.py): S1 RA Kartini +
  TB Simatupang through, S2 TB Simatupang through + right, S3 north, S4 south; amber 2 s + all-red 2 s.
  'fixed' runs the Dishub greens 80/12/78/45 s; the adaptive controllers request a stage at 5 s decision
  instants and the plant keeps min green 5 s / max green 90 s, serves a stage with standing vehicles after
  at most 240 s of red, and runs the transition phases;
- ambulances obey the signals (priority only via the signal); a car directly ahead of an ambulance
  moves toward the kerb when the lane change is safe (SUMO checks it), a boxed-in car stays put;
- motorcycles filter anywhere on a road but ride the middle of their lane across a junction (and 3 s
  after it), because SUMO's junction model only checks crossing and merging movements against each other;
- safety: lane collisions are an invariant (must stay 0); overlaps inside junction areas are a known
  sublane-model limitation and are counted separately (n_junction_overlaps); teleports are counted.
"""

from __future__ import annotations

import math
import os
import time

from .build_net import (ALLRED_S, APPROACHES, DISHUB_GREEN_S, JUNCTION_LAT, JUNCTION_LON, STAGE_APPROACHES, STAGE_LABEL,
                        STAGE_OF_APPROACH, STAGES, TLS_ID, YELLOW_S)
from .demand import DATA, NET
from .geometry3d import raised_z

VTYPES = os.path.join(DATA, "demand", "vtypes.add.xml")
TYPE_CODE = {"car": 0, "ambulance": 1, "moto": 2, "bus": 3, "truck": 4}
ARMS = ("N", "E", "S", "W")
CONTROLLERS = ("fixed", "longest_queue", "max_pressure", "priority_mp", "preempt_naive", "random", "still")
PREEMPT_ETA_S = 20.0                                            # naive pre-emption: ambulance this close in time
INTERGREEN_S = YELLOW_S + ALLRED_S                              # amber 2 + all-red 2 (Dishub)
START_LOSS_S = 2.0                                              # start-up lost time
STARVE_S = 240.0                                                # longest red for a stage with standing vehicles
STUDY_M = 300.0                                                 # the study area: sensors lie within it
VKM_EVERY_S = 10.0
_LAYOUT = None
_RAISED = None


def sensor_layout() -> dict:
    """The sensor inventory of the current network (built once per process)."""
    global _LAYOUT
    if _LAYOUT is None:
        from .sensors import build_layout
        _LAYOUT = build_layout()
    return _LAYOUT


def display_raised() -> tuple:
    """Lanes and junctions whose heights the viewer shows: the toll road's (geometry3d.raised_parts)."""
    global _RAISED
    if _RAISED is None:
        import sumolib
        from .geometry3d import raised_parts
        _RAISED = raised_parts(sumolib.net.readNet(NET, withInternal=False))
    return _RAISED


def type_code(type_id: str) -> int:
    for k, v in TYPE_CODE.items():
        if type_id.startswith(k):
            return v
    return 0


class FatmawatiWorld:
    def __init__(self, routes: str, start_hour: float = 8.0, seed: int = 1, sublane: float = 0.8,
                 controller: str = "fixed", min_green: float = 5.0, max_green: float = 90.0, decision_s: float = 5.0,
                 lateral_mode: str = "centre", centre_hold_s: float = 3.0, sensors: bool = False, record_from: float = 0.0,
                 store_rows: bool = True, vtypes: str = VTYPES, extra_args: tuple = ()):
        if lateral_mode not in ("centre", "off"):
            raise ValueError(f"lateral_mode must be 'centre' or 'off', not {lateral_mode!r}")
        if controller not in CONTROLLERS and not controller.startswith("learned:"):
            raise ValueError(f"controller must be one of {CONTROLLERS} or 'learned:<model path>', not {controller!r}")
        if controller != "fixed" and not sensors:
            raise ValueError(f"the {controller} controller reads the virtual sensors: use sensors=True")
        import libsumo as ts
        self.ts = ts
        self.start_hour = start_hour
        additional = vtypes
        if sensors:
            from .sensors import write_additional
            additional += "," + write_additional(sensor_layout())
        cmd = ["sumo", "-n", NET, "-r", routes, "-a", additional, "--step-length", "0.5", "--lateral-resolution", str(sublane),
               "--collision.action", "warn", "--collision.check-junctions", "true", "--collision.mingap-factor", "0",
               "--time-to-teleport", "300", "--no-step-log", "true", "--no-warnings", "true", "--ignore-route-errors", "true",
               "--seed", str(seed)] + list(extra_args)
        ts.start(cmd)
        ts.trafficlight.setProgram(TLS_ID, "dishub")
        self.x0, self.y0 = ts.simulation.convertGeo(JUNCTION_LON, JUNCTION_LAT, True)
        self.controller = controller
        self.min_green, self.max_green, self.decision_s = min_green, max_green, decision_s
        self.t = 0.0
        logic = [p for p in ts.trafficlight.getAllProgramLogics(TLS_ID) if p.programID == "dishub"][0]
        self.phase_names = [ph.name for ph in logic.phases]
        self.green_idx = {st: self.phase_names.index(f"{st} green") for st in STAGES}
        self.trans_idx = {(a, b): (self.phase_names.index(f"{a}>{b} amber"), self.phase_names.index(f"{a}>{b} all-red"))
                          for a in STAGES for b in STAGES if a != b}
        self.green_stage = "S1"
        self.green_since = 0.0
        self.red_since = {st: 0.0 for st in STAGES}
        self.next_decision = min_green
        self.transition = None                     # (target stage, 0 amber / 1 all-red, start time)
        self._set(self.green_idx["S1"])
        self.n_lane_collisions = 0
        self.n_junction_overlaps = 0
        self.n_lane_collisions_study = 0               # within STUDY_M of the junction centre (the sensors' area)
        self.n_junction_overlaps_study = 0
        self.veh_km = self.veh_km_study = 0.0           # vehicle-km travelled (sampled every VKM_EVERY_S)
        import sumolib
        _net = sumolib.net.readNet(NET, withInternal=False)
        self._edges = [e.getID() for e in _net.getEdges()]
        self._study_edges = {e.getID() for e in _net.getEdges()
                             if any(math.hypot(x - self.x0, y - self.y0) <= STUDY_M for x, y in e.getShape())}
        self.n_teleports = 0
        self.n_arrived = 0
        self.n_departed = 0
        self.n_yields = 0
        self.n_ev_arrived = 0
        self.ev_travel = []
        self.ev_depart = {}
        self._coll_seen = set()
        self.events = []
        self._sub = set()
        self.lateral_mode, self.centre_hold_s = lateral_mode, centre_hold_s
        self.motos = {}                            # motorcycle id -> (alignment in force, centred until t)
        self.sensors = None
        if sensors:
            from .sensors import SensorBank
            self.sensors = SensorBank(ts, sensor_layout(), record_from=record_from, store_rows=store_rows)
        self.signal_log = []                       # controller log: one row per phase change
        self._phase, self._phase_since = None, 0.0
        self.last_request = self.green_stage
        self.preempting = None
        import numpy as np
        self.ctrl_rng = np.random.default_rng([seed, 991])   # own stream: the 'random' controller leaves traffic draws alone
        self.policy, self.n_invalid_requests = None, 0
        self.force = None          # hindsight replays: (stage, from t, ambulance id), requested until that ambulance crosses
        if controller.startswith("learned:"):                # a trained model: feature row -> stage (learn.py)
            from .learn import load_policy
            self.policy = load_policy(controller.split(":", 1)[1])
        self.wall0 = time.perf_counter()

    # ---- signal plant at Fatmawati -------------------------------------------------------------
    def approach_chain(self, arm: str, length_m: float = 300.0) -> list:
        """The approach edge and the edges of the same street upstream of it, up to length_m: the
        stretch over which an approach queue is counted (the stop-line edge alone is only ~47 m)."""
        if not hasattr(self, "_chains"):
            self._chains = {}
        if arm in self._chains:
            return self._chains[arm]
        ts = self.ts
        chain, e, dist = [], APPROACHES[arm], 0.0
        name = ts.edge.getStreetName(e)
        while e and dist < length_m:
            chain.append(e)
            dist += ts.lane.getLength(e + "_0")
            # predecessor of the same street feeding this edge
            cands = [c for c in self._pred(e) if ts.edge.getStreetName(c) in (name, "") and c not in chain]
            e = cands[0] if len(cands) == 1 else None
        self._chains[arm] = chain
        return chain

    def _pred(self, edge_id: str) -> list:
        if not hasattr(self, "_preds"):
            import sumolib
            net = sumolib.net.readNet(NET, withInternal=False)
            self._preds = {e.getID(): [i.getID() for i in e.getIncoming()] for e in net.getEdges()}
        return self._preds.get(edge_id, [])

    def approach_queue(self, arm: str) -> int:
        return sum(self.ts.edge.getLastStepHaltingNumber(e) for e in self.approach_chain(arm))

    def queues(self) -> dict:
        """Standing vehicles per approach: the queue cameras when the sensors are on, else the approach edges."""
        if self.sensors is not None:
            return {a: self.sensors.cam_now.get(f"QC_{a}", {}).get("q_veh", 0) for a in ARMS}
        return {a: self.approach_queue(a) for a in ARMS}

    def stage_queues(self, pcu: bool = False) -> dict:
        """Standing vehicles each stage would serve: the camera lane groups its green reaches."""
        out = {st: 0.0 for st in STAGES}
        key = "q_groups_pcu" if pcu else "q_groups"
        for a in ARMS:
            for g, q in self.sensors.cam_now.get(f"QC_{a}", {}).get(key, {}).items():
                for st in g.split("+"):
                    out[st] += q
        return out

    def stage_pressures(self) -> dict:
        """Max-pressure per stage: each approach's pressure (sensors.SensorBank.pressures) shared over its lane
        groups in proportion to their standing PCU, summed over the groups a stage serves."""
        p = self.sensors.pressures()
        out = {st: 0.0 for st in STAGES}
        for a in ARMS:
            grp = self.sensors.cam_now.get(f"QC_{a}", {}).get("q_groups_pcu", {})
            tot = sum(grp.values())
            for g, q in grp.items():
                for st in g.split("+"):
                    out[st] += p[a] * (q / tot if tot > 0 else 1.0 / len(grp))
        return out

    def _best(self, score: dict, margin: float) -> str:
        best = max(STAGES, key=lambda st: (score[st], -STAGES.index(st)))
        return best if score[best] > score[self.green_stage] + margin else self.green_stage

    def max_pressure_request(self) -> str:
        """Stage with the largest pressure, switching only for a margin of 1 PCU."""
        return self._best(self.stage_pressures(), 1.0)

    def priority_stage(self):
        """Stage of an ambulance that must be served now: its priority request says its route crosses the signal
        and its ETA is no more than the time to switch (intergreen + start-up loss), discharge the PCU queue ahead
        of it at the approach's saturation flow, and wait for the next decision instant.  Several: the smallest ETA."""
        best = None
        for cam in sensor_layout()["cameras"]:
            if cam["kind"] != "queue_camera":
                continue
            c = self.sensors.cam_now.get(cam["sensor_id"], {})
            eta = c.get("ev_eta_s")
            if eta is None or not c.get("ev_needs_signal"):
                continue
            # + one decision interval: the next chance to switch is decision_s away, so act at the last instant before
            # the deadline rather than up to decision_s after it
            need = (INTERGREEN_S + START_LOSS_S + 3600.0 / cam["sat_flow_pcu_h"] * c.get("ev_q_ahead_pcu", 0.0)
                    + self.decision_s)
            if eta <= need and (best is None or eta < best[0]):
                best = (eta, STAGE_OF_APPROACH[cam["leg"]])
        return best[1] if best else None

    def teacher_request(self) -> str:
        """The priority-aware teacher (label y_teacher_stage, controller 'priority_mp'): serve an ambulance that must
        be served now, else max-pressure."""
        return self.priority_stage() or self.max_pressure_request()

    def naive_preempt_stage(self):
        """Stage of an ambulance whose route crosses the signal and that is within PREEMPT_ETA_S of the stop line;
        a stage already pre-empted is held until its ambulance has left the approach camera zone (crossed)."""
        best = None
        for cam in sensor_layout()["cameras"]:
            if cam["kind"] != "queue_camera":
                continue
            c, st = self.sensors.cam_now.get(cam["sensor_id"], {}), STAGE_OF_APPROACH[cam["leg"]]
            eta = c.get("ev_eta_s")
            if eta is not None and c.get("ev_needs_signal") and (eta <= PREEMPT_ETA_S or st == self.preempting):
                if best is None or eta < best[0]:
                    best = (eta, st)
        self.preempting = best[1] if best else None
        return self.preempting

    def feature_row(self) -> dict:
        """The ticks_5s row at this instant without episode_id and the y_* labels: a learned controller's input."""
        row = {**self.sensors.tick_row(self.t), **self.tick_context()}
        return {k: v for k, v in row.items() if not k.startswith("y_")}

    def request(self) -> str:
        """Controller: 'fixed' runs the Dishub greens in the order S1-S2-S3-S4; 'longest_queue' serves the stage
        with the most standing vehicles (margin 3); 'max_pressure' the largest pressure; 'priority_mp' is
        max-pressure that serves an ambulance in time (the teacher); 'preempt_naive' is the Dishub plan that
        jumps to an ambulance's stage when it is 20 s away and holds it until the ambulance has crossed;
        'random' asks for a uniformly random stage; 'still' keeps the current stage (so max green and the
        starvation guard decide)."""
        if self.force is not None:
            st, t_from, vid = self.force
            r = self.sensors.ev.get(vid)
            if r is None or "t_stopline" in r or r.get("exits"):
                self.force = None                            # it has crossed (or used a slip and left)
            elif self.t >= t_from - 1e-9:
                return st
        if self.policy is not None:
            st = self.policy(self.feature_row())
            if st not in STAGES:
                self.n_invalid_requests += 1
                return self.green_stage
            return st
        if self.controller == "priority_mp":
            return self.teacher_request()
        if self.controller == "random":
            return STAGES[int(self.ctrl_rng.integers(len(STAGES)))]
        if self.controller == "still":
            return self.green_stage
        if self.controller == "preempt_naive":
            st = self.naive_preempt_stage()
            if st is not None:
                return st
        if self.controller == "max_pressure":
            return self.max_pressure_request()
        if self.controller == "longest_queue":
            return self._best(self.stage_queues(), 3.0)
        if self.t - self.green_since >= DISHUB_GREEN_S[self.green_stage] - 1e-9:
            return STAGES[(STAGES.index(self.green_stage) + 1) % 4]
        return self.green_stage

    def _set(self, idx: int) -> None:
        self.ts.trafficlight.setPhase(TLS_ID, idx)
        self.ts.trafficlight.setPhaseDuration(TLS_ID, 1e6)

    def plant_step(self) -> None:
        if self.transition is not None:
            target, k, since = self.transition
            amber_i, allred_i = self.trans_idx[(self.green_stage, target)]
            if self.t - since >= (YELLOW_S if k == 0 else ALLRED_S) - 1e-9:
                if k == 0:
                    self.transition = (target, 1, self.t)
                    self._set(allred_i)
                else:
                    self.green_stage, self.green_since, self.transition = target, self.t, None
                    self.next_decision = self.t + self.min_green
                    self._set(self.green_idx[target])
            return
        if self.controller in ("fixed", "preempt_naive"):     # switches exactly at the end of each Dishub green
            want = self.last_request = self.request()
        else:
            if self.t + 1e-9 < self.next_decision:
                return
            self.next_decision = self.t + self.decision_s
            want = self.last_request = self.request()
            demand = self.stage_queues()
            starving = [st for st in STAGES if st != self.green_stage and demand[st] > 0
                        and self.t - self.red_since[st] >= STARVE_S]
            if starving:
                want = min(starving, key=lambda st: self.red_since[st])
            elif want == self.green_stage and self.t - self.green_since >= self.max_green - 1e-9:
                want = STAGES[(STAGES.index(self.green_stage) + 1) % 4]
        if want != self.green_stage and self.t - self.green_since >= self.min_green - 1e-9:
            self.red_since[self.green_stage] = self.t
            self.transition = (want, 0, self.t)
            self._set(self.trans_idx[(self.green_stage, want)][0])

    # ---- ambulance rule: cars move aside toward the kerb where there is room ---------------------
    def ambulance_step(self) -> None:
        ts = self.ts
        for vid in ts.vehicle.getIDList():
            if not vid.startswith("ev"):
                continue
            lead = ts.vehicle.getLeader(vid, 80.0)
            if not lead or not lead[0] or lead[0].startswith("ev"):
                continue
            car = lead[0]
            if ts.vehicle.getSpeed(car) < 1.0:
                continue                                   # a standing queue cannot clear a path
            lane_idx = ts.vehicle.getLaneIndex(car)
            if lane_idx > 0 and ts.vehicle.couldChangeLane(car, -1):
                ts.vehicle.changeLaneRelative(car, -1, 4.0)
                self.n_yields += 1

    # ---- motorcycles keep to their lane while crossing a junction ---------------------------------
    def junction_step(self) -> None:
        """Motorcycles filter anywhere on a road (latAlignment arbitrary: kerb, lane lines, the motorcycle box),
        but ride the middle of their lane while on a junction and for centre_hold_s after it.  SUMO only checks
        crossing and merging movements against each other, so filtering motorcycles at the outer edges of
        side-by-side movements overlapped inside junctions and right at their exits (30 min at design demand,
        three seeds: about 9 junction overlaps and no teleports with this rule, 22 and 18 teleports without)."""
        ts = self.ts
        for vid, (align, until) in self.motos.items():
            lane = ts.vehicle.getLaneID(vid)
            if not lane:                                   # teleporting
                continue
            if lane.startswith(":"):
                until = self.t + self.centre_hold_s
            want = "center" if self.t < until else "arbitrary"
            if want != align:
                ts.vehicle.setLateralAlignment(vid, want)
            self.motos[vid] = (want, until)

    # ---- stepping ----------------------------------------------------------------------------------
    def step(self) -> None:
        ts = self.ts
        self.plant_step()
        if int(self.t * 2) % 2 == 0:
            self.ambulance_step()
        if self.lateral_mode != "off":
            self.junction_step()
        ts.simulationStep()
        self.t = ts.simulation.getTime()
        self.n_teleports += ts.simulation.getStartingTeleportNumber()
        self.n_departed += ts.simulation.getDepartedNumber()
        for vid in ts.simulation.getDepartedIDList():
            if self.lateral_mode != "off" and ts.vehicle.getTypeID(vid).startswith("moto"):
                self.motos[vid] = ("arbitrary", -1.0)
            if vid.startswith("ev"):
                self.ev_depart[vid] = self.t
                if self.sensors is not None:
                    self.sensors.ev_departed(vid, self.t)
                self.log(f"ambulance {vid} on its way to RSUP Fatmawati")
        for vid in ts.simulation.getArrivedIDList():
            self.motos.pop(vid, None)
            self.n_arrived += 1
            if self.sensors is not None and vid.startswith("ev"):
                self.sensors.ev_arrived(vid, self.t)
            if vid in self.ev_depart:
                self.n_ev_arrived += 1
                tt = self.t - self.ev_depart.pop(vid)
                self.ev_travel.append(tt)
                self.log(f"ambulance {vid} reached RSUP Fatmawati after {tt:.0f} s")
        for c in ts.simulation.getCollisions():
            key = (c.collider, c.victim)
            if key in self._coll_seen:
                continue
            self._coll_seen.add(key)
            try:
                x, y = ts.vehicle.getPosition(c.collider)
                near = math.hypot(x - self.x0, y - self.y0) <= STUDY_M
            except Exception:
                near = False
            if c.lane.startswith(":"):
                self.n_junction_overlaps += 1
                self.n_junction_overlaps_study += near
            else:
                self.n_lane_collisions += 1
                self.n_lane_collisions_study += near
        if int(self.t * 2) % int(VKM_EVERY_S * 2) == 0:      # distance travelled: vehicles x mean speed per edge
            for e in self._edges:
                n = ts.edge.getLastStepVehicleNumber(e)
                if n:
                    km = n * ts.edge.getLastStepMeanSpeed(e) * VKM_EVERY_S / 1000.0
                    self.veh_km += km
                    if e in self._study_edges:
                        self.veh_km_study += km
        phase = ts.trafficlight.getPhase(TLS_ID)
        if phase != self._phase:
            self._phase, self._phase_since = phase, self.t
            stage, _, state = self.phase_names[phase].partition(" ")
            self.signal_log.append(dict(t=self.t, phase=phase, stage=stage, state=state, controller=self.controller))
        if self.sensors is not None:
            n = len(self.sensors.rows_ticks)
            self.sensors.step(self.t)
            if len(self.sensors.rows_ticks) > n:
                self.sensors.rows_ticks[-1].update(self.tick_context())

    def tick_context(self) -> dict:
        """Controller-side columns of a 5 s tick: what the signal is doing and what the teacher would request."""
        stage, _, state = self.phase_names[self._phase].partition(" ")
        sq, sp = self.stage_queues(), self.stage_pressures()
        return dict(clock=self.clock(), controller=self.controller, stage=self.green_stage, phase_state=state,
                    **{f"stage_q_{st}": int(sq[st]) for st in STAGES}, **{f"stage_pressure_{st}": round(sp[st], 2) for st in STAGES},
                    transition_to=self.transition[0] if self.transition else None,
                    t_in_phase_s=round(self.t - self._phase_since, 1), t_since_green_s=round(self.t - self.green_since, 1),
                    controller_request=self.last_request, y_teacher_stage=self.teacher_request())

    def log(self, text: str) -> None:
        h = (self.start_hour * 3600.0 + self.t) % 86400.0
        self.events.append(f"{int(h // 3600):02d}:{int(h % 3600 // 60):02d}:{int(h % 60):02d} {text}")
        del self.events[:-40]

    def clock(self) -> str:
        h = (self.start_hour * 3600.0 + self.t) % 86400.0
        return f"{int(h // 3600):02d}:{int(h % 3600 // 60):02d}:{int(h % 60):02d}"

    # ---- the viewer frame -------------------------------------------------------------------------
    def frame(self) -> dict:
        ts = self.ts
        vs = []
        n_moto = n_halt = 0
        raised_lanes, raised_nodes = display_raised()
        for vid in ts.vehicle.getIDList():
            x, y, z = ts.vehicle.getPosition3D(vid)
            if z:                                    # the heights the viewer draws: toll road only (geometry3d)
                lane = ts.vehicle.getLaneID(vid)
                if lane in raised_lanes:
                    z = raised_z(raised_lanes[lane], ts.vehicle.getLanePosition(vid))
                elif not (lane.startswith(":") and lane[1:].rsplit("_", 2)[0] in raised_nodes):
                    z = 0.0
            code = 1 if vid.startswith("ev") else type_code(ts.vehicle.getTypeID(vid))
            sp = ts.vehicle.getSpeed(vid)
            n_moto += code == 2
            n_halt += sp < 0.5
            num = int(vid[2:]) + 10_000_000 if vid.startswith("ev") else int(vid[1:]) if vid[1:].isdigit() else hash(vid) % 10_000_000
            vs.append([num, round(x - self.x0, 2), round(y - self.y0, 2), round(z, 2), round(ts.vehicle.getAngle(vid), 1), code, round(sp, 1)])
        q = self.queues()
        cams = {}
        if self.sensors is not None:                 # what the cameras see (the panel shows sensor readings)
            for a in ARMS:
                c, x = self.sensors.cam_now.get(f"QC_{a}", {}), self.sensors.cam_now.get(f"XC_{a}", {})
                cams[a] = dict(q_veh=c.get("q_veh", 0), q_m=c.get("q_m", 0.0), moto_box=c.get("moto_box", 0),
                               speed_in_kmh=c.get("speed_kmh"), speed_out_kmh=x.get("speed_kmh"))
        phase = ts.trafficlight.getPhase(TLS_ID)
        return dict(
            t=round(self.t, 1), clock=self.clock(), vehicles=vs,
            tls=ts.trafficlight.getRedYellowGreenState(TLS_ID), green=self.green_stage,
            stage_label=STAGE_LABEL[self.green_stage], phase_name=self.phase_names[phase],
            cams=cams,
            kpi=dict(n_vehicles=len(vs), n_motorcycles=n_moto, n_halting=n_halt, queues=q,
                     n_lane_collisions=self.n_lane_collisions, n_junction_overlaps=self.n_junction_overlaps,
                     n_teleports=self.n_teleports, n_arrived=self.n_arrived, n_yields=self.n_yields,
                     ev_arrived=self.n_ev_arrived,
                     ev_mean_travel_s=round(sum(self.ev_travel) / len(self.ev_travel), 1) if self.ev_travel else None,
                     speed_x=round(self.t / max(time.perf_counter() - self.wall0, 1e-6), 1)),
            events=list(self.events[-10:]))

    def close(self) -> None:
        self.ts.close()
