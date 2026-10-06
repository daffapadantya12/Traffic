"""Near-miss and collision monitor (DECISIONS §12.4): surrogate safety measures every step.

Traffic-engineering idea: crashes are too rare to count, so road-safety studies count
*conflicts* - situations that end in a crash unless someone brakes or swerves - with
surrogate safety measures (FHWA SSAM, Gettman et al. 2008):

- following pairs (a vehicle and the vehicle body ahead of it on its own path: same lane,
  shared start of two box paths, exit lane, or a lane changer still beside it): time to
  collision TTC = gap / (v - v_lead) while closing, and the deceleration rate to avoid the
  crash DRAC = (v - v_lead)^2 / (2 gap).
  * conflict  : TTC < ttc_conflict_s (1.5 s, SSAM default)
  * near miss : a conflict whose DRAC reached drac_near_miss (3.4 m/s^2, AASHTO comfortable
                deceleration; Archer 2005).  DRAC = dv / (2 TTC), so slow creeping toward a
                queue is never a near miss - severity needs time margin AND speed.
  * serious   : a near miss with TTC < 1.0 s;  critical: DRAC > b_emerg (unavoidable).
- junction boxes: every pair of conflicting paths of different approaches has a zone on each
  path.  Entry (front reaches s_in) and exit (rear leaves s_out) are interpolated inside the
  step; post-encroachment time PET = entry of the next vehicle - exit of the previous one
  (other approach).  Two such vehicles inside their zones at once is a zone collision.
- 1-D collisions: a follower whose front ended a step inside the vehicle ahead.

Events are de-duplicated per (pair, type) with a 1.0 s hysteresis and kept as rows of
conflicts.csv (raw TTC, DRAC, gap, PET, so analysts can re-threshold).  KPIs count events
that start in the main period; the invariants (collisions, critical events) count the whole
episode.  The drivers in this model have no reaction time or errors, so the counts measure
geometry and control conflicts, not human crash risk.
"""

from __future__ import annotations

import math

HYSTERESIS_S = 1.0
PET_WINDOW_S = 5.0          # SSAM records PET up to 5 s


class SafetyMonitor:
    def __init__(self, world):
        self.world = world
        cfg = world.cfg
        self.cfg = cfg
        self.events = {}                 # (min id, max id, type) -> open event dict
        self.rows = []                   # closed events (conflicts.csv)
        self.next_event_id = 0
        self.n_collisions_1d = 0
        self.n_collisions_zone = 0
        self.n_critical = 0
        self.min_gap_moving = math.inf
        self.max_lat_accel = 0.0
        # zones: zone id -> {veh id: (approach, t_in)}; last exit per zone: (t_out, approach)
        self.zone_occ = {}
        self.zone_last_exit = {}
        self.pet_values = []             # (t, pet, junction, is_main)
        self._coll_pairs = set()

    # ---- following pairs -------------------------------------------------------------
    def follow(self, veh, leader, gap: float, v_lead: float, kind: str = "rear_end") -> None:
        """Called once per step for a vehicle whose closest body ahead is another vehicle."""
        if veh.v >= 2.0:
            self.min_gap_moving = min(self.min_gap_moving, gap)
        closing = veh.v - v_lead
        if closing <= 0.0 or gap <= 0.0:
            return
        ttc = gap / closing
        if ttc >= self.cfg.ttc_conflict_s:
            return
        drac = closing * closing / (2.0 * gap)
        t = self.world.t
        key = (min(veh.id, leader.id), max(veh.id, leader.id), kind)
        ev = self.events.get(key)
        if ev is None or t - ev["t_last"] > HYSTERESIS_S + 1e-9:
            if ev is not None:
                self._close(key)
            ev = self._open(key, veh, leader, kind, t)
        ev["t_last"] = t
        if ttc < ev["min_ttc_s"]:
            ev.update(min_ttc_s=ttc, v_a_mps=veh.v, v_b_mps=v_lead)
        ev["max_drac_mps2"] = max(ev["max_drac_mps2"], drac)
        ev["min_gap_m"] = min(ev["min_gap_m"], gap)
        veh.min_ttc = min(veh.min_ttc, ttc)
        veh.max_drac = max(veh.max_drac, drac)

    def _open(self, key, veh, other, kind, t, junction_id=-1):
        ev = dict(event_id=self.next_event_id, type=kind, veh_a=veh.id, veh_b=other.id,
                  vtype_a=veh.vehicle_type, vtype_b=other.vehicle_type,
                  junction_id=junction_id, edge_id=(veh.edge.id if veh.edge is not None else
                                                   (veh.conn.id if veh.conn is not None else "")),
                  t_start=t, t_last=t, min_ttc_s=math.inf, max_drac_mps2=0.0, min_gap_m=math.inf,
                  v_a_mps=veh.v, v_b_mps=other.v, pet_s=math.nan, collision=False,
                  a_id=veh, b_id=other)
        self.next_event_id += 1
        self.events[key] = ev
        return ev

    def _close(self, key) -> None:
        ev = self.events.pop(key)
        cfg = self.cfg
        near = ev["max_drac_mps2"] >= cfg.drac_near_miss
        serious = near and ev["min_ttc_s"] < 1.0
        critical = ev["max_drac_mps2"] > cfg.b_emerg
        a, b = ev.pop("a_id"), ev.pop("b_id")
        ev["t_end"] = ev.pop("t_last")
        ev.update(near_miss=near, serious=serious, critical=critical, is_main=cfg.is_main(ev["t_start"]))
        if not ev["collision"]:
            if near:
                a.n_near_misses += 1
            a.n_conflicts += 1
        if critical:
            self.n_critical += 1
        self.rows.append(ev)

    # ---- collisions ----------------------------------------------------------------------
    def collision_1d(self, follower, leader) -> None:
        pair = (min(follower.id, leader.id), max(follower.id, leader.id), "1d")
        if pair in self._coll_pairs:
            return
        self._coll_pairs.add(pair)
        self.n_collisions_1d += 1
        follower.n_collisions += 1
        leader.n_collisions += 1
        t = self.world.t
        ev = self._open(pair, follower, leader, "collision", t)
        ev.update(min_ttc_s=0.0, min_gap_m=0.0, collision=True)
        self._close(pair)

    # ---- junction boxes ------------------------------------------------------------------
    def zone_step(self, t0: float, dt: float, records) -> None:
        """records: (veh, connector, junction id, s_old, s_new) in path coordinates for every body
        that touched a box path this step.  Entries before exits are processed in time order."""
        net = self.world.net
        crossings = []
        for veh, conn, jid, s_old, s_new in records:
            ds = s_new - s_old
            for zid, s_in, s_out in net.zones_on_path[conn.key]:
                key = (jid, zid)
                if s_old < s_in <= s_new and ds > 1e-9:
                    crossings.append((t0 + dt * (s_in - s_old) / ds, 0, key, veh, conn.approach))
                r_old, r_new = s_old - veh.l_veh, s_new - veh.l_veh
                if r_old < s_out <= r_new and ds > 1e-9:
                    crossings.append((t0 + dt * (s_out - r_old) / ds, 1, key, veh, conn.approach))
        crossings.sort(key=lambda c: (c[0], c[1], c[3].id))
        cfg = self.cfg
        for tc, kind, key, veh, ap in crossings:
            jid, zid = key
            occ = self.zone_occ.setdefault(key, {})
            if kind == 0:                                  # front enters the zone
                foe = (jid, net.zone_foe[zid])
                foes = self.zone_occ.get(foe, {})
                hit = [(vid, v2) for vid, (ap2, _, v2) in foes.items() if ap2 != ap]
                if hit:
                    for vid, v2 in hit:
                        pair = (min(veh.id, vid), max(veh.id, vid), "zone")
                        if pair not in self._coll_pairs:
                            self._coll_pairs.add(pair)
                            self.n_collisions_zone += 1
                            veh.n_collisions += 1
                            v2.n_collisions += 1
                            self.rows.append(dict(
                                event_id=self.next_event_id, type="collision", veh_a=veh.id, veh_b=v2.id,
                                vtype_a=veh.vehicle_type, vtype_b=v2.vehicle_type, junction_id=jid,
                                edge_id=net.junctions[jid].node, t_start=tc, t_end=tc, min_ttc_s=0.0,
                                max_drac_mps2=math.nan, min_gap_m=0.0, v_a_mps=veh.v, v_b_mps=v2.v, pet_s=math.nan,
                                collision=True, near_miss=False, serious=False, critical=False, is_main=cfg.is_main(tc)))
                            self.next_event_id += 1
                else:
                    last = self.zone_last_exit.get(foe)
                    if last is not None and last[1] != ap:
                        pet = tc - last[0]
                        if pet <= PET_WINDOW_S:
                            self.pet_values.append((tc, pet, jid, cfg.is_main(tc)))
                            other = last[2]
                            self.rows.append(dict(
                                event_id=self.next_event_id, type="zone_pet", veh_a=veh.id, veh_b=other.id,
                                vtype_a=veh.vehicle_type, vtype_b=other.vehicle_type, junction_id=jid,
                                edge_id=net.junctions[jid].node, t_start=tc, t_end=tc, min_ttc_s=math.nan,
                                max_drac_mps2=math.nan, min_gap_m=math.nan, v_a_mps=veh.v, v_b_mps=other.v, pet_s=pet,
                                collision=False, near_miss=False, serious=pet < cfg.pet_conflict_s, critical=False,
                                is_main=cfg.is_main(tc)))
                            self.next_event_id += 1
                occ[veh.id] = (ap, tc, veh)
            else:                                          # rear leaves the zone
                if veh.id in occ:
                    del occ[veh.id]
                self.zone_last_exit[key] = (tc, ap, veh)

    def forget(self, veh) -> None:
        """A vehicle left the network: it can no longer occupy a zone."""
        for occ in self.zone_occ.values():
            occ.pop(veh.id, None)

    # ---- end of step / episode -------------------------------------------------------------
    def end_step(self, t: float) -> None:
        stale = [k for k, ev in self.events.items() if t - ev["t_last"] > HYSTERESIS_S + 1e-9]
        for k in stale:
            self._close(k)

    def finish(self) -> None:
        for k in list(self.events):
            self._close(k)

    def kpis(self, n_vehicles_main: int) -> dict:
        main = [r for r in self.rows if r["is_main"] and not r["collision"] and r["type"] != "zone_pet"]
        near = [r for r in main if r["near_miss"]]
        pets = [p for _, p, _, m in self.pet_values if m]
        return dict(
            n_conflicts=len(main), n_near_misses=len(near),
            n_serious=sum(1 for r in near if r["serious"]),
            near_miss_per_1000_veh=round(1000.0 * len(near) / max(n_vehicles_main, 1), 3),
            min_ttc_s=round(min((r["min_ttc_s"] for r in main), default=math.nan), 3),
            max_drac_mps2=round(max((r["max_drac_mps2"] for r in main), default=0.0), 3),
            min_pet_s=round(min(pets), 3) if pets else math.nan,
            n_pet_events=len(pets), n_pet_serious=sum(1 for p in pets if p < self.cfg.pet_conflict_s),
            n_collisions=self.n_collisions_1d + self.n_collisions_zone,
            min_gap_moving_m=round(self.min_gap_moving, 3) if self.min_gap_moving < math.inf else math.nan,
            max_lat_accel_mps2=round(self.max_lat_accel, 3),
        )

    def conflict_rows(self, episode_id: int) -> list:
        out = []
        for r in self.rows:
            row = {k: v for k, v in r.items()}
            row["episode_id"] = episode_id
            for k in ("min_ttc_s", "max_drac_mps2", "min_gap_m", "v_a_mps", "v_b_mps", "t_start", "t_end", "pet_s"):
                if isinstance(row[k], float) and math.isfinite(row[k]):
                    row[k] = round(row[k], 3)
            out.append(row)
        return out
