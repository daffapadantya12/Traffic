# DECISIONS.md — authoritative design decisions (v1)

This file wins over `design_panel_synthesis.md` wherever they disagree. Everything numeric here must appear as a field of the single config dataclass in `traffic_twin/config.py` with the stated default.

## 1. Student decisions (2026-10-03)

| Decision | Choice |
|---|---|
| Lanes per direction | **2** (`lanes_per_dir = 2`; must also run with 1) |
| Driving side | **left** (Indonesia). `drive_side = "left"`; `"right"` must also work |
| ML model shape | **one shared per-junction model** applied at all 4 junctions through the same feature row, with neighbour features |
| Demand band | **calibrated from the saturation flow measured in the simulator** (never the textbook 1800 veh/h/lane) |
| Reinforcement learning | **future work**; course deliverable = tabular dataset + supervised models |
| Hand-in format | **CSV only** (no Parquet dependency anywhere) |
| Signals | red/green display only; internal GREEN / ALLRED plant states; **split phasing** (one approach green at a time, all three movements) |

## 2. Units, coordinates, time

- SI units: m, s, m/s, m/s². Origin at the centre of the south-west junction, x east, y north. The viewer flips y.
- Physics step `dt = 0.5 s` (ballistic update, see §5). Logging of per-junction aggregates at 1 Hz. Decision instants per junction: first at `green_onset + min_green`, then every `decision_interval = 5 s` while GREEN; none during ALLRED.
- Episode time axis: `t = 0` is the start of warm-up. `warmup_s = 300`, `main_s = 3600`, `cooldown_s = 300` (spawning off). Flags on every logged row: `is_warmup = t < 300`, `is_cooldown = t >= 3900`, `is_tail = (t > 3900 - 30) and not is_cooldown`. KPIs and ML labels use only rows/vehicles with `not is_warmup and not is_cooldown` (vehicles: spawned in the main period).

## 3. Network ("#", built by `build_grid(cfg)`, must also build 1x1)

- `grid_x = 2, grid_y = 2` (1x1 must work for M0 validation). `L_block = 200 m` centre-to-centre, `L_stub = 150 m` centre to boundary, `w_lane = 3.5 m`.
- Road width = `2 * lanes_per_dir * w_lane` = 14 m; junction box = 14 m square (`w_box`). Internal edge length `L_block - w_box = 186 m`, stub edge length `L_stub - w_box/2 = 143 m`. Connector length = `w_box` = 14 m for every movement in v1 (arc radii are drawing-only).
- Nodes: junctions `J(i,j)` and boundary nodes. Directed edges run box-edge to box-edge. Edge id `"<from>_<to>"`. Stop line 2.0 m upstream of the box edge.
- **Lane geometry (left-hand traffic):** a vehicle drives on the left of the road axis relative to its heading. For heading unit vector `h = (hx, hy)`, left normal `n = (-hy, hx)`. Lane `k` (k = 0 kerb lane, k = lanes_per_dir-1 median lane) has lateral offset `n * w_lane * (lanes_per_dir - k - 0.5)` from the axis. Under right-hand traffic use `-n`. Derive from the flag; never hard-code.
- **Movements:** T (through), NS (near-side turn = does not cross opposing traffic = LEFT turn under left-hand traffic), FS (far-side turn = crosses opposing traffic). Exit heading is computed programmatically: `NS: rotate(h, +90° CCW)` under left-hand traffic, `-90°` under right-hand; `FS` the opposite rotation; `T` unchanged. Unit-test: southbound vehicle under left-hand traffic turns NS to the EAST arm. No U-turns, no turn-on-red.
- **Storage** per lane `floor(L / (l_veh + s0))`: internal 28 per lane, stub 22 per lane. Queue features are normalised by the storage of their own edge, not a global constant.

## 4. Two lanes per direction without lane changing

- No mid-link lane changes. A vehicle's lane on an edge is fixed when it enters that edge.
- Lane use at the stop line (left-hand): **lane 0 (kerb) serves NS + T; lane 1 (median) serves FS + T.** With 1 lane per direction that lane serves everything.
- Lane choice at edge entry, from the vehicle's *next* movement at the downstream junction: NS → lane 0; FS → lane 1; T → the lane with fewer vehicles on that edge (tie → lane 0). If the edge ends at a boundary node, pick the lane with fewer vehicles.
- The connector delivers the vehicle to whichever lane of the exit edge its *following* movement needs (this is a "merge inside the box" simplification; state it in the data card).
- Entering a connector requires: approach GREEN (or vehicle COMMITTED) AND free space at the start of the **target lane** of the exit edge ≥ `l_veh + s0 + 1.0 m`. The free-space condition is anticipatory and colour-independent: when it fails for a vehicle's next movement, the stop line is that vehicle's virtual stationary leader (checked from ≤ 100 m upstream). Committed vehicles always enter.
- Leader for a vehicle inside a connector = the nearest of (a) the last vehicle on its target exit lane (gap = remaining connector length + that vehicle's rear position on the exit lane) and (b) any vehicle on a sibling connector of the same approach targeting the same exit lane with smaller remaining distance. Speed on turn connectors capped at `v_turn = 5.0 m/s`; through connectors at `min(v0, v_limit)`.
- Clearance gate in v1 = **box empty** (no vehicle on any connector of that junction) — the 12x12 conflict matrix is a v2 item.

## 5. Vehicles (IDM, per `vehicle_model` in the synthesis, with these corrections)

- Parameters drawn once at spawn **request** time (so common random numbers hold): `alpha ~ Beta(2,2)`; `v0 = v_limit*(1+0.20*(alpha-0.5))*LN(0,0.08)` clipped to `[0.85, 1.15]*v_limit`; `a_max = 1.5*(1+0.50*(alpha-0.5))*LN(0,0.10)` clip `[0.8, 2.5]`; `b = 2.0*(1+0.30*(alpha-0.5))*LN(0,0.10)` clip `[1.2, 3.0]`; `T = 1.2*(1-0.40*(alpha-0.5))*LN(0,0.10)` clip `[0.8, 1.8]`; `l_veh = 4.5`, `s0 = 2.0`, `b_emerg = 7.0` fixed. `v_limit = 13.9 m/s` default.
- **Desired speed is NOT capped at v_limit**: `v0_local = v0` for through movements (speeding is allowed and is what alpha is for). Turn anticipation: `v0_local = min(v0, sqrt(v_turn^2 + 2*b*d_to_box))` from 60 m upstream when the next movement is a turn.
- IDM: `a = a_max*(1 - (v/v0_local)^4 - (s_star/gap)^2)`, `s_star = s0 + max(0, v*T + v*(v - v_lead)/(2*sqrt(a_max*b)))`. Do not clamp inside IDM; after computing, log `hard_brake` if `a < -3.0`, then clip `a >= -b_emerg`.
- Ballistic update with stop-exactly branch; post-step overlap guard at 0.3 m with an `overlap_fixed` counter (health metric, must stay < 1 per 10,000 vehicle-steps).
- **Stop-or-commit at red onset:** evaluate EVERY vehicle on the approach edge, nearest the stop line first, each against its own `d` and `v` with `b_stop = 3.0`: `d >= v^2/(2*b_stop)` → STOPPING (stop line = virtual leader), else COMMITTED (latched; crosses during ALLRED). Committed chains are allowed. Vehicles entering the edge after red onset default to STOPPING. Only non-committed vehicles see the stop line.
- Free-flow time `= sum(edge_len / v0_local_edge) + n_turns*(w_box/v_turn) + n_through*(w_box/min(v0, v_limit))`. `total_delay = entry_queue_delay + travel_time - free_flow_time`.
- Spawn: per step per boundary node, Bernoulli with `p = lambda_e(t)*dt`; requires first 10 m of the chosen lane free and gap ≥ `s0 + v_spawn*T`; else waits in a per-node entry queue (delay counted). Destination ~ OD row (default relational row: far end of own road 0.30, two ends of the parallel road 0.15 each, four ends of the crossing roads 0.10 each; `uniform` = 1/7). Route by Dijkstra on the node graph with turn penalties (T 0, NS +10, FS +25 m), ties broken by the routing RNG.
- RNG streams from `SeedSequence(seed).spawn(4)`: demand (arrivals, OD, routing ties), vehicles (parameters), controller (eps / random dwell), sensor (noise). Seed = `100000 + episode_id`. Determinism test: two runs with the same seed give identical vehicles CSV.

## 6. Signal plant

- Per junction: `phase ∈ {0:N,1:E,2:S,3:W}` (approach named by the arm the vehicle arrives from), `state ∈ {GREEN, ALLRED}`, `time_in_phase`, `pending_phase`, `n_switches`.
- `min_green = 5`, `max_green = 60`, `allred_min = 4`, `decision_interval = 5`.
- Switch procedure, max-green rule (forces a switch only if another approach has `n_a > 0`, to the approach with the largest `n_a`, ties in rotation order), ALLRED extended until box empty, requests during ALLRED ignored with `forced = 3`: exactly as in the synthesis `signal_model`.
- Lost time per phase = measured start-up lost time (~2 s) + all-red 4 s. **Effective green = displayed green + all-red − lost time**, e.g. 20 + 4 − 6 = 18 s. Use 18 s, not 20 s, in any capacity or Webster calculation.
- Safety invariants (pytest, every episode): never two approaches green; box empty at every green onset; no stop-line crossing on red unless COMMITTED; min green respected; all-red ≥ 4 s; vehicle conservation.
- Controllers (`controllers.py`): `fixed_time` (rotation N,E,S,W, 20 s green), `longest_queue`, `max_pressure` (pressure_a = n_a − mean(down_n over a's three exits); exits whose target lane is blocked are **excluded from the mean**; hysteresis +2 vehicles; this exact function is also the label function), `random_dwell` (hold U(5,40) s then random other approach), `max_pressure_eps` (eps = 0.3 random request). `y_mp_action` is the *request*, not the applied phase.

## 7. Dataset (CSV only, v1 = ~40 columns in signal_ticks)

Files in `data/<run_name>/`: `episodes.csv`, `vehicles.csv`, `signal_ticks.csv`, `junction_1hz.csv` (per-junction 1 Hz aggregates: `episode_id, junction_id, t, q_total, n_total, wait_sum_total, crossings_cum`; needed for the 30 s-ahead labels), `network_edges.csv`, `data_dictionary.csv`, `config.json`, `DATACARD.md`. Showcase trajectories `vehicle_states_<episode_id>.csv` (1 Hz) only when `log_trajectories = True`.

`signal_ticks` columns (one row per junction per decision instant; PK `episode_id, junction_id, t`):
- keys/flags: `episode_id, junction_id, t, tick_idx, is_warmup, is_cooldown, is_tail`
- signal: `phase, time_in_phase_s, max_green_remaining_s`
- per approach a ∈ {N,E,S,W}: `q_a` (vehicles with v < 2 m/s on the approach edge, both lanes), `n_a` (all vehicles on the approach edge), `q_a_l0`, `q_a_l1` (per-lane queue; `-1` when lanes_per_dir = 1), `arr_a_30s`
- downstream per exit direction d: `down_occ_d` (vehicles on the outgoing edge / its storage; stubs use their real counts), `down_blocked_d`
- neighbour per direction d: `nb_d_exists`, `nb_d_phase` (−1 if none), `nb_d_green_toward_me`
- actions: `action_requested, action_applied, forced, switched`
- labels: `y_mp_action, y_q_total_t30` (NaN on tail/cooldown)
- sensor copy: `obs_q_a` (4 columns) = `Binomial(q_a, 1-p_miss) + Poisson(p_miss/2)` with `sensor_noise_level ∈ {0, 0.1}`; identical to `q_a` at 0.
- NOT included (leakage/clock proxies): `throughput_cum`, `n_switches`, `t_frac`, time of day, `oracle_*`, `junction_id` is a key not a feature.

`episodes` columns: `episode_id, seed, config_hash, generator_version, grid_x, grid_y, lanes_per_dir, drive_side, L_block_m, v_limit_mps, dt_s, warmup_s, main_s, cooldown_s, demand_level, lambda_base_vph, demand_asym, od_pattern, controller, controller_eps, sensor_noise_level, split, n_spawned, n_arrived, n_in_network_end, throughput_vph, mean_total_delay_s, p90_total_delay_s, mean_stops, max_queue_veh, switches_per_min, n_red_commits, n_hard_brakes, n_overlap_fixed, n_box_blocked, allred_extension_mean_s, gridlock, gridlock_t, measured_sat_flow_vphpl, label_switch_rate, runtime_s`.

`vehicles` columns: `episode_id, veh_id, spawn_request_t, enter_t, exit_t, completed, is_warmup, origin, destination, route_movements, route_edges, route_len_m, n_junctions, n_turns, free_flow_time_s, travel_time_s, entry_queue_delay_s, total_delay_s, total_wait_s, n_stops, n_red_commits, n_hard_brakes, accel_energy, alpha, v0_mps, a_max_mps2, b_comf_mps2, T_headway_s, vehicle_type`.

Scenario mixture per episode (stratified): demand_level {low 25%, design 40%, high 25%, stress 10%}; demand_asym {sym, ns_heavy}; od_pattern {default 70%, uniform 30%}; controller {max_pressure 40%, max_pressure_eps 30%, fixed_time 20%, random_dwell 10%}; sensor_noise_level {0: 60%, 0.1: 40%}. Splits by episode: 70/15/15 train/val/test stratified by (demand_level, controller); **all stress episodes → `test_ood`**; random_dwell episodes train-only. `high` is trained on; the OOD claim is made on `stress`.

Demand levels are defined by v/c on the busiest internal approach under fixed-time control: low 0.4, design 0.7, high 0.95, stress 1.15, with capacity = `measured_sat_flow * lanes_per_dir * 18 / 96` per approach. `calibrate_demand()` derives `lambda_base` per level from the measured saturation flow and writes it to `config.json`.

## 8. Validation experiments (M0, figures in `reports/figures/`)

1. Queue discharge: 20-vehicle standing queue on one lane, green on; saturation headway = mean headway of vehicles 5–15 (expect 1.9–2.2 s → 1600–1900 veh/h/lane); start-up lost time. Writes `measured_sat_flow_vphpl`.
2. Single junction (1x1 grid), fixed-time, demand sweep: mean delay vs demand must rise smoothly and blow up near capacity (Webster comparison optional).
3. 2x2 grid, design demand, max_pressure, one episode: zero invariant violations, plausible delay (30–90 s/veh), histogram of all-red extensions, `overlap_fixed` rate.
4. Ring-road fundamental diagram (optional): expect approximately triangular, jam density ≈ 154 veh/km/lane, capacity 1600–1900 veh/h/lane. Do not fail the gate on shape.

## 9. Effort-reduction rules

No pygame, no YAML, no ZeroMQ, no SUMO, no Parquet, no vehicle classes, no acceleration noise, no counterfactual branching labels, no GNN, no RL in v1. Dependencies: numpy, pandas, matplotlib, pytest (scikit-learn later for the ML notebook). Plain Python loop; vectorise only if a 4200 s episode takes > 3 min.

## 10. Ambulance extension (healthcare framing for ITI5145; this section describes what `traffic_twin` v1.0 implements)

The assignment brief requires a healthcare scenario, so the simulator carries emergency vehicles (EVs). Everything in §§1–9 stays. Design rule: cars are bit-identical with EVs on or off for the same seed (separate RNG stream), so "with vs without ambulance" is a paired comparison.

- **RNG.** `SeedSequence(seed).spawn(5)`; the fifth child is `rng_ev`; the first four streams are unchanged.
- **Scenario factor `ev_rate ∈ {"0" (30%), "default" (70%)}`.** At `"default"` every boundary node spawns ambulance requests with `p_ev = (lambda_ev_vph / 3600) * dt`, `lambda_ev_vph = 4.0` per node (≈ 35 ambulances per episode; far above reality, chosen for statistical power and stated in the data card). Independent of demand level and asymmetry. Destination from the same OD row and route by the same Dijkstra, both drawn from `rng_ev`. The request joins the node's entry queue in FIFO order like a car.
- **Parameters (fixed, no alpha draw).** `v0 = 1.25 * v_limit = 17.4 m/s`, `a_max = 2.5`, `b = 3.0`, `T = 0.9 s`, `l_veh = 6.0 m`, `alpha = 1.0` stored, `vehicle_type = "ambulance"`. Same IDM, same stop-or-commit rule, same turn cap, same box rules. **Ambulances never run a red light**: priority comes only from the signal, so every safety invariant is unchanged. No lane changing, so an ambulance behind a queue waits for it to discharge (stated limitation).
- **Observable features (namespace `ev`, GPS/AVL-style, no sensor noise), per approach `a`:** `ev_present_a` (an ambulance is on the approach edge within `ev_horizon_m = 300 m` of the stop line), `ev_dist_a` (m to the stop line, −1 if none), `ev_eta_a = ev_dist_a / max(v_ev, 3.0 m/s)` (s, −1 if none), `ev_q_ahead_a` (queued vehicles, `v < 2 m/s`, ahead of it in its lane; −1 if none). With several ambulances the nearest one is reported.
- **Naive preemption baseline `preempt_naive`** (wraps `max_pressure`): request the approach of the ambulance with the smallest ETA among those with `ev_eta_a <= preempt_eta_s = 20 s`; otherwise the max-pressure request. The request still passes through the plant (min green, all-red ≥ 4 s extended until the box is empty, max-green override), so safety is unchanged.
- **Priority-aware teacher `priority_mp`** (= the label function `y_mp_action` on every row): request the ambulance's approach only when `ev_eta_a <= t_clear_a`, where `t_clear_a = allred_min + startup_lost_time_s + ev_q_ahead_a * sat_headway_s` is the time needed to discharge the queue ahead of it; before that it serves max-pressure, and after the ambulance has crossed it returns to max-pressure. On rows with no ambulance present `y_mp_action` is exactly the max-pressure request.
- **New KPIs in episodes.csv:** `ev_rate, lambda_ev_vph, n_ambulances` (requests in the whole episode), `ev_mean_travel_time_s, ev_mean_delay_s, ev_p90_delay_s, ev_mean_stops` (ambulances spawned in the main period; NaN if none), `car_mean_delay_penalty_s` (mean car delay minus the same quantity in the paired run without ambulances, same seed and controller; `generate.py` runs that paired episode unless `--no-paired`), `recovery_time_s` (mean time after an ambulance clears a junction until that junction's `q_total` is back within 10% of its value when the ambulance was first sighted on an approach).
- **Evaluation protocol:** controllers {fixed_time, max_pressure, preempt_naive, priority_mp, ml_v1} × demand {design, high} × ≥ 8 paired seeds; primary metric `ev_mean_delay_s`, secondary `car_mean_delay_penalty_s` and `recovery_time_s`; t-based 95% intervals on paired differences.
- **Future variants (not implemented):** a distance-based trigger (request as soon as the ambulance enters the approach), ambulance requests placed at the front of the entry queue, GPS noise and delivery delay on `ev_dist_a` (twin hooks).

## 11. Amendments after review round 1 (2026-10-03; `traffic_twin` v1.1)

Everything in §§1–10 stands. These additions close the review findings that did not contradict the sections above; findings that asked for a different number than §§1–10 state (hysteresis 2, min_green 5, b_stop 3.0 against the front-bumper distance, capacity = measured_sat_flow × lanes × 18/96) were NOT applied and are documented as known properties in the DATACARD instead.

- **11.1 Plant starvation guard** (`starvation_s = 180` = 3 × max_green; config field). At a decision instant, if an approach with `n_a > 0` has had no green for ≥ `starvation_s` (time since its green last ended, or since t = 0), the plant forces a switch to the longest-waiting such approach (ties in rotation order), logged as `forced = 4`. It pre-empts the max-green rule of §6, which alone can starve a third approach (the max-green rule always picks the largest `n_a`). `y_mp_action` is unchanged (it is the request, §6); `action_applied ≠ y_mp_action` on forced rows as before. New KPIs: `max_approach_red_s`, `n_forced_starvation`.
- **11.2 Clearance gate includes committed vehicles still upstream.** Green onset requires the box to be empty (§4) AND no vehicle latched COMMITTED at the previous red onset to be still on an approach edge of that junction, because committed vehicles enter regardless of colour. A COMMITTED vehicle that comes to rest (`v < stop_speed`) before the stop line is re-latched STOPPING (`n_commit_revoked`), so a stalled commit cannot freeze the junction. Invariant (3) is hardened: a crossing while the plant is GREEN for another approach is a red violation regardless of the latch; the COMMITTED exemption holds only during the all-red that follows its own red onset. New invariant counter `committed_at_onset_violations` (must be 0).
- **11.3 Gridlock per junction.** The §5 rule (no stop-line crossing for `gridlock_window_s` while ≥ `gridlock_min_vehicles` are present) is evaluated network-wide as before AND per junction (crossings at that junction, vehicles on its four approach edges). The episode ends at the first detection; `gridlock_junction` records which (−1 = network-wide rule).
- **11.4 Zipper merge at the box entry.** Two vehicles in different lanes that aim for the same connector slot/exit lane are ordered by position with ties broken by vehicle id (the one further along leads); the follower drops back at its own comfortable `b` while the longitudinal gap is ≤ 0 (IDM is not evaluated for a non-positive gap; `idm_accel` raises), and is held stopped at the box edge until the slot it would take is free of a vehicle it must follow (`n_merge_holds`). The in-box overlap guard never pushes a connector position below 0.
- **11.5 One integration per step.** A vehicle that leaves a connector in a step is not integrated again by the edge loop of the same step (it used to advance an extra `v·dt` at every junction crossing).
- **11.6 Data dictionary** covers `network_edges.csv` and the new `episodes` columns (`gridlock_junction, max_approach_red_s, max_total_wait_s, n_forced_starvation, n_emergency_brakes, n_commit_revoked, n_merge_holds`). `validate.py` additionally reports the switch-rate band `switch_rate_band = (0.7, 3.5)`, hard brakes per red onset, the realised-vs-formula capacity ratio of the 1x1 sweep, and a paired max_pressure-vs-fixed_time comparison at design and high demand.

## 11 (continued). Amendments after the recheck (2026-10-03; `traffic_twin` v1.2)

Everything in §§1–11.6 stands except where stated. These entries close the recheck findings NF-02 to NF-05, the stale data-card strings and the calibration correction.

- **11.7 Zipper merge without instantaneous stops (NF-02).** The merge hold of §11.4 used to set a yielding vehicle's speed to 0 at the box edge in one step (an implied deceleration above `b_emerg`, unlogged). Now (a) the merge-leader rule of `leader_on_edge` covers vehicles on sibling connectors and on the vehicle's own connector with the gap measured through the box to the occupant's rear bumper; while the yielding vehicle is abreast of the slot occupant (gap < `overlap_tolerance`, on the edge or into the box) it sees the **box edge as a standing IDM leader** and additionally drops back at least at its comfortable `b` (§11.4), so IDM stops it before the edge end. (b) An edge's vehicles are processed in position order (ties by id) so the vehicle behind, in whichever lane, sees a fresh exit-lane claim of the vehicle ahead. (c) A through/boundary-bound vehicle never plans the exit lane already claimed by the head of the other lane that is ahead of it and about to cross (the "T → fewer vehicles" rule of §4 now excludes the claimed lane), keeps its plan unless the other lane is emptier by more than `merge_plan_hysteresis = 1` vehicle or its lane is blocked, and a moving vehicle keeps its plan within `merge_commit_dist = 15 m` of the box; the plan is not recomputed at box entry. A residual hold at the edge is still counted in `n_merge_holds`; one whose instantaneous stop implies a deceleration above `b_emerg` (`v > b_emerg·dt`) is counted in the new KPI **`merge_hard_stops`** (expected 0; reported by `validate.py`). Effect at the old design demand (2x2, max_pressure, seed 21): merge holds 15 → 0, hard brakes 890 → 842, mean delay unchanged (94 s); realised 1x1 fixed-time capacity 536 → 549 veh/h/approach.
- **11.8 Lane-level liveness KPIs (NF-03).** The starvation guard (§11.1) bounds the time to a *green*, not the time a lane head stands when its exit lane is blocked by spillback (observed ~1200 s under stress). Two KPIs that do not touch the pressure function or the gridlock rule (§5, §11.3): **`max_head_wait_s`** = longest time any lane-head vehicle (first on its lane of an approach edge) stood with `v < stop_speed`, whole episode; **`spillback_lock`** / **`n_spillback_locks`** / `spillback_lock_t` / `spillback_lock_junction` = per junction, at 1 Hz, an approach lane that has been occupied without a stop-line crossing for `gridlock_window_s` although its approach was green for at least two full green periods' worth (`2 × fixed_time_green = 40 s`, counting the running green) within that window; latched per lane until a crossing or until the lane empties, counted once per latch. The lock does not end the episode. Measured worst case (2x2, stress, `random_dwell`, 5 seeds, full episodes): see README.
- **11.9 Calibration against the realised capacity (changes §7).** The §7 formula capacity `measured_sat_flow × lanes × 18/96` (612 veh/h per approach) overstates what the simulated junction realises under fixed time (two-lane zipper merging, turning vehicles capped at `v_turn`): nominal v/c 0.7 ran at ~0.8 and 'high' was over capacity. `calibrate.measure_capacity()` now runs one saturated 1x1 fixed-time probe episode (nominal `x = capacity_probe_x = 1.3` of the formula value, 300 s warm-up + `capacity_probe_main_s = 1200 s`) and takes its main-period throughput / 4 approaches as **`capacity_vph`**; the formula value is kept as **`capacity_formula_vph`** and **`capacity_realisation`** = measured / formula. `lambda_base` per level = v/c target × `capacity_vph` / (flow on the busiest internal approach per unit lambda). All three are `SimConfig` fields written to `config.json` and the DATACARD. With the default config: saturation flow 1631 veh/h/lane, formula 612 veh/h, measured 549 veh/h, realisation 0.897; `lambda_base` low 186.1, design 325.7, high 442.0, stress 535.0 veh/h per entry node (v1.1: 207.4 / 362.9 / 492.5 / 596.2). `validate.py` compares, on the 1x1 sweep, the v/c per level realised in the sweep with the one implied by the probe (`demand_sweep_capacity.vc_levels_probe_vs_sweep`, `probe_vs_sweep_within_5pct`); this is a consistency check of the probe against the sweep, not an independent validation (see §11.12). CLI and API of `generate.py` are unchanged; `calibrate_demand(cfg, sat_flow_vphpl=None, seed=0, capacity_vph=None)` gains an optional `capacity_vph` that skips the probe. Datasets generated with v1.0/v1.1 keep their nominal levels; `data/traffic_v1` and `data/dev10` were regenerated with v1.2 afterwards (§11.12).
- **11.10 Documentation corrections (NF-04/NF-05, stale strings).** README: the liveness claim now states the measured worst case (§11.8) instead of "never more than ~190 s". DATACARD: the hard-brake sentence quotes the measured per-red-onset rate and the emergency-brake share from `reports/validation_summary.json` when present and claims no number otherwise. `data_dictionary`: `y_mp_action` is described as the `priority_mp` teacher's request (= the max-pressure request on every row without an ambulance); `vehicle_type` is `{car, ambulance}`; the data card lists the five RNG streams (demand, vehicles, controller, sensor, ev) and names the two vehicle types in the limitations.
- **11.11 Versioning.** `GENERATOR_VERSION = traffic_twin_v1.2`; `config_hash` changes because of the new fields (`merge_commit_dist`, `merge_plan_hysteresis`, `capacity_formula_vph`, `capacity_vph`, `capacity_realisation`, `capacity_probe_x`, `capacity_probe_main_s`). New `episodes` columns: `merge_hard_stops, max_head_wait_s, spillback_lock, n_spillback_locks, spillback_lock_t, spillback_lock_junction`. Tests: `tests/test_review_round2.py`.

- **11.12 Final review (2026-10-03; `traffic_twin` v1.2, no change to the simulator physics).** (a) *Releases.* `data/traffic_v1` (120 episodes) and `data/dev10` (10 episodes) were regenerated with v1.2 (`config_hash ecd36a4b3cad`); this supersedes the last sentence of §11.9. (b) *Validation wording.* `demand_sweep_capacity.realised_vc_per_level` is renamed `vc_levels_probe_vs_sweep` and `realised_vc_within_5pct` is renamed `probe_vs_sweep_within_5pct`: the sweep and the probe run the same simulator, so their agreement checks the calibration arithmetic, not the capacity itself. The independent evidence is the saturation-flow measurement (§6) and the fixed-time delay curve. `validation_summary.json` now carries `generator_version`, `config_hash` and `created_utc`, and `generate.py` quotes a summary in the data card only when its `generator_version` equals the current one. (c) *Load on the 2x2 grid.* The 1x1 calibration does not make the nominal levels sub-capacity for every controller on the grid. New key `grid_levels_2x2` (design, high and stress under `fixed_time` and `max_pressure`, one seed) and the table of mean delay, throughput, `n_in_network_end` and `switches_per_min` per level and controller in the data card report it. In the release, 11 of 17 high-demand `max_pressure` / `max_pressure_eps` episodes end with more than 380 vehicles in the network (median 399, maximum 512), against a mean of 68 (maximum 198) for `fixed_time` at the same level: for the max-pressure family 'high' is over capacity, and the ML models trained on that teacher inherit it. (d) *Liveness and gridlock sentences.* The data card computes them from `episodes.csv` (longest red wait 296.5 s, longest lane-head standing time 581.5 s, 3 of 120 episodes in gridlock, all `random_dwell` at high demand, none under stress) and `README.md` quotes the same release maxima; the earlier wording was contradicted by the release. (e) *Residual merge holds.* `n_merge_holds` counts every hold at the box edge; `merge_hard_stops` counts only holds whose instantaneous stop implies a deceleration above `b_emerg` (speed above `b_emerg * dt` = 3.5 m/s). A hold at 1.5 < v <= 3.5 m/s implies an abrupt stop between 3 and 7 m/s^2 that is counted in `n_merge_holds` but not flagged. The release has 1 hold in 120 episodes and no hard stop, so this is documented and not changed; changing the rule or the counters would change the released data. (f) *Documentation.* `data_dictionary.csv` and `DATACARD.md` no longer cite DECISIONS sections or script names (they travel with the data); `python3 -m traffic_twin.generate --rewrite-docs DIR` rewrites both from the release's own `episodes.csv`, `signal_ticks.csv` and `config.json` without simulating. The CSV tables are unchanged; only the `data_dictionary.csv` line of `SHA256SUMS.txt` was refreshed.

## 12. Safe driving, lane changing and safety measurement (2026-10-03; `traffic_twin` v2.0)

The user asked for driving that cannot produce the near misses and touches of v1.2 ("cars that near-miss hit each other can be dangerous") and for lane changing and the related realism items. v2.0 is coded and tested; by the user's decision the v1.2 release `data/traffic_v1` is kept and the grid is **not** regenerated (the grid stays as the validated fallback of the v3 twin, §13).

- **12.1 Junction box and paths.** The box is `w_box = road_width + 2·corner_radius` (corner radius 6 m, so 26 m with two lanes). Each junction has 24 lane-level paths (through k→k; near-side turn 0→{0,1}; far-side turn 1→{1,0}), tangent-continuous with lead-in arcs, and a discrete curve-speed profile from `a_lat_max = 3.0 m/s²` reached with `b_curve = 2.0 m/s²`. Conflict zones between paths are precomputed (`network_paths.csv`); a vehicle enters only when every zone on its path is clear (`s_clear` gate). An all-red that cannot finish because the box will not clear re-picks after `allred_stall_s = 10 s`.
- **12.2 Car following.** ACC (improved IDM + constant-acceleration heuristic, coolness 0.99), capped by the free-road acceleration. A vehicle takes the minimum acceleration over every obstacle it sees: leader, stop line (`s0_line = 0.5 m`), protruding box bodies, conflict-zone occupants. A standing obstacle caps braking kinematically: a ≥ −max(v²/(2(gap−s0)), b). Red onset: a vehicle stops if d ≥ v·t_pr + v²/(2·b_stop), t_pr = 1 s, otherwise it commits.
- **12.3 Lane changing.** MOBIL with safe braking `lc_b_safe = 2.0` (urgent 3.0), minimum gap 2 m, minimum TTC 1.5 s, cooldown 4 s, no changes within 30 m of the stop line, a 3 s shadow on both lanes while changing, no hopping between standing queues. Look-ahead lane needs make vehicles reach the lane of their next movement early; a vehicle that reaches the solid line in the wrong lane reroutes (or changes destination through `exit_node`) instead of forcing in.
- **12.4 Ambulances.** Cars yield where there is room: space-limited yielding toward the kerb, no cut-in within 100 m in front of an ambulance, a 20 s block on returning, and a mandatory return with cooperation from the vehicles behind. A full two-lane queue at red cannot yield, so the signal still has to clear the queue ahead (the proposal's 2.4 claim). The ambulance is drawn in yellow.
- **12.5 Safety measurement.** `SafetyMonitor` logs conflicts with raw values (TTC < 1.5 s, DRAC ≥ 3.4 m/s², PET, zone collisions; hysteresis per pair) to `conflicts.csv`. Invariants: collisions, critical conflicts, `decel_limit`, `lane_discipline` (all must be 0 in tests).
- **12.6 Switches.** `vehicle_mix` (optional motorcycles, off), `lane_changes` (off for the saturation-flow measurement of §11.9). `GENERATOR_VERSION = traffic_twin_v2.0`. Tests: `tests/test_v2_safety.py` plus the updated earlier suites (82 tests).

## 13. Real-place twin: Perempatan Fatmawati (2026-10-03; `traffic_twin.fatmawati`, v3.0)

User decisions (2026-10-03):
- **Engine and scope:** replace the "#" grid with a copy of Perempatan Fatmawati (Cilandak, South Jakarta), main roads within 2.5 km, in SUMO 1.27.1.
- **Vehicles and viewer:** motorcycles with a research-based share; a 3D viewer with zoom.
- **Traffic data:** TomTom live only, not stored.
- **Dataset:** "tabular stored in csv for sensor not engine".
- **Signals:** the real Dishub 4-stage plan.
- **Calibration:** demand calibrated to the observed congestion.

- **13.1 Network.** `build_net.py` downloads OSM (motorway to tertiary, 2.7 km box; © OpenStreetMap contributors, ODbL) and runs netconvert.
  - netconvert options: left-hand traffic, `--ignore-widths` (uniform 3.25 m lanes), `--check-lane-foes.all`, `--no-turnarounds` (U-turns only where OSM maps a median opening, such as the loops 51 m east and 169 m west of the junction).
  - Two-way single-lane streets are widened to 3 m. A Jakarta type map sets arterial speeds and keeps motorcycles off toll roads.
  - The junction is two crossings 55 m apart under the JORR viaduct, run as one controller `fatmawati`.
  - Free-left slips and the U-turn loops are unsignalised.
  - Motorcycle boxes (RHK, SE PUPR 52/SE/M/2015): cars stop 8 m behind the motorcycle stop line (stopOffset exceptions="motorcycle").
  - Removed because the counts show none: the RA Kartini right turn and U-turns between the median segments.
  - Program `dishub` (Sumabrata et al. 2026, Dishub static plan):
    - S1: RA Kartini (W) through + TB Simatupang (E) through.
    - S2: E through + E right.
    - S3: north. S4: south.
    - Greens 80/12/78/45 s, amber 2 s, all-red 2 s; cycle 231 s (the paper prints 253 s, but its stage greens sum to 231 s).
    - For every ordered stage pair: an amber and an all-red phase. Links green in both stages stay green; median links keep green through the amber and get amber in the all-red.
  - Result: 706 edges, 431 nodes, 10 signal controllers; `fatmawati` has 20 links.
- **13.2 Vehicles.** EIDM for every class, five latent-aggressiveness profiles per class, sublane model at 0.8 m lateral resolution.
  - Mix: 80% motorcycle, 19% car, 0.5% bus, 0.5% truck. This follows the north-leg counts of 77–82% motorcycles and under 1% heavy vehicles (Yusuf & Isradi 2025). 70% (Jakarta-wide) is an evaluation-only sensitivity level.
  - Motorcycles: 2.0 × 0.8 m, a ≈ 2.0 m/s², T ≈ 0.8 s, v0 ≈ 11 m/s (modelling assumptions); standstill gap 0.2 m and lateral gap 0.3 m from the VISSIM model of this junction.
  - Motorcycles filter anywhere on a road (latAlignment arbitrary) but ride the lane centre while on a junction and for 3 s after it. SUMO checks only crossing and merging movements against each other, so filtering motorcycles at the outer edges of side-by-side movements overlapped. Measured over 30 min at design demand, three seeds: about 9 junction overlaps and no teleports with the rule, against 22 overlaps and 18 teleports without. Two alternatives were tested and rejected because they were worse: a lateral band before junctions, and centring only after split paths separate.
  - Ambulances (yellow) obey the signals. A moving car directly ahead within 80 m moves toward the kerb if SUMO's lane-change check allows it.
- **13.3 Demand and calibration.**
  - Junction demand is the 13 movements counted by CCTV in a weekday peak hour (Sumabrata, Haqueline & Mauramdha 2026; about 25,900 veh/h entering).
  - Each movement enters at its leg's upstream end and leaves at the exit leg's far end. It is routed through that leg's entry gate and the exit leg's exit gate, so the movement happens at Fatmawati.
  - Background fringe-to-fringe trips use explicit routes that stay 200 m from the junction.
  - Levels are 0.75 / 1.0 / 1.15 × the calibrated peak, times the hour's share of the daily-profile peak.
  - Calibration (`calibrate.py`) runs the Dishub fixed plan at a grid of volume scales. Per approach it takes the scale at which the peak-hour maximum queue matches the VISSIM maximum queue: N 289.5, S 166.6, E 239.0, W 155.5 m. The queue uses the VISSIM queue definition: below 5 km/h, contiguous from the stop line with gaps up to 20 m.
  - Validation: PKJI 2023 delays and the VISSIM junction mean of 67.4 s. The result is in `data/fatmawati/calibration.json` (numbers in 13.11).
- **13.4 Simulation layer** (`sumo_world.py`).
  - libsumo, 0.5 s steps. Collisions are checked on lanes and junctions with `collision.mingap-factor 0`.
  - The plant runs the stages: min green 5 s, max green 90 s (adaptive), decisions every 5 s. The starvation guard serves a stage with standing vehicles after 240 s of red, oldest red first.
  - Controllers:
    - `fixed` (Dishub greens);
    - `longest_queue` (camera lane groups, margin 3);
    - `max_pressure` (approach pressure shared over lane groups by standing PCU, margin 1 PCU);
    - `priority_mp` (the teacher): the stage of an ambulance whose route crosses the signal when ETA ≤ I + l_s + (3600/S)·Σ e_k n_k + Δ, with I = 4 s, l_s = 2 s, S = PKJI 2023 base 600·W_e, e = 0.15 / 1.0 / 1.3 and Δ = 5 s; otherwise max-pressure;
    - `preempt_naive` (Dishub plan, jumps to the ambulance's stage at ETA ≤ 20 s and holds until it has crossed);
    - `random`, from its own RNG stream;
    - `still`;
    - `learned:<path>`.
  - Quality diagnostics: lane collisions, junction overlaps, teleports.
  - Known limitation: SUMO's sublane model gives rare lane collisions between motorcycles at walking pace (centimetre overlaps), more at high density. They are reported per episode, not hidden.
- **13.5 3D viewer** (`webapp3d.py`, `static/index.html`).
  - Three.js with OSM buildings and lanes from the network; instanced multi-part vehicles: a car with glass and wheels, a motorcycle with its rider, a yellow ambulance with a flashing bar, a bus and a truck.
  - Camera presets (overview, junction, motorcycle box, toll flyover, hospital, follow a vehicle type) and `?cam=`.
  - A 10-minute warm-up before the first frame; panels show camera readings; compact layout on narrow windows.
  - The JORR toll road (Jalan Tol Lingkar Luar Jakarta) crosses between the two Fatmawati crossings on a viaduct.
    - In the network this was already so: the toll edges share no node with the streets below, so toll traffic never conflicts with the junction.
    - netconvert lifts OSM bridge ways by 6 m per layer (`--osm.layer-elevation 6`), so the deck is 6 m up and the toll vehicles' SUMO z is on it.
    - The viewer draws the deck (road surface, parapets, girder sides, underside), hammerhead piers about every 30 m and lamp posts (`geometry3d.flyover`).
    - Piers are never placed in a lane of another road or in a ground-level junction; the span grows instead, as over Jl. RS Fatmawati.
    - Buttons: "Toll flyover" (camera) and "See-through flyover" (watch the junction under the deck).
  - Heights are shown on the toll roads only (`geometry3d.raised_parts`).
    - netconvert also lifts short river bridges, e.g. on RA Kartini about 300 m west, into 6 m humps; in the simulation that adds only a few centimetres of lane length.
    - The viewer draws all other roads, and the vehicles on them, at ground level.
    - A toll ramp that lands on such a lifted street junction comes down to the ground over its last 60 m.
    - The network itself is unchanged, so the released data and the evaluation still match it.
  - The deck height is netconvert's 6 m, not a surveyed value. The real JORR deck is probably higher (5.1 m minimum clearance plus the girder).
- **13.6 Sensors** (`sensors.py`). The dataset is what a roadside installation would record. Sensors are exact.
  - Stop-line loops: one per approach lane, 1 m before the stop line.
  - Entry and exit gates: every lane of each leg at 150 m (185 m on RA Kartini, outside its U-turn loop). Re-identification of entry and exit gives turning counts and cordon travel times.
  - Queue cameras: 300 m upstream of each stop line, reporting standing vehicles (< 0.1 m/s) by class, the VISSIM queue length, motorcycles in the box, speed, and standing vehicles per lane group (the stages that serve the lanes).
  - Exit cameras: the first 150 m of each exit.
  - Priority-request detector: an ambulance in an approach camera zone, with distance, ETA, the PCU queue ahead of it, and whether its route crosses the stop line.
  - A vehicle counts once per gate. Vehicle identities are not written out.
- **13.7 Release** (`generate.py`, CSV only, ODbL).
  - Tables: `sensors`, `episodes` (scenario and sensor-derived outcomes: stopped delay per approach, fairness ratio, cordon travel time, ambulance travel time and recovery), `loops_1min`, `cameras_1min`, `ticks_5s` (features, `y_teacher_stage`, `y_q_veh_30s_<leg>`; roles in `data_dictionary.csv`), `signal_log`, `turning_counts_15min`, `ambulance_trips`, `simulation_qc` (engine diagnostics, not sensor readings).
  - Plan: 3 windows (07:00, 12:00, 17:00) × 3 levels × 4 controllers (fixed, longest_queue, max_pressure, priority_mp) × 3 seeds = 108 episodes. Each has 15 min warm-up then 60 min recorded.
  - The same trips go to every controller within a window, level and seed.
- **13.8 TomTom, display only** (`tomtom.py`).
  - The free tier forbids storing results (Cache-Control max-age=0, T&C 11.4), derived databases and mixing with our or OSM data (11.6.1, 20.2.3), and use in health-critical systems (11.6.9).
  - The viewer therefore shows TomTom's current and free-flow speed per carriageway next to the twin's camera speeds, labelled ©TomTom, to the first viewer only.
  - Polling runs only while a viewer is open: 8 probes per 60 s, with a monthly cap of 18,000 of the 20,000 free requests. Only the request count is stored.
  - Nothing feeds the simulation or any decision.
- **13.9 Google Cloud** (`deploy/gcp/`).
  - The live twin runs as a private Cloud Run service in asia-southeast2: max 1 instance, scales to zero, opened through `gcloud run services proxy`.
  - A Cloud Storage bucket holds the CSV releases and the TomTom request counter (a volume mount). The TomTom key is in Secret Manager.
  - The user creates the project, billing and TomTom key and runs `deploy.sh`, which asks before each step.
- **13.10 Evaluation grid** (`evaluate.py`; design agreed with the methodology session).
  - Ambulances are released every 10 min at fixed points 400 m before the stop line: N through, E free-left, and the W detour (W through, U-turn over the toll ramps east, E free-left). Releases cover 8 slices of the 231 s cycle.
  - Each run has a paired no-ambulance run and an empty-road reference (signal off: N 46.5, E 41.0, W 174.5 s).
  - Outputs:
    - `passages.csv`: junction time, delay, travel time, recovery (censored at episode end, flagged when it overlaps the next release), extra wait per approach, detection signal state.
    - `episodes_eval.csv`: stopped delay, served count and longest red per approach.
    - `evaluation_qc.csv`.
  - Options: `--moto-share`; `--hindsight` (best switch time per passage, for one seed per cell) is planned.
- **13.11 Calibration outcome and the motorcycle gaps (2026-10-03, late).**
  - **Gaps:** motorcycle gaps are back at 0.5 m standstill and 0.5 m lateral.
    - At calibrated demand, the VISSIM values (0.2 m / 0.3 m) gave 1.41 lane collisions per 1,000 veh-km against 0.09 with 0.5 m / 0.5 m (2 seeds × 75 min each), with the same queue fit.
    - Blocking lateral moves while standing removed the rest in the study area but raised congestion, so it was not kept.
    - This departs from the approved calibration option, which named the VISSIM gaps, because crash-free driving is the user's stated priority.
  - **Scales:** damped multiplicative rounds overshot. Near capacity an approach's hourly maximum queue tips from a short queue to spill-back past the 300 m camera; at a fixed scale it varies between 125 and 270 m across seeds on TB Simatupang; and the approaches interact through shared exits. The scales were therefore bracketed and checked over 4 seeds.

    | Approach | Scale | Max queue | Target | Status |
    |---|---|---|---|---|
    | N | 0.242 | 306 m | 289.5 m | +6%, fills the camera zone |
    | E | 0.239 | 194 ± 59 m | 239 m | −19% |
    | S | 0.220 | 127 ± 4 m | 167 m | −24% |
    | W | 0.360 | 73 m | 155.5 m | capacity-limited: RA Kartini backs up upstream of the camera above about 0.38 |

  - **Volumes:** the twin carries 22–35% of the counted volumes at these queues, so it reproduces the congestion, not the counts.
  - **Stopped delay:** N 191, E 98, S 86, W 42 s (PKJI control delay 80 / 79 / 106 / 78 s).
  - **Safety per 75 min at this density:** 7 lane collisions (2.5 within 300 m of the junction), about 280 junction overlaps (about 180 within 300 m), about 38 teleports. All are reported in `simulation_qc.csv` and the data card.
