# Traffic-light digital twin (v1: the simulation model)

A small, readable microscopic traffic simulator of a 2x2 grid of signalised
junctions (left-hand traffic, two lanes per direction) that produces a tabular
dataset for supervised learning. Every design number is in
`traffic_twin/config.py`; the authoritative decisions are in `docs/DECISIONS.md`.

* Vehicles follow the Intelligent Driver Model, with a latent driver
  aggressiveness `alpha` that shifts desired speed, acceleration and headway.
* Signals use split phasing (one approach green at a time). A plant per
  junction enforces min/max green, an all-red of at least 4 s that is extended
  until the box is empty and no committed vehicle is still upstream of it, and
  a starvation guard (`forced = 4`: an occupied approach is served after 180 s
  of red), so every controller is safe by construction. Liveness is bounded,
  not guaranteed: the guard bounds the wait for a *green*, but a blocked box
  extends the all-red and a blocked exit lane keeps a lane head standing under
  green. Measured worst case in the 120-episode release (`data/traffic_v1`):
  under `random_dwell` the longest red an occupied approach saw was 296.5 s,
  the longest a lane-head vehicle stood was 581.5 s (`episodes.max_head_wait_s`),
  one episode had 7 spillback locks (`episodes.n_spillback_locks`) and 3 of the
  4 `random_dwell` episodes at high demand ended in gridlock. No `fixed_time`,
  `max_pressure` or `max_pressure_eps` episode ended in gridlock; for them the
  longest red wait was 192 s and the longest lane-head standing time 191.5 s.
  Gridlock (no crossing for 120 s
  with >= 10 vehicles waiting, network-wide or at one junction) ends the
  episode and is kept as a labelled failure.
* Controllers (`fixed_time`, `longest_queue`, `max_pressure`,
  `max_pressure_eps`, `random_dwell`) read the same feature row that is logged
  to `signal_ticks.csv`; the max-pressure rule is also the label `y_mp_action`.

* Healthcare scenario (DECISIONS §10): ambulances spawn from their own random
  stream (cars keep common random numbers), are observed per approach through
  `ev_present/ev_dist/ev_eta/ev_q_ahead`, and two more controllers exist:
  `preempt_naive` (force green when an ambulance is 20 s away) and the
  priority-aware teacher `priority_mp`, which is the label `y_mp_action`.

## Install and test

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q                      # geometry, IDM, stop-or-commit, invariants, determinism, labels
```

## Validate the physics (M0)

```bash
python3 -m traffic_twin.validate          # add --quick for a 4-point demand sweep
```

Measures the saturation flow from a 20-vehicle queue discharge (expect 1600-1900
veh/h/lane), measures the capacity the junction realises under fixed time in a
saturated 1x1 probe episode (549 veh/h per approach = 0.90 x the DECISIONS §7
formula value 612; demand levels are calibrated against the measured value,
DECISIONS §11.9), sweeps demand on a single junction under fixed time (delay must
blow up near capacity; the realised v/c per level is a probe-versus-sweep
consistency check, not an independent test), runs one 2x2 design-demand episode
with `max_pressure` (zero invariant violations, delay 30-90 s/veh, switch-rate
band 0.7-3.5/min, hard brakes per red onset, merge hard stops = 0, lane-level
liveness KPIs), draws a ring-road fundamental diagram, compares `max_pressure`
with `fixed_time` on paired seeds at design and high demand, and runs the 2x2
grid at design, high and stress demand under `fixed_time` and `max_pressure`
(`grid_levels_2x2`: network load and queue growth, because the 1x1 calibration
does not guarantee that the nominal levels are sub-capacity for every
controller on the grid). Figures go to `reports/figures/`, numbers to
`reports/validation_summary.json` (stamped with generator version, config hash
and creation time). Known property of the specified teacher: it flickers (~4.2
switches/min) and loses to fixed time at high/stress demand; see the DATACARD
and DECISIONS §11.

## Generate a dataset (M1/M2)

```bash
python3 -m traffic_twin.generate --episodes 10 --out data/dev10 --trajectories 0
python3 -m traffic_twin.viewer data/dev10/vehicle_states_0.csv     # optional animation
```

`data/<run>/` contains `episodes.csv` (one row per episode with settings and
KPIs), `vehicles.csv` (one row per vehicle), `signal_ticks.csv` (the ML table:
one row per junction per decision instant, features + actions + labels),
`junction_1hz.csv` (1 Hz aggregates used for the 30 s-ahead label),
`network_edges.csv`, `data_dictionary.csv`, `config.json` and `DATACARD.md`.
Demand levels are calibrated before the first episode (`traffic_twin/calibrate.py`)
from the saturation flow AND the fixed-time capacity measured in the simulator
(`config.json`: `measured_sat_flow_vphpl`, `capacity_formula_vph`, `capacity_vph`,
`capacity_realisation`), never from the textbook 1800 veh/h/lane. Datasets shipped
in `data/` were generated by v1.0/v1.1 (nominal levels); see DECISIONS §11.9.

## Train the models, close the loop, render the R demo (M3)

All three commands run from the project root and read `data/traffic_v1`
(generator `traffic_twin_v1.2`); outputs go to `reports/ml` and `reports/r_demo`.

```bash
# 1. Supervised models (Task A imitation of the priority-aware teacher, Task B queue 30 s ahead).
#    Feature sets are selected by namespace from data_dictionary.csv and leak-checked in code.
#    Writes reports/ml/results.json, RESULTS.md (sections 1-3), models/*.joblib, figures/, ~5 min.
python3 -m traffic_twin.ml --data data/traffic_v1 --out reports/ml

# 2. Closed loop (DECISIONS §10 protocol + the proposal's 10% acceptance bound).
#    Deploys reports/ml/models/taskA_hgb_sensor.joblib as controller `ml_sensor` (alias `ml_v1`):
#    {fixed_time, max_pressure, preempt_naive, priority_mp, ml_sensor} x {design, high} x 8 paired seeds,
#    each also run with ambulances off (car_mean_delay_penalty_s), plus ml_sensor and max_pressure at
#    sensor_noise_level 0.1. Asserts the simulator's config hash equals the dataset's. ~8 min with 8 workers.
#    Writes closed_loop.csv, closed_loop_diffs.csv, closed_loop_bound.csv, closed_loop_summary.csv,
#    closed_loop_meta.json, figures/closed_loop*.png and RESULTS.md sections 4-6.
python3 -m traffic_twin.evaluate --data data/traffic_v1 --models reports/ml/models --out reports/ml
python3 -m traffic_twin.evaluate --report-only --out reports/ml      # rebuild tables/figures/RESULTS.md only

# 3. R Markdown demonstration of the dataset (reports/r_demo/Assignment3_demo.Rmd -> .html).
RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64 \
  Rscript -e 'rmarkdown::render("reports/r_demo/Assignment3_demo.Rmd")'

# tests of the ML pipeline and the evaluator (leak guards, splits, controller contract, acceptance bound)
python3 -m pytest -q tests/test_ml.py
```

Headline numbers and the honest verdict are in `reports/ml/RESULTS.md`; every
number there is generated from `results.json` and `closed_loop*.csv`.

## Layout

| file | what it does |
|---|---|
| `traffic_twin/config.py` | `SimConfig`: every number, `to_json`, `config_hash` |
| `traffic_twin/network.py` | grid, directed edges with lanes, connectors, turn exits by heading rotation, Dijkstra routing |
| `traffic_twin/vehicles.py` | parameter sampling, IDM, ballistic update |
| `traffic_twin/signals.py` | signal plant per junction |
| `traffic_twin/controllers.py` | baselines and the max-pressure label function |
| `traffic_twin/features.py` | `build_features()` = the `signal_ticks` row |
| `traffic_twin/sim.py` | `World`: spawning, routing, car following, box rules, logging, KPIs |
| `traffic_twin/calibrate.py` | saturation flow and realised capacity measurement, v/c -> arrival rates |
| `traffic_twin/validate.py` | M0 validation experiments and figures |
| `traffic_twin/generate.py` | dataset CLI (CSV only) |
| `traffic_twin/viewer.py` | matplotlib animation of a trajectory file |
| `traffic_twin/ml.py` | supervised models on `signal_ticks` (feature sets by namespace, leak assertions, Task A / Task B) |
| `traffic_twin/evaluate.py` | closed-loop protocol: `MLController`, paired seeds, acceptance bound, secondary metrics, noise variant |
| `reports/ml/` | `RESULTS.md`, `results.json`, `closed_loop*.csv`, `models/`, `figures/` |
| `reports/r_demo/` | the R Markdown demonstration (`Assignment3_demo.Rmd`) |
| `tests/` | pytest suite |

## The Fatmawati twin (v3)

A copy of Perempatan Fatmawati (Cilandak, South Jakarta) in Eclipse SUMO, with the main roads within 2.5 km. Details and the reasons for each choice are in `docs/DECISIONS.md` §13.

- **Network:** OSM, © OpenStreetMap contributors, ODbL.
- **Signal:** Dishub's real 4-stage plan.
- **Traffic:** 80% motorcycles with motorcycle boxes. Demand comes from the 13 movements counted at the junction, scaled so the twin reproduces the observed queues.
- **Dataset:** CSV tables of virtual-sensor readings (loops, counting gates, queue cameras, signal log, ambulance trips), not simulator internals.

Requires `deploy/gcp/requirements-twin.txt` (eclipse-sumo, libsumo, sumolib 1.27.1).

Build the network and the 3D geometry:

```bash
python3 -m traffic_twin.fatmawati.build_net
```

```bash
python3 -m traffic_twin.fatmawati.geometry3d
```

Run the live 3D viewer, then open http://localhost:8770. The camera buttons cover the junction, the motorcycle box, the JORR toll flyover, the hospital and following a vehicle. "See-through flyover" makes the toll deck transparent so you can watch the junction underneath.

```bash
python3 -m traffic_twin.fatmawati.webapp3d --port 8770
```

Re-fit the demand to the published queues. This writes `data/fatmawati/calibration.json`.

```bash
python3 -m traffic_twin.fatmawati.calibrate --workers 8 --start '{"N":0.242,"E":0.239,"S":0.22,"W":0.36}'
```

Generate the sensor release: 108 episodes, about 2 h on 8 workers.

```bash
python3 -m traffic_twin.fatmawati.generate --plan release --out data/fatmawati_v3 --workers 8
```

Run the ambulance evaluation grid. It writes `passages.csv`, `episodes_eval.csv` and `evaluation_qc.csv`.

```bash
python3 -m traffic_twin.fatmawati.evaluate --out data/fatmawati_eval --levels low,design,high --seeds 1,2,3,4 --workers 8
```

Run the hindsight ceiling: the best switch time per passage, from forked exact replays.

```bash
python3 -m traffic_twin.fatmawati.evaluate --hindsight --out data/fatmawati_eval_hindsight --levels low,design,high --seeds 1
```

Run the twin's tests:

```bash
python3 -m pytest -q tests/test_fatmawati.py tests/test_fatmawati_eval.py tests/test_tomtom.py
```

Live TomTom speeds are display only and need `TOMTOM_API_KEY` (see `traffic_twin/fatmawati/tomtom.py`). To run the viewer on Google Cloud, see `deploy/gcp/README.md`.
