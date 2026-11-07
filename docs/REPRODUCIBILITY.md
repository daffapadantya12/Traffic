# Reproducibility: re-running the Fatmawati study, and taking it to a second site

This is the runbook for `traffic_twin.fatmawati` (v3.0): what it takes to reproduce the released dataset
and results, and the protocol for repeating the study at another junction. Bundaran Hotel Indonesia
(Bundaran HI, Central Jakarta) is worked through as the transfer case in Section 7. Design reasons are in
`docs/DECISIONS.md` §13; the site facts are in `data/sites/`.

## 1. Four levels of reproduction

| Level | What is repeated | What to expect | How to check |
|---|---|---|---|
| 0 | the same checkout, the same machine, the same seeds | byte-identical CSV tables | `shasum -a 256 -c data/fatmawati_v3/SHA256SUMS.txt` after a regenerate into a fresh folder, then compare |
| 1 | a fresh install from the pinned versions (Section 2) on another computer | the same numbers; wall times differ | the rebuilt `net/fatmawati.net.xml` without netconvert's header comment (`<!-- generated on ... -->`, which carries the build time and absolute paths) hashes to `646b3acc5bd6f28d2e71694ca30f63322a9bd65232170c4e1f164861ea65f50b`; the released file's full hash `d389d8d6ceea97b6b243957f2c742bc7e09f6bc05789e27ddaf6f195cb0d5eff` (stamped in `config.json`) is reproduced only by keeping that file; `pytest tests/test_fatmawati.py` passes |
| 2 | a new OpenStreetMap extract (`build_net --force-download`) | a different network: edge ids, lane counts and geometry move with the map, so every raw id in `data/sites/fatmawati.json` must be re-found and the calibration refitted | the hash differs; treat it as a new release, not a reproduction |
| 3 | a new site | the protocol in Section 5, and the remaining code changes in Section 6 (the stage model) first | `python3 -c "from traffic_twin.fatmawati import site; site.load_site('<slug>')"` names the first missing value; then the site's own smoke release |

Level 2 is why the OSM extracts are committed (`data/fatmawati/osm/*.osm.xml`, Overpass `osm_base`
2026-10-03T11:43:02Z) even though they are large: the study is reproducible from the pinned extract, not
from "OpenStreetMap today".

## 2. Environment

| Component | Version used for the release | Where pinned |
|---|---|---|
| Python | 3.13.2 (miniconda, macOS arm64) | `data/fatmawati_v3/config.json` `python` |
| Eclipse SUMO, libsumo, sumolib | 1.27.1 (pip wheels; `netconvert` comes from the wheel's `SUMO_HOME`) | `deploy/gcp/requirements-twin.txt`; stamped per episode in `simulation_qc.csv` `sumo_version` |
| numpy, pandas, pyproj | 2.2.6, 2.3.3, >= 3.6 | same file (lower bounds only) |
| scikit-learn, matplotlib, pytest | 1.7.2 measured on 2026-11-07; not pinned | `requirements.txt` (no versions) |
| R, rmarkdown, RStudio's pandoc | RStudio-bundled pandoc at `/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64/pandoc`; headless Google Chrome prints PDFs | `reports/*/build_*.py` |

Install:

```bash
python3 -m pip install -r requirements.txt -r deploy/gcp/requirements-twin.txt
```

Known environment notes: libsumo warns that pyarrow 22 may not match its libarrow (harmless for this
pipeline); `rtree` is absent, so sumolib falls back to brute-force spatial queries (slower, same results);
`learn.py` must run with one OpenMP thread per estimator, because two threads beside eight SUMO workers made
histogram gradient boosting 30 to 50 times slower (`learn.py` imports `sklearn.ensemble` before setting the
thread limit for that reason). Record `pip freeze` with any new release; the pipeline does not do it.

## 3. The command sequence

All commands run from the project root. Steps 2 and 10 to 12 are not in `README.md`. Times are what this
project measured on a 10-core Apple-silicon Mac that was also running other work, so plan with them, not with
test-run speeds. Every command takes `--site <slug>` (default `fatmawati`, or `$TWIN_SITE`); the site facts
named below (`build_net.py` constants, `MOVEMENTS`, `TARGET_MAX_QUEUE_M`, ...) are read from
`data/sites/<slug>.json` since 2026-11-07 (DECISIONS §13.13).

| # | Step | Command | Reads | Writes | Seeds | Measured time |
|---|---|---|---|---|---|---|
| 1 | Network | `python3 -m traffic_twin.fatmawati.build_net` | Overpass (only without the cached extract), `patches/jakarta.typ.xml`, the constants in `build_net.py` | `osm/fatmawati_roads.osm.xml`, `net/fatmawati_raw.net.xml`, `net/fatmawati.net.xml`, five generated patch files | none | minutes |
| 2 | Sensor layout (implicit) | first call of `sumo_world.sensor_layout()` or `demand.gate_edges()` | the net and its signal program | `sensors/sensors.add.xml` | none | seconds |
| 3 | Vehicle types and a demand smoke file (optional) | `python3 -m traffic_twin.fatmawati.demand` | net, `calibration.json` if present | `demand/vtypes.add.xml`, `demand/design_0800.rou.xml` | 0 | seconds |
| 4 | Calibration | `python3 -m traffic_twin.fatmawati.calibrate --workers 8 --start '{"N":0.242,"E":0.239,"S":0.22,"W":0.36}'` | net, vtypes, `MOVEMENTS`, `TARGET_MAX_QUEUE_M` | `data/fatmawati/calibration.json` | fit 11, 12; verify 11 to 14 | hours (5 scales x 2 seeds x 75 min simulated, 3 to 5 x real time) |
| 5 | Smoke release | `python3 -m traffic_twin.fatmawati.generate --plan smoke --out data/fatmawati_smoke --workers 1` | net, calibration | the nine tables, `data_dictionary.csv`, `DATACARD.md`, `config.json`, `SHA256SUMS.txt` | 1 | ~2 min |
| 6 | Release | `python3 -m traffic_twin.fatmawati.generate --plan release --out data/fatmawati_v3 --workers 8` | same | same, 108 episodes | 1, 2, 3 | 82 min to 2 h |
| 7 | Verify the release | `shasum -a 256 -c data/fatmawati_v3/SHA256SUMS.txt` and `python3 -m pytest -q tests/test_fatmawati.py` | | | | minutes |
| 8 | Label validation and training | `python3 -m traffic_twin.fatmawati.learn validate --data data/fatmawati_v3` then `python3 -m traffic_twin.fatmawati.learn train --data data/fatmawati_v3 --out reports/a3/learning --jobs 2` | the release | `reports/a3/learning/` (`models/taskA_*.joblib`, metrics) | `random_state=0`; test split = seed 3 | 25 min |
| 9 | Evaluation grid | `python3 -m traffic_twin.fatmawati.evaluate --out data/fatmawati_eval --levels low,design,high --seeds 1,2,3,4 --workers 8` | net, calibration | `passages.csv`, `episodes_eval.csv`, `evaluation_qc.csv`, `evaluation_config.json` | 1 to 4 | 144 runs, ~21 h |
| 10 | Closed loop with learned controllers | same, with `--methods learned:reports/a3/learning/models/taskA_gradient_boosting.joblib,... --no-reference --out data/fatmawati_eval_learned` | plus the joblib files | same tables | 1 to 4 | low only: 20 runs; design + high: 40 runs in 14.6 h |
| 11 | Hindsight ceiling and the 70% mix | `... evaluate --hindsight --out data/fatmawati_eval_hindsight --levels low --seeds 1` and `... evaluate --moto-share 0.7 --no-reference --out data/fatmawati_eval_moto70 ...` | | `passages_hindsight.csv`; the usual tables | 1 | hours each |
| 12 | KPIs and the equilibrium | `python3 -m traffic_twin.fatmawati.kpi summary --eval data/fatmawati_eval,data/fatmawati_eval_learned,data/fatmawati_eval_hindsight --out reports/a3/kpi --levels low` and `python3 -m traffic_twin.fatmawati.kpi hand-check --out reports/a3/kpi` | the evaluation folders | `kpi_methods.csv`, `nash.csv`, `equilibrium.json`, `kpi_by_level.csv`, figures | bootstrap seed in `kpi.py` | minutes; hand-check runs SUMO |
| 13 | Documents | `python3 reports/a3/build_report.py`, `python3 reports/methodology_v3/build_methodology.py`, `RSTUDIO_PANDOC=... Rscript -e 'rmarkdown::render("reports/a3/Assignment3_Demo.Rmd")'` | the release and `reports/a3/kpi` | md, docx, html, pdf | | minutes |
| 14 | Viewer (optional) | `python3 -m traffic_twin.fatmawati.geometry3d` then `python3 -m traffic_twin.fatmawati.webapp3d --port 8770` | net, `osm/buildings_1600m.json`, `osm/mrt.json` | `viewer/geometry.json`, live route files | | |

Every parallel step uses `concurrent.futures.ProcessPoolExecutor`; keep at least two cores free. A long
run must be detached from the terminal session (`caffeinate -i nice -n 10 python3 -m ... &` under
`setsid`/`start_new_session`), because a run tied to an interactive session died with it once and lost
13 hours. `evaluate.py` writes one line per finished run to `<out>/evaluate.log`.

### What each output stamps, and what it does not

| Output | Stamped | Missing (record by hand in a `RUN.md` beside the folder) |
|---|---|---|
| `data/<release>/config.json` | generator version `traffic_twin_fatmawati_v3.0`, plan, windows, levels, calibration scales, gate radii, `network_sha256`, Python version | SUMO version (it is per episode in `simulation_qc.csv`), git commit, `pip freeze` |
| `data/<release>/SHA256SUMS.txt` | hash of every table | |
| `data/<release>/simulation_qc.csv` | `sumo_version`, `wall_s` per episode, collision and teleport counts | |
| `data/<eval>/evaluation_config.json` | methods, levels, seeds, `t_empty_s`, release points, mix, schedule, cycle, warm-up | generator version, network hash, SUMO version, git commit |
| `reports/a3/learning/` | metrics and models | library versions |

## 4. Determinism

- SUMO runs with `--seed <seed>` and `--step-length 0.5` (`sumo_world.py:90`). The `random` controller has
  its own stream, `np.random.default_rng([seed, 991])` (`sumo_world.py:139`), so it never disturbs the
  traffic draws. Trips come from `np.random.default_rng(seed)` in `demand.routes_xml()` (`demand.py:216`).
  Within one window, level and seed every controller gets the same trips.
- Seed sets: calibration fits on 11 and 12 and verifies on 11 to 14; the release uses 1 to 3 (seed 3 is the
  held-out test seed for learning); the evaluation grid uses 1 to 4 with its own demand draws, so seed 4
  appears nowhere in the release.
- Hindsight replays fork the libsumo process, because SUMO's own save/load-state is not exact in 1.27.1.
  Forked children share the parent's open trip-file offset, so SUMO must load all trips at start
  (`--route-steps 0`, passed through `FatmawatiWorld(extra_args=...)`). The hindsight base run is its own
  reference because its RNG draw order differs from the grid's.
- Sublane motorcycle overlaps and teleports are deterministic for a given seed but sensitive to any change in
  vehicle types, the network or the SUMO build: re-measure the safety counters after any such change.
- A re-downloaded OSM extract is a different network (Level 2 above).

## 5. Site transfer protocol

Run these in order. Each step names the field group of `data/sites/<site>.json` it fills and the pipeline
step from Section 3 it unblocks.

1. **Describe the site** (`site`, `junction.legs`, `junction.street_names`). Centre, radius, driving side,
   junction type, and the context that changes the study: bus lanes, plate restrictions, time-of-day
   movement bans, closures. Copy `data/sites/bundaran_hi.json` as the template: every unknown stays `null`
   with a `*_todo` note, and `tests/test_site_spec.py` enforces that.
2. **Collect the field data.** The pipeline needs four observed inputs. Without the first three nothing past
   step 1 of Section 3 is meaningful.

   | Datum | Survey | Definition to follow | Fills | Unblocks |
   |---|---|---|---|---|
   | Turning-movement counts | one weekday peak hour, 15-min bins, every entry x exit leg including U-turns, by class (motorcycle, car, bus, truck); CCTV with the operator's consent or manual counts (one counter per approach, two on the busiest) | PKJI 2023 classified count | `demand.movements_veh_h`, `demand.mix` | calibration, release |
   | Signal plan | the operator's plan (Dishub DKI ATCS) or a stopwatch survey of at least three cycles per period: stage sequence, green, amber, all-red, and which movements run in each stage; separate plans per period if they differ | `build_net.py` stage model | `signal.*` | network step 1 |
   | Queue lengths | the longest queue per cycle on each approach during the counted hour, from the stop line, counting vehicles slower than 5 km/h with gaps up to 20 m (VISSIM's queue counter, which the twin's cameras imitate, `sensors.py:37`) | VISSIM queue definition | `calibration.target_max_queue_m` | calibration |
   | Delay and degree of saturation (optional) | PKJI 2023 worksheet from the count and plan, or a published study | PKJI 2023 | `calibration.pkji_*` | validation text only |

   Fatmawati took all four from one published VISSIM study (Sumabrata et al. 2026) and the mix from a
   second (Yusuf & Isradi 2025). A site without a published study needs the survey; budget one weekday
   peak with four to six counters plus two people for timings and queues, and a second day if the plan
   changes by period.
3. **Build the raw network** at the new centre (`build_net --radius-km`, pointed at the new site once the
   code changes in Section 6 exist) and read the ids into `junction.crossings`, `approaches`, `exits`,
   `median`, `leg_source`, `leg_sink`. Pin the extract: commit the `.osm.xml`, record `osm_base`.
4. **Encode the stage model** (`signal.stage_links`, `stage_of_approach`). Check the compiled program the
   way `tests/test_fatmawati.py` does: the phase count, the greens, no link left out of every stage
   (`stage_link_sets()` raises on that).
5. **Check the sensor layout**: every queue camera reaches 300 m up its own street, gates lie outside slips
   and U-turn loops (`sensors.gate_r_m`), lane groups match the stages.
6. **Calibrate** to the queue targets with the fixed plan (`calibrate --workers 8`), verify over four seeds,
   and record which approaches are capacity-limited; write the scales, flags and relative errors into
   `calibration.fitted_scale` and the DATACARD. State the carried share of the counted volume (Fatmawati:
   22% to 35%).
7. **Pre-register** before any result: the acceptance gates (`evaluation.gates`), the seeds, the primary
   regime, the ambulance movements and the hospital (`evaluation.*`). Fatmawati fixed 10% over max-pressure
   and 20% per approach before results were seen.
8. **Run** the smoke release, the release, validation and training, the grid, the closed loop, the KPIs
   (Section 3 steps 5 to 12). Keep the same plan sizes unless the pre-registration says otherwise.
9. **Document**: a DATACARD with the sources, licences (OSM/ODbL, SUMO EPL 2.0, the study's licence), the
   survey dates, the stamps of Section 3, and `RUN.md` with git commit and `pip freeze`.

## 6. What must change in the code before a second site can run

Status 2026-11-07 (DECISIONS §13.13): items 1, 3 and 4 are done; items 2 and 5 remain. The pipeline reads
`data/sites/<slug>.json` through `traffic_twin/fatmawati/site.py`; the site is chosen by `--site <slug>` on
any CLI, else `$TWIN_SITE`, else `fatmawati`. The ranked list with file and line is in `data/sites/README.md`
("Coupling points, hardest first"). In the order the work has to happen:

1. **Done.** A site loader: `site.py` reads `data/sites/<slug>.json` and exposes what `build_net`, `demand`,
   `calibrate`, `sensors`, `evaluate`, `learn`, `kpi`, `jorr` and `mrt` used to define as literals; the module
   constants keep their names. `DATA` is `data/<slug>` and generated files take the slug prefix. A required
   field that is `null` fails the load with the field name and its `*_todo` note (`bundaran_hi.json` stops at
   `junction.crossings`). Acceptance met: with `fatmawati.json` the rebuilt network body is byte-identical
   (sha256 of the file without netconvert's header comment: `646b3acc…`; the header carries the build
   timestamp and absolute paths, so the full-file hash `d389d8d6…` is a property of the released file, not of
   any rebuild) and the smoke release runs.
2. **Open.** A general stage model: stages as sets of signal groups (tls link indices) with per-pair amber and
   all-red rules, replacing the two-crossing "approach / median" construction. A ring with N signalised nodes
   is then one controller with N stop lines. (The `% 4` cycle order is already `% len(STAGES)`.)
3. **Done.** The leg set comes from the spec everywhere the tuple `("N","E","S","W")` was written, including
   column names and the data dictionary; `fairness_ratio` and `turn_prior` are sized by the number of legs.
4. **Done.** The ambulance study per site: hospital (name, position, street), release legs, hospital side and
   the hand-check stage map (`evaluation.ev_stage`) come from the spec; the Fatmawati routes remain prose
   (`ev_routes`) because the routes are found by SUMO's router.
5. **Open.** The viewer's flyover and MRT models made optional: `jorr.ENDS` and `mrt.RAIL_Z` are read from the
   spec and may be `null`, and `--toll-vph 0` / `--no-mrt` skip them at run time, but `geometry3d` and the
   viewer tests still assume them.

To run the protocol of Section 5 at a new site, step 3 (the raw network before the ids are known) needs the
loader relaxed for `build_raw` alone; today `build_net --site <slug>` stops at the first `null`, so fill the
centre and run the raw pass by hand (`build_net.build_raw`) until that flag exists.

Remaining effort: the stage model (item 2) is the real work and depends on how the second site's ring is
signalled; item 5 is a day.

## 7. Transfer case: Bundaran HI

Verdict: the method transfers, the data does not exist yet, and the roundabout needs the stage-model
generalisation before anything runs. The skeleton spec is `data/sites/bundaran_hi.json`.

**What is known publicly (checked 2026-11-07).** Dishub DKI reported the two-way volume at the roundabout
at "nearly 98 thousand vehicles" for the week of 23 to 30 June 2022 (ANTARA), a figure that is not a
movement count and whose period is not stated. Since 19 July 2022 the Sudirman to Imam Bonjol movement is
closed every day from 16:00 to 21:00, TransJakarta and VVIP vehicles excepted (detik, VIVA, Kompas TV).
No journal article, thesis or Dishub document with per-movement counts, a stage plan or queue lengths
for Bundaran HI was found. A citywide UGM thesis on ATCS confirms Central Jakarta's signals are
ATCS-integrated, which is where the plan would come from.

**What transfers as is.** The OSM download and netconvert pass (with the Jakarta typemap), the sensor
engine and its queue definition, the trip generator and the daily profile, the calibration loop, the
dataset writer and schema, the learning validators and models, the KPI statistics and the equilibrium
rule, the TomTom overlay.

**What is blocked on a survey.** Movements and mix (demand), the stage plan (network), queue targets
(calibration). The pm window (17:00) falls inside the movement ban, so the pm plan and movement set differ
from the am ones: either count both periods or drop the pm window.

**What breaks in the code.** The ring: Fatmawati's stage model knows two crossings and two median links;
Bundaran HI has one signalised node per approach on a ring (to be confirmed on site) and the ring segments
between them carry the conflicts. The leg set: four legs are likely (Thamrin north, Thamrin/Sudirman south,
Imam Bonjol, Kebon Kacang Raya), but the ring may have more stop lines than legs. The busway: TransJakarta
Corridor 1 runs in the median with its own signal heads, which Fatmawati never modelled. The viewer's
flyover and the elevated MRT do not apply (the MRT station is underground).

**What must be re-chosen.** The hospital and the ambulance movements: RSUP Fatmawati sits 326 m from
its junction; at Bundaran HI the nearest candidates (RSPAD Gatot Soebroto to the north-east; RS Bunda
Menteng and RS Abdi Waluyo east via Imam Bonjol) are 1.5 to 2.5 km away, so the "passage" ends at the
exit gate, not at the hospital, and the release legs follow from the chosen one.

**Effort.** Survey: one weekday with four to six counters and two timing/queue observers, repeated for
the evening programme; refactor: Section 6; compute at the release plan sizes: calibration hours, release
about 2 h, grid about 21 h, closed loop and stress about 15 h, on eight workers.

## 8. Known issues found while writing this runbook (not fixed)

- Fixed 2026-11-07 (DECISIONS §13.13): `learn.py` `_sat_flow()` imported `sensor_layout` from `.sensors`
  instead of `sumo_world`, so the label validation always fell back to Fatmawati's saturation flows. It now
  reads the network's values and falls back to the site file's `sat_flow_pcu_h` only without SUMO.
- `evaluation_config.json` carries no generator version, network hash, SUMO version or git commit
  (`evaluate.py:341-343`); the release `config.json` does.
- `data/fatmawati/osm/buildings_1600m.json` has no downloader; the Overpass query is not recorded.
- `docs/DECISIONS.md` §13.2 still states motorcycle gaps of 0.2 m / 0.3 m; the shipped values are
  0.5 m / 0.5 m (§13.11 has the correction).
- `README.md` omits the `learn` and `kpi` steps (Section 3 here has them).

## Sources for the Bundaran HI facts

- ANTARA (2022), "Volume kendaraan di Bundaran HI naik 9 persen per minggu",
  https://www.antaranews.com/berita/2978073/volume-kendaraan-di-bundaran-hi-naik-9-persen-per-minggu
- detik (19 July 2022), "Rekayasa Lalin di Bundaran HI Permanen, Diterapkan Setiap Sore",
  https://news.detik.com/berita/d-6186783/rekayasa-lalin-di-bundaran-hi-permanen-diterapkan-setiap-sore
- VIVA (2022), "Traffic Engineering Policy at the HI Roundabout Becomes Permanent",
  https://www.viva.co.id/english/1499271-traffic-engineering-policy-at-the-hi-roundabout-becomes-permanent
- Kompas TV (2022), "Mulai hari ini Dishub DKI terapkan rekayasa lalu lintas di Bundaran HI",
  https://www.kompas.tv/article/305522/mulai-hari-ini-dishub-dki-terapkan-rekayasa-lalu-lintas-di-bundaran-hi
- UGM thesis record, "Area Traffic Control System (ATCS) for Supporting Urban Traffic Management in DKI
  Jakarta", https://libmast.utm.my/Record/oai:generic.eprints.org:281686/Description
