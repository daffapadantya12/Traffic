# Site specifications

One JSON file per junction, holding every site fact the Fatmawati pipeline (`traffic_twin/fatmawati/`)
needs. Today the pipeline reads none of these files: the values live in Python module constants, and
`tests/test_site_spec.py` asserts that `fatmawati.json` equals those constants, so the spec cannot drift from
the code. The spec exists so that a second site can be described completely before any code is changed, and
so that the transfer protocol in `docs/REPRODUCIBILITY.md` has a checklist to point at.

| File | What it is |
|---|---|
| `fatmawati.json` | the released study, verified against the code and the released `config.json` files |
| `bundaran_hi.json` | a skeleton for Bundaran Hotel Indonesia: public facts filled in, every unknown `null` with a `*_todo` note |

Conventions: `"status"` says how much of a file is verified. A `null` value always has a sibling
`<field>_todo` string saying how to obtain it. Stage links are written symbolically (`"approach:W"`,
`"exit:N"`, `"median:south"`, `"*"`) and resolved through `approaches` / `exits` / `median`, so the signal
plan reads without raw OpenStreetMap ids. Movements are keyed `"<entry>><exit>"`.

## Field reference

"Where used" names the constant that holds the value today. R = required for a run, O = optional
(validation text, data card), A = an assumption carried from Fatmawati that a new site may keep.

### `site`

| Field | Meaning | Unit | R/O/A | Where used |
|---|---|---|---|---|
| `centre_lat`, `centre_lon` | box centre at the signal nodes | deg | R | `build_net.py:42` `JUNCTION_LAT/LON`; imported by `sensors.py`, `sumo_world.py`, `demand.py` (cordon), `geometry3d.py` (viewer origin) |
| `radius_km` | half-width of the OSM box | km | R | `build_net.py:73` `bbox()`, CLI `--radius-km` (`:229`); repeated in `mrt.py:26` |
| `osm_bbox_s_w_n_e` | the box as downloaded | deg | O | `data/fatmawati/osm/bbox.txt`, read by `build_methodology.py` |
| `osm_extract_timestamp`, `osm_extract_file` | pin of the Overpass extract | | R | `build_net.py:79` `download_osm()` caches the file; a re-download changes edge ids |
| `osm_road_types` | highway classes kept | | A | `build_net.py:45` `ROAD_TYPES` |
| `driving_side` | | | R | `--lefthand` in every netconvert call (`build_net.py:109,159,223`) |
| `typemap` | speeds and class bans | | A | `data/fatmawati/patches/jakarta.typ.xml` (the only handwritten patch; the other five are written by `build_net`) |
| `network_file`, `network_sha256` | the built net and its hash | | R | `generate.py:438` stamps `network_sha256` into `config.json` |
| `junction_type` | prose | | O | documents only |

### `junction`

| Field | Meaning | R/O/A | Where used |
|---|---|---|---|
| `tls_id`, `program_id` | SUMO traffic-light id and program | R | `build_net.py:44`; `sumo_world.py:92` `setProgram(TLS_ID, "dishub")`; `sensors.py`, `evaluate.py`, `geometry3d.py` |
| `crossings` | signal nodes merged into one controller | R | `build_net.py:43` `CROSSINGS`, written to `fatmawati_tls.nod.xml` |
| `legs`, `street_names` | the arms | R | `sensors.py:28` `LEGS`, `sumo_world.py:31` `ARMS`, `kpi.py:37`, `learn.py:29`; column names in every table |
| `approaches`, `exits`, `median` | raw edge ids at the stop lines, leaving the junction, between the crossings | R | `build_net.py:48-50`; sensors walk from them (`sensors.py:100` `build_layout`) |
| `leg_source`, `leg_sink` | where movement trips start and end | R | `demand.py:46-47` |
| `deleted_connections` | banned or uncounted turns | R | `build_net.py:155-157` connection deletes |
| `rhk_depth_m` | motorcycle box depth | A | `build_net.py:65`, CLI `--rhk-depth-m` |

### `signal`

| Field | Meaning | R/O/A | Where used |
|---|---|---|---|
| `stages`, `green_s`, `yellow_s`, `allred_s`, `cycle_s` | the city's fixed plan | R | `build_net.py:51-53`; `evaluate.py:50` `CYCLE_S` (release slices) |
| `stage_links` | links green in each stage | R | `build_net.py:55-60`, compiled by `stage_program()` (`:184`) into `fatmawati_program.tll.xml` |
| `stage_approaches`, `stage_of_approach` | which stage serves an approach | R | `build_net.py:61-62`; `sumo_world.py` (pressures), `learn.py:31`, `evaluate.py` |
| `stage_label` | names for the dictionary | O | `build_net.py:63`, `generate.py` dictionary text |
| `plant.*` | adaptive plant limits | A | `sumo_world.py:71` (`min_green`, `max_green`, `decision_s`), `:36` `STARVE_S` |

### `demand`

| Field | Meaning | Unit | R/O/A | Where used |
|---|---|---|---|---|
| `movements_veh_h` | counted peak-hour turning movements | veh/h | R | `demand.py:43-45` `MOVEMENTS`; `calibrate.py:34` sums them per leg |
| `mix` | class shares | | R | `demand.py:35` `MIX`; `vtypes.add.xml` |
| `mix_sensitivity` | evaluation-only mix | | A | `evaluate.py` `--moto-share` |
| `level_factor` | low / design / high multipliers | | A | `demand.py:49` |
| `base_inflow_vph_at_design`, `cordon_m` | background trips | | A | `demand.py:50-51` |
| `daily_profile` | hourly multipliers | | A | `traffic_twin/config.py:21` |
| `windows_h` | release windows | h | A | `generate.py` `WINDOWS` |
| `ev_per_hour`, `ev_legs` | ambulance rate and origins | | R | `demand.py:52-53` |

### `calibration`

| Field | Meaning | Unit | R/O/A | Where used |
|---|---|---|---|---|
| `target_max_queue_m` | observed longest queue per approach | m | R | `calibrate.py:30` `TARGET_MAX_QUEUE_M` |
| `queue_definition` | how the queue is measured | | R | `sensors.py:37` `QUEUE_MS, QUEUE_GAP_M` |
| `pkji_delay_s`, `pkji_ds`, `vissim_junction_delay_s` | published validation numbers | | O | `calibrate.py:31-33`, DATACARD text |
| `grid`, `fit_seeds`, `verify_seeds`, `warm_s`, `main_s` | the search | | A | `calibrate.py:35-37`, `verify_and_save()` (`:160`) |
| `fitted_scale`, `capacity_limited`, `rel_error` | the result | | derived | `data/fatmawati/calibration.json`, read by `demand.load_calibration()` (`demand.py:56`) |

### `sensors`

| Field | Meaning | Unit | R/O/A | Where used |
|---|---|---|---|---|
| `gate_r_m` | entry/exit gate distance per leg | m | R | `sensors.py:29` `GATE_R` |
| `camera_m`, `exit_camera_m`, `loop_setback_m` | zone lengths | m | A | `sensors.py:30-32` |
| `study_m` | safety-count radius | m | A | `sumo_world.py:37` `STUDY_M` |
| `pcu` | passenger-car units | | A | `sensors.py:35` (PKJI 2023) |
| `sat_flow_pcu_h` | per-approach saturation flow | pcu/h | derived | `sensors.py:154` (600 x width); `learn.py:33` `SAT_FLOW_DEFAULT` |

### `evaluation`

| Field | Meaning | R/O/A | Where used |
|---|---|---|---|
| `hospital_name`, `hospital_lonlat`, `hospital_street_name`, `hospital_leg` | the ambulance destination | R | `demand.py:40` `HOSPITAL_LONLAT`, `demand.py:138` street-name filter, `evaluate.py:46` `HOSPITAL_LEG` |
| `ev_moves`, `ev_routes` | release legs and their paths | R | `evaluate.py:45` `EV_MOVES`; `kpi.py:562` hand-check stages |
| `release_min_m`, `release_points`, `t_empty_s` | release geometry and empty-road times | derived | `evaluate.py:48`; written to `evaluation_config.json` (`evaluate.py:341`) |
| `slices`, `release_every_s`, `warm_s`, `tail_s`, `extra_s` | the schedule | A | `evaluate.py:47-53` |
| `methods` | controllers compared | A | `evaluate.py:54` `METHODS` |
| `gates` | pre-registered acceptance margins | R | `kpi.py:38` `GEN_GATE, APPROACH_GATE`; safety test in `kpi.gates()` |
| `grid_seeds`, `release_seeds` | | A | `evaluate.py` `--seeds`; `generate.py:45` plan `release` |

### `viewer_only`

Nothing here enters the dataset, the calibration or the evaluation: JORR through-traffic (`jorr.py:22-23`),
the MRT viaduct (`mrt.py:27` `RAIL_Z`), building footprints (`geometry3d.py` reads
`osm/buildings_1600m.json`; no code downloads it) and the TomTom overlay (`tomtom.py`, display only).

## Coupling points, hardest first

What has to change in the code before a site other than Fatmawati can run. Ranked by effort.

1. **The signal and stage model.** `build_net.py:43-64` and `:163-224` define stages as (from-edge, to-edge)
   pairs over exactly two crossings and two median links; the amber/all-red builder treats only "approach"
   and "median" links; the cycle order is `% 4` (`build_net.py:200`, `sumo_world.py:294,327`);
   `STAGE_OF_APPROACH` assumes one stage serves each approach (`sumo_world.py`, `learn.py:31`,
   `evaluate.py`). A signalised ring needs stages defined per signal group with ring clearance. An
   unsignalised junction cannot run at all: `sumo_world.py:92` sets the program unconditionally and every
   controller is a stage requester (`sumo_world.py:259` `request()`).
2. **The ambulance study.** Hospital location and street name (`demand.py:40,138`), `EV_LEGS` /
   `EV_MOVES` / `HOSPITAL_LEG`, the Fatmawati-specific routes in the `evaluate.py` docstring, the hand-check
   stage map (`kpi.py:562`), release slices tied to the fixed cycle. This is a study redesign per site.
3. **Four legs named N/E/S/W.** `sensors.py:28`, `sumo_world.py:31`, `kpi.py:37`, `learn.py:29`, the 4 x 4
   `turn_prior` (`sensors.py:219`), `fairness_ratio` needing four values (`generate.py:145`), column names
   in every table and the data dictionary, and the tests.
4. **Raw OSM ids.** `CROSSINGS`, `APPROACHES`, `EXITS`, `MEDIAN`, `LEG_SOURCE`, `LEG_SINK`, `jorr.ENDS`,
   edge ids in `tests/test_fatmawati.py`. Found by hand after `build_raw`; they change with every
   re-download. Could be derived from centre, street name and bearing, or kept in the site JSON.
5. **Observed data and targets.** `MOVEMENTS`, `MIX`, `TARGET_MAX_QUEUE_M`, PKJI values, citation strings
   in `calibrate.py:141-143`, `generate.py` and the DATACARD. Dictionary-driven already; the hard part is
   the survey.
6. **Viewer structures.** Flyover and MRT viaduct (`geometry3d.py`, `jorr.py`, `mrt.py`), labels and camera
   presets in `static/index.html`, tests asserting them.
7. **Sensor geometry.** `GATE_R` per leg and the 300 m same-street-name chain walk (`sensors.py:44-53`).
8. **Centre, radius, typemap.** `JUNCTION_LAT/LON`, `--radius-km`, the duplicate `mrt.RADIUS_KM`.
9. **Paths and names.** `DATA = data/fatmawati` (`build_net.py:40`, `demand.py:32`), `fatmawati_*` file
   names, default output folders, `TLS_ID`, program id, class name `FatmawatiWorld`, log strings.
