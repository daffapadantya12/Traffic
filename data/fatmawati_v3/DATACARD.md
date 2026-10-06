# Fatmawati twin sensor dataset (traffic_twin_fatmawati_v3.0, plan 'release')

Virtual-sensor readings from a SUMO digital twin of Perempatan Fatmawati (Cilandak, South Jakarta): what a roadside installation of loops, counting gates and queue cameras would record. The tables hold sensor readings only; no vehicle trajectories, model states or driver parameters. Vehicle identities were used inside the simulator to match entries with exits (as video or ANPR re-identification does) and are not written out.

## Calibration

Demand is the 13 movements counted at the junction in a weekday peak hour (Sumabrata, Haqueline & Mauramdha 2026), scaled per approach so that, under the Dishub fixed plan, the peak-hour maximum queue approaches that study's VISSIM maximum queue. Scales N 0.242, E 0.239, S 0.220, W 0.360. Check over 4 seeds, maximum queue: N 306 m (SD 2) vs 290 m, +6% of target; E 194 m (SD 59) vs 239 m, -19% of target; S 127 m (SD 4) vs 167 m, -24% of target; W 73 m (SD 16) vs 156 m, capacity-limited (its demand backs up upstream before the camera queue reaches the target). Near capacity an approach's hourly maximum queue tips from a short queue to spill-back past the 300 m camera and varies strongly between seeds, so a closer fit was not possible. Stopped delay N 191 s (PKJI control delay 80 s), E 98 s (PKJI control delay 79 s), S 86 s (PKJI control delay 106 s), W 42 s (PKJI control delay 78 s). The twin carries N 22%, E 24%, S 22%, W 35% of the observed volumes at these queues: SUMO's motorcycles discharge far less densely than Jakarta's, so the twin reproduces the congestion, not the absolute counts. Vehicle mix 80% motorcycles (Yusuf & Isradi 2025).

Sensors are exact (no detection errors).

## Sensors

- Stop-line loops: one per approach lane, 1 m before the stop line (4 approaches).
- Entry and exit gates: a loop on every lane of each leg, N 150 m, E 150 m, S 150 m, W 185 m from the junction centre, outside the free-left slips and the median U-turn loops.
- Queue cameras: 300 m upstream of each stop line; exit cameras: first 150 m of each exit.
- Priority-request detector: an ambulance inside an approach camera zone.

## Episodes

108 episodes: windows ['am_peak', 'midday', 'pm_peak'], demand levels ['low', 'design', 'high'], controllers ['fixed', 'longest_queue', 'max_pressure', 'priority_mp'], seeds [1, 2, 3]; 15 min warm-up (not recorded) then 60 min recorded.

## Tables

- sensors.csv: 39 rows
- episodes.csv: 108 rows
- loops_1min.csv: 200880 rows
- cameras_1min.csv: 51840 rows
- ticks_5s.csv: 77760 rows
- signal_log.csv: 77202 rows
- turning_counts_15min.csv: 15238 rows
- ambulance_trips.csv: 732 rows
- simulation_qc.csv: 108 rows

## Simulation quality (simulation_qc.csv)

- whole 2.5 km network: lane collisions 373, junction overlaps 23976, teleports 2019; within 300 m of the junction (the sensors' area): lane collisions 115, junction overlaps 19292; unmatched exits 96. Lane collisions are rare overlaps between motorcycles at walking pace, a limitation of SUMO's sublane model with filtering motorcycles, and are reported rather than hidden.

## Licence and attribution

The road network and sensor positions derive from OpenStreetMap: © OpenStreetMap contributors, available under the Open Database License (ODbL 1.0); this dataset is shared under the ODbL. Simulation: Eclipse SUMO (EPL 2.0). No TomTom or Google data were stored or used to make it.
