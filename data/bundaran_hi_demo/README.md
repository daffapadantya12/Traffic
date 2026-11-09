# Bundaran HI demo (uncalibrated)

A SUMO demonstration only. It is not part of the traffic_twin study and its numbers are not evidence about the real
junction. The study's own spec, `data/sites/bundaran_hi.json`, is untouched and still refuses to load.

Open `replay.html` to watch the last five minutes of the run. Rebuild everything (about one minute):

    python -I scripts/bundaran_hi_demo.py --tag s04 --replay

For the live 3D view in the style of the Fatmawati twin, start the server and open http://127.0.0.1:8792/ :

    python -I scripts/bundaran_hi_3d.py

It runs the network, demand and seed of the 40 % run below (at 1,800 s the live run matched the batch run: 1,156 trips
finished, 221 vehicles running, 1 teleport). The selector "ring signals off" switches the three ring-entry signals to
SUMO's `off` program while the simulation runs. The clock starts at 07:00 for display only; demand is constant.

**From OpenStreetMap (real):** road geometry and lane counts, extract of 2026-11-09, 1.5 km around the roundabout; for the
3D view also the building footprints and heights within 1.6 km, the pool outline and the monument's position. The monument
itself is drawn schematically.

**Assumed:** the volumes (98,000 veh/day from ANTARA 2022, read as a daily total, x 7 % peak share = 6,860 veh/h, then
split by entry lanes), turning shares, the mix (70 % motorcycles), which ring lanes may exit (netconvert allows only the
outermost one), and the signals (OSM tags none on the ring, so SUMO's automatic program runs on the three entries).

**Left out:** the TransJakarta busway and the 16:00-21:00 movement ban.

**OSM versus the spec's provisional layout:** the ring has three entries (Thamrin, Imam Bonjol, Sudirman) and four exits;
the west leg (Saleh Ishak) is exit-only in OSM.

## Runs (30 min, last 15 measured)

| Run | Veh/h | Teleports (jam) | Collisions | Trips finished | Mean time loss (s) |
|---|---|---|---|---|---|
| signals, 20 % | 1,373 | 0 (0) | 4 | 230 | 41.0 |
| signals, 30 % | 2,058 | 7 (6) | 13 | 343 | 93.6 |
| **signals, 40 % (replay)** | 2,743 | 1 (0) | 9 | 466 | 49.1 |
| signals, 50 % | 3,428 | 28 (27) | 30 | 225 | 63.8 |
| signals, 70 % | 4,800 | 15 (10) | 114 | 447 | 109.5 |
| signals, 100 % | 6,862 | 36 (28) | 74 | 151 | 262.9 |
| no signals, 40 % | 2,743 | 0 (0) | 9 | 484 | 17.8 |
| no signals, 50 % | 3,428 | 18 (16) | 52 | 472 | 43.0 |
| no signals, 70 % | 4,800 | 9 (5) | 95 | 519 | 189.0 |
| no signals, 100 % | 6,862 | 20 (18) | 25 | 202 | 147.9 |

Seed 42 throughout, except that the 40 % signal run was repeated with seeds 1 to 3: no jam teleports, 460 to 464 trips
finished, 43 to 46 s mean time loss. Trips still under way at the end of the run are not in the means, so the jammed
runs look better than they are. From 50 % up the Thamrin approach queues back about 900 m.
