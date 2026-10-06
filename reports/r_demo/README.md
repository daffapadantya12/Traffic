# Assignment 3 demonstration (R Markdown)

`Assignment3_demo.Rmd` is the ITI5145 Assignment 3 demonstration: it loads
the `data/traffic_v1` dataset, prints its release front matter
(generator version, configuration hash, row counts read from `config.json`
and `DATACARD.md`), verifies the `SHA256SUMS.txt` checksums, documents the
tables and the `signal_ticks` feature namespaces, checks keys, column
documentation and missingness, plots the basic characteristics (delay by
demand level and controller, delay by entry node as a fairness view, queues
over time, label balance, hidden aggressiveness vs delay, ambulance vs car
delay), fits two depth-5 `rpart` trees that predict the teacher label
`y_mp_action` on held-out episodes (raw features, then with the max-pressure
quantities computed from the row) and describes each tree from the fitted
object, pulls the headline metrics of the Python ML pipeline from
`reports/ml/results.json` when it exists, and cross-checks the release
against the methodology. `Assignment3_demo.html` is the rendered result.

Every number in the text is computed at render time (inline R); the only
hard assertions (`stopifnot`) are structural: unique primary keys and every
column documented. Row counts are *compared* with the data card and shown as
a pass/fail table, so the same `.Rmd` renders on any release (it was also
tested on the 10-episode `data/dev10`).

## Render

Requirements: R >= 4.x with `rmarkdown`, `knitr`, `dplyr`, `tidyr`, `ggplot2`,
`scales`, `rpart`, `jsonlite` (all on CRAN; `digest` is optional, for the
checksum table) and pandoc. The dataset must sit at `../../data/traffic_v1`
relative to this folder (i.e. `<project>/data/traffic_v1`); another folder can
be passed as a parameter without editing the file:

```bash
Rscript -e 'rmarkdown::render("reports/r_demo/Assignment3_demo.Rmd")'
# or, for another release:
Rscript -e 'rmarkdown::render("reports/r_demo/Assignment3_demo.Rmd", params = list(data_dir = "/path/to/data/dev10"))'
```

If pandoc is not on the PATH, point R at the one bundled with RStudio
(on this Mac):

```bash
RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64 \
  Rscript -e 'rmarkdown::render("reports/r_demo/Assignment3_demo.Rmd")'
```

Rendering the 120-episode release takes about 40 s on an Apple-silicon Mac
(reading the 102 MB `signal_ticks.csv`, hashing the seven released files and
fitting two trees on 168k rows). Or open the `.Rmd` in RStudio and press Knit.

## Before submitting, replace

| placeholder | where | with |
|---|---|---|
| `[STUDENT_NAME] ([STUDENT_ID])` | YAML `author` | your name and student id |
| `[DATASET_DOWNLOAD_LINK]` | section 1 (front-matter table and "Download link") | a public link to the `data/traffic_v1` folder (zip of the five CSVs + `config.json`, `DATACARD.md`, `data_dictionary.csv`, `SHA256SUMS.txt`), e.g. a Google Drive / OneDrive / Zenodo link |

The licence line in the front-matter table ("CC BY 4.0 for the data, MIT for
the generator code, proposed") follows the methodology report; confirm it
before publishing the link.

Upload both the `.Rmd` and the rendered `.html` with the report, as the unit
brief asks for the R Markdown file.

## Key numbers printed by the demo (traffic_v1, generator traffic_twin_v1.2, config_hash ecd36a4b3cad)

- 120 episodes, 360,443 vehicles, 300,598 signal_ticks rows, 1,996,932 junction_1hz rows; 253,275 main-period rows. All seven SHA-256 checksums match.
- `y_mp_action` balance (main rows): N 0.274, E 0.225, S 0.278, W 0.223; 61.0% of rows ask for a switch.
- `y_q_total_t30` is missing only on tail and cool-down rows plus 14 rows each of episodes 74 and 93, which ended early by gridlock (episode 20 also gridlocked, but during its cool-down, so no main-period row is affected).
- Plotting rule: 61 of 329,727 main-period cars (0.02%) have a delay below 1 s and are dropped from the log-scale figures only; 36 of them are negative.
- Depth-5 tree, by-episode split (train 167,542 rows / test 31,390 rows): accuracy 0.466 on raw features (root split `ev_present_W < 0.5`), 0.815 with the per-approach pressures computed from the row (root split `best_dir < 0.5`; 0.752 on the stress episodes). Rows without an ambulance in sight: 79.8% of the main rows.
- Fairness: worst/best origin ratio of mean car delay is 1.20 (low), 1.10 (design), 1.07 (high), 1.32 (stress).
- Saturation flow 1631.35 veh/h/lane, realised capacity 549 veh/h per approach (formula 611.76), invariant violations 0, seeds = 100000 + episode_id; 11 of 11 methodology cross-checks pass.
- The ML results table flags that `reports/ml/results.json` was computed on an earlier release (254,414 main rows vs 253,275 here) until the Python pipeline is re-run.

The canonical copy of the demo is this folder; `reports/rmarkdown/Traffic_Demo.Rmd`
is an older duplicate that is scheduled for removal.
