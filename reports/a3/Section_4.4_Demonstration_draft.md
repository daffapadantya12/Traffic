## 4.4 Demonstration in R

The demonstration uses the project's own dataset, downloadable from [download link to be added by the group], and the R Markdown file `Assignment3_Demonstration.Rmd` submitted with this report. **The data is generated, but calibrated on real counts:** every row is a virtual-sensor reading from the SUMO twin, whose roads, signal plan and turning movements come from OpenStreetMap and Sumabrata et al. (2026), and whose demand was tuned until simulated queues matched that study's published queues.

**Process.** The file (1) compares the twin with the real inputs, (2) describes the nine tables against the data dictionary, (3) checks data quality, (4) summarises vehicle mix, queues, delay and ambulance trips, (5) tests whether the data can support learning and paired evaluation, and (6) fits a first model for Task A, linear discriminant analysis (LDA), trained on seeds 1–2 and tested on seed 3.

| Check | Result |
|---|---|
| Realism | 80.2% of entering vehicles are motorcycles (published 77–82%); the twin carries 21–32% of counted volumes in the 17:00 design hour |
| Contents | 108 episodes in a full factorial design; 77,760 five-second rows with 75 features and 5 labels; 732 ambulance trips |
| Quality | No documented column missing; gaps are structural; 0.19 lane collisions per 1,000 vehicle-km |
| Label | Teacher's stage balanced from 11.5% (S2) to 37.1% (S3); an ambulance is present in 13.6% of rows |
| Signal in the features | The label equals the max-pressure stage in 86% of rows without an ambulance and 71% with one |
| Paired t-tests vs fixed plan | At low demand all adaptive controllers cut stopped delay by 24–27 s per vehicle (p ≤ 0.0001); at design and high demand no difference is significant (p ≥ 0.31) |
| Task A with LDA | Macro-F1 0.67 against 0.28 for the best naive baseline; 76% accurate on ambulance rows, above the max-pressure rule's 70% |

*Table 3. Results of the R demonstration.*

**What it shows.** The data is usable as published. It reproduces the junction's motorcycle share and queues, not its volumes, so it compares controllers rather than predicting real delays. The label is balanced and tied to the features, and even a linear model learns part of the ambulance priority, so a learned controller is feasible; its weaker overall score motivates the non-linear models of Section 4.3. Paired tests compare controllers fairly, but three seeds per cell cannot separate them at high demand, which is why Section 4.3 uses a dedicated ambulance grid. Two limits appear: no ambulance approaches from the south, and the 308 detected on the east take a slip road without a signal, so only 406 of 732 trips test signal priority.
