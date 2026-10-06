---
title: "Green for the Ambulance at Perempatan Fatmawati: A Digital Twin for Learning Emergency-Vehicle Signal Priority in Motorcycle-Dominated Traffic"
subtitle: "ITI5145 Foundations of Data Science, Term 4 2026 — Assignment 3: Report"
author: "Group members: [Name, Student ID]; [Name, Student ID]; [Name, Student ID]"
date: "Business scenario: Healthcare (emergency medical services)"
lang: en-AU
---

**Topic.** The topic is unchanged from Assignment 1. Sections 1 to 3 revise the proposal; Sections 4 and 5 are new.

<div class="pagebreak"></div>

**Feedback on Assignment 1 and how we used it.** [To be completed by the group once Assignment 1 feedback is returned: the lecturer's feedback in one or two sentences, then how this report responds, at most 150 words in total.]

<div class="pagebreak"></div>

# 1. Introduction

## 1.1 Problem and context

Perempatan Fatmawati, under South Jakarta's JORR toll viaduct, lies 326 m north of RSUP Fatmawati, a national referral hospital with a 24-hour emergency department (Direktorat Jenderal Kesehatan Lanjutan, n.d.; Figure 1). About four in five vehicles here are motorcycles (Yusuf & Isradi, 2025); at red they fill the motorcycle box and every gap between cars. Indonesian law gives ambulances right of way and exempts them from signals (Undang-Undang Nomor 22 Tahun 2009, arts. 134–135), yet a boxed-in queue cannot move aside: only a timely green for the ambulance's approach clears the path. The junction already runs at the worst level of service, averaging 85 s of delay per vehicle (Sumabrata et al., 2026).

![Figure 1. The study area in a 5.4 km square (a) and the junction as simulated (b).](figures/F1_study_area.png){width=100%}

**Problem statement.** No data-driven ambulance-priority rule exists for motorcycle-dominated junctions, because ambulance passages cannot be recorded safely on live roads. We built a digital twin of the junction, generated open sensor data on ambulance passages, and learned priority controllers that cut ambulance delay while keeping every direction's delay fair and bounded.

## 1.2 Why it matters

Survival from cardiac arrest falls by about two percentage points for every minute advanced life support is delayed (Larsen et al., 1993). Among Jakarta emergency-department patients, the median ambulance response was 24 minutes and only 9% arrived by ambulance (Brice et al., 2022).

## 1.3 Goals

1. **Twin.** The junction and surrounding main roads in SUMO from OpenStreetMap, with motorcycle boxes, the city's four-stage signal plan, and drivers yielding to ambulances only where there is room, calibrated to published queue lengths.
2. **Dataset.** {{DATA_HOURS}} simulated hours of virtual sensor readings as CSV tables, with a data dictionary and data card.
3. **Controllers.** Five model families trained to imitate a priority-aware rule that requests the ambulance's stage just in time to clear the queue ahead, against stillness and random baselines.
4. **Evaluation.** Every method on one grid of ambulance cases, scored against an empty-road reference, flattened over directions and judged by a Nash bargaining equilibrium.

# 2. Related Work

Fixed-time plans follow Webster (1958), adapted for Indonesia's mixed traffic (Direktorat Jenderal Bina Marga, 2023). Max-pressure control (Varaiya, 2013) uses only local queues. Deep reinforcement learning dominates research but is hard to audit (Wei et al., 2019); on tabular data, tree ensembles match deep models and stay interpretable (Grinsztajn et al., 2022).

Chen et al. (2024) time preemption around queue spillback; Roy et al. (2024) tune it in a Vissim-based digital twin; EMVLight (Su et al., 2023) treats yielding as a speed rule. Savithramma et al. (2022) predict queue-clearing green time with gradient-boosted trees, but they, like a Surabaya SUMO study (Izzuddin et al., 2025), treat two-wheelers only as counts. A transport digital twin joins a model, a live data link and synchronisation (Kušić et al., 2023); like other SUMO twins (Schaffland et al., 2024), ours defers synchronisation, because TomTom's terms forbid storage and safety-critical use (TomTom, n.d.).

**Gap and novelty.** A real Jakarta junction beside a national hospital, mostly motorcycles, with boxes and the city's own signal plan; yielding limited to real gaps; an open dataset of ambulance passages with verified labels, which we have not found elsewhere; and interpretable control behind a safety shield, chosen by an equilibrium that neglects no direction.

# 3. Business Model

Three public bodies share the problem: RSUP Fatmawati's emergency department, Jakarta's 119 ambulance service (Pusat Krisis dan Kegawatdaruratan Kesehatan Daerah, n.d.) and the Transport Agency's signal-control unit. The value is seconds saved for patients and crews at a bounded, fair cost to everyone else, and evidence before any capital spending, from a pipeline that rebuilds any junction from open maps.

| Stakeholder | Need | How the project meets it |
|---|---|---|
| Patients and ambulance crews | Fewer stops, predictable passage | Flattened ambulance delay is the headline measure |
| RSUP Fatmawati emergency department | Earlier, predictable arrivals | Passage times for every direction |
| Signal operators | No manual intervention, no gridlock | Controllers only request stages; the plant enforces safety |
| Motorcyclists and other road users | Fair, bounded delay | Delay per approach against today; worst approach shown |
| City government and researchers | Evidence and reusable data | Paired-seed tests; open tables and data card |

# 4. Characterising and Analysing Data

## 4.1 Data sources and the four V's

- **Sources.** OpenStreetMap roads (ODbL); Indonesian design rules; published counts, the city's signal plan and a calibrated simulation of the junction for calibration (Sumabrata et al., 2026); the twin's own sensor tables, which are the dataset. Ambulance passages cannot be recorded safely on live roads, so synthetic data from a validated generator is the accepted substitute (Jordon et al., 2022).
- **Volume.** {{DATA_EPISODES}} episodes, {{TICK_ROWS}} learning rows and {{AMB_TRIPS}} ambulance trips in under a gigabyte of CSV, plus {{GRID_PASSAGES}} evaluation passages.
- **Velocity.** The simulation steps every 0.5 s; a controller decides every 5 s, which every model serves in milliseconds.
- **Variety.** Nine tables at seven grains, from sensor and minute to tick, signal phase, 15 minutes, ambulance trip and episode, with a data dictionary that labels each column's role.
- **Veracity.** Labels are recomputed from the logged columns before training ({{VALIDATION_SENTENCE}}); calibration was fitted to the published queue lengths ({{CALIBRATION_SENTENCE}}); results describe the twin, not the street.

## 4.2 Platforms, software and tools

| Role | Now | If the project expands |
|---|---|---|
| Simulation | SUMO 1.27 through its Python library (Alvarez Lopez et al., 2018) | many junctions on a cloud batch service |
| Modelling | Python, pandas and scikit-learn (Pedregosa et al., 2011) | gradient-boosting libraries on GPUs |
| Demonstration | R with dplyr, ggplot2 and rpart (R Core Team, 2026) | dashboards on the same tables |
| Storage | CSV, configuration, checksums | Parquet in object storage |

## 4.3 Analysis techniques and statistical methods

**Supervised learning.** Each 5 s tick is a row of sensor features, labelled with the stage a priority-aware teacher requests: the ambulance's stage once its arrival time falls within the time to clear the queue ahead, otherwise the max-pressure stage. Five families compete on splits that keep whole episodes and seeds apart: logistic regression, a decision tree, a random forest (Breiman, 2001), gradient boosting and a small neural network, each minimising a log-loss in which every stage counts equally. Each must beat three baselines: keeping the current stage, the most common stage and a random stage. Two further tasks forecast each queue 30 s ahead and the ambulance's arrival time.

{{MODEL_TABLE}}

{{MODEL_SENTENCE}}

**Evaluation.** Every method meets the same grid of ambulance cases: three hospital movements, eight arrival moments across the 231 s signal cycle, three demand levels and {{GRID_SEEDS}} seeds. Ambulance delay is measured against an empty-road run and averaged with equal weight per direction. Each approach's delay is indexed to the city's fixed plan. Methods must pass a safety gate, keep general delay within 10% of max-pressure and leave no approach more than 20% worse than today. Among those on the Pareto front, a Nash bargaining rule (Nash, 1950) chooses the method that maximises the product of every player's gain.

Each run holds its demand level for 4.4 hours. At the design and high levels the north approach, calibrated at capacity, never clears: its queue grows past the ambulances' release point, and up to {{STRESS_UNFINISHED_MAX}} of ambulances fail to reach the hospital before the run ends. Those levels are therefore stress tests, and the main result uses the low level, the only one that stays stationary. An ambulance that never arrives counts at the delay it had built up when the run stopped, so stranding ambulances is never rewarded.

{{KPI_TABLE}}

{{EQUILIBRIUM_SENTENCE}} {{STRESS_SENTENCE}} {{MOTO_SENTENCE}}

{{CLOSED_LOOP_SENTENCE}} {{HINDSIGHT_SENTENCE}} {{SAFETY_SENTENCE}}

## 4.4 Demonstration in R

The dataset can be downloaded from {{DATA_URL}}. The R Markdown file reads the tables, checks every column against the data dictionary, summarises scenarios and missing values, plots the vehicle mix, queues, delay by controller and ambulance passages, and fits a shallow decision tree to the teacher's label on two seeds, tested on the third. {{DEMO_SENTENCE}}

![Figure 2. Methods at the low demand level on the trade-off between ambulance delay and everyone else's delay, with 95% intervals over seed resamples; the Pareto front of the methods that pass the gates in blue.](figures/tradeoff.png){width=92%}

# 5. Standards, Data Governance and Management

- **Process.** CRISP-DM (Chapman et al., 2000) structures the work: the proposal is business understanding, calibration and validators are data understanding, the generator and grouped splits are data preparation, Section 4.3 is modelling and evaluation, and the live view is a research deployment.
- **Data standards.** FAIR principles (Wilkinson et al., 2016) guide the release: a download link and data card make it findable, CSV keeps it accessible, SI units and the dictionary make it interoperable, and seeds make any row reproducible. The data card follows the datasheets pattern (Gebru et al., 2021).
- **Licences.** The road network derives from OpenStreetMap and is shared under the ODbL with attribution; the sensor tables are a produced work under it (Open Data Commons, 2009). TomTom speeds are shown live only, never stored or combined with these data (TomTom, n.d.).
- **Accessibility, security and integrity.** Public download with SHA-256 checksums and versioned releases.
- **Confidentiality and ethics.** The dataset holds no personal data. A real deployment would ingest ambulance location traces, which are personal and, linked to patients, health data under Indonesia's data-protection law, which also requires an impact assessment (Undang-Undang Nomor 27 Tahun 2022, arts. 4, 34). Priority shifts delay onto other road users, mostly motorcyclists, so delay is reported per approach. {{STARVATION_SENTENCE}}

# References

{{REFERENCES}}

# Use of generative AI

We used Claude (Anthropic, via Claude Code, September–November 2026) to brainstorm, critique the simulator design, draft and revise this report from our decisions and notes, draft simulation, analysis and figure code that we reviewed and tested, run label checks, and check references by web search. The problem, decisions and sources are ours; we re-checked every reference.
