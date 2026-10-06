# Supervised models on the 120-episode release

Imitation of the priority-aware signal teacher (Task A), a 30 s-ahead queue forecast (Task B) and, in sections 4-6, the learned controller deployed in the simulator against the proposal's acceptance bound. Every number in this report is computed from the stored results of the same run.

## 1. Data

Release of 120 episodes (generator `traffic_twin_v1.2`, config hash `ecd36a4b3cad`): 253275 main-period decision rows from the signal-tick table (warm-up, cool-down and tail rows removed), split by episode: train 167542, val 30029, test 31390, test_ood 24314.
Episodes per split: train 79, val 14, test 15, test_ood 12. Split integrity verified in code: every stress episode is test_ood and test_ood holds only stress episodes; random_dwell episodes are train-only; no episode is in two splits.

Scenario mixture (episodes): controllers max_pressure 48, max_pressure_eps 36, fixed_time 24, random_dwell 12; demand levels design 48, low 30, high 30, stress 12; sensor_noise_level 0.0 72, 0.1 48; ev_rate default 84, 0 36; demand_asym sym 60, ns_heavy 60; od_pattern default 84, uniform 36.
Calibrated arrival rates (veh/h per entry node, mean over episodes): design 324.1, high 443.7, low 185.9, stress 522.0.
Ambulance episodes 84 (2854 ambulances in total); gridlocked episodes 3; invariant violations 0.
Label balance of `y_mp_action` on the main rows: N 27.4%, E 22.5%, S 27.8%, W 22.3%; the label differs from the current phase (a switch request) on 61.0% of rows; an ambulance is present on some approach in 20.2% of rows; `y_q_total_t30` is NaN on 0.01% of main rows (tail rows already removed).

Total runtime of this pipeline: 563 s (scikit-learn 1.7.2).

### Feature sets (selected by namespace from the data dictionary)

- **truth** (63 columns): `phase`, `time_in_phase_s`, `max_green_remaining_s`, `q_N`, `n_N`, `q_N_l0`, `q_N_l1`, `arr_N_30s`, `q_E`, `n_E`, `q_E_l0`, `q_E_l1`, `arr_E_30s`, `q_S`, `n_S`, `q_S_l0`, `q_S_l1`, `arr_S_30s`, `q_W`, `n_W`, `q_W_l0`, `q_W_l1`, `arr_W_30s`, `down_n_N`, `down_occ_N`, `down_blocked_N`, `down_n_E`, `down_occ_E`, `down_blocked_E`, `down_n_S`, `down_occ_S`, `down_blocked_S`, `down_n_W`, `down_occ_W`, `down_blocked_W`, `nb_N_exists`, `nb_N_phase`, `nb_N_green_toward_me`, `nb_E_exists`, `nb_E_phase`, `nb_E_green_toward_me`, `nb_S_exists`, `nb_S_phase`, `nb_S_green_toward_me`, `nb_W_exists`, `nb_W_phase`, `nb_W_green_toward_me`, `ev_present_N`, `ev_dist_N`, `ev_eta_N`, `ev_q_ahead_N`, `ev_present_E`, `ev_dist_E`, `ev_eta_E`, `ev_q_ahead_E`, `ev_present_S`, `ev_dist_S`, `ev_eta_S`, `ev_q_ahead_S`, `ev_present_W`, `ev_dist_W`, `ev_eta_W`, `ev_q_ahead_W`
- **sensor** (47 columns): `phase`, `time_in_phase_s`, `max_green_remaining_s`, `obs_q_N`, `obs_q_E`, `obs_q_S`, `obs_q_W`, `down_n_N`, `down_occ_N`, `down_blocked_N`, `down_n_E`, `down_occ_E`, `down_blocked_E`, `down_n_S`, `down_occ_S`, `down_blocked_S`, `down_n_W`, `down_occ_W`, `down_blocked_W`, `nb_N_exists`, `nb_N_phase`, `nb_N_green_toward_me`, `nb_E_exists`, `nb_E_phase`, `nb_E_green_toward_me`, `nb_S_exists`, `nb_S_phase`, `nb_S_green_toward_me`, `nb_W_exists`, `nb_W_phase`, `nb_W_green_toward_me`, `ev_present_N`, `ev_dist_N`, `ev_eta_N`, `ev_q_ahead_N`, `ev_present_E`, `ev_dist_E`, `ev_eta_E`, `ev_q_ahead_E`, `ev_present_S`, `ev_dist_S`, `ev_eta_S`, `ev_q_ahead_S`, `ev_present_W`, `ev_dist_W`, `ev_eta_W`, `ev_q_ahead_W`
- **regression** (64 columns): `phase`, `time_in_phase_s`, `max_green_remaining_s`, `q_N`, `n_N`, `q_N_l0`, `q_N_l1`, `arr_N_30s`, `q_E`, `n_E`, `q_E_l0`, `q_E_l1`, `arr_E_30s`, `q_S`, `n_S`, `q_S_l0`, `q_S_l1`, `arr_S_30s`, `q_W`, `n_W`, `q_W_l0`, `q_W_l1`, `arr_W_30s`, `down_n_N`, `down_occ_N`, `down_blocked_N`, `down_n_E`, `down_occ_E`, `down_blocked_E`, `down_n_S`, `down_occ_S`, `down_blocked_S`, `down_n_W`, `down_occ_W`, `down_blocked_W`, `nb_N_exists`, `nb_N_phase`, `nb_N_green_toward_me`, `nb_E_exists`, `nb_E_phase`, `nb_E_green_toward_me`, `nb_S_exists`, `nb_S_phase`, `nb_S_green_toward_me`, `nb_W_exists`, `nb_W_phase`, `nb_W_green_toward_me`, `ev_present_N`, `ev_dist_N`, `ev_eta_N`, `ev_q_ahead_N`, `ev_present_E`, `ev_dist_E`, `ev_eta_E`, `ev_q_ahead_E`, `ev_present_S`, `ev_dist_S`, `ev_eta_S`, `ev_q_ahead_S`, `ev_present_W`, `ev_dist_W`, `ev_eta_W`, `ev_q_ahead_W`, `action_applied`

Never used as features (asserted in code): keys (`episode_id`, `junction_id`), time (`t`, `tick_idx`, period flags), `action_requested`, `forced`, `switched`, and the labels `y_*`. The sensor set keeps only what a loop-detector installation plus the ambulance GPS channel could deliver: no `n_*`, no per-lane queues, no arrival counts, `obs_q_*` instead of `q_*`. Note that the teacher rule is computed from `n_*` and `down_n_*`, so the sensor set can only approximate it.

## 2. Task A: imitation of the priority-aware teacher (4 classes, target `y_mp_action`)

Models: class-balanced logistic regression (one-hot phase columns + standardised numerics), HistGradientBoostingClassifier(max_iter=400, lr=0.05, depth=6, l2=1.0, early stopping, class-balanced) and a depth-5 decision tree. `switch_f1` is the F1 of the binary event *label != current phase* against *prediction != current phase*.

| model | split | n | balanced_accuracy | macro_f1 | switch_f1 | accuracy |
|---|---|---|---|---|---|---|
| logreg_truth | val | 30029 | 0.941 | 0.939 | 0.963 | 0.940 |
| logreg_truth | test | 31390 | 0.941 | 0.940 | 0.963 | 0.940 |
| logreg_truth | test_ood | 24314 | 0.919 | 0.918 | 0.960 | 0.919 |
| hgb_truth | val | 30029 | 0.959 | 0.958 | 0.977 | 0.959 |
| hgb_truth | test | 31390 | 0.949 | 0.948 | 0.970 | 0.949 |
| hgb_truth | test_ood | 24314 | 0.887 | 0.887 | 0.951 | 0.888 |
| tree_truth | val | 30029 | 0.472 | 0.457 | 0.750 | 0.478 |
| tree_truth | test | 31390 | 0.455 | 0.434 | 0.743 | 0.460 |
| tree_truth | test_ood | 24314 | 0.440 | 0.416 | 0.769 | 0.445 |
| logreg_sensor | val | 30029 | 0.736 | 0.735 | 0.787 | 0.737 |
| logreg_sensor | test | 31390 | 0.719 | 0.719 | 0.790 | 0.722 |
| logreg_sensor | test_ood | 24314 | 0.778 | 0.778 | 0.907 | 0.779 |
| hgb_sensor | val | 30029 | 0.768 | 0.767 | 0.843 | 0.769 |
| hgb_sensor | test | 31390 | 0.760 | 0.759 | 0.841 | 0.761 |
| hgb_sensor | test_ood | 24314 | 0.796 | 0.796 | 0.913 | 0.797 |
| tree_sensor | val | 30029 | 0.456 | 0.374 | 0.708 | 0.408 |
| tree_sensor | test | 31390 | 0.418 | 0.351 | 0.701 | 0.384 |
| tree_sensor | test_ood | 24314 | 0.476 | 0.444 | 0.774 | 0.458 |

### GroupKFold(5, groups = episode_id) on the train split, HGB

| model | balanced_accuracy | macro_f1 | folds |
|---|---|---|---|
| hgb_truth | 0.948 +- 0.010 | 0.947 +- 0.010 | 0.945, 0.959, 0.933, 0.959, 0.942 |
| hgb_sensor | 0.770 +- 0.011 | 0.768 +- 0.011 | 0.767, 0.766, 0.791, 0.762, 0.762 |

### Per-class F1 on the test split

| model | F1_N | F1_E | F1_S | F1_W | support_N | support_E | support_S | support_W |
|---|---|---|---|---|---|---|---|---|
| logreg_truth | 0.944 | 0.935 | 0.945 | 0.936 | 8551 | 6830 | 9128 | 6881 |
| hgb_truth | 0.949 | 0.946 | 0.954 | 0.945 | 8551 | 6830 | 9128 | 6881 |
| tree_truth | 0.549 | 0.436 | 0.442 | 0.309 | 8551 | 6830 | 9128 | 6881 |
| logreg_sensor | 0.731 | 0.701 | 0.745 | 0.699 | 8551 | 6830 | 9128 | 6881 |
| hgb_sensor | 0.769 | 0.739 | 0.782 | 0.747 | 8551 | 6830 | 9128 | 6881 |
| tree_sensor | 0.268 | 0.446 | 0.242 | 0.449 | 8551 | 6830 | 9128 | 6881 |

### Confusion matrices

Figure: Task A confusion matrices (counts and row shares) for every model and split. HGB-truth on test:

| label | pred_N | pred_E | pred_S | pred_W |
|---|---|---|---|---|
| N | 8093 | 154 | 169 | 135 |
| E | 117 | 6489 | 123 | 101 |
| S | 149 | 132 | 8700 | 147 |
| W | 143 | 120 | 116 | 6502 |

HGB-sensor on test:

| label | pred_N | pred_E | pred_S | pred_W |
|---|---|---|---|---|
| N | 6525 | 618 | 766 | 642 |
| E | 629 | 5063 | 603 | 535 |
| S | 712 | 653 | 7121 | 642 |
| W | 557 | 538 | 598 | 5188 |

### Permutation importance, HGB-sensor, test split (20000 rows, 5 repeats, balanced accuracy)

| feature | mean | std |
|---|---|---|
| phase | 0.2585 | 0.0033 |
| obs_q_S | 0.1753 | 0.0015 |
| obs_q_N | 0.1630 | 0.0013 |
| obs_q_E | 0.1338 | 0.0021 |
| obs_q_W | 0.1273 | 0.0023 |
| ev_present_W | 0.0621 | 0.0010 |
| ev_present_E | 0.0584 | 0.0012 |
| ev_present_N | 0.0507 | 0.0008 |
| ev_present_S | 0.0473 | 0.0006 |
| down_n_W | 0.0410 | 0.0012 |
| down_n_E | 0.0314 | 0.0012 |
| down_n_N | 0.0231 | 0.0011 |
| down_n_S | 0.0195 | 0.0020 |
| time_in_phase_s | 0.0162 | 0.0020 |
| down_occ_N | 0.0133 | 0.0014 |

Figure: permutation importance of the HGB-sensor model on the test split.

### Depth-5 decision tree (first three levels, truth features)

```
|--- ev_dist_W <= -0.50
|   |--- ev_eta_E <= -0.50
|   |   |--- ev_q_ahead_N <= -0.50
|   |   |   |--- n_S <= 5.50
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- n_S >  5.50
|   |   |   |   |--- truncated branch of depth 2
|   |   |--- ev_q_ahead_N >  -0.50
|   |   |   |--- ev_present_S <= 0.50
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- ev_present_S >  0.50
|   |   |   |   |--- truncated branch of depth 2
|   |--- ev_eta_E >  -0.50
|   |   |--- ev_eta_N <= -0.50
|   |   |   |--- ev_eta_S <= -0.50
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- ev_eta_S >  -0.50
|   |   |   |   |--- truncated branch of depth 2
|   |   |--- ev_eta_N >  -0.50
|   |   |   |--- ev_dist_N <= 21.44
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- ev_dist_N >  21.44
|   |   |   |   |--- truncated branch of depth 2
|--- ev_dist_W >  -0.50
|   |--- ev_dist_N <= -0.50
|   |   |--- ev_dist_E <= -0.50
|   |   |   |--- ev_eta_S <= -0.50
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- ev_eta_S >  -0.50
|   |   |   |   |--- truncated branch of depth 2
|   |   |--- ev_dist_E >  -0.50
|   |   |   |--- ev_dist_E <= 14.97
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- ev_dist_E >  14.97
|   |   |   |   |--- truncated branch of depth 2
|   |--- ev_dist_N >  -0.50
|   |   |--- ev_eta_W <= 5.33
|   |   |   |--- ev_eta_N <= 2.82
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- ev_eta_N >  2.82
|   |   |   |   |--- truncated branch of depth 2
|   |   |--- ev_eta_W >  5.33
|   |   |   |--- ev_dist_N <= 34.41
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- ev_dist_N >  34.41
|   |   |   |   |--- truncated branch of depth 2
```
Full rules: `taskA_tree_rules_truth.txt`, `taskA_tree_rules_sensor.txt`.

## 3. Task B: queue 30 s ahead (target `y_q_total_t30`, feature set `regression` = truth + action_applied)

| model | split | n | mae | rmse | r2 | target_mean |
|---|---|---|---|---|---|---|
| naive: q_total now | val | 30029 | 3.40 | 4.65 | 0.978 | 23.07 |
| ridge | val | 30029 | 2.91 | 3.85 | 0.985 | 23.07 |
| hgb | val | 30029 | 2.41 | 3.22 | 0.989 | 23.07 |
| naive: q_total now | test | 31390 | 3.68 | 5.29 | 0.973 | 27.36 |
| ridge | test | 31390 | 3.09 | 4.23 | 0.983 | 27.36 |
| hgb | test | 31390 | 2.56 | 3.41 | 0.989 | 27.36 |
| naive: q_total now | test_ood | 24314 | 5.92 | 7.61 | 0.928 | 97.08 |
| ridge | test_ood | 24314 | 4.79 | 6.07 | 0.954 | 97.08 |
| hgb | test_ood | 24314 | 4.22 | 5.36 | 0.964 | 97.08 |

### Partial dependence of the HGB prediction on `action_applied` (31390 test rows)

| action_name | mean_pred_queue | mean_pred_queue_when_hold | mean_pred_queue_when_switch | n_rows |
|---|---|---|---|---|
| N | 27.37 | 27.71 | 27.25 | 31390 |
| E | 27.72 | 28.06 | 27.61 | 31390 |
| S | 27.33 | 27.65 | 27.22 | 31390 |
| W | 27.93 | 26.66 | 28.34 | 31390 |

Holding the current phase gives a mean predicted queue of 27.52 veh; the best other action 26.91 veh; in 69.8% of test rows some switch is predicted to lower the queue at t+30. Figure: partial dependence of the Task B prediction on the applied action.

The 30 s-ahead queue is mostly explained by the queue now (see the naive baseline), so R^2 alone flatters; the control-relevant quantity is the action effect above, which is small compared with the state effect.

### Reading Tasks A and B

- The teacher is a deterministic rule of `n_*`, `down_n_*`, `down_blocked_*`, `time_in_phase_s` and the `ev_*` channel. With truth features HGB reaches balanced accuracy 0.949 on test and 0.887 on stress (test_ood); the remaining errors sit on the hysteresis boundary (pressure difference close to 2) and on max-green rows.
- With sensor features (queues only, noisy in the noisy episodes) HGB reaches 0.760 on test and 0.796 on stress: the gap to the truth model is the price of not observing moving vehicles (`n_a`) and arrivals.
- Logistic regression (truth) reaches 0.941: the rule is a difference of counts with a threshold, which a linear model can only partly express across phases.
- Task B: HGB MAE 2.56 veh vs naive 3.68 veh on test; on stress 4.22 vs 5.92.

## 4. Closed loop: the learned controller deployed in the simulator

Protocol: controllers ['fixed_time', 'max_pressure', 'preempt_naive', 'priority_mp', 'ml_sensor'] x demand ['design', 'high'] x 8 paired seeds (900001..900008, disjoint from the training seeds 100000..100119), demand_asym sym, od default, ev_rate default (lambda_ev = 4.0 /h per node), sensor_noise_level 0. Every (controller, demand, seed) was also run with ambulances off (same seed; 112 such episodes) to measure `car_mean_delay_penalty_s`. 224 episodes in total, wall clock 352 s with 8 worker processes (serial inference per worker, BLAS/OMP threads = 1). The ML controller is the gradient-boosted-tree classifier on sensor features, called `ml_sensor` in the tables, with the hysteresis rule request argmax only if P(best) > P(current) + 0.10. Invariant violations over all episodes: 0; gridlocked episodes: 0.

Simulator `traffic_twin_v1.2`, config hash `ecd36a4b3cad`; dataset `traffic_twin_v1.2`, config hash `ecd36a4b3cad` (asserted equal: True).

### 4.1 Mean over seeds per controller and demand level (ambulances on)

| demand | controller | n | ev_mean_travel_time_s | ev_mean_delay_s | ev_p90_delay_s | car_mean_delay_s | car_p90_delay_s | throughput_vph | switches_per_min | gridlock_rate | n_red_commits | n_ambulances |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| design | fixed_time | 8 | 123.78 | 91.51 | 146.21 | 100.48 | 156.61 | 2622 | 2.50 | 0.00 | 287 | 34.0 |
| design | max_pressure | 8 | 100.49 | 68.23 | 109.78 | 71.83 | 123.09 | 2609 | 4.22 | 0.00 | 755 | 34.0 |
| design | preempt_naive | 8 | 50.85 | 18.58 | 28.50 | 73.19 | 126.19 | 2608 | 4.19 | 0.00 | 756 | 34.0 |
| design | priority_mp | 8 | 56.93 | 24.69 | 40.31 | 73.47 | 125.76 | 2607 | 4.21 | 0.00 | 754 | 34.0 |
| design | ml_sensor | 8 | 56.30 | 24.05 | 37.41 | 91.31 | 148.04 | 2603 | 3.84 | 0.00 | 668 | 34.0 |
| high | fixed_time | 8 | 192.40 | 160.22 | 249.47 | 155.41 | 254.47 | 3520 | 2.50 | 0.00 | 429 | 34.0 |
| high | max_pressure | 8 | 335.76 | 309.97 | 491.54 | 333.56 | 550.76 | 3147 | 4.41 | 0.00 | 1240 | 34.0 |
| high | preempt_naive | 8 | 131.67 | 102.94 | 219.18 | 282.14 | 458.77 | 3252 | 4.09 | 0.00 | 1132 | 34.0 |
| high | priority_mp | 8 | 192.62 | 178.02 | 310.56 | 367.71 | 621.10 | 3091 | 4.68 | 0.00 | 1459 | 34.0 |
| high | ml_sensor | 8 | 160.87 | 144.04 | 270.41 | 338.68 | 568.46 | 3164 | 4.27 | 0.00 | 1184 | 34.0 |

### 4.2 The proposal's acceptance bound: car mean delay under the ML controller <= 1.10 x `max_pressure`

Per seed the ratio car_delay(ml_sensor) / car_delay(max_pressure) is computed on the paired episodes; PASS iff ratio <= 1.10. `mean_ratio` carries a t-based 95% interval over seeds; `bound_met_ci` = that interval's upper end is below the factor. The proposal states the same bound on the paired difference: `bound_met_methodology` = the upper end of the 95% t-interval on mean(ml - max_pressure) is at most `margin_s` = 0.10 x the mean max_pressure car delay of that cell.

| sensor_noise_level | demand_level | n_pairs | n_pass | pass_rate | mean_ratio | ci95_low | ci95_high | min_ratio | max_ratio | mean_ml | mean_reference | bound_met_all_seeds | bound_met_ci |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | design | 8 | 0 | 0.000 | 1.272 | 1.251 | 1.293 | 1.241 | 1.320 | 91.3 | 71.8 | False | False |
| 0 | high | 8 | 8 | 1.000 | 1.014 | 0.971 | 1.056 | 0.948 | 1.097 | 338.7 | 333.6 | True | True |
| 0 | all | 16 | 8 | 0.500 | 1.143 | 1.069 | 1.217 | 0.948 | 1.320 | 215.0 | 202.7 | False | False |
| 0.1 | design | 8 | 0 | 0.000 | 1.324 | 1.299 | 1.349 | 1.276 | 1.362 | 95.1 | 71.8 | False | False |
| 0.1 | high | 8 | 8 | 1.000 | 0.954 | 0.906 | 1.002 | 0.872 | 1.042 | 317.8 | 333.6 | True | True |
| 0.1 | all | 16 | 8 | 0.500 | 1.139 | 1.034 | 1.243 | 0.872 | 1.362 | 206.5 | 202.7 | False | False |

The same cells in the methodology's paired-difference form (seconds of car delay):

| sensor_noise_level | demand_level | n_pairs | diff_mean_s | diff_ci95_half_s | diff_ci95_high_s | margin_s | bound_met_methodology |
|---|---|---|---|---|---|---|---|
| 0 | design | 8 | 19.48 | 1.28 | 20.77 | 7.18 | False |
| 0 | high | 8 | 5.11 | 14.43 | 19.54 | 33.36 | True |
| 0 | all | 16 | 12.30 | 7.44 | 19.74 | 20.27 | True |
| 0.1 | design | 8 | 23.28 | 2.17 | 25.44 | 7.18 | False |
| 0.1 | high | 8 | -15.72 | 17.32 | 1.60 | 33.36 | True |
| 0.1 | all | 16 | 3.78 | 13.15 | 16.93 | 20.27 | True |

Figure: car-delay ratio of the ML controller to the reference per demand level, with the 1.10 bound.

### 4.3 Paired differences versus `max_pressure` (mean +- t-based 95% half-width, n = pairs)

Negative delay / penalty / recovery / fairness differences favour the controller; positive throughput differences favour it. `*` = the 95% interval excludes zero.

**design demand**

| controller | ev_mean_travel_time_s | ev_mean_delay_s | mean_total_delay_s | p90_total_delay_s | throughput_vph | switches_per_min | n_red_commits | gridlock |
|---|---|---|---|---|---|---|---|---|
| fixed_time | +23.29 +- 6.47 (n=8) * | +23.29 +- 6.48 (n=8) * | +28.65 +- 1.58 (n=8) * | +33.51 +- 2.47 (n=8) * | +13.00 +- 9.50 (n=8) * | -1.72 +- 0.02 (n=8) * | -468.12 +- 30.08 (n=8) * | +0.00 +- 0.00 (n=8) |
| preempt_naive | -49.64 +- 5.02 (n=8) * | -49.64 +- 5.01 (n=8) * | +1.36 +- 1.44 (n=8) | +3.10 +- 3.70 (n=8) | -1.38 +- 4.47 (n=8) | -0.03 +- 0.05 (n=8) | +0.50 +- 25.74 (n=8) | +0.00 +- 0.00 (n=8) |
| priority_mp | -43.56 +- 4.57 (n=8) * | -43.54 +- 4.58 (n=8) * | +1.64 +- 1.21 (n=8) * | +2.67 +- 2.30 (n=8) * | -2.12 +- 4.20 (n=8) | -0.00 +- 0.05 (n=8) | -1.00 +- 26.36 (n=8) | +0.00 +- 0.00 (n=8) |
| ml_sensor | -44.19 +- 4.60 (n=8) * | -44.18 +- 4.60 (n=8) * | +19.48 +- 1.28 (n=8) * | +24.95 +- 2.30 (n=8) * | -5.50 +- 5.38 (n=8) * | -0.38 +- 0.06 (n=8) * | -87.12 +- 25.61 (n=8) * | +0.00 +- 0.00 (n=8) |

**high demand**

| controller | ev_mean_travel_time_s | ev_mean_delay_s | mean_total_delay_s | p90_total_delay_s | throughput_vph | switches_per_min | n_red_commits | gridlock |
|---|---|---|---|---|---|---|---|---|
| fixed_time | -143.36 +- 32.61 (n=8) * | -149.75 +- 34.47 (n=8) * | -178.15 +- 25.21 (n=8) * | -296.29 +- 34.83 (n=8) * | +372.88 +- 40.33 (n=8) * | -1.91 +- 0.02 (n=8) * | -810.62 +- 33.72 (n=8) * | +0.00 +- 0.00 (n=8) |
| preempt_naive | -204.08 +- 30.26 (n=8) * | -207.03 +- 29.50 (n=8) * | -51.42 +- 19.77 (n=8) * | -91.99 +- 27.78 (n=8) * | +104.62 +- 21.31 (n=8) * | -0.32 +- 0.06 (n=8) * | -107.12 +- 41.66 (n=8) * | +0.00 +- 0.00 (n=8) |
| priority_mp | -143.14 +- 26.32 (n=8) * | -131.95 +- 17.93 (n=8) * | +34.15 +- 11.57 (n=8) * | +70.34 +- 30.54 (n=8) * | -56.62 +- 23.97 (n=8) * | +0.27 +- 0.07 (n=8) * | +219.12 +- 47.20 (n=8) * | +0.00 +- 0.00 (n=8) |
| ml_sensor | -174.89 +- 25.65 (n=8) * | -165.93 +- 21.17 (n=8) * | +5.11 +- 14.43 (n=8) | +17.70 +- 33.54 (n=8) | +16.50 +- 25.00 (n=8) | -0.14 +- 0.08 (n=8) * | -55.50 +- 32.62 (n=8) * | +0.00 +- 0.00 (n=8) |

### 4.4 Secondary metrics (ambulance, car penalty, recovery, fairness)

`ev_mean_travel_time_s` = mean ambulance travel time; `car_mean_delay_penalty_s` = car mean delay with ambulances minus the paired episode without ambulances (same seed and controller); `recovery_time_s` = mean time after an ambulance clears a junction until that junction's queue is back within 10% of its value at first sighting; `fairness_ratio` = max / min of the mean car delay over the 8 entry nodes (1 = perfectly even).

| demand | controller | ev_mean_travel_time_s | car_mean_delay_penalty_s | recovery_time_s | fairness_ratio | worst_origin (mode) |
|---|---|---|---|---|---|---|
| design | fixed_time | 123.78 | 0.09 | 82.28 | 1.358 | E1 |
| design | max_pressure | 100.49 | 2.04 | 54.64 | 1.128 | E1 |
| design | preempt_naive | 50.85 | 3.40 | 58.93 | 1.158 | E0 |
| design | priority_mp | 56.93 | 3.68 | 62.54 | 1.128 | S0 |
| design | ml_sensor | 56.30 | 1.86 | 60.44 | 1.127 | E1 |
| high | fixed_time | 192.40 | 3.65 | 69.20 | 1.322 | N0 |
| high | max_pressure | 335.76 | 15.80 | 126.58 | 1.208 | E1 |
| high | preempt_naive | 131.67 | -35.62 | 46.15 | 1.195 | N1 |
| high | priority_mp | 192.62 | 49.95 | 93.93 | 1.356 | N0 |
| high | ml_sensor | 160.87 | 23.89 | 62.03 | 1.369 | W1 |

Paired differences of the secondary metrics versus `max_pressure`:

**design demand**

| controller | ev_mean_travel_time_s | car_mean_delay_penalty_s | recovery_time_s | fairness_ratio |
|---|---|---|---|---|
| fixed_time | +23.29 +- 6.47 (n=8) * | -1.95 +- 1.80 (n=8) * | +27.64 +- 39.26 (n=8) | +0.230 +- 0.053 (n=8) * |
| preempt_naive | -49.64 +- 5.02 (n=8) * | +1.36 +- 1.44 (n=8) | +4.30 +- 23.60 (n=8) | +0.030 +- 0.040 (n=8) |
| priority_mp | -43.56 +- 4.57 (n=8) * | +1.64 +- 1.21 (n=8) * | +7.90 +- 32.98 (n=8) | +0.000 +- 0.036 (n=8) |
| ml_sensor | -44.19 +- 4.60 (n=8) * | -0.18 +- 3.45 (n=8) | +5.80 +- 32.25 (n=8) | -0.001 +- 0.032 (n=8) |

**high demand**

| controller | ev_mean_travel_time_s | car_mean_delay_penalty_s | recovery_time_s | fairness_ratio |
|---|---|---|---|---|
| fixed_time | -143.36 +- 32.61 (n=8) * | -12.15 +- 10.47 (n=8) * | -57.38 +- 93.26 (n=8) | +0.114 +- 0.125 (n=8) |
| preempt_naive | -204.08 +- 30.26 (n=8) * | -51.42 +- 19.77 (n=8) * | -80.43 +- 41.66 (n=8) * | -0.013 +- 0.097 (n=8) |
| priority_mp | -143.14 +- 26.32 (n=8) * | +34.15 +- 11.57 (n=8) * | -32.65 +- 52.95 (n=8) | +0.148 +- 0.131 (n=8) * |
| ml_sensor | -174.89 +- 25.65 (n=8) * | +8.09 +- 16.39 (n=8) | -64.55 +- 47.06 (n=8) * | +0.160 +- 0.211 (n=8) |

### 4.5 Sensor-noise variant (the sensor-realistic claim)

`ml_sensor` and `max_pressure` re-run on the same 8 seeds x ['design', 'high'] at sensor_noise_level 0.1 (obs_q_a = Binomial(q_a, 1-p_miss) + Poisson(p_miss/2)); the ML controller reads `obs_q_*`, max_pressure reads the true `n_*` and is therefore unaffected by construction except through the ambulance stream, which is identical. Means over seeds:

| demand | controller | n | car_delay_noise0 | car_delay_noise | ev_delay_noise0 | ev_delay_noise | throughput_noise0 | throughput_noise | agreement_noise0 | agreement_noise | switches_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|
| design | max_pressure | 8 | 71.83 | 71.83 | 68.23 | 68.23 | 2609 | 2609 | 0.900 | 0.900 | 4.22 |
| design | ml_sensor | 8 | 91.31 | 95.10 | 24.05 | 25.09 | 2603 | 2598 | 0.674 | 0.652 | 3.77 |
| high | max_pressure | 8 | 333.56 | 333.56 | 309.97 | 309.97 | 3147 | 3147 | 0.695 | 0.695 | 4.41 |
| high | ml_sensor | 8 | 338.68 | 317.84 | 144.04 | 121.52 | 3164 | 3212 | 0.707 | 0.682 | 4.15 |

Paired differences `ml_sensor` - `max_pressure` at noise 0.1:

| demand | mean_total_delay_s | ev_mean_delay_s | throughput_vph | car_mean_delay_penalty_s | fairness_ratio |
|---|---|---|---|---|---|
| design | +23.28 +- 2.17 (n=8) * | -43.14 +- 4.00 (n=8) * | -10.38 +- 9.99 (n=8) * | +0.13 +- 3.32 (n=8) | +0.013 +- 0.030 (n=8) |
| high | -15.72 +- 17.32 (n=8) | -188.45 +- 23.83 (n=8) * | +64.25 +- 27.51 (n=8) * | +12.82 +- 17.49 (n=8) | +0.109 +- 0.205 (n=8) |

The acceptance bound at this noise level is in the table of 4.2 (rows with sensor_noise_level 0.1).

### 4.6 Request statistics on main-period decision rows (per controller, mean over episodes, noise 0)

`agreement_with_teacher` = share of decision instants where the controller's request equals the priority-aware teacher label recomputed on the same row (for `priority_mp` it is 1 by construction); `request_switch_rate` = share of requests that differ from the current phase; `accepted_switch_rate` = share of instants at which the plant actually started an all-red; `forced_share` = share of instants where the plant overrode the request (max-green or starvation guard).

| demand | controller | agreement_with_teacher | disagreement_rate | request_switch_rate | teacher_switch_rate | accepted_switch_rate | forced_share | n_decisions_main |
|---|---|---|---|---|---|---|---|---|
| design | fixed_time | 0.370 | 0.630 | 0.250 | 0.711 | 0.250 | 0.000 | 2395 |
| design | max_pressure | 0.900 | 0.100 | 0.490 | 0.535 | 0.490 | 0.000 | 2065 |
| design | preempt_naive | 0.987 | 0.013 | 0.485 | 0.485 | 0.485 | 0.000 | 2071 |
| design | priority_mp | 1.000 | 0.000 | 0.490 | 0.490 | 0.490 | 0.000 | 2066 |
| design | ml_sensor | 0.674 | 0.326 | 0.431 | 0.574 | 0.431 | 0.001 | 2138 |
| high | fixed_time | 0.322 | 0.678 | 0.250 | 0.738 | 0.250 | 0.000 | 2396 |
| high | max_pressure | 0.695 | 0.305 | 0.522 | 0.671 | 0.522 | 0.000 | 2027 |
| high | preempt_naive | 0.917 | 0.083 | 0.469 | 0.533 | 0.470 | 0.002 | 2089 |
| high | priority_mp | 1.000 | 0.000 | 0.567 | 0.567 | 0.569 | 0.004 | 1975 |
| high | ml_sensor | 0.707 | 0.293 | 0.498 | 0.593 | 0.499 | 0.003 | 2055 |

Figure: ambulance and car delay per controller and demand level, with paired differences to the reference. The per-episode rows (ambulance-off and noise-variant episodes included), the paired differences, the bound and the means per cell are published as tables with this report.

## 5. Verdict

- **Imitation (Task A).** HGB-truth reaches balanced accuracy 0.949 on test and 0.887 on stress (test_ood); HGB-sensor, the deployed model, 0.760 / 0.796 (switch F1 0.841 on test). The gap between the two is the price of not observing moving vehicles and arrivals; the teacher is a rule of `n_*`, which no loop detector delivers.
- **Queue forecast (Task B).** HGB MAE 2.56 veh on test vs 3.68 veh for the naive 'queue now' baseline (R^2 0.989); on stress 4.22 vs 5.92. The action effect in the partial dependence is -0.61 veh (best other action minus hold), small against the state effect.
- **Acceptance bound, design, noise 0**: 0/8 seeds pass (pass rate 0.00); mean ratio 1.272 [1.251, 1.293] (car delay 91.3 s vs 71.8 s). The bound is NOT met on every seed. Methodology form: paired difference +19.5 s, 95% upper end +20.8 s against a margin of 7.2 s: NOT met.
- **Acceptance bound, high, noise 0**: 8/8 seeds pass (pass rate 1.00); mean ratio 1.014 [0.971, 1.056] (car delay 338.7 s vs 333.6 s). The bound is met on every seed and the interval lies below 1.10. Methodology form: paired difference +5.1 s, 95% upper end +19.5 s against a margin of 33.4 s: met.
- **Acceptance bound, pooled over demand levels, noise 0**: pass rate 0.50 (8/16), mean ratio 1.143 [1.069, 1.217].
- **Acceptance bound, design, noise 0.1**: 0/8 seeds pass, mean ratio 1.324 [1.299, 1.349].
- **Acceptance bound, high, noise 0.1**: 8/8 seeds pass, mean ratio 0.954 [0.906, 1.002].
- **design**: on ambulance delay the ML controller beats max_pressure (-44.2 +- 4.6 s); on car delay it loses to max_pressure (+19.5 +- 1.3 s). Against its own teacher priority_mp the ML controller's ambulance delay differs by -0.6 s (mean over seeds). The teacher itself changes ambulance delay by -43.5 +- 4.6 s and naive preemption by -49.6 +- 5.0 s versus max_pressure.
  - car delay penalty of the ambulances: ML does not differ significantly from max_pressure (-0.178 +- 3.447; ML 1.858 vs 2.036).
  - recovery time: ML does not differ significantly from max_pressure (+5.801 +- 32.253; ML 60.440 vs 54.639).
  - fairness ratio: ML does not differ significantly from max_pressure (-0.001 +- 0.032; ML 1.127 vs 1.128).
- **high**: on ambulance delay the ML controller beats max_pressure (-165.9 +- 21.2 s); on car delay it does not differ significantly from max_pressure (+5.1 +- 14.4 s). Against its own teacher priority_mp the ML controller's ambulance delay differs by -34.0 s (mean over seeds). The teacher itself changes ambulance delay by -131.9 +- 17.9 s and naive preemption by -207.0 +- 29.5 s versus max_pressure.
  - car delay penalty of the ambulances: ML does not differ significantly from max_pressure (+8.088 +- 16.386; ML 23.890 vs 15.801).
  - recovery time: ML beats max_pressure (-64.549 +- 47.062; ML 62.031 vs 126.580).
  - fairness ratio: ML does not differ significantly from max_pressure (+0.160 +- 0.211; ML 1.369 vs 1.208).
- **Reference check, high**: plain fixed_time beats max_pressure on car delay (-178.1 +- 25.2 s, throughput +373 +- 40 veh/h). The max-pressure rule as specified for this simulator (split phasing, one-approach pressure, 2-vehicle hysteresis) is therefore not a strong baseline at this demand; the ML controller imitates a teacher built on that rule and inherits the weakness. This is a finding about the simulator's controllers, reported as is.
- **Agreement.** Deployed, the ML controller's request equals the teacher's on 69.0% of main-period decision instants (mean over its episodes at noise 0); the residual disagreement is concentrated on hysteresis-boundary and max-green rows and on ambulance rows, where the sensor set cannot see `n_*`.
- **Safety.** 0 invariant violations in 224 closed-loop episodes: the signal plant, not the model, guarantees safety.
- **Scope.** All of this is one simulator with its own teacher rule and calibrated demand; the closed-loop seeds are disjoint from training but the demand model, OD pattern and vehicle population are the same as in training. Nothing here is evidence about a real junction.

## 6. Reproducibility

- Simulator: `traffic_twin_v1.2`, config hash `ecd36a4b3cad`; dataset `traffic_twin_v1.2`, config hash `ecd36a4b3cad`, 120 episodes.
- Software at evaluation time: Python 3.13.2, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, joblib 1.5.2; macOS-15.2-arm64-arm-64bit-Mach-O.
- Supervised pipeline: 563 s; scikit-learn 1.7.2, dataset config hash `ecd36a4b3cad`.
- Closed loop: 352 s, 8 workers, seeds 900001..900008, hysteresis margin 0.1, noise variant 0.1.
- Every number in this report is computed from the stored per-episode results; the report can be rebuilt without re-simulating.
