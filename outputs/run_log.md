# Run log: Phase 2 extensions to `fixed_model.ipynb`

Facts only; interpretation is left to the report.

## 1. Environment
- Date of run: 2026-10-11 22:40 to 2026-10-12 01:56 (3 h 16 min, first run, all caches empty except `nested_*.csv`)
- Machine: Windows 11, AMD Ryzen 5 7535HS, 12 logical CPUs, 15.2 GB RAM
- Python 3.12.14 (project `.venv`, kernel `ml-assignment-venv`), scikit-learn 1.6.1, numpy 2.2.6, pandas 2.2.3, scipy 1.15.2, joblib 1.6.0, psutil 7.2.2
- `SEED = 1`; all `RUN_HEAVY` flags `False` (missing caches were computed automatically)

## 2. Regression check (Step 6a)
Passed for DR14, DR16, DR17 (tuned-RF accuracy and macro-F1 within 1e-4 of the original 12 h run).

## 3. Timings
The in-kernel `TIMINGS` dict was lost when the kernel shut down, so these are wall times of the notebook cells from the runner's progress log.

| Step (cell) | Time (min) |
|---|---|
| Step 6 searches, all 3 releases (cell 17) | 90.7 |
| 6b probability metrics | 0.0 |
| 6c learning curves, 3 releases | 18.1 |
| 6c validation curve DR17 | 21.7 |
| Step 7 nested CV (loaded from `nested_*.csv`) | 0.0 |
| Step 8 ablations | 5.4 |
| 8b re-tuned no-redshift RF, DR17 | 30.7 |
| Step 9 grouped / stratified CV (+ DR16->DR17) | 18.8 |
| Step 10 untuned baselines | 1.6 |
| 10c tuned HistGB, 3 releases | 3.0 |
| 10b efficiency DR17 | 2.9 |
| Step 11 stress tests | 3.1 |

Note: the new searches are not comparable with the original 12 h figure, which was dominated by nested CV (loaded from CSV this time).

## 4. Results

### 4.1 Probability metrics (Step 6b, test split)
| Release | Model | log loss | QSO AP | ROC-AUC macro OvR | QSO prevalence |
|---|---|---|---|---|---|
| DR14 | default RF | 0.0745 | 0.9825 | 0.9971 | 0.085 |
| DR14 | tuned RF | 0.0436 | 0.9922 | 0.9989 | 0.085 |
| DR16 | default RF | 0.0488 | 0.9895 | 0.9981 | 0.1058 |
| DR16 | tuned RF | 0.0267 | 0.9918 | 0.9990 | 0.1058 |
| DR17 | default RF | 0.1107 | 0.9804 | 0.9954 | 0.1896 |
| DR17 | tuned RF | 0.0659 | 0.9848 | 0.9968 | 0.1896 |

### 4.2 Overfitting diagnostics (Step 6c)
Train vs test macro-F1 (gap):

| Release | default RF | tuned RF |
|---|---|---|
| DR14 | 1.0000 vs 0.9828 (0.0172) | 0.9910 vs 0.9863 (0.0047) |
| DR16 | 1.0000 vs 0.9888 (0.0112) | 0.9931 vs 0.9893 (0.0038) |
| DR17 | 1.0000 vs 0.9754 (0.0246) | 0.9844 vs 0.9762 (0.0083) |

Learning curves (tuned RF, 5-fold; DR16/DR17 capped at 200 trees), smallest -> largest training size:
- DR14: gap 0.0149 -> 0.0047; validation F1 0.9725 -> 0.9849 (training sizes 640 -> 6400)
- DR16: gap 0.0055 -> 0.0038; validation F1 0.9854 -> 0.9893 (6400 -> 64000)
- DR17: gap 0.0099 -> 0.0081; validation F1 0.9710 -> 0.9761 (6399 -> 63999)
- Validation F1 over the last two training sizes: DR14 0.9846 -> 0.9849, DR16 0.9889 -> 0.9893, DR17 0.9759 -> 0.9761.

Validation curve, DR17, `min_samples_leaf` 1/2/4/8/16/32/64 (200 trees): validation F1 0.9761, 0.9762, 0.9760, 0.9761, 0.9755, 0.9752, 0.9737; train F1 0.9953, 0.9894, 0.9869, 0.9825, 0.9792, 0.9771, 0.9752. Selected value in the Step 6 search: see notebook (`best_params_`).

### 4.3 Tuned HistGB vs tuned RF (Step 10c, same split/features/search protocol)
| Release | RF acc | HGB acc | RF macro-F1 | HGB macro-F1 | McNemar b (RF right, HGB wrong) | c (RF wrong, HGB right) | p |
|---|---|---|---|---|---|---|---|
| DR14 | 0.9920 | 0.9885 | 0.9863 | 0.9828 | 11 | 4 | 0.1213 |
| DR16 | 0.99425 | 0.99095 | 0.9893 | 0.9862 | 77 | 11 | 4.2e-12 |
| DR17 | 0.9796 | 0.97825 | 0.9762 | 0.9748 | 77 | 50 | 0.0210 |

Best HistGB parameters:
- DR14: lr 0.1896, max_iter 804, max_leaf_nodes 246, max_depth 6, min_samples_leaf 150, l2 3.4e-6, max_features 0.5, class_weight None
- DR16: lr 0.0170, max_iter 244, max_leaf_nodes 110, max_depth 8, min_samples_leaf 106, l2 0.0133, max_features 0.7, class_weight None
- DR17: lr 0.1074, max_iter 212, max_leaf_nodes 116, max_depth None, min_samples_leaf 12, l2 0.0625, max_features 0.7, class_weight None

Other tuned HistGB numbers (log loss / QSO AP / train-test gap): DR14 0.0421 / 0.9918 / 0.0166; DR16 0.0263 / 0.9924 / 0.0056; DR17 0.0725 / 0.9846 / 0.0208.

### 4.4 Re-tuned no-redshift RF (Step 8b, DR17 only)
| Metric | tuned RF (full) | no-redshift (reused params) | no-redshift (re-tuned) |
|---|---|---|---|
| accuracy | 0.9796 | 0.8910 | 0.8923 |
| macro-F1 | 0.9762 | 0.8589 | 0.8619 |
| recall QSO | 0.9314 | 0.8125 | 0.8307 |
| recall STAR | 0.9993 | 0.7854 | 0.8006 |

Re-tuned parameters: class_weight balanced, entropy, max_depth 40, max_features log2, min_samples_leaf 3, min_samples_split 6, n_estimators 721.

### 4.5 Efficiency, DR17 (Step 10b; RF pipelines with n_jobs=-1)
| Model | fit (s) | batch predict (s) | throughput (obj/s) | single latency (ms) | pickle (MB) | nodes / iterations |
|---|---|---|---|---|---|---|
| Default RF | 5.1 | 0.099 | 202,525 | 20.1 | 33.7 | 400,722 nodes, mean depth 31.3 |
| Tuned RF | 83.8 | 0.349 | 57,235 | 61.8 | 71.3 | 846,984 nodes, mean depth 28.4 |
| Tuned HistGB | 2.4 | 0.076 | 264,191 | 4.0 | 1.9 | 51 iterations |
| GaussianNB | 0.11 | 0.011 | 1,856,493 | 2.2 | 0.002 | |
| kNN | 0.21 | 3.19 | 6,262 | 3.2 | 10.1 | |
| LogReg | 0.52 | 0.007 | 2,923,207 | 2.1 | 0.002 | |
| MLP | 43.1 | 0.015 | 1,330,796 | 2.2 | 0.064 | |
| HistGB (untuned) | 0.87 | 0.065 | 308,129 | 4.1 | 0.65 | 61 iterations |

RSS deltas are approximate (e.g. Tuned RF reports -1.1 MB, i.e. noise-level).

### 4.6 Stress tests (Step 11, unchanged logic, now cached)
DR17 magnitude noise, macro-F1 (standard / no-redshift): sigma 0.02: 0.9763 / 0.8544; 0.05: 0.9756 / 0.8221; 0.1: 0.9742 / 0.7594; 0.2: 0.9655 / 0.6688; 0.5: 0.9418 / 0.4814.
DR14 noise columns, macro-F1 (RF / kNN): k=0: 0.9828 / 0.9470; 10: 0.9813 / 0.8538; 50: 0.9821 / 0.7427; 200: 0.9777 / 0.6312.

## 5. Exported files
- Tables (`outputs/tables/`): `6b_prob_metrics_{DR14,DR16,DR17}`, `6c_train_test_gap`, `6c_learning_curves`, `6c_validation_curve_DR17`, `8b_noz_retuned_DR17`, `10b_efficiency_DR17`, `10c_hgb_cv_results_{DR14,DR16,DR17}`, `10c_rf_vs_hgb_{DR14,DR16,DR17}`, `10c_mcnemar_rf_vs_hgb`, `11_stress_gaussian_noise`, `11_stress_noise_features`, `12_summary` (all `.csv`)
- Figures (`outputs/figures/`): `6b_roc_pr_{DR14,DR16,DR17}`, `6c_learning_curves`, `6c_validation_curve_DR17`, `11_stress_gaussian_noise`, `11_stress_noise_features` (all `.png`, dpi 200)
- Caches (`cache/`, git-ignored): `search_*`, `grouped_*`, `lc_*`, `vc_DR17`, `hgb_search_*`, `noz_search_DR17`, `stress_noise`, `stress_noisecols` (`.joblib`). The pre-extension notebook backup and the cell-by-cell progress log were removed during repository cleanup.

## 6. Deviations, skips and caveats
- **Step 8b** re-tuned DR17 only (the brief's allowed shortcut); DR14/DR16 were not re-tuned (`NOZ_RELEASES` controls this).
- **Step 10c optional nested CV for HistGB on DR14 was skipped.**
- **Step 11:** the Gaussian-noise cache also stores the RNG state, so the noise-column test keeps the original random stream; it is a small addition beyond the brief's file list (`stress_noise.joblib` holds a (table, rng_state) tuple).
- **Step 6c:** learning/validation curves clone the tuned pipeline with `rf__n_jobs=1` and `rf__oob_score=False` (the stored estimator has `n_jobs=-1` and OOB on, which would oversubscribe the CPU); DR16/DR17 capped at 200 trees as specified.
- **Step 10b:** `efficiency()` uses `isinstance` checks for the node and iteration counts so only RF and HistGB report them. Pickle size is exact, RSS growth is approximate, single-object latency is dominated by Python overhead.
- **Step 12:** additional columns are filled for tuned-RF and tuned-HistGB rows only; paper and default-RF rows have NaN there.
- `search_DR{14,16,17}.csv` in the project root were rewritten by the Step 6 re-run (same hyper-parameter results; fit/score times differ). The original files were not kept in the repository.
- Python is 3.12.14 (project `.venv`); `TIMINGS` could not be exported from the kernel, so Section 3 uses cell wall times from the runner's progress log.
- Nothing failed; no cell raised an error.
