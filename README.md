# SDSS Stellar Classification: Audit and Corrected Model

- Student Index: KIC-BSCCSAI-001 (Machine Learning)
- Selected Research: Solorio-Ramírez et al. (2023), *Algorithms* 16(6), 293

## Contents

| Path | What it is |
|---|---|
| `baseline_model.ipynb` | Reproduces the paper's pipeline on the real data, as the baseline |
| `fixed_model.ipynb` | Phases 2 and 3: leak-free pipeline, tuned RF, probability metrics, over/underfitting diagnostics, nested CV, ablations, grouped CV, untuned and tuned baselines (HistGB), efficiency, stress tests |
| `search_DR*.csv` | Hyperparameter search results (`cv_results_`) written by `fixed_model.ipynb` |
| `nested_DR*.csv` | Nested cross-validation scores written by `fixed_model.ipynb` (also used as its cache) |
| `outputs/tables/`, `outputs/figures/` | Every exported table (`.csv`) and figure (`.png`) |
| `outputs/run_log.md` | Machine, versions, timings and a factual summary of the latest full run |
| `Datasets/` | SDSS DR14, DR16 and DR17 CSVs (10k / 100k / 100k rows) |

## Setup

Tested with Python 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks from the repository root, since they load data from `Datasets/`.

## Runtime and caching

`baseline_model.ipynb` runs in a few minutes.

`fixed_model.ipynb` caches every heavy step in `cache/` (git-ignored, created on
the first run). Control it with the `RUN_HEAVY` flags in Step A:

- **Fresh clone (no `cache/`):** about 3–3.5 hours on a 6-core / 12-thread laptop
  CPU. Nested CV is not recomputed because `nested_DR*.csv` is in the repository;
  set `RUN_HEAVY['nested'] = True` to recompute it (about 5.5 hours more).
- **With `cache/` present:** a few minutes; only fast fits and plots rerun.

A regression check after Step 6 asserts that the tuned Random Forest reproduces
the original test-set accuracy and macro-F1 to within 1e-4.
