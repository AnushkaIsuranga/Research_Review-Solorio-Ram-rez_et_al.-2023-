# NB6006CEM Phase 1 - reproducibility package

Supports the audit of Solorio-Ramirez et al. (2023), Algorithms 16(6), 293.

## IMPORTANT: what is and is not in here
- **No SDSS dataset was downloaded or used.** The sandbox could not reach Kaggle/SDSS. Every number analysed comes from the paper itself, transcribed into `data/*.csv`.
- The confusion-matrix counts were **read from the paper's figure images by eye** (Figs. 4, 6, 8). Please check them against the PDF; a transcription slip is possible.
- `scripts/verify_metrics.py` is the exact script run during the audit. Its numbers are real calculations on the paper's printed values.
- `scripts/generate_figures.py` makes all 9 figures. Only R3 (class distribution), R7 (confusion matrices) and R8 (Table 3 vs Fig. 7) are computed from paper numbers. R1, R2, R6, R9 are designed diagrams. R4 (feature scales; magnitudes assigned from domain knowledge) and R5 (redshift; synthetic curves) are **illustrative sketches, not data plots**.
- The scripts keep the paper numbers as literals inside the code (as originally run). The CSVs in `data/` hold the same values in readable form for cross-checking.

## Run (tested with Python 3.12.3)
    python -m venv .venv
    # Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/verify_metrics.py                              # prints all audit numbers
    cd figures_output && python ../scripts/generate_figures.py    # writes R1..R9 PNGs here

A virtual environment itself is not shipped (it is OS-specific); requirements.txt pins the exact versions used.

## Where each number in the document comes from
- Tables 4 and 5 (macro vs GALAXY-positive, per-class, MCC, Wilson CI): verify_metrics.py, per-dataset block
- Table 7 expected test counts: "expected 20% per class" lines
- Friedman chi2 = 12.71 (p = 0.0262), p = 0.0305 with corrected DR16 IB3 value, Nemenyi CD 4.35: last blocks of verify_metrics.py
- Combinatorics (97% / 94.7% / 23.5%), bootstrap 0.632, float64 spacing 256: final lines of verify_metrics.py
- Normalised class entropy 0.838 / 0.863 / 0.870: printed by generate_figures.py
- Friedman corrected-value check and Nemenyi CD: scripts/friedman_check.py
