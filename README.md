# Nanozyme Kinetics PIML

Companion repository for the manuscript **“A Physics-Motivated Multi-Output Machine Learning Framework for Predicting Nanozyme Kinetic Parameters: Proof-of-Concept on a Curated Dataset.”**

This repository contains the frozen analysis code, DOI-annotated 86-record dataset, manuscript figures, and supplementary validation tables used in the revised study.

## What this repository demonstrates

The study evaluates a small-data, physics-motivated machine-learning pipeline for predicting three nanozyme kinetic parameters: **Km**, **kcat**, and **Vmax**.

The central methodological result is that conventional shuffled cross-validation gives materially more optimistic estimates than publication-aware validation on this curated literature dataset. The final analysis therefore distinguishes:

- **within-dataset predictive utility** — leakage-controlled random 5-fold CV;
- **cross-publication transfer** — 5-fold `GroupKFold` using reconstructed publication/source identifiers;
- **pooled out-of-fold (OOF) performance** — one strictly out-of-publication prediction per record.

For the frozen 86-record cohort, the full model obtained random-CV R² values of approximately **0.345 (Km), 0.465 (kcat), and 0.321 (Vmax)**. Publication-grouped pooled OOF R² values were approximately **0.074, -0.014, and -0.803**, respectively. These results indicate useful structure within the curated dataset but limited transfer to unseen studies.

The ablation analysis also shows that the engineered representation contributes most strongly for Km, RegressorChain adds little incremental value, and the Arrhenius descriptor provides little additional predictive benefit once catalytic activity class is encoded explicitly.

## Repository structure

```text
nanozyme-kinetics-piml/
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── code/
│   └── PIML_revision_grouped_oof.py
├── data/
│   ├── Database_86_with_DOI.xlsx
│   └── README.md
├── results/
│   └── README.md
├── figures/
│   ├── Figure1_ParityPlots.pdf
│   ├── Figure2_FeatureImportance.pdf
│   └── Figure3_Residuals.pdf
└── supplementary/
    └── CV_tables_final_publication_ready.pdf
```

## Reproducing the analysis

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the full analysis from the repository root:

```bash
python code/PIML_revision_grouped_oof.py \
  --data data/Database_86_with_DOI.xlsx \
  --sheet Matched_86 \
  --out results/full_run
```

The script performs the fixed held-out analysis, leakage-controlled random 5-fold cross-validation, publication-aware `GroupKFold`, pooled OOF prediction export, M1–M4 and A0–A3 ablations, and the known-activity sensitivity analysis.

### Reproducibility safeguards

The final pipeline keeps all data-dependent operations inside training partitions:

- target winsorization is fitted on training targets only;
- validation/test targets remain unmodified;
- imputation and scaling are fitted on training features only;
- activity-specific Arrhenius scaling parameters are selected using training data only;
- publication-aware validation prevents records from the same reconstructed publication group from appearing in both train and validation folds;
- each OOF record is predicted exactly once, with an explicit integrity check.

## Dataset

`data/Database_86_with_DOI.xlsx` contains the same **86 complete records** used in the manuscript, augmented with publication provenance for publication-aware validation. The final cohort contains **23 reconstructed publication groups**.

See [`data/README.md`](data/README.md) for provenance, scope, and reuse notes.

## Results

The repository is intended to hold the frozen summary outputs from the final manuscript run. See [`results/README.md`](results/README.md) for the expected files and upload instructions.

## Important interpretation note

This repository does **not** present a deployment-ready virtual-screening model. Random CV should be interpreted as predictive utility within the curated literature dataset. Publication-grouped CV/OOF is the stricter estimate of cross-study transfer, and it shows substantial publication-level dependence.

The Arrhenius term is retained as a **physics-motivated predictive descriptor**. The present data do not establish that it recovers independent temperature-dependent catalytic energetics.

## License

The **software/code** in this repository is released under the MIT License unless otherwise noted.

The curated dataset is derived from NanozymeDB and source publications and is **not relicensed under MIT**. See `data/README.md` for provenance and attribution guidance.

## Citation

GitHub can generate a citation from [`CITATION.cff`](CITATION.cff). Once the associated manuscript receives a DOI, this file can be updated with the final article citation.

## Contact

Maintained by **NanoBioML** / Daniel Zhakupov.
