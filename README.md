# Nanozyme Kinetics — Physics-Motivated ML

Companion repository for the manuscript **“Physics-Motivated Multi-Output Machine Learning for Nanozyme Kinetics: Predictive Utility and Limits of Cross-Publication Transfer.”**

This project studies how far machine learning can go on a **small, heterogeneous, literature-derived nanozyme kinetic dataset**, and where apparent predictive performance breaks down when validation is separated by publication source.

The repository contains the frozen 86-record dataset, publication provenance, leakage-controlled modeling code, publication-aware validation, reviewer-driven benchmark analyses, ablations, manuscript figures, and machine-readable result tables.

## Research question

Can low-dimensional, physics-motivated descriptors improve prediction of **Km**, **kcat**, and **Vmax** from sparse literature-derived nanozyme data, and do those gains transfer across independent publication sources?

The final study distinguishes:

- **within-dataset predictive utility** — shuffled leakage-controlled 5-fold CV;
- **cross-publication transfer** — 5-fold `GroupKFold` using reconstructed publication/source identifiers.

## Dataset

The frozen analytical cohort contains:

- **86 complete kinetic records**
- **23 reconstructed publication groups**
- targets: **Km, kcat, Vmax**
- experimental descriptors: pH and temperature
- coarse material descriptors: electronegativity and redox potential
- publication provenance for source-aware validation

The data were curated from NanozymeDB and the cited source publications. The software is MIT-licensed; the curated dataset is **not relicensed under MIT**.

## Modeling and engineering

The reference model uses **ExtraTreesRegressor + RegressorChain** with the prespecified target order:

`Km → kcat → Vmax`

The pipeline includes train-only target winsorization, median imputation, robust scaling, physics-motivated feature engineering, activity-dependent Arrhenius transformation, publication-aware `GroupKFold`, pooled OOF prediction export, deterministic leakage checks, paired ablations, conventional-model benchmarking, and feature-group ablation.

The former Vmax post-processing clamp is **not used for primary predictions**; it is retained only as a labeled sensitivity diagnostic.

## Final predictive results

| Target | Shuffled 5-fold R² mean ± SD | Publication-grouped pooled OOF R² |
|---|---:|---:|
| Km | **0.345 ± 0.363** | **0.074** |
| kcat | **0.465 ± 0.106** | **−0.014** |
| Vmax | **0.321 ± 0.116** | **−0.803** |

The central finding is **not** that the model is a deployment-ready screening system. The analysis shows moderate structure within the curated record pool but poor transfer across publication sources.

## Conventional-model benchmark

The final revision compares the reference model against Ridge regression, Random Forest, HistGradientBoosting, and independent-target ExtraTrees using identical outer folds and train-only preprocessing.

The proposed ExtraTrees + RegressorChain architecture was **not consistently superior**:

- HistGradientBoosting with physics-motivated features achieved the best shuffled-CV Km score (**R² ≈ 0.381**).
- Random Forest with the same representation achieved the best shuffled-CV kcat and Vmax scores (**R² ≈ 0.539** and **0.331**).
- No benchmark model transferred robustly across publications for all three targets.

This indicates that the main contribution is better understood as **representation + rigorous validation**, rather than a uniquely superior estimator architecture.

## Feature-group ablation

Keeping the estimator fixed while varying input representation showed that:

- removing **all pH information** reduced shuffled-CV R² by about **0.194 (Km), 0.201 (kcat), and 0.249 (Vmax)**;
- removing engineered temperature descriptors reduced **Km** R² by about **0.524**, with smaller effects on kcat and Vmax;
- the two coarse material descriptors added little measurable predictive value;
- pH- and temperature-related descriptors contain useful within-dataset signal, but these associations do **not** transfer robustly across publications.

The physicochemical interpretation is intentionally narrow: these are **physics-motivated predictive descriptors**, not evidence that the model recovered transferable catalytic mechanisms or activation energetics.

## Repository structure

```text
nanozyme-kinetics-piml/
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── code/
│   ├── PIML_revision_grouped_oof.py
│   └── PIML_round3_reviewer_analysis_FINAL_CLEAN.py
├── data/
│   ├── Database_86_with_DOI.xlsx
│   └── README.md
├── results/
│   ├── README.md
│   └── final_round3/
├── figures/
│   ├── Figure1_ParityPlots.pdf
│   ├── Figure2_FeatureImportance.pdf
│   ├── Figure3_Residuals.pdf
│   └── graphical_abstract_round3.png
└── supplementary/
    ├── CV_tables_final_publication_ready.pdf
    └── Supporting_Information_Round3.pdf
```

## Reproducing the analysis

Python 3.10+ is recommended.

```bash
python -m venv .venv
pip install -r requirements.txt
```

Run the final Round 3 pipeline:

```bash
python code/PIML_round3_reviewer_analysis_FINAL_CLEAN.py   --data data/Database_86_with_DOI.xlsx   --sheet Matched_86
```

Each Round 3 execution writes to a fresh timestamped output directory so older runs are not overwritten.

## Reproducibility safeguards

All data-dependent operations are fitted inside training partitions:

- validation/test targets remain unmodified;
- target winsorization is train-only;
- imputation and scaling are train-only;
- Arrhenius-parameter selection is train-only;
- grouped validation prevents any publication from appearing in both train and validation within an outer fold;
- every OOF record is predicted exactly once by a model that did not train on that record;
- reviewer comparisons use identical outer folds within each analysis.

## Main conclusion

**Physics-motivated features capture useful within-dataset associations, but current sparse literature-derived nanozyme data do not support robust cross-publication kinetic prediction.**

## Citation

GitHub can generate a citation from [`CITATION.cff`](CITATION.cff). Once the associated journal article receives a DOI, the metadata can be updated with the final publication record.

## Contact

Maintained by **NanoBioML / Daniel Zhakupov**.
