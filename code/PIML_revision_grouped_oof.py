#!/usr/bin/env python3
"""
Scientifically rigorous revision of the physics-motivated multi-output nanozyme model.

Primary goals
-------------
1. Reproduce the manuscript architecture (ExtraTrees + RegressorChain).
2. Keep all data-dependent preprocessing train-only.
3. Evaluate against RAW held-out targets (test labels are never winsorized).
4. Tune activity-specific Arrhenius activation energies using training data only.
5. Perform fold-safe cross-validation: imputation, target winsorization, and Ea tuning
   are re-fitted inside every validation fold.
6. Evaluate the principal M1-M4 and A0-A3 comparisons under identical outer folds,
   rather than treating repeated held-out-set analyses as independent confirmation.
7. Support publication-aware GroupKFold validation when a complete stable source ID is available.
8. Report uncertainty with deterministic bootstrap confidence intervals for the fixed held-out set.
9. Remove the heuristic Vmax post-processing clamp from PRIMARY predictions.
   The previous clamp is retained only as a sensitivity diagnostic.
10. Add reference baselines, ablations, and sensitivity analyses:
      - Dummy median predictor
      - Linear regression
      - M1/M2/M3/M4 ablation (raw/physics x independent/chain)
      - Arrhenius vs activity-type categorical encoding
      - Activity-encoding baseline without Arrhenius under the same CV procedure
      - Exclusion of the unresolved ``unknown`` activity class

Input
-----
Primary revision dataset: ``Database_86_with_DOI.xlsx``, sheet ``Matched_86``.
The workbook contains the same 86 complete records used in the manuscript, now
augmented with publication provenance. The stable ``publication_id`` field is
used for publication-aware GroupKFold validation.

Pass an alternative path/sheet if needed:
    python PIML_revision_grouped_oof.py --data /path/to/file.xlsx --sheet Matched_86

Important
---------
This code intentionally prioritizes methodological rigor over reproducing legacy
numbers exactly. If primary metrics differ from an earlier manuscript version,
the revised manuscript should report the metrics produced by THIS final pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# =============================================================================
# REPRODUCIBILITY / CONSTANTS
# =============================================================================

RANDOM_STATE = 42
LOG_FLOOR_TRAIN = 1e-12
LOG_FLOOR_METRICS = 1e-6

TARGETS = ["Km (mM)", "Kcat (s⁻¹)", "Vmax (nM s⁻¹)"]

# Manuscript "raw/base" numerical representation.  This intentionally retains the
# two coarse material descriptors derived from the material name, matching the
# submitted methodology.  Here "raw" means that the pH-zone, inverse-temperature,
# thermal-energy, and Arrhenius transformations have NOT yet been added.
# Do not remove these columns in a revision-only rerun: that would redefine M1/M2.
RAW_COLS = [
    "pH",
    "Temp (°C)",
    "mat_electronegativity",
    "mat_redox_potential",
]

PHYSICS_BASE_COLS = [
    "pH",
    "Temp (°C)",
    "inv_T",
    "RT_eV",
    "pH_pod_zone",
    "pH_cat_zone",
    "pH_oxd_zone",
    "mat_electronegativity",
    "mat_redox_potential",
]

EA_GRID_EV = np.linspace(0.1, 1.5, 15)
EA_DEFAULT_EV = 0.40

# Only identifiers that are plausibly stable at the individual-publication level are
# auto-detected. Broader columns such as journal name or free-text citation are NOT
# guessed automatically; they can be selected explicitly with --group-column.
PUBLICATION_ID_CANDIDATES = [
    "DOI",
    "Publication DOI",
    "Source DOI",
    "Reference DOI",
    "Article DOI",
    "PMID",
    "PubMed ID",
    "PMCID",
    "Paper ID",
    "Publication ID",
    "Reference ID",
    "Source ID",
]


class PhysicsConstants:
    R_eV_per_K = 8.617e-5
    PH_PEROXIDASE = 4.0
    PH_CATALASE = 7.0
    PH_OXIDASE = 9.0

    # Retained only for a sensitivity analysis of the legacy rule.
    LEGACY_E0_MIN_NM = 1e-3
    LEGACY_E0_MAX_NM = 1e6


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def safe_log10(values: np.ndarray, floor: float = LOG_FLOOR_METRICS) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.log10(np.maximum(arr, floor))


def make_extra_trees(
    *,
    n_estimators: int = 300,
    max_depth: int = 15,
    min_samples_leaf: int = 3,
    min_samples_split: int = 6,
) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features="sqrt",
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Required columns are missing from the dataset: "
            + ", ".join(repr(x) for x in missing)
        )


def clean_numeric_columns(df: pd.DataFrame, targets: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in list(targets) + ["Temp (°C)", "pH"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# =============================================================================
# DETERMINISTIC SUBSTRATE / ACTIVITY HANDLING
# =============================================================================

_SUBSTRATE_SYNONYMS = {
    "tmb": ("tmb", "tetramethylbenzidine", "3 3 5 5 tetramethylbenzidine"),
    "abts": ("abts",),
    "opd": ("opd", "o phenylenediamine", "ortho phenylenediamine"),
    "dopa": ("dopa",),
    "h2o2": ("h2o2", "hydrogen peroxide", "peroxide"),
}


def normalize_chemical_text(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "unknown"


def canonicalize_substrate(value: object) -> str:
    """
    Deterministic canonicalization.

    This avoids an environment-dependent SBERT/TF-IDF fallback and therefore
    produces the same mapping on every machine.
    """
    text = normalize_chemical_text(value)
    for canonical, variants in _SUBSTRATE_SYNONYMS.items():
        if any(v in text for v in variants):
            return canonical
    return text


def detect_activity_type(df: pd.DataFrame) -> pd.Series:
    """
    Prefer the explicit 'Enzyme Like Activity' annotation when available.
    Use deterministic substrate heuristics only for unresolved rows.
    """
    activity = pd.Series("unknown", index=df.index, dtype="object")

    if "Enzyme Like Activity" in df.columns:
        raw = df["Enzyme Like Activity"].fillna("").astype(str).str.lower()
        activity.loc[raw.str.contains("peroxid", regex=False)] = "peroxidase"
        activity.loc[raw.str.contains("catal", regex=False)] = "catalase"
        activity.loc[raw.str.contains("oxidase", regex=False)] = "oxidase"

    if "Substrate" in df.columns:
        substrate = df["Substrate"].apply(canonicalize_substrate)
        unresolved = activity.eq("unknown")
        activity.loc[
            unresolved & substrate.isin(["tmb", "abts", "opd", "dopa"])
        ] = "peroxidase"
        activity.loc[unresolved & substrate.eq("h2o2")] = "catalase"

    return activity


# =============================================================================
# PHYSICS-MOTIVATED FEATURE ENGINEERING
# =============================================================================

def prepare_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    require_columns(out, ["Temp (°C)", "pH"])

    out["T_kelvin"] = out["Temp (°C)"] + 273.15
    invalid_t = (
        (out["T_kelvin"] <= 0)
        | (out["Temp (°C)"] < 0)
        | (out["Temp (°C)"] > 100)
    )
    out["inv_T"] = 1.0 / out["T_kelvin"]
    out.loc[invalid_t, "inv_T"] = np.nan
    out["RT_eV"] = PhysicsConstants.R_eV_per_K * out["T_kelvin"]

    out["pH_pod_zone"] = np.exp(
        -((out["pH"] - PhysicsConstants.PH_PEROXIDASE) ** 2) / 1.5
    )
    out["pH_cat_zone"] = np.exp(
        -((out["pH"] - PhysicsConstants.PH_CATALASE) ** 2) / 1.5
    )
    out["pH_oxd_zone"] = np.exp(
        -((out["pH"] - PhysicsConstants.PH_OXIDASE) ** 2) / 1.5
    )

    metals = {
        "au": {"electronegativity": 2.54, "redox_potential": 1.50},
        "pt": {"electronegativity": 2.28, "redox_potential": 1.18},
        "fe": {"electronegativity": 1.83, "redox_potential": -0.44},
        "ag": {"electronegativity": 1.93, "redox_potential": 0.80},
        "cu": {"electronegativity": 1.90, "redox_potential": 0.34},
    }

    if "Name" in out.columns:
        names = out["Name"].fillna("").astype(str).str.lower()

        def lookup(name: str, prop: str) -> float:
            # Deliberately coarse manuscript heuristic: the first recognized
            # elemental keyword in the fixed dictionary order supplies the
            # approximate descriptor.  This is deterministic but is NOT a
            # compositionally complete representation of multi-element materials.
            for metal, props in metals.items():
                if metal in name:
                    return float(props[prop])
            return np.nan

        out["mat_electronegativity"] = names.map(
            lambda x: lookup(x, "electronegativity")
        )
        out["mat_redox_potential"] = names.map(
            lambda x: lookup(x, "redox_potential")
        )
    else:
        out["mat_electronegativity"] = np.nan
        out["mat_redox_potential"] = np.nan

    return out


def add_arrhenius_factor(
    X: pd.DataFrame,
    activity_types: pd.Series,
    ea_by_activity: Mapping[str, float],
) -> pd.DataFrame:
    """
    Classical Arrhenius factor from the manuscript:
        exp[-Ea / (R*T)] = exp[-Ea * (1/T) / R].
    """
    out = X.copy()
    inv_t = out["inv_T"].to_numpy(dtype=float)
    acts = activity_types.astype(str).to_numpy()
    ea_values = np.array(
        [ea_by_activity.get(a, EA_DEFAULT_EV) for a in acts],
        dtype=float,
    )
    out["arrhenius_factor"] = np.exp(
        -ea_values * inv_t / PhysicsConstants.R_eV_per_K
    )
    return out


# =============================================================================
# TRAIN-ONLY TARGET PREPROCESSING
# =============================================================================

@dataclass
class TargetWinsorizer:
    lower_pct: float = 1.0
    upper_pct: float = 99.0
    bounds_: Optional[List[Tuple[float, float]]] = None

    def fit(self, Y_train: np.ndarray) -> "TargetWinsorizer":
        Y = np.asarray(Y_train, dtype=float)
        self.bounds_ = []
        for j in range(Y.shape[1]):
            ylog = np.log10(np.maximum(Y[:, j], LOG_FLOOR_TRAIN))
            finite = np.isfinite(ylog)
            if not finite.any():
                raise ValueError(f"Target column {j} contains no finite values.")
            lo = float(np.percentile(ylog[finite], self.lower_pct))
            hi = float(np.percentile(ylog[finite], self.upper_pct))
            self.bounds_.append((lo, hi))
        return self

    def transform_train(self, Y_train: np.ndarray) -> np.ndarray:
        if self.bounds_ is None:
            raise RuntimeError("TargetWinsorizer must be fitted first.")
        Y = np.asarray(Y_train, dtype=float).copy()
        for j, (lo, hi) in enumerate(self.bounds_):
            ylog = np.log10(np.maximum(Y[:, j], LOG_FLOOR_TRAIN))
            ylog = np.clip(ylog, lo, hi)
            Y[:, j] = 10.0 ** ylog
        return Y


# =============================================================================
# TRAIN-ONLY ACTIVITY-SPECIFIC Ea OPTIMIZATION
# =============================================================================

@dataclass
class EaOptimizationResult:
    ea_by_activity: Dict[str, float]
    best_cv_r2: Dict[str, float]
    diagnostics: Dict[str, Dict[str, object]]


def learn_activity_specific_ea(
    X_train: pd.DataFrame,
    Y_train_processed: np.ndarray,
    activity_train: pd.Series,
    *,
    candidate_grid: np.ndarray = EA_GRID_EV,
    min_samples: int = 10,
    groups_train: Optional[Sequence[object]] = None,
) -> EaOptimizationResult:
    """
    Tune Ea using only the OUTER training data.

    Objective: Vmax in log10 space, simplified ExtraTrees model, 3-fold CV.
    Imputation and scaling are inside the inner-CV Pipeline so every inner fold
    learns preprocessing from its own training portion.

    When publication/source groups are supplied, the inner tuning also uses
    GroupKFold. If an activity subset contains fewer than two distinct groups, Ea
    is not tuned for that activity and the prespecified default is retained.
    """
    if Y_train_processed.shape[1] < 3:
        raise ValueError("Ea optimization expects Vmax as the third target.")

    ea_by_activity: Dict[str, float] = {}
    scores: Dict[str, float] = {}
    diagnostics: Dict[str, Dict[str, object]] = {}

    activity_array = activity_train.astype(str).to_numpy()
    group_array = (
        np.asarray(groups_train, dtype=object)
        if groups_train is not None
        else None
    )
    if group_array is not None and len(group_array) != len(activity_array):
        raise ValueError("groups_train must have the same length as X_train.")

    for activity in sorted(np.unique(activity_array)):
        mask = activity_array == activity
        n = int(mask.sum())
        n_groups: Optional[int] = None

        if group_array is not None:
            n_groups = int(len(np.unique(group_array[mask])))

        if n < min_samples:
            ea_by_activity[activity] = EA_DEFAULT_EV
            scores[activity] = float("nan")
            diagnostics[activity] = {
                "n_training_records": n,
                "n_training_groups": n_groups,
                "inner_cv_kind": None,
                "inner_cv_splits": 0,
                "used_default_Ea": True,
                "selection_reason": f"fewer_than_{min_samples}_training_records",
            }
            continue

        X_sub = X_train.loc[mask, PHYSICS_BASE_COLS].reset_index(drop=True)
        a_sub = pd.Series(activity_array[mask])
        y_sub = safe_log10(Y_train_processed[mask, 2])

        cv_groups = None
        if group_array is not None:
            cv_groups = group_array[mask]
            if n_groups is None or n_groups < 2:
                ea_by_activity[activity] = EA_DEFAULT_EV
                scores[activity] = float("nan")
                diagnostics[activity] = {
                    "n_training_records": n,
                    "n_training_groups": n_groups,
                    "inner_cv_kind": "GroupKFold",
                    "inner_cv_splits": 0,
                    "used_default_Ea": True,
                    "selection_reason": "fewer_than_2_publication_groups",
                }
                continue
            inner_splits = min(3, n_groups)
            cv = GroupKFold(n_splits=inner_splits)
            inner_cv_kind = "GroupKFold"
        else:
            inner_splits = min(3, n)
            cv = KFold(
                n_splits=inner_splits,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
            inner_cv_kind = "KFold"

        best_ea = EA_DEFAULT_EV
        best_score = -np.inf

        for ea in candidate_grid:
            candidate = add_arrhenius_factor(
                X_sub,
                a_sub,
                {activity: float(ea)},
            )

            pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=50,
                            max_depth=8,
                            max_features="sqrt",
                            min_samples_leaf=3,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

            fold_scores = cross_val_score(
                pipeline,
                candidate,
                y_sub,
                cv=cv,
                groups=cv_groups,
                scoring="r2",
                error_score=np.nan,
            )
            score = float(np.nanmean(fold_scores))

            if np.isfinite(score) and score > best_score:
                best_score = score
                best_ea = float(ea)

        finite_score = bool(np.isfinite(best_score))
        ea_by_activity[activity] = best_ea
        scores[activity] = best_score if finite_score else float("nan")
        diagnostics[activity] = {
            "n_training_records": n,
            "n_training_groups": n_groups,
            "inner_cv_kind": inner_cv_kind,
            "inner_cv_splits": int(inner_splits),
            "used_default_Ea": bool((not finite_score) and best_ea == EA_DEFAULT_EV),
            "selection_reason": (
                "grid_search_best_inner_cv_R2"
                if finite_score
                else "no_finite_inner_cv_R2_default_retained"
            ),
        }

    if "unknown" not in ea_by_activity:
        ea_by_activity["unknown"] = EA_DEFAULT_EV
        scores["unknown"] = float("nan")
        diagnostics["unknown"] = {
            "n_training_records": 0,
            "n_training_groups": 0 if group_array is not None else None,
            "inner_cv_kind": None,
            "inner_cv_splits": 0,
            "used_default_Ea": True,
            "selection_reason": "class_absent_from_training_partition",
        }

    return EaOptimizationResult(ea_by_activity, scores, diagnostics)


# =============================================================================
# MODEL
# =============================================================================

class NanozymeRegressor:
    """
    Configurable ExtraTrees model used for the primary analysis and ablations.
    """

    def __init__(
        self,
        *,
        use_physics: bool,
        use_arrhenius: bool,
        use_activity_onehot: bool,
        use_chain: bool,
        ea_by_activity: Optional[Mapping[str, float]] = None,
    ):
        if use_arrhenius and not use_physics:
            raise ValueError("Arrhenius factor requires physics features.")

        self.use_physics = use_physics
        self.use_arrhenius = use_arrhenius
        self.use_activity_onehot = use_activity_onehot
        self.use_chain = use_chain
        self.ea_by_activity = dict(
            ea_by_activity or {"unknown": EA_DEFAULT_EV}
        )

        self.numeric_cols = PHYSICS_BASE_COLS if use_physics else RAW_COLS
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.activity_encoder: Optional[OneHotEncoder] = None
        self.estimator = None
        self.feature_names_: List[str] = []
        self.target_names_: List[str] = []

    def _numeric_design(
        self,
        X: pd.DataFrame,
        activity: pd.Series,
    ) -> pd.DataFrame:
        Xn = X[self.numeric_cols].copy()
        if self.use_arrhenius:
            Xn = add_arrhenius_factor(
                Xn,
                activity,
                self.ea_by_activity,
            )
        return Xn

    def _fit_design(
        self,
        X: pd.DataFrame,
        activity: pd.Series,
    ) -> np.ndarray:
        Xn = self._numeric_design(X, activity)
        self.feature_names_ = list(Xn.columns)

        Xi = self.imputer.fit_transform(Xn)
        Xs = self.scaler.fit_transform(Xi)

        if self.use_activity_onehot:
            self.activity_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=float,
            )
            A = self.activity_encoder.fit_transform(
                activity.astype(str).to_numpy().reshape(-1, 1)
            )
            categories = self.activity_encoder.categories_[0]
            self.feature_names_.extend(
                [f"activity_{x}" for x in categories]
            )
            Xs = np.column_stack([Xs, A])

        return Xs

    def _transform_design(
        self,
        X: pd.DataFrame,
        activity: pd.Series,
    ) -> np.ndarray:
        Xn = self._numeric_design(X, activity)
        Xi = self.imputer.transform(Xn)
        Xs = self.scaler.transform(Xi)

        if self.use_activity_onehot:
            if self.activity_encoder is None:
                raise RuntimeError("Activity encoder has not been fitted.")
            A = self.activity_encoder.transform(
                activity.astype(str).to_numpy().reshape(-1, 1)
            )
            Xs = np.column_stack([Xs, A])

        return Xs

    def fit(
        self,
        X: pd.DataFrame,
        Y_train_processed: np.ndarray,
        activity: pd.Series,
        target_names: Sequence[str],
    ) -> "NanozymeRegressor":
        self.target_names_ = list(target_names)

        Xd = self._fit_design(X, activity)
        Ylog = safe_log10(Y_train_processed)

        base = make_extra_trees()

        if self.use_chain:
            # Fixed a priori to match the manuscript: Km -> Kcat -> Vmax.
            # The order is not validation-tuned and is not given a causal meaning;
            # the M3-vs-M4 ablation tests whether chaining adds predictive value.
            self.estimator = RegressorChain(
                base,
                order=list(range(Ylog.shape[1])),
                random_state=RANDOM_STATE,
            )
        else:
            self.estimator = MultiOutputRegressor(
                base,
                n_jobs=-1,
            )

        self.estimator.fit(Xd, Ylog)
        return self

    def predict(
        self,
        X: pd.DataFrame,
        activity: pd.Series,
    ) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model must be fitted before prediction.")
        Xd = self._transform_design(X, activity)
        return 10.0 ** self.estimator.predict(Xd)

    def final_target_feature_importance(
        self,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Impurity-based feature importance for the final target estimator.

        This is descriptive, not causal.
        """
        if self.estimator is None:
            raise RuntimeError("Model must be fitted first.")

        if self.use_chain:
            final_est = self.estimator.estimators_[-1]
            names = list(self.feature_names_)
            names.extend(
                [f"pred_{t}" for t in self.target_names_[:-1]]
            )
            values = np.asarray(
                final_est.feature_importances_,
                dtype=float,
            )
        else:
            final_est = self.estimator.estimators_[-1]
            names = list(self.feature_names_)
            values = np.asarray(
                final_est.feature_importances_,
                dtype=float,
            )

        if len(names) != len(values):
            raise RuntimeError(
                f"Feature-name mismatch: {len(names)} vs {len(values)}."
            )
        return names, values


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class MetricResult:
    r2: float
    rmse: float
    mae: float
    within_2x: float
    within_5x: float
    ci_low: float
    ci_high: float


def bootstrap_r2_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
) -> Tuple[float, float]:
    yt = safe_log10(y_true)
    yp = safe_log10(y_pred)
    rng = np.random.default_rng(seed)

    boot: List[float] = []
    n = len(yt)

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)

        # R2 is undefined for a constant resampled target.
        if np.var(yt[idx]) <= 1e-15:
            continue

        score = r2_score(yt[idx], yp[idx])
        if np.isfinite(score):
            boot.append(float(score))

    if len(boot) < max(100, n_boot // 10):
        return float("nan"), float("nan")

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_targets(
    Y_true_raw: np.ndarray,
    Y_pred: np.ndarray,
    target_names: Sequence[str],
) -> Dict[str, MetricResult]:
    results: Dict[str, MetricResult] = {}

    for j, target in enumerate(target_names):
        yt = np.asarray(Y_true_raw[:, j], dtype=float)
        yp = np.asarray(Y_pred[:, j], dtype=float)

        lt = safe_log10(yt)
        lp = safe_log10(yp)

        ratio = np.maximum(
            yp, LOG_FLOOR_METRICS
        ) / np.maximum(yt, LOG_FLOOR_METRICS)

        ci_lo, ci_hi = bootstrap_r2_ci(
            yt,
            yp,
            seed=RANDOM_STATE + j,
        )

        results[target] = MetricResult(
            r2=float(r2_score(lt, lp)),
            rmse=float(np.sqrt(mean_squared_error(lt, lp))),
            mae=float(mean_absolute_error(lt, lp)),
            within_2x=float(
                np.mean((ratio > 0.5) & (ratio < 2.0)) * 100.0
            ),
            within_5x=float(
                np.mean((ratio > 0.2) & (ratio < 5.0)) * 100.0
            ),
            ci_low=ci_lo,
            ci_high=ci_hi,
        )

    return results


def print_metric_block(
    title: str,
    results: Mapping[str, MetricResult],
) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

    for target, m in results.items():
        print(
            f"{target:<24} "
            f"R²={m.r2:+.4f}  "
            f"95% CI=[{m.ci_low:+.4f}, {m.ci_high:+.4f}]  "
            f"RMSE={m.rmse:.4f}  "
            f"MAE={m.mae:.4f}  "
            f"W2x={m.within_2x:.1f}%  "
            f"W5x={m.within_5x:.1f}%"
        )


# =============================================================================
# SIMPLE REFERENCE BASELINES
# =============================================================================

def fit_raw_feature_matrix(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    train_i = imputer.fit_transform(X_train[RAW_COLS])
    test_i = imputer.transform(X_test[RAW_COLS])

    return (
        scaler.fit_transform(train_i),
        scaler.transform(test_i),
    )


def evaluate_simple_baselines(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train_processed: np.ndarray,
    Y_test_raw: np.ndarray,
    target_names: Sequence[str],
) -> Dict[str, Dict[str, MetricResult]]:
    Xtr, Xte = fit_raw_feature_matrix(X_train, X_test)
    Ylog = safe_log10(Y_train_processed)

    outputs: Dict[str, Dict[str, MetricResult]] = {}

    dummy = MultiOutputRegressor(
        DummyRegressor(strategy="median")
    )
    dummy.fit(Xtr, Ylog)
    outputs["DummyMedian"] = evaluate_targets(
        Y_test_raw,
        10.0 ** dummy.predict(Xte),
        target_names,
    )

    linear = MultiOutputRegressor(LinearRegression())
    linear.fit(Xtr, Ylog)
    outputs["LinearRegression"] = evaluate_targets(
        Y_test_raw,
        10.0 ** linear.predict(Xte),
        target_names,
    )

    return outputs


# =============================================================================
# LEAKAGE-FREE OUTER CROSS-VALIDATION
# =============================================================================


def _fold_metric_rows(
    model_label: str,
    fold: int,
    Y_true_raw: np.ndarray,
    Y_pred: np.ndarray,
    target_names: Sequence[str],
) -> List[Dict[str, float]]:
    """Return fold-level metrics without bootstrap resampling."""
    rows: List[Dict[str, float]] = []
    for j, target in enumerate(target_names):
        yt = np.asarray(Y_true_raw[:, j], dtype=float)
        yp = np.asarray(Y_pred[:, j], dtype=float)
        lt = safe_log10(yt)
        lp = safe_log10(yp)
        ratio = np.maximum(yp, LOG_FLOOR_METRICS) / np.maximum(
            yt, LOG_FLOOR_METRICS
        )
        r2 = (
            float(r2_score(lt, lp))
            if np.var(lt) > 1e-15
            else float("nan")
        )
        rows.append(
            {
                "model": model_label,
                "fold": int(fold),
                "target": str(target),
                "R2": r2,
                "RMSE_log10": float(np.sqrt(mean_squared_error(lt, lp))),
                "MAE_log10": float(mean_absolute_error(lt, lp)),
                "within_2x_pct": float(
                    np.mean((ratio > 0.5) & (ratio < 2.0)) * 100.0
                ),
                "within_5x_pct": float(
                    np.mean((ratio > 0.2) & (ratio < 5.0)) * 100.0
                ),
            }
        )
    return rows


def summarize_cv_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize outer-fold performance; SD uses ddof=0 to match legacy reporting."""
    rows = []
    for (model, target), group in fold_df.groupby(["model", "target"], sort=False):
        rows.append(
            {
                "model": model,
                "target": target,
                "n_folds": int(group["fold"].nunique()),
                "mean_R2": float(np.nanmean(group["R2"])),
                "std_R2": float(np.nanstd(group["R2"], ddof=0)),
                "mean_RMSE_log10": float(np.nanmean(group["RMSE_log10"])),
                "std_RMSE_log10": float(np.nanstd(group["RMSE_log10"], ddof=0)),
                "mean_MAE_log10": float(np.nanmean(group["MAE_log10"])),
                "mean_within_2x_pct": float(np.nanmean(group["within_2x_pct"])),
                "mean_within_5x_pct": float(np.nanmean(group["within_5x_pct"])),
            }
        )
    return pd.DataFrame(rows)


def paired_cv_deltas(fold_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute paired fold-wise R2 differences for the analysis comparisons.

    Positive deltas mean the model named second outperformed the model named first
    on the same outer fold.
    """
    comparisons = [
        ("M1_raw_independent", "M3_physics_independent", "M3_minus_M1_engineered_representation"),
        ("M3_physics_independent", "M4_full_physics_chain", "M4_minus_M3_chain"),
        ("A0_physics_no_arrhenius", "A1_physics_plus_activity_category", "A1_minus_A0_activity_category"),
        ("A0_physics_no_arrhenius", "A2_physics_plus_arrhenius", "A2_minus_A0_arrhenius"),
        ("A1_physics_plus_activity_category", "A3_physics_plus_activity_and_arrhenius", "A3_minus_A1_arrhenius_beyond_activity"),
    ]

    delta_rows = []
    for base, enhanced, label in comparisons:
        left = fold_df.loc[
            fold_df["model"].eq(base),
            ["fold", "target", "R2"],
        ].rename(columns={"R2": "R2_base"})
        right = fold_df.loc[
            fold_df["model"].eq(enhanced),
            ["fold", "target", "R2"],
        ].rename(columns={"R2": "R2_enhanced"})
        merged = left.merge(right, on=["fold", "target"], how="inner")
        for _, row in merged.iterrows():
            delta_rows.append(
                {
                    "comparison": label,
                    "base_model": base,
                    "enhanced_model": enhanced,
                    "fold": int(row["fold"]),
                    "target": row["target"],
                    "delta_R2": float(row["R2_enhanced"] - row["R2_base"]),
                }
            )

    delta_df = pd.DataFrame(delta_rows)
    summary_rows = []
    if not delta_df.empty:
        for (comparison, target), group in delta_df.groupby(
            ["comparison", "target"], sort=False
        ):
            summary_rows.append(
                {
                    "comparison": comparison,
                    "target": target,
                    "n_folds": int(group["fold"].nunique()),
                    "mean_delta_R2": float(np.nanmean(group["delta_R2"])),
                    "std_delta_R2": float(np.nanstd(group["delta_R2"], ddof=0)),
                }
            )
    return delta_df, pd.DataFrame(summary_rows)


def summarize_oof_predictions(oof_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize pooled out-of-fold predictions across all outer validation folds.

    These pooled metrics are complementary to the mean +/- SD of fold-wise metrics:
    every record is predicted exactly once by a model that did not train on that
    record (and, for grouped CV, did not train on any record from the same
    publication).  Pooled R2 is computed in log10 target space, matching the
    manuscript evaluation convention.
    """
    if oof_df.empty:
        return pd.DataFrame()

    rows = []
    for (analysis, model, target), group in oof_df.groupby(
        ["analysis", "model", "target"], sort=False
    ):
        yt = group["y_true"].to_numpy(dtype=float)
        yp = group["y_pred"].to_numpy(dtype=float)
        lt = safe_log10(yt)
        lp = safe_log10(yp)
        ratio = np.maximum(yp, LOG_FLOOR_METRICS) / np.maximum(
            yt, LOG_FLOOR_METRICS
        )

        r2 = (
            float(r2_score(lt, lp))
            if np.var(lt) > 1e-15
            else float("nan")
        )

        publication_count = None
        if "publication_group" in group.columns:
            known_groups = group["publication_group"].dropna().astype(str)
            if len(known_groups):
                publication_count = int(known_groups.nunique())

        rows.append(
            {
                "analysis": analysis,
                "model": model,
                "target": target,
                "n_samples": int(len(group)),
                "n_publications": publication_count,
                "pooled_OOF_R2": r2,
                "pooled_OOF_RMSE_log10": float(
                    np.sqrt(mean_squared_error(lt, lp))
                ),
                "pooled_OOF_MAE_log10": float(mean_absolute_error(lt, lp)),
                "pooled_OOF_within_2x_pct": float(
                    np.mean((ratio > 0.5) & (ratio < 2.0)) * 100.0
                ),
                "pooled_OOF_within_5x_pct": float(
                    np.mean((ratio > 0.2) & (ratio < 5.0)) * 100.0
                ),
            }
        )

    return pd.DataFrame(rows)


def cross_validate_model_specs(
    X_all: pd.DataFrame,
    Y_all_raw: np.ndarray,
    activity_all: pd.Series,
    target_names: Sequence[str],
    specs: Sequence["ModelSpec"],
    *,
    n_splits: int = 5,
    groups: Optional[Sequence[object]] = None,
    sample_ids: Optional[Sequence[object]] = None,
    analysis_label: str = "random_5fold",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Evaluate multiple model specifications on IDENTICAL outer folds.

    In every outer fold:
      - target winsorization is fitted on fold-train only;
      - Ea is retuned on fold-train only when required;
      - imputer/scaler/one-hot encoder are fitted on fold-train only;
      - validation labels remain raw;
      - all model specifications use the same fold assignment.

    If ``groups`` are supplied, GroupKFold is used in both the outer split and,
    where feasible, the inner Ea-tuning procedure.

    The function also returns record-level out-of-fold predictions.  Thus every
    sample has exactly one prediction from an outer-fold model that did not train
    on that sample.  Under GroupKFold, the training fold also contains no records
    from the validation sample's publication group.
    """
    if len(X_all) != len(Y_all_raw) or len(X_all) != len(activity_all):
        raise ValueError("X, Y, and activity arrays must have equal length.")

    if sample_ids is None:
        sample_id_array = np.asarray(X_all.index.to_numpy(), dtype=object)
    else:
        sample_id_array = np.asarray(sample_ids, dtype=object)
        if len(sample_id_array) != len(X_all):
            raise ValueError("sample_ids must have the same length as X_all.")

    group_array = None
    if groups is not None:
        group_array = np.asarray(groups, dtype=object)
        if len(group_array) != len(X_all):
            raise ValueError("groups must have the same length as X_all.")
        n_groups = len(np.unique(group_array))
        if n_groups < 2:
            raise ValueError("Grouped CV requires at least two distinct groups.")
        actual_splits = min(n_splits, n_groups)
        splitter = GroupKFold(n_splits=actual_splits)
        split_indices = list(
            splitter.split(X_all, Y_all_raw, groups=group_array)
        )
        split_kind = "GroupKFold"
    else:
        actual_splits = n_splits
        splitter = KFold(
            n_splits=actual_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        split_indices = list(splitter.split(X_all))
        split_kind = "KFold"

    fold_rows: List[Dict[str, float]] = []
    oof_rows: List[Dict[str, object]] = []
    ea_records = []
    needs_arrhenius = any(spec.use_arrhenius for spec in specs)

    for fold, (tr_idx, va_idx) in enumerate(split_indices, start=1):
        Xtr = X_all.iloc[tr_idx].copy()
        Xva = X_all.iloc[va_idx].copy()
        Ytr_raw = Y_all_raw[tr_idx].copy()
        Yva_raw = Y_all_raw[va_idx].copy()
        atr = activity_all.iloc[tr_idx].copy()
        ava = activity_all.iloc[va_idx].copy()
        gtr = group_array[tr_idx] if group_array is not None else None
        gva = group_array[va_idx] if group_array is not None else None

        if group_array is not None:
            overlap = set(gtr).intersection(set(gva))
            if overlap:
                raise RuntimeError(
                    "Publication leakage detected between outer train and validation "
                    f"folds: {sorted(overlap)!r}"
                )

        winsor = TargetWinsorizer().fit(Ytr_raw)
        Ytr = winsor.transform_train(Ytr_raw)

        if needs_arrhenius:
            ea_result = learn_activity_specific_ea(
                Xtr,
                Ytr,
                atr,
                groups_train=gtr,
            )
            ea_map = ea_result.ea_by_activity
            for activity, ea in sorted(ea_map.items()):
                diag = ea_result.diagnostics.get(activity, {})
                ea_records.append(
                    {
                        "analysis": analysis_label,
                        "fold": int(fold),
                        "activity": activity,
                        "Ea_eV": float(ea),
                        "inner_cv_R2": float(
                            ea_result.best_cv_r2.get(activity, float("nan"))
                        ),
                        "n_training_records": diag.get("n_training_records"),
                        "n_training_groups": diag.get("n_training_groups"),
                        "inner_cv_kind": diag.get("inner_cv_kind"),
                        "inner_cv_splits": diag.get("inner_cv_splits"),
                        "used_default_Ea": diag.get("used_default_Ea"),
                        "selection_reason": diag.get("selection_reason"),
                    }
                )
        else:
            ea_map = {"unknown": EA_DEFAULT_EV}

        for spec in specs:
            model = NanozymeRegressor(
                use_physics=spec.use_physics,
                use_arrhenius=spec.use_arrhenius,
                use_activity_onehot=spec.use_activity_onehot,
                use_chain=spec.use_chain,
                ea_by_activity=ea_map,
            )
            model.fit(Xtr, Ytr, atr, target_names)
            pred = model.predict(Xva, ava)
            fold_rows.extend(
                _fold_metric_rows(
                    spec.label,
                    fold,
                    Yva_raw,
                    pred,
                    target_names,
                )
            )

            for local_i, global_i in enumerate(va_idx):
                publication_group = (
                    str(group_array[global_i])
                    if group_array is not None
                    else None
                )
                for j, target in enumerate(target_names):
                    y_true = float(Yva_raw[local_i, j])
                    y_pred = float(pred[local_i, j])
                    log_true = float(safe_log10(np.array([y_true]))[0])
                    log_pred = float(safe_log10(np.array([y_pred]))[0])
                    oof_rows.append(
                        {
                            "analysis": analysis_label,
                            "model": spec.label,
                            "fold": int(fold),
                            "sample_id": str(sample_id_array[global_i]),
                            "source_index": str(X_all.index[global_i]),
                            "publication_group": publication_group,
                            "activity_type": str(activity_all.iloc[global_i]),
                            "target": str(target),
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "log10_true": log_true,
                            "log10_pred": log_pred,
                            "log10_error_pred_minus_true": log_pred - log_true,
                            "abs_log10_error": abs(log_pred - log_true),
                            "prediction_to_true_ratio": float(
                                max(y_pred, LOG_FLOOR_METRICS)
                                / max(y_true, LOG_FLOOR_METRICS)
                            ),
                        }
                    )

        print(
            f"{analysis_label}: fold {fold}/{actual_splits} complete; "
            f"validation n={len(va_idx)}"
        )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = summarize_cv_folds(fold_df)
    oof_df = pd.DataFrame(oof_rows)

    expected_oof_rows = int(len(specs) * len(X_all) * len(target_names))
    if len(oof_df) != expected_oof_rows:
        raise RuntimeError(
            f"OOF integrity check failed: got {len(oof_df)} rows, "
            f"expected {expected_oof_rows}."
        )
    counts = oof_df.groupby(["model", "sample_id", "target"]).size()
    if not bool((counts == 1).all()):
        raise RuntimeError(
            "OOF integrity check failed: every model/sample/target must be "
            "predicted exactly once."
        )

    metadata = {
        "analysis_label": analysis_label,
        "splitter": split_kind,
        "n_splits": int(actual_splits),
        "n_samples": int(len(X_all)),
        "n_groups": (
            int(len(np.unique(group_array)))
            if group_array is not None
            else None
        ),
        "ea_tuning_records": ea_records,
        "oof_prediction_rows": int(len(oof_df)),
        "oof_integrity_checked": True,
    }
    return fold_df, summary_df, oof_df, metadata



# =============================================================================
# MODEL ABLATIONS
# =============================================================================


@dataclass(frozen=True)
class ModelSpec:
    label: str
    use_physics: bool
    use_arrhenius: bool
    use_activity_onehot: bool
    use_chain: bool


def model_ablation_specs() -> List[ModelSpec]:
    """All M1-M4 and A0-A3 specifications used in the ablation analysis."""
    return [
        # Main M1-M4 ablation
        ModelSpec(
            "M1_raw_independent",
            False, False, False, False,
        ),
        ModelSpec(
            "M2_raw_chain",
            False, False, False, True,
        ),
        ModelSpec(
            "M3_physics_independent",
            True, True, False, False,
        ),
        ModelSpec(
            "M4_full_physics_chain",
            True, True, False, True,
        ),

        # Direct Arrhenius-vs-category analysis
        ModelSpec(
            "A0_physics_no_arrhenius",
            True, False, False, True,
        ),
        ModelSpec(
            "A1_physics_plus_activity_category",
            True, False, True, True,
        ),
        ModelSpec(
            "A2_physics_plus_arrhenius",
            True, True, False, True,
        ),
        ModelSpec(
            "A3_physics_plus_activity_and_arrhenius",
            True, True, True, True,
        ),
    ]


def arrhenius_activity_specs() -> List[ModelSpec]:
    """A0-A3 only; used for the unknown-activity sensitivity analysis."""
    return [spec for spec in model_ablation_specs() if spec.label.startswith("A")]


def run_model_ablations(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train_processed: np.ndarray,
    Y_test_raw: np.ndarray,
    activity_train: pd.Series,
    activity_test: pd.Series,
    target_names: Sequence[str],
    ea_by_activity: Mapping[str, float],
) -> Dict[str, Dict[str, MetricResult]]:
    """
    Legacy fixed-test-set ablations retained only as an exploratory diagnostic.

    Principal analysis comparisons are produced by
    ``cross_validate_model_specs`` on identical outer folds.
    """
    results: Dict[str, Dict[str, MetricResult]] = {}

    for spec in model_ablation_specs():
        model = NanozymeRegressor(
            use_physics=spec.use_physics,
            use_arrhenius=spec.use_arrhenius,
            use_activity_onehot=spec.use_activity_onehot,
            use_chain=spec.use_chain,
            ea_by_activity=ea_by_activity,
        )
        model.fit(
            X_train,
            Y_train_processed,
            activity_train,
            target_names,
        )
        pred = model.predict(
            X_test,
            activity_test,
        )
        results[spec.label] = evaluate_targets(
            Y_test_raw,
            pred,
            target_names,
        )

    return results


# =============================================================================
# PUBLICATION/SOURCE GROUP AUDIT
# =============================================================================


def resolve_group_column(
    df: pd.DataFrame,
    requested: Optional[str] = None,
) -> Optional[str]:
    """Find a stable publication/source-ID column without guessing from free text."""
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"Requested group column {requested!r} is not present in the dataset."
            )
        return requested

    casefold_map = {str(col).casefold(): str(col) for col in df.columns}
    for candidate in PUBLICATION_ID_CANDIDATES:
        match = casefold_map.get(candidate.casefold())
        if match is not None:
            return match
    return None


def publication_group_audit(
    df: pd.DataFrame,
    group_column: str,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, object]]:
    """Normalize source IDs and quantify repeated/missing publication identifiers."""
    raw = df[group_column]
    groups = raw.map(
        lambda value: (
            None
            if pd.isna(value) or not str(value).strip()
            else str(value).strip().lower()
        )
    )
    known = groups.dropna()
    counts = (
        known.value_counts()
        .rename_axis("publication_group")
        .reset_index(name="record_count")
    )
    n_known = int(len(known))
    n_unique = int(known.nunique())
    audit = {
        "group_column": group_column,
        "n_records": int(len(df)),
        "n_with_group_id": n_known,
        "n_missing_group_id": int(groups.isna().sum()),
        "n_unique_known_groups": n_unique,
        "n_multi_record_groups": int((counts["record_count"] > 1).sum()) if not counts.empty else 0,
        "n_records_in_multi_record_groups": int(
            counts.loc[counts["record_count"] > 1, "record_count"].sum()
        ) if not counts.empty else 0,
        "max_group_size": int(counts["record_count"].max()) if not counts.empty else 0,
        "grouped_cv_eligible": bool(len(df) > 0 and groups.notna().all() and n_unique >= 2),
    }
    return groups, counts, audit


# =============================================================================
# LEGACY Vmax CLAMP SENSITIVITY — NOT USED FOR PRIMARY METRICS
# =============================================================================

def legacy_vmax_clamp_sensitivity(
    Y_pred_direct: np.ndarray,
) -> Dict[str, float]:
    """
    Diagnose the legacy Vmax/kcat concentration clamp without applying it.

    Because Vmax is expressed in nM s^-1 and kcat in s^-1, their ratio has
    units of nM.  This diagnostic is retained only for manuscript continuity;
    it has no role in fitting, scoring, or primary prediction generation.
    """
    pred = np.asarray(
        Y_pred_direct,
        dtype=float,
    ).copy()

    if pred.ndim < 2 or pred.shape[1] < 3:
        return {
            "changed_count": 0.0,
            "total_count": float(len(pred)),
            "max_relative_change": 0.0,
            "mean_relative_change": 0.0,
        }

    kcat = pred[:, 1]
    vmax = pred[:, 2]
    implied_e0 = vmax / (kcat + 1e-12)
    clipped_e0 = np.clip(
        implied_e0,
        PhysicsConstants.LEGACY_E0_MIN_NM,
        PhysicsConstants.LEGACY_E0_MAX_NM,
    )
    corrected = kcat * clipped_e0

    changed = ~np.isclose(
        vmax,
        corrected,
        rtol=1e-10,
        atol=1e-12,
    )
    relative = np.abs(
        corrected - vmax
    ) / np.maximum(np.abs(vmax), 1e-12)

    return {
        "changed_count": float(changed.sum()),
        "total_count": float(len(vmax)),
        "max_relative_change": (
            float(np.max(relative)) if len(relative) else 0.0
        ),
        "mean_relative_change": (
            float(np.mean(relative)) if len(relative) else 0.0
        ),
    }


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def results_to_dataframe(
    results: Mapping[str, Mapping[str, MetricResult]]
) -> pd.DataFrame:
    rows = []

    for model_name, target_results in results.items():
        for target, m in target_results.items():
            rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "R2": m.r2,
                    "CI_low": m.ci_low,
                    "CI_high": m.ci_high,
                    "RMSE_log10": m.rmse,
                    "MAE_log10": m.mae,
                    "within_2x_pct": m.within_2x,
                    "within_5x_pct": m.within_5x,
                }
            )

    return pd.DataFrame(rows)


def save_primary_figures(
    out_dir: Path,
    model: NanozymeRegressor,
    Y_test_raw: np.ndarray,
    Y_pred: np.ndarray,
    target_names: Sequence[str],
    metrics: Mapping[str, MetricResult],
) -> None:
    labels = [
        r"$K_m$ (mM)",
        r"$k_{cat}$ (s$^{-1}$)",
        r"$V_{max}$ (nM s$^{-1}$)",
    ]

    # Figure 1: parity plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for j, (ax, target, label) in enumerate(
        zip(axes, target_names, labels)
    ):
        yt = safe_log10(Y_test_raw[:, j])
        yp = safe_log10(Y_pred[:, j])

        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))

        ax.scatter(
            yt,
            yp,
            alpha=0.75,
            s=55,
            edgecolors="black",
            linewidths=0.6,
        )
        ax.plot(
            [lo, hi],
            [lo, hi],
            linestyle="--",
            linewidth=1.5,
            label="1:1 line",
        )
        ax.fill_between(
            [lo, hi],
            [
                lo - math.log10(2),
                hi - math.log10(2),
            ],
            [
                lo + math.log10(2),
                hi + math.log10(2),
            ],
            alpha=0.15,
            label="2× error band",
        )

        ax.set_xlabel(
            f"Experimental log10({label})"
        )
        ax.set_ylabel(
            f"Predicted log10({label})"
        )
        ax.set_title(
            f"{label}\nR² = {metrics[target].r2:.3f}"
        )
        ax.legend()
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        out_dir / "Figure1_ParityPlots.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "Figure1_ParityPlots.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 2: feature importance
    names, importance = (
        model.final_target_feature_importance()
    )
    order = np.argsort(
        importance
    )[::-1][:10]

    replacements = {
        "arrhenius_factor": "Arrhenius factor",
        "pH": "pH",
        "pH_pod_zone": "pH (peroxidase zone)",
        "pH_cat_zone": "pH (catalase zone)",
        "pH_oxd_zone": "pH (oxidase zone)",
        "inv_T": "Inverse temperature (1/T)",
        "RT_eV": "Thermal energy (RT)",
        "mat_electronegativity": "Electronegativity",
        "mat_redox_potential": "Redox potential",
    }

    display = [
        replacements.get(
            str(name),
            str(name).replace("_", " "),
        )
        for name in np.array(
            names,
            dtype=object,
        )[order]
    ]
    values = importance[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        np.arange(len(values)),
        values,
    )
    ax.set_yticks(
        np.arange(len(values))
    )
    ax.set_yticklabels(display)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Impurity-based feature importance"
    )
    ax.set_title(
        r"Top-10 feature importances for $V_{max}$ prediction"
    )
    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        out_dir / "Figure2_FeatureImportance.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "Figure2_FeatureImportance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 3: residual distributions
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
    )

    for j, (ax, label) in enumerate(
        zip(axes, labels)
    ):
        residuals = (
            safe_log10(Y_test_raw[:, j])
            - safe_log10(Y_pred[:, j])
        )

        ax.hist(
            residuals,
            bins=min(
                12,
                max(5, len(residuals) // 2),
            ),
            edgecolor="black",
            alpha=0.75,
        )
        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=1.5,
            label="Zero residual",
        )
        ax.axvline(
            float(np.mean(residuals)),
            linestyle=":",
            linewidth=1.5,
            label=f"Mean = {np.mean(residuals):.3f}",
        )
        ax.set_xlabel(
            "Residual (log10 true - log10 predicted)"
        )
        ax.set_ylabel("Frequency")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        out_dir / "Figure3_Residuals.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "Figure3_Residuals.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Rigorous physics-motivated "
            "nanozyme kinetic model"
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir / "Database_86_with_DOI.xlsx",
        help=(
            "Path to Database_86_with_DOI.xlsx "
            "(default: next to this script)."
        ),
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="Matched_86",
        help=(
            "Excel sheet containing the 86 manuscript records "
            "(default: Matched_86)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=script_dir / "PIML_results",
        help="Output directory",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help=(
            "Skip computationally expensive "
            "fold-safe cross-validation analyses."
        ),
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="publication_id",
        help=(
            "Stable publication/source identifier used for GroupKFold "
            "(default: publication_id from the matched 86-record workbook)."
        ),
    )
    parser.add_argument(
        "--skip-group-cv",
        action="store_true",
        help="Skip publication-aware GroupKFold even when complete source IDs exist.",
    )
    parser.add_argument(
        "--skip-unknown-sensitivity",
        action="store_true",
        help="Skip A0-A3 CV after excluding the unresolved 'unknown' activity class.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = args.data.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Place Database.xlsx next to the script "
            "or pass --data."
        )

    print("=" * 88)
    print(
        "PHYSICS-MOTIVATED NANOZYME MODEL "
        "— SCIENTIFIC REVISION"
    )
    print("=" * 88)
    print(f"Data: {data_path}")
    print(f"Sheet: {args.sheet}")
    print(f"Output: {out_dir}")

    # 1. Load and clean.  The revision workbook contains multiple audit sheets;
    # use the explicitly matched 86-record sheet rather than relying on sheet order.
    try:
        df = pd.read_excel(data_path, sheet_name=args.sheet)
    except ValueError as exc:
        raise ValueError(
            f"Sheet {args.sheet!r} was not found in {data_path.name}. "
            "Use --sheet to select the record sheet explicitly."
        ) from exc
    df.columns = df.columns.str.strip()

    require_columns(
        df,
        ["pH", "Temp (°C)", *TARGETS],
    )

    df = clean_numeric_columns(
        df,
        TARGETS,
    )
    df = df.dropna(
        subset=["pH", "Temp (°C)"]
    ).copy()

    df["activity_type"] = detect_activity_type(df)
    df = prepare_physics_features(df)

    complete = df[TARGETS].notna().all(axis=1)
    data = df.loc[complete].copy()

    # Deliberate manuscript-reproducibility guard, not a generic package constraint.
    # The revised paper reports the frozen 86-record complete-case cohort; silently
    # running a different cohort would make the numerical results non-comparable.
    if len(data) != 86:
        raise ValueError(
            f"Revision dataset produced {len(data)} complete samples, not 86. "
            "Stop and verify that Database_86_with_DOI.xlsx / Matched_86 is being used "
            "and that the manuscript sample has not changed."
        )

    X_all = data[
        PHYSICS_BASE_COLS
    ].copy()
    Y_all_raw = data[
        TARGETS
    ].to_numpy(dtype=float)
    activity_all = data[
        "activity_type"
    ].copy()

    print(
        f"Complete three-target records: {len(data)}"
    )
    print("Activity counts:")
    print(
        activity_all.value_counts(
            dropna=False
        ).to_string()
    )

    # Publication/source dependence audit. Grouped CV is run only when a stable
    # identifier is available for every complete record; missing IDs are never
    # silently treated as independent singleton publications.
    group_column = resolve_group_column(data, args.group_column)
    publication_groups = None
    group_counts_df = pd.DataFrame()
    group_audit = {
        "group_column": None,
        "grouped_cv_eligible": False,
        "reason": "No stable publication/source ID column was identified.",
    }
    if group_column is not None:
        publication_groups, group_counts_df, group_audit = publication_group_audit(
            data, group_column
        )
        print("\nPublication/source audit:")
        print(json.dumps(group_audit, indent=2))
        if not group_audit["grouped_cv_eligible"]:
            print(
                "Grouped CV will NOT be run automatically because publication IDs "
                "are incomplete or fewer than two distinct groups are available."
            )
    else:
        print(
            "\nPublication/source audit: no stable ID-like column detected; "
            "publication-aware CV is unavailable unless --group-column is supplied."
        )

    # 2. Held-out split
    stratify = activity_all
    counts = activity_all.value_counts()

    if (counts < 2).any():
        print(
            "WARNING: a class has <2 samples; "
            "using an unstratified split."
        )
        stratify = None

    (
        X_train,
        X_test,
        Y_train_raw,
        Y_test_raw,
        activity_train,
        activity_test,
    ) = train_test_split(
        X_all,
        Y_all_raw,
        activity_all,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    # 3. Train-only target winsorization
    winsor = TargetWinsorizer().fit(
        Y_train_raw
    )
    Y_train_processed = (
        winsor.transform_train(
            Y_train_raw
        )
    )

    print(
        "\nTraining target winsorization "
        "bounds (log10):"
    )
    for target, bounds in zip(
        TARGETS,
        winsor.bounds_ or [],
    ):
        print(
            f"  {target}: "
            f"[{bounds[0]:.4f}, {bounds[1]:.4f}]"
        )

    print(
        "Test targets are evaluated in their "
        "original, unmodified form."
    )

    # 4. Train-only Ea optimization
    ea_result = learn_activity_specific_ea(
        X_train,
        Y_train_processed,
        activity_train,
    )

    print(
        "\nActivity-specific Ea selected "
        "using training data only:"
    )
    for activity in sorted(
        ea_result.ea_by_activity
    ):
        score = ea_result.best_cv_r2.get(
            activity,
            float("nan"),
        )
        diag = ea_result.diagnostics.get(activity, {})
        print(
            f"  {activity:<12} "
            f"Ea={ea_result.ea_by_activity[activity]:.2f} eV  "
            f"inner-CV R²={score:+.4f}  "
            f"n={diag.get('n_training_records')}  "
            f"groups={diag.get('n_training_groups')}  "
            f"reason={diag.get('selection_reason')}"
        )

    # 5. Primary full model
    full_model = NanozymeRegressor(
        use_physics=True,
        use_arrhenius=True,
        use_activity_onehot=False,
        use_chain=True,
        ea_by_activity=(
            ea_result.ea_by_activity
        ),
    )

    full_model.fit(
        X_train,
        Y_train_processed,
        activity_train,
        TARGETS,
    )

    Y_pred = full_model.predict(
        X_test,
        activity_test,
    )

    primary = evaluate_targets(
        Y_test_raw,
        Y_pred,
        TARGETS,
    )

    print_metric_block(
        "SECONDARY FIXED HELD-OUT RESULTS "
        "(RAW TEST TARGETS; ONE DATA PARTITION)",
        primary,
    )

    # 6. Old Vmax clamp: sensitivity only
    clamp_sensitivity = (
        legacy_vmax_clamp_sensitivity(
            Y_pred
        )
    )

    print(
        "\nLegacy Vmax-clamp sensitivity "
        "(NOT applied to primary metrics):"
    )
    print(
        json.dumps(
            clamp_sensitivity,
            indent=2,
        )
    )

    # 7. Reference baselines
    simple = evaluate_simple_baselines(
        X_train,
        X_test,
        Y_train_processed,
        Y_test_raw,
        TARGETS,
    )

    for label, result in simple.items():
        print_metric_block(
            f"SIMPLE BASELINE — {label}",
            result,
        )

    # 8. Model ablations
    ablation = run_model_ablations(
        X_train,
        X_test,
        Y_train_processed,
        Y_test_raw,
        activity_train,
        activity_test,
        TARGETS,
        ea_result.ea_by_activity,
    )

    for label, result in ablation.items():
        print_metric_block(
            f"EXPLORATORY HELD-OUT ABLATION — {label}",
            result,
        )

    # 9. Fold-safe cross-validation analyses
    cv_payload: Dict[str, object] = {}
    cv_fold_df = pd.DataFrame()
    cv_summary_df = pd.DataFrame()
    cv_delta_df = pd.DataFrame()
    cv_delta_summary_df = pd.DataFrame()
    cv_oof_df = pd.DataFrame()
    cv_oof_summary_df = pd.DataFrame()
    grouped_fold_df = pd.DataFrame()
    grouped_summary_df = pd.DataFrame()
    grouped_delta_df = pd.DataFrame()
    grouped_delta_summary_df = pd.DataFrame()
    grouped_oof_df = pd.DataFrame()
    grouped_oof_summary_df = pd.DataFrame()
    random_vs_grouped_full_df = pd.DataFrame()
    random_vs_grouped_oof_full_df = pd.DataFrame()
    unknown_fold_df = pd.DataFrame()
    unknown_oof_df = pd.DataFrame()
    unknown_summary_df = pd.DataFrame()

    if not args.skip_cv:
        specs = model_ablation_specs()
        sample_ids_all = (
            data["record_id"].astype(str).to_numpy()
            if "record_id" in data.columns
            else data.index.astype(str).to_numpy()
        )

        # Primary generalization evidence: all M1-M4 and A0-A3 on identical
        # leakage-controlled outer folds.
        cv_fold_df, cv_summary_df, cv_oof_df, cv_meta = cross_validate_model_specs(
            X_all,
            Y_all_raw,
            activity_all,
            TARGETS,
            specs,
            n_splits=5,
            sample_ids=sample_ids_all,
            analysis_label="random_5fold_cv",
        )
        cv_oof_summary_df = summarize_oof_predictions(cv_oof_df)
        cv_delta_df, cv_delta_summary_df = paired_cv_deltas(cv_fold_df)
        cv_payload["random_5fold"] = cv_meta

        print("\n" + "=" * 88)
        print("PRIMARY GENERALIZATION EVIDENCE — LEAKAGE-CONTROLLED 5-FOLD CV")
        print("=" * 88)
        full_rows = cv_summary_df.loc[
            cv_summary_df["model"].eq("M4_full_physics_chain")
        ]
        for _, row in full_rows.iterrows():
            print(
                f"{row['target']:<24} "
                f"CV R²={row['mean_R2']:+.4f} ± {row['std_R2']:.4f}"
            )

        # Publication-aware validation is deliberately conditional. We do not
        # manufacture group IDs for records with missing source information.
        if (
            not args.skip_group_cv
            and publication_groups is not None
            and bool(group_audit.get("grouped_cv_eligible", False))
        ):
            grouped_fold_df, grouped_summary_df, grouped_oof_df, grouped_meta = (
                cross_validate_model_specs(
                    X_all,
                    Y_all_raw,
                    activity_all,
                    TARGETS,
                    specs,
                    n_splits=5,
                    groups=publication_groups.to_numpy(dtype=object),
                    sample_ids=sample_ids_all,
                    analysis_label="publication_grouped_cv",
                )
            )
            grouped_oof_summary_df = summarize_oof_predictions(grouped_oof_df)
            grouped_delta_df, grouped_delta_summary_df = paired_cv_deltas(
                grouped_fold_df
            )
            cv_payload["publication_grouped"] = grouped_meta

            print("\n" + "=" * 88)
            print(
                "PUBLICATION-AWARE GENERALIZATION — 5-FOLD GROUPKFold "
                f"({group_audit.get('n_unique_known_groups')} publication groups)"
            )
            print("=" * 88)
            grouped_full_rows = grouped_summary_df.loc[
                grouped_summary_df["model"].eq("M4_full_physics_chain")
            ]
            for _, row in grouped_full_rows.iterrows():
                print(
                    f"{row['target']:<24} "
                    f"Grouped CV R²={row['mean_R2']:+.4f} ± {row['std_R2']:.4f}"
                )

            print("\nPooled publication-grouped OOF performance (all 86 records):")
            grouped_oof_full = grouped_oof_summary_df.loc[
                grouped_oof_summary_df["model"].eq("M4_full_physics_chain")
            ]
            for _, row in grouped_oof_full.iterrows():
                print(
                    f"{row['target']:<24} "
                    f"Pooled OOF R²={row['pooled_OOF_R2']:+.4f}  "
                    f"RMSE={row['pooled_OOF_RMSE_log10']:.4f}  "
                    f"MAE={row['pooled_OOF_MAE_log10']:.4f}"
                )

        # Direct comparison of the frozen full model under
        # ordinary shuffled CV versus publication-aware grouped CV.
        if not grouped_summary_df.empty:
            random_full = cv_summary_df.loc[
                cv_summary_df["model"].eq("M4_full_physics_chain"),
                ["target", "mean_R2", "std_R2"],
            ].rename(
                columns={
                    "mean_R2": "random_5fold_mean_R2",
                    "std_R2": "random_5fold_std_R2",
                }
            )
            grouped_full = grouped_summary_df.loc[
                grouped_summary_df["model"].eq("M4_full_physics_chain"),
                ["target", "mean_R2", "std_R2"],
            ].rename(
                columns={
                    "mean_R2": "publication_grouped_mean_R2",
                    "std_R2": "publication_grouped_std_R2",
                }
            )
            random_vs_grouped_full_df = random_full.merge(
                grouped_full, on="target", how="inner"
            )
            random_vs_grouped_full_df["grouped_minus_random_R2"] = (
                random_vs_grouped_full_df["publication_grouped_mean_R2"]
                - random_vs_grouped_full_df["random_5fold_mean_R2"]
            )

            random_oof_full = cv_oof_summary_df.loc[
                cv_oof_summary_df["model"].eq("M4_full_physics_chain"),
                ["target", "pooled_OOF_R2", "pooled_OOF_RMSE_log10", "pooled_OOF_MAE_log10"],
            ].rename(
                columns={
                    "pooled_OOF_R2": "random_pooled_OOF_R2",
                    "pooled_OOF_RMSE_log10": "random_pooled_OOF_RMSE_log10",
                    "pooled_OOF_MAE_log10": "random_pooled_OOF_MAE_log10",
                }
            )
            grouped_oof_full = grouped_oof_summary_df.loc[
                grouped_oof_summary_df["model"].eq("M4_full_physics_chain"),
                ["target", "pooled_OOF_R2", "pooled_OOF_RMSE_log10", "pooled_OOF_MAE_log10"],
            ].rename(
                columns={
                    "pooled_OOF_R2": "publication_grouped_pooled_OOF_R2",
                    "pooled_OOF_RMSE_log10": "publication_grouped_pooled_OOF_RMSE_log10",
                    "pooled_OOF_MAE_log10": "publication_grouped_pooled_OOF_MAE_log10",
                }
            )
            random_vs_grouped_oof_full_df = random_oof_full.merge(
                grouped_oof_full, on="target", how="inner"
            )
            random_vs_grouped_oof_full_df["grouped_minus_random_pooled_OOF_R2"] = (
                random_vs_grouped_oof_full_df["publication_grouped_pooled_OOF_R2"]
                - random_vs_grouped_oof_full_df["random_pooled_OOF_R2"]
            )

        # Minor-comment sensitivity: remove unresolved activity annotations and
        # repeat the A0-A3 comparison under the same random outer-CV protocol.
        if not args.skip_unknown_sensitivity:
            known_mask = activity_all.ne("unknown").to_numpy()
            known_n = int(known_mask.sum())
            print(
                f"Known-activity sensitivity cohort: {known_n}/{len(activity_all)} records "
                "after excluding 'unknown'."
            )
            if known_n >= 20:
                unknown_fold_df, unknown_summary_df, unknown_oof_df, unknown_meta = (
                    cross_validate_model_specs(
                        X_all.loc[known_mask].copy(),
                        Y_all_raw[known_mask].copy(),
                        activity_all.loc[known_mask].copy(),
                        TARGETS,
                        arrhenius_activity_specs(),
                        n_splits=5,
                        sample_ids=sample_ids_all[known_mask],
                        analysis_label="known_activity_only_random_5fold",
                    )
                )
                cv_payload["known_activity_only"] = unknown_meta
            else:
                print(
                    "Skipping unknown-activity sensitivity: fewer than 20 "
                    "known-activity records remain."
                )

    # 10. Save outputs
    results_to_dataframe(
        {"FullModel": primary}
    ).to_csv(
        out_dir / "HeldOut_Secondary_Results.csv",
        index=False,
    )

    results_to_dataframe(
        simple
    ).to_csv(
        out_dir / "Simple_Baselines.csv",
        index=False,
    )

    results_to_dataframe(
        ablation
    ).to_csv(
        out_dir / "Exploratory_HeldOut_Ablations.csv",
        index=False,
    )

    if not cv_fold_df.empty:
        cv_fold_df.to_csv(
            out_dir / "CV_Model_Ablations_Foldwise.csv",
            index=False,
        )
        cv_summary_df.to_csv(
            out_dir / "CV_Model_Ablations_Summary.csv",
            index=False,
        )
        cv_delta_df.to_csv(
            out_dir / "CV_Paired_R2_Deltas_Foldwise.csv",
            index=False,
        )
        cv_delta_summary_df.to_csv(
            out_dir / "CV_Paired_R2_Deltas_Summary.csv",
            index=False,
        )
        cv_oof_df.to_csv(
            out_dir / "Random_CV_OOF_Predictions.csv",
            index=False,
        )
        cv_oof_summary_df.to_csv(
            out_dir / "Random_CV_OOF_Summary.csv",
            index=False,
        )

    if not grouped_fold_df.empty:
        grouped_fold_df.to_csv(
            out_dir / "Publication_Grouped_CV_Foldwise.csv",
            index=False,
        )
        grouped_summary_df.to_csv(
            out_dir / "Publication_Grouped_CV_Summary.csv",
            index=False,
        )
        grouped_delta_df.to_csv(
            out_dir / "Publication_Grouped_CV_Paired_R2_Deltas_Foldwise.csv",
            index=False,
        )
        grouped_delta_summary_df.to_csv(
            out_dir / "Publication_Grouped_CV_Paired_R2_Deltas_Summary.csv",
            index=False,
        )
        grouped_oof_df.to_csv(
            out_dir / "Publication_Grouped_OOF_Predictions.csv",
            index=False,
        )
        grouped_oof_summary_df.to_csv(
            out_dir / "Publication_Grouped_OOF_Summary.csv",
            index=False,
        )
        if not random_vs_grouped_full_df.empty:
            random_vs_grouped_full_df.to_csv(
                out_dir / "Random_vs_Publication_Grouped_FullModel.csv",
                index=False,
            )
        if not random_vs_grouped_oof_full_df.empty:
            random_vs_grouped_oof_full_df.to_csv(
                out_dir / "Random_vs_Publication_Grouped_OOF_FullModel.csv",
                index=False,
            )

    if not unknown_fold_df.empty:
        unknown_fold_df.to_csv(
            out_dir / "KnownActivityOnly_A0_A3_CV_Foldwise.csv",
            index=False,
        )
        unknown_summary_df.to_csv(
            out_dir / "KnownActivityOnly_A0_A3_CV_Summary.csv",
            index=False,
        )
        if not unknown_oof_df.empty:
            unknown_oof_df.to_csv(
                out_dir / "KnownActivityOnly_A0_A3_OOF_Predictions.csv",
                index=False,
            )

    if group_column is not None:
        group_counts_df.to_csv(
            out_dir / "Publication_Group_Counts.csv",
            index=False,
        )

    fixed_ea_rows = []
    for activity, ea in sorted(ea_result.ea_by_activity.items()):
        diag = ea_result.diagnostics.get(activity, {})
        fixed_ea_rows.append(
            {
                "activity": activity,
                "Ea_eV": float(ea),
                "inner_cv_R2": float(
                    ea_result.best_cv_r2.get(activity, float("nan"))
                ),
                **diag,
            }
        )
    pd.DataFrame(fixed_ea_rows).to_csv(
        out_dir / "FixedSplit_Ea_Tuning_Diagnostics.csv",
        index=False,
    )

    outer_ea_rows = []
    for analysis_name, analysis_meta in cv_payload.items():
        if isinstance(analysis_meta, dict):
            outer_ea_rows.extend(analysis_meta.get("ea_tuning_records", []))
    if outer_ea_rows:
        pd.DataFrame(outer_ea_rows).to_csv(
            out_dir / "CV_Ea_Tuning_Diagnostics.csv",
            index=False,
        )

    metadata = {
        "random_state": RANDOM_STATE,
        "n_total_complete": int(len(data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "targets": TARGETS,
        "ea_by_activity": (
            ea_result.ea_by_activity
        ),
        "ea_inner_cv_r2": (
            ea_result.best_cv_r2
        ),
        "winsor_bounds_log10_train_only": (
            winsor.bounds_
        ),
        "legacy_vmax_clamp_sensitivity": (
            clamp_sensitivity
        ),
        "cv_analyses": cv_payload,
        "publication_group_audit": group_audit,
        "method_notes": {
            "heldout_role": "secondary_single_partition",
            "principal_generalization_evidence": "outer_cross_validation",
            "test_targets_winsorized": False,
            "vmax_postprocessing_applied": False,
            "outer_cv_refits_preprocessing": True,
            "outer_cv_retunes_Ea": True,
            "cv_model_specs_share_identical_outer_folds": True,
            "publication_grouped_cv_requires_complete_stable_ids": True,
            "publication_group_source": "matched DOI/publication_id for all 86 manuscript records",
            "publication_grouped_cv_inner_Ea_tuning_group_aware": True,
            "outer_cv_out_of_fold_predictions_saved": True,
            "oof_prediction_integrity_rule": (
                "each model/sample/target is predicted exactly once by an outer-fold model"
            ),
            "pooled_oof_metrics_role": (
                "complementary overall performance estimate; fold-wise mean +/- SD retained "
                "to show between-fold heterogeneity"
            ),
            "missing_publication_ids_are_not_assumed_independent": True,
            "raw_base_feature_columns": RAW_COLS,
            "raw_base_definition": (
                "pH, temperature, and the same coarse name-derived material descriptors; "
                "excludes pH-zone, inverse-temperature, thermal-energy, and Arrhenius transforms"
            ),
            "regressor_chain_order": TARGETS,
            "regressor_chain_order_validation_tuned": False,
            "material_descriptor_rule": (
                "deterministic first recognized Au/Pt/Fe/Ag/Cu keyword; coarse heuristic only"
            ),
            "expected_complete_case_cohort_n": 86,
        },
    }

    with open(
        out_dir / "Run_Metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    save_primary_figures(
        out_dir,
        full_model,
        Y_test_raw,
        Y_pred,
        TARGETS,
        primary,
    )

    # Test predictions with original row index
    pred_df = pd.DataFrame(
        {
            "source_row_index": (
                X_test.index.to_numpy()
            ),
            "activity_type": (
                activity_test.to_numpy()
            ),
        }
    )

    for j, target in enumerate(TARGETS):
        pred_df[
            f"experimental_{target}"
        ] = Y_test_raw[:, j]
        pred_df[
            f"predicted_{target}"
        ] = Y_pred[:, j]

    pred_df.to_csv(
        out_dir / "HeldOut_Test_Predictions.csv",
        index=False,
    )

    print("\n" + "=" * 88)
    print("DONE")
    print("=" * 88)

    for path in sorted(
        out_dir.iterdir()
    ):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
