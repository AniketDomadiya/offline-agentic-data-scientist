"""
Modelling Tools
===============
Plan-aware preprocessing, model selection, and training.

Changes from previous version
------------------------------
impute_strategy parameter
    ``build_preprocessor`` now accepts ``impute_strategy="mean"|"median"``
    (default "median").  The Planner adds TASK_IMPUTE_MEAN when no skewed
    features are detected; otherwise median (the safer default) is used.

consider_severe_imbalance
    When the plan contains ``consider_severe_imbalance`` (ratio > 10),
    ``select_models`` skips LogisticRegression entirely - it tends to
    under-perform severely imbalanced datasets even with class_weight -
    and increases n_estimators for tree ensembles for better minority
    coverage.

emphasize_ensemble
    Triggered during replan when F1 < 0.50.  Removes LogisticRegression,
    ensures GradientBoosting is always included, increases RF/ET estimators.

StratifiedKFold cross-validation
    Activated by ``cross_validate_folds > 0`` (from TASK_CROSS_VAL).
    Computes averaged metrics over k folds, then refits on all of X_train
    for the final held-out test evaluation.

ExtraTreesClassifier
    Always included as a candidate - fast, competitive, and diverse from RF.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
    LabelEncoder,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


# OneHotEncoder sklearn compat shim
def _make_ohe(**kwargs) -> OneHotEncoder:
    """Support sklearn < 1.2 (sparse=False) and ≥ 1.2 (sparse_output=False)."""
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)

# Preprocessing builder
def build_preprocessor(
    profile: Dict[str, Any],
    use_power_transform: bool = False,
    handle_high_cardinality: bool = False,
    impute_strategy: str = "median",
) -> ColumnTransformer:
    """
    Build a ColumnTransformer preprocessing pipeline tailored to the profile.

    Parameters
    ----------
    profile : dict
        Output of ``profile_dataset()``.
    use_power_transform : bool
        Apply PowerTransformer (Yeo-Johnson) after scaling.
        Triggered by TASK_SKEW_TRANSFORM in the plan.
    handle_high_cardinality : bool
        Split categorical columns into low-card (OHE) and high-card (Ordinal).
        Triggered by TASK_HIGH_CARD in the plan.
    impute_strategy : str
        "mean" or "median" for numeric imputation.
        "mean" → triggered by TASK_IMPUTE_MEAN (no skewed features present).
        "median" → default; safer when skewness is present.

    Returns
    -------
    sklearn ColumnTransformer
    """
    num_cols  = profile["feature_types"]["numeric"]
    cat_cols  = profile["feature_types"]["categorical"]
    high_card = profile.get("high_cardinality_features", [])

    # ── Numeric pipeline ───────────────────────────────────────────────────
    numeric_steps = [
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("scaler",  StandardScaler()),
    ]
    if use_power_transform:
        # PowerTransformer after scaling handles residual skewness
        numeric_steps.append(
            ("power", PowerTransformer(method="yeo-johnson", standardize=True))
        )
    numeric_pipeline = Pipeline(steps=numeric_steps)

    # ── Categorical pipeline(s) ────────────────────────────────────────────
    transformers: List[Tuple[str, Any, List[str]]] = []

    if num_cols:
        transformers.append(("num", numeric_pipeline, num_cols))

    if handle_high_cardinality and high_card and cat_cols:
        low_card_cols  = [c for c in cat_cols if c not in high_card]
        high_card_cols = [c for c in cat_cols if c in high_card]

        if low_card_cols:
            low_card_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot",  _make_ohe(handle_unknown="ignore")),
            ])
            transformers.append(("cat_low", low_card_pipeline, low_card_cols))

        if high_card_cols:
            # OrdinalEncoder prevents feature explosion on high-cardinality cols
            high_card_pipeline = Pipeline(steps=[
                ("imputer",  SimpleImputer(strategy="most_frequent")),
                ("ordinal",  OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )),
            ])
            transformers.append(("cat_high", high_card_pipeline, high_card_cols))

    elif cat_cols:
        # Default: all categoricals → OneHotEncoder
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  _make_ohe(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")

# Model selection
def select_models(
    profile: Dict[str, Any],
    seed: int = 42,
    plan: Optional[List[str]] = None,
) -> List[Tuple[str, Any]]:
    """
    Choose candidate classifiers based on dataset characteristics and the plan.

    Plan-driven decisions
    ---------------------
    consider_imbalance
        → class_weight='balanced' for all classifiers that support it.
    consider_severe_imbalance
        → class_weight='balanced' AND LogisticRegression excluded (too weak
          on severely imbalanced data) AND higher n_estimators for RF/ET.
    emphasize_ensemble
        → LogisticRegression excluded; GradientBoosting always included;
          RF/ET n_estimators raised to 400.
    prioritize_model:<n>
        → Named model moved to front of candidate list.

    Size-based decisions
    --------------------
    rows > 50 000
        → GradientBoosting excluded by default (training cost); overridden by
          emphasize_ensemble or consider_severe_imbalance.
    rows > 20 000 OR cols > 200
        → SVC excluded (too expensive post one-hot expansion).

    Parameters
    ----------
    profile : dict
    seed    : int
    plan    : List[str] - current plan list for conditional logic

    Returns
    -------
    List of (name, unfitted_estimator) tuples, sorted with prioritised
    model at front if applicable.
    """
    plan = plan or []
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["cols"]
    imb  = float(profile.get("imbalance_ratio") or 1.0)

    # Determine class_weight
    use_balanced       = imb >= 3.0 or "consider_imbalance" in plan or "consider_severe_imbalance" in plan
    cw                 = "balanced" if use_balanced else None
    severe_imb         = "consider_severe_imbalance" in plan
    emphasize_ensemble = "emphasize_ensemble" in plan

    # Number of estimators: raise under severe imbalance or ensemble emphasis
    n_est = 400 if (severe_imb or emphasize_ensemble) else 300

    # ── Build candidate list ────────────────────────────────────────────────
    candidates: List[Tuple[str, Any]] = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
    ]

    # LogisticRegression: excluded under severe imbalance or ensemble emphasis
    # (it struggles to fit minority classes even with balanced weights)
    if not severe_imb and not emphasize_ensemble:
        candidates.append((
            "LogisticRegression",
            LogisticRegression(max_iter=2000, class_weight=cw, random_state=seed),
        ))

    candidates += [
        ("RandomForest", RandomForestClassifier(
            n_estimators=n_est, random_state=seed, n_jobs=-1, class_weight=cw
        )),
        ("ExtraTrees", ExtraTreesClassifier(
            n_estimators=n_est, random_state=seed, n_jobs=-1, class_weight=cw
        )),
    ]

    # GradientBoosting: no native class_weight; included when affordable or demanded
    include_gb = (
        rows <= 50_000
        or emphasize_ensemble
        or severe_imb
    )
    if include_gb:
        candidates.append((
            "GradientBoosting",
            GradientBoostingClassifier(n_estimators=200, random_state=seed),
        ))

    # SVC: expensive after one-hot expansion
    if rows <= 20_000 and cols <= 200 and not severe_imb and not emphasize_ensemble:
        candidates.append((
            "SVC_RBF",
            SVC(kernel="rbf", probability=True, class_weight=cw, random_state=seed),
        ))

    # ── Honour memory hint: move prioritised model to front ─────────────────
    for task in plan:
        if task.startswith("prioritize_model:"):
            model_name = task.split(":", 1)[1]
            idx = next(
                (i for i, (n, _) in enumerate(candidates) if n == model_name),
                None,
            )
            if idx is not None and idx != 0:
                candidates.insert(0, candidates.pop(idx))
            break

    return candidates


# Training
def _compute_metrics(name: str, y_true, y_pred) -> Dict[str, Any]:
    """Compute the standard set of classification metrics."""
    return {
        "model":             name,
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro",  zero_division=0)),
        "precision_macro":   float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":      float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
    }


def train_models(
    df: pd.DataFrame,
    target: str,
    preprocessor: ColumnTransformer,
    candidates: List[Tuple[str, Any]],
    seed: int,
    test_size: float,
    output_dir: str,
    verbose: bool = True,
    cross_validate_folds: int = 0,
) -> Dict[str, Any]:
    """
    Train all candidate models and return results sorted by balanced accuracy.

    Parameters
    ----------
    df, target, preprocessor, candidates, seed, test_size, output_dir, verbose
        Standard parameters.
    cross_validate_folds : int
        If > 0, run StratifiedKFold CV on the training split and report
        averaged metrics.  A final full-training-data fit is then made for
        held-out test evaluation.  Triggered by TASK_CROSS_VAL (use 5 folds).

    Returns
    -------
    {
      "results"     : list of result dicts sorted by balanced_accuracy desc
      "best"        : single best result dict
      "all_metrics" : list of metrics dicts for all models
    }
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe.")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # If target is float dtype, assume discrete classes encoded as floats - label-encode to integers
    label_encoder = None
    if pd.api.types.is_float_dtype(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
        label_encoder = le
        if verbose:
            print(
                f"[Modelling] Target '{target}' was float dtype; applied LabelEncoder to convert classes to integers.",
                flush=True,
            )

    # Drop rows where target is missing
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]

    # Stratified split
    stratify = y if y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    results: List[Dict[str, Any]] = []

    for name, model in candidates:
        if verbose:
            print(f"[Modelling] Training: {name}", flush=True)

        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model",      model),
        ])

        # ── Cross-validation path (small datasets) ─────────────────────────
        if cross_validate_folds > 1 and len(X_train) >= cross_validate_folds * 2:
            cv = StratifiedKFold(n_splits=cross_validate_folds, shuffle=True, random_state=seed)
            cv_results = cross_validate(
                pipe, X_train, y_train, cv=cv,
                scoring=["balanced_accuracy", "f1_macro"],
                return_train_score=False,
            )
            cv_ba = float(np.mean(cv_results["test_balanced_accuracy"]))
            cv_f1 = float(np.mean(cv_results["test_f1_macro"]))
            if verbose:
                print(
                    f"[Modelling]   CV bal_acc={cv_ba:.3f} "
                    f"± {np.std(cv_results['test_balanced_accuracy']):.3f}",
                    flush=True,
                )
            # Final fit on all training data
            pipe.fit(X_train, y_train)
            y_pred  = pipe.predict(X_test)
            metrics = _compute_metrics(name, y_test, y_pred)
            metrics["cv_balanced_accuracy"] = round(cv_ba, 4)
            metrics["cv_f1_macro"]          = round(cv_f1, 4)
        else:
            # ── Standard single fit / predict ──────────────────────────────
            pipe.fit(X_train, y_train)
            y_pred  = pipe.predict(X_test)
            metrics = _compute_metrics(name, y_test, y_pred)

        results.append({
            "name":     name,
            "pipeline": pipe,
            "metrics":  metrics,
            "X_test":   X_test,
            "y_test":   y_test,
            "y_pred":   y_pred,
        })

    # Sort by balanced_accuracy (primary), then f1_macro (tiebreak)
    results.sort(
        key=lambda r: (r["metrics"]["balanced_accuracy"], r["metrics"]["f1_macro"]),
        reverse=True,
    )

    return {
        "results":     results,
        "best":        results[0],
        "all_metrics": [r["metrics"] for r in results],
        "label_encoder": label_encoder,
    }