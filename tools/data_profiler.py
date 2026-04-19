"""
Enhanced Data Profiler
======================
Always classification-only.

Target detection
----------------
``infer_target_column`` scores all columns and then iterates from highest
score downward, returning the first candidate that is actually suitable
for classification (discrete, 2–50 unique values).  This means if the
top-scored column turns out to be continuous the function automatically
tries the next best candidate rather than returning an unusable column.

``is_classification_suitable`` defines "suitable": a column must have
at least 2 and at most 200 unique values, and for float columns the
bar is tighter (≤ 20 unique values), since floats are almost always
continuous measurements unless they are encoded labels.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Thresholds ─────────────────────────────────────────────────────────────
_UNIQUE_RATIO_ID     = 0.95
_TARGET_KEYWORDS     = frozenset(
    ["target", "label", "class", "output", "result", "status", "y", "outcome"]
)
_HIGH_CARD_THRESHOLD = 20
_CORR_THRESHOLD      = 0.80
_SKEW_HIGH           = 1.0
_SKEW_MOD            = 0.5
_DATETIME_CONV_RATIO = 0.80
_CAT_UNIQUE_RATIO    = 0.05
_CAT_VALUE_RANGE     = 20


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _detect_id_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that behave as row identifiers (no predictive signal)."""
    id_cols: List[str] = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        ur = s.nunique() / n
        if "id" in col.lower():
            id_cols.append(col)
        elif ur > _UNIQUE_RATIO_ID and not pd.api.types.is_numeric_dtype(s):
            id_cols.append(col)
        elif ur > _UNIQUE_RATIO_ID and s.is_monotonic_increasing:
            id_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            diffs = s.diff().dropna()
            if len(diffs) > 1 and diffs.nunique() == 1:
                id_cols.append(col)
        elif s.dtype == "object":
            avg_len = s.astype(str).str.len().mean()
            if avg_len > 15 and ur > _UNIQUE_RATIO_ID:
                id_cols.append(col)
    return list(set(id_cols))


def _behaves_like_categorical(series: pd.Series) -> bool:
    """Return True if an integer series encodes discrete categories."""
    s = series.dropna()
    if len(s) == 0 or not pd.api.types.is_integer_dtype(s):
        return False
    ur = s.nunique() / len(s)
    vr = float(s.max() - s.min())
    return ur < _CAT_UNIQUE_RATIO and vr < _CAT_VALUE_RANGE


def _classify_columns(
    df: pd.DataFrame,
    id_cols: List[str],
) -> Dict[str, str]:
    """Assign each non-ID column a type: numeric | categorical | datetime | text."""
    schema: Dict[str, str] = {}
    for col in df.columns:
        if col in id_cols:
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            schema[col] = "categorical"
        elif pd.api.types.is_datetime64_any_dtype(s):
            schema[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            schema[col] = "categorical" if _behaves_like_categorical(s) else "numeric"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_datetime(s, errors="coerce")
            if converted.notna().sum() / max(len(s), 1) >= _DATETIME_CONV_RATIO:
                schema[col] = "datetime"
            else:
                threshold = max(10, _CAT_UNIQUE_RATIO * len(s))
                schema[col] = "categorical" if s.nunique() <= threshold else "text"
    return schema


def _score_target_candidates(
    df: pd.DataFrame,
    id_cols: List[str],
) -> Dict[str, float]:
    """
    Score each column by how likely it is to be the classification target.

    Positive signals:
      +4  name matches a known target keyword
      +2  last column (common dataset convention)

    Negative signals:
      -4  detected as an identifier column
      -4  column looks like a datetime
    """
    scores: Dict[str, float] = {}
    for col in df.columns:
        score = 0.0
        if col in id_cols:
            score -= 4
        if col.lower() in _TARGET_KEYWORDS:
            score += 4
        if col == df.columns[-1]:
            score += 2
        if (
            not pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_bool_dtype(df[col])
            and col not in id_cols
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().sum() / max(len(df[col]), 1) >= _DATETIME_CONV_RATIO:
                score -= 4
        scores[col] = score
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# Public helpers
# ═══════════════════════════════════════════════════════════════════════════

def is_classification_suitable(series: pd.Series) -> bool:
    """
    Return True if a column is genuinely suitable as a classification target.

    Rules
    -----
    - At least 2 distinct non-null values (need at least two classes).
    - String / category columns: up to 200 unique values.
    - Boolean: always suitable (binary classification).
    - Integer numeric: 2–50 unique values (encoded labels, ratings, etc.).
    - Float numeric: 2–20 unique values only.
      Rationale: floats almost always represent continuous measurements;
      if there are only a handful of unique floats they are likely encoded
      labels (e.g. 0.0 / 1.0) — otherwise reject.
    """
    s = series.dropna()
    if len(s) == 0:
        return False
    unique = int(s.nunique())
    if unique < 2:
        return False  # can't classify with only one value

    if pd.api.types.is_bool_dtype(s):
        return True
    if s.dtype == "object" or str(s.dtype).startswith("category"):
        return unique <= 200
    if pd.api.types.is_integer_dtype(s):
        return unique <= 50
    if pd.api.types.is_float_dtype(s):
        return unique <= 20  # tight limit: floats are almost always continuous
    return unique <= 50


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identify the best classification target column via heuristic scoring.

    Algorithm
    ---------
    1. Score every column (keyword match, position bias, ID/datetime penalties).
    2. Sort by score descending.
    3. Return the first positively-scored column that passes
       ``is_classification_suitable``.
    4. If nothing positively scored is suitable, check the last column
       as a final fallback (many public datasets put the label last).
    5. Return None if nothing suitable is found.

    This means if the top-scored column is a continuous float the function
    automatically tries the next best candidate, so the agent never silently
    picks an unusable regression target.
    """
    if df.empty or len(df.columns) == 0:
        return None

    id_cols = _detect_id_columns(df)
    scores  = _score_target_candidates(df, id_cols)

    # Try candidates in descending score order
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for col, score in sorted_candidates:
        if score <= 0:
            break
        if is_classification_suitable(df[col]):
            return col

    # Fallback: last column regardless of score
    last = df.columns[-1]
    if is_classification_suitable(df[last]):
        return last

    return None


def dataset_fingerprint(df: pd.DataFrame, target: str) -> str:
    """Produce a stable hash-based fingerprint for a (dataset, target) pair."""
    cols = ",".join(df.columns.astype(str).tolist())
    base = f"{df.shape[0]}x{df.shape[1]}|{target}|{cols}"
    return f"fp_{abs(hash(base)) % (10 ** 12)}"


# ═══════════════════════════════════════════════════════════════════════════
# Main profiler
# ═══════════════════════════════════════════════════════════════════════════

def profile_dataset(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """
    Produce a rich, classification-focused profile of a dataset.

    Raises
    ------
    ValueError
        If ``target`` is not a column in ``df``, with a message that
        lists all available columns.
    ValueError
        If the target column is not suitable for classification (e.g. it is
        a continuous float with many unique values).
    """
    if target not in df.columns:
        available = ", ".join(f"'{c}'" for c in df.columns.tolist()[:20])
        raise ValueError(
            f"Target column '{target}' not found in the dataset. "
            f"Available columns: {available}"
            + (" …" if len(df.columns) > 20 else "")
        )

    y = df[target]

    # Validate that the chosen target is actually usable for classification
    if not is_classification_suitable(y):
        n_unique = y.nunique(dropna=True)
        raise ValueError(
            f"Target column '{target}' does not appear to be suitable for "
            f"classification: it has {n_unique} unique values "
            f"(dtype={y.dtype}). "
            "For classification the target should be discrete with ≤50 unique "
            "values (≤20 for float columns). "
            "If your target is continuous please specify a different column "
            "with --target <column_name>, or check that 'auto' detection "
            "can find a suitable column."
        )

    X = df.drop(columns=[target])

    profile: Dict[str, Any] = {}

    # Shape and duplicates
    profile["shape"]           = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    profile["duplicate_count"] = int(df.duplicated().sum())
    profile["columns"]         = df.columns.astype(str).tolist()

    # Missing values
    missing = (df.isna().mean() * 100).round(2)
    profile["missing_pct"] = {str(k): float(v) for k, v in missing.items()}

    # Column typing
    id_cols = _detect_id_columns(X)
    profile["id_features"] = id_cols
    schema  = _classify_columns(X, id_cols)
    profile["feature_types"] = {
        "numeric":     [c for c, t in schema.items() if t == "numeric"],
        "categorical": [c for c, t in schema.items() if t == "categorical"],
        "datetime":    [c for c, t in schema.items() if t == "datetime"],
        "text":        [c for c, t in schema.items() if t == "text"],
    }

    # Target — always classification
    profile["target"]           = str(target)
    profile["target_dtype"]     = str(y.dtype)
    profile["is_classification"] = True
    profile["n_classes"]         = int(y.nunique(dropna=True))

    vc = y.value_counts(dropna=False)
    profile["class_counts"]    = {str(k): int(v) for k, v in vc.items()}
    profile["imbalance_ratio"] = (
        round(float(vc.max() / max(vc.min(), 1)), 3) if len(vc) >= 2 else 1.0
    )

    # Cardinality
    profile["n_unique_by_col"] = {
        str(c): int(df[c].nunique(dropna=True)) for c in df.columns.astype(str)
    }
    profile["high_cardinality_features"] = [
        c for c in profile["feature_types"]["categorical"]
        if df[c].nunique(dropna=True) > _HIGH_CARD_THRESHOLD
    ]

    # Skewness
    num_cols: List[str] = profile["feature_types"]["numeric"]
    if num_cols:
        skew = X[num_cols].skew()
        profile["skewness_values"] = {str(k): round(float(v), 3) for k, v in skew.items()}
        profile["highly_skewed_features"]    = [c for c, v in profile["skewness_values"].items() if abs(v) > _SKEW_HIGH]
        profile["moderately_skewed_features"] = [c for c, v in profile["skewness_values"].items() if _SKEW_MOD < abs(v) <= _SKEW_HIGH]
    else:
        profile["skewness_values"]            = {}
        profile["highly_skewed_features"]     = []
        profile["moderately_skewed_features"] = []

    # Outliers (IQR)
    outlier_cols: List[str] = []
    for col in num_cols:
        q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0 and ((X[col] < q1 - 1.5 * iqr) | (X[col] > q3 + 1.5 * iqr)).any():
            outlier_cols.append(col)
    profile["outlier_columns"] = outlier_cols

    # Multicollinearity
    high_corr: List[Tuple[str, str]] = []
    if len(num_cols) >= 2:
        corr  = X[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        for ci in upper.columns:
            for ri in upper.index:
                val = upper.loc[ri, ci]
                if pd.notna(val) and val > _CORR_THRESHOLD:
                    high_corr.append((str(ri), str(ci)))
    profile["high_corr_pairs"] = high_corr

    # Notes
    notes: List[str] = []
    rows = profile["shape"]["rows"]
    if rows < 500:
        notes.append(
            f"Small dataset ({rows} rows): prefer simpler models; "
            "cross-validation will be used for reliable metric estimates."
        )
    if profile["shape"]["cols"] > 100:
        notes.append("High dimensionality (>100 cols): ordinal encoding recommended.")
    if profile["duplicate_count"] > 0:
        notes.append(f"{profile['duplicate_count']} duplicate rows detected — will be removed.")
    if profile["imbalance_ratio"] >= 3.0:
        notes.append(
            f"Class imbalance (ratio={profile['imbalance_ratio']:.1f}): "
            "balanced class weights and macro-F1 will be used."
        )
    if profile["highly_skewed_features"]:
        notes.append(
            f"{len(profile['highly_skewed_features'])} highly skewed feature(s): "
            "PowerTransformer will be applied."
        )
    if profile["outlier_columns"]:
        notes.append(f"Outliers in {len(profile['outlier_columns'])} column(s) (IQR method).")
    if profile["high_corr_pairs"]:
        notes.append(f"{len(profile['high_corr_pairs'])} highly correlated pair(s) detected.")
    if profile["high_cardinality_features"]:
        notes.append(f"High-cardinality categorical(s): {profile['high_cardinality_features']}.")
    if profile["feature_types"]["text"]:
        notes.append(f"Text feature(s) detected — will be excluded from modelling.")
    if profile["feature_types"]["datetime"]:
        notes.append(f"Datetime feature(s) — year/month will be extracted.")

    profile["notes"] = notes
    return profile