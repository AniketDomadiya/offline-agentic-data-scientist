"""
Memory System
=============
Lightweight persistent memory backed by a single JSON file.

Each entry is keyed by dataset fingerprint and records the best model, performance metrics, and dataset metadata needed for similarity matching.

Meta-learning additions
-----------------------
``store_reflection_outcome``
    Called after each run to record what issues were flagged, whether a
    replan happened, and the resulting F1. Over multiple runs on the same
    or similar datasets the agent accumulates a track record for different
    suggestion types.

``get_suggestion_effectiveness``
    Returns a dict mapping suggestion category → success_rate for a given
    fingerprint. The Reflector uses this to deprioritise suggestions that
    have already been tried and failed on this dataset.

``get_failed_strategies``
    Returns a set of suggestion categories that were tried in prior replans
    and did NOT lead to meaningful F1 improvement. Used by ``should_replan``
    to detect diminishing returns and by the Reflector to avoid repeating
    stale advice.

Similarity-based retrieval
--------------------------
When an exact fingerprint match is unavailable, ``get_hint`` returns the
most similar stored record scored across four dimensions:
  - Size bucket      (tiny / small / medium / large)  weight 0.30
  - Number of classes                                  weight 0.25
  - Imbalance ratio  (log scale)                       weight 0.25
  - Numeric / categorical feature ratio                weight 0.20
"""

import json
import math
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


# Helpers
def now_iso() -> str:
    """Return current UTC time as ISO-8601 string (no microseconds)."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _size_bucket(rows: int) -> str:
    if rows < 500:    return "tiny"
    if rows < 5_000:  return "small"
    if rows < 50_000: return "medium"
    return "large"

def _compute_similarity(
    profile: Dict[str, Any],
    record: Dict[str, Any],
) -> float:
    """
    Compute a weighted similarity score ∈ [0, 1] between a live profile
    and a stored memory record.

    Dimensions
    ----------
    Size bucket match   0.30  - exact categorical match
    n_classes proximity 0.25  - linear decay, max penalty at diff ≥ 10
    Imbalance ratio     0.25  - log-scale proximity
    Feature-type ratio  0.20  - |Δratio_numeric|
    """
    score = 0.0

    rows_a   = profile.get("shape", {}).get("rows", 0)
    bucket_a = _size_bucket(rows_a)
    bucket_b = record.get("size_bucket", "")
    score   += 0.30 if bucket_a == bucket_b else 0.0

    nc_a   = float(profile.get("n_classes", 2))
    nc_b   = float(record.get("n_classes", 2))
    score += 0.25 * max(0.0, 1.0 - abs(nc_a - nc_b) / 10.0)

    imb_a    = float(profile.get("imbalance_ratio") or 1.0)
    imb_b    = float(record.get("imbalance_ratio") or 1.0)
    imb_dist = abs(math.log(max(imb_a, 1.0)) - math.log(max(imb_b, 1.0)))
    score   += 0.25 * max(0.0, 1.0 - imb_dist / 3.0)

    ftype   = profile.get("feature_types", {})
    n_num   = len(ftype.get("numeric", []))
    n_cat   = len(ftype.get("categorical", []))
    total_a = max(n_num + n_cat, 1)
    ratio_a = n_num / total_a

    n_num_b  = float(record.get("n_numeric", 0))
    n_cat_b  = float(record.get("n_categorical", 0))
    total_b  = max(n_num_b + n_cat_b, 1)
    ratio_b  = n_num_b / total_b
    score   += 0.20 * max(0.0, 1.0 - abs(ratio_a - ratio_b))

    return round(score, 4)


# Main class
class JSONMemory:
    """
    Persistent agent memory backed by a JSON file.

    Dataset record fields
    ---------------------
    last_seen           ISO timestamp
    target              target column name
    shape               {rows, cols}
    size_bucket         coarse size category
    n_classes           number of target classes
    imbalance_ratio     majority / minority ratio
    n_numeric           count of numeric features
    n_categorical       count of categorical features
    best_model          name of best-performing model
    best_metrics        full metrics dict for best model
    all_metrics         metrics for all models
    plan                execution plan used
    reflection_status   "ok" | "needs_attention"
    notes               list of important observations
    reflection_history  list of {ts, issues, suggestions, f1_before,
                                  f1_after, replan_count, improved}
                        - meta-learning track record
    """

    _MIN_SIMILARITY      = 0.50   # below this, similarity hints are discarded
    _MIN_F1_IMPROVEMENT  = 0.02   # improvement below this → suggestion "failed"

    def __init__(self, path: str = "agent_memory.json"):
        self.path = path
        self.data: Dict[str, Any] = {"datasets": {}, "notes": []}
        self._load()

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except (json.JSONDecodeError, OSError):
            backup = self.path + ".bak"
            try:
                shutil.copy(self.path, backup)
            except OSError:
                pass
            self.data = {
                "datasets": {},
                "notes": [{"ts": now_iso(), "msg": f"Memory reset; backup at {backup}"}],
            }

    def save(self) -> None:
        """Persist to disk via atomic write-then-rename."""
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    # ── Exact-match access ───────────────────────────────────────────────────

    def get_dataset_record(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Return the stored record for this exact fingerprint, or None."""
        return self.data.get("datasets", {}).get(fingerprint)

    def upsert_dataset_record(
        self,
        fingerprint: str,
        record: Dict[str, Any],
    ) -> None:
        """Insert or overwrite the record for this fingerprint."""
        self.data.setdefault("datasets", {})[fingerprint] = record
        self.save()

    # ── Meta-learning ────────────────────────────────────────────────────────

    def store_reflection_outcome(
        self,
        fingerprint: str,
        issues: List[str],
        suggestions: List[str],
        f1_before: float,
        f1_after: float,
        replan_count: int,
    ) -> None:
        """
        Append a reflection outcome to the dataset record's history.

        Called by the Orchestrator at the END of each cycle (after reflect())
        if a replan happened. This lets us track whether acting on specific
        types of suggestions actually improved performance.

        Parameters
        ----------
        fingerprint  : Dataset fingerprint.
        issues       : List of issue strings from reflect().
        suggestions  : List of suggestion strings from reflect().
        f1_before    : Best F1 before this replan cycle.
        f1_after     : Best F1 achieved after acting on suggestions.
        replan_count : How many replans have happened so far.
        """
        record = self.data.get("datasets", {}).get(fingerprint)
        if record is None:
            return  # no record yet; orchestrator will create one shortly

        history = record.setdefault("reflection_history", [])

        # Categorise suggestions for efficient lookup
        categories = _categorise_suggestions(suggestions)
        improved   = (f1_after - f1_before) >= self._MIN_F1_IMPROVEMENT

        history.append({
            "ts":           now_iso(),
            "issues":       issues[:5],          # cap to keep JSON compact
            "categories":   categories,
            "f1_before":    round(f1_before, 4),
            "f1_after":     round(f1_after, 4),
            "improved":     improved,
            "replan_count": replan_count,
        })

        self.save()

    def get_suggestion_effectiveness(
        self, fingerprint: str
    ) -> Dict[str, float]:
        """
        Return category → success_rate for a given fingerprint.

        success_rate is the fraction of past cycles where acting on that
        suggestion category led to meaningful F1 improvement.

        Used by the Reflector to downweight suggestions with a poor track
        record so the agent doesn't repeat advice that isn't working.
        """
        record = self.data.get("datasets", {}).get(fingerprint)
        if record is None:
            return {}

        counts:   Dict[str, int] = {}
        successes: Dict[str, int] = {}

        for entry in record.get("reflection_history", []):
            for cat in entry.get("categories", []):
                counts[cat]    = counts.get(cat, 0) + 1
                if entry.get("improved", False):
                    successes[cat] = successes.get(cat, 0) + 1

        return {
            cat: round(successes.get(cat, 0) / cnt, 2)
            for cat, cnt in counts.items()
        }

    def get_failed_strategies(
        self, fingerprint: str
    ) -> Set[str]:
        """
        Return the set of suggestion categories that have been tried at least
        once and NEVER led to meaningful improvement on this fingerprint.

        Used by ``should_replan`` to detect diminishing returns and by the
        Reflector to avoid repeating stale advice.
        """
        effectiveness = self.get_suggestion_effectiveness(fingerprint)
        return {cat for cat, rate in effectiveness.items() if rate == 0.0}

    def has_prior_replan(self, fingerprint: str) -> bool:
        """Return True if at least one replan has been recorded for this fingerprint."""
        record = self.data.get("datasets", {}).get(fingerprint)
        if record is None:
            return False
        return len(record.get("reflection_history", [])) > 0

    def get_prior_f1(self, fingerprint: str) -> Optional[float]:
        """
        Return the best F1 from the PREVIOUS run for this fingerprint.

        Used for diminishing returns detection: if current F1 is not
        meaningfully better than prior F1 despite replanning, stop.
        """
        record = self.data.get("datasets", {}).get(fingerprint)
        if record is None:
            return None
        history = record.get("reflection_history", [])
        if history:
            return float(history[-1].get("f1_before", 0.0))
        # Fall back to stored best_metrics if no history yet
        best = record.get("best_metrics", {})
        return float(best.get("f1_macro", 0.0)) if best else None

    # ── Similarity-based retrieval ───────────────────────────────────────────

    def get_similar_record(
        self,
        profile: Dict[str, Any],
        min_similarity: float = _MIN_SIMILARITY,
    ) -> Optional[Dict[str, Any]]:
        """Return the most similar stored record, or None if below threshold."""
        best_score  = -1.0
        best_record: Optional[Dict[str, Any]] = None

        for _fp, record in self.data.get("datasets", {}).items():
            sim = _compute_similarity(profile, record)
            if sim > best_score:
                best_score  = sim
                best_record = record

        if best_record is not None and best_score >= min_similarity:
            return {**best_record, "similarity_score": best_score}
        return None

    def get_hint(
        self,
        fingerprint: str,
        profile: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Return the best available memory hint for the Planner.

        Priority:
          1. Exact fingerprint match  → match_type: "exact"
          2. Similarity-based match   → match_type: "similar"
          3. None
        """
        exact = self.get_dataset_record(fingerprint)
        if exact:
            return {**exact, "match_type": "exact", "similarity_score": 1.0}

        similar = self.get_similar_record(profile)
        if similar:
            return {**similar, "match_type": "similar"}

        return None

    # ── Notes ────────────────────────────────────────────────────────────────

    def add_note(self, msg: str) -> None:
        self.data.setdefault("notes", []).append({"ts": now_iso(), "msg": msg})
        self.save()

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        n = len(self.data.get("datasets", {}))
        return f"JSONMemory: {n} dataset record(s) at '{self.path}'."

    def best_models_seen(self) -> List[str]:
        return list({
            rec.get("best_model", "")
            for rec in self.data.get("datasets", {}).values()
            if rec.get("best_model")
        })


#  Suggestion categorisation (used by meta-learning)
def _categorise_suggestions(suggestions: List[str]) -> List[str]:
    """
    Map free-text suggestion strings to coarse categories.

    Categories are used as stable keys in the effectiveness tracker so that
    similar suggestions across different runs are grouped together.

    Mapping heuristics (keyword-based):
      imbalance      - any mention of class_weight, oversampling, imbalance,
                       SMOTE, threshold
      feature_eng    - feature engineering, encoding, cardinality
      ensemble       - ensemble, GradientBoosting, ExtraTrees
      regularisation - regulariz, overfitting, complexity
      data_quality   - leakage, label, duplicate, correlated, missing
      preprocessing  - transform, skew, impute, scale
      cross_val      - cross-validation, stratified, fold
      more_data      - more examples, small dataset, collect
    """
    keywords: Dict[str, List[str]] = {
        "imbalance":      ["imbalance", "class_weight", "oversamp", "smote", "threshold"],
        "feature_eng":    ["feature engineer", "encoding", "cardinality", "ordinal"],
        "ensemble":       ["ensemble", "gradientboosting", "extratrees", "boosting"],
        "regularisation": ["regulariz", "overfitting", "complexity", "simpler"],
        "data_quality":   ["leakage", "label", "duplicate", "correlated", "missing"],
        "preprocessing":  ["transform", "skew", "impute", "scale", "power"],
        "cross_val":      ["cross-valid", "stratified", "fold"],
        "more_data":      ["more example", "small dataset", "collect"],
    }

    categories: List[str] = []
    combined = " ".join(suggestions).lower()
    for cat, kws in keywords.items():
        if any(kw in combined for kw in kws):
            categories.append(cat)
    return categories if categories else ["general"]