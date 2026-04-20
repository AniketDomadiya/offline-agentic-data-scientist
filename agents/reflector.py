"""
Reflector Agent
===============
Analyses execution results, diagnoses problems, generates prioritised actionable suggestions, and decides whether to trigger replanning.

What this module covers
-----------------------
1.  Dummy baseline + effect size (Cohen's h) + 95 % confidence interval.
    Uses Wilson interval (more accurate than normal approximation for proportions near 0 or 1).
2.  Absolute performance bands (weak / moderate thresholds).
3.  Suspiciously high performance (leakage detection).
4.  Imbalance bias  (accuracy >> balanced accuracy divergence).
5.  Per-class analysis: flag low-F1 classes and large F1 spread.
6.  Precision-recall tradeoff: flag conservative (high-P, low-R) or
    liberal (low-P, high-R) bias, with threshold-tuning advice.
7.  Overfitting / underfitting detection:
    - Uses cv_balanced_accuracy vs test balanced_accuracy when CV ran.
    - Overfitting: CV ba >> test ba by > 0.10 gap.
    - Underfitting: both CV and test ba are low (< 0.60).
    - Without CV: heuristic based on absolute performance level.
8.  Confusion matrix pattern analysis:
    - Which class-pair is most confused.
    - Any class with zero correct predictions (fully misclassified).
    - Whether the matrix is dominated by off-diagonal entries.
    Requires the raw CM ndarray passed in from evaluation.py.
9.  Model diversity: std of balanced-accuracy across non-dummy models.
10. Root cause diagnosis: synthesises all signals into a single
    structured cause label (majority_class_bias / weak_feature_signal /
    class_confusion / overfitting / underfitting / insufficient_data /
    data_quality / ok).
11. Data quality issue detection from profile signals (missing, skewness,
    correlations, duplicates still present).
12. Suggestion prioritisation by expected impact score (high / medium / low).
13. Meta-learning from past reflections:
    - Looks up which suggestion categories were tried before on this
      fingerprint (via JSONMemory) and whether they helped.
    - Downweights stale suggestions with a 0 % success rate.
    - Adds a memory note when a suggestion has previously failed.
14. Sophisticated ``should_replan`` policy:
    - Diminishing returns detection (F1 improvement < 0.02 after prior replan).
    - Memory check for failed strategies (don't repeat what didn't work).
    - Leakage suspicion blocks replan (it's a different class of problem).
    - Adaptive F1/BA thresholds adjusted for problem difficulty (n_classes).
15. ``apply_replan_strategy``: delegates plan to ``create_replan`` in
    planner.py (no duplicate logic); annotates profile with structured notes.
"""

import math
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from agents.planner import create_replan
from agents.memory  import JSONMemory, _categorise_suggestions


# Thresholds  (all in one place for easy tuning)
_MIN_IMPROVEMENT     = 0.05   # min balanced-acc gain over dummy to be "meaningful"
_F1_WEAK             = 0.60   # below → poor performance
_F1_MODERATE         = 0.75   # below → moderate performance
_BA_MODERATE         = 0.70   # balanced-accuracy moderate threshold
_PER_CLASS_F1_LOW    = 0.50   # per-class F1 below this → flag
_MODEL_STD_CONVERGED = 0.03   # std bal-acc below this → all models similar
_LEAKAGE_BA          = 0.97   # suspiciously high balanced-accuracy
_OVERFIT_GAP         = 0.10   # CV ba − test ba gap above this → overfitting
_PR_GAP              = 0.15   # |precision − recall| above this → P-R imbalance
_DIMINISHING_DELTA   = 0.02   # F1 improvement below this → diminishing returns
_CI_Z                = 1.96   # z-score for 95 % confidence interval


# Helper functions

# 1. Effect size and confidence interval 
def _compute_effect_size_and_ci(
    bal_acc: float,
    dummy_ba: float,
    n_test: int,
) -> Dict[str, Any]:
    """
    Compute Cohen's h effect size and 95 % Wilson confidence interval.

    Cohen's h measures the practical significance of the difference between
    two proportions (best model vs dummy) independently of sample size.
    It answers "is this improvement large enough to matter?" rather than
    just "is it non-zero?".

    Wilson interval is more accurate than the normal approximation near 0/1.

    Returns
    -------
    {
      "cohen_h"    : float   effect size (positive = improvement over dummy)
      "label"      : str     "negligible" | "small" | "medium" | "large"
      "ci_lower"   : float   lower bound of 95 % CI for bal_acc
      "ci_upper"   : float   upper bound of 95 % CI for bal_acc
    }
    """
    # Cohen's h: h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))
    h = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, bal_acc)))) \
      - 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, dummy_ba))))

    if abs(h) < 0.20:
        label = "negligible"
    elif abs(h) < 0.50:
        label = "small"
    elif abs(h) < 0.80:
        label = "medium"
    else:
        label = "large"

    # Wilson confidence interval
    z2  = _CI_Z ** 2
    n   = max(n_test, 1)
    p   = bal_acc
    center    = (n * p + z2 / 2) / (n + z2)
    half_width = _CI_Z * math.sqrt(max(0.0, n * p * (1 - p) + z2 / 4)) / (n + z2)
    ci_lower   = max(0.0, round(center - half_width, 4))
    ci_upper   = min(1.0, round(center + half_width, 4))

    return {
        "cohen_h":  round(h, 4),
        "label":    label,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# 2. Classification report parser
def _parse_classification_report(report_str: str) -> Dict[str, Dict[str, float]]:
    """
    Parse sklearn's classification_report string into per-class metric dicts.

    Returns {class_label: {"precision", "recall", "f1", "support"}}.
    Skips summary rows (accuracy, macro avg, weighted avg).
    Handles multi-word class labels gracefully.
    """
    class_metrics: Dict[str, Dict[str, float]] = {}
    if not report_str:
        return class_metrics

    _SKIP = {"accuracy", "macro avg", "weighted avg"}
    for line in report_str.strip().split("\n")[2:]:
        parts = line.split()
        if not parts:
            continue
        nums: List[float] = []
        for tok in reversed(parts):
            try:
                nums.insert(0, float(tok))
                if len(nums) == 4:
                    break
            except ValueError:
                if nums:
                    break
        label = " ".join(parts[: len(parts) - len(nums)]).strip()
        if label in _SKIP or not label or len(nums) < 4:
            continue
        class_metrics[label] = {
            "precision": nums[0],
            "recall":    nums[1],
            "f1":        nums[2],
            "support":   nums[3],
        }
    return class_metrics

# 3. Per-class performance analysis 
def _analyse_per_class(
    class_metrics: Dict[str, Dict[str, float]],
    n_classes: int,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Return (issues, suggestions) from per-class metric analysis.

    Each item is a (text, impact_score) tuple for prioritisation.
    """
    issues:      List[Tuple[str, float]] = []
    suggestions: List[Tuple[str, float]] = []

    if not class_metrics:
        return issues, suggestions

    worst_label, worst_m = min(class_metrics.items(), key=lambda x: x[1]["f1"])
    best_label,  best_m  = max(class_metrics.items(), key=lambda x: x[1]["f1"])

    # Flag classes with very low F1
    low_f1 = [
        (lbl, m["f1"], int(m["support"]))
        for lbl, m in class_metrics.items()
        if m["f1"] < _PER_CLASS_F1_LOW
    ]
    if low_f1:
        names = ", ".join(f"'{l}' (F1={f:.2f}, n={s})" for l, f, s in low_f1)
        issues.append((
            f"Poor per-class F1 on: {names}. "
            "Model fails to learn these classes reliably.",
            0.85,
        ))
        suggestions.append((
            "Minority-class improvement: try class-specific threshold tuning, "
            "oversampling the low-support class, or collecting more labelled "
            "examples for the under-represented categories.",
            0.85,
        ))

    # Large F1 spread (multi-class only)
    if n_classes > 2:
        gap = best_m["f1"] - worst_m["f1"]
        if gap > 0.30:
            issues.append((
                f"Large per-class F1 spread: best='{best_label}' ({best_m['f1']:.2f}) "
                f"vs worst='{worst_label}' ({worst_m['f1']:.2f}), gap={gap:.2f}.",
                0.70,
            ))
            suggestions.append((
                "Large cross-class gap: review class label definitions, inspect the "
                "confusion matrix for specific confusion patterns, and consider "
                "class-specific feature engineering.",
                0.70,
            ))

    return issues, suggestions


# 4. Precision-recall tradeoff
def _analyse_precision_recall_tradeoff(
    evaluation: Dict[str, Any],
) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Detect whether the model has a conservative or liberal prediction bias.

    Conservative (precision >> recall)
        The model requires strong evidence before predicting a non-majority
        class. Many true positives are missed (high false-negative rate).
        Cause: the decision threshold is effectively too high.

    Liberal (recall >> precision)
        The model over-predicts non-majority classes. Many false positives.
        Cause: the decision threshold is effectively too low.

    Returns
    -------
    (note_str, suggestions)
        note_str: short human-readable diagnosis string (or "")
        suggestions: list of (text, impact) tuples
    """
    prec  = float(evaluation.get("precision_macro", 0.0))
    rec   = float(evaluation.get("recall_macro", 0.0))
    gap   = prec - rec
    suggestions: List[Tuple[str, float]] = []
    note  = ""

    if gap > _PR_GAP:
        note = (
            f"Conservative bias: macro precision ({prec:.3f}) >> "
            f"recall ({rec:.3f}). Model misses many true positives."
        )
        suggestions.append((
            "Conservative bias detected: lower the decision threshold for the "
            "minority class (e.g. predict class X if P(X) > 0.3 instead of 0.5) "
            "to recover missed true positives.",
            0.65,
        ))
    elif -gap > _PR_GAP:
        note = (
            f"Liberal bias: macro recall ({rec:.3f}) >> "
            f"precision ({prec:.3f}). Many false positives."
        )
        suggestions.append((
            "Liberal bias detected: raise the decision threshold or add stronger "
            "regularisation to reduce false positives.",
            0.60,
        ))

    return note, suggestions


# 5. Overfitting / underfitting detection
def _detect_overfitting_underfitting(
    all_metrics: List[Dict[str, Any]],
    bal_acc: float,
    f1_macro: float,
) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Detect overfitting or underfitting from available signals.

    Primary signal: cv_balanced_accuracy (available when StratifiedKFold ran)
    vs test balanced_accuracy.
      - cv_ba >> test_ba (gap > 0.10) → overfitting: model memorised training data.
      - Both are low (< _F1_WEAK)     → underfitting: model too simple.

    Fallback (no CV): heuristic based on absolute test performance.
      - If test ba < 0.60 without other explanatory signals → likely underfitting.

    Returns
    -------
    (diagnosis_str, suggestions)
        diagnosis_str: "overfitting" | "underfitting" | "ok" | ""
    """
    suggestions: List[Tuple[str, float]] = []
    diagnosis = ""

    # Collect CV metrics from non-dummy models
    cv_values = [
        (float(m.get("cv_balanced_accuracy", -1)), float(m.get("balanced_accuracy", 0)))
        for m in all_metrics
        if "Dummy" not in m.get("model", "") and m.get("cv_balanced_accuracy") is not None
    ]

    if cv_values:
        cv_mean   = statistics.mean(v[0] for v in cv_values if v[0] >= 0)
        test_mean = statistics.mean(v[1] for v in cv_values)

        if cv_mean - test_mean > _OVERFIT_GAP:
            diagnosis = "overfitting"
            suggestions.append((
                f"Overfitting detected: CV balanced-accuracy ({cv_mean:.3f}) >> "
                f"test balanced-accuracy ({test_mean:.3f}). "
                "Add regularisation (C↓ for LR, max_depth for trees) or collect more data.",
                0.80,
            ))
        elif cv_mean < _F1_WEAK and test_mean < _F1_WEAK:
            diagnosis = "underfitting"
            suggestions.append((
                f"Underfitting detected: both CV ({cv_mean:.3f}) and test "
                f"({test_mean:.3f}) balanced-accuracy are low. "
                "Try more complex models, more features, or reduce regularisation.",
                0.75,
            ))
    else:
        # No CV available - heuristic
        if bal_acc < _F1_WEAK and f1_macro < _F1_WEAK:
            diagnosis = "underfitting"
            suggestions.append((
                "Both balanced-accuracy and macro-F1 are low without CV evidence. "
                "Possible underfitting: consider more expressive models or richer features.",
                0.55,
            ))

    return diagnosis, suggestions


# 6. Confusion matrix pattern analysis
def _analyse_confusion_matrix(
    cm: Optional[np.ndarray],
    labels: List[str],
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Analyse a confusion matrix for problematic patterns.

    Checks performed
    ----------------
    1. Most confused pair: off-diagonal cell with highest count.
    2. Zero-correct class: any class where the diagonal is 0 (fully misclassified).
    3. Off-diagonal dominance: fraction of total predictions that are wrong.

    Parameters
    ----------
    cm     : numpy ndarray of shape (n_classes, n_classes), or None.
    labels : List of class label strings aligned with cm rows/cols.

    Returns
    -------
    (issues, suggestions) - each item is a (text, impact_score) tuple.
    """
    issues:      List[Tuple[str, float]] = []
    suggestions: List[Tuple[str, float]] = []

    if cm is None or cm.size == 0 or len(labels) < 2:
        return issues, suggestions

    n_total   = int(cm.sum())
    n_correct = int(np.trace(cm))
    error_rate = 1.0 - (n_correct / max(n_total, 1))

    # 1. Most confused class pair
    cm_off = cm.astype(float).copy()
    np.fill_diagonal(cm_off, 0)
    max_idx = np.unravel_index(np.argmax(cm_off), cm_off.shape)
    max_val = int(cm_off[max_idx])
    if max_val > 0:
        true_lbl = labels[max_idx[0]] if max_idx[0] < len(labels) else str(max_idx[0])
        pred_lbl = labels[max_idx[1]] if max_idx[1] < len(labels) else str(max_idx[1])
        issues.append((
            f"Most confused pair: true='{true_lbl}' predicted as '{pred_lbl}' "
            f"{max_val} times. These classes may share overlapping features.",
            0.65,
        ))
        suggestions.append((
            f"Class confusion '{true_lbl}'→'{pred_lbl}': inspect feature distributions "
            "for these two classes, consider adding discriminative features, or review "
            "whether the class labels are consistently defined.",
            0.65,
        ))

    # 2. Fully misclassified class (zero on diagonal)
    zero_classes = [
        labels[i]
        for i in range(min(cm.shape[0], len(labels)))
        if cm[i, i] == 0 and cm[i, :].sum() > 0
    ]
    if zero_classes:
        issues.append((
            f"Class(es) with ZERO correct predictions: {zero_classes}. "
            "The model never predicts these classes correctly.",
            0.90,
        ))
        suggestions.append((
            f"Completely misclassified class(es) {zero_classes}: ensure sufficient "
            "training examples, verify the label is not mislabelled, and consider "
            "class_weight='balanced' or targeted oversampling.",
            0.90,
        ))

    # 3. High overall error rate
    if error_rate > 0.50:
        issues.append((
            f"Overall confusion matrix error rate is {error_rate:.1%}: "
            "more than half of all predictions are wrong.",
            0.75,
        ))

    return issues, suggestions


# 7. Model diversity
def _analyse_model_diversity(
    all_metrics: List[Dict[str, Any]],
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], float]:
    """
    Return (issues, suggestions, spread) from model diversity analysis.
    spread = std of balanced-accuracy across non-dummy models.
    """
    issues:      List[Tuple[str, float]] = []
    suggestions: List[Tuple[str, float]] = []

    non_dummy = [m for m in all_metrics if "Dummy" not in m.get("model", "")]
    if len(non_dummy) < 2:
        return issues, suggestions, 0.0

    ba_values = [float(m.get("balanced_accuracy", 0)) for m in non_dummy]
    try:
        spread = round(statistics.stdev(ba_values), 4)
    except Exception:
        return issues, suggestions, 0.0

    if spread < _MODEL_STD_CONVERGED:
        issues.append((
            f"All non-dummy models converge to similar balanced-accuracy "
            f"(std={spread:.4f}): likely a data quality issue or ceiling effect.",
            0.70,
        ))
        suggestions.append((
            "Model convergence: investigate target leakage, verify label "
            "correctness, and review whether features carry real predictive signal.",
            0.70,
        ))

    return issues, suggestions, spread


# 8. Data quality issue detection
def _detect_data_quality_issues(
    dataset_profile: Dict[str, Any],
    plan: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """
    Flag data quality issues from profile signals that may explain poor performance.

    Checks
    ------
    - High missing values in important columns (>20 %)
    - Unaddressed highly skewed features (TASK_SKEW_TRANSFORM not in plan)
    - Unaddressed highly correlated pairs (TASK_DROP_CORRELATED not in plan)
    - Duplicate rows not removed (TASK_DROP_DUPES not in plan)

    Returns list of (suggestion_text, impact_score) tuples.
    """
    suggestions: List[Tuple[str, float]] = []
    plan = plan or []

    missing_pct = dataset_profile.get("missing_pct", {})
    high_missing = {
        col: pct for col, pct in missing_pct.items()
        if pct > 20.0
    }
    if high_missing:
        cols = ", ".join(f"'{c}' ({p:.1f}%)" for c, p in list(high_missing.items())[:3])
        suggestions.append((
            f"Data quality: {len(high_missing)} column(s) have >20 % missing values "
            f"({cols}). Consider more sophisticated imputation (KNN, iterative).",
            0.65,
        ))

    skewed = dataset_profile.get("highly_skewed_features", [])
    if skewed and "apply_skew_transform" not in plan:
        suggestions.append((
            f"Data quality: {len(skewed)} highly skewed feature(s) ({skewed[:3]}) "
            "were not transformed. PowerTransformer (Yeo-Johnson) may improve "
            "performance of linear and distance-based models.",
            0.55,
        ))

    corr_pairs = dataset_profile.get("high_corr_pairs", [])
    if corr_pairs and "drop_correlated" not in plan:
        suggestions.append((
            f"Data quality: {len(corr_pairs)} highly correlated feature pair(s) "
            "present. Dropping redundant columns may reduce noise and improve "
            "generalisation.",
            0.50,
        ))

    dupes = dataset_profile.get("duplicate_count", 0)
    if dupes > 0 and "drop_duplicates" not in plan:
        suggestions.append((
            f"Data quality: {dupes} duplicate rows were not removed. "
            "Duplicates can inflate training metrics and cause data leakage.",
            0.70,
        ))

    return suggestions


# 9. Root cause diagnosis
def _diagnose_root_cause(
    bal_acc: float,
    f1_macro: float,
    accuracy: float,
    imb: float,
    improvement_over_dummy: Optional[float],
    model_spread: float,
    overfit_diagnosis: str,
    zero_classes: bool,
    n_classes: int,
    rows: int,
) -> str:
    """
    Synthesise all signals into one structured root-cause label.

    Precedence
    ----------
    1. majority_class_bias - imbalance + accuracy >> balanced_accuracy
    2. overfitting         - CV >> test gap
    3. underfitting        - both CV and test are low
    4. zero_class          - at least one class never predicted correctly
    5. weak_feature_signal - improvement over dummy < 0.05
    6. data_quality        - all models converge (std < 0.03)
    7. insufficient_data   - small dataset + poor performance
    8. class_confusion     - multi-class + per-class spread > 0.30
    9. ok                  - none of the above

    This label is stored in the reflection dict and in the markdown report
    so the agent's diagnosis is auditable.
    """
    if imb >= 3.0 and accuracy > bal_acc + 0.10:
        return "majority_class_bias"
    if overfit_diagnosis == "overfitting":
        return "overfitting"
    if overfit_diagnosis == "underfitting":
        return "underfitting"
    if zero_classes:
        return "zero_class"
    if improvement_over_dummy is not None and improvement_over_dummy < _MIN_IMPROVEMENT:
        return "weak_feature_signal"
    if model_spread < _MODEL_STD_CONVERGED and f1_macro < _F1_MODERATE:
        return "data_quality"
    if rows < 500 and f1_macro < _F1_WEAK:
        return "insufficient_data"
    if n_classes > 2 and f1_macro < _F1_MODERATE:
        return "class_confusion"
    return "ok"


# 10. Suggestion prioritisation
def _prioritize_suggestions(
    raw_suggestions: List[Tuple[str, float]],
) -> List[str]:
    """
    Sort (suggestion_text, impact_score) tuples by score descending.

    Returns a plain list of suggestion strings for the reflection output.
    Impact scores are assigned by each generator function (0–1 scale):
      0.90 + : critical (zero-class, severe overfitting)
      0.70-0.89: high impact
      0.50-0.69: medium impact
      < 0.50  : low / informational
    """
    sorted_sug = sorted(raw_suggestions, key=lambda x: x[1], reverse=True)
    return [text for text, _ in sorted_sug]

# 11. Meta-learning: learn from past reflections
def _learn_from_memory(
    memory: Optional[JSONMemory],
    fingerprint: str,
    current_suggestions_raw: List[Tuple[str, float]],
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Adjust suggestion list based on past suggestion effectiveness from memory.

    If a suggestion category has a 0 % success rate for this fingerprint,
    its impact score is reduced and a memory note is added warning that
    this approach has been tried before without improvement.

    Returns
    -------
    (adjusted_suggestions, memory_notes)
        adjusted_suggestions : suggestion list with impact scores lowered for
                               previously-failed categories
        memory_notes         : list of note strings about past failures
    """
    if memory is None or not fingerprint:
        return current_suggestions_raw, []

    effectiveness  = memory.get_suggestion_effectiveness(fingerprint)
    failed_cats    = {cat for cat, rate in effectiveness.items() if rate == 0.0}
    memory_notes: List[str] = []

    if not failed_cats:
        return current_suggestions_raw, []

    adjusted: List[Tuple[str, float]] = []
    for text, score in current_suggestions_raw:
        # Categorise this suggestion
        cats = _categorise_suggestions([text])
        if any(c in failed_cats for c in cats):
            # Reduce impact score - still show suggestion but ranked lower
            adjusted.append((text, score * 0.40))
            cats_str = ", ".join(c for c in cats if c in failed_cats)
            memory_notes.append(
                f"[Memory] Suggestion category '{cats_str}' was tried in a prior "
                "replan for this dataset without meaningful improvement. "
                "Consider a different strategy."
            )
        else:
            adjusted.append((text, score))

    return adjusted, memory_notes


# 12. Diminishing returns check
def _check_diminishing_returns(
    memory: Optional[JSONMemory],
    fingerprint: str,
    replan_count: int,
    current_f1: float,
) -> bool:
    """
    Return True if replanning is unlikely to help further.

    A diminishing returns situation is detected when:
      - A replan has already happened (replan_count > 0), AND
      - The F1 improvement over the prior run is < _DIMINISHING_DELTA (0.02).

    This prevents the agent from thrashing in an infinite improvement loop
    when the data fundamentally limits achievable performance.
    """
    if replan_count == 0 or memory is None:
        return False

    prior_f1 = memory.get_prior_f1(fingerprint)
    if prior_f1 is None:
        return False

    improvement = current_f1 - prior_f1
    return improvement < _DIMINISHING_DELTA


def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    all_metrics: List[Dict[str, Any]],
    classification_report_str: str = "",
    confusion_matrix: Optional[np.ndarray] = None,
    confusion_matrix_labels: Optional[List[str]] = None,
    replan_count: int = 0,
    memory: Optional[JSONMemory] = None,
    fingerprint: str = "",
    plan: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyse execution results and produce a structured, prioritised reflection.

    Parameters
    ----------
    dataset_profile           : Output of ``profile_dataset()``.
    evaluation                : Best model's metrics dict.
    all_metrics               : Metrics for all trained models.
    classification_report_str : sklearn classification_report string.
    confusion_matrix          : Raw confusion matrix ndarray from evaluation.py.
    confusion_matrix_labels   : Class label strings aligned with CM rows/cols.
    replan_count              : Number of replans already performed this run.
    memory                    : JSONMemory instance for meta-learning lookups.
    fingerprint               : Dataset fingerprint for memory lookups.
    plan                      : Current plan list (for data-quality checks).

    Returns
    -------
    dict with keys:
      status, best_model, balanced_accuracy, f1_macro,
      issues, suggestions,
      replan_recommended, diminishing_returns,
      per_class_metrics, model_spread,
      confidence_interval_95, effect_size_cohen_h, effect_size_label,
      overfit_underfitting_diagnosis, precision_recall_note,
      root_cause, memory_notes
    """
    best_model = evaluation.get("model", "Unknown")
    bal_acc    = float(evaluation.get("balanced_accuracy", 0.0))
    f1_macro   = float(evaluation.get("f1_macro", 0.0))
    accuracy   = float(evaluation.get("accuracy", 0.0))
    imb        = float(dataset_profile.get("imbalance_ratio") or 1.0)
    n_classes  = int(dataset_profile.get("n_classes", 2))
    rows       = dataset_profile["shape"]["rows"]

    # All issues and suggestions are (text, impact_score) tuples until
    # the end where they are flattened and sorted.
    raw_issues:      List[Tuple[str, float]] = []
    raw_suggestions: List[Tuple[str, float]] = []

    # 1. Dummy baseline + effect size + confidence interval
    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)
    improvement_over_dummy: Optional[float] = None
    effect_info: Dict[str, Any] = {}

    if dummy is not None:
        dummy_ba    = float(dummy.get("balanced_accuracy", 0.0))
        improvement = bal_acc - dummy_ba
        improvement_over_dummy = improvement

        # Estimate n_test from per-class supports
        per_class_metrics = _parse_classification_report(classification_report_str)
        n_test = int(sum(m.get("support", 0) for m in per_class_metrics.values()))
        if n_test == 0:
            n_test = max(int(rows * 0.2), 10)  # fallback estimate

        effect_info = _compute_effect_size_and_ci(bal_acc, dummy_ba, n_test)

        if improvement < _MIN_IMPROVEMENT:
            raw_issues.append((
                f"Best model improves only {improvement:.3f} over dummy "
                f"(best={bal_acc:.3f} vs dummy={dummy_ba:.3f}): very weak signal. "
                f"Cohen's h={effect_info['cohen_h']:.3f} ({effect_info['label']}).",
                0.90,
            ))
            raw_suggestions.append((
                "Marginal gain over dummy: check for target leakage, label noise, "
                "or verify features carry information about the target.",
                0.90,
            ))
        elif effect_info["label"] == "small":
            raw_suggestions.append((
                f"Improvement over dummy is statistically real "
                f"(Cohen's h={effect_info['cohen_h']:.3f}) but small. "
                "More feature engineering may increase effect size.",
                0.50,
            ))
    else:
        per_class_metrics = _parse_classification_report(classification_report_str)
        n_test = max(int(rows * 0.2), 10)

    # 2. Absolute performance bands
    if f1_macro < _F1_WEAK:
        raw_issues.append((
            f"Macro F1={f1_macro:.3f} is below the weak threshold ({_F1_WEAK}). "
            "Overall classification performance is poor.",
            0.85,
        ))
        raw_suggestions.append((
            "Low macro F1: try GradientBoosting/ExtraTrees, additional feature "
            "engineering, or hyperparameter tuning.",
            0.80,
        ))
    elif f1_macro < _F1_MODERATE:
        raw_suggestions.append((
            f"Macro F1={f1_macro:.3f} is moderate (<{_F1_MODERATE}). "
            "Ensemble methods or targeted hyperparameter search may help.",
            0.55,
        ))

    if bal_acc < _BA_MODERATE and rows >= 200:
        raw_issues.append((
            f"Balanced accuracy={bal_acc:.3f} < {_BA_MODERATE}: model may not be "
            "learning a robust decision boundary.",
            0.80,
        ))

    # 3. Leakage suspicion
    if bal_acc >= _LEAKAGE_BA:
        raw_issues.append((
            f"Balanced accuracy={bal_acc:.3f} ≥ {_LEAKAGE_BA}: suspiciously high. "
            "Possible target leakage or trivially separable classes.",
            0.95,
        ))
        raw_suggestions.append((
            "Investigate target leakage: ensure no feature directly encodes "
            "or derives from the target label.",
            0.95,
        ))

    # 4. Imbalance bias
    if imb >= 3.0:
        if accuracy > bal_acc + 0.10:
            raw_issues.append((
                f"Accuracy ({accuracy:.3f}) >> balanced accuracy ({bal_acc:.3f}): "
                "model likely predicting majority class most of the time.",
                0.85,
            ))
        raw_suggestions.append((
            f"Imbalance ratio={imb:.1f}: verify class_weight='balanced' is active; "
            "consider decision-threshold tuning for the minority class.",
            0.70,
        ))

    # 5. Per-class analysis
    cls_issues, cls_suggestions = _analyse_per_class(per_class_metrics, n_classes)
    raw_issues.extend(cls_issues)
    raw_suggestions.extend(cls_suggestions)

    # 6. Precision-recall tradeoff
    pr_note, pr_suggestions = _analyse_precision_recall_tradeoff(evaluation)
    raw_suggestions.extend(pr_suggestions)

    # 7. Overfitting / underfitting
    overfit_diagnosis, overfit_suggestions = _detect_overfitting_underfitting(
        all_metrics, bal_acc, f1_macro
    )
    raw_suggestions.extend(overfit_suggestions)

    # 8. Confusion matrix patterns
    cm_issues, cm_suggestions = _analyse_confusion_matrix(
        confusion_matrix,
        confusion_matrix_labels or [],
    )
    raw_issues.extend(cm_issues)
    raw_suggestions.extend(cm_suggestions)

    # Track whether any class has zero correct predictions (used in root cause)
    zero_class_present = any(
        "ZERO correct" in text for text, _ in cm_issues
    )

    # 9. Model diversity
    div_issues, div_suggestions, model_spread = _analyse_model_diversity(all_metrics)
    raw_issues.extend(div_issues)
    raw_suggestions.extend(div_suggestions)

    # 10. Data quality issues from profile
    dq_suggestions = _detect_data_quality_issues(dataset_profile, plan)
    raw_suggestions.extend(dq_suggestions)

    # 11. Small-dataset reliability
    if rows < 500 and f1_macro < _F1_MODERATE:
        raw_suggestions.append((
            f"Small dataset ({rows} rows): single train-test split metrics may be "
            "unreliable. StratifiedKFold cross-validation recommended.",
            0.60,
        ))

    # 12. Top-2 model tie
    ranked = sorted(
        all_metrics,
        key=lambda m: float(m.get("balanced_accuracy", 0)),
        reverse=True,
    )
    if len(ranked) >= 2:
        gap = float(ranked[0].get("balanced_accuracy", 0)) - float(ranked[1].get("balanced_accuracy", 0))
        if gap < 0.01:
            raw_suggestions.append((
                f"'{best_model}' and '{ranked[1].get('model')}' are nearly tied "
                f"(Δbal_acc < 0.01): the simpler model may be preferable.",
                0.40,
            ))

    # 13. Root cause diagnosis
    root_cause = _diagnose_root_cause(
        bal_acc=bal_acc,
        f1_macro=f1_macro,
        accuracy=accuracy,
        imb=imb,
        improvement_over_dummy=improvement_over_dummy,
        model_spread=model_spread,
        overfit_diagnosis=overfit_diagnosis,
        zero_classes=zero_class_present,
        n_classes=n_classes,
        rows=rows,
    )

    # 14. Meta-learning from memory
    raw_suggestions, memory_notes = _learn_from_memory(
        memory, fingerprint, raw_suggestions
    )

    # 15. Prioritise suggestions
    issues_text     = _prioritize_suggestions(raw_issues)
    suggestions_text = _prioritize_suggestions(raw_suggestions)

    # 16. Diminishing returns check
    diminishing_returns = _check_diminishing_returns(
        memory, fingerprint, replan_count, f1_macro
    )

    # 17. Replan decision (sophisticated)
    # Adaptive F1 threshold: harder problems (many classes) get a lower bar
    adaptive_f1_threshold  = max(0.55, _F1_MODERATE - 0.03 * max(0, n_classes - 2))
    adaptive_ba_threshold  = max(0.55, _BA_MODERATE - 0.02 * max(0, n_classes - 2))

    replan_recommended = (
        bool(issues_text)                   # at least one issue identified
        and f1_macro  < adaptive_f1_threshold   # performance genuinely poor
        and bal_acc   < adaptive_ba_threshold   # not just a metric artefact
        and bal_acc   < _LEAKAGE_BA             # not a leakage situation
        and not diminishing_returns             # not stuck in a loop
    )

    status = "needs_attention" if issues_text else "ok"

    return {
        # Core
        "status":               status,
        "best_model":           best_model,
        "balanced_accuracy":    bal_acc,
        "f1_macro":             f1_macro,
        "issues":               issues_text,
        "suggestions":          suggestions_text,
        "replan_recommended":   replan_recommended,
        # Statistical context
        "confidence_interval_95": {
            "lower": effect_info.get("ci_lower", 0.0),
            "upper": effect_info.get("ci_upper", 1.0),
        },
        "effect_size_cohen_h":  effect_info.get("cohen_h"),
        "effect_size_label":    effect_info.get("label", ""),
        # Diagnostics
        "per_class_metrics":            per_class_metrics,
        "model_spread":                 model_spread,
        "overfit_underfitting_diagnosis": overfit_diagnosis,
        "precision_recall_note":        pr_note,
        "root_cause":                   root_cause,
        # Meta-learning
        "memory_notes":       memory_notes,
        "diminishing_returns": diminishing_returns,
    }


def should_replan(reflection: Dict[str, Any]) -> bool:
    """
    Decide whether to trigger a new execution cycle based on the reflection.

    Sophisticated policy
    --------------------
    Returns True only when ALL of the following hold:
      1. ``replan_recommended`` flag is set (set by reflect() using adaptive
         thresholds adjusted for problem difficulty).
      2. At least one concrete issue was identified.
      3. Diminishing returns NOT detected (prior replan already tried and
         F1 improvement was negligible - no point trying again).
      4. Not a leakage situation (near-perfect balanced accuracy is a
         different class of problem that replanning won't fix).

    Note: the max_replans budget check is handled in the Orchestrator.
    """
    if not reflection.get("replan_recommended", False):
        return False
    if not reflection.get("issues"):
        return False
    if reflection.get("diminishing_returns", False):
        return False
    if float(reflection.get("balanced_accuracy", 0)) >= _LEAKAGE_BA:
        return False
    return True


def apply_replan_strategy(
    plan: List[str],
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Return a (revised_plan, updated_profile) pair for the next execution cycle.

    Plan
    ----
    Delegates entirely to ``create_replan`` in ``agents.planner``.
    There is NO duplicate plan-editing logic here - the Planner owns all
    plan construction and revision logic.

    Profile annotations
    -------------------
    Appends structured notes to explain what triggered the replan and what
    strategy is being applied. These appear in the markdown report and ensure
    the agent's decisions are auditable.

    Strategy labels (stored in notes for auditability)
    --------------------------------------------------
    conservative  - only the single highest-impact change
    standard      - standard replan (imbalance + ensemble if needed)
    aggressive    - all possible improvements at once (low F1 + many issues)

    The actual strategy implementation is in ``create_replan``; the label
    here is descriptive metadata.
    """
    new_plan    = create_replan(plan, dataset_profile, reflection)
    new_profile = dict(dataset_profile)
    notes       = list(new_profile.get("notes", []))

    f1       = float(reflection.get("f1_macro", 1.0))
    imb      = float(dataset_profile.get("imbalance_ratio") or 1.0)
    spread   = float(reflection.get("model_spread", 1.0))
    issues   = reflection.get("issues", [])
    root     = reflection.get("root_cause", "unknown")
    n_issues = len(issues)
    replan_count = sum(1 for t in new_plan if t == "replan_attempt")

    # Choose strategy label based on problem severity
    if n_issues >= 3 and f1 < 0.50:
        strategy = "aggressive"
    elif n_issues == 1 or f1 >= 0.60:
        strategy = "conservative"
    else:
        strategy = "standard"

    notes.append(
        f"Replan #{replan_count} [{strategy}]: triggered by {n_issues} issue(s). "
        f"Root cause='{root}'. F1={f1:.3f}, bal_acc={reflection.get('balanced_accuracy', 0):.3f}."
    )

    # Specific strategy notes keyed to root cause
    root_notes: Dict[str, str] = {
        "majority_class_bias":  "Replan: imbalance strategy injected or strengthened.",
        "overfitting":          "Replan: model complexity will be reduced (fewer estimators).",
        "underfitting":         "Replan: switching to more expressive ensemble models.",
        "weak_feature_signal":  "Replan: model convergence - investigating data quality.",
        "data_quality":         "Replan: correlated-feature drop added to break ceiling.",
        "class_confusion":      "Replan: ensemble emphasis to better separate confused classes.",
        "zero_class":           "Replan: class_weight='balanced' enforced for zero-recall class.",
        "insufficient_data":    "Replan: cross-validation added for more reliable estimates.",
    }
    if root in root_notes:
        notes.append(root_notes[root])

    # Memory-based note
    memory_notes = reflection.get("memory_notes", [])
    notes.extend(memory_notes)

    new_profile["notes"] = notes
    return new_plan, new_profile