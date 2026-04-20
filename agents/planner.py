"""
Planner Agent
=============
Generates a context-aware, conditional execution plan for the Agentic Data Scientist.

Design principles
-----------------
* A primary scenario is detected first to give the plan a clear narrative.
* Signal-based tasks are then layered on top of the scenario template.
* Every task constant is a string so the Orchestrator can check membership with ``if TASK_X in plan`` without magic strings anywhere else.

Plan task reference
-------------------
Core (always present)
  profile_dataset           - Confirms schema and extracts all signals.
  build_preprocessor        - Builds ColumnTransformer pipeline.
  select_models             - Picks candidate classifiers.
  train_models              - Trains all candidates.
  evaluate                  - Picks best; saves confusion matrix.
  reflect                   - Analyses results; decides whether to replan.
  write_report              - Saves markdown report and artefacts.

Data preparation (conditional - executed before train/test split)
  drop_duplicates           - Duplicate rows detected (leakage risk).
  drop_id_columns           - Identifier columns detected (no signal).
  drop_severe_missing       - A column is >40 % missing; drop that column.
  extract_datetime          - Datetime columns → extract year/month features.
  drop_correlated           - High Pearson pairs (|r|>0.8); drop one per pair.
  handle_text_features      - Text-like columns (will be excluded, logged).

Preprocessing strategy (consumed by build_preprocessor)
  impute_numeric_mean       - No skewness → mean imputation for numerics.
                              Default (no task) = median (safer for skewed data).
  apply_skew_transform      - ≥1 highly skewed feature → PowerTransformer.
  handle_high_cardinality   - High-cardinality categoricals → OrdinalEncoder.
  consider_imbalance        - Ratio 3–10 → class_weight='balanced'.
  consider_severe_imbalance - Ratio >10 → balanced weights + ensemble-only.

Training strategy (conditional)
  use_cross_validation      - Dataset <500 rows → StratifiedKFold.
  prioritize_model:<name>   - Memory hint → move named model to front.
  emphasize_ensemble        - Replan after very low F1 → skip LR, push GB/ET.
  replan_attempt            - Marks this as a revised execution cycle.

Scenario tags (informational, embedded in plan after profile_dataset)
  scenario:tiny             - <200 rows
  scenario:small            - 200–499 rows
  scenario:large            - ≥50 000 rows
  scenario:high_dim         - >100 columns
  scenario:heavy_missing    - max column missingness >20 %
  scenario:severe_imb       - imbalance ratio >10
  scenario:imbalanced       - imbalance ratio 3–10
  scenario:standard         - none of the above
"""

from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Task-name constants
# ═══════════════════════════════════════════════════════════════════════════

# Core
TASK_PROFILE           = "profile_dataset"
TASK_BUILD_PRE         = "build_preprocessor"
TASK_SELECT_MODELS     = "select_models"
TASK_TRAIN             = "train_models"
TASK_EVALUATE          = "evaluate"
TASK_REFLECT           = "reflect"
TASK_REPORT            = "write_report"

# Data preparation
TASK_DROP_DUPES        = "drop_duplicates"
TASK_DROP_ID           = "drop_id_columns"
TASK_DROP_SEVERE_MISS  = "drop_severe_missing"
TASK_EXTRACT_DATETIME  = "extract_datetime"
TASK_DROP_CORRELATED   = "drop_correlated"
TASK_HANDLE_TEXT       = "handle_text_features"

# Preprocessing strategy
TASK_IMPUTE_MEAN       = "impute_numeric_mean"
TASK_SKEW_TRANSFORM    = "apply_skew_transform"
TASK_HIGH_CARD         = "handle_high_cardinality"
TASK_IMBALANCE         = "consider_imbalance"
TASK_IMBALANCE_SEVERE  = "consider_severe_imbalance"

# Training strategy
TASK_CROSS_VAL         = "use_cross_validation"
TASK_REPLAN            = "replan_attempt"

# Scenario prefix (informational)
_SCENARIO_PREFIX       = "scenario:"

# Thresholds
_IMBALANCE_MILD    = 3.0    # ratio ≥ → apply balanced class weights
_IMBALANCE_SEVERE  = 10.0   # ratio ≥ → severe strategy (ensemble-only)
_MISSING_SEVERE    = 40.0   # column missing % above → drop that column
_MISSING_MODERATE  = 20.0   # column missing % above → heavy-missing scenario
_TINY_ROWS         = 200    # < → scenario:tiny
_SMALL_ROWS        = 500    # < → scenario:small (also triggers CV)
_LARGE_ROWS        = 50_000 # ≥ → scenario:large
_HIGH_DIM_COLS     = 100    # > → scenario:high_dim


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════
def _detect_scenario(dataset_profile: Dict[str, Any]) -> str:
    """
    Return the single most dominant scenario label.

    Precedence (highest to lowest):
      tiny        < 200 rows           - dominates; all strategies limited
      severe_imb  imbalance > 10       - model selection drastically changes
      high_dim    cols > 100           - encoding strategy dominates
      heavy_miss  max missing > 20 %   - imputation focus
      small       200–499 rows         - cross-validation warranted
      imbalanced  ratio 3–10           - class weights needed
      large       ≥ 50 000 rows        - can afford full model suite
      standard    none of the above
    """
    rows        = dataset_profile["shape"]["rows"]
    cols        = dataset_profile["shape"]["cols"]
    imb         = float(dataset_profile.get("imbalance_ratio") or 1.0)
    missing_pct = dataset_profile.get("missing_pct", {})
    max_missing = max(missing_pct.values(), default=0.0) if missing_pct else 0.0

    if rows < _TINY_ROWS:
        return "tiny"
    if imb >= _IMBALANCE_SEVERE:
        return "severe_imb"
    if cols > _HIGH_DIM_COLS:
        return "high_dim"
    if max_missing > _MISSING_MODERATE:
        return "heavy_missing"
    if rows < _SMALL_ROWS:
        return "small"
    if imb >= _IMBALANCE_MILD:
        return "imbalanced"
    if rows >= _LARGE_ROWS:
        return "large"
    return "standard"


def _base_plan_for_scenario(scenario: str) -> List[str]:
    """
    Return the ordered base task list for a scenario.

    The scenario tag is the second item so the report records which
    template was chosen. Signal-based tasks are merged in on top by
    ``_inject_signal_tasks``.
    """
    core_tail = [
        TASK_BUILD_PRE,
        TASK_SELECT_MODELS,
        TASK_TRAIN,
        TASK_EVALUATE,
        TASK_REFLECT,
        TASK_REPORT,
    ]

    bases: Dict[str, List[str]] = {
        # tiny: force CV; skip heavy models (too slow / over-fit risk on <200 rows)
        "tiny": [
            TASK_PROFILE,
            f"{_SCENARIO_PREFIX}tiny",
            TASK_BUILD_PRE,
            TASK_SELECT_MODELS,
            TASK_TRAIN,
            TASK_CROSS_VAL,
            TASK_EVALUATE,
            TASK_REFLECT,
            TASK_REPORT,
        ],
        # small: CV recommended
        "small": [
            TASK_PROFILE,
            f"{_SCENARIO_PREFIX}small",
            TASK_BUILD_PRE,
            TASK_SELECT_MODELS,
            TASK_TRAIN,
            TASK_CROSS_VAL,
            TASK_EVALUATE,
            TASK_REFLECT,
            TASK_REPORT,
        ],
        # severe imbalance: force both imbalance tasks + ensemble-only modelling
        "severe_imb": [
            TASK_PROFILE,
            f"{_SCENARIO_PREFIX}severe_imb",
            TASK_IMBALANCE,
            TASK_IMBALANCE_SEVERE,
        ] + core_tail,
        # high-dimensional: ordinal encoding always active
        "high_dim": [
            TASK_PROFILE,
            f"{_SCENARIO_PREFIX}high_dim",
            TASK_HIGH_CARD,
        ] + core_tail,
        # heavy missing: severe-missing drop always active
        "heavy_missing": [
            TASK_PROFILE,
            f"{_SCENARIO_PREFIX}heavy_missing",
            TASK_DROP_SEVERE_MISS,
        ] + core_tail,
        # imbalanced: apply balanced class weights
        "imbalanced": [
            TASK_PROFILE,
            f"{_SCENARIO_PREFIX}imbalanced",
            TASK_IMBALANCE,
        ] + core_tail,
        # large / standard: minimal pipeline, full model suite
        "large":    [TASK_PROFILE, f"{_SCENARIO_PREFIX}large"]    + core_tail,
        "standard": [TASK_PROFILE, f"{_SCENARIO_PREFIX}standard"] + core_tail,
    }

    return list(bases.get(scenario, bases["standard"]))


def _inject_signal_tasks(
    plan: List[str],
    dataset_profile: Dict[str, Any],
    memory_hint: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Augment the base scenario plan with tasks driven by concrete dataset signals.

    Injection order
    ---------------
    1. Data-level mutations (before BUILD_PRE): duplicates, ID columns, datetime
       extraction, severe-missing drops, correlated-feature drops, text warnings.
    2. Preprocessing strategy: imputation method, skew transform, high-cardinality
       encoding, tiered imbalance (if not already added by the scenario template).
    3. Training strategy: cross-validation (if not from template), memory hint.
    """
    in_plan = set(plan)

    rows        = dataset_profile["shape"]["rows"]
    imb         = float(dataset_profile.get("imbalance_ratio") or 1.0)
    dup_count   = dataset_profile.get("duplicate_count", 0)
    id_feats    = dataset_profile.get("id_features", [])
    high_card   = dataset_profile.get("high_cardinality_features", [])
    highly_skew = dataset_profile.get("highly_skewed_features", [])
    text_feats  = dataset_profile.get("feature_types", {}).get("text", [])
    datetime_f  = dataset_profile.get("feature_types", {}).get("datetime", [])
    high_corr   = dataset_profile.get("high_corr_pairs", [])
    missing_pct = dataset_profile.get("missing_pct", {})
    max_missing = max(missing_pct.values(), default=0.0) if missing_pct else 0.0

    # ── Find BUILD_PRE anchor for insertion of data-prep tasks ─────────────
    try:
        pre_idx = plan.index(TASK_BUILD_PRE)
    except ValueError:
        pre_idx = len(plan)

    # ── 1. Data preparation tasks ──────────────────────────────────────────
    data_prep: List[str] = []

    if dup_count > 0 and TASK_DROP_DUPES not in in_plan:
        data_prep.append(TASK_DROP_DUPES)

    if id_feats and TASK_DROP_ID not in in_plan:
        data_prep.append(TASK_DROP_ID)

    # Datetime extraction: creates year/month columns, drops original datetime col
    if datetime_f and TASK_EXTRACT_DATETIME not in in_plan:
        data_prep.append(TASK_EXTRACT_DATETIME)

    # Severe-missing column drop (if not already in plan from scenario template)
    if max_missing > _MISSING_SEVERE and TASK_DROP_SEVERE_MISS not in in_plan:
        data_prep.append(TASK_DROP_SEVERE_MISS)

    # Correlated feature drop: reduces redundancy, can improve generalisation
    if high_corr and TASK_DROP_CORRELATED not in in_plan:
        data_prep.append(TASK_DROP_CORRELATED)

    if text_feats and TASK_HANDLE_TEXT not in in_plan:
        data_prep.append(TASK_HANDLE_TEXT)

    # Insert all data-prep tasks in order, immediately before BUILD_PRE
    for task in reversed(data_prep):
        plan.insert(pre_idx, task)
    pre_idx += len(data_prep)  # shift anchor to reflect insertions

    # ── 2. Preprocessing strategy ──────────────────────────────────────────

    # Imputation strategy selection:
    # If NO features are highly skewed, mean imputation is appropriate.
    # If any are skewed, keep default (median) and apply PowerTransformer.
    # IMPUTE_MEAN and SKEW_TRANSFORM are mutually exclusive by design.
    if not highly_skew and TASK_IMPUTE_MEAN not in in_plan and TASK_SKEW_TRANSFORM not in in_plan:
        plan.insert(pre_idx, TASK_IMPUTE_MEAN)
        pre_idx += 1
    elif highly_skew and TASK_SKEW_TRANSFORM not in in_plan:
        plan.insert(pre_idx, TASK_SKEW_TRANSFORM)
        pre_idx += 1

    # High-cardinality encoding (if not from scenario template)
    if high_card and TASK_HIGH_CARD not in in_plan:
        plan.insert(pre_idx, TASK_HIGH_CARD)
        pre_idx += 1

    # Tiered imbalance strategy (if not from scenario template)
    if imb >= _IMBALANCE_SEVERE and TASK_IMBALANCE_SEVERE not in in_plan:
        if TASK_IMBALANCE not in in_plan:
            plan.insert(pre_idx, TASK_IMBALANCE)
            pre_idx += 1
            in_plan.add(TASK_IMBALANCE)
        plan.insert(pre_idx, TASK_IMBALANCE_SEVERE)
        pre_idx += 1
    elif imb >= _IMBALANCE_MILD and TASK_IMBALANCE not in in_plan:
        plan.insert(pre_idx, TASK_IMBALANCE)
        pre_idx += 1

    # ── 3. Training strategy ───────────────────────────────────────────────

    # Cross-validation for small datasets (if not from scenario template)
    if rows < _SMALL_ROWS and TASK_CROSS_VAL not in in_plan:
        try:
            train_idx = plan.index(TASK_TRAIN)
            plan.insert(train_idx + 1, TASK_CROSS_VAL)
        except ValueError:
            plan.append(TASK_CROSS_VAL)

    # Memory-guided model prioritisation
    if memory_hint:
        best_prev  = memory_hint.get("best_model")
        match_type = memory_hint.get("match_type", "exact")
        sim_score  = float(memory_hint.get("similarity_score", 1.0))
        hint_task  = f"prioritize_model:{best_prev}"
        already    = any(t.startswith("prioritize_model:") for t in plan)
        if best_prev and not already and (match_type == "exact" or sim_score >= 0.60):
            try:
                sel_idx = plan.index(TASK_SELECT_MODELS)
                plan.insert(sel_idx + 1, hint_task)
            except ValueError:
                plan.append(hint_task)

    return plan


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def create_plan(
    dataset_profile: Dict[str, Any],
    memory_hint: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Generate a full ordered execution plan from dataset signals and optional memory.

    Steps
    -----
    1. Detect the primary scenario (dominant dataset characteristic).
    2. Build the scenario's base plan template.
    3. Layer signal-driven tasks on top.

    Args:
        dataset_profile : Output of ``profile_dataset()``.
        memory_hint     : Output of ``JSONMemory.get_hint()``, or None.

    Returns:
        Ordered list of task-name strings.

    Example
    -------
    >>> profile = {
    ...     "shape": {"rows": 350, "cols": 8},
    ...     "imbalance_ratio": 4.5,
    ...     "highly_skewed_features": ["revenue"],
    ...     "duplicate_count": 0,
    ...     "high_corr_pairs": [],
    ...     "feature_types": {"numeric": ["a"], "categorical": [], "datetime": [], "text": []},
    ...     "missing_pct": {},
    ...     "n_classes": 3,
    ... }
    >>> create_plan(profile)
    ['profile_dataset', 'scenario:small', 'apply_skew_transform',
     'consider_imbalance', 'build_preprocessor', 'select_models',
     'train_models', 'use_cross_validation', 'evaluate', 'reflect',
     'write_report']
    """
    scenario = _detect_scenario(dataset_profile)
    plan     = _base_plan_for_scenario(scenario)
    plan     = _inject_signal_tasks(plan, dataset_profile, memory_hint)
    return plan


def create_replan(
    original_plan: List[str],
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> List[str]:
    """
    Generate a revised execution plan after a poor-quality run.

    This is the *single source* of replan logic. The Reflector's
    ``apply_replan_strategy`` calls this function - it does not contain its
    own duplicate plan-editing code.

    Strategies applied
    ------------------
    Imbalance not yet handled
        → Insert TASK_IMBALANCE (and TASK_IMBALANCE_SEVERE if ratio > 10)
          before TASK_TRAIN.
    Very low F1 (< 0.50)
        → Add ``emphasize_ensemble`` → modelling skips LR, emphasises GB/ET.
    Model convergence (spread < 0.03) AND correlated pairs exist
        → Add TASK_DROP_CORRELATED if not already present.
    Stale memory hint that under-performed
        → Remove ``prioritize_model:X`` so model selection runs freely.
    Small dataset without cross-validation
        → Add TASK_CROSS_VAL for more reliable estimates.
    Always
        → Append TASK_REPLAN to mark this as a revised run.

    Args:
        original_plan    : Plan from the previous execution cycle.
        dataset_profile  : Dataset profile (may have been updated by Reflector).
        reflection       : Output of ``reflect()``.

    Returns:
        New ordered plan list.
    """
    # Start fresh from original plan, stripping prior replan markers
    new_plan = [t for t in original_plan if t != TASK_REPLAN]
    in_plan  = set(new_plan)

    imb      = float(dataset_profile.get("imbalance_ratio") or 1.0)
    f1       = float(reflection.get("f1_macro", 1.0))
    spread   = float(reflection.get("model_spread", 1.0))
    rows     = dataset_profile["shape"]["rows"]
    high_corr = dataset_profile.get("high_corr_pairs", [])

    # Helper: safe insert before a given anchor task
    def _insert_before(task: str, anchor: str) -> None:
        try:
            idx = new_plan.index(anchor)
        except ValueError:
            idx = len(new_plan)
        new_plan.insert(idx, task)
        in_plan.add(task)

    # Strategy 1: Force imbalance handling if absent
    if imb >= _IMBALANCE_MILD and TASK_IMBALANCE not in in_plan:
        _insert_before(TASK_IMBALANCE, TASK_TRAIN)
    if imb >= _IMBALANCE_SEVERE and TASK_IMBALANCE_SEVERE not in in_plan:
        _insert_before(TASK_IMBALANCE_SEVERE, TASK_TRAIN)

    # Strategy 2: Very low F1 → ensemble emphasis
    if f1 < 0.50 and "emphasize_ensemble" not in in_plan:
        new_plan.append("emphasize_ensemble")
        in_plan.add("emphasize_ensemble")

    # Strategy 3: All models converge + correlated pairs → try dropping correlates
    if spread < 0.03 and high_corr and TASK_DROP_CORRELATED not in in_plan:
        _insert_before(TASK_DROP_CORRELATED, TASK_BUILD_PRE)

    # Strategy 4: Remove stale memory hint if the suggested model under-performed
    prior_best = reflection.get("best_model", "")
    stale_hint = f"prioritize_model:{prior_best}"
    if stale_hint in new_plan and f1 < 0.50:
        new_plan = [t for t in new_plan if t != stale_hint]
        in_plan.discard(stale_hint)

    # Strategy 5: Add CV if small dataset and it's missing
    if rows < _SMALL_ROWS and TASK_CROSS_VAL not in in_plan:
        _insert_before(TASK_CROSS_VAL, TASK_EVALUATE)

    # Always mark as a replan attempt
    new_plan.append(TASK_REPLAN)
    return new_plan


# ═══════════════════════════════════════════════════════════════════════════
# Plan explainability
# ═══════════════════════════════════════════════════════════════════════════

_TASK_DESCRIPTIONS: Dict[str, str] = {
    TASK_PROFILE:          "Always first - confirms dataset schema and extracts all signals.",
    TASK_BUILD_PRE:        "Build ColumnTransformer (imputation, scaling, encoding).",
    TASK_SELECT_MODELS:    "Choose candidate classifiers based on size and characteristics.",
    TASK_TRAIN:            "Train all candidate models; compute per-model metrics.",
    TASK_EVALUATE:         "Pick best model; generate confusion matrix and classification report.",
    TASK_REFLECT:          "Analyse results; identify issues; decide whether to replan.",
    TASK_REPORT:           "Write markdown report and persist all artefacts.",
    TASK_DROP_DUPES:       "Duplicate rows detected - remove before split (leakage prevention).",
    TASK_DROP_ID:          "Identifier column(s) detected - excluded (no predictive signal).",
    TASK_DROP_SEVERE_MISS: "Column(s) >40 % missing - dropped (imputation unreliable at this level).",
    TASK_EXTRACT_DATETIME: "Datetime column(s) - extract year/month features; drop original.",
    TASK_DROP_CORRELATED:  "Highly correlated pairs (|r|>0.8) - drop one per pair to reduce redundancy.",
    TASK_HANDLE_TEXT:      "Text-like column(s) - excluded (no text encoder configured).",
    TASK_IMPUTE_MEAN:      "No skewed features - mean imputation is appropriate for numeric columns.",
    TASK_SKEW_TRANSFORM:   "Highly skewed numeric feature(s) - PowerTransformer (Yeo-Johnson) applied.",
    TASK_HIGH_CARD:        "High-cardinality categorical(s) - OrdinalEncoder (prevents feature explosion).",
    TASK_IMBALANCE:        "Class imbalance ratio 3–10 - class_weight='balanced'; report macro-F1.",
    TASK_IMBALANCE_SEVERE: "Severe imbalance (ratio >10) - ensemble-only selection; balanced weights.",
    TASK_CROSS_VAL:        "Small dataset - StratifiedKFold CV for reliable metric estimates.",
    TASK_REPLAN:           "Revised execution triggered by poor prior-run performance.",
}


def explain_plan(plan: List[str], dataset_profile: Dict[str, Any]) -> str:
    """
    Return a human-readable justification for every task in the plan.

    Saved as ``plan_explanation.txt`` and embedded in the markdown report
    so agent decisions are fully auditable.
    """
    rows = dataset_profile["shape"]["rows"]
    cols = dataset_profile["shape"]["cols"]
    imb  = float(dataset_profile.get("imbalance_ratio") or 1.0)
    n_cl = dataset_profile.get("n_classes", "?")
    scenario_task = next((t for t in plan if t.startswith(_SCENARIO_PREFIX)), None)
    scenario_name = scenario_task.replace(_SCENARIO_PREFIX, "") if scenario_task else "standard"

    lines = [
        "=== Execution Plan Justification ===",
        f"Dataset  : {rows} rows × {cols} cols",
        f"Classes  : {n_cl}   |   Imbalance ratio : {imb:.2f}",
        f"Scenario : {scenario_name}",
        "",
    ]

    for task in plan:
        if task.startswith(_SCENARIO_PREFIX):
            lines.append(f"  {task}: Scenario template applied - see header above.")
        elif task.startswith("prioritize_model:"):
            model = task.split(":", 1)[1]
            lines.append(
                f"  {task}: Memory hint - '{model}' worked well on a similar "
                "dataset; evaluated first."
            )
        elif task == "emphasize_ensemble":
            lines.append(
                "  emphasize_ensemble: Replan - prior F1 very low; "
                "LogisticRegression excluded; GB/ET/RF emphasised."
            )
        elif task in _TASK_DESCRIPTIONS:
            lines.append(f"  {task}: {_TASK_DESCRIPTIONS[task]}")
        else:
            lines.append(f"  {task}")

    return "\n".join(lines)