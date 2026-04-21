# CE888 - Technical Report
# Offline Agentic Data Scientist

**Module:** CE888 Data Science and Decision Making  
**Academic Year:** 2025–2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Dataset Understanding](#3-dataset-understanding)
4. [Planning Logic](#4-planning-logic)
5. [Modelling and Evaluation](#5-modelling-and-evaluation)
6. [Reflection and Re-planning](#6-reflection-and-re-planning)
7. [Memory and Learning](#7-memory-and-learning)
8. [Ethics and Limitations](#8-ethics-and-limitations)
9. [Conclusion and Future Work](#9-conclusion-and-future-work)

---

## 1. Introduction

The standard approach to machine learning is manual: a data scientist loads a dataset, explores it, decides how to handle missing values and imbalanced classes, picks models, evaluates results, and writes up findings. This works when a knowledgeable person is in the loop - but it does not scale, it is inconsistent across datasets, and it does not learn from what it has done before.

This project builds an alternative: an **Offline Agentic Data Scientist** that receives an unseen CSV file and autonomously produces a complete classification pipeline. The system profiles the data, forms a conditional plan, prepares the data, trains and evaluates candidate models, reflects on what went wrong, and if performance is poor enough, revises its strategy and tries again.

### Why Agentic?

The core problem with a fixed pipeline is that **no single set of steps fits all datasets**. Consider just three examples:

| Dataset type | What is needed | What a fixed pipeline does wrong |
|---|---|---|
| 150-row dataset | Cross-validation; simpler models | Gives unreliable single-split metrics |
| 10:1 class imbalance | Balanced weights; ensemble models | Over-predicts majority class silently |
| Skewed numeric features | PowerTransformer before scaling | Linear models produce poor boundaries |

An agent that reads the data first and then decides what to do handles all of these cases correctly. A fixed pipeline either ignores them or requires manual configuration - defeating the purpose of automation.

### Scope

- Language and libraries: Python, scikit-learn, pandas, NumPy, matplotlib
- Task type: tabular binary and multi-class classification
- Execution: fully offline, no APIs, no cloud services, no LLMs
- Testing: four diverse classification datasets

---

## 2. System Architecture

The system is organised into two layers: **agents** that reason and decide, and **tools** that do computational work. The orchestrator coordinates everything.

### Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AgenticDataScientist (Orchestrator)              │
│                         agentic_data_scientist.py                   │
└────────────┬──────────────┬──────────────┬──────────────────────────┘
             │              │              │
    ┌────────▼──────┐  ┌────▼────────┐  ┌─▼──────────────┐
    │    AGENTS     │  │    TOOLS    │  │    MEMORY       │
    │               │  │             │  │                 │
    │  planner.py   │  │data_profiler│  │  memory.py      │
    │  reflector.py │  │modelling.py │  │  agent_memory   │
    │               │  │evaluation.py│  │  .json          │
    └───────────────┘  └─────────────┘  └─────────────────┘
```

### End-to-End Data Flow

```
CSV file
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. LOAD & VALIDATE                                               │
│    • Read CSV  →  check shape, empty check, column count         │
│    • Validate or infer target column (scoring-based detection)   │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. PROFILE                                                       │
│    • Column types, missing %, skewness, outliers, correlations   │
│    • Class distribution, imbalance ratio, duplicate count        │
│    • Produces: dataset_profile dict                              │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. MEMORY LOOKUP                                                 │
│    • Exact fingerprint match → prior best model known            │
│    • Similarity match → experience from similar datasets         │
│    • No match → plan from scratch                                │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. PLAN                                                          │
│    • Detect scenario (tiny / imbalanced / large / ...)           │
│    • Inject signal-driven tasks on top of scenario template      │
│    • Produces: ordered task list + plain-English justification   │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                     ┌─────────▼──────────────────────────────────┐
                     │ 5. EXECUTION LOOP (may repeat on replan)   │
                     │                                             │
                     │   DATA PREP  →  PREPROCESS  →  TRAIN       │
                     │       │             │            │          │
                     │  drop_dupes   impute+scale   RF/ET/GB/LR    │
                     │  extract_dt   OHE/ordinal    DummyBaseline  │
                     │  drop_corr    PowerTransform  StratKFold?   │
                     │                                             │
                     │   EVALUATE  →  REFLECT  →  REPLAN?         │
                     │       │             │            │          │
                     │  confusion  per-class F1   create_replan    │
                     │  matrix     root cause     (if warranted)   │
                     └─────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 6. PERSIST ARTEFACTS                                             │
│    report.md │ plan.json │ metrics.json │ reflection.json        │
│    plan_explanation.txt │ confusion_matrix.png │ eda_summary.json│
│    agent_memory.json (updated for future runs)                   │
└──────────────────────────────────────────────────────────────────┘
```

### Files and Responsibilities

| File | Layer | Responsibility |
|---|---|---|
| `run_agent.py` | Entry point | CLI argument parsing; calls orchestrator |
| `agentic_data_scientist.py` | Orchestrator | Coordinates all components; manages replan loop; handles errors |
| `agents/planner.py` | Agent | Scenario detection; plan template selection; signal-task injection; replan logic |
| `agents/reflector.py` | Agent | Multi-dimensional result analysis; root cause diagnosis; replan decision |
| `agents/memory.py` | Agent | Persistent storage; similarity retrieval; meta-learning from outcomes |
| `tools/data_profiler.py` | Tool | Signal extraction; target detection; column classification |
| `tools/modelling.py` | Tool | Preprocessing pipeline; model selection; training with optional CV |
| `tools/evaluation.py` | Tool | Metrics; confusion matrix; markdown report writing |

---

## 3. Dataset Understanding

Before planning anything, the agent must understand the data. The `profile_dataset` function extracts a comprehensive set of signals that drive every downstream decision.

### Signals Extracted

| Signal | How detected | What it drives |
|---|---|---|
| **Column type** | dtype + heuristics (integer range, datetime parse attempt, unique ratio) | Which transformer to apply |
| **ID columns** | "id" in name, monotonic, high unique ratio, constant step | Excluded from modelling |
| **Target column** | Scoring: keyword match +4, last column +2, ID −4, datetime −4 | Defines the classification task |
| **Missing values** | Per-column `isna().mean() * 100` | Imputation strategy or column drop |
| **Class imbalance** | `majority_count / minority_count` | Balanced weights, metric choice |
| **Skewness** | `df.skew()`, threshold |skew| > 1.0 | PowerTransformer in preprocessor |
| **Outliers** | IQR method per numeric column | Noted in report |
| **Correlations** | Pearson |r| > 0.80 | Drop one column per pair |
| **High-cardinality cats** | Unique count > 20 | OrdinalEncoder instead of OHE |
| **Datetime columns** | `pd.to_datetime` conversion rate ≥ 80% | Extract year / month features |

### Target Detection Logic

Target detection uses a **scoring system** rather than a simple lookup. The key insight is that after scoring, candidates are checked for **classification suitability** before being accepted - a highly-scored column that turns out to be a continuous float is skipped, and the next candidate is tried automatically.

```
For each column → score:
  + 4  if column name matches: target, label, class, outcome, y, status, result, output
  + 2  if it is the last column in the dataset
  − 4  if detected as an identifier column
  − 4  if it looks like a datetime

Sort by score descending →
  For each candidate (high score first):
    if is_classification_suitable(column):  ← 2–50 unique for int, ≤20 for float
      return this column ✓
    else: skip, try next candidate
```

### Four Test Datasets: Signal Comparison

| Signal | Sales.csv | Iris | Titanic | Adult Income |
|---|---|---|---|---|
| Rows | 30 000 | 150 | 891 | 48 842 |
| Classes | 21 | 3 | 2 | 2 |
| Imbalance ratio | ~1.05 | 1.0 | ~1.7 | ~3.1 |
| Missing values | None | None | Age 20 % | Occupation 5.7 % |
| Datetime columns | `order_date` | - | - | - |
| ID columns | `order_id` | - | - | - |
| High-cardinality cats | `model_name` | - | - | `native_country` |
| Skewed features | `revenue_usd` | - | - | `capital_gain`, `capital_loss` |
| Correlated pairs | price columns | - | - | `education` / `education_num` |
| Agent scenario | standard | **tiny** | standard | **large** |

No two datasets trigger exactly the same combination of plan tasks, which is the whole point of diverse evaluation.

---

## 4. Planning Logic

The planner is the most important component for demonstrating autonomy. It takes the profile and produces an **ordered, named list of task strings** that the orchestrator executes conditionally.

### Step 1 - Scenario Detection

The planner first classifies the dataset into one primary scenario using a strict precedence order:

```
1.  tiny          rows < 200                    (dominates everything)
2.  severe_imb    imbalance_ratio > 10
3.  high_dim      cols > 100
4.  heavy_missing max column missing > 20%
5.  small         rows 200–499
6.  imbalanced    imbalance_ratio 3–10
7.  large         rows ≥ 50 000
8.  standard      none of the above
```

Each scenario has a base plan template. For example:

| Scenario | Base plan tasks always included |
|---|---|
| `tiny` | `profile_dataset` → `build_preprocessor` → `select_models` → `train_models` → **`use_cross_validation`** → `evaluate` → `reflect` → `write_report` |
| `severe_imb` | ... → **`consider_imbalance`** → **`consider_severe_imbalance`** → ... |
| `large` | Standard pipeline; SVC excluded; full model suite |
| `standard` | Minimal pipeline; all models evaluated |

### Step 2 - Signal-Task Injection

On top of the scenario base, individual signals inject additional tasks:

```
Signal detected                     →  Task inserted (before build_preprocessor)
────────────────────────────────────────────────────────────────────────────
Duplicate rows > 0                  →  drop_duplicates
ID columns detected                 →  drop_id_columns
Datetime columns exist              →  extract_datetime
Any column > 40% missing            →  drop_severe_missing
High-correlation pairs (|r|>0.8)    →  drop_correlated
Text-like columns present           →  handle_text_features
No highly-skewed features           →  impute_numeric_mean   ─┐ mutually
Highly-skewed features exist        →  apply_skew_transform  ─┘ exclusive
High-cardinality categoricals       →  handle_high_cardinality
Imbalance ratio 3–10                →  consider_imbalance
Imbalance ratio > 10                →  consider_imbalance + consider_severe_imbalance
rows < 500 (if not in scenario)     →  use_cross_validation
Memory hint available               →  prioritize_model:<name>
```

### Real Plan Comparison

Here are the actual plans generated for three of the four test datasets:

**Iris (150 rows, balanced, all-numeric):**
```
profile_dataset → scenario:tiny → impute_numeric_mean →
build_preprocessor → select_models → train_models →
use_cross_validation → evaluate → reflect → write_report
```

**Sales.csv (30 000 rows, mixed types, datetime, high-cardinality):**
```
profile_dataset → scenario:standard → drop_id_columns →
extract_datetime → drop_correlated → handle_high_cardinality →
impute_numeric_mean → build_preprocessor → select_models →
train_models → evaluate → reflect → write_report
```

**Adult Income (49K rows, imbalanced, skewed features):**
```
profile_dataset → scenario:large → apply_skew_transform →
handle_high_cardinality → consider_imbalance →
build_preprocessor → select_models → train_models →
evaluate → reflect → write_report
```

Every plan step is justified in plain English in `plan_explanation.txt`. For example, the `drop_correlated` task in the Sales plan includes the note: *"Highly correlated pairs (|r|>0.8) - drop one per pair to reduce redundancy."*

---

## 5. Modelling and Evaluation

### Preprocessing Pipeline

The `build_preprocessor` function constructs a `ColumnTransformer` with different pipelines for different column types. The specific steps depend on the plan:

```
Numeric columns:
   SimpleImputer (mean or median based on plan)
   → StandardScaler
   → [PowerTransformer Yeo-Johnson, if apply_skew_transform in plan]

Categorical columns (low cardinality):
   SimpleImputer (most frequent)
   → OneHotEncoder (handle_unknown='ignore')

Categorical columns (high cardinality, if handle_high_cardinality in plan):
   SimpleImputer (most frequent)
   → OrdinalEncoder (unknown_value=-1)
```

Why OrdinalEncoder for high-cardinality columns rather than always using OHE: a column like `native_country` with 41 unique values would produce 41 new binary columns. On larger datasets this inflates training time and memory without meaningful benefit over a simple integer encoding.

### Candidate Models

| Model | Always included? | Excluded when |
|---|---|---|
| `DummyMostFrequent` | Yes - baseline | Never |
| `LogisticRegression` | Yes | `consider_severe_imbalance` or `emphasize_ensemble` in plan |
| `RandomForest` | Yes | Never |
| `ExtraTrees` | Yes | Never |
| `GradientBoosting` | Rows ≤ 50 000 | Large datasets (unless replan forces it) |
| `SVC (RBF)` | Rows ≤ 20 000, cols ≤ 200 | Large or high-dimensional datasets |

`class_weight='balanced'` is applied to all classifiers that support it whenever `consider_imbalance` is in the plan. This instructs each model to penalise misclassification of minority classes proportionally more than majority classes.

### Primary Metrics

Accuracy alone is not used as the primary metric. The agent uses:

| Metric | Why |
|---|---|
| **Balanced accuracy** | Average recall per class - gives equal weight regardless of class size |
| **Macro F1** | Averages F1 across classes without weighting by support |
| **Per-class F1** | Reveals which specific classes are hardest to learn |

### Statistical Context

The reflector adds statistical context alongside raw metrics:

- **Cohen's h effect size** - measures whether improvement over the dummy baseline is practically meaningful (not just numerically non-zero). Categories: negligible < 0.20, small 0.20–0.50, medium 0.50–0.80, large > 0.80.
- **95% Wilson confidence interval** on balanced accuracy - quantifies uncertainty given test set size. More reliable than the normal approximation near 0 or 1.

---

## 6. Reflection and Re-planning

### What the Reflector Analyses

After every training cycle, `reflect()` runs eight analytical checks:

```
1.  Dummy baseline comparison    →  effect size (Cohen's h) + Wilson 95% CI
2.  Absolute performance bands   →  F1 < 0.60 (weak) or < 0.75 (moderate)
3.  Leakage suspicion            →  balanced accuracy ≥ 0.97 (too good?)
4.  Imbalance bias               →  accuracy >> balanced accuracy by > 0.10
5.  Per-class analysis           →  F1 per class, F1 spread across classes
6.  Precision-recall tradeoff    →  |precision − recall| > 0.15 (bias check)
7.  Overfitting / underfitting   →  CV balanced accuracy vs test accuracy gap
8.  Confusion matrix patterns    →  most confused pair, zero-correct classes
```

### Root Cause Diagnosis

All checks are synthesised into a single root cause label. This makes the agent's diagnosis auditable:

| Root cause | Condition | What changes in replan |
|---|---|---|
| `majority_class_bias` | Accuracy >> balanced accuracy with imbalance | Force `consider_imbalance` |
| `overfitting` | CV accuracy >> test accuracy (gap > 0.10) | Flag for regularisation |
| `underfitting` | Both CV and test accuracy < 0.60 | Add `emphasize_ensemble` |
| `zero_class` | Any class with zero correct predictions | Force balanced weights |
| `weak_feature_signal` | < 0.05 improvement over dummy | Investigate leakage / features |
| `data_quality` | All models converge (std < 0.03) | Add `drop_correlated` |
| `class_confusion` | Multi-class, large F1 spread | Ensemble emphasis |
| `insufficient_data` | < 500 rows + poor F1 | Add cross-validation |
| `ok` | None of the above | No replan needed |

### Replan Decision Logic

Replanning is only triggered when **all four conditions** are met:

```
✓ At least one issue identified
✓ Macro F1 < adaptive threshold  (relaxes for harder multi-class problems)
✓ Balanced accuracy < 0.70
✓ No diminishing returns detected (prior replan improved F1 by < 0.02 → stop)
✓ Not a leakage situation (balanced accuracy ≥ 0.97 → replan won't help)
```

The **diminishing returns guard** is important: without it, the agent would keep replanning with small variations and never converge. If the last replan produced less than 0.02 improvement in F1, further replanning is blocked even if other conditions are met.

### Replan Strategies

Different root causes trigger different plan changes:

```
Root cause: majority_class_bias   →  insert consider_imbalance before train_models
Root cause: underfitting          →  append emphasize_ensemble (drop LR, boost GB/ET)
Root cause: data_quality          →  insert drop_correlated (break the convergence)
Root cause: overfitting           →  noted in report; replan with smaller models
Stale memory hint + low F1        →  remove prioritize_model:<name> from plan
Very low F1 (< 0.50)              →  append emphasize_ensemble
```

---

## 7. Memory and Learning

The memory system serves two separate functions: **warm starting new runs** and **avoiding repeated failures** on datasets the agent has seen before.

### What is Stored

Every completed run updates the JSON memory store with:

```json
{
  "fp_123456789": {
    "last_seen": "2026-04-15T14:22:10Z",
    "target": "income",
    "shape": {"rows": 48842, "cols": 14},
    "size_bucket": "large",
    "n_classes": 2,
    "imbalance_ratio": 3.1,
    "n_numeric": 6,
    "n_categorical": 8,
    "best_model": "GradientBoosting",
    "best_metrics": {"balanced_accuracy": 0.812, "f1_macro": 0.786},
    "plan": ["profile_dataset", "scenario:large", ...],
    "reflection_status": "ok",
    "reflection_history": [...]
  }
}
```

### Retrieval Hierarchy

```
New dataset arrives
        │
        ▼
Exact fingerprint match? ─── YES ──→ Use prior best model directly
        │                              (match_type: "exact")
        NO
        │
        ▼
Similarity score > 0.50? ─── YES ──→ Use as hint, lower confidence
        │                              (match_type: "similar")
        NO
        │
        ▼
Plan from scratch (no memory hint)
```

**Similarity** is computed across four dimensions:

| Dimension | Weight | Why |
|---|---|---|
| Size bucket (tiny/small/medium/large) | 30% | Dataset size is the strongest driver of strategy |
| Number of classes | 25% | Binary vs multi-class changes metric choice |
| Imbalance ratio (log scale) | 25% | Imbalance handling is a major decision point |
| Numeric/categorical feature ratio | 20% | Affects preprocessing choices |

### Meta-Learning from Reflection Outcomes

When a replan happens, the orchestrator records the outcome afterwards:

```
store_reflection_outcome(
    fingerprint = "fp_123...",
    suggestions = ["Try ensemble methods", "Apply class_weight='balanced'"],
    f1_before   = 0.48,
    f1_after    = 0.61,
    improved    = True   ← computed as f1_after - f1_before ≥ 0.02
)
```

On the next run against the same dataset, the reflector checks `get_suggestion_effectiveness()` which returns the historical success rate per suggestion category (`imbalance`, `ensemble`, `regularisation`, etc.). Suggestions from categories with a 0% success rate have their impact score reduced by 60%, pushing them lower in the sorted suggestion list. A `[Memory]` warning note is added to the reflection output: *"Suggestion category 'ensemble' was tried in a prior replan without meaningful improvement."*

This prevents the agent from endlessly giving the same advice that has not worked.

---

## 8. Ethics and Limitations

### Ethical Considerations

**Fairness** is the most significant concern. The agent makes no distinction between features that are technically informative and features that raise ethical concerns. On the Adult Income dataset, `sex`, `race`, and `native_country` are all included as features and the agent will use them because they correlate with income. A deployed system of this type would need a separate fairness audit - for example, checking whether prediction error rates differ systematically across demographic groups.

**Transparency** is well handled. Every decision is recorded: the plan with its justification, the reflection with its root cause diagnosis, and the memory with its history. A user can always trace why a model was chosen and what the agent thought about its performance.

**Data quality propagation.** If training data reflects historical biases, the model learns those patterns. The reflector checks for label noise and class imbalance, but it cannot detect when imbalance itself results from unfair historical processes.

### Technical Limitations

| Limitation | Detail |
|---|---|
| Classification only | No regression, time-series, multi-label, or text classification |
| No hyperparameter search | Models use default or lightly tuned hyperparameters |
| No SMOTE / oversampling | Imbalance is handled only via class weights, not synthetic samples |
| Feature importance blind | The agent does not read tree feature importances to inform re-planning |
| Scale ceiling (~50k rows for GB) | Above this GradientBoosting is excluded; XGBoost / LightGBM would be better choices |
| Memory based on fingerprint | Structural renames or column reorders produce a different fingerprint even for the same underlying dataset |

---

## 9. Conclusion and Future Work

### What Was Built

This project demonstrates that a rule-based agentic system can perform genuine autonomous reasoning about data without any large language model. The key design principle throughout was: **every agent decision must be traceable to a specific data signal.** The plan is not a hard-coded checklist - it is assembled from what the profiler actually finds.

The four test datasets show this in practice. Iris gets cross-validation. Sales.csv gets datetime extraction, identifier removal, and ordinal encoding. Adult Income gets balanced class weighting, skew transformation, and ordinal encoding for high-cardinality categoricals. No two datasets trigger the same plan, which is the evidence for genuine adaptability.

### What Worked Well

- **Plan explainability** - every task has a plain-English justification saved to disk
- **Root cause diagnosis** - synthesising multiple checks into a single label makes the agent's reasoning clear
- **Graceful error handling** - unknown column types, continuous targets, empty datasets, and training failures all produce user-friendly messages rather than Python tracebacks
- **Memory similarity matching** - the four-dimension similarity score meaningfully captures dataset relatedness

### Future Work

| Direction | What it would add |
|---|---|
| Hyperparameter search (Bayesian) | Better performance without manual tuning |
| Fairness auditing (Fairlearn) | Demographic parity and equal opportunity checks |
| SMOTE / ADASYN oversampling | Alternative to class weighting for severe imbalance |
| Feature importance feedback | Tree importances fed back into reflector for better root cause analysis |
| XGBoost / LightGBM support | Better scaling beyond 50k rows |
| Regression and multi-label support | Broader applicability |
| Data drift detection | Flag when a new dataset is statistically unlike anything in memory |


