"""
Evaluation Tools
================
Evaluates the best trained model and produces artefacts:
  - Confusion matrix (PNG + raw ndarray returned in-memory for the Reflector)
  - Classification report string (returned for per-class analysis)
  - All-models metrics table
  - Markdown summary report

JSON serialisation note
-----------------------
``evaluate_best`` returns ``confusion_matrix_array`` as a numpy ndarray.
This is kept in-memory for the Reflector but is NOT written to metrics.json
(ndarrays are not JSON-serialisable, and the PNG already captures it visually).
Use ``metrics_for_saving(eval_payload)`` to get a JSON-safe copy before saving.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# ── Serialisation helpers ──────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """
    Fallback JSON serialiser for non-standard types.

    Handles numpy scalars and arrays (the most common failure cases).
    Falls back to str() for anything else.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()           # ndarray → nested list
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()         # single-element numpy scalar
        except (ValueError, AttributeError):
            return str(obj)
    return str(obj)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_safe)


def metrics_for_saving(eval_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a JSON-safe copy of eval_payload for writing to metrics.json.

    ``confusion_matrix_array`` is a numpy ndarray needed only in-memory by
    the Reflector. It is excluded here to keep the JSON file clean and small.
    """
    return {
        k: v for k, v in eval_payload.items()
        if k != "confusion_matrix_array"
    }


# ── Confusion matrix plot ──────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    out_path: str,
    title: str,
) -> None:
    """Render and save a colour-mapped confusion matrix PNG."""
    n = max(len(labels), 2)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=13, pad=12)
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    ax.set_ylabel("True label",      fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


# ── Main evaluation ────────────────────────────────────────────────────────

def evaluate_best(
    training_payload: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Evaluate the best trained model and save artefacts.

    Returns
    -------
    dict with keys:
      best_metrics            — accuracy, balanced_accuracy, f1_macro, etc.
      all_metrics             — list of metrics dicts for all models
      confusion_matrix_path   — path to PNG
      confusion_matrix_array  — raw numpy ndarray (in-memory only; not saved to JSON)
      confusion_matrix_labels — class label strings aligned with CM rows/cols
      classification_report   — sklearn classification_report string
    """
    best        = training_payload["best"]
    all_metrics = training_payload["all_metrics"]

    y_test = best["y_test"]
    y_pred = best["y_pred"]

    labels  = sorted([str(x) for x in y_test.dropna().unique().tolist()])
    cm      = confusion_matrix(y_test, y_pred, labels=labels)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels, cm_path, f"Confusion Matrix — {best['name']}")

    cls_report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "best_metrics":            best["metrics"],
        "all_metrics":             all_metrics,
        "confusion_matrix_path":   cm_path,
        "confusion_matrix_array":  cm,       # numpy ndarray — in-memory only
        "confusion_matrix_labels": labels,
        "classification_report":   cls_report,
    }


# ── Markdown report ────────────────────────────────────────────────────────

def write_markdown_report(
    out_path: str,
    ctx: Any,
    fingerprint: str,
    dataset_profile: Dict[str, Any],
    plan: List[str],
    plan_explanation: str,
    eval_payload: Dict[str, Any],
    reflection: Dict[str, Any],
    memory_hint: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a comprehensive markdown report for this agent run."""
    best    = eval_payload["best_metrics"]
    cls_rpt = eval_payload.get("classification_report", "")

    def short_list(xs, n=12):
        if not xs:
            return "_(none)_"
        return ", ".join(str(x) for x in xs[:n]) + (" …" if len(xs) > n else "")

    numeric     = dataset_profile.get("feature_types", {}).get("numeric", [])
    categorical = dataset_profile.get("feature_types", {}).get("categorical", [])
    notes       = dataset_profile.get("notes", [])

    model_rows = "\n".join(
        f"| {m['model']:<30} | {m['balanced_accuracy']:.3f} | "
        f"{m['f1_macro']:.3f} | {m['accuracy']:.3f} |"
        for m in eval_payload.get("all_metrics", [])
    )

    if memory_hint:
        mt  = memory_hint.get("match_type", "unknown")
        sim = memory_hint.get("similarity_score", "—")
        sim_str = f"{sim:.2f}" if isinstance(sim, float) else str(sim)
        memory_md = (
            f"- Match type: **{mt}** (similarity={sim_str})\n"
            f"- Prior best model: `{memory_hint.get('best_model', '—')}`\n"
            f"- Prior balanced-accuracy: "
            f"{memory_hint.get('best_metrics', {}).get('balanced_accuracy', '—')}"
        )
    else:
        memory_md = "- No prior knowledge available for this dataset."

    ci = reflection.get("confidence_interval_95", {})
    ci_str = (
        f"[{ci.get('lower', 0):.3f}, {ci.get('upper', 0):.3f}]"
        if ci else "_(not computed)_"
    )
    effect_h   = reflection.get("effect_size_cohen_h")
    effect_lbl = reflection.get("effect_size_label", "")
    effect_str = f"{effect_h:.3f} ({effect_lbl})" if effect_h is not None else "_(not computed)_"

    root_cause  = reflection.get("root_cause", "unknown")
    overfit_str = reflection.get("overfit_underfitting_diagnosis", "")
    pr_note     = reflection.get("precision_recall_note", "")

    md = f"""# Agentic Data Scientist — Run Report

**Run ID:** `{ctx.run_id}`  
**Started (UTC):** {ctx.started_at}  
**Dataset:** `{ctx.data_path}`  
**Target:** `{ctx.target}`  
**Fingerprint:** `{fingerprint}`

---

## 1. Dataset Profile

| Property | Value |
|---|---|
| Rows | **{dataset_profile['shape']['rows']}** |
| Columns | **{dataset_profile['shape']['cols']}** |
| Duplicates | **{dataset_profile.get('duplicate_count', 0)}** |
| Classes | **{dataset_profile.get('n_classes', '?')}** |
| Imbalance ratio | **{dataset_profile.get('imbalance_ratio', '?')}** |

**Numeric features ({len(numeric)}):** {short_list(numeric)}  
**Categorical features ({len(categorical)}):** {short_list(categorical)}  
**Highly skewed:** {short_list(dataset_profile.get('highly_skewed_features', []))}

**Warnings:**
{chr(10).join(f'- {n}' for n in notes) if notes else '- _(none)_'}

---

## 2. Execution Plan

```
{chr(10).join(f'  {i+1:2d}. {t}' for i, t in enumerate(plan))}
```

### Justification

```
{plan_explanation}
```

---

## 3. Model Results

| Model | Bal. Acc | F1 Macro | Accuracy |
|---|---|---|---|
{model_rows}

### Best Model: `{best.get('model')}`

| Metric | Value |
|---|---|
| Accuracy | {best.get('accuracy', 0):.4f} |
| Balanced accuracy | {best.get('balanced_accuracy', 0):.4f} |
| Macro F1 | {best.get('f1_macro', 0):.4f} |
| Macro Precision | {best.get('precision_macro', 0):.4f} |
| Macro Recall | {best.get('recall_macro', 0):.4f} |

**95% CI (balanced accuracy):** {ci_str}  
**Effect size vs dummy (Cohen's h):** {effect_str}

### Classification Report

```
{cls_rpt}
```

---

## 4. Reflection

**Status:** `{reflection.get('status', 'unknown')}`  
**Root cause:** `{root_cause}`  
**Replan recommended:** `{reflection.get('replan_recommended', False)}`  
**Diminishing returns:** `{reflection.get('diminishing_returns', False)}`  
**Model spread (std):** `{reflection.get('model_spread', 0):.4f}`

{f'**Overfitting/underfitting:** {overfit_str}' if overfit_str else ''}
{f'**Precision-recall bias:** {pr_note}' if pr_note else ''}

**Issues:**
{chr(10).join(f'- ⚠ {s}' for s in reflection.get('issues', [])) or '- _(none)_'}

**Suggestions (by impact):**
{chr(10).join(f'- 💡 {s}' for s in reflection.get('suggestions', [])) or '- _(none)_'}

---

## 5. Memory

{memory_md}

---

## 6. Artefacts

- `confusion_matrix.png`
- `eda_summary.json`
- `plan.json`
- `plan_explanation.txt`
- `metrics.json`
- `reflection.json`
- `report.md`
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)