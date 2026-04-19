"""
Agentic Data Scientist — Orchestrator
======================================
Coordinates the full agentic pipeline for offline classification tasks.

Changes in this version
-----------------------
reflect() extended params
    Now passes confusion_matrix, confusion_matrix_labels, replan_count,
    memory, fingerprint, and plan so the Reflector can run its full
    analytical suite (CM pattern analysis, overfitting detection,
    meta-learning, diminishing returns check).

store_reflection_outcome (meta-learning)
    At the END of each execution cycle where a replan happened, the
    Orchestrator calls memory.store_reflection_outcome() to record which
    suggestions were given, the F1 before/after, and whether performance
    improved. The Reflector uses this on subsequent runs to deprioritise
    suggestions that have not worked.
"""

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ── Agent components ────────────────────────────────────────────────────────
from agents.planner import (
    create_plan,
    explain_plan,
    TASK_DROP_DUPES,
    TASK_DROP_ID,
    TASK_DROP_SEVERE_MISS,
    TASK_EXTRACT_DATETIME,
    TASK_DROP_CORRELATED,
    TASK_SKEW_TRANSFORM,
    TASK_HIGH_CARD,
    TASK_IMPUTE_MEAN,
    TASK_CROSS_VAL,
)
from agents.reflector import reflect, should_replan, apply_replan_strategy
from agents.memory    import JSONMemory

# ── Tools ───────────────────────────────────────────────────────────────────
from tools.data_profiler import profile_dataset, infer_target_column, dataset_fingerprint
from tools.modelling     import build_preprocessor, select_models, train_models
from tools.evaluation    import evaluate_best, write_markdown_report, save_json


# ── Run context ─────────────────────────────────────────────────────────────

@dataclass
class RunContext:
    run_id:      str
    started_at:  str
    data_path:   str
    target:      str
    output_dir:  str
    seed:        int
    test_size:   float
    max_replans: int


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _size_bucket(rows: int) -> str:
    if rows < 500:    return "tiny"
    if rows < 5_000:  return "small"
    if rows < 50_000: return "medium"
    return "large"


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class AgenticDataScientist:
    """
    Offline Agentic Data Scientist (classification-focused).

    Pipeline
    --------
    1. Load CSV; infer / validate target column.
    2. Profile dataset (rich signal extraction).
    3. Consult memory for prior knowledge (exact or similarity match).
    4. Create plan conditioned on signals + memory.
    5. Apply data-level preparation (drops, datetime extraction, etc.)
       → returns updated profile so column lists stay consistent.
    6. Build preprocessor, select models, train, evaluate.
    7. Reflect (full analytical suite including CM patterns, overfitting
       detection, meta-learning, diminishing returns).
    8. Persist all artefacts and update memory.
    9. If replan recommended and budget allows, update plan and repeat.
    10. Store reflection outcome in memory for meta-learning.
    """

    def __init__(self, memory_path: str = "agent_memory.json", verbose: bool = True):
        self.verbose = verbose
        self.memory  = JSONMemory(memory_path)
        self.ctx:   Optional[RunContext] = None
        self.state: Dict[str, Any]       = {}

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[AgenticDataScientist] {msg}", flush=True)

    # ── Data loading ────────────────────────────────────────────────────────

    def load_data(self, path: str) -> pd.DataFrame:
        self.log(f"Loading dataset: {path}")
        df = pd.read_csv(path)
        self.log(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        return df

    # ── Data preparation ────────────────────────────────────────────────────

    def _apply_data_preparation(
        self,
        df: pd.DataFrame,
        plan: List[str],
        profile: Dict[str, Any],
        target: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply all plan-specified data mutations BEFORE the train/test split.

        Returns (df_prepared, updated_profile) so downstream column lists
        stay consistent with what is actually in the dataframe.

        Mutations (in order)
        --------------------
        1. Drop rows with null target (always).
        2. drop_duplicates — remove duplicate rows (leakage prevention).
        3. drop_id_columns — explicitly drop detected identifier columns.
        4. extract_datetime — extract year/month; drop original datetime col.
        5. drop_severe_missing — drop columns with >40 % missing values.
        6. drop_correlated — from each high-correlation pair, drop second col.
        """
        df         = df.copy()
        profile    = dict(profile)
        ftype      = {k: list(v) for k, v in profile.get("feature_types", {}).items()}
        notes      = list(profile.get("notes", []))
        dropped:   List[str] = []

        # 1. Drop rows with null target
        before = len(df)
        df     = df[df[target].notna()]
        n_dropped = before - len(df)
        if n_dropped > 0:
            self.log(f"Dropped {n_dropped} rows with null target.")
            notes.append(f"Dropped {n_dropped} rows where target '{target}' was null.")

        # 2. Duplicate rows
        if TASK_DROP_DUPES in plan:
            before = len(df)
            df     = df.drop_duplicates()
            removed = before - len(df)
            if removed > 0:
                self.log(f"Dropped {removed} duplicate rows.")
                notes.append(f"Dropped {removed} duplicate rows (leakage prevention).")

        # 3. Identifier columns
        if TASK_DROP_ID in plan:
            id_cols = [c for c in profile.get("id_features", [])
                       if c in df.columns and c != target]
            if id_cols:
                df = df.drop(columns=id_cols)
                dropped.extend(id_cols)
                self.log(f"Dropped ID column(s): {id_cols}")
                notes.append(f"Dropped identifier column(s): {id_cols}")

        # 4. Datetime feature extraction
        if TASK_EXTRACT_DATETIME in plan:
            dt_cols = [c for c in ftype.get("datetime", [])
                       if c in df.columns and c != target]
            new_num: List[str] = []
            for col in dt_cols:
                try:
                    dt = pd.to_datetime(df[col], errors="coerce")
                    yr, mo = f"{col}_year", f"{col}_month"
                    df[yr]  = dt.dt.year.astype("Int64")
                    df[mo]  = dt.dt.month.astype("Int64")
                    df      = df.drop(columns=[col])
                    new_num.extend([yr, mo])
                    dropped.append(col)
                    self.log(f"Extracted year/month from '{col}'.")
                    notes.append(f"Extracted year/month from datetime column '{col}'.")
                except Exception as exc:
                    self.log(f"Could not parse datetime '{col}': {exc}")
            ftype["datetime"] = []
            ftype["numeric"]  = ftype.get("numeric", []) + new_num

        # 5. Severe-missing columns
        if TASK_DROP_SEVERE_MISS in plan:
            sev = [c for c in df.columns
                   if c != target and df[c].isna().mean() * 100 > 40.0]
            if sev:
                df = df.drop(columns=sev)
                dropped.extend(sev)
                self.log(f"Dropped severely missing columns: {sev}")
                notes.append(f"Dropped {len(sev)} column(s) with >40 % missing: {sev}")

        # 6. Correlated feature drop
        if TASK_DROP_CORRELATED in plan:
            pairs = profile.get("high_corr_pairs", [])
            corr_drop: List[str] = []
            seen: set = set()
            for col_a, col_b in pairs:
                if col_b not in seen and col_b in df.columns and col_b != target:
                    corr_drop.append(col_b)
                    seen.add(col_b)
            if corr_drop:
                df = df.drop(columns=corr_drop)
                dropped.extend(corr_drop)
                self.log(f"Dropped correlated column(s): {corr_drop}")
                notes.append(
                    f"Dropped {len(corr_drop)} correlated column(s) (|r|>0.8): {corr_drop}"
                )

        # Update profile to reflect structural changes
        all_dropped = set(dropped)
        for dtype in ("numeric", "categorical", "datetime", "text"):
            ftype[dtype] = [c for c in ftype.get(dtype, []) if c not in all_dropped]

        profile["feature_types"] = ftype
        profile["missing_pct"]   = {
            k: v for k, v in profile.get("missing_pct", {}).items()
            if k not in all_dropped
        }
        profile["notes"]  = notes
        profile["shape"]  = {"rows": int(len(df)), "cols": int(df.shape[1])}

        return df.reset_index(drop=True), profile

    # ── Main run ────────────────────────────────────────────────────────────

    def run(
        self,
        data_path:   str,
        target:      str,
        output_root: str   = "outputs",
        seed:        int   = 42,
        test_size:   float = 0.2,
        max_replans: int   = 1,
    ) -> str:
        """
        End-to-end agentic classification pipeline.

        Returns path to the run's output directory.
        """
        run_id     = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = os.path.join(output_root, run_id)
        os.makedirs(output_dir, exist_ok=True)

        self.ctx = RunContext(
            run_id=run_id, started_at=_now_iso(),
            data_path=data_path, target=target,
            output_dir=output_dir, seed=seed,
            test_size=test_size, max_replans=max_replans,
        )
        self.state = {"replan_count": 0, "prior_f1": None}

        # ── Load ────────────────────────────────────────────────────────────
        df = self.load_data(data_path)

        if target.strip().lower() == "auto":
            inferred = infer_target_column(df)
            if not inferred:
                raise ValueError(
                    "Could not infer target column. "
                    "Please provide --target <column_name>."
                )
            self.ctx.target = inferred
            self.log(f"Inferred target: '{inferred}'")

        # ── Profile ─────────────────────────────────────────────────────────
        profile = profile_dataset(df, self.ctx.target)
        fp      = dataset_fingerprint(df, self.ctx.target)
        self.log(
            f"Profile: {profile['shape']['rows']} rows × {profile['shape']['cols']} cols, "
            f"imbalance_ratio={profile['imbalance_ratio']:.2f}, "
            f"n_classes={profile['n_classes']}"
        )
        for note in profile.get("notes", []):
            self.log(f"  NOTE: {note}")

        # ── Memory lookup ────────────────────────────────────────────────────
        memory_hint = self.memory.get_hint(fp, profile)
        if memory_hint:
            self.log(
                f"Memory {memory_hint['match_type']} hit: "
                f"prior best='{memory_hint.get('best_model')}' "
                f"(sim={memory_hint.get('similarity_score', 1.0):.2f})"
            )
        else:
            self.log("No memory hint — planning from scratch.")

        # ── Initial plan ─────────────────────────────────────────────────────
        plan = create_plan(profile, memory_hint=memory_hint)
        plan_explanation = explain_plan(plan, profile)
        self.log(f"Plan ({len(plan)} tasks): {plan}")
        self.log(f"\n{plan_explanation}\n")

        # ═══════════════════════════════════════════════════════════════════
        # Execution loop
        # ═══════════════════════════════════════════════════════════════════
        while True:
            cycle = self.state["replan_count"] + 1
            self.log(f"--- Execution cycle {cycle} ---")

            # ── Data preparation ────────────────────────────────────────────
            df_prepared, profile = self._apply_data_preparation(
                df, plan, profile, self.ctx.target
            )

            # ── Preprocessor flags from plan ────────────────────────────────
            use_power_transform = TASK_SKEW_TRANSFORM in plan
            handle_high_card    = TASK_HIGH_CARD      in plan
            impute_strategy     = "mean" if TASK_IMPUTE_MEAN in plan else "median"

            self.log(
                f"Preprocessor: impute={impute_strategy}, "
                f"power_transform={use_power_transform}, "
                f"high_card_ordinal={handle_high_card}"
            )

            preprocessor = build_preprocessor(
                profile,
                use_power_transform=use_power_transform,
                handle_high_cardinality=handle_high_card,
                impute_strategy=impute_strategy,
            )

            # ── Model selection ──────────────────────────────────────────────
            candidates = select_models(profile, seed=self.ctx.seed, plan=plan)
            self.log(f"Candidates: {[n for n, _ in candidates]}")

            # ── Training ─────────────────────────────────────────────────────
            cv_folds = 5 if TASK_CROSS_VAL in plan else 0
            if cv_folds:
                self.log(f"Using {cv_folds}-fold StratifiedKFold CV.")

            results = train_models(
                df=df_prepared,
                target=self.ctx.target,
                preprocessor=preprocessor,
                candidates=candidates,
                seed=self.ctx.seed,
                test_size=self.ctx.test_size,
                output_dir=self.ctx.output_dir,
                verbose=self.verbose,
                cross_validate_folds=cv_folds,
            )

            # ── Evaluation ───────────────────────────────────────────────────
            eval_payload = evaluate_best(results, output_dir=self.ctx.output_dir)
            best_m = eval_payload["best_metrics"]
            self.log(
                f"Best: {best_m['model']}  "
                f"bal_acc={best_m['balanced_accuracy']:.3f}  "
                f"f1_macro={best_m['f1_macro']:.3f}"
            )

            # ── Reflection ───────────────────────────────────────────────────
            reflection = reflect(
                dataset_profile           = profile,
                evaluation                = eval_payload["best_metrics"],
                all_metrics               = eval_payload["all_metrics"],
                classification_report_str = eval_payload.get("classification_report", ""),
                confusion_matrix          = eval_payload.get("confusion_matrix_array"),
                confusion_matrix_labels   = eval_payload.get("confusion_matrix_labels"),
                replan_count              = self.state["replan_count"],
                memory                    = self.memory,
                fingerprint               = fp,
                plan                      = plan,
            )
            self.log(
                f"Reflection: status={reflection['status']}, "
                f"root_cause={reflection['root_cause']}, "
                f"replan={reflection['replan_recommended']}, "
                f"diminishing_returns={reflection['diminishing_returns']}"
            )
            for issue in reflection.get("issues", []):
                self.log(f"  ISSUE: {issue}")
            for note in reflection.get("memory_notes", []):
                self.log(f"  MEM:   {note}")

            # ── Save artefacts ───────────────────────────────────────────────
            save_json(os.path.join(self.ctx.output_dir, "eda_summary.json"), profile)
            save_json(os.path.join(self.ctx.output_dir, "plan.json"),        {"plan": plan})
            save_json(os.path.join(self.ctx.output_dir, "metrics.json"),     eval_payload)
            save_json(os.path.join(self.ctx.output_dir, "reflection.json"),  reflection)

            with open(os.path.join(self.ctx.output_dir, "plan_explanation.txt"), "w") as f:
                f.write(plan_explanation)

            write_markdown_report(
                out_path         = os.path.join(self.ctx.output_dir, "report.md"),
                ctx              = self.ctx,
                fingerprint      = fp,
                dataset_profile  = profile,
                plan             = plan,
                plan_explanation = plan_explanation,
                eval_payload     = eval_payload,
                reflection       = reflection,
                memory_hint      = memory_hint,
            )

            # ── Update memory ────────────────────────────────────────────────
            ftype = profile.get("feature_types", {})
            self.memory.upsert_dataset_record(fp, {
                "last_seen":         _now_iso(),
                "target":            self.ctx.target,
                "shape":             profile["shape"],
                "size_bucket":       _size_bucket(profile["shape"]["rows"]),
                "n_classes":         profile.get("n_classes", 2),
                "imbalance_ratio":   profile.get("imbalance_ratio", 1.0),
                "n_numeric":         len(ftype.get("numeric", [])),
                "n_categorical":     len(ftype.get("categorical", [])),
                "best_model":        eval_payload["best_metrics"]["model"],
                "best_metrics":      eval_payload["best_metrics"],
                "all_metrics":       eval_payload["all_metrics"],
                "plan":              plan,
                "reflection_status": reflection["status"],
                "notes":             profile.get("notes", []),
            })

            # ── Meta-learning: record outcome ────────────────────────────────
            current_f1 = float(best_m["f1_macro"])
            if self.state["prior_f1"] is not None:
                self.memory.store_reflection_outcome(
                    fingerprint  = fp,
                    issues       = reflection.get("issues", []),
                    suggestions  = reflection.get("suggestions", []),
                    f1_before    = float(self.state["prior_f1"]),
                    f1_after     = current_f1,
                    replan_count = self.state["replan_count"],
                )
                self.log(
                    f"Meta-learning: stored reflection outcome "
                    f"(f1_before={self.state['prior_f1']:.3f}, "
                    f"f1_after={current_f1:.3f})."
                )

            self.state["prior_f1"] = current_f1
            self.log(f"Memory updated (fp={fp}).")

            # ── Replan decision ──────────────────────────────────────────────
            if not should_replan(reflection):
                self.log("No replan. Finishing.")
                break

            if self.state["replan_count"] >= self.ctx.max_replans:
                self.log(
                    f"Replan suggested but max_replans={self.ctx.max_replans} reached."
                )
                break

            self.state["replan_count"] += 1
            self.log(f"Replanning (attempt #{self.state['replan_count']})...")
            plan, profile = apply_replan_strategy(plan, profile, reflection)
            plan_explanation = explain_plan(plan, profile)
            self.log(f"Revised plan: {plan}")

        self.log(f"Done. Outputs: {self.ctx.output_dir}")
        return self.ctx.output_dir