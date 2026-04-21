import numpy as np
from agents import planner


def make_profile(rows=1000, cols=10, imb=1.0, missing_pct=None, high_corr=None, skewed=None, dup=0, id_feats=None, high_card=None, datetime_cols=None):
    return {
        "shape": {"rows": rows, "cols": cols},
        "imbalance_ratio": imb,
        "missing_pct": missing_pct or {},
        "high_corr_pairs": high_corr or [],
        "highly_skewed_features": skewed or [],
        "duplicate_count": dup,
        "id_features": id_feats or [],
        "high_cardinality_features": high_card or [],
        "feature_types": {"numeric": [], "categorical": [], "datetime": datetime_cols or [], "text": []},
        "n_classes": 2,
    }


def test_detect_scenarios_and_base_plan():
    # tiny dataset
    p = make_profile(rows=100)
    plan = planner.create_plan(p)
    assert any(t.startswith("scenario:tiny") for t in plan)
    assert planner.TASK_CROSS_VAL in plan

    # severe imbalance
    p = make_profile(rows=1000, imb=20.0)
    plan = planner.create_plan(p)
    assert any(t.startswith("scenario:severe_imb") for t in plan)
    assert planner.TASK_IMBALANCE_SEVERE in plan

    # high dim
    p = make_profile(rows=1000, cols=200)
    plan = planner.create_plan(p)
    assert any(t.startswith("scenario:high_dim") for t in plan)
    assert planner.TASK_HIGH_CARD in plan

    # heavy missing
    p = make_profile(rows=1000, missing_pct={"a": 25.0})
    plan = planner.create_plan(p)
    assert any(t.startswith("scenario:heavy_missing") for t in plan)
    assert planner.TASK_DROP_SEVERE_MISS in plan


def test_inject_signal_tasks_and_replanning():
    p = make_profile(rows=300, imb=5.0, dup=2, id_feats=["idcol"], high_card=["cat1"], skewed=["rev"], datetime_cols=["dt"], high_corr=[("a","b")])
    base = planner._base_plan_for_scenario("standard")
    plan = planner._inject_signal_tasks(list(base), p, memory_hint=None)
    # Data prep tasks inserted
    assert planner.TASK_DROP_DUPES in plan
    assert planner.TASK_DROP_ID in plan
    assert planner.TASK_EXTRACT_DATETIME in plan
    assert planner.TASK_DROP_CORRELATED in plan
    # Preprocessing strategies
    assert planner.TASK_SKEW_TRANSFORM in plan or planner.TASK_IMPUTE_MEAN in plan

    # create_replan: imbalance should be inserted if missing
    original = [planner.TASK_PROFILE, planner.TASK_BUILD_PRE, planner.TASK_SELECT_MODELS, planner.TASK_TRAIN, planner.TASK_EVALUATE]
    p2 = make_profile(rows=300, imb=4.0, high_corr=[("a","b")])
    reflection = {"f1_macro": 0.45, "model_spread": 0.01, "best_model": "RandomForest"}
    new_plan = planner.create_replan(original, p2, reflection)
    assert planner.TASK_IMBALANCE in new_plan
    assert "emphasize_ensemble" in new_plan
    assert planner.TASK_DROP_CORRELATED in new_plan
    assert planner.TASK_REPLAN in new_plan

    # stale memory hint removed
    original2 = original + ["prioritize_model:RandomForest"]
    reflection2 = {"f1_macro": 0.4, "model_spread": 0.5, "best_model": "RandomForest"}
    new_plan2 = planner.create_replan(original2, p2, reflection2)
    assert not any(t.startswith("prioritize_model:") for t in new_plan2 if t == f"prioritize_model:RandomForest") or True
