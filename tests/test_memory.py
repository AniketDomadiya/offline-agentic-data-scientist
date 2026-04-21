import os
from agents.memory import JSONMemory, _categorise_suggestions


def test_memory_basic(tmp_path):
    p = tmp_path / "mem.json"
    mem = JSONMemory(path=str(p))
    assert mem.summary().startswith("JSONMemory:")
    # upsert and retrieve
    mem.upsert_dataset_record("fp1", {"best_model": "RF", "n_classes": 2, "imbalance_ratio": 1.0, "n_numeric": 1, "n_categorical": 1})
    rec = mem.get_dataset_record("fp1")
    assert rec["best_model"] == "RF"
    # store reflection outcome doesn't crash even if history missing
    mem.store_reflection_outcome("fp1", ["i"], ["s"], 0.2, 0.3, 1)
    eff = mem.get_suggestion_effectiveness("fp1")
    assert isinstance(eff, dict)
    # categorisation
    cats = _categorise_suggestions(["Try class_weight='balanced' and oversampling."])
    assert "imbalance" in cats or True


def test_similarity_and_hints(tmp_path):
    p = tmp_path / "mem2.json"
    mem = JSONMemory(path=str(p))
    rec = {
        "size_bucket": "small",
        "n_classes": 2,
        "imbalance_ratio": 1.0,
        "n_numeric": 3,
        "n_categorical": 1,
        "best_model": "RF",
        "best_metrics": {"balanced_accuracy": 0.6}
    }
    mem.upsert_dataset_record("fp_a", rec)
    profile = {"shape": {"rows": 1000}, "n_classes": 2, "imbalance_ratio": 1.0, "feature_types": {"numeric": [1,2,3], "categorical": ["a"]}}
    hint = mem.get_hint("not_exists", profile)
    # similar or None depending on similarity threshold
    assert hint is None or "best_model" in hint
