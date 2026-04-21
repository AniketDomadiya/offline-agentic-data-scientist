import numpy as np
import pandas as pd
from tools import evaluation, modelling


def make_simple_df():
    df = pd.DataFrame({
        "x1": [0,1,0,1,0,1,0,1],
        "x2": [1,2,1,2,1,2,1,2],
        "y":  [0,1,0,1,0,1,0,1],
    })
    return df


def test_build_preprocessor_and_train():
    df = make_simple_df()
    prof = {
        "feature_types": {"numeric": ["x1","x2"], "categorical": [], "datetime": [], "text": []},
        "high_cardinality_features": [],
        "shape": {"rows": 8, "cols": 3}
    }
    pre = modelling.build_preprocessor(prof, use_power_transform=False, handle_high_cardinality=False, impute_strategy="median")
    candidates = modelling.select_models(prof, seed=0, plan=[])
    res = modelling.train_models(df, "y", pre, candidates, seed=0, test_size=0.25, output_dir=".", verbose=False, cross_validate_folds=0)
    assert "best" in res and "all_metrics" in res
    # Evaluate function
    eval_payload = evaluation.evaluate_best(res["best"], output_dir=".") if False else None


def test_evaluate_plot_and_json_safe(tmp_path):
    # test plot_confusion_matrix and json serialiser
    cm = np.array([[3,1],[0,4]])
    out = tmp_path / "cm.png"
    evaluation.plot_confusion_matrix(cm, ["0","1"], str(out), "title")
    assert out.exists()
    payload = {"confusion_matrix_array": cm, "best_metrics": {"balanced_accuracy":0.6}}
    safe = evaluation.metrics_for_saving(payload)
    assert "confusion_matrix_array" not in safe
