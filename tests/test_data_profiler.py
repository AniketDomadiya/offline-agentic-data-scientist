import pandas as pd
from tools import data_profiler


def test_is_classification_suitable_and_infer():
    df = pd.DataFrame({
        "a": [0,1,0,1],
        "b": ["x","y","x","y"],
        "target": [0,1,0,1],
    })
    assert data_profiler.is_classification_suitable(df["target"]) is True
    # When removing the true target, infer_target_column may still find a suitable column
    inferred = data_profiler.infer_target_column(df.drop(columns=["target"]))
    assert inferred is None or inferred in {"a", "b"}


def test_profile_dataset_basic():
    df = pd.DataFrame({"f1": [1,2,1,2], "label": [0,1,0,1]})
    prof = data_profiler.profile_dataset(df, "label")
    assert prof["shape"]["rows"] == 4
    assert prof["n_classes"] == 2
    assert isinstance(prof["feature_types"], dict)
