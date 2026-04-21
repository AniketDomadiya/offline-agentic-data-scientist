import pandas as pd
from agentic_data_scientist import AgenticDataScientist


def test_apply_data_preparation_basic():
    agent = AgenticDataScientist(verbose=False, memory_path=":memory:")
    df = pd.DataFrame({
        "id": [1,2,3,4],
        "dt": ["2020-01-01","2020-02-01","2020-03-01","2020-04-01"],
        "x": [1,2,3,4],
        "y": [0,1,0,1],
    })
    profile = {
        "feature_types": {"numeric": ["x"], "categorical": [], "datetime": ["dt"], "text": []},
        "id_features": ["id"],
        "missing_pct": {},
        "shape": {"rows": 4, "cols": 4},
        "notes": []
    }
    plan = ["drop_duplicates","drop_id_columns","extract_datetime","drop_severe_missing","drop_correlated"]
    df2, prof2 = agent._apply_data_preparation(df, plan, profile, target="y")
    assert "dt_year" in df2.columns or "dt_month" in df2.columns
    assert "id" not in df2.columns
