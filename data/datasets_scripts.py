"""
download_datasets.py
====================
Run this once from your project root to download all four datasets
and save them as CSV files in data/

Usage:
    python download_datasets.py

Requirements: pandas, scikit-learn, seaborn  (already in requirements.txt)
"""

import os
import warnings
import pandas as pd

os.makedirs("data", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. IRIS  (sklearn built-in)
#    150 rows · 3 balanced classes · 4 numeric features · no missing
#    Exercises: scenario:tiny, use_cross_validation
# ═══════════════════════════════════════════════════════════════════════════
print("Saving iris.csv ...")
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df_iris = iris.frame.copy()
df_iris["target"] = iris.target_names[iris.target]
df_iris.drop(columns=["target"], inplace=False)  # keep numeric target name
# rename columns to be clean
df_iris.columns = [
    "sepal_length", "sepal_width", "petal_length", "petal_width", "species"
]
df_iris.to_csv("data/iris.csv", index=False)
print(f"  iris.csv  — {df_iris.shape[0]} rows, {df_iris.shape[1]} cols, "
      f"target='species', classes={df_iris['species'].unique().tolist()}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. TITANIC  (seaborn built-in)
#    891 rows · 2 classes (survived 0/1) · mixed types · missing Age/Cabin
#    Exercises: imputation, drop_severe_missing (Cabin ~77% missing),
#               OHE for categoricals, mild imbalance
# ═══════════════════════════════════════════════════════════════════════════
print("Saving titanic.csv ...")
import seaborn as sns
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_titanic = sns.load_dataset("titanic")

# Keep only the columns that make sense as features + target
keep_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]
df_titanic = df_titanic[keep_cols].copy()
df_titanic.to_csv("data/titanic.csv", index=False)
vc = df_titanic["survived"].value_counts()
print(f"  titanic.csv — {df_titanic.shape[0]} rows, {df_titanic.shape[1]} cols, "
      f"target='survived', class counts={vc.to_dict()}, "
      f"missing Age={df_titanic['age'].isna().sum()}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. WINE QUALITY  (UCI — direct CSV URL, no account needed)
#    1 599 rows · 6 classes (quality 3–8) · all numeric · skewed features
#    imbalance ratio ~5–7 (quality 5 and 6 dominate)
#    Exercises: scenario:imbalanced, apply_skew_transform,
#               consider_imbalance, per-class analysis
# ═══════════════════════════════════════════════════════════════════════════
print("Saving wine_quality.csv  (downloading from UCI) ...")
try:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-red.csv"
    )
    df_wine = pd.read_csv(url, sep=";")
    # Rename so target column is clearly named
    df_wine.rename(columns={"quality": "wine_quality"}, inplace=True)
    df_wine.to_csv("data/wine_quality.csv", index=False)
    vc_w = df_wine["wine_quality"].value_counts().sort_index()
    print(f"  wine_quality.csv — {df_wine.shape[0]} rows, {df_wine.shape[1]} cols, "
          f"target='wine_quality', class counts={vc_w.to_dict()}")
except Exception as exc:
    print(f"  WARNING: could not download wine quality: {exc}")
    print("  Manual fallback: go to https://archive.ics.uci.edu/dataset/186/wine+quality")
    print("  Download winequality-red.csv, rename column 'quality' → 'wine_quality',")
    print("  save as data/wine_quality.csv")


# ═══════════════════════════════════════════════════════════════════════════
# 4. ADULT INCOME  (UCI — direct URL, no account needed)
#    48 842 rows · 2 classes (<=50K / >50K) · mixed types
#    high-cardinality categoricals (occupation, native-country)
#    missing values encoded as ' ?'  →  cleaned to NaN here
#    imbalance ratio ~3:1
#    Exercises: scenario:large, handle_high_cardinality,
#               consider_imbalance, imputation
# ═══════════════════════════════════════════════════════════════════════════
print("Saving adult_income.csv  (downloading from UCI) ...")
try:
    col_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income",
    ]
    url_train = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )
    url_test  = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    )
    df_train = pd.read_csv(url_train, header=None, names=col_names,
                           skipinitialspace=True)
    df_test  = pd.read_csv(url_test,  header=None, names=col_names,
                           skipinitialspace=True, skiprows=1)

    df_adult = pd.concat([df_train, df_test], ignore_index=True)

    # Strip trailing dots from the test set income values  (">50K." → ">50K")
    df_adult["income"] = df_adult["income"].str.rstrip(".")

    # Replace ' ?' placeholders with actual NaN so the agent can impute them
    df_adult.replace("?", float("nan"), inplace=True)

    # Drop fnlwgt (sampling weight, not a real feature)
    df_adult.drop(columns=["fnlwgt"], inplace=True)

    df_adult.to_csv("data/adult_income.csv", index=False)
    vc_a = df_adult["income"].value_counts()
    print(f"  adult_income.csv — {df_adult.shape[0]} rows, {df_adult.shape[1]} cols, "
          f"target='income', class counts={vc_a.to_dict()}, "
          f"missing occupation={df_adult['occupation'].isna().sum()}")
except Exception as exc:
    print(f"  WARNING: could not download adult income: {exc}")
    print("  Manual fallback: go to https://archive.ics.uci.edu/dataset/2/adult")
    print("  Download adult.data + adult.test, combine them, add column names,")
    print("  replace '?' with NaN, save as data/adult_income.csv")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print()
print("Done. Run the agent on each dataset:")
print()
print("  python run_agent.py --data data/iris.csv         --target species   --max_replans 1")
print("  python run_agent.py --data data/titanic.csv      --target survived  --max_replans 1")
print("  python run_agent.py --data data/wine_quality.csv --target wine_quality --max_replans 2")
print("  python run_agent.py --data data/adult_income.csv --target income    --max_replans 1")