# Dataset Documentation

This folder contains the four classification datasets used to evaluate the Offline Agentic Data Scientist across diverse scenarios. Each dataset was selected to exercise a different combination of agent code paths, together they cover small, standard, imbalanced, and large-scale classification challenges.

---

## Overview

| Dataset | File | Rows | Features | Classes | Imbalance | Agent Scenario |
|---|---|---|---|---|---|---|
| Sales (Footwear) | `Sales.csv` | 30 000 | 17 | 21 (ratings 3.0–5.0) | Near-balanced | `scenario:standard` |
| Iris | `iris.csv` | 150 | 4 | 3 | Balanced | `scenario:tiny` |
| Titanic | `titanic.csv` | 891 | 7 | 2 | Mild (~1.7×) | `scenario:standard` |
| Adult Income | `adult_income.csv` | 48 842 | 14 | 2 | Moderate (~3×) | `scenario:large` |

---

## Dataset 1 - Sales (Footwear)

**File:** `Sales.csv`  
**Source:** Custom synthetic dataset used for Deliverable 1 (EDA notebook)  
**Task:** Multi-class classification - predict customer rating

### Description

A synthetic retail sales dataset representing footwear transactions across multiple brands, countries, and sales channels. Each row is one order. The classification task is to predict `customer_rating`, a float score in the range 3.0–5.0 with 21 distinct values (steps of 0.1), making this a challenging fine-grained multi-class problem.

### Schema

| Column | Type | Description |
|---|---|---|
| `order_id` | identifier | Unique order ID - excluded by agent (no signal) |
| `order_date` | datetime | Transaction date - agent extracts year and month features |
| `brand` | categorical | Footwear brand (Adidas, Nike, Puma, Reebok, New Balance) |
| `model_name` | categorical (high-card) | Specific model - high cardinality, agent uses ordinal encoding |
| `category` | categorical | Product category (Running, Gym, Lifestyle, Training) |
| `gender` | categorical | Target gender (Men, Women, Unisex) |
| `size` | categorical | UK shoe size (7–11) |
| `color` | categorical | Colour (Red, Blue, White, Grey) |
| `base_price_usd` | numeric | Original price before discount |
| `discount_percent` | numeric | Discount percentage applied |
| `final_price_usd` | numeric | Price after discount - highly correlated with `base_price_usd` |
| `units_sold` | numeric (behaves categorical) | Units per order (1–4) |
| `revenue_usd` | numeric | Total revenue - moderately skewed, outliers present |
| `payment_method` | categorical | Card, Cash, or Wallet |
| `sales_channel` | categorical | Online or Retail Store |
| `country` | categorical | Country of purchase (India, Pakistan, UAE, UK, USA) |
| `customer_income_level` | categorical | Customer income bracket (Low, Medium, High) |
| `customer_rating` | **target** | Rating score 3.0–5.0 with 21 classes |

### Key Characteristics

- **30 000 rows, 18 columns** (17 usable features after dropping `order_id`)
- **No missing values** in any column
- **No duplicate rows**
- **21 classes** - near-uniform distribution (~4.8–5.3 % per class), except 3.0 and 5.0 which are slightly rarer (~2.4 %)
- **High-cardinality column:** `model_name` (many unique values) - triggers `handle_high_cardinality` → OrdinalEncoder
- **Datetime column:** `order_date` - triggers `extract_datetime` → year and month extracted as numeric features
- **Identifier column:** `order_id` - triggers `drop_id_columns`
- **Highly correlated pair:** `final_price_usd` / `base_price_usd` (|r| > 0.8) - one dropped by `drop_correlated`
- **Moderately skewed:** `revenue_usd` (skewness ≈ 0.78) - outliers detected by IQR method


### How to Obtain

Already present in `data/Sales.csv` - used as the primary dataset for Deliverable 1.

---

## Dataset 2 - Iris

**File:** `iris.csv`  
**Source:** `sklearn.datasets.load_iris`
**Task:** Multi-class classification - predict iris species

### Description

The classic iris flower dataset introduced by Ronald Fisher in 1936. Each row represents one flower measurement. Three species of iris are classified based on four physical measurements of sepal and petal dimensions. This is a textbook classification problem with clean, well-separated classes.

### Schema

| Column | Type | Description |
|---|---|---|
| `sepal_length` | numeric | Sepal length in cm |
| `sepal_width` | numeric | Sepal width in cm |
| `petal_length` | numeric | Petal length in cm |
| `petal_width` | numeric | Petal width in cm |
| `species` | **target** | Iris species: setosa, versicolor, virginica |

### Key Characteristics

- **150 rows, 5 columns** (4 features)
- **3 classes**, perfectly balanced - exactly 50 samples per class
- **No missing values**
- **No duplicate rows**
- All features are continuous numeric - no categorical encoding needed
- Classes are linearly separable for setosa; versicolor and virginica overlap slightly


### How to Obtain

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = iris.target_names[iris.target]
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df.to_csv("data/iris.csv", index=False)
```

---

## Dataset 3 - Titanic

**File:** `titanic.csv`  
**Source:** `seaborn.load_dataset("titanic")`  
**Task:** Binary classification - predict passenger survival

### Description

The Titanic passenger survival dataset. Each row represents one passenger aboard the RMS Titanic. The task is to predict whether a passenger survived (1) or did not survive (0) based on demographic and ticket information. This is one of the most widely studied binary classification datasets.

### Schema

| Column | Type | Description |
|---|---|---|
| `pclass` | numeric (1/2/3) | Passenger class (1st, 2nd, 3rd) |
| `sex` | categorical | Passenger sex (male, female) |
| `age` | numeric | Age in years - has missing values (~20%) |
| `sibsp` | numeric | Number of siblings/spouses aboard |
| `parch` | numeric | Number of parents/children aboard |
| `fare` | numeric | Ticket fare paid |
| `embarked` | categorical | Port of embarkation: C (Cherbourg), Q (Queenstown), S (Southampton) |
| `survived` | **target** | 0 = did not survive, 1 = survived |

### Key Characteristics

- **891 rows, 8 columns** (7 features)
- **2 classes**: 342 survived (38.4 %), 549 did not (61.6 %) - imbalance ratio ≈ 1.7×
- **Missing values:** `age` ≈ 19.9 % missing, `embarked` ≈ 0.2 % missing
- **Mixed feature types:** numeric (`age`, `fare`, `sibsp`, `parch`) and categorical (`sex`, `embarked`, `pclass`)
- `pclass` is an integer but behaves categorically (only 3 values) - the agent's `behaves_like_categorical` heuristic handles this

### How to Obtain

```python
import seaborn as sns
df = sns.load_dataset("titanic")
df = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]]
df.to_csv("data/titanic.csv", index=False)
```

---

## Dataset 4 - Adult Income

**File:** `adult_income.csv`  
**Source:** UCI Machine Learning Repository - [Adult dataset](https://archive.ics.uci.edu/dataset/2/adult)  
**Task:** Binary classification - predict whether annual income exceeds $50K

### Description

The Adult Income dataset (also called the Census Income dataset) was extracted from the 1994 US Census database. Each row represents one individual. The classification task is to predict whether a person's income is `>50K` or `<=50K` based on demographic and employment attributes. It is a standard benchmark for classification with mixed feature types, categorical encoding challenges, and class imbalance.

### Schema

| Column | Type | Description |
|---|---|---|
| `age` | numeric | Age in years |
| `workclass` | categorical | Employment type (Private, Self-emp, Government, etc.) |
| `education` | categorical | Highest education level achieved |
| `education_num` | numeric | Numeric encoding of education level (correlated with `education`) |
| `marital_status` | categorical | Marital status |
| `occupation` | categorical (high-card) | Job type - 14 unique values, has missing values |
| `relationship` | categorical | Relationship to household head |
| `race` | categorical | Race |
| `sex` | categorical | Sex (Male, Female) |
| `capital_gain` | numeric | Capital gains - highly skewed (most values are 0) |
| `capital_loss` | numeric | Capital losses - highly skewed (most values are 0) |
| `hours_per_week` | numeric | Hours worked per week |
| `native_country` | categorical (high-card) | Country of origin - 41 unique values, has missing values |
| `income` | **target** | `<=50K` or `>50K` |

Note: `fnlwgt` (census sampling weight) is excluded - it is not a predictive feature.

### Key Characteristics

- **48 842 rows, 15 columns** (14 features after dropping `fnlwgt`)
- **2 classes:** `<=50K` ≈ 75.9 %, `>50K` ≈ 24.1 % - imbalance ratio ≈ 3.1×
- **Missing values:** `occupation` ≈ 5.7 % missing, `workclass` ≈ 5.6 % missing, `native_country` ≈ 1.8 % missing - stored as NaN (original dataset uses `?` as placeholder, converted on load)
- **High-cardinality categoricals:** `native_country` (41 unique), `occupation` (14 unique) - triggers `handle_high_cardinality`
- **Highly skewed features:** `capital_gain` and `capital_loss` (most rows are 0, extreme outliers exist) - triggers `apply_skew_transform`
- **Correlated columns:** `education` and `education_num` encode the same information - `drop_correlated` may remove one

### How to Obtain

```python
import pandas as pd

col_names = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
df_train = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None, names=col_names, skipinitialspace=True
)
df_test = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    header=None, names=col_names, skipinitialspace=True, skiprows=1
)
df = pd.concat([df_train, df_test], ignore_index=True)
df["income"] = df["income"].str.rstrip(".")   # remove trailing dot in test set
df.replace("?", float("nan"), inplace=True)
df.drop(columns=["fnlwgt"], inplace=True)
df.to_csv("data/adult_income.csv", index=False)
```

---

## Running the Agent

```bash
# Dataset 1 - Sales (21-class, large, mixed types)
python run_agent.py --data data/Sales.csv --target customer_rating --max_replans 2

# Dataset 2 - Iris (tiny, balanced, clean)
python run_agent.py --data data/iris.csv --target species --max_replans 1

# Dataset 3 - Titanic (standard, binary, missing values)
python run_agent.py --data data/titanic.csv --target survived --max_replans 1

# Dataset 4 - Adult Income (large, binary, imbalanced, high-cardinality)
python run_agent.py --data data/adult_income.csv --target income --max_replans 1
```

---

## Diversity Justification

These four datasets were selected to ensure broad coverage of the agent's decision space:

| Dimension | Sales | Iris | Titanic | Adult Income |
|---|---|---|---|---|
| Dataset size | Large (30k) | Tiny (150) | Small (891) | Large (49k) |
| Number of classes | 21 | 3 | 2 | 2 |
| Class balance | Near-uniform | Perfectly balanced | Mild imbalance | Moderate imbalance |
| Missing values | None | None | Yes (~20% Age) | Yes (occupation, workclass) |
| Feature types | Mixed | Numeric only | Mixed | Mixed |
| High-cardinality cats | Yes (model_name) | No | No | Yes (native_country) |
| Skewed features | Moderate (revenue) | No | No | Yes (capital_gain) |
| Correlated pairs | Yes (price cols) | No | No | Yes (education cols) |
| Datetime features | Yes (order_date) | No | No | No |
| Agent scenario | standard | tiny | standard | large |

