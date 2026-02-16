# 1. Key Findings from the Sales Dataset

During exploration of the Sales.csv dataset, the dataset characteristics, statistics, and EDA outputs are printed in the `eda.ipynb` notebook. This document summarizes the key findings and challenges identified for the Sales dataset and highlights aspects that are particularly relevant for designing the agentic system.

## 1.1 Ground Truth is Not Obvious

One of the first challenges was identifying the ground truth column. This dataset does not explicitly define a target variable.

At first glance, multiple columns could reasonably serve as a prediction target: `customer_rating`, `revenue_usd`, `units_sold`. Each of these represents a meaningful outcome of a transaction. Because of this, the target detection process is not straightforward.

To address this, A heuristic scoring approach is implemented that ranks columns by how likely they are to be targets. Instead of selecting only one hardcoded column, the system stores a ranked list of possible targets with associated confidence scores. So, if modelling with the top-ranked target produces weak or unstable results, the agent can attempt the next most likely candidate and re-evaluate.

This design acknowledges that target detection itself is uncertain, and the system should not assume it is always correct on the first attempt.

## 1.2 Interdependence Between Target Detection and Problem Type

Another complexity is that determining the problem type (classification vs regression) depends on the predicted target column, but identifying the correct target may itself depend on knowing the problem type.

For example, the code detected `customer_rating` as the most likely target. However:
- It has relatively low cardinality.
- It contains floating-point values.
- It lies within a limited range (3.0–5.0).

Because of the limited number of unique values, the automatic logic classified it as a classification problem. However, since the values are continuous and ordinal in nature, it could also reasonably be treated as a regression task.

If instead `revenue_usd` or `units_sold` were selected as targets, the problem would clearly be regression.
This creates a circular dependency:
- Target → determines problem type
- Problem type → influences how target should be interpreted

This ambiguity is an important finding from the dataset and highlights why fully rigid pipelines are risky.

## 1.3 Structural Redundancy in Features

The dataset contains deterministic relationships:
- `revenue_usd` is derived from `units_sold × final_price_usd`
- `final_price_usd` depends on `base_price_usd` and `discount_percent`

This introduces strong multicollinearity and redundancy. If all these variables are used together in linear models, it may destabilise coefficient estimation. This becomes clear by inspecting correlation matrix.

## 1.4 Skewness and Legitimate Extremes

Financial variables such as revenue and pricing show strong right skew.

However, these extreme values appear to represent legitimate high-value transactions rather than data errors. Removing them would likely discard useful information. So, transformation (e.g., log scaling) is more appropriate than deletion.

# 2. Key Dataset Challenges

Identified Modelling challenges:
- Ambiguous ground truth selection
- Interdependence between target detection and problem type
- Feature redundancy and multicollinearity
- Skewed financial variables
- Mixed feature types (ex. order_id column is not just numeric therfore, to identify it as id column, the more in-depth logic is needed)


# 3. Proposed Plan for the Offline Agentic Data Scientist

The final system will not assume a predefined target or fixed workflow. Instead, it will reason in stages.

## 3.1 Target Ranking Rather Than Single Selection

If --target is specified as 'auto' while running the system, then the first challenge is identifying the ground truth..

Instead of selecting a single column and committing to it, the system will:
1. Generate a ranked list of possible target candidates using heuristic scoring (name patterns, distribution properties, uniqueness ratio, etc.).
2. Attempt modelling using the highest-ranked candidate.
3. Evaluate performance stability.
4. If performance is weak, inconsistent, or suspiciously high (possible leakage), the system will try the next candidate in the ranked list.
This creates a controlled retry mechanism rather than blind repetition.

Retry will be triggered when:
- Cross-validation variance is very high.
- Performance is near-random.
- Performance is unrealistically high (indicating leakage).
- Target confidence score was low.
This approach acknowledges uncertainty in target detection and avoids rigid assumptions.

## 3.2 Metrics will be chosen conditionally:
- Balanced classes → Accuracy + F1-score
- Imbalanced classes → F1-score or weighted F1
- Very severe imbalance → Consider ROC-AUC

## 3.3 Model Selection Strategy
Instead of trying many models randomly, the system will choose model families based on dataset signals.

- If multicollinearity is high → Prefer regularised linear models (Ridge, Lasso)

- If non-linear patterns suspected → Prefer tree-based models (Random Forest)

- If dataset is small → Prefer simpler models to avoid overfitting

Initial baseline model will always be trained first for reference. More complex models will only be tried if baseline underperforms.

## 3.4 Data-Aware Preprocessing Decisions

Preprocessing will depend on extracted dataset signals:
- High skew → apply log transformation
- Strong outliers → robust scaling
- Strong multicollinearity → remove redundant features
- High-cardinality categoricals → alternative encoding
- Datetime features → extract components (year, month)

These decisions will not be hardcoded to specific column names.

## 3.5 Reproducibility
Deterministic Execution:
- Fixed random seeds for:
    - train_test_split
    - Cross-validation
    - Models (e.g., RandomForest random_state)
- Controlled retry depth to avoid non-deterministic loops.

Logged Configuration State:

- Every experiment attempt will log: Target candidate used, Problem type assumption, Model selected, Preprocessing steps applied, Metric chosen, Random seed used

This ensures the system can reproduce identical results, explain its decisions and be audited.

## 3.6 Reflection and ReAct Mechanism

The system will follow a simplified ReAct-style loop:

1. Observe dataset signals.
2. Plan preprocessing and model.
3. Act (train and evaluate).
4. Reflect on results.
5. Re-plan if necessary.

After initial training:
- If performance is unstable across folds → reconsider preprocessing.
- If one feature dominates → check for leakage.
- If model underperforms baseline → try alternative model family.
- If target confidence was low → attempt next-ranked target candidate.

Re-planning actions may include:
- Switching model family.
- Modifying preprocessing.
- Trying next-ranked target candidate.
- Changing evaluation metric.

The system will not loop indefinitely. A maximum retry depth will be enforced to prevent uncontrolled recursion.

# 4. Limitations and Realistic Considerations

Target detection relies on heuristic scoring and cannot guarantee correctness. In ambiguous real-world datasets, fully autonomous systems may require optional human confirmation.

The goal of the final system is not to maximise predictive accuracy, but to demonstrate structured reasoning under uncertainty.
