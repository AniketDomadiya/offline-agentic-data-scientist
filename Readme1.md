# Key Findings from the Sales Dataset

During exploration of the Sales.csv dataset, several non-obvious issues emerged that significantly affect how the system should reason about the problem.

## 1.1 Ground Truth is Not Obvious

One of the first challenges was identifying the ground truth column. Unlike standard ML datasets, this dataset does not explicitly define a target variable.

At first glance, multiple columns could reasonably serve as a prediction target: `customer_rating`, `revenue_usd`, `units_sold`. Each of these represents a meaningful outcome of a transaction. Because of this, the target detection process is not straightforward.

To address this, I implemented a heuristic scoring approach that ranks columns by how likely they are to be targets. Instead of selecting only one hardcoded column, the system stores a ranked list of possible targets with associated confidence scores. So, if modelling with the top-ranked target produces weak or unstable results, the agent can attempt the next most likely candidate and re-evaluate.

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

Instead of selecting a single ground truth column, the system will:
1. Score all potential target candidates.
2. Rank them by confidence.
3. Attempt modelling with the highest-ranked candidate.
4. If performance is poor or inconsistent, attempt the next candidate.

This allows limited re-planning without human intervention.

However, in highly ambiguous datasets, human confirmation may still be necessary. The system will explicitly flag low-confidence target detection cases rather than silently proceeding.

---

## 3.2 Conditional Problem Type Inference

Problem type will initially be inferred from the selected target using:
- Cardinality
- Datatype
- Distribution properties

However, because of the ambiguity observed in `customer_rating`, the system will allow:
- Trying both regression and classification interpretations (if uncertainty is high)
- Comparing validation performance
- Selecting the formulation that behaves more consistently

This avoids rigid assumptions.

---

## 3.3 Data-Aware Preprocessing Decisions

Preprocessing will depend on extracted dataset signals:
- High skew → apply log transformation
- Strong multicollinearity → remove redundant features
- High-cardinality categoricals → alternative encoding
- Datetime features → extract temporal components

These decisions will not be hardcoded to specific column names.

---

## 3.4 Reflection and Re-Planning

After initial training:
- If performance is unstable across folds → reconsider preprocessing.
- If one feature dominates → check for leakage.
- If model underperforms baseline → try alternative model family.
- If target confidence was low → attempt next-ranked target candidate.

Reflection is triggered by measurable signals rather than arbitrary loops.

# 4. Limitations and Realistic Considerations

Target detection relies on heuristic scoring and cannot guarantee correctness. In ambiguous real-world datasets, fully autonomous systems may require optional human confirmation.

The goal of the final system is not to maximise predictive accuracy, but to demonstrate structured reasoning under uncertainty.
