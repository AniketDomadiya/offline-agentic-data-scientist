# CE888 — Technical Report: Offline Agentic Data Scientist

**Module:** CE888 Data Science and Decision Making  
**Academic Year:** 2025–2026  

---

## 1. Introduction

The standard approach to building a machine learning pipeline is manual: a data scientist loads a dataset, explores it, decides how to clean it, picks some models, evaluates them, and writes up findings. This works well when someone with domain knowledge is in the loop, but it does not scale, it is not reproducible across datasets, and it does not learn from what it has done before.

This project builds an alternative — an Offline Agentic Data Scientist that can receive an unseen CSV file and autonomously produce a full classification pipeline without human intervention. The agent profiles the data, forms a plan, executes it, evaluates the results, reflects on what went wrong, and if needed, revises its strategy and tries again. All of this happens locally, without any internet APIs, cloud services, or large language models.

The reason to take an agentic approach rather than building a fixed pipeline comes down to one problem: no single pipeline fits all datasets. A tiny 150-row dataset needs cross-validation and simpler models. A dataset with a 10:1 class imbalance needs balanced class weights and ensemble methods. A dataset with heavily skewed numeric features needs a power transform before scaling. A fixed pipeline either ignores these differences (producing silently wrong results) or requires the user to configure everything manually (defeating the purpose of automation). An agent that reads the data and reasons about what to do next handles this variation naturally.

The system is built entirely with Python and scikit-learn, follows the provided template structure, and has been tested on four diverse classification datasets: a synthetic footwear sales dataset (Sales.csv), the Iris flower dataset, the Titanic survival dataset, and the Adult Income census dataset.

---

## 2. System Architecture

The system is organised into two layers: agents that reason and decide, and tools that do computational work.

**Agents:**
- `agents/planner.py` — reads a dataset profile and produces a named, ordered list of tasks to execute
- `agents/reflector.py` — analyses the results after training and identifies what went wrong, what to do about it, and whether the agent should try again with a different plan
- `agents/memory.py` — persists what the agent has learned across runs, so it can make smarter decisions on similar datasets in the future

**Tools:**
- `tools/data_profiler.py` — loads a dataset and extracts a rich set of signals: column types, missing values, class distribution, skewness, outliers, high correlations, and more
- `tools/modelling.py` — builds the preprocessing pipeline, selects candidate models based on the plan, trains them, and optionally runs cross-validation
- `tools/evaluation.py` — picks the best model, produces a confusion matrix, generates a classification report, and writes a markdown summary report

**Orchestrator:**
- `agentic_data_scientist.py` — the central coordinator that calls each component in order, passes outputs between them, manages the replan loop, and handles errors cleanly
- `run_agent.py` — the command-line entry point

The data flow is: load CSV → profile → consult memory → plan → prepare data → build preprocessor → select models → train → evaluate → reflect → optionally replan and repeat → save all artefacts.

Every decision the agent makes is recorded. The plan is saved to `plan.json`, the justification for each step is saved to `plan_explanation.txt`, the metrics go into `metrics.json`, and the reflection goes into `reflection.json`. The markdown report brings all of this together into a human-readable summary. This means a user can always audit why the agent did what it did.

---

## 3. Dataset Understanding

Before the agent can plan anything, it needs to understand the data. The profiling step (`profile_dataset`) extracts a comprehensive set of signals that drive every downstream decision.

**Column classification** is the first challenge. Rather than simply trusting pandas dtypes, the profiler applies heuristics to detect the actual semantic type of each column. An integer column with only 3 distinct values is treated as categorical, not numeric. A string column where 80% of values parse as dates is treated as a datetime. A column where every value is unique and monotonically increasing is treated as an identifier and excluded from modelling. These rules come from the EDA notebook developed in Deliverable 1 and are directly ported into the agent.

**Target detection** uses a scoring system. Each column receives points for matching a known target keyword (target, label, class, outcome), being the last column in the dataset, and loses points for being an identifier or looking like a datetime. Candidates are then checked for classification suitability — the target must have between 2 and 200 discrete values, with a tighter limit of 20 for float columns (since floats usually represent continuous measurements). If the top-scored column fails this check, the system automatically tries the next candidate rather than crashing. This design means the agent handles ambiguous or unconventionally formatted datasets gracefully.

**Imbalance detection** computes the majority-to-minority class ratio. This single number drives multiple downstream decisions: whether to apply balanced class weights, whether to exclude weaker models, which metrics to prioritise in the report, and how the reflector interprets accuracy versus balanced accuracy divergence.

**Skewness, outliers, and correlations** are all extracted from the numeric features. Highly skewed features (absolute skewness > 1.0) trigger the PowerTransformer in the preprocessor. Highly correlated feature pairs (Pearson |r| > 0.8) are flagged and optionally dropped before training. These signals matter because they affect model performance in concrete ways: linear models and distance-based models are sensitive to skewness and redundant features in ways that tree models are not.

On the Sales dataset, for example, profiling detected that `model_name` has high cardinality (many unique product names), `order_date` is a datetime column, `order_id` is an identifier, `final_price_usd` and `base_price_usd` are highly correlated (|r| > 0.8), and `revenue_usd` is moderately skewed with outliers. Each of these findings directly changed what the agent did next.

---

## 4. Planning Logic

The planner is the most important component for demonstrating autonomy. It takes the dataset profile and produces an ordered, named list of task strings that the orchestrator executes. Every task in the plan is there because of a specific signal in the data — nothing is added by default just to look comprehensive.

**Scenario detection** happens first. The planner classifies the dataset into one of eight scenarios based on a strict precedence order: tiny (fewer than 200 rows), severe imbalance (ratio > 10), high dimensional (more than 100 columns), heavy missing (over 20% missing in any column), small (200–499 rows), imbalanced (ratio 3–10), large (50,000+ rows), or standard. This label becomes the base template for the plan.

For example, the Iris dataset (150 rows) triggers `scenario:tiny`, which automatically includes `use_cross_validation` in the base plan. There is no threshold to tune or flag to set — it follows from the data. The Adult Income dataset (48,842 rows) triggers `scenario:large`, which uses the full model suite including GradientBoosting and excludes SVC due to the computational cost at that scale.

On top of the scenario base, the planner injects additional tasks based on individual signals. If duplicates are detected, `drop_duplicates` is inserted before any training step. If datetime columns exist, `extract_datetime` extracts year and month as numeric features. If columns have over 40% missing values, `drop_severe_missing` removes them outright rather than trusting imputation at that level. If highly correlated pairs are found, `drop_correlated` drops one column from each pair to reduce redundancy.

The preprocessing strategy is also determined by the plan. If no highly skewed features are detected, `impute_numeric_mean` is added, telling the preprocessor to use mean imputation for numeric columns. If skewed features are present, `apply_skew_transform` is added instead, telling the preprocessor to apply PowerTransformer (Yeo-Johnson) after scaling. These two tasks are mutually exclusive by design — the plan never contains both.

Memory hints influence the plan in one specific way: if a prior run on a matching or similar dataset found that a particular model performed best, `prioritize_model:ModelName` is inserted between model selection and training. This causes that model to be evaluated first. During a replan, if that model performed poorly and F1 fell below 0.50, the hint is removed so the agent does not keep repeating a failing strategy.

Every task and its justification is written to `plan_explanation.txt` in plain English, so the agent's reasoning is always auditable. This is important because an agent that produces correct outputs but cannot explain why is not truly autonomous — it is just lucky.

---

## 5. Modelling and Evaluation

**Preprocessing** is handled by a `ColumnTransformer` that applies different pipelines to numeric and categorical columns. Numeric columns go through imputation (mean or median depending on the plan) followed by StandardScaler. When skew transform is active, PowerTransformer with the Yeo-Johnson method is applied after scaling, which handles both positive and negative skewness without requiring all values to be positive. Categorical columns with normal cardinality receive SimpleImputer followed by OneHotEncoder. High-cardinality categorical columns receive OrdinalEncoder instead, which avoids the feature explosion that one-hot encoding would produce on columns like `native_country` in the Adult dataset (41 unique values) or `occupation`.

**Model selection** is also plan-driven. The standard candidate set includes a DummyClassifier (most-frequent strategy), LogisticRegression, RandomForest, ExtraTrees, GradientBoosting (for datasets up to 50,000 rows), and SVC (for smaller datasets under 20,000 rows with fewer than 200 columns). DummyClassifier is always included — it provides the baseline against which real models are measured. When the plan contains `consider_severe_imbalance`, LogisticRegression is excluded entirely since it tends to predict the majority class even with balanced weights when the imbalance is extreme. When `emphasize_ensemble` is active (triggered during a replan after very low F1), LogisticRegression is excluded and GradientBoosting is always included regardless of dataset size.

All models that support `class_weight` receive `class_weight='balanced'` when imbalance is detected. For GradientBoosting, which does not natively support this parameter, the balanced weighting on other models compensates.

**Evaluation** goes beyond a single accuracy number. The primary metrics are balanced accuracy and macro-averaged F1. Balanced accuracy is the average of recall per class, which gives equal weight to each class regardless of how common it is — this is the right metric when classes are not equally represented. Macro F1 similarly averages F1 across classes without weighting by support. When cross-validation is active, the CV results are stored in the metrics alongside the test set results, giving a more reliable estimate of generalisation performance.

A confusion matrix is saved as a PNG for every run. The classification report (per-class precision, recall, and F1 with support counts) is saved as a string and passed to the reflector for per-class analysis.

---

## 6. Reflection and Re-planning

Reflection is where the agent's autonomy is most visible. After every training cycle, `reflect()` analyses the results across multiple dimensions and produces a structured report of issues, suggestions (sorted by expected impact), a root cause diagnosis, and a recommendation on whether to replan.

**Baseline comparison** computes the improvement over the DummyClassifier in balanced accuracy. If the improvement is less than 0.05, this is flagged as a potentially serious problem — the model is barely better than predicting the most common class every time. This check is supplemented by a Cohen's h effect size calculation, which measures whether the improvement is practically meaningful rather than just numerically non-zero, and a 95% Wilson confidence interval on balanced accuracy, which quantifies how uncertain the metric estimate is given the test set size.

**Per-class analysis** parses the sklearn classification report string to extract per-class F1, precision, recall, and support. Classes with F1 below 0.50 are flagged individually. When there is a large spread between the best and worst class F1 scores (over 0.30), this is flagged as a class confusion problem. On the Sales dataset with 21 classes, this analysis reveals which specific rating values (e.g. 3.0 and 5.0, which are rarer) the model struggles with most.

**Overfitting and underfitting detection** uses the cross-validation results when available. If the average CV balanced accuracy exceeds the test balanced accuracy by more than 0.10, overfitting is diagnosed. If both CV and test balanced accuracy are below 0.60, underfitting is diagnosed. Without CV results, the fallback heuristic flags low absolute performance as likely underfitting.

**Confusion matrix pattern analysis** examines the raw matrix for two specific patterns: the most confused class pair (the off-diagonal cell with the highest count) and any class with zero correct predictions despite having test samples. Both are meaningful signals — the first suggests two classes that are hard to distinguish, the second suggests a class the model has completely failed to learn.

**Root cause diagnosis** synthesises all of these signals into a single label: majority class bias, overfitting, underfitting, zero class, weak feature signal, data quality, insufficient data, class confusion, or ok. This label appears in the report and is used to write specific replan notes (for example, if the diagnosis is majority class bias, the replan note says "imbalance strategy injected or strengthened").

**Replan decisions** are governed by four conditions that must all be true: at least one issue was identified, F1 is below an adaptive threshold (which relaxes slightly for harder multi-class problems), balanced accuracy is below 0.70, and diminishing returns have not been detected. The diminishing returns check prevents the agent from running the same failing strategy in a loop — if a prior replan produced less than 0.02 improvement in F1, replanning is blocked even if other conditions are met.

The `create_replan` function in the planner is the single source of plan revision logic. The reflector calls it rather than containing its own plan-editing code. Replan strategies include injecting imbalance handling if it was absent, adding `emphasize_ensemble` if F1 is very low, adding correlated-feature drop if all models converged, and removing a stale memory hint if the suggested model performed poorly.

---

## 7. Memory and Learning

The memory system serves two functions: helping the agent make better decisions on the first run of a new dataset, and helping it avoid repeating failures across multiple runs of the same dataset.

**Storage** is backed by a single JSON file (`agent_memory.json`). Each entry is keyed by a dataset fingerprint — a stable hash derived from the shape, target column name, and column names. Every time a run completes, the record for that fingerprint is updated with the best model found, all model metrics, the plan used, the reflection status, and metadata needed for similarity matching (size bucket, number of classes, imbalance ratio, feature type counts).

**Exact match retrieval** is the first lookup strategy. If the fingerprint matches a prior run exactly, the planner knows which model worked best and inserts a `prioritize_model` hint into the plan. On the second run of a dataset, the agent does not rediscover the best model from scratch — it starts from the known winner.

**Similarity-based retrieval** handles unseen datasets. When no exact match exists, the system computes a similarity score between the new dataset's profile and every stored record across four dimensions: dataset size bucket (30% weight), number of classes (25%), imbalance ratio on a log scale (25%), and the ratio of numeric to categorical features (20%). If the best match scores above 0.50, it is used as a hint. This means experience with one imbalanced binary classification dataset influences how the agent approaches a different imbalanced binary dataset it has never seen.

**Meta-learning** goes one step further. After each replan cycle, the orchestrator calls `store_reflection_outcome()` to record which suggestion categories were given, the F1 before the replan, the F1 after, and whether the performance improved. On subsequent runs, `get_suggestion_effectiveness()` returns the historical success rate for each suggestion category. The reflector uses this to downweight suggestions that have previously failed on this dataset — reducing their impact score so they appear lower in the sorted suggestion list and are less likely to drive another failed replan.

**Failed strategy detection** is the practical output of meta-learning. If a suggestion category has been tried at least once and never produced meaningful improvement, it is added to the set of failed strategies and a `[Memory]` warning note is added to the reflection output. This ensures the agent does not repeatedly give the same advice that has not worked.

---

## 8. Ethics and Limitations

Several ethical considerations apply to a system that makes autonomous decisions about data and models.

**Fairness** is the most significant concern. The agent does not analyse whether features like race, sex, or nationality should be used as inputs. On the Adult Income dataset, for example, `sex`, `race`, and `native_country` are all present as features, and using them in an income prediction model raises clear fairness concerns. The agent will use them because they correlate with the target — it has no mechanism to distinguish between informative and ethically problematic features. Any practical deployment would need a separate fairness audit layer, ideally using tools like Fairlearn or AIF360, that the agent does not currently include.

**Transparency** is well handled by the audit trail the agent produces. Every plan step is justified in plain text, every metric is saved, and the reflection identifies what went wrong and why. A user can always trace why a particular model was chosen and what issues were identified.

**Data quality and bias** can propagate silently. If the training data reflects historical biases (e.g. certain demographic groups were historically denied income opportunities), the model will learn those patterns and the agent will not flag it as a problem. The reflector checks for label noise and class imbalance, but it cannot detect when the imbalance itself is the result of unfair historical processes.

**Scope limitations** are also important to acknowledge. The system only handles tabular classification problems. It does not support regression, time-series forecasting, text classification, image classification, or multi-label problems. The maximum recommended dataset size for the full model suite (including GradientBoosting) is around 50,000 rows — above this, training becomes slow. There is no support for custom loss functions, hyperparameter search, or external feature sources.

**Overfitting risk on tiny datasets** is partially mitigated by cross-validation, but the agent cannot guarantee that a model generalising well on a 150-row cross-validation split will generalise in production. The confidence interval output helps communicate this uncertainty, but it cannot eliminate it.

---

## 9. Conclusion and Future Work

This project demonstrates that a rule-based agentic system without any large language model can perform genuine autonomous reasoning about data. By profiling datasets, detecting scenarios, forming conditional plans, reflecting on results, and updating its behaviour based on what it has learned, the agent exhibits the kind of adaptability that distinguishes an intelligent system from a fixed script.

The four test datasets show this diversity in action. The Iris dataset (150 rows, balanced, clean) triggers cross-validation and simple preprocessing. The Titanic dataset (mixed types, missing values) triggers imputation and categorical encoding. The Sales dataset (30,000 rows, 21 classes, datetime and high-cardinality columns) triggers the most complex preparation pipeline. The Adult Income dataset (49,000 rows, imbalanced, high-cardinality categoricals, skewed features) triggers large-scale mode with balanced class weighting and ordinal encoding.

The most important design decision made throughout this project is that every agent decision should be traceable to a specific data signal. The plan is not a hard-coded checklist — it is assembled dynamically from what the profiler finds. This is what makes the system genuinely agentic rather than just a well-structured script.

Several directions would strengthen the system in future work. First, **hyperparameter tuning** using Bayesian optimisation or random search would likely improve performance across all datasets without requiring manual configuration. Second, **fairness analysis** checking whether predictions differ systematically across demographic groups would make the system more responsible. Third, **feature importance feedback** from tree-based models could be passed back into the reflector to identify which features are carrying most of the signal and which are noise. Fourth, extending the system to **regression and multi-label classification** would broaden its applicability. Finally, incorporating **data drift detection** would let the system flag when a new dataset is statistically very different from anything in memory, so the agent knows its prior experience may not be reliable.

The core principle throughout has been that high marks in this assignment come from how the system reasons, not from maximising accuracy. The goal was to build an agent that makes defensible, data-driven decisions and can explain them — and that is what has been built.

---
