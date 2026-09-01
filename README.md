# Offline Agentic Data Scientist

An end-to-end, offline machine learning workflow designed to autonomously profile tabular datasets, build a task plan, train and evaluate models, and reflect on performance without depending on external LLM APIs or cloud services.

This project is built around the idea of an agentic data science pipeline: it inspects the dataset, reasons about dataset quality and structure, chooses a suitable modelling strategy, runs model training and evaluation, and then critiques the results to decide whether a re-plan is necessary.

The objective of this project is not to build a perfect or highly-optimised ML system with the best possible accuracy. Instead, the goal is to understand how an agentic data scientist works under the hood: how it profiles data, how it decides what actions to take, how different ML components connect together, how it reacts to weak performance, and how it reflects and adapts its strategy over time.

This project is intentionally designed as an offline agentic data scientist to avoid dependence on online LLM APIs or cloud-based model services. The focus is on building the system from scratch using local logic, heuristics, data analysis, model evaluation, and structured reasoning.

---

## What this project does

The agent is designed to handle classification-focused tabular data and perform the following steps automatically:

- dataset profiling and target detection
- identification of data quality issues such as duplicates, missing values, and high-cardinality features
- plan generation based on dataset characteristics
- preprocessing strategy selection
- model selection and training
- evaluation using standard classification metrics
- reflection and optional re-planning when performance is weak
- saving outputs for review and reporting

This agent will use rule-based reasoning, heuristics, and meta-learning to autonomously:

- Profile datasets
- Plan execution workflows
- Train and evaluate models
- Reflect on results
- Learn from experience

This is an offline, lightweight approach to agentic data science using rule-based logic and heuristics rather than external AI services.

---

## Why this project

This repository demonstrates a practical, self-contained framework for building an autonomous data science agent in a local environment. It is useful for:

- understanding the internal structure of an agentic data science workflow
- exploring how planning, reflection, and model selection fit together
- building offline ML automation for research or portfolio projects
- learning how an autonomous system adapts to dataset characteristics without hard-coded assumptions
- demonstrating a local, privacy-friendly, no-API ML workflow in a portfolio project

---

## Core workflow

1. Load a CSV dataset.
2. Detect or validate the target column.
3. Profile the dataset for shape, type distribution, imbalance, missingness, skewness, and correlations.
4. Generate an execution plan based on the dataset profile.
5. Build a preprocessing pipeline.
6. Train multiple candidate models.
7. Evaluate the best model.
8. Reflect on the results and decide if the plan should be revised.
9. Save outputs and reporting artefacts.

---

## Project structure

```text
.
├── README.md
├── requirements.txt
├── run_agent.py
├── agentic_data_scientist.py
├── agent_memory.json
├── agents/
│   ├── __init__.py
│   ├── planner.py
│   ├── reflector.py
│   └── memory.py
├── tools/
│   ├── __init__.py
│   ├── data_profiler.py
│   ├── modelling.py
│   └── evaluation.py
├── data/
│   ├── README.md
│   ├── example_dataset.csv
│   └── other dataset files
├── outputs/
├── report/
├── tests/
│   ├── sanity_check.py
│   └── other test files
└── my_outputs/
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate it

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Quick start

Run the agent on a dataset using auto target detection:

```bash
python run_agent.py --data data/example_dataset.csv --target auto
```

Run it with an explicit target column:

```bash
python run_agent.py --data data/your_dataset.csv --target target_column_name
```

### Optional arguments

```bash
python run_agent.py \
  --data data/your_dataset.csv \
  --target target_column_name \
  --output_root my_outputs \
  --seed 42 \
  --test_size 0.2 \
  --max_replans 2 \
  --quiet
```

Arguments:

- `--data`: path to the CSV file
- `--target`: target column name or `auto`
- `--output_root`: output directory for generated artefacts
- `--seed`: random seed for reproducibility
- `--test_size`: test split proportion
- `--max_replans`: maximum re-planning attempts
- `--quiet`: reduce console logging

---

## Example outputs

After a run, the output directory contains files such as:

- `report.md` — markdown summary of the run
- `eda_summary.json` — dataset profiling information
- `plan.json` — execution plan generated by the planner
- `metrics.json` — evaluation performance metrics
- `reflection.json` — reflection notes and suggestions
- `confusion_matrix.png` — confusion matrix visualisation

---

## Notes

- The project is intended for tabular classification tasks.
- It is designed to run completely offline without external model APIs.
- The agent uses rule-based heuristics and lightweight model selection, rather than large foundation models.
- It is a practical prototype for exploring autonomous data-science behaviour in a local environment.

---

## License

This project is available under the repository license. See the license file in the project root for details.

---

## Contact / portfolio note

This repository showcases an offline agentic data science workflow combining dataset profiling, planning, model evaluation, and reflective decision-making in a single local pipeline.

It is intended as a portfolio project and demonstration of autonomous ML pipeline design.
