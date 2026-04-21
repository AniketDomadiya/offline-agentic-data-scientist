# CE888 Agentic Data Scientist

**Assignment:** Offline Agentic AI for Data Science  
**Module:** CE888  
**Academic Year:** 2025/2026

---

## Overview

This repository contains the skeleton code for building an **Offline Agentic Data Scientist** - an autonomous agent that performs end-to-end classification tasks without relying on Large Language Models.

This agent will use **rule-based reasoning, heuristics, and meta-learning** to autonomously:
- Profile datasets
- Plan execution workflows
- Train and evaluate models
- Reflect on results
- Learn from experience

---

## Quick Start

### 1. Clone this repository

```bash
git clone https://github.com/sagihaider/ce888-agentic-data-scientist.git
cd ce888-agentic-data-scientist
```

### 2. Set up your environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Test the skeleton

```bash
python run_agent.py --data data/example_dataset.csv --target auto
```

You should see the agent run through the basic pipeline and generate outputs in `outputs/[timestamp]/`

---

## Project Structure

```
ce888-agentic-data-scientist/
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
│
├── agentic_data_scientist.py      # Core agent (Executor) - extend this
├── run_agent.py                   # Entry point - students run this
│
├── agents/                        # Agent components
│   ├── __init__.py
│   ├── planner.py                 # TODO: EXTEND THIS
│   ├── reflector.py               # TODO: EXTEND THIS
│   └── memory.py                  # Basic implementation - can extend
│
├── tools/                         # Data science tools
│   ├── __init__.py
│   ├── data_profiler.py          # Provided - can extend
│   ├── modelling.py              # Provided - can extend
│   └── evaluation.py             # Provided - can extend
│
├── data/                          # Datasets
│   ├── README.md                 # Dataset documentation template
│   └── example_dataset.csv       # Small demo dataset
│
├── outputs/                       # Generated outputs (gitignored)
│   └── .gitkeep
│
├── report/                        # Your technical report
│   ├── README.md                 # Report guidelines
│   └── REPORT.md                 # TODO: WRITE YOUR REPORT HERE
│
└── tests/                         # Test suite
    ├── __init__.py
    └── sanity_check.py           # Basic sanity test
```


## Running the Agent

### Basic Usage

```bash
python run_agent.py --data data/example_dataset.csv --target auto
```

### Custom Parameters

```bash
python run_agent.py \
    --data data/your_dataset.csv \
    --target target_column_name \
    --output_root my_outputs \
    --seed 42 \
    --test_size 0.2 \
    --max_replans 2
```

### Arguments

- `--data`: Path to CSV dataset (required)
- `--target`: Target column name or 'auto' for automatic detection (required)
- `--output_root`: Output directory (default: 'outputs')
- `--seed`: Random seed for reproducibility (default: 42)
- `--test_size`: Test set fraction (default: 0.2)
- `--max_replans`: Maximum replanning attempts (default: 1)
- `--quiet`: Reduce logging output

---

## Testing

Run the sanity check:

```bash
python tests/sanity_check.py
```

Run all tests (once you add them):

```bash
pytest tests/
```

With coverage:

```bash
pytest --cov=agents --cov=tools --cov-report=html tests/
```

---

## Expected Output

After running the agent, check `outputs/[timestamp]/` for:

- `report.md` - Human-readable summary report
- `eda_summary.json` - Dataset profile and characteristics
- `plan.json` - Generated execution plan
- `metrics.json` - Model performance metrics
- `reflection.json` - Agent's self-assessment and suggestions
- `confusion_matrix.png` - Confusion matrix visualization
