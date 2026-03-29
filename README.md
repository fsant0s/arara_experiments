# ARARA Experiments

Experimental monorepo for the **ARARA** (Adaptive Recommender Agent with Reflection and Action) framework.

## Projects

| Folder | Description |
|--------|-------------|
| [`advisory_contextual_bandits/`](./advisory_contextual_bandits/) | **Advisor experiment** — a bandit-guided mediation agent that assists users navigating multiple conversational recommender systems (CRS) in parallel. |
| [`pilot/`](./pilot/) | **Pilot experiment** — first evaluation of the ARARA framework using Neo4j, multi-agent orchestration, and structured recommendation scenarios across books and movies. |
| [`datasets/`](./datasets/) | **Shared data** — datasets used across experiments. Not duplicated per project. |

## Repository Structure

```
arara_experiments/
├── advisory_contextual_bandits/  # Advisor agent experiment
│   ├── pyproject.toml
│   ├── README.md
│   ├── .venv/         # Local env (uv venv / uv sync)
│   ├── components/    # Belief state, bandit policy, reward
│   ├── config/        # LLM client configs
│   ├── docs/          # Documentation and paper drafts
│   ├── analysis_results/
│   └── results_*/     # Experiment outputs
│
├── pilot/             # First ARARA experiment
│   ├── pyproject.toml
│   ├── README.md
│   ├── .venv/         # Local env (includes editable ../../arara)
│   ├── modules/       # Book and movie recommendation modules
│   ├── tools/         # LLM tool wrappers
│   └── test/          # Test scripts
│
├── datasets/          # Shared datasets (not duplicated)
│   └── README.md
│
└── .gitignore
```

## Setup

Each project has its **own** virtual environment under `advisory_contextual_bandits/.venv` and `pilot/.venv`. Create or refresh them with [uv](https://github.com/astral-sh/uv) from each folder:

```bash
cd advisory_contextual_bandits && uv venv && uv sync
cd ../pilot && uv venv && uv sync
```

**Pilot** installs the **Arara** framework as an editable dependency from a sibling folder `arara/` (same parent directory as `arara_experiments/`). If your layout differs, edit `[tool.uv.sources]` in `pilot/pyproject.toml`.

Activate per project:

```bash
source advisory_contextual_bandits/.venv/bin/activate   # advisory_contextual_bandits only
source pilot/.venv/bin/activate      # pilot only
```

## Environment Variables

Copy `.env.example` to `.env` (if available) and fill in your API keys:

```bash
OPENAI_API_KEY=sk-...
```
