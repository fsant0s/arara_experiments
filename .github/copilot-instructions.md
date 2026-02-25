# Copilot Instructions for arara_experiments

## Project Overview
- This repository is a research/experimentation environment for recommender systems, LLM-based agents, and evaluation frameworks.
- The codebase is organized by domain (e.g., `fidelity/`, `pilot/`, `datasets/`) and by function (e.g., `recsys/`, `advisor/`, `utils/`).
- Data and results are stored in `datasets/`, with subfolders for different sources and formats (e.g., `INSPIRED2-main/`, `instructrec/`, `multi-aspect-reviews/`).

## Key Components
- **fidelity/**: Main experiments, metrics, and evaluation logic. Contains notebooks and Python modules for running and analyzing experiments.
- **pilot/**: Contains agent logic, user simulation, and orchestration modules. Entry point for orchestrating agent interactions.
- **datasets/**: Raw and processed data, including mappings, reviews, and results. Subfolders are named by dataset or experiment.
- **recsys/**: LLM-based recommender system logic, including agent creation and user message formatting.
- **advisor.py**: Implements advisor agent logic.
- **utils/**: Dataset loading, configuration, and utility functions.

## Developer Workflows
- **Run experiments**: Use Jupyter notebooks in `fidelity/` (e.g., `main.ipynb`) or Python scripts in `pilot/`.
- **Data access**: Use `InstructRecDataset` and `ItemIndex` from `utils/` for loading and accessing dataset items.
- **Agent orchestration**: Use `Module` and `Orchestrator` classes (imported from `agents`) to define agent interactions.
- **Add new agents**: Implement in `pilot/` or `fidelity/`, register in orchestration logic.

## Patterns & Conventions
- **Path management**: Use `os.path` and `sys.path` to ensure modules and data are importable from notebooks/scripts.
- **Agent communication**: Agents interact via `talk_to()` and are orchestrated using `Module` and `Orchestrator`.
- **Data format**: Datasets are often in `.pkl`, `.csv`, `.json`, or `.tsv` formats. Use provided loaders.
- **Results**: Evaluation results are stored in `eval_results/` and `llm_results/` as JSON or JSONL files.

## Integration & Dependencies
- **LLM integration**: LLM-based recommenders are in `recsys/llms/`.
- **Neo4j**: Some scripts (e.g., `neo4j_client.py`) interact with Neo4j for knowledge graph experiments.
- **No monolithic entrypoint**: Workflows are modular; use notebooks or scripts as needed.

## Examples
- See `fidelity/main.ipynb` for a full experiment pipeline: data loading, agent setup, orchestration, and evaluation.
- See `pilot/arara_user.py` and `pilot/evaluation.py` for user simulation and evaluation logic.

## Tips
- When adding new datasets, follow the structure in `datasets/` and update loaders in `utils/`.
- For new agent types, subclass or follow patterns in `pilot/` and register in orchestration logic.
- Use relative imports and update `sys.path` as needed for cross-module access in notebooks.
