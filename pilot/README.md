# Pilot Experiment

First evaluation of the **ARARA** framework. This experiment implements multi-agent recommender system evaluation for books and movies, using Neo4j as a knowledge graph backend and LLM-based agents for orchestration.

## Role in ARARA

The pilot is the **first experimental version** of ARARA, validating the core framework components — agent orchestration, knowledge graph querying, tool usage, and evaluation metrics.

## Project Structure

```
pilot/
├── book.py                  # Book recommendation experiment entry point
├── movie.py                 # Movie recommendation experiment entry point
├── clients.py               # LLM client factory (OpenAI, Groq, Ollama)
├── neo4j_client.py          # Neo4j database connection and queries
├── arara_user.py            # Custom ARARA user agent
├── evaluation.py            # Metrics computation and reporting
├── user_history_book.py     # Book user history simulation
├── user_history_movie.py    # Movie user history simulation
├── modules/
│   ├── books/               # Book recommendation modules (explicit, implicit, misinformed)
│   └── movies/              # Movie recommendation modules (explicit, implicit, misinformed)
├── tools/
│   ├── books.py             # LLM-callable tools for book recommendations
│   └── movies.py            # LLM-callable tools for movie recommendations
├── test/                    # Test scripts for each scenario (explicit/implicit/misinformed)
└── main.ipynb               # Exploratory notebook
```

## Scenarios

Each domain (books, movies) is evaluated under three scenarios:
- **Explicit**: user clearly states preferences
- **Implicit**: preferences inferred from behaviour history
- **Misinformed**: user has incorrect beliefs about their preferences

## Running

**Prerequisites:** install dependencies (from `pilot/`). This installs **Arara** in editable mode from `../../arara` (sibling of `arara_experiments`). Ensure Neo4j is running at `neo4j://127.0.0.1:7687`.

```bash
uv venv && uv sync
source .venv/bin/activate
```

**Run book experiment:**

```bash
python book.py
```

**Run movie experiment:**

```bash
python movie.py
```

**Run individual scenario tests:**

```bash
python test/book_explicit.py
python test/book_implicit.py
python test/book_misinformed.py
```

## Dependencies

See `pyproject.toml`. Core: `openai`, `neo4j`, `numpy`, `scipy`, plus editable **Arara** (provides the `agents` package).
