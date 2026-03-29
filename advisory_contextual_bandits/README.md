# Advisory Experiment

Advisor agent for multi-CRS interaction. A **User-Side Advisor** mediates between a simulated user and two independent conversational recommender systems, selecting mediation actions via a contextual bandit (LinUCB) to reduce cognitive overload and guide the user to a better decision.

## Architecture

```
User (gpt-4o-mini)
    ↕ queries / reads responses
RS1 (llama3.1)    RS2 (mixtral)
    ↘              ↙
       Advisor (gpt-4o)
       └── LinUCB bandit
       └── Belief state (LLM-based)
```

The Advisor never recommends items directly. It observes both RS conversations, maintains a structured belief state, selects a mediation action (e.g., summarize, compare, simplify), and generates a guidance message for the user.

## Project Structure

```
advisory_contextual_bandits/
├── run_experiment.py       # Main CLI entry point
├── advisor.py              # AdvisorMediator class
├── runner.py               # Session orchestration (advisor + baseline)
├── simulated_user.py       # LLM-based user simulation
├── rs_agent.py             # Conversational recommender agent
├── evaluation.py           # Metrics computation
├── evaluate_experiment.py  # Offline evaluation CLI
├── train_bandit.py         # Offline LinUCB training
├── gt_inject.py            # Ground-truth injection into RS prompts
├── llm_utils.py            # LLM client wrappers
├── components/
│   ├── policy.py           # LinUCB + action space
│   ├── belief_state.py     # Belief state update via LLM
│   └── reward.py           # Reward function
├── config/
│   └── llm_clients.py      # Model configs and temperatures
├── docs/                   # Documentation, paper drafts (.tex), analysis .md
├── analysis_results/       # Figures and analysis scripts
├── notebooks/              # Exploratory notebooks
└── results_*/              # Experiment outputs (train/test sessions, bandit model)
```

## Running Experiments

**Prerequisites:** create the project environment (from `advisory_contextual_bandits/`):

```bash
uv venv && uv sync
source .venv/bin/activate
```

**Full train/test pipeline (200 train, 50 test users):**

```bash
python run_experiment.py \
    --dataset ../datasets/recassistbench/dataset/book/ExplicitQuery.json \
    --dataset_type recassistbench --domain book \
    --query_field direct_description_query \
    --n_users 250 --train_test_split 0.8 \
    --seed 42 --output_dir results_book
```

**Evaluate results:**

```bash
python evaluate_experiment.py \
    --advisor results_book/test_advisor \
    --baseline results_book/test_baseline \
    --dataset ../datasets/recassistbench/dataset/book/ExplicitQuery.json \
    --dataset_type recassistbench --domain book
```

## Models

| Role | Model | Temperature |
|------|-------|-------------|
| Simulated user | `gpt-4o-mini` | 0.7 |
| Advisor + belief extractor | `gpt-4o` | 0.7 |
| RS1 (analytical) | `llama3.1:latest` (Ollama) | 0.7 |
| RS2 (exploratory) | `mixtral:8x7b` (Ollama) | 0.7 |

## Dependencies

See `pyproject.toml`. Core: `openai`, `numpy`, `scipy`.
