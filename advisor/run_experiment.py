#!/usr/bin/env python3
"""
CLI entry point for running the Advisor experiment on InstructRec Books.

Usage:
    python run_experiment.py --dataset ../datasets/instructrec/booksAll_recagent.pkl --n_users 5
    python run_experiment.py --help
"""

import argparse
import os
import sys
from pathlib import Path


def _load_env():
    """Load .env from the project root before any API clients are initialized."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.llm_clients import get_user_config, get_advisor_config, get_recsys_configs
from runner import ExperimentConfig, run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run the Advisor experiment on InstructRec Books"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="../datasets/instructrec/booksAll_recagent.pkl",
        help="Path to InstructRec .pkl dataset",
    )
    parser.add_argument("--n_users", type=int, default=5, help="Number of users to sample")
    parser.add_argument("--max_turns", type=int, default=6, help="Max turns per advisor session")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--advisor_model", type=str, default="qwen2.5:7b", help="LLM for the Advisor (Ollama)"
    )
    parser.add_argument(
        "--user_model", type=str, default="mistral:7b", help="LLM for the simulated user (Ollama)"
    )
    parser.add_argument(
        "--user_temp", type=float, default=0.7, help="Temperature for simulated user"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment_results", help="Output directory"
    )
    parser.add_argument(
        "--user_type",
        type=str,
        default="simulated",
        choices=["simulated", "confused"],
        help="Type of simulated user to run: 'simulated' (default) or 'confused'.",
    )
    parser.add_argument(
        "--bandit_model",
        type=str,
        default=None,
        help="Path to pre-trained bandit model (.npz). Starts from scratch if not provided.",
    )

    args = parser.parse_args()

    user_llm_config = get_user_config(args.user_model, args.user_temp)
    advisor_llm_config = get_advisor_config(args.advisor_model)
    recsys_llm_configs = get_recsys_configs()

    config = ExperimentConfig(
        dataset_path=args.dataset,
        n_users=args.n_users,
        seed=args.seed,
        max_turns=args.max_turns,
        advisor_model=args.advisor_model,
        user_model=args.user_model,
        user_temperature=args.user_temp,
        output_dir=args.output_dir,
        user_type=args.user_type,
        bandit_model_path=args.bandit_model,
        user_llm_config=user_llm_config,
        advisor_llm_config=advisor_llm_config,
        recsys_llm_configs=recsys_llm_configs,
    )

    print("=" * 60)
    print("ADVISOR EXPERIMENT")
    print("=" * 60)
    print(f"  Dataset:       {config.dataset_path}")
    print(f"  Users:         {config.n_users}")
    print(f"  Max turns:     {config.max_turns}")
    print(f"  Advisor model: {config.advisor_model}")
    print(f"  User model:    {config.user_model} (temp={config.user_temperature})")
    print(f"  User type:     {config.user_type}")
    print(f"  Policy:        LinUCB bandit")
    if config.bandit_model_path:
        print(f"  Bandit model:  {config.bandit_model_path}")
    print(f"  Seed:          {config.seed}")
    print(f"  Output:        {config.output_dir}")
    print("=" * 60)

    report = run_experiment(config)

    print("\n" + "=" * 60)
    print("ADVISOR RUN SUMMARY")
    print("=" * 60)

    n_adv = report.get("n_advisor", 0) if isinstance(report, dict) else 0
    print(f"  Advisor sessions: {n_adv}")

    print("=" * 60)


if __name__ == "__main__":
    main()
