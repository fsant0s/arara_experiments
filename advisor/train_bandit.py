#!/usr/bin/env python3
"""
Offline training script for the LinUCB bandit.

Reads a JSONL log of (context, action, reward) tuples collected
during bandit sessions and trains (or retrains) a LinUCBPolicy.

Usage:
    python train_bandit.py --logs experiment_results/bandit_logs.jsonl --output bandit_model.npz
    python train_bandit.py --help
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.policy import LinUCBPolicy, ACTIONS, CONTEXT_KEYS


def train(log_path: str, output_path: str, alpha: float = 1.0) -> None:
    policy = LinUCBPolicy(
        n_actions=len(ACTIONS),
        d=len(CONTEXT_KEYS),
        alpha=alpha,
    )

    action_value_to_type = {a.value: a for a in ACTIONS}
    n_tuples = 0

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ctx_dict = {k: v for k, v in zip(CONTEXT_KEYS, record["context"])}
            action = action_value_to_type.get(record["action"])
            if action is None:
                continue
            reward = float(record["reward"])
            policy.update(action, ctx_dict, reward)
            n_tuples += 1

    policy.save(output_path)
    print(f"Trained on {n_tuples} tuples. Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LinUCB bandit offline")
    parser.add_argument(
        "--logs", type=str, required=True,
        help="Path to JSONL log file with (context, action, reward) tuples",
    )
    parser.add_argument(
        "--output", type=str, default="bandit_model.npz",
        help="Output path for the trained model (.npz)",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="UCB exploration parameter",
    )
    args = parser.parse_args()
    train(args.logs, args.output, args.alpha)


if __name__ == "__main__":
    main()
