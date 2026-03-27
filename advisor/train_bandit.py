#!/usr/bin/env python3
"""
Offline training script for the LinUCB bandit.

Reads a JSONL log of (context, action, reward) tuples collected
during advisor sessions and trains (or retrains) a LinUCBPolicy.

When --session_logs is provided (directory with session JSON files),
the script applies GT-based reward boosting: if the user's chosen items
match any ground-truth title, the final turn's reward is boosted.

Usage:
    python train_bandit.py --logs results/train_advisor/bandit_logs.jsonl --output bandit_model.npz
    python train_bandit.py --logs results/train_advisor/bandit_logs.jsonl \\
        --session_logs results/train_advisor/ --output bandit_model.npz
"""

import argparse
import glob
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.policy import LinUCBPolicy, ACTIONS, CONTEXT_KEYS


def _normalize_title(title: str) -> str:
    import html
    import re
    s = html.unescape(str(title or "")).strip().lower()
    return re.sub(r"\s+", " ", s)


def _load_session_gt_and_chosen(session_logs_dir: str) -> dict:
    """Load ground truth and chosen items from session JSON files.

    Returns {session_id: {"gt": [titles], "chosen": [titles]}}.
    """
    result = {}
    pattern = os.path.join(session_logs_dir, "user_*_advisor.json")
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                data = json.load(f)
            session_id = data.get("user_id", "")
            gt = data.get("ground_truth", [])
            chosen_items = data.get("chosen_items", [])
            if not chosen_items:
                single = data.get("chosen_item", "")
                chosen_items = [single] if single and single != "none" else []
            if session_id:
                result[session_id] = {
                    "gt": [_normalize_title(t) for t in gt if t],
                    "chosen": [_normalize_title(t) for t in chosen_items if t],
                }
        except (json.JSONDecodeError, OSError):
            continue
    return result


def _gt_hit(chosen_list: list, gt_list: list) -> bool:
    """Check if any chosen item matches any GT title (fuzzy substring)."""
    for chosen in chosen_list:
        if not chosen or chosen == "none":
            continue
        for gt in gt_list:
            if not gt:
                continue
            if chosen in gt or gt in chosen:
                return True
    return False


def train(
    log_path: str,
    output_path: str,
    alpha: float = 1.0,
    session_logs_dir: str = None,
    gt_hit_reward: float = 1.0,
) -> None:
    policy = LinUCBPolicy(
        n_actions=len(ACTIONS),
        d=len(CONTEXT_KEYS),
        alpha=alpha,
    )

    action_value_to_type = {a.value: a for a in ACTIONS}
    n_tuples = 0
    n_boosted = 0

    session_info = {}
    if session_logs_dir and os.path.isdir(session_logs_dir):
        session_info = _load_session_gt_and_chosen(session_logs_dir)
        print(f"Loaded GT/chosen info for {len(session_info)} sessions.")

    # Track last turn per session for GT-hit boost
    session_last_turn: dict = {}

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            sid = record.get("session_id", "")
            turn = record.get("turn", 0)
            if sid not in session_last_turn or turn > session_last_turn[sid]:
                session_last_turn[sid] = turn

    for record in records:
        ctx_raw = record.get("context", {})
        if isinstance(ctx_raw, dict):
            ctx_dict = {k: float(ctx_raw.get(k, 0.0)) for k in CONTEXT_KEYS}
        elif isinstance(ctx_raw, (list, tuple)):
            ctx_dict = {k: float(v) for k, v in zip(CONTEXT_KEYS, ctx_raw)}
        else:
            continue

        action = action_value_to_type.get(record.get("action"))
        if action is None:
            continue
        reward = float(record.get("reward", 0.0))

        session_id = record.get("session_id", "")
        turn = record.get("turn", 0)

        if (
            session_id in session_info
            and session_id in session_last_turn
            and turn == session_last_turn[session_id]
        ):
            info = session_info[session_id]
            if _gt_hit(info["chosen"], info["gt"]):
                reward = max(reward, gt_hit_reward)
                n_boosted += 1

        policy.update(action, ctx_dict, reward)
        n_tuples += 1

    policy.save(output_path)
    print(f"Trained on {n_tuples} tuples ({n_boosted} GT-hit boosted). Model → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LinUCB bandit offline")
    parser.add_argument("--logs", type=str, required=True,
                        help="Path to JSONL log file")
    parser.add_argument("--output", type=str, default="bandit_model.npz",
                        help="Output path for trained model (.npz)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="UCB exploration parameter")
    parser.add_argument("--session_logs", type=str, default=None,
                        help="Directory with session JSONs (for GT reward boost)")
    parser.add_argument("--gt_hit_reward", type=float, default=1.0,
                        help="Reward boost when chosen matches GT")
    args = parser.parse_args()
    train(args.logs, args.output, args.alpha,
          session_logs_dir=args.session_logs, gt_hit_reward=args.gt_hit_reward)


if __name__ == "__main__":
    main()
