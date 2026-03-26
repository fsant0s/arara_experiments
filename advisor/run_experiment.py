#!/usr/bin/env python3
"""
CLI entry point for running the Advisor experiment.

Supports InstructRec (.pkl) and RecAssistBench (.json) datasets in book/movie domains.

Usage (single condition — InstructRec):
    python run_experiment.py --dataset ../datasets/instructrec/booksAll_recagent.pkl --n_users 5

Usage (single condition — RecAssistBench, books):
    python run_experiment.py \\
        --dataset ../datasets/recassistbench/dataset/book/ExplicitQuery.json \\
        --dataset_type recassistbench --domain book \\
        --query_field direct_description_query --n_users 20

Usage (train/test split — automatic pipeline):
    python run_experiment.py \\
        --dataset ../datasets/recassistbench/dataset/book/ExplicitQuery.json \\
        --dataset_type recassistbench --domain book \\
        --query_field direct_description_query \\
        --n_users 100 --train_test_split 0.8 \\
        --seed 42 --output_dir results_book_split

    This will:
      1. Sample 100 users, split 80 train / 20 test
      2. Run advisor on the 80 train users (collecting bandit logs)
      3. Train the bandit offline on ALL train logs → bandit_model.npz
      4. Run advisor (warm-started with bandit_model.npz) on 20 test users
      5. Run baseline (no_advisor) on the SAME 20 test users, reusing each user's
         recsys_outputs from step 4 (identical pool for paired evaluation)
      6. Save a split_manifest.json with train/test indices
"""

import argparse
import json
import os
import random
import sys
from copy import deepcopy
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

from config.llm_clients import (
    get_user_config,
    get_advisor_config,
    get_recsys_configs,
    ADVISOR_MODEL,
    USER_MODEL,
    USER_TEMPERATURE,
)
from runner import ExperimentConfig, run_experiment, load_dataset_entries
from train_bandit import train as train_bandit


def _print_banner(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _build_config(args, user_llm, advisor_llm, recsys_llm) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_path=args.dataset,
        n_users=args.n_users,
        seed=args.seed,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
        user_type=args.user_type,
        bandit_model_path=args.bandit_model,
        user_llm_config=user_llm,
        advisor_llm_config=advisor_llm,
        recsys_llm_configs=recsys_llm,
        dataset_type=args.dataset_type,
        domain=args.domain,
        query_field=args.query_field,
        inject_random_gt_in_pool=args.inject_random_gt_in_pool,
        no_advisor_recsys_cache_dir=getattr(args, "no_advisor_recsys_cache_dir", None),
    )


def _print_config(config: ExperimentConfig, condition: str, extra: str = ""):
    print(f"  Dataset:       {config.dataset_path}")
    print(f"  Dataset type:  {config.dataset_type}")
    print(f"  Domain:        {config.domain}")
    if config.dataset_type == "recassistbench":
        print(f"  Query field:   {config.query_field}")
    print(f"  Max turns:     {config.max_turns}")
    print(f"  Advisor model: {ADVISOR_MODEL} (OpenAI)")
    print(f"  User model:    {USER_MODEL} (temp={USER_TEMPERATURE})")
    print(f"  User type:     {config.user_type}")
    print(f"  Policy:        LinUCB bandit")
    if config.bandit_model_path:
        print(f"  Bandit model:  {config.bandit_model_path}")
    print(f"  Seed:          {config.seed}")
    if getattr(config, "inject_random_gt_in_pool", False):
        print(f"  GT in pool:    each GT → random agent + random slot (inject_random_gt_in_pool)")
    if getattr(config, "no_advisor_recsys_cache_dir", None):
        print(f"  Baseline cache: reuse recsys from → {config.no_advisor_recsys_cache_dir}")
    print(f"  Output:        {config.output_dir}")
    if extra:
        print(f"  {extra}")


def run_single_condition(args, config: ExperimentConfig):
    """Original flow: run one condition (advisor or no_advisor)."""
    _print_banner(f"EXPERIMENT — condition: {args.condition}")
    _print_config(config, args.condition, f"Users: {config.n_users}")
    print("=" * 60)

    report = run_experiment(config, condition=args.condition)

    _print_banner("RUN SUMMARY")
    key = f"n_{args.condition}"
    n_sessions = report.get(key, 0) if isinstance(report, dict) else 0
    print(f"  {args.condition} sessions: {n_sessions}")
    print("=" * 60)


def run_train_test_pipeline(args, config: ExperimentConfig, train_fraction: float):
    """
    Full train/test pipeline:
      1) Sample n_users indices, split into train (train_fraction) and test
      2) Train phase: advisor on train indices → bandit_logs.jsonl
      3) Offline bandit training on ALL train logs → bandit_model.npz
      4) Test phase: advisor (with bandit_model.npz) on test indices
      5) Test phase: baseline (no_advisor) on same test indices
      6) Save split_manifest.json

    If split_manifest.json already exists, reuses its indices (resume mode).
    """
    base_dir = config.output_dir
    os.makedirs(base_dir, exist_ok=True)

    manifest_path = os.path.join(base_dir, "split_manifest.json")
    resume = getattr(args, "resume", True)

    # ── 1. Sample and split (or load existing manifest) ─────
    _print_banner("TRAIN/TEST SPLIT")

    if resume and os.path.exists(manifest_path):
        print(f"  [Resume] Loading existing manifest: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        train_indices = manifest["train_indices"]
        test_indices = manifest["test_indices"]
        n_train = len(train_indices)
        n_test = len(test_indices)
        n_sample = n_train + n_test
        print(f"  [Resume] Train: {n_train}, Test: {n_test}")
    else:
        _, n_total = load_dataset_entries(config)
        random.seed(config.seed)
        n_sample = min(config.n_users, n_total)
        all_indices = random.sample(range(n_total), n_sample)

        n_train = max(1, min(n_sample - 1, int(n_sample * train_fraction)))
        n_test = n_sample - n_train
        train_indices = all_indices[:n_train]
        test_indices = all_indices[n_train:]

        print(f"  Total sampled:  {n_sample}")
        print(f"  Train:          {n_train}  ({train_fraction*100:.0f}%)")
        print(f"  Test:           {n_test}  ({(1-train_fraction)*100:.0f}%)")
        print(f"  Seed:           {config.seed}")

        manifest = {
            "seed": config.seed,
            "n_total_sampled": n_sample,
            "train_fraction": train_fraction,
            "n_train": n_train,
            "n_test": n_test,
            "train_indices": train_indices,
            "test_indices": test_indices,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Manifest:       {manifest_path}")

    # ── 2. Train phase: advisor on train set ─────────────────
    train_dir = os.path.join(base_dir, "train_advisor")
    train_config = deepcopy(config)
    train_config.output_dir = train_dir
    train_config.bandit_model_path = None

    _print_banner(f"PHASE 1/4 — TRAIN ADVISOR ({n_train} users)")
    _print_config(train_config, "advisor", f"Users (train): {n_train}")
    print("=" * 60)

    run_experiment(
        train_config,
        condition="advisor",
        indices=train_indices,
        training_phase=True,
        resume=resume,
    )

    # ── 3. Offline bandit training ───────────────────────────
    _print_banner("PHASE 2/4 — TRAIN BANDIT (offline on all train logs, with GT boost)")
    bandit_log_path = os.path.join(train_dir, train_config.bandit_log_path)
    bandit_model_path = os.path.join(base_dir, "bandit_model.npz")

    if os.path.exists(bandit_log_path):
        train_bandit(
            bandit_log_path, bandit_model_path,
            session_logs_dir=train_dir,
            gt_hit_reward=1.0,
        )
        print(f"  Bandit model saved to {bandit_model_path}")
    else:
        print(f"  WARNING: {bandit_log_path} not found — skipping bandit training.")
        bandit_model_path = None

    # ── 4. Test phase: advisor with trained bandit ───────────
    test_adv_dir = os.path.join(base_dir, "test_advisor")
    test_adv_config = deepcopy(config)
    test_adv_config.output_dir = test_adv_dir
    test_adv_config.bandit_model_path = bandit_model_path

    _print_banner(f"PHASE 3/4 — TEST ADVISOR ({n_test} users, warm-started bandit)")
    _print_config(test_adv_config, "advisor", f"Users (test): {n_test}")
    print("=" * 60)

    run_experiment(
        test_adv_config,
        condition="advisor",
        indices=test_indices,
        training_phase=False,
        resume=resume,
    )

    # ── 5. Test phase: baseline on same test indices ─────────
    test_bas_dir = os.path.join(base_dir, "test_baseline")
    test_bas_config = deepcopy(config)
    test_bas_config.output_dir = test_bas_dir
    test_bas_config.bandit_model_path = None
    test_bas_config.no_advisor_recsys_cache_dir = test_adv_dir

    _print_banner(f"PHASE 4/4 — TEST BASELINE ({n_test} users, no advisor, same recsys as test advisor)")
    _print_config(test_bas_config, "no_advisor", f"Users (test): {n_test}")
    print("=" * 60)

    run_experiment(
        test_bas_config,
        condition="no_advisor",
        indices=test_indices,
        resume=resume,
    )

    # ── 6. Summary ───────────────────────────────────────────
    _print_banner("TRAIN/TEST PIPELINE COMPLETE")
    print(f"  Train advisor logs:   {train_dir}/")
    print(f"  Bandit model:         {bandit_model_path or '(not created)'}")
    print(f"  Test advisor logs:    {test_adv_dir}/")
    print(f"  Test baseline logs:   {test_bas_dir}/")
    print(f"  Split manifest:       {manifest_path}")
    print()
    print("  To evaluate the test results:")
    print(f"    python evaluate_experiment.py \\")
    print(f"        --advisor {test_adv_dir} \\")
    print(f"        --baseline {test_bas_dir} \\")
    print(f"        --dataset {config.dataset_path} \\")
    print(f"        --dataset_type {config.dataset_type} --domain {config.domain}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run the Advisor experiment (InstructRec or RecAssistBench)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="../datasets/instructrec/booksAll_recagent.pkl",
        help="Path to dataset (.pkl for InstructRec, .json for RecAssistBench)",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="instructrec",
        choices=["instructrec", "recassistbench"],
        help="Dataset format: 'instructrec' (.pkl) or 'recassistbench' (.json)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="book",
        choices=["book", "movie"],
        help="Recommendation domain: 'book' or 'movie'",
    )
    parser.add_argument(
        "--query_field",
        type=str,
        default="direct_description_query",
        help="RecAssistBench JSON field to use as the user query "
             "(e.g. direct_description_query, situational_description_query)",
    )
    parser.add_argument("--n_users", type=int, default=5, help="Number of users to sample")
    parser.add_argument("--max_turns", type=int, default=6, help="Max turns per advisor session")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="experiment_results", help="Output directory"
    )
    parser.add_argument(
        "--user_type",
        type=str,
        default="simulated",
        choices=["simulated"],
        help="Simulated user type (currently only 'simulated' is supported).",
    )
    parser.add_argument(
        "--bandit_model",
        type=str,
        default=None,
        help="Path to pre-trained bandit model (.npz). Starts from scratch if not provided.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="advisor",
        choices=["advisor", "no_advisor"],
        help="Condition to run (ignored when --train_test_split is set).",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=None,
        metavar="FRACTION",
        help="Enable train/test pipeline. Fraction of n_users for training (e.g. 0.8). "
             "Remaining users are test set for both advisor and baseline.",
    )
    parser.add_argument(
        "--inject_random_gt_in_pool",
        action="store_true",
        help="After recsys, inject every GT title: each into a random recsys agent at a random "
             "top-k slot (distinct slots when possible; reproducible via --seed + user id).",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start from scratch even if partial results exist (default: resume).",
    )
    parser.add_argument(
        "--no_advisor_recsys_cache_dir",
        type=str,
        default=None,
        metavar="DIR",
        help="For condition=no_advisor only: directory with prior user_*_advisor.json files; "
             "reuse each file's recsys_outputs instead of calling recsys again (paired pool).",
    )

    args = parser.parse_args()
    args.resume = not args.no_resume

    user_llm_config = get_user_config()
    advisor_llm_config = get_advisor_config()
    recsys_llm_configs = get_recsys_configs()

    config = _build_config(args, user_llm_config, advisor_llm_config, recsys_llm_configs)

    if args.train_test_split is not None:
        frac = args.train_test_split
        if not (0.0 < frac < 1.0):
            parser.error("--train_test_split must be between 0 and 1 (exclusive)")
        run_train_test_pipeline(args, config, frac)
    else:
        run_single_condition(args, config)


if __name__ == "__main__":
    main()
