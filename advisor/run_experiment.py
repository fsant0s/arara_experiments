#!/usr/bin/env python3
"""
CLI entry point for the multi-party Advisor experiment.

Usage (train/test split — 40 train / 10 test):
    python run_experiment.py \\
        --dataset ../datasets/recassistbench/dataset/book/ExplicitQuery.json \\
        --dataset_type recassistbench --domain book \\
        --query_field direct_description_query \\
        --n_users 50 --train_test_split 0.8 \\
        --seed 42 --output_dir results_book

Usage (single condition):
    python run_experiment.py --dataset ... --condition advisor --n_users 5

Train/test pipeline:
  1. Sample n_users, split into train / test
  2. Train: advisor on train users (bandit logs)
  3. Offline bandit training → bandit_model.npz
  4. Test: advisor (warm-started bandit) on test users
  5. Test: baseline (no advisor) on same test users
  6. Save split_manifest.json
"""
import argparse
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path


def _load_env():
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
    get_rs1_config,
    get_rs2_config,
    ADVISOR_MODEL,
    USER_MODEL,
    USER_TEMPERATURE,
    rs_models_display,
)
from runner import ExperimentConfig, run_experiment, load_dataset_entries
from train_bandit import train as train_bandit


def _print_banner(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _build_config(args, user_llm, advisor_llm, rs1_llm, rs2_llm) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_path=args.dataset,
        n_users=args.n_users,
        seed=args.seed,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
        bandit_model_path=args.bandit_model,
        user_llm_config=user_llm,
        advisor_llm_config=advisor_llm,
        rs1_llm_config=rs1_llm,
        rs2_llm_config=rs2_llm,
        dataset_type=args.dataset_type,
        domain=args.domain,
        query_field=args.query_field,
        inject_gt=not args.no_inject_gt,
    )


def _print_config(config: ExperimentConfig, condition: str, extra: str = ""):
    print(f"  Dataset:       {config.dataset_path}")
    print(f"  Dataset type:  {config.dataset_type}")
    print(f"  Domain:        {config.domain}")
    print(f"  Max turns:     {config.max_turns}")
    print(f"  Advisor model: {ADVISOR_MODEL}")
    print(f"  User model:    {USER_MODEL} (temp={USER_TEMPERATURE})")
    rs1_m, rs2_m = rs_models_display()
    print(f"  RS1 model:     {rs1_m}")
    print(f"  RS2 model:     {rs2_m}")
    print(f"  GT injection:  {'yes' if config.inject_gt else 'no'}")
    if config.bandit_model_path:
        print(f"  Bandit model:  {config.bandit_model_path}")
    print(f"  Seed:          {config.seed}")
    print(f"  Output:        {config.output_dir}")
    if extra:
        print(f"  {extra}")


def run_single_condition(args, config: ExperimentConfig):
    _print_banner(f"EXPERIMENT — {args.condition}")
    _print_config(config, args.condition, f"Users: {config.n_users}")
    print("=" * 60)
    report = run_experiment(config, condition=args.condition)
    _print_banner("DONE")
    print(f"  Sessions: {report.get(f'n_{args.condition}', 0)}")


def run_train_test_pipeline(args, config: ExperimentConfig, train_fraction: float):
    base_dir = config.output_dir
    os.makedirs(base_dir, exist_ok=True)

    manifest_path = os.path.join(base_dir, "split_manifest.json")
    resume = getattr(args, "resume", True)

    _print_banner("TRAIN/TEST SPLIT")

    if resume and os.path.exists(manifest_path):
        print(f"  [Resume] Loading manifest: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        train_indices = manifest["train_indices"]
        test_indices = manifest["test_indices"]
        n_train, n_test = len(train_indices), len(test_indices)
    else:
        _, n_total = load_dataset_entries(config)
        random.seed(config.seed)
        n_sample = min(config.n_users, n_total)
        all_indices = random.sample(range(n_total), n_sample)
        n_train = max(1, min(n_sample - 1, int(n_sample * train_fraction)))
        n_test = n_sample - n_train
        train_indices = all_indices[:n_train]
        test_indices = all_indices[n_train:]

        print(f"  Total: {n_sample}  Train: {n_train}  Test: {n_test}  Seed: {config.seed}")

        manifest = {
            "seed": config.seed, "n_total_sampled": n_sample,
            "train_fraction": train_fraction, "n_train": n_train, "n_test": n_test,
            "train_indices": train_indices, "test_indices": test_indices,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # Phase 1: Train advisor
    train_dir = os.path.join(base_dir, "train_advisor")
    train_config = deepcopy(config)
    train_config.output_dir = train_dir
    train_config.bandit_model_path = None

    _print_banner(f"PHASE 1/4 — TRAIN ADVISOR ({n_train} users)")
    _print_config(train_config, "advisor", f"Users (train): {n_train}")
    print("=" * 60)

    run_experiment(train_config, condition="advisor", indices=train_indices,
                   training_phase=True, resume=resume)

    # Phase 2: Offline bandit training
    _print_banner("PHASE 2/4 — TRAIN BANDIT")
    bandit_log_path = os.path.join(train_dir, train_config.bandit_log_path)
    bandit_model_path = os.path.join(base_dir, "bandit_model.npz")

    if os.path.exists(bandit_log_path):
        train_bandit(bandit_log_path, bandit_model_path,
                     session_logs_dir=train_dir, gt_hit_reward=1.0)
        print(f"  Bandit model → {bandit_model_path}")
    else:
        print(f"  WARNING: {bandit_log_path} not found")
        bandit_model_path = None

    # Phase 3: Test advisor (warm-started)
    test_adv_dir = os.path.join(base_dir, "test_advisor")
    test_adv_config = deepcopy(config)
    test_adv_config.output_dir = test_adv_dir
    test_adv_config.bandit_model_path = bandit_model_path

    _print_banner(f"PHASE 3/4 — TEST ADVISOR ({n_test} users)")
    _print_config(test_adv_config, "advisor", f"Users (test): {n_test}")
    print("=" * 60)

    run_experiment(test_adv_config, condition="advisor", indices=test_indices,
                   training_phase=False, resume=resume)

    # Phase 4: Test baseline
    test_bas_dir = os.path.join(base_dir, "test_baseline")
    test_bas_config = deepcopy(config)
    test_bas_config.output_dir = test_bas_dir
    test_bas_config.bandit_model_path = None

    _print_banner(f"PHASE 4/4 — TEST BASELINE ({n_test} users)")
    _print_config(test_bas_config, "no_advisor", f"Users (test): {n_test}")
    print("=" * 60)

    run_experiment(test_bas_config, condition="no_advisor", indices=test_indices, resume=resume)

    # Summary
    _print_banner("PIPELINE COMPLETE")
    print(f"  Train logs:    {train_dir}/")
    print(f"  Bandit model:  {bandit_model_path or '(not created)'}")
    print(f"  Test advisor:  {test_adv_dir}/")
    print(f"  Test baseline: {test_bas_dir}/")
    print(f"  Manifest:      {manifest_path}")
    print()
    print("  To evaluate:")
    print(f"    python evaluate_experiment.py \\")
    print(f"        --advisor {test_adv_dir} \\")
    print(f"        --baseline {test_bas_dir} \\")
    print(f"        --dataset {config.dataset_path} \\")
    print(f"        --dataset_type {config.dataset_type} --domain {config.domain}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Multi-party Advisor experiment")
    parser.add_argument("--dataset", type=str,
                        default="../datasets/recassistbench/dataset/book/ExplicitQuery.json")
    parser.add_argument("--dataset_type", type=str, default="recassistbench",
                        choices=["recassistbench"])
    parser.add_argument("--domain", type=str, default="book", choices=["book", "movie"])
    parser.add_argument("--query_field", type=str, default="direct_description_query")
    parser.add_argument("--n_users", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="experiment_results")
    parser.add_argument("--bandit_model", type=str, default=None)
    parser.add_argument("--condition", type=str, default="advisor",
                        choices=["advisor", "no_advisor"])
    parser.add_argument("--train_test_split", type=float, default=None, metavar="FRAC",
                        help="Enable train/test pipeline (e.g. 0.8)")
    parser.add_argument("--no_inject_gt", action="store_true",
                        help="Disable GT injection into RS prompts")
    parser.add_argument("--no_resume", action="store_true")

    args = parser.parse_args()
    args.resume = not args.no_resume

    user_llm = get_user_config()
    advisor_llm = get_advisor_config()
    rs1_llm = get_rs1_config()
    rs2_llm = get_rs2_config()

    config = _build_config(args, user_llm, advisor_llm, rs1_llm, rs2_llm)

    if args.train_test_split is not None:
        frac = args.train_test_split
        if not (0.0 < frac < 1.0):
            parser.error("--train_test_split must be between 0 and 1")
        run_train_test_pipeline(args, config, frac)
    else:
        run_single_condition(args, config)


if __name__ == "__main__":
    main()
