#!/usr/bin/env python3
"""
Evaluate advisor vs baseline for the multi-party experiment.

Usage:
    python evaluate_experiment.py \\
        --advisor results/test_advisor \\
        --baseline results/test_baseline \\
        --dataset ../datasets/recassistbench/dataset/book/ExplicitQuery.json \\
        --dataset_type recassistbench --domain book \\
        --output evaluation_report.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import (
    SessionLog,
    TurnLog,
    compute_session_metrics,
    compare_conditions,
)


def _load_session(path: str) -> SessionLog:
    with open(path) as f:
        d = json.load(f)
    turns = []
    for t in d.get("turns", []):
        um = (t.get("user_message") or t.get("user_response", "") or "").strip()
        action = (t.get("advisor_action") or t.get("action", "")).strip()
        turns.append(TurnLog(
            turn=t["turn"],
            action=action,
            advisor_message=t.get("advisor_message", ""),
            user_message=um,
            user_target=t.get("user_target", ""),
            rs_responses=t.get("rs_responses", {}),
            belief_state=t.get("belief_state", {}),
            items_mentioned=t.get("items_mentioned", []),
        ))
    return SessionLog(
        user_id=d.get("user_id", ""),
        condition=d.get("condition", ""),
        turns=turns,
        chosen_items=d.get("chosen_items", []),
        ground_truth=d.get("ground_truth", []),
        rs1_gt_items=d.get("rs1_gt_items", []),
        rs2_gt_items=d.get("rs2_gt_items", []),
        rs1_history=d.get("rs1_history", []),
        rs2_history=d.get("rs2_history", []),
        final_state=d.get("final_state"),
        initial_user_message=d.get("initial_user_message", ""),
        shared_relationships=d.get("shared_relationships", []),
        n_choices=d.get("n_choices", 1),
    )


def _load_all_sessions(directory: str) -> list[SessionLog]:
    sessions = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        fname = os.path.basename(path)
        if fname.endswith("_summary.json") or fname == "split_manifest.json":
            continue
        try:
            sessions.append(_load_session(path))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: skipping {fname}: {e}")
    return sessions


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-party advisor experiment")
    parser.add_argument("--advisor", required=True, help="Directory with advisor session JSONs")
    parser.add_argument("--baseline", required=True, help="Directory with baseline session JSONs")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset_type", default="recassistbench")
    parser.add_argument("--domain", default="book")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Loading advisor sessions from: {args.advisor}")
    adv_logs = _load_all_sessions(args.advisor)
    print(f"  Loaded {len(adv_logs)} advisor sessions")

    print(f"Loading baseline sessions from: {args.baseline}")
    bas_logs = _load_all_sessions(args.baseline)
    print(f"  Loaded {len(bas_logs)} baseline sessions")

    if not adv_logs and not bas_logs:
        print("No sessions found. Nothing to evaluate.")
        return

    # Per-session metrics
    print("\n--- Per-session metrics (advisor) ---")
    for log in adv_logs:
        m = compute_session_metrics(log)
        chosen = ", ".join(log.chosen_items) or "none"
        gt = ", ".join(log.ground_truth[:3]) or "none"
        print(f"  {log.user_id}: recall={m['recall']:.2f} prec={m['precision']:.2f} "
              f"turns={m['n_turns']} cross_rs={m['cross_rs_hits']} chosen=[{chosen}] gt=[{gt}]")

    # Aggregate comparison
    if adv_logs and bas_logs:
        print("\n--- Aggregate comparison ---")
        comp = compare_conditions(adv_logs, bas_logs)

        header = f"{'Metric':<35} {'Advisor':>10} {'Baseline':>10} {'Delta':>10} {'p-value':>10}"
        print(header)
        print("-" * len(header))

        display_keys = [
            ("Recall@chosen", "recall"),
            ("Precision@chosen", "precision"),
            ("N turns", "n_turns"),
            ("Cross-RS discovery", "cross_rs_rate"),
            ("Advisor follow rate", "follow_rate"),
            ("Overload risk (final)", "overload_risk_final"),
            ("Preference specificity", "preference_specificity"),
        ]

        for label, key in display_keys:
            if key not in comp:
                continue
            c = comp[key]
            p_str = f"{c['p_value']:.4f}" if "p_value" in c else "n/a"
            print(f"  {label:<33} {c['advisor_mean']:>10.3f} {c['baseline_mean']:>10.3f} "
                  f"{c['delta']:>+10.3f} {p_str:>10}")

    # Save report
    report = {
        "advisor_sessions": len(adv_logs),
        "baseline_sessions": len(bas_logs),
        "per_session_advisor": [
            {"user_id": l.user_id, **compute_session_metrics(l)} for l in adv_logs
        ],
        "per_session_baseline": [
            {"user_id": l.user_id, **compute_session_metrics(l)} for l in bas_logs
        ],
    }
    if adv_logs and bas_logs:
        report["comparison"] = compare_conditions(adv_logs, bas_logs)

    output_path = args.output or os.path.join(
        os.path.dirname(args.advisor), "evaluation_report.json"
    )
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
