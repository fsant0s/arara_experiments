#!/usr/bin/env python3
"""
Evaluate advisor vs baseline experiments.

With ``--dataset``: behavioral metrics (articulation, bias resistance, turns,
recall/precision vs GT pool) via ``compare_conditions``, **plus** whether the
user's **chosen** title matches the ground-truth list (chosen ∈ GT + McNemar).

Usage (sem ground truth):
    python evaluate_experiment.py \
        --advisor experiment_results_v2_eval \
        --baseline experiment_results_v2_baseline

Uso (com ground truth do InstructRec):
    python evaluate_experiment.py \
        --advisor experiment_results_v2_eval \
        --baseline experiment_results_v2_baseline \
        --dataset ../datasets/instructrec/booksAll_recagent.pkl \
        --dataset_type instructrec \
        --item_index ../datasets/instructrec/combined_books_asin_mapping.csv
        --output evaluation_report.json

Uso (com ground truth do RecAssistBench — books):
    python evaluate_experiment.py \
        --advisor results_recassist_book_adv \
        --baseline results_recassist_book_bas \
        --dataset ../datasets/recassistbench/dataset/book/ExplicitQuery.json \
        --dataset_type recassistbench --domain book \
        --output evaluation_report.json

Uso (com ground truth do RecAssistBench — movies):
    python evaluate_experiment.py \
        --advisor results_recassist_movie_adv \
        --baseline results_recassist_movie_bas \
        --dataset ../datasets/recassistbench/dataset/movie/ExplicitQuery.json \
        --dataset_type recassistbench --domain movie \
        --output evaluation_report.json
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import (
    SessionLog,
    TurnLog,
    classify_chosen_vs_gt,
    classify_chosen_set_vs_gt,
    compare_conditions,
    compute_articulation,
    compute_bias_resistance,
    compute_chosen_recall_precision,
    compute_ground_truth,
    compute_relationship_match,
    normalize_title_for_gt_match,
)


def _load_session(path: str) -> Optional[SessionLog]:
    with open(path) as f:
        d = json.load(f)
    turns = []
    for t in d.get("turns", []):
        turns.append(TurnLog(
            turn=t["turn"],
            action=t["action"],
            advisor_message=t.get("advisor_message", ""),
            user_response=t.get("user_response", ""),
            belief_state=t.get("belief_state", {}),
            items_considered_for_internal_turn_state=t.get(
                "items_considered_for_internal_turn_state", []
            ),
        ))
    chosen_items = d.get("chosen_items", [])
    chosen_item = d.get("chosen_item", "none")
    if not chosen_items and chosen_item and chosen_item != "none":
        chosen_items = [chosen_item]

    return SessionLog(
        user_id=d["user_id"],
        condition=d["condition"],
        turns=turns,
        chosen_item=chosen_item,
        chosen_items=chosen_items,
        final_state=d.get("final_state"),
        recsys_outputs=d.get("recsys_outputs"),
        initial_user_message=d.get("initial_user_message", ""),
        ground_truth=d.get("ground_truth", []),
        shared_relationships=d.get("shared_relationships", []),
    )


def _load_dir(directory: str, condition: str) -> List[SessionLog]:
    pattern = os.path.join(directory, f"user_*_{condition}.json")
    logs = []
    for path in sorted(glob.glob(pattern)):
        log = _load_session(path)
        if log:
            logs.append(log)
    return logs


def _build_ground_truths_instructrec(
    dataset_path: str,
    item_index_path: str,
) -> Dict[str, List[str]]:
    """
    Build {user_id: [ground_truth_title]} from InstructRec .pkl,
    using the first item of each user's ranked_lists resolved via ItemIndex.
    """
    from load_dataset import InstructRecDataset, ItemIndex

    print(f"  Loading InstructRec dataset: {dataset_path}")
    ds = InstructRecDataset(dataset_path)
    item_idx = ItemIndex(item_index_path)

    ground_truths: Dict[str, List[str]] = {}
    import numpy as np

    for i in range(len(ds)):
        entry = ds[i]
        ranked = entry.get("ranked_lists")
        if ranked is None:
            continue
        if isinstance(ranked, np.ndarray):
            ranked = ranked.tolist()
        if not isinstance(ranked, (list, tuple)) or len(ranked) == 0:
            continue
        first_idx = ranked[0]
        try:
            item = item_idx.get_item_by_index(int(first_idx))
            title = item.get("title", "")
            if title:
                ground_truths[f"user_{i}"] = [title]
        except (KeyError, TypeError, ValueError):
            pass

    print(f"  Ground truth loaded for {len(ground_truths)} users.")
    return ground_truths


def _build_ground_truths_recassistbench(
    dataset_path: str,
    domain: str = "book",
) -> tuple[Dict[str, List[str]], Dict[str, List]]:
    """
    Build {user_id: [gt_title, ...]} and {user_id: sharedRelationships}
    from a RecAssistBench JSON.

    GT comes from ``bookSubset`` (books) or ``movieSubset`` (movies) —
    multi-title ground truth per entry.
    user_id is ``user_{data_idx}`` to match the runner convention.
    """
    print(f"  Loading RecAssistBench ({domain}): {dataset_path}")
    with open(dataset_path) as f:
        entries = json.load(f)

    subset_key = "bookSubset" if domain == "book" else "movieSubset"
    ground_truths: Dict[str, List[str]] = {}
    shared_rels_map: Dict[str, List] = {}
    for entry in entries:
        data_idx = entry.get("data_idx")
        if data_idx is None:
            continue
        user_id = f"user_{data_idx}"
        titles = entry.get(subset_key, [])
        if titles:
            ground_truths[user_id] = list(titles)
        rels = entry.get("sharedRelationships", [])
        if rels:
            shared_rels_map[user_id] = rels

    print(f"  Ground truth loaded for {len(ground_truths)} users "
          f"({subset_key}, avg {sum(len(v) for v in ground_truths.values()) / max(len(ground_truths), 1):.1f} GT items each).")
    print(f"  Shared relationships loaded for {len(shared_rels_map)} users.")
    return ground_truths, shared_rels_map


def _print_section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _ground_truth_hit_rate(
    logs: List[SessionLog],
    ground_truths: Dict[str, List[str]],
) -> float:
    """
    Fraction of sessions where ``chosen_item`` matches any GT title (single-item legacy).
    """
    hits = 0
    total = 0
    for log in logs:
        gt_list = ground_truths.get(log.user_id, [])
        if not gt_list:
            continue
        total += 1
        if classify_chosen_vs_gt(log.chosen_item or "", gt_list) is not None:
            hits += 1
    return hits / total if total > 0 else 0.0


def _ground_truth_set_hit_rate(
    logs: List[SessionLog],
    ground_truths: Dict[str, List[str]],
) -> float:
    """
    Fraction of sessions where ANY item in ``chosen_items`` matches ANY GT title.

    This is the multi-choice hit metric: the user picks ``bookCount`` items,
    and we check if the intersection with bookSubset is non-empty.
    """
    hits = 0
    total = 0
    for log in logs:
        gt_list = ground_truths.get(log.user_id, [])
        if not gt_list:
            continue
        total += 1
        chosen_set = log.chosen_items or ([log.chosen_item] if log.chosen_item and log.chosen_item != "none" else [])
        if classify_chosen_set_vs_gt(chosen_set, gt_list):
            hits += 1
    return hits / total if total > 0 else 0.0


def _gt_chosen_hit(log: SessionLog, ground_truths: Dict[str, List[str]]) -> bool:
    """True if chosen item matches any GT (strict or fuzzy) — single-item legacy."""
    gt_list = ground_truths.get(log.user_id, [])
    if not gt_list:
        return False
    return classify_chosen_vs_gt(log.chosen_item or "", gt_list) is not None


def _gt_chosen_set_hit(log: SessionLog, ground_truths: Dict[str, List[str]]) -> bool:
    """True if ANY item in chosen_items matches ANY GT title."""
    gt_list = ground_truths.get(log.user_id, [])
    if not gt_list:
        return False
    chosen_set = log.chosen_items or ([log.chosen_item] if log.chosen_item and log.chosen_item != "none" else [])
    return classify_chosen_set_vs_gt(chosen_set, gt_list)


def _mcnemar_binary_pvalue(table) -> Optional[float]:
    """
    Exact McNemar test on 2x2 table (paired binary outcomes):
        [[both False, adv F bas T],
         [adv T bas F, both True]]
    Rows = advisor hit?, cols = baseline hit?.
    Discordant pairs: b = t[0,1], c = t[1,0]. Under H0, b ~ Binomial(b+c, 0.5).

    Uses ``scipy.stats.binomtest`` (SciPy >= 1.7). SciPy 1.17+ removed ``stats.mcnemar``,
    so we do not import it.
    """
    try:
        import numpy as np
        from scipy.stats import binomtest

        t = np.asarray(table, dtype=int)
        b = int(t[0, 1])
        c = int(t[1, 0])
        n = b + c
        if n == 0:
            return None
        k = min(b, c)
        return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        return None


def _paired_binary_tables(
    adv_logs: List[SessionLog],
    bas_logs: List[SessionLog],
    ground_truths: Dict[str, List[str]],
    hit_fn: Callable[[SessionLog, Dict[str, List[str]]], bool],
):
    """
    Build 2x2 for McNemar: row index = advisor hit, col index = baseline hit.
    Only users present in both logs with non-empty GT.
    """
    adv_by = {l.user_id: l for l in adv_logs}
    bas_by = {l.user_id: l for l in bas_logs}
    common = set(adv_by) & set(bas_by)
    uids = sorted(u for u in common if ground_truths.get(u))
    import numpy as np

    table = np.zeros((2, 2), dtype=int)
    for uid in uids:
        a = bool(hit_fn(adv_by[uid], ground_truths))
        b = bool(hit_fn(bas_by[uid], ground_truths))
        table[int(a), int(b)] += 1
    return table, uids


def _recsys_union_fingerprint(log: SessionLog) -> frozenset:
    """Normalized title set from recsys_outputs (order-free, for equality checks)."""
    return frozenset(_recsys_titles(log))


def warn_if_recsys_pools_differ(
    adv_logs: List[SessionLog],
    bas_logs: List[SessionLog],
) -> int:
    """
    Warn when advisor and baseline sessions for the same user_id used different
    raw recsys outputs. The train/test pipeline runs recsys twice (test advisor,
    then test baseline), so LLM sampling yields different pools — paired McNemar /
    delta on \"Recsys GT coverage\" are then **not** causal comparisons of methods.
    Returns the number of user_ids with mismatched recsys unions.
    """
    by_user = {l.user_id: l for l in bas_logs}
    mismatches = []
    for a in adv_logs:
        b = by_user.get(a.user_id)
        if not b:
            continue
        fa = _recsys_union_fingerprint(a)
        fb = _recsys_union_fingerprint(b)
        if fa != fb:
            mismatches.append(a.user_id)

    if mismatches:
        print()
        print("  ⚠  RECSYS POOL MISMATCH (stochastic recsys across runs)")
        print("     Advisor vs baseline JSONs differ for "
              f"{len(mismatches)}/{len(adv_logs)} paired user(s): "
              f"{', '.join(mismatches[:5])}"
              + (" …" if len(mismatches) > 5 else ""))
        print("     → Paired comparisons are not on the same raw recsys draw.")
        print("       Fix: re-run baseline with train_test_split (baseline auto-uses advisor")
        print("       recsys cache), or: run_experiment.py --condition no_advisor")
        print("       --no_advisor_recsys_cache_dir <test_advisor_dir> …")
        print("       Also helps: lower recsys temperature in config/llm_clients.py.")
        print()

    return len(mismatches)


def _recsys_titles(log: SessionLog) -> List[str]:
    """
    Collect all titles recommended by the recsys (union across all LLMs),
    normalised, from SessionLog.recsys_outputs.

    recsys_outputs format:
        { "llm_name": [{"title": ..., "explanation": ...}, ...], ... }
    """
    out: List[str] = []
    seen: set = set()
    for items in (log.recsys_outputs or {}).values():
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            raw = it.get("title") or it.get("book_title") or it.get("movie_title") or ""
            t = normalize_title_for_gt_match(raw)
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _descriptive_stats(
    logs: List[SessionLog],
    label: str,
    ground_truths: Optional[Dict[str, List[str]]] = None,
):
    """Sessions, actions, articulation, bias resistance; optional mean recall/precision vs GT pool."""
    n = len(logs)
    if n == 0:
        print(f"  {label}: no sessions found.")
        return

    decided = [l for l in logs if l.chosen_item and l.chosen_item != "none"]
    turns = [len(l.turns) for l in logs]

    action_counts: Counter = Counter()
    for l in logs:
        for t in l.turns:
            action_counts[t.action] += 1

    print(f"  Sessions:        {n}")
    print(f"  Decided:         {len(decided)} / {n}  ({100 * len(decided) / n:.0f}%)")
    print(f"  Turns (mean):    {sum(turns) / n:.1f}")
    print(f"  Turns (min/max): {min(turns)} / {max(turns)}")
    print(
        f"  Chosen items:    {', '.join(l.chosen_item for l in decided[:5])}"
        + (" ..." if len(decided) > 5 else "")
    )

    if any(a.startswith("a") for a in action_counts):
        print("  Action distribution:")
        for act in sorted(action_counts):
            print(f"    {act}: {action_counts[act]}")

    artic = [compute_articulation(l) for l in logs]
    biases = [compute_bias_resistance(l) for l in logs]
    anch = [b["anchoring"] for b in biases]
    conf = [b["confirmation"] for b in biases]
    overl = [b["overload"] for b in biases]

    print(f"  Articulation Δσ (mean):     {sum(artic) / n:.3f}")
    print(f"  Anchoring resist (mean):    {sum(anch) / n:.3f}")
    print(f"  Confirmation resist (mean): {sum(conf) / n:.3f}")
    print(f"  Overload resist (mean):     {sum(overl) / n:.3f}")

    if ground_truths:
        pool_recalls: List[float] = []
        pool_precs: List[float] = []
        chosen_recalls: List[float] = []
        chosen_precs: List[float] = []
        for l in logs:
            gt = ground_truths.get(l.user_id, [])
            if not gt:
                continue
            m = compute_ground_truth(l, gt)
            pool_recalls.append(m["recall_at_k"])
            pool_precs.append(m["precision_at_k"])
            ch = compute_chosen_recall_precision(l, gt)
            chosen_recalls.append(ch["chosen_recall"])
            chosen_precs.append(ch["chosen_precision"])
        if pool_recalls:
            print(f"  Cobertura do pool (Recall@pool):        {sum(pool_recalls) / len(pool_recalls):.3f}")
            print(f"  Precisão do pool (Precision@pool):      {sum(pool_precs) / len(pool_precs):.3f}")
        if chosen_recalls:
            print(f"  Acerto das escolhas (Recall@chosen):    {sum(chosen_recalls) / len(chosen_recalls):.3f}")
            print(f"  Precisão das escolhas (Precision@chosen):{sum(chosen_precs) / len(chosen_precs):.3f}")


def _print_chosen_in_gt_eval(
    logs: List[SessionLog],
    label: str,
    ground_truths: Optional[Dict[str, List[str]]] = None,
):
    """Print single-item and multi-choice GT hit evaluation."""
    n = len(logs)
    if n == 0:
        print(f"  {label}: no sessions found.")
        return

    print(f"  {label}:")
    print(f"  Sessions: {n}")
    if not ground_truths:
        print("  (No --dataset: cannot evaluate chosen ∈ ground_truth.)")
        return

    hit_rate_single = _ground_truth_hit_rate(logs, ground_truths)
    hit_rate_set = _ground_truth_set_hit_rate(logs, ground_truths)
    n_with_gt = sum(1 for l in logs if ground_truths.get(l.user_id))
    print(f"  Chosen ∈ GT (single best):  {hit_rate_single:.3f}  (sessions with GT: {n_with_gt})")
    print(f"  Chosen_set ∩ GT ≠ ∅ (multi): {hit_rate_set:.3f}  (sessions with GT: {n_with_gt})")
    print(f"  Per user (✓ = set hit, ○ = single hit only, ✗ = no match):")
    for l in sorted(logs, key=lambda x: x.user_id):
        gt_list = ground_truths.get(l.user_id, [])
        if not gt_list:
            continue
        chosen_set = l.chosen_items or ([l.chosen_item] if l.chosen_item and l.chosen_item != "none" else [])
        set_hit = classify_chosen_set_vs_gt(chosen_set, gt_list)
        single_hit = classify_chosen_vs_gt(l.chosen_item or "", gt_list) is not None
        mark = "✓" if set_hit else ("○" if single_hit else "✗")
        items_str = ", ".join(chosen_set[:3]) if chosen_set else (l.chosen_item or "none")
        n_chosen = len(chosen_set)
        print(f"    {mark}  {l.user_id:<14}  [{n_chosen} chosen] {items_str[:80]}")


def _relationship_match_rate(
    logs: List[SessionLog],
    shared_rels_map: Optional[Dict[str, List]] = None,
) -> float:
    """Fraction of sessions where chosen item matches any sharedRelationship."""
    hits = total = 0
    for log in logs:
        rels = (shared_rels_map or {}).get(log.user_id, log.shared_relationships)
        if not rels:
            continue
        total += 1
        result = compute_relationship_match(log, shared_relationships=rels)
        if result["match"]:
            hits += 1
    return hits / total if total else 0.0


def _print_relationship_match_eval(
    logs: List[SessionLog],
    label: str,
    shared_rels_map: Optional[Dict[str, List]] = None,
):
    """Print per-user relationship match results."""
    n = len(logs)
    if n == 0:
        print(f"  {label}: no sessions found.")
        return

    print(f"  {label}:")
    hits = total = 0
    for log in sorted(logs, key=lambda x: x.user_id):
        rels = (shared_rels_map or {}).get(log.user_id, log.shared_relationships)
        if not rels:
            continue
        total += 1
        result = compute_relationship_match(log, shared_relationships=rels)
        if result["match"]:
            hits += 1
        mark = "✓" if result["match"] else "✗"
        chosen = result["chosen"][:60] or "none"
        rels_str = ", ".join(f"{r[0]}:{r[1]}" for r in rels if isinstance(r, (list, tuple)) and len(r) >= 2)
        matched_str = ""
        if result["matches"]:
            matched_str = f"  [matched: {', '.join(m[1] for m in result['matches'])}]"
        print(f"    {mark}  {log.user_id:<14}  chosen: {chosen:<45} rels: {rels_str}{matched_str}")
    rate = hits / total if total else 0.0
    print(f"  Rate: {rate:.3f}  ({hits}/{total})")
    return rate


def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment results")
    parser.add_argument("--advisor", type=str, required=True, help="Advisor results directory")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline results directory")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset for ground truth (.pkl or .json)")
    parser.add_argument("--dataset_type", type=str, default="instructrec",
                        choices=["instructrec", "recassistbench"],
                        help="Dataset format: 'instructrec' or 'recassistbench'")
    parser.add_argument("--domain", type=str, default="book",
                        choices=["book", "movie"],
                        help="Domain for RecAssistBench GT: 'book' or 'movie'")
    parser.add_argument("--item_index", type=str,
                        default="../datasets/instructrec/combined_books_asin_mapping.csv",
                        help="Path to item index CSV with columns: index, title, description (InstructRec only)")
    parser.add_argument("--output", type=str, default=None, help="Save JSON report to this path")
    args = parser.parse_args()

    ground_truths: Dict[str, List[str]] = {}
    shared_rels_map: Dict[str, List] = {}
    if args.dataset:
        _print_section("Loading ground truth")
        if args.dataset_type == "recassistbench":
            ground_truths, shared_rels_map = _build_ground_truths_recassistbench(args.dataset, args.domain)
        else:
            ground_truths = _build_ground_truths_instructrec(args.dataset, args.item_index)

    adv_logs = _load_dir(args.advisor, "advisor")

    print("=" * 60)
    print("  EXPERIMENT EVALUATION")
    print("=" * 60)

    _print_section(f"ADVISOR  ({args.advisor})")
    _descriptive_stats(adv_logs, "Advisor", ground_truths or None)

    bas_logs: List[SessionLog] = []
    if args.baseline:
        bas_logs = _load_dir(args.baseline, "no_advisor")
        _print_section(f"BASELINE  ({args.baseline})")
        _descriptive_stats(bas_logs, "Baseline", ground_truths or None)

    report: Dict = {}

    def _p_str(p):
        if p is None:
            return "-"
        if isinstance(p, float) and (math.isnan(p) or math.isinf(p)):
            return "-"
        return f"{p:.4f}"

    if adv_logs and bas_logs:
        n_recsys_mismatch = warn_if_recsys_pools_differ(adv_logs, bas_logs)

        _print_section("ADVISOR vs BASELINE (compare_conditions)")
        report = compare_conditions(adv_logs, bas_logs, ground_truths=ground_truths or None)
        report["recsys_pool_mismatch_paired"] = n_recsys_mismatch

        skip_compare_print = {
            "n_advisor",
            "n_baseline",
            "recsys_pool_mismatch_paired",
            "chosen_in_gt_rate_advisor",
            "chosen_in_gt_rate_baseline",
            "chosen_in_gt_rate_delta",
            "chosen_in_gt_mcnemar_table",
            "chosen_in_gt_mcnemar_p",
            "chosen_in_gt_n_paired",
            "chosen_set_gt_rate_advisor",
            "chosen_set_gt_rate_baseline",
            "chosen_set_gt_rate_delta",
            "chosen_set_gt_mcnemar_table",
            "chosen_set_gt_mcnemar_p",
            "chosen_set_gt_n_paired",
        }
        _METRIC_FRIENDLY_NAMES = {
            "articulation":            "Articulação (Articulation)",
            "anchoring_resistance":    "Resist. ancoragem (Anchoring Res.)",
            "confirmation_resistance": "Resist. confirmação (Confirmation Res.)",
            "overload_resistance":     "Resist. sobrecarga (Overload Res.)",
            "bias_composite":          "Viés composto (Bias Composite)",
            "n_turns":                 "Nº de turnos (N Turns)",
            "recall_at_k":             "Cobertura do pool (Recall@pool)",
            "precision_at_k":          "Precisão do pool (Precision@pool)",
            "chosen_recall":           "Acerto das escolhas (Recall@chosen)",
            "chosen_precision":        "Precisão das escolhas (Precision@chosen)",
        }
        print(f"  {'Métrica':<44} {'Advisor':>8} {'Baseline':>8} {'Delta':>8} {'p-value':>8}")
        print(f"  {'-' * 78}")
        for key, vals in report.items():
            if key in skip_compare_print or not isinstance(vals, dict):
                continue
            adv_m = vals.get("advisor_mean", float("nan"))
            bas_m = vals.get("baseline_mean", float("nan"))
            delta = vals.get("delta", float("nan"))
            pval = vals.get("p_value", "")
            pval_str = f"{pval:.4f}" if isinstance(pval, float) else "-"
            label = _METRIC_FRIENDLY_NAMES.get(key, key)
            print(f"  {label:<44} {adv_m:>8.3f} {bas_m:>8.3f} {delta:>+8.3f} {pval_str:>8}")

        if ground_truths:
            report["chosen_in_gt_rate_advisor"] = _ground_truth_hit_rate(adv_logs, ground_truths)
            report["chosen_in_gt_rate_baseline"] = _ground_truth_hit_rate(bas_logs, ground_truths)
            report["chosen_in_gt_rate_delta"] = (
                report["chosen_in_gt_rate_advisor"] - report["chosen_in_gt_rate_baseline"]
            )

            report["chosen_set_gt_rate_advisor"] = _ground_truth_set_hit_rate(adv_logs, ground_truths)
            report["chosen_set_gt_rate_baseline"] = _ground_truth_set_hit_rate(bas_logs, ground_truths)
            report["chosen_set_gt_rate_delta"] = (
                report["chosen_set_gt_rate_advisor"] - report["chosen_set_gt_rate_baseline"]
            )

            tbl_chosen, n_chosen = _paired_binary_tables(
                adv_logs, bas_logs, ground_truths, _gt_chosen_hit
            )
            report["chosen_in_gt_mcnemar_table"] = tbl_chosen.tolist()
            report["chosen_in_gt_mcnemar_p"] = _mcnemar_binary_pvalue(tbl_chosen)
            report["chosen_in_gt_n_paired"] = len(n_chosen)

            tbl_set, n_set = _paired_binary_tables(
                adv_logs, bas_logs, ground_truths, _gt_chosen_set_hit
            )
            report["chosen_set_gt_mcnemar_table"] = tbl_set.tolist()
            report["chosen_set_gt_mcnemar_p"] = _mcnemar_binary_pvalue(tbl_set)
            report["chosen_set_gt_n_paired"] = len(n_set)

            _print_section("CHOSEN ∈ GROUND TRUTH (lista bookSubset / movieSubset)")
            _print_chosen_in_gt_eval(adv_logs, "Advisor", ground_truths)
            print()
            _print_chosen_in_gt_eval(bas_logs, "Baseline", ground_truths)

            print(f"\n  --- Single best item ---")
            print(f"  Chosen ∈ GT rate:  Adv {report['chosen_in_gt_rate_advisor']:.3f}  "
                  f"Bas {report['chosen_in_gt_rate_baseline']:.3f}  "
                  f"Δ {report['chosen_in_gt_rate_delta']:+.3f}")
            npair = report.get("chosen_in_gt_n_paired", 0)
            pc = report.get("chosen_in_gt_mcnemar_p")
            print(f"  McNemar ({npair} paired): 2x2 {report.get('chosen_in_gt_mcnemar_table')}  p = {_p_str(pc)}")

            print(f"\n  --- Multi-choice set (chosen_set ∩ GT ≠ ∅) ---")
            print(f"  Set hit rate:  Adv {report['chosen_set_gt_rate_advisor']:.3f}  "
                  f"Bas {report['chosen_set_gt_rate_baseline']:.3f}  "
                  f"Δ {report['chosen_set_gt_rate_delta']:+.3f}")
            npair_s = report.get("chosen_set_gt_n_paired", 0)
            ps = report.get("chosen_set_gt_mcnemar_p")
            print(f"  McNemar ({npair_s} paired): 2x2 {report.get('chosen_set_gt_mcnemar_table')}  p = {_p_str(ps)}")

        if shared_rels_map:
            _print_section("CHOSEN MATCHES RELATIONSHIP (author/category/topic)")
            adv_rel_rate = _print_relationship_match_eval(adv_logs, "Advisor", shared_rels_map)
            print()
            bas_rel_rate = _print_relationship_match_eval(bas_logs, "Baseline", shared_rels_map)
            report["rel_match_rate_advisor"] = adv_rel_rate or 0.0
            report["rel_match_rate_baseline"] = bas_rel_rate or 0.0
            report["rel_match_rate_delta"] = (
                report["rel_match_rate_advisor"] - report["rel_match_rate_baseline"]
            )
            print(f"\n  Relationship match rate:  "
                  f"Advisor {report['rel_match_rate_advisor']:.3f}  "
                  f"Baseline {report['rel_match_rate_baseline']:.3f}  "
                  f"Δ {report['rel_match_rate_delta']:+.3f}")

        print(f"\n  N advisor:  {report.get('n_advisor', 0)}")
        print(f"  N baseline: {report.get('n_baseline', 0)}")
    elif adv_logs and not bas_logs:
        print("\n  (No baseline provided; skipping compare_conditions & paired GT tests.)")
        if ground_truths:
            report["chosen_in_gt_rate_advisor"] = _ground_truth_hit_rate(adv_logs, ground_truths)
            report["chosen_set_gt_rate_advisor"] = _ground_truth_set_hit_rate(adv_logs, ground_truths)
            _print_section("CHOSEN ∈ GROUND TRUTH")
            _print_chosen_in_gt_eval(adv_logs, "Advisor", ground_truths)
        if shared_rels_map:
            _print_section("CHOSEN MATCHES RELATIONSHIP (author/category/topic)")
            adv_rel_rate = _print_relationship_match_eval(adv_logs, "Advisor", shared_rels_map)
            report["rel_match_rate_advisor"] = adv_rel_rate or 0.0

    if args.output and (report or ground_truths):
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved to {args.output}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
