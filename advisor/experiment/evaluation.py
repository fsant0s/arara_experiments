from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class TurnLog:
    turn: int
    action: str
    advisor_message: str
    user_response: str
    belief_state: dict
    items_shown: list

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "action": self.action,
            "advisor_message": self.advisor_message,
            "user_response": self.user_response,
            "belief_state": self.belief_state,
            "items_shown": self.items_shown,
        }


@dataclass
class SessionLog:
    user_id: str
    condition: str
    turns: List[TurnLog]
    chosen_item: str
    final_state: Optional[dict]
    recsys_outputs: Optional[dict]

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "condition": self.condition,
            "turns": [t.to_dict() for t in self.turns],
            "chosen_item": self.chosen_item,
            "final_state": self.final_state,
            "recsys_outputs": self.recsys_outputs,
        }


def compute_alignment(
    session_log: SessionLog,
    llm_client: Callable[[str], str],
) -> float:
    """
    Preference-Decision Alignment: does the chosen item match the user's
    articulated preferences?

    Uses an LLM to extract experiential dimensions of the chosen item,
    then computes overlap with the user's stated preference dimensions.
    """
    if not session_log.chosen_item or session_log.chosen_item == "none":
        return 0.0

    final_state = session_log.final_state or {}
    prefs = final_state.get("preference_dimensions", {})
    if not prefs:
        return 0.0

    prompt = f"""Given this item title: "{session_log.chosen_item}"
And the context of a recommendation conversation, estimate the item's experiential dimensions.
Return JSON with keys from [tone, pace, complexity, emotional_register, formality, scope].
Only include dimensions you can reasonably infer. Return ONLY valid JSON."""

    raw = llm_client(prompt)
    try:
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            lines = raw_clean.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_clean = "\n".join(lines)
        item_dims = json.loads(raw_clean)
    except (json.JSONDecodeError, ValueError):
        item_dims = {}

    if not item_dims:
        return 0.0

    shared_keys = set(prefs.keys()) & set(item_dims.keys())
    if not shared_keys:
        return 0.0

    matches = sum(1 for k in shared_keys if str(prefs[k]).lower() == str(item_dims[k]).lower())
    return matches / len(shared_keys)


def compute_articulation(session_log: SessionLog) -> float:
    """
    Articulation Improvement: σ(T) - σ(0).
    Measures how much the user's preference specificity increased.
    """
    if not session_log.turns:
        return 0.0

    sigma_0 = session_log.turns[0].belief_state.get("preference_specificity", 0.0)

    final = session_log.final_state or {}
    sigma_t = final.get("preference_specificity", 0.0)

    return sigma_t - sigma_0


def compute_bias_resistance(session_log: SessionLog) -> Dict[str, float]:
    """
    Bias Resistance: anchoring, confirmation, overload sub-scores.
    """
    result = {"anchoring": 0.0, "confirmation": 0.0, "overload": 0.0}

    if not session_log.turns or not session_log.chosen_item or session_log.chosen_item == "none":
        return result

    first_items_shown = session_log.turns[0].items_shown if session_log.turns else []
    if first_items_shown:
        first_item = first_items_shown[0].strip().lower()
        chosen = session_log.chosen_item.strip().lower()
        result["anchoring"] = 0.0 if chosen == first_item else 1.0
    else:
        result["anchoring"] = 1.0

    recsys = session_log.recsys_outputs or {}
    if recsys:
        chosen_lower = session_log.chosen_item.strip().lower()
        llms_with_chosen = 0
        for items in recsys.values():
            titles = [it.get("title", "").strip().lower() for it in items]
            if chosen_lower in titles:
                llms_with_chosen += 1
        n_llms = len(recsys)
        positives = session_log.final_state.get("items_positive", []) if session_log.final_state else []
        llms_considered = set()
        for items in recsys.values():
            titles_set = {it.get("title", "").strip().lower() for it in items}
            for pos in positives:
                if pos.lower() in titles_set:
                    llms_considered.add(id(items))
                    break
        result["confirmation"] = len(llms_considered) / n_llms if n_llms > 0 else 0.0

    max_turns = 6
    n_turns = len(session_log.turns)
    result["overload"] = 1.0 if (session_log.chosen_item != "none" and n_turns <= max_turns) else 0.0

    return result


def compute_ground_truth(
    session_log: SessionLog,
    ground_truth_items: List[str],
    k: Optional[int] = None,
) -> Dict[str, float]:
    """
    Ground-Truth Alignment: Recall@K and Precision@K of presented items
    against ground truth.
    """
    if not ground_truth_items:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0}

    gt_set = {t.strip().lower() for t in ground_truth_items}
    if k is None:
        k = len(gt_set)

    presented = set()
    for turn in session_log.turns:
        for title in turn.items_shown:
            presented.add(title.strip().lower())

    presented_list = list(presented)[:k] if k else list(presented)
    presented_set = set(presented_list)

    hits = len(presented_set & gt_set)
    recall = hits / len(gt_set) if gt_set else 0.0
    precision = hits / len(presented_set) if presented_set else 0.0

    return {"recall_at_k": recall, "precision_at_k": precision}


def compute_reward(
    session_log: SessionLog,
    llm_client: Optional[Callable[[str], str]] = None,
    ground_truth: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Composite reward: w1*align + w2*artic + w3*bias + w4*gt.
    """
    w = weights or {"align": 0.3, "artic": 0.3, "bias": 0.3, "gt": 0.1}

    r_align = compute_alignment(session_log, llm_client) if llm_client else 0.0
    r_artic = compute_articulation(session_log)

    bias = compute_bias_resistance(session_log)
    r_bias = (bias["anchoring"] + bias["confirmation"] + bias["overload"]) / 3.0

    gt_metrics = compute_ground_truth(session_log, ground_truth or [])
    r_gt = gt_metrics["recall_at_k"]

    return (
        w.get("align", 0.3) * r_align
        + w.get("artic", 0.3) * r_artic
        + w.get("bias", 0.3) * r_bias
        + w.get("gt", 0.1) * r_gt
    )


def compare_conditions(
    advisor_logs: List[SessionLog],
    baseline_logs: List[SessionLog],
    llm_client: Optional[Callable[[str], str]] = None,
    ground_truths: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """
    Compare advisor vs baseline across all users.
    Returns aggregated metrics with means and standard deviations.
    """
    import math
    import numpy as np

    ground_truths = ground_truths or {}

    def _compute_metrics(logs: List[SessionLog]) -> Dict[str, List[float]]:
        metrics = {
            "articulation": [],
            "anchoring_resistance": [],
            "confirmation_resistance": [],
            "overload_resistance": [],
            "bias_composite": [],
            "n_turns": [],
            "recall_at_k": [],
            "precision_at_k": [],
        }
        for log in logs:
            metrics["articulation"].append(compute_articulation(log))
            bias = compute_bias_resistance(log)
            metrics["anchoring_resistance"].append(bias["anchoring"])
            metrics["confirmation_resistance"].append(bias["confirmation"])
            metrics["overload_resistance"].append(bias["overload"])
            metrics["bias_composite"].append(
                (bias["anchoring"] + bias["confirmation"] + bias["overload"]) / 3.0
            )
            metrics["n_turns"].append(len(log.turns))

            gt = ground_truths.get(log.user_id, [])
            gt_m = compute_ground_truth(log, gt)
            metrics["recall_at_k"].append(gt_m["recall_at_k"])
            metrics["precision_at_k"].append(gt_m["precision_at_k"])

        return metrics

    adv_m = _compute_metrics(advisor_logs)
    bas_m = _compute_metrics(baseline_logs)

    def _safe_mean_std(arr: np.ndarray) -> tuple[float, float]:
        """
        Compute mean/std defensively.

        - Empty array     → (nan, nan)
        - Single element  → (value, 0.0)
        - >= 2 elements   → (np.mean, np.std)
        """
        if arr.size == 0:
            return math.nan, math.nan
        if arr.size == 1:
            v = float(arr[0])
            return v, 0.0
        return float(np.mean(arr)), float(np.std(arr))

    report = {}
    for key in adv_m:
        adv_arr = np.array(adv_m[key])
        bas_arr = np.array(bas_m[key])

        adv_mean, adv_std = _safe_mean_std(adv_arr)
        bas_mean, bas_std = _safe_mean_std(bas_arr)

        report[key] = {
            "advisor_mean": adv_mean,
            "advisor_std": adv_std,
            "baseline_mean": bas_mean,
            "baseline_std": bas_std,
            "delta": float(adv_mean - bas_mean)
            if not (math.isnan(adv_mean) or math.isnan(bas_mean))
            else math.nan,
        }

        if adv_arr.size >= 2 and bas_arr.size >= 2:
            try:
                from scipy.stats import ttest_ind
                stat, pval = ttest_ind(adv_arr, bas_arr, equal_var=False)
                report[key]["t_stat"] = float(stat)
                report[key]["p_value"] = float(pval)
            except ImportError:
                pass

    report["n_advisor"] = len(advisor_logs)
    report["n_baseline"] = len(baseline_logs)
    return report
