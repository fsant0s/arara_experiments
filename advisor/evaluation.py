from __future__ import annotations

import html
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional


def normalize_title_for_gt_match(title: str) -> str:
    """
    Normalize titles for ground-truth comparison: HTML entities (e.g. &amp;),
    lowercase, collapse whitespace.
    """
    s = html.unescape(str(title or "")).strip().lower()
    return re.sub(r"\s+", " ", s)


def _strip_title(title: str) -> str:
    """Aggressive normalization: remove diacritics, punctuation, parenthetical
    suffixes, and subtitles after ':' or '—'."""
    s = normalize_title_for_gt_match(title)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s*[:(—\-–]\s*.{0,}$", "", s)
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def classify_chosen_vs_gt(chosen_item: str, gt_list: List[str]) -> Optional[str]:
    """
    Classify whether the chosen title matches any ground-truth title.

    Returns
    -------
    None
        No valid chosen item or no match.
    "strict"
        Normalized chosen string equals normalized GT (exact after cleanup),
        OR stripped versions match exactly.
    "fuzzy"
        Substring match, token Jaccard >= 0.5, or SequenceMatcher ratio >= 0.75.
    """
    chosen_n = normalize_title_for_gt_match(chosen_item or "")
    if not chosen_n or chosen_n == "none":
        return None
    normalized_gts = [normalize_title_for_gt_match(g) for g in gt_list if g]
    normalized_gts = [g for g in normalized_gts if g]
    if not normalized_gts:
        return None

    chosen_s = _strip_title(chosen_item or "")

    for gt_n in normalized_gts:
        if chosen_n == gt_n:
            return "strict"
    for gt_n in normalized_gts:
        gt_s = _strip_title(gt_n)
        if chosen_s and gt_s and chosen_s == gt_s:
            return "strict"

    for gt_n in normalized_gts:
        if chosen_n in gt_n or gt_n in chosen_n:
            return "fuzzy"

    for gt_n in normalized_gts:
        gt_s = _strip_title(gt_n)
        if not chosen_s or not gt_s:
            continue
        if chosen_s in gt_s or gt_s in chosen_s:
            return "fuzzy"
        if _token_jaccard(chosen_s, gt_s) >= 0.5:
            return "fuzzy"
        if SequenceMatcher(None, chosen_s, gt_s).ratio() >= 0.75:
            return "fuzzy"

    return None


def classify_chosen_set_vs_gt(chosen_items: List[str], gt_list: List[str]) -> bool:
    """Return True if ANY item in ``chosen_items`` matches ANY title in ``gt_list``."""
    for item in chosen_items:
        if classify_chosen_vs_gt(item, gt_list) is not None:
            return True
    return False


@dataclass
class TurnLog:
    turn: int
    action: str
    advisor_message: str
    user_response: str
    belief_state: dict
    # Triangulation/debiasing pool for this turn (belief, reward, bandit); may not appear in advisor text.
    items_considered_for_internal_turn_state: list

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "action": self.action,
            "user_response": self.user_response,
            "advisor_message": self.advisor_message,
            "belief_state": self.belief_state,
            "items_considered_for_internal_turn_state": self.items_considered_for_internal_turn_state,
        }


@dataclass
class SessionLog:
    user_id: str
    condition: str
    turns: List[TurnLog]
    chosen_item: str
    final_state: Optional[dict]
    recsys_outputs: Optional[dict]
    initial_user_message: str = ""
    ground_truth: List[str] = field(default_factory=list)
    shared_relationships: List = field(default_factory=list)
    chosen_items: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "condition": self.condition,
            "initial_user_message": self.initial_user_message,
            "ground_truth": self.ground_truth,
            "shared_relationships": self.shared_relationships,
            "recsys_outputs": self.recsys_outputs,
            "turns": [t.to_dict() for t in self.turns],
            "chosen_item": self.chosen_item,
            "chosen_items": self.chosen_items,
            "final_state": self.final_state,
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

    first_internal = (
        session_log.turns[0].items_considered_for_internal_turn_state if session_log.turns else []
    )
    if first_internal:
        first_item = first_internal[0].strip().lower()
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
    Ground-truth alignment over titles shown in the session pool
    (``items_considered_for_internal_turn_state``), exact match on
    lowercased stripped strings.

    - If ``k`` is None: use the **full** set of distinct titles shown
      (union across turns). Recall = |GT ∩ presented| / |GT|;
      precision = |GT ∩ presented| / |presented|.
    - If ``k`` is set: evaluate only the first ``k`` **distinct** titles
      in encounter order (turn order, then list order within each turn).
      This approximates recall/precision@K when the pool is treated as
      an ordered stream.
    """
    if not ground_truth_items:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0}

    gt_set = {
        normalize_title_for_gt_match(t)
        for t in ground_truth_items
        if t and str(t).strip()
    }

    ordered_presented: List[str] = []
    seen: set = set()
    for turn in session_log.turns:
        for title in turn.items_considered_for_internal_turn_state:
            t = normalize_title_for_gt_match(title)
            if not t or t in seen:
                continue
            seen.add(t)
            ordered_presented.append(t)

    if k is None:
        presented_eval = set(ordered_presented)
    else:
        presented_eval = set(ordered_presented[: max(0, int(k))])

    hits = len(presented_eval & gt_set)
    recall = hits / len(gt_set) if gt_set else 0.0
    precision = hits / len(presented_eval) if presented_eval else 0.0

    return {"recall_at_k": recall, "precision_at_k": precision}


def compute_chosen_recall_precision(
    session_log: SessionLog,
    ground_truth_items: List[str],
) -> Dict[str, float]:
    """
    Ground-truth alignment based on what the user ACTUALLY CHOSE (chosen_items).

    recall  = |GT ∩ chosen_items| / |GT|
    precision = |GT ∩ chosen_items| / |chosen_items|

    This is the standard CRS evaluation metric: did the user end up picking
    items that match their ground truth?  Unlike compute_ground_truth (which
    measures coverage in the internal pool), this measures the *decision* quality.
    """
    if not ground_truth_items:
        return {"chosen_recall": 0.0, "chosen_precision": 0.0}

    gt_set = {
        normalize_title_for_gt_match(t)
        for t in ground_truth_items
        if t and str(t).strip()
    }

    chosen = session_log.chosen_items or (
        [session_log.chosen_item]
        if session_log.chosen_item and session_log.chosen_item != "none"
        else []
    )
    chosen_norm = {normalize_title_for_gt_match(c) for c in chosen if c}

    hits = len(chosen_norm & gt_set)
    recall = hits / len(gt_set) if gt_set else 0.0
    precision = hits / len(chosen_norm) if chosen_norm else 0.0

    return {"chosen_recall": recall, "chosen_precision": precision}


def compute_relationship_match(
    session_log: SessionLog,
    shared_relationships: Optional[List] = None,
) -> Dict[str, Any]:
    """Check whether the chosen item matches the user's sharedRelationships.

    This is a softer GT metric than exact title match: a user who asked for
    "books by Alison Weir" and chose *The Six Wives of Henry VIII* (by Alison
    Weir, but not in bookSubset) is still a relationship-level hit.

    Returns
    -------
    dict with:
        match : bool   – any relationship matched
        matches : list  – which (rel_type, value) pairs matched
        chosen  : str   – the chosen item
    """
    rels = shared_relationships or session_log.shared_relationships or []
    chosen = (session_log.chosen_item or "").strip()
    if not chosen or chosen == "none" or not rels:
        return {"match": False, "matches": [], "chosen": chosen}

    chosen_lower = chosen.lower()

    recsys_text = ""
    for items in (session_log.recsys_outputs or {}).values():
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            title = (it.get("title") or "").strip()
            if title.lower() == chosen_lower:
                recsys_text = (it.get("explanation") or "").lower()
                break
        if recsys_text:
            break

    advisor_text = ""
    for turn in session_log.turns:
        if chosen_lower in turn.advisor_message.lower():
            advisor_text += " " + turn.advisor_message.lower()

    search_text = f"{chosen_lower} {recsys_text} {advisor_text}"

    matched = []
    for rel in rels:
        if not isinstance(rel, (list, tuple)) or len(rel) < 2:
            continue
        rel_type, rel_value = str(rel[0]), str(rel[1])
        val_lower = rel_value.lower()
        if val_lower in search_text:
            matched.append((rel_type, rel_value))
        else:
            val_words = val_lower.split()
            if len(val_words) > 1 and all(w in search_text for w in val_words):
                matched.append((rel_type, rel_value))

    return {"match": len(matched) > 0, "matches": matched, "chosen": chosen}


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
            "chosen_recall": [],
            "chosen_precision": [],
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

            ch_m = compute_chosen_recall_precision(log, gt)
            metrics["chosen_recall"].append(ch_m["chosen_recall"])
            metrics["chosen_precision"].append(ch_m["chosen_precision"])

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
