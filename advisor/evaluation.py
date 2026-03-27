"""Evaluation metrics for the multi-party advisor experiment."""
from __future__ import annotations

import html
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional


# ─── Title normalization ──────────────────────────────────────

def normalize_title(title: str) -> str:
    s = html.unescape(str(title or "")).strip().lower()
    return re.sub(r"\s+", " ", s)


def _strip_title(title: str) -> str:
    s = normalize_title(title)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\s*[:(—\-–;,]\s*.{0,}$", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    stop = {"a", "the", "an", "of", "in", "for", "and", "to", "by"}
    tokens = s.split()
    if len(tokens) > 3:
        s = " ".join(t for t in tokens if t not in stop)
    return s


def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def title_matches_gt(chosen: str, gt_list: List[str]) -> Optional[str]:
    """Return 'strict', 'fuzzy', or None."""
    cn = normalize_title(chosen)
    if not cn or cn == "none":
        return None
    gts = [normalize_title(g) for g in gt_list if g]
    if not gts:
        return None
    cs = _strip_title(chosen)
    for g in gts:
        if cn == g:
            return "strict"
        gs = _strip_title(g)
        if cs == gs:
            return "strict"
    for g in gts:
        if cn in g or g in cn:
            return "fuzzy"
        gs = _strip_title(g)
        if cs and gs and (cs in gs or gs in cs):
            return "fuzzy"
        if _token_jaccard(cs, gs) >= 0.45:
            return "fuzzy"
        if SequenceMatcher(None, cs, gs).ratio() >= 0.70:
            return "fuzzy"
    return None


# ─── Data structures ──────────────────────────────────────────

@dataclass
class TurnLog:
    turn: int
    action: str
    advisor_message: str
    user_message: str
    user_target: str = ""
    rs_responses: Dict[str, str] = field(default_factory=dict)
    belief_state: dict = field(default_factory=dict)
    items_mentioned: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "user_message": self.user_message,
            "user_target": self.user_target,
            "rs_responses": self.rs_responses,
            "advisor_action": self.action,
            "advisor_message": self.advisor_message,
            "belief_state": self.belief_state,
            "items_mentioned": self.items_mentioned,
        }


@dataclass
class SessionLog:
    user_id: str
    condition: str
    turns: List[TurnLog]
    chosen_items: List[str]
    ground_truth: List[str]
    rs1_gt_items: List[str] = field(default_factory=list)
    rs2_gt_items: List[str] = field(default_factory=list)
    rs1_history: List[dict] = field(default_factory=list)
    rs2_history: List[dict] = field(default_factory=list)
    final_state: Optional[dict] = None
    initial_user_message: str = ""
    shared_relationships: List = field(default_factory=list)
    n_choices: int = 1

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "condition": self.condition,
            "turns": [t.to_dict() for t in self.turns],
            "chosen_items": self.chosen_items,
            "ground_truth": self.ground_truth,
            "rs1_gt_items": self.rs1_gt_items,
            "rs2_gt_items": self.rs2_gt_items,
            "rs1_history": self.rs1_history,
            "rs2_history": self.rs2_history,
            "final_state": self.final_state,
            "initial_user_message": self.initial_user_message,
            "shared_relationships": self.shared_relationships,
            "n_choices": self.n_choices,
        }


# ─── Per-session metrics ──────────────────────────────────────

def compute_chosen_recall_precision(
    chosen: List[str], gt: List[str],
) -> Dict[str, float]:
    """Recall and precision of chosen items vs ground truth."""
    if not gt:
        return {"recall": 0.0, "precision": 0.0, "n_hits": 0}
    hits = 0
    for c in chosen:
        if title_matches_gt(c, gt):
            hits += 1
    recall = hits / len(gt) if gt else 0.0
    precision = hits / len(chosen) if chosen else 0.0
    return {"recall": recall, "precision": precision, "n_hits": hits}


def compute_cross_rs_discovery(
    chosen: List[str],
    rs1_gt: List[str],
    rs2_gt: List[str],
    rs1_history: List[dict],
    rs2_history: List[dict],
) -> Dict[str, float]:
    """Did the user discover GT items that were only in the 'other' RS?"""
    rs1_text = " ".join(m.get("content", "") for m in rs1_history if m.get("role") == "assistant").lower()
    rs2_text = " ".join(m.get("content", "") for m in rs2_history if m.get("role") == "assistant").lower()

    cross_hits = 0
    total_gt = len(rs1_gt) + len(rs2_gt)
    for c in chosen:
        cn = normalize_title(c)
        for gt_title in rs1_gt:
            gn = normalize_title(gt_title)
            if (cn in gn or gn in cn) and gn in rs2_text:
                cross_hits += 1
        for gt_title in rs2_gt:
            gn = normalize_title(gt_title)
            if (cn in gn or gn in cn) and gn in rs1_text:
                cross_hits += 1

    return {
        "cross_rs_hits": cross_hits,
        "cross_rs_rate": cross_hits / max(total_gt, 1),
    }


def compute_session_metrics(log: SessionLog) -> Dict[str, Any]:
    """Compute all metrics for one session."""
    m: Dict[str, Any] = {}
    m["n_turns"] = len(log.turns)
    m["n_chosen"] = len(log.chosen_items)

    rp = compute_chosen_recall_precision(log.chosen_items, log.ground_truth)
    m.update(rp)

    cross = compute_cross_rs_discovery(
        log.chosen_items, log.rs1_gt_items, log.rs2_gt_items,
        log.rs1_history, log.rs2_history,
    )
    m.update(cross)

    rs_targets = [t.user_target for t in log.turns if t.user_target and t.user_target != "decision"]
    m["n_rs1_talks"] = sum(1 for t in rs_targets if t == "RS1")
    m["n_rs2_talks"] = sum(1 for t in rs_targets if t == "RS2")

    if log.final_state and isinstance(log.final_state, dict):
        m["overload_risk_final"] = log.final_state.get("overload_risk", 0.0)
        m["preference_specificity"] = log.final_state.get("preference_specificity", 0.0)
        followed = log.final_state.get("advisor_suggestions_followed", 0)
        total = log.final_state.get("advisor_suggestions_total", 0)
        m["follow_rate"] = followed / max(total, 1)
    else:
        m["overload_risk_final"] = 0.0
        m["preference_specificity"] = 0.0
        m["follow_rate"] = 0.0

    return m


# ─── Aggregate comparison ─────────────────────────────────────

def compare_conditions(
    advisor_logs: List[SessionLog],
    baseline_logs: List[SessionLog],
) -> Dict[str, Any]:
    """Compare advisor vs baseline across all sessions."""
    adv_metrics = [compute_session_metrics(l) for l in advisor_logs]
    bas_metrics = [compute_session_metrics(l) for l in baseline_logs]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = _mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

    keys = ["recall", "precision", "n_hits", "n_turns", "cross_rs_hits",
            "cross_rs_rate", "follow_rate", "overload_risk_final",
            "preference_specificity", "n_rs1_talks", "n_rs2_talks"]

    result: Dict[str, Any] = {"n_advisor": len(adv_metrics), "n_baseline": len(bas_metrics)}

    for k in keys:
        av = [m.get(k, 0.0) for m in adv_metrics]
        bv = [m.get(k, 0.0) for m in bas_metrics]
        result[k] = {
            "advisor_mean": round(_mean(av), 4),
            "advisor_std": round(_std(av), 4),
            "baseline_mean": round(_mean(bv), 4),
            "baseline_std": round(_std(bv), 4),
            "delta": round(_mean(av) - _mean(bv), 4),
        }

        try:
            from scipy.stats import ttest_ind
            if len(av) >= 2 and len(bv) >= 2:
                _, p = ttest_ind(av, bv, equal_var=False)
                result[k]["p_value"] = round(p, 4)
        except ImportError:
            pass

    return result
