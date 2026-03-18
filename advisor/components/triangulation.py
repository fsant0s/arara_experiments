from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TriangulationResult:
    item_convergence: float
    explanation_convergence: float
    framing_divergence: bool
    consensus_items: List[dict]
    divergent_items: Dict[str, List[dict]]
    all_items: List[dict]
    per_llm_items: Dict[str, List[dict]] = field(default_factory=dict)


_sbert_model = None


def _get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def _normalize_title(title: str) -> str:
    return title.strip().lower()


def compute_item_convergence(outputs: Dict[str, List[dict]]) -> float:
    """Jaccard similarity across all LLM Top-K item sets."""
    if len(outputs) < 2:
        return 0.0

    title_sets = []
    for items in outputs.values():
        titles = {_normalize_title(it["title"]) for it in items if "title" in it}
        title_sets.append(titles)

    intersection = set.intersection(*title_sets)
    union = set.union(*title_sets)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def compute_explanation_convergence(
    outputs: Dict[str, List[dict]],
    overlapping_titles: set[str],
) -> float:
    """Average SBERT cosine similarity of explanations for overlapping items."""
    if not overlapping_titles:
        return 0.0

    model = _get_sbert_model()
    similarities = []

    for title in overlapping_titles:
        explanations = []
        for items in outputs.values():
            for it in items:
                if _normalize_title(it.get("title", "")) == title and it.get("explanation"):
                    explanations.append(it["explanation"])
                    break

        if len(explanations) < 2:
            continue

        embeddings = model.encode(explanations, convert_to_numpy=True)
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                cos = float(np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9
                ))
                similarities.append(cos)

    return float(np.mean(similarities)) if similarities else 0.0


def detect_framing_divergence(
    outputs: Dict[str, List[dict]],
    threshold: float = 0.7,
) -> bool:
    """True if LLMs interpret the query through substantially different lenses."""
    if len(outputs) < 2:
        return False

    model = _get_sbert_model()
    centroids = []

    for items in outputs.values():
        explanations = [it["explanation"] for it in items if it.get("explanation")]
        if not explanations:
            continue
        embeddings = model.encode(explanations, convert_to_numpy=True)
        centroid = embeddings.mean(axis=0)
        centroids.append(centroid)

    if len(centroids) < 2:
        return False

    min_cos = 1.0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            cos = float(np.dot(centroids[i], centroids[j]) / (
                np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j]) + 1e-9
            ))
            min_cos = min(min_cos, cos)

    return min_cos < threshold


def _build_item_maps(outputs: Dict[str, List[dict]]) -> Tuple[List[dict], List[dict], Dict[str, List[dict]]]:
    """Separate consensus items from divergent items."""
    title_to_sources: Dict[str, List[str]] = {}
    title_to_item: Dict[str, dict] = {}

    for llm_name, items in outputs.items():
        for it in items:
            norm = _normalize_title(it.get("title", ""))
            if not norm:
                continue
            title_to_sources.setdefault(norm, []).append(llm_name)
            if norm not in title_to_item:
                title_to_item[norm] = {
                    "title": it.get("title", ""),
                    "explanation": it.get("explanation", ""),
                    "rank": it.get("rank", 0),
                    "sources": [],
                }
            title_to_item[norm]["sources"] = list(set(title_to_sources[norm]))

    n_llms = len(outputs)
    consensus = []
    divergent: Dict[str, List[dict]] = {}

    for norm_title, sources in title_to_sources.items():
        item = title_to_item[norm_title]
        if len(set(sources)) >= 2:
            consensus.append(item)
        else:
            source = sources[0]
            divergent.setdefault(source, []).append(item)

    consensus.sort(key=lambda x: (-len(x["sources"]), x.get("rank", 999)))

    all_items = consensus + [it for items in divergent.values() for it in items]
    return consensus, divergent, all_items


def triangulate(outputs: Dict[str, List[dict]]) -> TriangulationResult:
    """
    Main entry point. Computes item convergence, explanation convergence,
    framing divergence, and separates consensus from divergent items.

    Parameters
    ----------
    outputs : dict
        Keys are LLM names, values are lists of
        {"title": str, "explanation": str, "rank": int}.
    """
    item_conv = compute_item_convergence(outputs)

    title_sets = []
    for items in outputs.values():
        title_sets.append({_normalize_title(it["title"]) for it in items if "title" in it})
    overlapping = set.intersection(*title_sets) if title_sets else set()

    expl_conv = compute_explanation_convergence(outputs, overlapping)
    framing_div = detect_framing_divergence(outputs)
    consensus, divergent, all_items = _build_item_maps(outputs)

    return TriangulationResult(
        item_convergence=item_conv,
        explanation_convergence=expl_conv,
        framing_divergence=framing_div,
        consensus_items=consensus,
        divergent_items=divergent,
        all_items=all_items,
        per_llm_items=dict(outputs),
    )
