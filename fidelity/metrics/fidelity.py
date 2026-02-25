from typing import Callable, List, Dict, Any, Sequence, Optional
import math

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def remove_items_from_profile(
    user_profile: Sequence[Any],
    items_to_remove: Sequence[Any]
) -> List[Any]:
    """
    Returns a copy of user_profile with all items in items_to_remove removed.
    """
    to_remove = set(items_to_remove)
    return [x for x in user_profile if x not in to_remove]


def add_items_to_profile(
    user_profile: Sequence[Any],
    items_to_add: Sequence[Any]
) -> List[Any]:
    """
    Returns a copy of user_profile with items_to_add appended
    (without duplicating items already in the profile).
    """
    profile_set = set(user_profile)
    new_profile = list(user_profile)
    for x in items_to_add:
        if x not in profile_set:
            new_profile.append(x)
            profile_set.add(x)
    return new_profile


def ndcg_at_k(
    ranking: Sequence[Any],
    relevance: Dict[Any, float],
    k: Optional[int] = None
) -> float:
    """
    Computes NDCG@k for a given ranking and a relevance dictionary.
    ranking: ordered list of item_ids (best first).
    relevance: dict item_id -> relevance (e.g., click (0/1), rating, etc.)
    k: cutoff; if None, uses the full ranking length.
    """
    if k is None:
        k = len(ranking)

    # DCG
    dcg = 0.0
    for idx, item in enumerate(ranking[:k]):
        rel = relevance.get(item, 0.0)
        if rel <= 0:
            continue
        dcg += (2.0**rel - 1.0) / math.log2(idx + 2.0)

    # Ideal DCG
    ideal_rels = sorted(relevance.values(), reverse=True)
    ideal_dcg = 0.0
    for idx, rel in enumerate(ideal_rels[:k]):
        if rel <= 0:
            continue
        ideal_dcg += (2.0**rel - 1.0) / math.log2(idx + 2.0)

    if ideal_dcg == 0.0:
        return 0.0

    return dcg / ideal_dcg

def pos_at_kr_ke_official(
    rank_fn, 
    user_profile, 
    target_item, # O item 'y' que foi recomendado
    explanation_items_ordered, 
    K_r, 
    K_e
):
    # 1. Criar perfil contrafactual removendo os Top-Ke itens explicativos
    items_to_remove = explanation_items_ordered[:K_e]
    cf_profile = [item for item in user_profile if item not in items_to_remove]
    
    # 2. Obter o novo ranking
    cf_ranking = rank_fn(cf_profile)
    
    # 3. Encontrar a nova posição do item alvo 'y'
    try:
        new_rank = cf_ranking.index(target_item) + 1 # +1 pois rank costuma ser 1-based
    except ValueError:
        new_rank = float('inf') # Se sumiu do ranking, o rank é muito alto
        
    # 4. Retornar 1 se ainda estiver no Top-Kr, 0 caso contrário (Função Indicadora)
    return 1.0 if new_rank <= K_r else 0.0


# -------------------------------------------------------------------
# 1) POS@K_r, K_e
# -------------------------------------------------------------------

def pos_at_kr_ke(
    rank_fn: Callable[[Sequence[Any]], List[Any]],
    user_profile: Sequence[Any],
    explanation_items_ordered: Sequence[Any],
    K_r: int,
    K_e: int
) -> float:
    """
    POS@K_r, K_e:
    Average ranking drop of the top-K_r items after removing the top-K_e
    explanation features from the user profile.

    rank_fn(profile) -> ranked list of item_ids (best first).
    user_profile: sequence of items describing the user (e.g., past interactions).
    explanation_items_ordered: items sorted by explanation importance (most -> least).
    K_r: number of top recommendations to track.
    K_e: number of explanation items to remove.
    """
    # Baseline ranking
    baseline_ranking = rank_fn(user_profile)
    if len(baseline_ranking) == 0:
        return 0.0

    K_r = min(K_r, len(baseline_ranking))
    baseline_top = baseline_ranking[:K_r]
    baseline_pos = {item: idx for idx, item in enumerate(baseline_ranking)}

    # Remove top-K_e explanation items
    items_to_remove = explanation_items_ordered[:K_e]
    cf_profile = remove_items_from_profile(user_profile, items_to_remove)

    cf_ranking = rank_fn(cf_profile)
    cf_pos = {item: idx for idx, item in enumerate(cf_ranking)}
    max_pos = len(cf_ranking)

    # Compute average position increase
    deltas = []
    for item in baseline_top:
        old = baseline_pos[item]
        new = cf_pos.get(item, max_pos)
        deltas.append(max(0, new - old))  # only count drops

    if not deltas:
        return 0.0

    return sum(deltas) / len(deltas)


# -------------------------------------------------------------------
# 2) CDCG@K_e
# -------------------------------------------------------------------

def cdcg_at_ke(
    rank_fn: Callable[[Sequence[Any]], List[Any]],
    user_profile: Sequence[Any],
    explanation_items_ordered: Sequence[Any],
    relevance: Dict[Any, float],
    cutoff_k: Optional[int] = None,
    K_e: int = 1
) -> float:
    """
    CDCG@K_e:
    Counterfactual NDCG loss after removing the top-K_e explanation items.

    rank_fn(profile) -> ranked list of item_ids (best first).
    user_profile: sequence of items describing the user.
    explanation_items_ordered: items sorted by explanation importance.
    relevance: dict item_id -> relevance (from ground truth, e.g., rating/click).
    cutoff_k: NDCG cutoff (if None, use full ranking).
    K_e: number of explanation items to remove.
    """
    baseline_ranking = rank_fn(user_profile)
    if len(baseline_ranking) == 0:
        return 0.0

    ndcg_base = ndcg_at_k(baseline_ranking, relevance, cutoff_k)

    items_to_remove = explanation_items_ordered[:K_e]
    cf_profile = remove_items_from_profile(user_profile, items_to_remove)
    cf_ranking = rank_fn(cf_profile)

    ndcg_cf = ndcg_at_k(cf_ranking, relevance, cutoff_k)

    # CDCG is the loss in NDCG (positive means worse after perturbation)
    return max(0.0, ndcg_base - ndcg_cf)


# -------------------------------------------------------------------
# 3) INS@K_e
# -------------------------------------------------------------------

def ins_at_ke(
    score_fn: Callable[[Sequence[Any], Any], float],
    user_profile: Sequence[Any],
    target_items: Sequence[Any],
    explanation_items_ordered: Sequence[Any],
    K_e: int
) -> Dict[Any, float]:
    """
    INS@K_e:
    Score changes for target_items when inserting the top-K_e explanation
    items into the user profile.

    score_fn(profile, item) -> real-valued score for 'item'.
    user_profile: current user representation.
    target_items: list of items whose scores will be tracked.
    explanation_items_ordered: items that can be inserted, ordered by importance.
    K_e: number of items to insert.
    Returns:
        dict item_id -> (score_with_insertion - baseline_score)
    """
    # Baseline scores
    baseline_scores = {
        item: score_fn(user_profile, item)
        for item in target_items
    }

    # Add top-K_e items
    items_to_add = explanation_items_ordered[:K_e]
    cf_profile = add_items_to_profile(user_profile, items_to_add)

    cf_scores = {
        item: score_fn(cf_profile, item)
        for item in target_items
    }

    deltas = {
        item: cf_scores[item] - baseline_scores[item]
        for item in target_items
    }

    return deltas


# -------------------------------------------------------------------
# 4) DEL@K_e
# -------------------------------------------------------------------

def del_at_ke(
    score_fn: Callable[[Sequence[Any], Any], float],
    user_profile: Sequence[Any],
    target_items: Sequence[Any],
    explanation_items_ordered: Sequence[Any],
    K_e: int
) -> Dict[Any, float]:
    """
    DEL@K_e:
    Score changes for target_items when removing the top-K_e explanation
    items from the user profile.

    score_fn(profile, item) -> real-valued score for 'item'.
    user_profile: current user representation.
    target_items: list of items whose scores will be tracked.
    explanation_items_ordered: items to remove, ordered by importance.
    K_e: number of items to remove.
    Returns:
        dict item_id -> (score_after_removal - baseline_score)
    """
    # Baseline scores
    baseline_scores = {
        item: score_fn(user_profile, item)
        for item in target_items
    }

    items_to_remove = explanation_items_ordered[:K_e]
    cf_profile = remove_items_from_profile(user_profile, items_to_remove)

    cf_scores = {
        item: score_fn(cf_profile, item)
        for item in target_items
    }

    deltas = {
        item: cf_scores[item] - baseline_scores[item]
        for item in target_items
    }

    return deltas
