"""
Funções de avaliação extraídas do eval_movie.py
Sem dependências de Neo4j para uso standalone
"""

import re
import numpy as np
from difflib import SequenceMatcher


def is_similar(name1, name2, threshold=0.8):
    """
    Verifica se duas strings são similares
    """
    similarity = SequenceMatcher(None, name1, name2).ratio()
    return similarity >= threshold


def recall(recommended_items, actual_items):
    """
    Calcula Recall: quantos dos itens corretos foram recomendados
    """
    hits = sum(1 for item in actual_items if item in recommended_items)
    return hits / len(actual_items) if actual_items else 0


def precision(recommended_items, actual_items):
    """
    Calcula Precision: proporção de itens recomendados que estão corretos
    """
    hits = sum(1 for item in actual_items if item in recommended_items)
    return hits / len(recommended_items) if recommended_items else 0


def ndcg(prediction, ground_truth):
    """
    Calcula NDCG (Normalized Discounted Cumulative Gain)
    """
    def dcg(relevance_scores):
        return np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])

    def idcg(ground_truth):
        relevance_scores = [1 if item in ground_truth else 0 for item in ground_truth]
        return dcg(relevance_scores)
    
    relevance_scores = [1 if movie in ground_truth else 0 for movie in prediction]
    dcg_value = dcg(relevance_scores)
    idcg_value = idcg(ground_truth)
    
    if idcg_value == 0:
        return 0.0
    
    return dcg_value / idcg_value


def ftr(response):
    """
    Failed to Recommend: verifica se o modelo não retornou no formato correto
    """
    if '[SEP]' not in response:
        return True
    else:
        return False


def get_predicted_movie_titles(prediction_response):
    """
    Extrai títulos de filmes da resposta do modelo
    """
    predicted_movie_titles = prediction_response.split("[SEP]")
    predicted_movie_titles = [movie.strip().strip('\'').strip('"') for movie in predicted_movie_titles]
    # Remove ano de cada título
    predicted_movie_titles = [re.sub(r'\s\(\d{4}\)$', '', movie.strip()).strip() for movie in predicted_movie_titles]
    return predicted_movie_titles


def preprocess_matching(recommended_items, actual_items, threshold=0.8):
    """
    Faz matching fuzzy entre itens recomendados e corretos
    """
    new_recommended_items = []
    for recommended_item in recommended_items:
        for actual_item in actual_items:
            if is_similar(recommended_item, actual_item, threshold):
                new_recommended_items.append(actual_item)
                break
        else:
            new_recommended_items.append(recommended_item)
    return new_recommended_items
