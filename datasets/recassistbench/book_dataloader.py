import json
import os
from typing import List, Dict, Any, Optional

from .eval.eval_book import (
    recall,
    precision,
    ndcg,
    ftr,
    get_predicted_book_titles,
    preprocess_matching
)


class Dataloader:

    def __init__(self, dataset: str):
        # Base directory relative to this file so imports work from any CWD
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(base_dir, "dataset") + "/"
        self.dataset = os.path.join(self.path, dataset)
        self._data = None

    def load(
        self, 
        source_user: Optional[str] = None, 
        condition_num: Optional[int] = None,
        book_count: Optional[int] = None,
        min_books: Optional[int] = None,
        max_books: Optional[int] = None,
        data_idx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Carrega o dataset do caminho especificado com filtros opcionais.
        
        Args:
            source_user: Filtrar por usuário específico
            condition_num: Filtrar por número de condição
            book_count: Filtrar por quantidade exata de livros
            min_books: Filtrar por quantidade mínima de livros
            max_books: Filtrar por quantidade máxima de livros
            data_idx: Filtrar por índice específico
            
        Returns:
            Lista de dicionários com os dados filtrados
        """
        # Carrega dados se ainda não foi carregado
        if self._data is None:
            with open(self.dataset, 'r') as file:
                self._data = json.load(file)
        
        # Começa com todos os dados
        filtered_data = self._data
        
        # Aplica filtros sequencialmente
        if source_user is not None:
            filtered_data = [item for item in filtered_data if item.get('source_user') == source_user]
        
        if condition_num is not None:
            filtered_data = [item for item in filtered_data if item.get('condition_num') == condition_num]
        
        if book_count is not None:
            filtered_data = [item for item in filtered_data if item.get('bookCount') == book_count]
        
        if min_books is not None:
            filtered_data = [item for item in filtered_data if item.get('bookCount', 0) >= min_books]
        
        if max_books is not None:
            filtered_data = [item for item in filtered_data if item.get('bookCount', 0) <= max_books]
        
        if data_idx is not None:
            filtered_data = [item for item in filtered_data if item.get('data_idx') == data_idx]
        
        return filtered_data
    
    def get_users(self) -> List[str]:
        """Retorna lista de todos os usuários únicos no dataset."""
        if self._data is None:
            self.load()
        return sorted(set(item.get('source_user') for item in self._data if 'source_user' in item))
    
    def get_conditions(self) -> List[int]:
        """Retorna lista de todas as condições únicas no dataset."""
        if self._data is None:
            self.load()
        return sorted(set(item.get('condition_num') for item in self._data if 'condition_num' in item))
    
    def count(self, **filters) -> int:
        """Retorna a contagem de itens com os filtros aplicados."""
        return len(self.load(**filters))

    def load_predictions(self, model_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Carrega todas as predições de um modelo específico.
        
        Args:
            model_name: Nome do modelo (ex: "llama-3.1-70b-instruct")
            
        Returns:
            Dicionário com data_idx como chave e predição como valor
        """
        # Extrai domain e query_type (ex: "book/ExplicitQuery.json" -> "book-ExplicitQuery")
        dataset_parts = self.dataset.replace(self.path, '').replace('.json', '').split('/')
        base_filename = f"{dataset_parts[0]}-{dataset_parts[1]}"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path_results = os.path.join(base_dir, "llm_results", model_name, f"{base_filename}_{model_name}-prediction.jsonl")
        
        predictions = {}
        with open(path_results, 'r') as file:
            for line in file:
                pred = json.loads(line.strip())
                predictions[pred['id']] = pred
        
        return predictions
    
    def get_result(self, data_idx: int, model_name: str) -> Optional[Dict[str, Any]]:
        predictions = self.load_predictions(model_name)
        if not isinstance(predictions, dict):
            raise TypeError(f"Predictions must be a dict, got {type(predictions)}")

        key_int = data_idx
        key_str = str(data_idx)

        if key_int in predictions:
            return predictions[key_int]
        elif key_str in predictions:
            return predictions[key_str]
        return None
    
    def get_eval_result(self, data_idx: int, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retorna o resultado da avaliação de um modelo para um data_idx específico.
        
        Args:
            data_idx: Índice do item no dataset
            model_name: Nome do modelo
            
        Returns:
            Dicionário com métricas de avaliação, ou None se não encontrado
        """
        # Extrai domain e query_type (ex: "book/ExplicitQuery.json" -> "book-ExplicitQuery")
        dataset_parts = self.dataset.replace(self.path, '').replace('.json', '').split('/')
        base_filename = f"{dataset_parts[0]}-{dataset_parts[1]}"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path_eval = os.path.join(base_dir, "eval_results", f"{base_filename}_{model_name}-prediction.json")
        
        with open(path_eval, 'r') as file:
            eval_results = json.load(file)
        
        for result in eval_results:
            if str(result.get('id')) == str(data_idx):
                return result
        
        return None
    
    def evaluate_response(self, response: str, data_idx: int, k: int = 0) -> Dict[str, Any]:
        """
        Avalia uma resposta em tempo real comparando com o groundtruth.
        
        Args:
            response: Resposta do modelo (livros separados por [SEP])
            data_idx: Índice do item no dataset para pegar o groundtruth
            k: Número de recomendações para avaliar (0 = todas)
            
        Returns:
            Dicionário com métricas: recall, precision, ndcg, ftr
        """
        # Busca o groundtruth
        groundtruth_data = self.load(data_idx=data_idx)
        if not groundtruth_data:
            raise ValueError(f"Groundtruth não encontrado para data_idx={data_idx}")
        
        groundtruth = groundtruth_data[0]
        
        # Calcula FTR
        results = {'ftr': ftr(response)}
        
        # Extrai títulos preditos
        predicted_titles = get_predicted_book_titles(response)
        
        # Aplica limite k se especificado
        if k != 0:
            predicted_titles = predicted_titles[:k]
            if len(predicted_titles) < k:
                predicted_titles += ['none'] * (k - len(predicted_titles))
        
        # Faz matching fuzzy
        matched_titles = preprocess_matching(predicted_titles, groundtruth['bookSubset'])
        
        # Calcula métricas
        results["recall"] = recall(matched_titles, groundtruth['bookSubset'])
        results["precision"] = precision(matched_titles, groundtruth['bookSubset'])
        results["ndcg"] = ndcg(matched_titles, groundtruth['bookSubset'])
        results["predicted_titles"] = predicted_titles
        results["matched_titles"] = matched_titles
        results["ground_truth"] = groundtruth['bookSubset']
        
        return results