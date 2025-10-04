import json
from typing import List, Dict, Any, Optional

# Importa funções de avaliação do módulo local
try:
    # Tenta import quando executado como módulo
    from .eval_utils import (
        recall,
        precision,
        ndcg,
        ftr,
        get_predicted_movie_titles,
        preprocess_matching
    )
except ImportError:
    # Import quando executado diretamente
    from eval_utils import (
        recall,
        precision,
        ndcg,
        ftr,
        get_predicted_movie_titles,
        preprocess_matching
    )


class Dataloader:

    def __init__(self, dataset: str):
        self.path = "datasets/RecAssistBench/dataset/"
        self.dataset = self.path + dataset
        self._data = None

    def load(
        self, 
        source_user: Optional[str] = None, 
        condition_num: Optional[int] = None,
        movie_count: Optional[int] = None,
        min_movies: Optional[int] = None,
        max_movies: Optional[int] = None,
        data_idx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Carrega o dataset do caminho especificado com filtros opcionais.
        
        Args:
            source_user: Filtrar por usuário específico
            condition_num: Filtrar por número de condição
            movie_count: Filtrar por quantidade exata de filmes
            min_movies: Filtrar por quantidade mínima de filmes
            max_movies: Filtrar por quantidade máxima de filmes
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
        
        if movie_count is not None:
            filtered_data = [item for item in filtered_data if item.get('movieCount') == movie_count]
        
        if min_movies is not None:
            filtered_data = [item for item in filtered_data if item.get('movieCount', 0) >= min_movies]
        
        if max_movies is not None:
            filtered_data = [item for item in filtered_data if item.get('movieCount', 0) <= max_movies]
        
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
        # Extrai domain e query_type (ex: "movie/ExplicitQuery.json" -> "movie-ExplicitQuery")
        dataset_parts = self.dataset.replace(self.path, '').replace('.json', '').split('/')
        base_filename = f"{dataset_parts[0]}-{dataset_parts[1]}"
        path_results = f"{self.path}../llm_results/{model_name}/{base_filename}_{model_name}-prediction.jsonl"
        
        predictions = {}
        with open(path_results, 'r') as file:
            for line in file:
                pred = json.loads(line.strip())
                predictions[pred['id']] = pred
        
        return predictions
    
    def get_result(self, data_idx: int, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retorna o resultado de um modelo para um data_idx específico.
        
        Args:
            data_idx: Índice do item no dataset
            model_name: Nome do modelo
            
        Returns:
            Dicionário com 'id' e 'response', ou None se não encontrado
        """
        predictions = self.load_predictions(model_name)
        return predictions.get(str(data_idx))
    
    def get_eval_result(self, data_idx: int, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retorna o resultado da avaliação de um modelo para um data_idx específico.
        
        Args:
            data_idx: Índice do item no dataset
            model_name: Nome do modelo
            
        Returns:
            Dicionário com métricas de avaliação, ou None se não encontrado
        """
        # Extrai domain e query_type (ex: "movie/ExplicitQuery.json" -> "movie-ExplicitQuery")
        dataset_parts = self.dataset.replace(self.path, '').replace('.json', '').split('/')
        base_filename = f"{dataset_parts[0]}-{dataset_parts[1]}"
        path_eval = f"{self.path}../eval_results/{base_filename}_{model_name}-prediction.json"
        
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
            response: Resposta do modelo (filmes separados por [SEP])
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
        predicted_titles = get_predicted_movie_titles(response)
        
        # Aplica limite k se especificado
        if k != 0:
            predicted_titles = predicted_titles[:k]
            if len(predicted_titles) < k:
                predicted_titles += ['none'] * (k - len(predicted_titles))
        
        # Faz matching fuzzy
        matched_titles = preprocess_matching(predicted_titles, groundtruth['movieSubset'])
        
        # Calcula métricas
        results["recall"] = recall(matched_titles, groundtruth['movieSubset'])
        results["precision"] = precision(matched_titles, groundtruth['movieSubset'])
        results["ndcg"] = ndcg(matched_titles, groundtruth['movieSubset'])
        results["predicted_titles"] = predicted_titles
        results["matched_titles"] = matched_titles
        results["ground_truth"] = groundtruth['movieSubset']
        
        return results


if __name__ == "__main__":
    # Exemplo de uso
    dataloader = Dataloader("movie/ExplicitQuery.json")
    
    # Pega um item do dataset
    dataset = dataloader.load()
    data = dataset[0]
    
    print("=" * 80)
    print("=== Groundtruth ===")
    print(f"Query: {data['direct_description_query']}")
    print(f"Filmes corretos: {data['movieSubset']}")
    
    # ===== Avalia modelo existente =====
    model_name = "llama-3.1-70b-instruct"
    prediction = dataloader.get_result(data_idx=data['data_idx'], model_name=model_name)
    
    print(f"\n{'=' * 80}")
    print(f"=== Predição do {model_name} ===")
    if prediction:
        print(f"Response: {prediction['response']}")
        
        # Pega resultado da avaliação pré-calculada
        eval_result = dataloader.get_eval_result(data_idx=data['data_idx'], model_name=model_name)
        
        print(f"\n=== Métricas (pré-calculadas) ===")
        if eval_result:
            print(f"Recall:          {eval_result.get('recall', 'N/A'):.3f}")
            print(f"Precision:       {eval_result.get('precision', 'N/A'):.3f}")
            print(f"NDCG:            {eval_result.get('ndcg', 'N/A'):.3f}")
            print(f"Satisfied Ratio: {eval_result.get('satisfied_ratio', 'N/A'):.3f}")

    # ===== Avalia resposta do Arara em tempo real =====
    arara_response = "Do the Right Thing [SEP] Malcolm X [SEP] 25th Hour [SEP] She's Gotta Have It [SEP] Mo' Better Blues [SEP] Crooklyn"
    
    print(f"\n{'=' * 80}")
    print(f"=== Predição do Arara (tempo real) ===")
    print(f"Response: {arara_response}")
    
    # Avalia em tempo real
    arara_eval = dataloader.evaluate_response(arara_response, data_idx=data['data_idx'])
    
    print(f"FTR:       {arara_eval['ftr']}")
    print(f"Recall:    {arara_eval['recall']:.3f}")
    print(f"Precision: {arara_eval['precision']:.3f}")
    print(f"NDCG:      {arara_eval['ndcg']:.3f}")
    print(f"\nFilmes preditos: {arara_eval['predicted_titles']}")
    print(f"Filmes matched:  {arara_eval['matched_titles']}")
    print(f"Ground truth:    {arara_eval['ground_truth']}")
    print("=" * 80)