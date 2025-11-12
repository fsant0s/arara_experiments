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
        Load the dataset from the specified path with optional filters.

        Args:
            source_user: Filter by a specific user
            condition_num: Filter by condition number
            book_count: Filter by an exact number of books
            min_books: Filter by minimum number of books
            max_books: Filter by maximum number of books
            data_idx: Filter by a specific index

        Returns:
            List of dictionaries containing the filtered data
        """
        # Load data if not already loaded
        if self._data is None:
            with open(self.dataset, 'r') as file:
                self._data = json.load(file)
        
        # Start with all data
        filtered_data = self._data
        
        # Apply filters sequentially
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
        """Returns a list of all unique users in the dataset."""
        if self._data is None:
            self.load()
        return sorted(set(item.get('source_user') for item in self._data if 'source_user' in item))
    
    def get_conditions(self) -> List[int]:
        """Returns a list of all unique conditions in the dataset."""
        if self._data is None:
            self.load()
        return sorted(set(item.get('condition_num') for item in self._data if 'condition_num' in item))
    
    def count(self, **filters) -> int:
        """Returns the count of items after applying filters."""
        return len(self.load(**filters))

    def load_predictions(self, model_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Loads all predictions from a specific model.
        
        Args:
            model_name: Model name (e.g., "llama-3.1-70b-instruct")
            
        Returns:
            Dictionary with data_idx as key and prediction as value
        """
        # Extract domain and query_type (e.g., "book/ExplicitQuery.json" -> "book-ExplicitQuery")
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
        """Returns a single prediction entry for a given data_idx."""
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
        Returns the evaluation result of a model for a specific data_idx.
        
        Args:
            data_idx: Index of the item in the dataset
            model_name: Model name
            
        Returns:
            Dictionary with evaluation metrics, or None if not found
        """
        # Extract domain and query_type (e.g., "book/ExplicitQuery.json" -> "book-ExplicitQuery")
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
        Evaluates a response in real time by comparing it with the ground truth.
        
        Args:
            response: Model response (books separated by [SEP])
            data_idx: Dataset item index used to retrieve the ground truth
            k: Number of recommendations to evaluate (0 = all)
            
        Returns:
            Dictionary with metrics: recall, precision, ndcg, ftr
        """
        # Fetch ground truth
        groundtruth_data = self.load(data_idx=data_idx)
        if not groundtruth_data:
            raise ValueError(f"Ground truth not found for data_idx={data_idx}")
        
        groundtruth = groundtruth_data[0]
        
        # Compute FTR
        results = {'ftr': ftr(response)}
        
        # Extract predicted titles
        predicted_titles = get_predicted_book_titles(response)
        
        # Apply top-k limit if specified
        if k != 0:
            predicted_titles = predicted_titles[:k]
            if len(predicted_titles) < k:
                predicted_titles += ['none'] * (k - len(predicted_titles))
        
        # Perform fuzzy matching
        matched_titles = preprocess_matching(predicted_titles, groundtruth['bookSubset'])
        
        # Compute metrics
        results["recall"] = recall(matched_titles, groundtruth['bookSubset'])
        results["precision"] = precision(matched_titles, groundtruth['bookSubset'])
        results["ndcg"] = ndcg(matched_titles, groundtruth['bookSubset'])
        results["predicted_titles"] = predicted_titles
        results["matched_titles"] = matched_titles
        results["ground_truth"] = groundtruth['bookSubset']
        
        return results
