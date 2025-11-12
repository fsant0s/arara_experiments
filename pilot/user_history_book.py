from typing import List, Dict, Any
import os
import random
import json


def get_data_path(filename: str) -> str:
    """Returns the correct path for the book data files."""
    if os.path.exists("datasets/recassistbench/dataset/book/"):
        base_path = "datasets/recassistbench/dataset/book/"
    else:
        base_path = "../datasets/recassistbench/dataset/book/"
    return os.path.join(base_path, filename)


def get_user_history(user_id: str) -> List[Dict[str, Any]]:
    """Fetches the rating history of a specific user for books."""
    ratings_path = get_data_path("ratings.dat")
    history = []
    
    try:
        with open(ratings_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[0] == user_id:
                    history.append({
                        "book_title": parts[1],
                        "rating": float(parts[2])
                    })
    except FileNotFoundError:
        print(f"❌ File not found: {ratings_path}")
        return []
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []
    
    return history


def get_book_info_from_jsonl(book_title: str) -> Dict[str, Any]:
    """Fetches information for a specific book from book_info.jsonl."""
    books_path = get_data_path("book_info.jsonl")
    
    try:
        with open(books_path, 'r', encoding='utf-8') as file:
            for line in file:
                book_data = json.loads(line.strip())
                if book_data.get("Title") == book_title:
                    return book_data
    except FileNotFoundError:
        print(f"❌ File not found: {books_path}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
    
    return {}


def get_user_book_titles(user_id: str) -> List[str]:
    """Returns only the titles of the books the user has rated."""
    history = get_user_history(user_id)
    return [item["book_title"] for item in history]


def get_filtered_user_history(user_id: str, groundtruth_book_ids: List[str] = None, neo4j_conditions: List[List[str]] = None, percentage: float = 1.0) -> List[str]:
    """Returns the user's history filtered with a random percentage."""
    # Imports the Neo4j client for books
    books_module = None
    try:
        import sys
        import os
        sys.path.append('.')
        sys.path.append('./pilot')
        
        # Import directly without going through __init__.py
        import importlib.util
        spec = importlib.util.spec_from_file_location('books', './pilot/tools/books.py')
        books_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(books_module)
        
        # Tests whether it can connect to Neo4j
        test_result = books_module.get_existing_nodes()
        if not test_result:  # If it returns an empty list, it may be a connection error
            books_module = None
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to Neo4j: {e}")
        books_module = None
    
    # Fetches the user's complete history
    history = get_user_history(user_id)
    user_book_titles = [item["book_title"] for item in history]
    
    # Remove books from the ground truth
    if groundtruth_book_ids:
        user_book_titles = [title for title in user_book_titles if title not in groundtruth_book_ids]
    
    # Remove books from the Neo4j conditions (only if connection succeeded)
    if neo4j_conditions and books_module is not None:
        try:
            neo4j_titles = []
            for relation, value in neo4j_conditions:
                if relation == "WRITTEN_BY":
                    titles = books_module.get_books_by_author(value, limit=200)
                elif relation == "BELONGS_TO":
                    titles = books_module.get_books_by_category(value, limit=200)
                else:
                    titles = books_module.get_books_by_relation(relation, value, limit=200)
                neo4j_titles.extend(titles)
            
            user_book_titles = [title for title in user_book_titles if title not in neo4j_titles]
        except Exception as e:
            print(f"⚠️ Warning: Error applying Neo4j filters: {e}")
    
    # Applies random percentage=1.0
    if percentage < 1.0:
        sample_size = int(len(user_book_titles) * percentage)
        user_book_titles = random.sample(user_book_titles, min(sample_size, len(user_book_titles)))
    
    return user_book_titles


def get_user_history_with_books(user_id: str) -> List[Dict[str, Any]]:
    """Fetches the user's history with book information."""
    history = get_user_history(user_id)
    enriched_history = []
    
    for item in history:
        book_info = get_book_info_from_jsonl(item["book_title"])
        enriched_item = {
            **item,
            "author": book_info.get("Author", ""),
            "category": book_info.get("Category", ""),
            "genre": book_info.get("Genre", "")
        }
        enriched_history.append(enriched_item)
    
    return enriched_history


def get_user_ratings_summary(user_id: str) -> Dict[str, Any]:
    """Returns a summary of the user's ratings."""
    history = get_user_history(user_id)
    
    if not history:
        return {"total_books": 0, "average_rating": 0.0, "ratings_distribution": {}}
    
    ratings = [item["rating"] for item in history]
    ratings_distribution = {}
    
    for rating in ratings:
        ratings_distribution[rating] = ratings_distribution.get(rating, 0) + 1
    
    return {
        "total_books": len(history),
        "average_rating": sum(ratings) / len(ratings),
        "ratings_distribution": ratings_distribution
    }


if __name__ == "__main__":
    # Simple test
    user_id = "A30TK6U7DNS82R"
    print(f"🔍 Testing user {user_id}")
    
    # Basic test
    history = get_user_history(user_id)
    print(f"   {len(history)} rated books:")
    for i, item in enumerate(history[:5], 1):
        print(f"   {i}. {item['book_title']} - {item['rating']}")
    
    print()
    
    # Test with filters
    filtered_titles = get_filtered_user_history(
        user_id=user_id,
        groundtruth_book_ids=["Dr. Seuss: American Icon"],
        neo4j_conditions=[["WRITTEN_BY", "Stephen King"]],
        percentage=0.5
    )
    
    print(f"   {len(filtered_titles)} books after filters (50% of history):")
    for i, title in enumerate(filtered_titles[:5], 1):
        print(f"   {i}. {title}")
    
    print()
    
    # Ratings summary
    summary = get_user_ratings_summary(user_id)
    print(f"📊 Ratings summary:")
    print(f"   Total books: {summary['total_books']}")
    print(f"   Average rating: {summary['average_rating']:.2f}")
    print(f"   Distribution: {summary['ratings_distribution']}")
