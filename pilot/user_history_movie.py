from typing import List, Dict, Any
import os
import random


def get_data_path(filename: str) -> str:
    """Returns the correct path for the data files."""
    if os.path.exists("datasets/recassistbench/dataset/movie/"):
        base_path = "datasets/recassistbench/dataset/movie/"
    else:
        base_path = "../datasets/recassistbench/dataset/movie/"
    return os.path.join(base_path, filename)


def get_user_history(user_id: int) -> List[Dict[str, Any]]:
    """Fetches the rating history of a specific user."""
    ratings_path = get_data_path("ratings.dat")
    history = []
    
    try:
        with open(ratings_path, 'r') as file:
            for line in file:
                if line.startswith(f"{user_id}::"):
                    parts = line.strip().split('::')
                    if len(parts) >= 3:
                        history.append({
                            "movie_id": int(parts[1]),
                            "rating": float(parts[2]),
                            "timestamp": int(parts[3]) if len(parts) > 3 else None
                        })
    except FileNotFoundError:
        print(f"❌ File not found: {ratings_path}")
        return []
    
    return history


def get_movie_info(movie_id: int) -> Dict[str, Any]:
    """Fetches information for a specific movie."""
    movies_path = get_data_path("movies.dat")
    
    try:
        with open(movies_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('::')
                if len(parts) >= 3 and int(parts[0]) == movie_id:
                    return {
                        "movie_id": int(parts[0]),
                        "title": parts[1],
                        "genres": parts[2].split('|') if len(parts) > 2 else []
                    }
    except FileNotFoundError:
        print(f"❌ File not found: {movies_path}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
    
    return {}


def get_user_movie_titles(user_id: int) -> List[str]:
    """Returns only the titles of the movies the user has rated."""
    history = get_user_history(user_id)
    titles = []
    
    for item in history:
        movie_info = get_movie_info(item["movie_id"])
        title = movie_info.get("title", "Unknown")
        if title != "Unknown":
            titles.append(title)
    
    return titles


def get_filtered_user_history(user_id: int, groundtruth_movie_ids: List[int] = None, neo4j_conditions: List[List[str]] = None, percentage: float = 1.0) -> List[str]:
    """Returns the user's history filtered with a random percentage."""
    # Imports the Neo4j client
    try:
        from .neo4j_client import connect_to_neo4j, get_items_from_query, close_connection
    except ImportError:
        try:
            from neo4j_client import connect_to_neo4j, get_items_from_query, close_connection
        except ImportError:
            neo4j_conditions = None
    
    # Fetches the user's complete history
    history = get_user_history(user_id)
    user_movie_titles = []
    
    # Convert to titles
    for item in history:
        movie_info = get_movie_info(item["movie_id"])
        title = movie_info.get("title", "Unknown")
        if title != "Unknown":
            user_movie_titles.append(title)
    
    # Remove ground truth movies
    if groundtruth_movie_ids:
        groundtruth_titles = []
        for movie_id in groundtruth_movie_ids:
            movie_info = get_movie_info(movie_id)
            title = movie_info.get("title", "Unknown")
            if title != "Unknown":
                groundtruth_titles.append(title)
        
        user_movie_titles = [title for title in user_movie_titles if title not in groundtruth_titles]
    
    # Remove movies that match Neo4j conditions
    if neo4j_conditions:
        try:
            if connect_to_neo4j():
                neo4j_titles = get_items_from_query(neo4j_conditions)
                user_movie_titles = [title for title in user_movie_titles if title not in neo4j_titles]
        except Exception as e:
            print(f"❌ Error connecting to Neo4j: {e}")
    
    # Apply random percentage
    if percentage < 1.0:
        sample_size = int(len(user_movie_titles) * percentage)
        user_movie_titles = random.sample(user_movie_titles, min(sample_size, len(user_movie_titles)))
    
    return user_movie_titles


def get_user_history_with_movies(user_id: int) -> List[Dict[str, Any]]:
    """Fetches a user's history with movie information."""
    history = get_user_history(user_id)
    enriched_history = []
    
    for item in history:
        movie_info = get_movie_info(item["movie_id"])
        enriched_item = {
            **item,
            "title": movie_info.get("title", "Unknown"),
            "genres": movie_info.get("genres", [])
        }
        enriched_history.append(enriched_item)
    
    return enriched_history
    
if __name__ == "__main__":
    # Simple test
    user_id = 1
    print(f"🔍 Testing user {user_id}")
    
    # Test with percentage
    filtered_titles = get_filtered_user_history(
        user_id=user_id,
        groundtruth_movie_ids=[1193, 661],
        neo4j_conditions=[["Genre", "Animation"]],
        percentage=0.3
    )
    
    print(f"   {len(filtered_titles)} movies (30% of history):")
    for i, title in enumerate(filtered_titles[:5], 1):
        print(f"   {i}. {title}")
