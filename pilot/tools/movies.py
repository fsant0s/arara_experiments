from typing import List, Dict, Any, Optional
import neo4j_client
from neo4j_client import connect_to_neo4j, NEO4J_DATABASE, close_connection


def _execute_query(query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Execute a Neo4j query and return results as a list of dictionaries.
    
    Args:
        query: Cypher query to execute
        params: Query parameters
        
    Returns:
        List of records as dictionaries, empty list if error occurs
    """
    # Verifica se o driver existe, caso contrário conecta
    if neo4j_client.driver is None:
        if not connect_to_neo4j():
            return []
    
    try:
        with neo4j_client.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]
    except Exception as e:
        print(f"Query error: {e}")
        return []

def get_existing_relations() -> List[str]:
    """
    Get all relationship types available in the database.
    
    Returns:
        Sorted list of relationship type names
    """
    query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
    results = _execute_query(query)
    return [r["relationshipType"] for r in results]

def get_existing_nodes() -> List[str]:
    """
    Get all node labels available in the database.
    
    Returns:
        Sorted list of node label names
    """
    query = "CALL db.labels() YIELD label RETURN label ORDER BY label"
    results = _execute_query(query)
    return [r["label"] for r in results]

def get_movies_by_relation(relation: str, target_name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get movies connected to a target node via a specific relationship.
    
    Args:
        relation: Relationship type (e.g., 'Genre', 'Directed_by', 'Starring')
        target_name: Name of the target node
        limit: Maximum number of results to return
        
    Returns:
        List of movies with movieId, title, and release_date
    """
    query = f"""
    MATCH (m:Movie)-[:{relation}]->(t {{name: $target_name}})
    RETURN m.Title as title, m.release_date as release_date
    ORDER BY m.release_date DESC
    LIMIT $limit
    """
    return _execute_query(query, {"target_name": target_name, "limit": limit})

def get_movies_by_genre(genre: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get movies filtered by genre.
    
    Args:
        genre: Genre name
        limit: Maximum number of results
        
    Returns:
        List of movies matching the genre
    """
    return get_movies_by_relation("Genre", genre, limit)

def get_movies_by_director(director_name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get movies directed by a specific director.
    
    Args:
        director_name: Director's name
        limit: Maximum number of results
        
    Returns:
        List of movies directed by the director
    """
    return get_movies_by_relation("Directed_by", director_name, limit)

def get_movies_by_actor(actor_name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get movies starring a specific actor.
    
    Args:
        actor_name: Actor's name
        limit: Maximum number of results
        
    Returns:
        List of movies starring the actor
    """
    return get_movies_by_relation("Starring", actor_name, limit)

def get_movies_by_language(language: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get movies in a specific language.
    
    Args:
        language: Language name
        limit: Maximum number of results
        
    Returns:
        List of movies in the specified language
    """
    return get_movies_by_relation("Language", language, limit)

def get_movies_by_production_company(company: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get movies produced by a specific company.
    
    Args:
        company: Production company name
        limit: Maximum number of results
        
    Returns:
        List of movies produced by the company
    """
    return get_movies_by_relation("Produced_by", company, limit)

def get_movies_by_year(year: int, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get movies released in a specific year.
    
    Args:
        year: Release year
        limit: Maximum number of results
        
    Returns:
        List of movies released in the specified year
    """
    query = """
    MATCH (m:Movie)
    WHERE m.release_date CONTAINS $year
    RETURN m.movieId as movieId, m.Title as title, m.release_date as release_date
    ORDER BY m.release_date DESC
    LIMIT $limit
    """
    return _execute_query(query, {"year": str(year), "limit": limit})

def get_movie_details(movie_id: str) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive details about a specific movie.
    
    Args:
        movie_id: Unique movie identifier
        
    Returns:
        Dictionary containing movie data and related entities (directors, actors, genres, etc.),
        or None if movie not found
    """
    # Verifica se o driver existe, caso contrário conecta
    if neo4j_client.driver is None:
        if not connect_to_neo4j():
            return None
    
    query = """
    MATCH (m:Movie {movieId: $movie_id})
    OPTIONAL MATCH (m)-[:Directed_by]->(d)
    OPTIONAL MATCH (m)-[:Starring]->(a)
    OPTIONAL MATCH (m)-[:Genre]->(g)
    OPTIONAL MATCH (m)-[:Language]->(l)
    OPTIONAL MATCH (m)-[:Produced_by]->(p)
    RETURN m, 
           collect(DISTINCT d.name) as directors,
           collect(DISTINCT a.name) as actors,
           collect(DISTINCT g.name) as genres,
           collect(DISTINCT l.name) as languages,
           collect(DISTINCT p.name) as producers
    """
    
    try:
        with neo4j_client.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, movie_id=movie_id)
            record = result.single()
            if record:
                return {
                    "movie": dict(record["m"]),
                    "directors": record["directors"],
                    "actors": record["actors"],
                    "genres": record["genres"],
                    "languages": record["languages"],
                    "producers": record["producers"]
                }
            return None
    except Exception as e:
        print(f"Query error: {e}")
        return None

def search_movies_by_title(title_query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search for movies by title using case-insensitive partial matching.
    
    Args:
        title_query: Search term to match against movie titles
        limit: Maximum number of results
        
    Returns:
        List of movies matching the search query
    """
    query = """
    MATCH (m:Movie)
    WHERE toLower(m.Title) CONTAINS toLower($title_query)
    RETURN m.movieId as movieId, m.Title as title, m.release_date as release_date
    ORDER BY m.Title
    LIMIT $limit
    """
    return _execute_query(query, {"title_query": title_query, "limit": limit})

def get_available_genres() -> List[str]:
    """
    Get all available genres in the database.
    
    Returns:
        Sorted list of genre names
    """
    query = """
    MATCH (g:Genre)
    RETURN g.name as genre
    ORDER BY g.name
    """
    results = _execute_query(query)
    return [r["genre"] for r in results]

def get_available_languages() -> List[str]:
    """
    Get all available languages in the database.
    
    Returns:
        Sorted list of language names
    """
    query = """
    MATCH (l)-[:Language]-(:Movie)
    RETURN DISTINCT l.name as language
    ORDER BY l.name
    """
    results = _execute_query(query)
    return [r["language"] for r in results]

def explore_database_schema() -> Dict[str, Any]:
    """
    Get comprehensive information about the database schema.
    
    Returns:
        Dictionary containing nodes, relations, genres, and languages
    """
    return {
        "nodes": get_existing_nodes(),
        "relations": get_existing_relations(),
        "available_genres": get_available_genres(),
        "available_languages": get_available_languages()
    }

tools = [
    # get_existing_relations,
    # get_existing_nodes,
    # get_available_genres,
    # get_available_languages,
    # explore_database_schema,
    # get_movies_by_relation,
    # get_movies_by_genre,
    get_movies_by_director,
    get_movies_by_actor,
    # get_movies_by_language,
    get_movies_by_production_company,
    # get_movies_by_year,
    # get_movie_details,
    # search_movies_by_title,
]