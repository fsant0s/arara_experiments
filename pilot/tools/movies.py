from typing import List, Dict, Any, Optional
import re
import neo4j_client
from neo4j_client import connect_to_neo4j, NEO4J_DATABASE

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

def list_nodes_by_type(node_type: str, limit: int = 50) -> List[str]:
    """
    List all nodes of a specific type, filtering out invalid names.
    
    Args:
        node_type: Type of node (e.g., 'Actor', 'Director', 'Genre', 'Language', 'Company')
        limit: Maximum number of results
        
    Returns:
        List of valid node names (excludes names starting with special characters)
    
    Examples:
        - list_nodes_by_type("Actor", 20) -> Returns names of 20 actors
        - list_nodes_by_type("Director") -> Returns names of directors
        - list_nodes_by_type("Company") -> Returns production company names
    """
    # Mapeia tipos e define queries específicas
    node_type_lower = node_type.lower()
    
    # Para atores e diretores, filtra Person nodes conectados a filmes
    if node_type_lower in ["actor", "actors"]:
        query = """
        MATCH (p:Person)<-[:Starring]-(m:Movie)
        WHERE p.name IS NOT NULL 
          AND NOT p.name STARTS WITH '('
          AND NOT p.name STARTS WITH '"'
          AND NOT p.name STARTS WITH '['
        RETURN DISTINCT p.name as name
        ORDER BY p.name
        LIMIT $limit
        """
    elif node_type_lower in ["director", "directors"]:
        query = """
        MATCH (p:Person)<-[:Directed_by]-(m:Movie)
        WHERE p.name IS NOT NULL 
          AND NOT p.name STARTS WITH '('
          AND NOT p.name STARTS WITH '"'
          AND NOT p.name STARTS WITH '['
          AND NOT p.name =~ '^[0-9]+$'
          AND size(p.name) > 2
        RETURN DISTINCT p.name as name
        ORDER BY p.name
        LIMIT $limit
        """
    elif node_type_lower in ["company", "companies", "corporation"]:
        query = """
        MATCH (c:Corporation)
        WHERE c.name IS NOT NULL 
          AND NOT c.name STARTS WITH '('
          AND NOT c.name STARTS WITH '/'
        RETURN DISTINCT c.name as name
        ORDER BY c.name
        LIMIT $limit
        """
    elif node_type_lower in ["genre", "genres"]:
        query = """
        MATCH (g:Genre)
        WHERE g.name IS NOT NULL
          AND NOT g.name =~ '^[0-9]+$'
        RETURN DISTINCT g.name as name
        ORDER BY g.name
        LIMIT $limit
        """
    elif node_type_lower in ["language", "languages"]:
        query = """
        MATCH (l:Country)
        WHERE l.name IS NOT NULL
        RETURN DISTINCT l.name as name
        ORDER BY l.name
        LIMIT $limit
        """
    else:
        # Query genérica para outros tipos
        query = f"""
        MATCH (n:{node_type})
        WHERE n.name IS NOT NULL
        RETURN DISTINCT n.name as name
        ORDER BY n.name
        LIMIT $limit
        """
    
    results = _execute_query(query, {"limit": limit})
    return [r["name"] for r in results if r.get("name")]

def get_movies_by_relation(relation: str, target_name: str, limit: int = 20) -> List[str]:
    """
    Get movies connected to a target node via a specific relationship.
    
    Args:
        relation: Relationship type (e.g., 'Genre', 'Directed_by', 'Starring')
        target_name: Name of the target node
        limit: Maximum number of results to return
        
    Returns:
        List of movie titles with dates replaced by [SEP]
    """
    query = f"""
    MATCH (m:Movie)-[:{relation}]->(t {{name: $target_name}})
    RETURN m.Title as title
    ORDER BY m.Title
    LIMIT $limit
    """
    results = _execute_query(query, {"target_name": target_name, "limit": limit})
    titles = [r["title"] for r in results]
    return titles

def get_movies_by_genre(genre: str, limit: int = 20) -> List[str]:
    """
    Get movies filtered by genre.
    
    Args:
        genre: Genre name
        limit: Maximum number of results
        
    Returns:
        List of movie titles
    """
    return get_movies_by_relation("Genre", genre, limit)

def get_movies_by_director(director_name: str, limit: int = 20) -> List[str]:
    """
    Get movies directed by a specific director.
    
    Args:
        director_name: Director's name
        limit: Maximum number of results
        
    Returns:
        List of movie titles
    """
    return get_movies_by_relation("Directed_by", director_name, limit)

def get_movies_by_actor(actor_name: str) -> List[str]:
    """
    Get movies starring a specific actor (ex: "Charlie Chaplin").
    
    Args:
        actor_name: Actor's name
        
    Returns:
        List of movie titles
    """
    return get_movies_by_relation("Starring", actor_name, 20)

def get_movies_by_language(language: str, limit: int = 20) -> List[str]:
    """
    Get movies in a specific language.
    
    Args:
        language: Language name
        limit: Maximum number of results
        
    Returns:
        List of movie titles
    """
    return get_movies_by_relation("Language", language, limit)

def get_movies_by_production_company(company: str, limit: int = 20) -> List[str]:
    """
    Get movies produced by a specific company.
    
    Args:
        company: Production company name
        limit: Maximum number of results
        
    Returns:
        List of movie titles
    """
    return get_movies_by_relation("Produced_by", company, limit)

def get_movies_by_year(year: int, limit: int = 20) -> List[str]:
    """
    Get movies released in a specific year.
    
    Args:
        year: Release year
        limit: Maximum number of results
        
    Returns:
        List of movie titles with dates replaced by [SEP]
    """
    query = """
    MATCH (m:Movie)
    WHERE m.release_date CONTAINS $year
    RETURN m.Title as title
    ORDER BY m.Title
    LIMIT $limit
    """
    results = _execute_query(query, {"year": str(year), "limit": limit})
    titles = [r["title"] for r in results]
    return _replace_dates_with_sep(titles)

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

def search_movies_by_title(title_query: str, limit: int = 20) -> List[str]:
    """
    Search for movies by title using case-insensitive partial matching.
    
    Args:
        title_query: Search term to match against movie titles
        limit: Maximum number of results
        
    Returns:
        List of movie titles with dates replaced by [SEP]
    """
    query = """
    MATCH (m:Movie)
    WHERE toLower(m.Title) CONTAINS toLower($title_query)
    RETURN m.Title as title
    ORDER BY m.Title
    LIMIT $limit
    """
    results = _execute_query(query, {"title_query": title_query, "limit": limit})
    titles = [r["title"] for r in results]
    return _replace_dates_with_sep(titles)

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
    get_existing_relations,
    get_existing_nodes,
    list_nodes_by_type,
    get_available_genres,
    get_available_languages,
    explore_database_schema,
    get_movies_by_relation,
    get_movies_by_genre,
    get_movies_by_director,
    get_movies_by_actor,
    get_movies_by_language,
    get_movies_by_production_company,
    get_movies_by_year,
    get_movie_details,
    search_movies_by_title,
]