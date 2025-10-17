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
    return titles

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
    return titles

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

def get_movie_id_by_exact_title(title: str) -> Optional[str]:
    """
    Resolve a Movie node's internal identifier (movieId) from an **exact** title match.

    Args:
        title: The exact title stored in the `Movie.Title` property (case-sensitive equality).

    Returns:
        The `movieId` string if a movie with exactly this Title exists; otherwise None.

    Notes:
        - This performs `MATCH (m:Movie {Title: $title})` and returns the first match.
        - Use this to bridge from a human-facing title to the internal ID required by `get_movie_details`.
    """
    query = "MATCH (m:Movie {Title: $title}) RETURN m.movieId as movieId LIMIT 1"
    res = _execute_query(query, {"title": title})
    return res[0]["movieId"] if res else None


def get_movie_details_by_title(title: str) -> Optional[Dict[str, Any]]:
    """
    Fetch full movie details by **exact** title, internally resolving `movieId` first.

    Args:
        title: The exact movie title (case-sensitive equality on `Movie.Title`).

    Returns:
        A dictionary with:
            - "movie": properties of the Movie node,
            - "directors": list of director names,
            - "actors": list of actor names,
            - "genres": list of genre names,
            - "languages": list of language names,
            - "producers": list of producer names,
        or None if the title is not found.

    Notes:
        - This is a convenience wrapper around `get_movie_id_by_exact_title` + `get_movie_details`.
        - Prefer this when you have a canonical Title and want authoritative credits (e.g., true director).
    """
    movie_id = get_movie_id_by_exact_title(title)
    if not movie_id:
        return None
    return get_movie_details(movie_id)

# tools_wrappers.py (ou dentro do mesmo arquivo do módulo, acima da arquitetura)

from tools import movies

def retrieve_titles_by_condition(relation: str, value: str, limit: int = 200) -> str:
    """
    Deterministically retrieve movie titles for a (relation, value) CONDITION and
    return them as a single ' [SEP] ' joined line. If no results, return ''.

    Args:
        relation: One of {'Directed_by','Starring','Genre','Language','Produced_by','Year','Music_by'}.
        value:    Canonical person/value name (string). If Year, can be str or int.
        limit:    Upper bound for retrieval (safety cap).

    Returns:
        A single string with titles separated by ' [SEP] ', or '' if empty.

    Notes:
        - Uses your existing tools under the hood (get_movies_by_director, get_movies_by_actor, etc.).
        - Applies deterministic dedup (case/underscore normalization) and sorts A–Z.
    """
    rel = (relation or "").strip()
    val = (value or "").strip()
    titles = []

    try:
        if rel == "Directed_by":
            titles = movies.get_movies_by_director(val, limit=limit)
        elif rel == "Starring":
            titles = movies.get_movies_by_actor(val)
            if limit and len(titles) > limit:
                titles = titles[:limit]
        elif rel == "Genre":
            titles = movies.get_movies_by_genre(val, limit=limit)
        elif rel == "Language":
            titles = movies.get_movies_by_language(val, limit=limit)
        elif rel == "Produced_by":
            titles = movies.get_movies_by_production_company(val, limit=limit)
        elif rel == "Year":
            try:
                year_int = int(val)
            except Exception:
                return ""
            titles = movies.get_movies_by_year(year_int, limit=limit)
        elif rel == "Music_by":
            titles = movies.get_movies_by_relation("Music_by", val, limit=limit)
        else:
            # Fallback genérico para qualquer relação suportada no grafo
            titles = movies.get_movies_by_relation(rel, val, limit=limit)
    except Exception:
        titles = []

    # Dedup determinístico + normalização simples
    seen = set()
    normalized = []
    for t in titles or []:
        key = t.strip().lower().replace("_", " ").strip("'\"")
        if key not in seen:
            seen.add(key)
            normalized.append(t.strip())

    normalized.sort()  # A–Z determinístico
    return " [SEP] ".join(normalized)



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
    get_movie_id_by_exact_title,
    get_movie_details_by_title,
    retrieve_titles_by_condition,
]