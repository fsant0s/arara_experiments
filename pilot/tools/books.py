from typing import List, Dict, Any, Optional
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Usar banco de livros
BOOKS_DATABASE = "books"

def _execute_query(query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Execute a Neo4j query and return results as a list of dictionaries.
    
    Args:
        query: Cypher query to execute
        params: Query parameters
        
    Returns:
        List of records as dictionaries, empty list if error occurs
    """
    try:
        # Importa e conecta ao Neo4j
        from neo4j_client import connect_to_neo4j, driver
        
        # Conecta se necessário
        if driver is None:
            if not connect_to_neo4j():
                return []
        
        # Verifica se o driver ainda é None após tentar conectar
        if driver is None:
            return []
        
        with driver.session(database=BOOKS_DATABASE) as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]
    except Exception as e:
        print(f"Query error: {e}")
        return []

def get_existing_relations() -> List[str]:
    """
    Get all relationship types available in the books database.
    
    Returns:
        Sorted list of relationship type names
    """
    query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
    results = _execute_query(query)
    return [r["relationshipType"] for r in results]

def get_existing_nodes() -> List[str]:
    """
    Get all node labels available in the books database.
    
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
        node_type: Type of node (e.g., 'Author', 'Category', 'Book')
        limit: Maximum number of results
        
    Returns:
        List of valid node names
    
    Examples:
        - list_nodes_by_type("Author", 20) -> Returns names of 20 authors
        - list_nodes_by_type("Category") -> Returns names of categories
        - list_nodes_by_type("Book") -> Returns book titles
    """
    node_type_lower = node_type.lower()
    
    if node_type_lower in ["author", "authors"]:
        query = """
        MATCH (a:Author)
        WHERE a.name IS NOT NULL 
          AND NOT a.name STARTS WITH '('
          AND NOT a.name STARTS WITH '"'
          AND NOT a.name STARTS WITH '['
        RETURN DISTINCT a.name as name
        ORDER BY a.name
        LIMIT $limit
        """
    elif node_type_lower in ["category", "categories"]:
        query = """
        MATCH (c:Category)
        WHERE c.name IS NOT NULL 
          AND NOT c.name STARTS WITH '('
          AND NOT c.name =~ '^[0-9]+$'
        RETURN DISTINCT c.name as name
        ORDER BY c.name
        LIMIT $limit
        """
    elif node_type_lower in ["book", "books"]:
        query = """
        MATCH (b:Book)
        WHERE b.Title IS NOT NULL
        RETURN DISTINCT b.Title as name
        ORDER BY b.Title
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

def get_books_by_relation(relation: str, target_name: str, limit: int = 20) -> List[str]:
    """
    Get books connected to a target node via a specific relationship.
    
    Args:
        relation: Relationship type (e.g., 'WRITTEN_BY', 'BELONGS_TO')
        target_name: Name of the target node
        limit: Maximum number of results to return
        
    Returns:
        List of book titles
    """
    query = f"""
    MATCH (b:Book)-[:{relation}]->(t {{name: $target_name}})
    RETURN b.Title as title
    ORDER BY b.Title
    LIMIT $limit
    """
    results = _execute_query(query, {"target_name": target_name, "limit": limit})
    return [r["title"] for r in results]

def get_books_by_author(author_name: str, limit: int = 20) -> List[str]:
    """
    Get books written by a specific author.
    
    Args:
        author_name: Author's name
        limit: Maximum number of results
        
    Returns:
        List of book titles
    """
    return get_books_by_relation("WRITTEN_BY", author_name, limit)

def get_books_by_category(category: str, limit: int = 20) -> List[str]:
    """
    Get books in a specific category.
    
    Args:
        category: Category name
        limit: Maximum number of results
        
    Returns:
        List of book titles
    """
    return get_books_by_relation("BELONGS_TO", category, limit)

def get_available_categories() -> List[str]:
    """
    Get all available categories in the database.
    
    Returns:
        Sorted list of category names
    """
    query = """
    MATCH (c:Category)
    RETURN c.name as category
    ORDER BY c.name
    """
    results = _execute_query(query)
    return [r["category"] for r in results]

def get_available_authors() -> List[str]:
    """
    Get all available authors in the database.
    
    Returns:
        Sorted list of author names
    """
    query = """
    MATCH (a:Author)
    RETURN a.name as author
    ORDER BY a.name
    """
    results = _execute_query(query)
    return [r["author"] for r in results]

def explore_database_schema() -> Dict[str, Any]:
    """
    Get comprehensive information about the books database schema.
    
    Returns:
        Dictionary containing nodes, relations, categories, and authors
    """
    return {
        "nodes": get_existing_nodes(),
        "relations": get_existing_relations(),
        "available_categories": get_available_categories(),
        "available_authors": get_available_authors()
    }

def _calculate_title_similarity(query: str, title: str) -> float:
    """
    Calculate similarity between query and title using multiple strategies.
    
    Args:
        query: Search query
        title: Book title to compare
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    from difflib import SequenceMatcher
    
    query_lower = query.lower().strip()
    title_lower = title.lower().strip()
    
    # 1. Match exato - prioridade máxima
    if query_lower == title_lower:
        return 1.0
    
    # 2. Match exato ignorando case - prioridade alta  
    if query.lower() == title.lower():
        return 0.95
    
    # 3. Verificar se o query está contido no título
    if query_lower in title_lower and query_lower != title_lower:
        return 0.85
    
    # 4. Similaridade direta
    direct_similarity = SequenceMatcher(None, query_lower, title_lower).ratio()
    
    # 5. Similaridade com palavras parciais
    if len(query_lower) < len(title_lower):
        best_partial = 0
        for i in range(len(title_lower) - len(query_lower) + 1):
            partial = title_lower[i:i + len(query_lower)]
            partial_sim = SequenceMatcher(None, query_lower, partial).ratio()
            best_partial = max(best_partial, partial_sim)
        
        # 6. Similaridade com palavras individuais
        query_words = query_lower.split()
        title_words = title_lower.split()
        word_similarities = []
        
        for q_word in query_words:
            best_word_sim = 0
            for t_word in title_words:
                word_sim = SequenceMatcher(None, q_word, t_word).ratio()
                best_word_sim = max(best_word_sim, word_sim)
            word_similarities.append(best_word_sim)
        
        avg_word_similarity = sum(word_similarities) / len(word_similarities) if word_similarities else 0
        
        return max(direct_similarity, best_partial, avg_word_similarity)
    else:
        return direct_similarity

def search_books_by_title(title_query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search for books by title using fuzzy matching and return full details.
    
    Args:
        title_query: Search term to match against book titles
        limit: Maximum number of results
        
    Returns:
        List of book details dictionaries ordered by similarity (highest first)
    """
    # Verifica se o driver existe, caso contrário conecta
    from neo4j_client import driver
    if driver is None:
        if not connect_to_neo4j():
            return []
    
    # Buscar todos os livros com seus detalhes
    query = """
    MATCH (b:Book)
    OPTIONAL MATCH (b)-[:WRITTEN_BY]->(a)
    OPTIONAL MATCH (b)-[:BELONGS_TO]->(c)
    RETURN b, 
           collect(DISTINCT a.name) as authors,
           collect(DISTINCT c.name) as categories
    """
    
    try:
        with driver.session(database=BOOKS_DATABASE) as session:
            result = session.run(query)
            all_books = []
            
            for record in result:
                book_data = dict(record["b"])
                book_details = {
                    "book": book_data,
                    "authors": [a for a in record["authors"] if a],
                    "categories": [c for c in record["categories"] if c]
                }
                all_books.append(book_details)
            
            # Calcular similaridade para cada livro
            matches = []
            exact_matches = []
            
            for book_details in all_books:
                title = book_details["book"].get("Title", "")
                if title:
                    similarity = _calculate_title_similarity(title_query, title)
                    if similarity >= 0.8:  # 80% de similaridade fixo
                        # Separar matches exatos dos outros
                        if similarity == 1.0:  # Match exato perfeito
                            exact_matches.append((book_details, similarity))
                        else:
                            matches.append((book_details, similarity))
            
            # Ordenar matches exatos primeiro (alfabeticamente), depois os outros por similaridade
            exact_matches.sort(key=lambda x: x[0]["book"].get("Title", ""))
            matches.sort(key=lambda x: (-x[1], x[0]["book"].get("Title", "")))
            
            # Dar prioridade absoluta ao match exato perfeito
            perfect_matches = []
            other_exact_matches = []
            
            for match in exact_matches:
                title = match[0]["book"].get("Title", "")
                if title.lower().strip() == title_query.lower().strip():
                    perfect_matches.append(match)
                else:
                    other_exact_matches.append(match)
            
            # Combinar: match perfeito primeiro, depois outros exatos, depois os outros
            all_matches = perfect_matches + other_exact_matches + matches
            
            # Retornar apenas os detalhes dos livros
            return [match[0] for match in all_matches[:limit]]
            
    except Exception as e:
        print(f"Error fetching book details: {e}")
        return []

def get_book_details_by_title(title: str) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive details about a specific book.
    
    Args:
        title: Book title
        
    Returns:
        A dictionary with:
            - "book": properties of the Book node,
            - "authors": list of author names,
            - "categories": list of category names,
        or None if the title is not found.
    """
    results = search_books_by_title(title, limit=1)
    return results[0] if results else None

def retrieve_titles_by_condition(relation: str, value: str, limit: int = 200) -> str:
    """
    Deterministically retrieve book titles for a (relation, value) CONDITION and
    return them as a single ' [SEP] ' joined line. If no results, return ''.

    Args:
        relation: One of {'WRITTEN_BY','BELONGS_TO'}.
        value:    Canonical author/category name (string).
        limit:    Upper bound for retrieval (safety cap).

    Returns:
        A single string with titles separated by ' [SEP] ', or '' if empty.

    Notes:
        - Uses your existing tools under the hood (get_books_by_author, get_books_by_category).
        - Applies deterministic dedup (case/underscore normalization) and sorts A–Z.
    """
    rel = (relation or "").strip()
    val = (value or "").strip()
    titles = []

    try:
        if rel == "WRITTEN_BY":
            titles = get_books_by_author(val, limit=limit)
        elif rel == "BELONGS_TO":
            titles = get_books_by_category(val, limit=limit)
        else:
            # Fallback genérico para qualquer relação suportada no grafo
            titles = get_books_by_relation(rel, val, limit=limit)
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
    get_available_categories,
    get_available_authors,
    explore_database_schema,
    get_books_by_relation,
    get_books_by_author,
    get_books_by_category,
    search_books_by_title,
    get_book_details_by_title,
    retrieve_titles_by_condition,
]