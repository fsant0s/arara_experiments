from typing import List, Dict, Any, Optional
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Usar banco de livros
BOOKS_DATABASE = "neo4j"

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
    Find books connected to a specific entity via a relationship.
    
    This tool searches for books that have a specific relationship to a target entity
    (author, category, etc.). Use this for precise relationship-based queries.
    
    Args:
        relation: Relationship type - MUST be exactly 'WRITTEN_BY' or 'BELONGS_TO'
        target_name: Exact name of the target entity (author name, category name, etc.)
        limit: Maximum number of results to return (default: 20).
        
    Returns:
        List of book titles that match the relationship
        
    Examples:
        - get_books_by_relation("WRITTEN_BY", "Stephen King") 
          → Returns books written by Stephen King
        - get_books_by_relation("BELONGS_TO", "Science Fiction")
          → Returns books in Science Fiction category
          
    Note: Use exact relation names as they appear in the database.
    """
    # Handle limit=0 as "no limit" by removing LIMIT clause entirely
    if limit > 0:
        query = f"""
        MATCH (b:Book)-[:{relation}]->(t {{name: $target_name}})
        RETURN b.Title as title
        ORDER BY b.Title
        LIMIT {limit}
        """
        results = _execute_query(query, {"target_name": target_name})
    else:
        query = f"""
        MATCH (b:Book)-[:{relation}]->(t {{name: $target_name}})
        RETURN b.Title as title
        ORDER BY b.Title
        LIMIT 20
        """
        results = _execute_query(query, {"target_name": target_name})
    return [r["title"] for r in results if r.get("title")]

def get_books_by_author(author_name: str, limit: int = 20) -> List[str]:
    """
    Find books written by a specific author.
    
    This is a convenience function that calls get_books_by_relation with "WRITTEN_BY".
    Use this for author-based queries to get all books by a specific author.
    
    Args:
        author_name: Exact name of the author (e.g., "Stephen King", "J.K. Rowling")
        limit: Maximum number of results to return (default: 20). Use 0 for no limit.
        
    Returns:
        List of book titles written by the author
        
    Examples:
        - get_books_by_author("Stephen King") 
          → Returns ["The Shining", "It", "Carrie", ...]
        - get_books_by_author("J.K. Rowling")
          → Returns ["Harry Potter and the Philosopher's Stone", ...]
          
    Note: Use the exact author name as it appears in the database.
    """
    return get_books_by_relation("WRITTEN_BY", author_name, limit)

def get_books_by_category(category: str, limit: int = 20) -> List[str]:
    """
    Find books in a specific category or genre.
    
    This is a convenience function that calls get_books_by_relation with "BELONGS_TO".
    Use this for category-based queries to get all books in a specific genre or category.
    
    Args:
        category: Exact name of the category (e.g., "Fiction", "Science Fiction", "Mystery")
        limit: Maximum number of results to return (default: 20)
        
    Returns:
        List of book titles in the specified category
        
    Examples:
        - get_books_by_category("Science Fiction") 
          → Returns ["Dune", "Foundation", "The Martian", ...]
        - get_books_by_category("Mystery")
          → Returns ["The Girl with the Dragon Tattoo", ...]
          
    Note: Use the exact category name as it appears in the database.
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
    Get comprehensive details about a specific book by its title.
    
    This tool performs fuzzy matching on book titles and returns detailed information
    including authors, categories, and all book properties. Use this when you need
    complete information about a specific book.
    
    Args:
        title: Book title to search for (can be partial or fuzzy match)
        
    Returns:
        A dictionary with:
            - "book": properties of the Book node (Title, ISBN, etc.)
            - "authors": list of author names (from WRITTEN_BY relation)
            - "categories": list of category names (from BELONGS_TO relation)
        or None if the title is not found
        
    Examples:
        - get_book_details_by_title("The Great Gatsby")
          → Returns {"book": {...}, "authors": ["F. Scott Fitzgerald"], "categories": ["American Literature"]}
        - get_book_details_by_title("1984")
          → Returns {"book": {...}, "authors": ["George Orwell"], "categories": ["Dystopian Fiction"]}
          
    Note: This function uses fuzzy matching, so partial titles work well.
    """
    results = search_books_by_title(title, limit=1)
    return results[0] if results else None

def retrieve_titles_by_condition(relation: str, value: str, limit: int = 200) -> str:
    """
    Retrieve book titles for a specific condition and return them as a single joined string.
    
    This tool is designed for implicit queries where you need to find books that share
    a specific attribute (author, category) with reference books. It returns results
    in a standardized format suitable for further processing.
    
    Args:
        relation: Relationship type - MUST be exactly 'WRITTEN_BY' or 'BELONGS_TO'
        value: Exact name of the entity (author name, category name, etc.)
        limit: Upper bound for retrieval (safety cap, default: 200)
        
    Returns:
        A single string with titles separated by ' [SEP] ', or empty string if no results
        
    Examples:
        - retrieve_titles_by_condition("WRITTEN_BY", "Stephen King")
          → Returns "The Shining [SEP] It [SEP] Carrie [SEP] ..."
        - retrieve_titles_by_condition("BELONGS_TO", "Science Fiction")
          → Returns "Dune [SEP] Foundation [SEP] The Martian [SEP] ..."
          
    Note: 
        - Uses deterministic deduplication and sorting
        - Results are normalized and sorted alphabetically
        - Empty result returns empty string, not None
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