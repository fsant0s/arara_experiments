from neo4j import GraphDatabase
from typing import List, Dict, Any

# Neo4j configuration
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "arara123"
NEO4J_DATABASE = "neo4j"

driver = None

def connect_to_neo4j():
    """Connects to Neo4j"""
    global driver
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run("RETURN 1")
        return True
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        return False

def close_connection():
    """Closes the connection with Neo4j"""
    global driver
    if driver:
        driver.close()

def get_movies_by_conditions(conditions: List[List[str]]) -> List[Dict[str, Any]]:
    """Fetches movies that satisfy the specified conditions"""
    if not driver:
        return []
    
    cypher_query = "MATCH (m:Movie)"
    where_clauses = []
    
    for condition in conditions:
        relationship_type, target_name = condition
        target_name = target_name.replace("'", "\\'")
        where_clauses.append(f"(m)-[:{relationship_type}]->({{name: '{target_name}'}})")
    
    if where_clauses:
        cypher_query += " WHERE " + " AND ".join(where_clauses)
    
    cypher_query += " RETURN m.movieId as movieId, m.Title as title LIMIT 20"
    
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher_query)
            movies = []
            for record in result:
                movies.append({
                    "movieId": record["movieId"],
                    "title": record["title"]
                })
            return movies
    except Exception as e:
        print(f"❌ Error executing query: {e}")
        return []

def get_items_from_query(conditions: List[List[str]]) -> List[str]:
    """Returns only the movie titles"""
    movies = get_movies_by_conditions(conditions)
    return [movie["title"] for movie in movies]

if __name__ == "__main__":
    # Simple test
    if connect_to_neo4j():
        conditions = [["Genre", "Animation"]]
        movies = get_movies_by_conditions(conditions)
        print(f"Found {len(movies)} animation movies")
        close_connection()
