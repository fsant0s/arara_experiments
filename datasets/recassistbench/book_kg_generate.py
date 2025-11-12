#!/usr/bin/env python3
"""
Script to build the Books Knowledge Graph in Neo4j
"""

import json
import re
from neo4j import GraphDatabase
from tqdm import tqdm
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "arara123"
NEO4J_DATABASE = "neo4j"

SCHEMA_FILE = "datasets/recassistbench/eval/book-schema.json"
BOOK_INFO_FILE = "datasets/recassistbench/dataset/book/book_data.csv"

CLEAR_DATABASE = False  # Do not clear database to keep existing movies

class BookKGBuilder:
    """Builds the Books Knowledge Graph in Neo4j"""
    
    def __init__(self, uri, username, password, database, schema_path):
        """Initialize the Neo4j driver, load the schema and prepare relation target mapping."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        
        # Load schema
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        
        # Mapping relation -> target node type
        self.relation_targets = {}
        for rel in self.schema['relations']:
            self.relation_targets[rel['type']] = rel['target']
        
        print(f"✓ Schema loaded: {len(self.schema['relations'])} relations")
    
    def create_database(self):
        """Use the default database"""
        print(f"✓ Using default database '{self.database}'")
        return True
    
    def test_connection(self):
        """Test connection to Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            print("✓ Connection OK")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    
    def clear_database(self):
        """Clear the database"""
        print("Clearing database...")
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Database cleared")
    
    def create_indexes(self):
        """Create indexes to improve performance"""
        indexes = [
            "CREATE INDEX book_title IF NOT EXISTS FOR (b:Book) ON (b.Title)",
            "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name)",
            "CREATE INDEX category_name IF NOT EXISTS FOR (c:Category) ON (c.name)",
        ]
        
        with self.driver.session(database=self.database) as session:
            for idx in indexes:
                try:
                    session.run(idx)
                except:
                    pass
        
        print("✓ Indexes created")
    
    def split_entities(self, value):
        """Split multiple values using ast.literal_eval and fallback parsing"""
        if not value:
            return []
        
        import ast
        
        try:
            # Try to parse as a Python list
            entities = ast.literal_eval(value)
            if isinstance(entities, list):
                return [str(entity).strip() for entity in entities if str(entity).strip()]
        except (ValueError, SyntaxError):
            pass
        
        # Fallback: manual cleanup
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            value = value[1:-1]
        
        # Split by comma
        entities = []
        for part in value.split(','):
            part = part.strip()
            # Remove quotes
            if (part.startswith("'") and part.endswith("'")) or \
               (part.startswith('"') and part.endswith('"')):
                part = part[1:-1]
            if part:
                entities.append(part)
        
        return entities
    
    def create_book_and_relations(self, book_data):
        """Create the book node and all its relations"""
        
        title = book_data.get('Title')
        if not title:
            return
        
        with self.driver.session(database=self.database) as session:
            # 1. Create the book node
            session.run("""
                MERGE (b:Book {Title: $title})
            """, title=title)
            
            # 2. Create relations based on the schema
            # WRITTEN_BY: Book -> Author
            if book_data.get('Author'):
                authors = self.split_entities(book_data['Author'])
                for author in authors:
                    if author:
                        try:
                            session.run("""
                                MERGE (a:Author {name: $author})
                                WITH a
                                MATCH (b:Book {Title: $title})
                                MERGE (b)-[:WRITTEN_BY]->(a)
                            """, author=author, title=title)
                        except Exception as e:
                            pass
            
            # BELONGS_TO: Book -> Category
            if book_data.get('Category'):
                categories = self.split_entities(book_data['Category'])
                for category in categories:
                    if category:
                        try:
                            session.run("""
                                MERGE (c:Category {name: $category})
                                WITH c
                                MATCH (b:Book {Title: $title})
                                MERGE (b)-[:BELONGS_TO]->(c)
                            """, category=category, title=title)
                        except Exception as e:
                            pass
    
    def build_graph(self, book_info_path):
        """Build the complete graph from the CSV file"""
        import csv
        
        with open(book_info_path, 'r', encoding='utf-8') as f:
            total = sum(1 for _ in f) - 1  # -1 for header
        
        print(f"Processing {total} books...")
        
        with open(book_info_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, total=total, desc="Building KG"):
                try:
                    # Convert CSV row to expected format
                    book_data = {
                        'Title': row.get('Title', ''),
                        'Author': row.get('authors', ''),  # CSV uses 'authors'
                        'Category': row.get('categories', '')  # CSV uses 'categories'
                    }
                    self.create_book_and_relations(book_data)
                except Exception as e:
                    pass
        
        print(f"✓ Done")
    
    def get_stats(self):
        """Return statistics of the graph"""
        with self.driver.session(database=self.database) as session:
            # Total nodes
            total_nodes = session.run("MATCH (n) RETURN count(n) as c").single()['c']
            
            # Nodes by type
            nodes_by_type = session.run("""
                MATCH (n) 
                RETURN labels(n)[0] AS type, count(*) AS count 
                ORDER BY count DESC
            """).data()
            
            # Total relations
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()['c']
            
            # Relations by type
            rels_by_type = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) AS type, count(*) AS count 
                ORDER BY count DESC
            """).data()
            
            return {
                'total_nodes': total_nodes,
                'nodes_by_type': nodes_by_type,
                'total_relations': total_rels,
                'relations_by_type': rels_by_type
            }
    
    def print_stats(self):
        """Print statistics of the graph"""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("BOOKS KNOWLEDGE GRAPH STATISTICS")
        print("="*70)
        
        print(f"\nTotal Nodes: {stats['total_nodes']:,}")
        print("\nNodes by Type:")
        for item in stats['nodes_by_type']:
            print(f"  • {item['type']:20s}: {item['count']:,}")
        
        print(f"\nTotal Relations: {stats['total_relations']:,}")
        print("\nRelations by Type:")
        for item in stats['relations_by_type']:
            print(f"  • {item['type']:20s}: {item['count']:,}")
        
        print("\n" + "="*70)

    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()


def main():
    """Build the Books Knowledge Graph in Neo4j"""
    
    # Check required files
    for path in [SCHEMA_FILE, BOOK_INFO_FILE]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
    
    print("="*70)
    print("BUILDING BOOKS KNOWLEDGE GRAPH")
    print("="*70)
    
    # Build KG
    builder = BookKGBuilder(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        schema_path=SCHEMA_FILE
    )
    
    try:
        # Create database if necessary
        if not builder.create_database():
            return
        
        if not builder.test_connection():
            return
        
        if CLEAR_DATABASE:
            builder.clear_database()
        
        builder.create_indexes()
        builder.build_graph(BOOK_INFO_FILE)
        builder.print_stats()
        
        print("Books Knowledge Graph built successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        builder.close()


if __name__ == '__main__':
    main()