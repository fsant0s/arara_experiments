#!/usr/bin/env python3
"""
Script to build the movie Knowledge Graph in Neo4j
"""

import json
import re
from neo4j import GraphDatabase
from tqdm import tqdm
import os

# ============================================================================
# SETTINGS
# ============================================================================
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "arara123"
NEO4J_DATABASE = "neo4j"

SCHEMA_FILE = "datasets/recassistbench/eval/movie-schema.json"
MOVIE_INFO_FILE = "datasets/recassistbench/dataset/movie/movie_info.jsonl"  # Use the filtered one!
MOVIES_DAT_FILE = "datasets/recassistbench/dataset/movie/movies.dat"

CLEAR_DATABASE = True  # True to clear the database before building


class MovieKGBuilder:
    """Builds the movie Knowledge Graph in Neo4j"""
    
    def __init__(self, uri, username, password, database, schema_path):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        
        # Load schema
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        
        # Process relation mappings
        self.relation_mappings = self.schema.get('relation_mappings', {})
        
        # Create reverse mapping: original_name -> standardized_name
        self.relation_map = {}
        for standard_name, variants in self.relation_mappings.items():
            for variant in variants:
                self.relation_map[variant] = standard_name
        
        # Mapping from relation -> target node type
        self.relation_targets = {}
        for rel in self.schema['relations']:
            self.relation_targets[rel['type']] = rel['target']
        
        print(f"✓ Schema loaded: {len(self.relation_map)} mappings")
    
    def test_connection(self):
        """Tests the connection with Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            print("✓ Connection OK")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def clear_database(self):
        """Clears the database"""
        print("Clearing database...")
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Database cleared")
    
    def create_indexes(self):
        """Creates indexes to improve performance"""
        indexes = [
            "CREATE INDEX movie_title IF NOT EXISTS FOR (m:Movie) ON (m.Title)",
            "CREATE INDEX movie_id IF NOT EXISTS FOR (m:Movie) ON (m.movieId)",
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX genre_name IF NOT EXISTS FOR (g:Genre) ON (g.name)",
            "CREATE INDEX corporation_name IF NOT EXISTS FOR (c:Corporation) ON (c.name)",
            "CREATE INDEX country_name IF NOT EXISTS FOR (c:Country) ON (c.name)",
            "CREATE INDEX thing_name IF NOT EXISTS FOR (t:Thing) ON (t.name)",
        ]
        
        with self.driver.session(database=self.database) as session:
            for idx in indexes:
                try:
                    session.run(idx)
                except:
                    pass
        
        print("✓ Indexes created")
    
    def load_movie_ids_and_genres(self, movies_dat_path):
        """Loads movie IDs and genres from movies.dat"""
        title_to_id = {}
        title_to_genres = {}
        id_to_title = {}  # Reverse mapping: ID -> title from movies.dat
        
        with open(movies_dat_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    movie_id = int(parts[0])
                    title = parts[1]
                    genres = parts[2].split('|')
                    
                    title_to_id[title] = movie_id
                    title_to_genres[title] = genres
                    id_to_title[movie_id] = title
        
        print(f"✓ {len(title_to_id)} movies loaded from movies.dat")
        return title_to_id, title_to_genres, id_to_title
    
    def normalize_title(self, title):
        """Normalizes title for matching"""
        # Remove special characters, parentheses with years, etc.
        normalized = re.sub(r'\s*\(\d{4}\)\s*', '', title)  # Remove (1994)
        normalized = re.sub(r'[^\w\s]', '', normalized)     # Remove punctuation
        normalized = normalized.strip().lower()
        return normalized
    
    def find_movie_id(self, title, title_to_id):
        """Finds the movie ID"""
        # Try exact match first
        if title in title_to_id:
            return title_to_id[title]
        
        # Try normalized match
        normalized = self.normalize_title(title)
        for dat_title, movie_id in title_to_id.items():
            if self.normalize_title(dat_title) == normalized:
                return movie_id
        
        return None
    
    def find_dat_title(self, movie_id, id_to_title):
        """Finds the movies.dat title by ID"""
        return id_to_title.get(movie_id)
    
    def split_entities(self, value):
        """Splits multiple values (separated by ; or ,)"""
        if not value:
            return []
        
        # Split by ; first, then by comma
        entities = []
        for part in value.split(';'):
            # If there are commas, split them as well
            if ',' in part:
                entities.extend([e.strip() for e in part.split(',') if e.strip()])
            else:
                if part.strip():
                    entities.append(part.strip())
        
        return [e for e in entities if e]
    
    def normalize_field_name(self, field_name):
        """Normalizes the field name to match the schema"""
        # Convert to lowercase and replace spaces/hyphens with underscores
        normalized = field_name.lower().replace(' ', '_').replace('-', '_')
        return normalized
    
    def find_standard_relation(self, field_name):
        """Finds the standard relation for a field, with normalization"""
        # Try exact match first
        if field_name in self.relation_map:
            return self.relation_map[field_name]
        
        # Try normalized match
        normalized = self.normalize_field_name(field_name)
        for variant, standard in self.relation_map.items():
            if self.normalize_field_name(variant) == normalized:
                return standard
        
        # Try case-insensitive match
        field_lower = field_name.lower()
        for variant, standard in self.relation_map.items():
            if variant.lower() == field_lower:
                return standard
        
        return None

    def create_movie_and_relations(self, movie_data, title_to_id, title_to_genres, id_to_title):
        """Creates the movie node and all its relations"""
        
        title = movie_data.get('Title')
        if not title:
            return
        
        # Find ID and genres
        movie_id = self.find_movie_id(title, title_to_id)
        movie_genres = title_to_genres.get(title, [])
        
        # If the ID was found in movies.dat, use the movies.dat title
        final_title = title
        if movie_id:
            dat_title = self.find_dat_title(movie_id, id_to_title)
            if dat_title:
                final_title = dat_title
        
        with self.driver.session(database=self.database) as session:
            # 1. Create the movie node
            movie_props = {'Title': final_title}
            if movie_id:
                movie_props['movieId'] = movie_id
            
            session.run("""
                MERGE (m:Movie {Title: $title})
                SET m += $props
            """, title=final_title, props=movie_props)
            
            # 2. Create relations based on the schema
            for field_name, value in movie_data.items():
                if field_name == 'Title' or not value:
                    continue
                
                # Check if it's a mapped relation (with normalization)
                standard_relation = self.find_standard_relation(field_name)
                if not standard_relation:
                    continue
                
                # Get target node type
                target_type = self.relation_targets.get(standard_relation, 'Thing')
                
                # Split multiple entities
                entities = self.split_entities(value)
                
                # Create node and relation for each entity
                for entity in entities:
                    if not entity:
                        continue
                    
                    try:
                        session.run(f"""
                            MERGE (e:{target_type} {{name: $entity}})
                            WITH e
                            MATCH (m:Movie {{Title: $title}})
                            MERGE (m)-[:{standard_relation}]->(e)
                        """, entity=entity, title=final_title)
                    except Exception as e:
                        pass
            
            # 3. Add genres from movies.dat if they don't exist in movie_info.jsonl
            if movie_genres and not movie_data.get('Genre') and not movie_data.get('Genres'):
                for genre in movie_genres:
                    if genre.strip():
                        try:
                            session.run("""
                                MERGE (g:Genre {name: $genre})
                                WITH g
                                MATCH (m:Movie {Title: $title})
                                MERGE (m)-[:Genre]->(g)
                            """, genre=genre.strip(), title=final_title)
                        except Exception as e:
                            pass
    
    def build_graph(self, movie_info_path, movies_dat_path):
        """Builds the complete graph"""
        title_to_id, title_to_genres, id_to_title = self.load_movie_ids_and_genres(movies_dat_path)
        
        with open(movie_info_path, 'r', encoding='utf-8') as f:
            total = sum(1 for _ in f)
        
        print(f"Processing {total} movies...")
        
        with open(movie_info_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total, desc="Building KG"):
                try:
                    movie_data = json.loads(line)
                    self.create_movie_and_relations(movie_data, title_to_id, title_to_genres, id_to_title)
                except:
                    pass
        
        print(f"✓ Completed")
    
    def get_stats(self):
        """Returns graph statistics"""
        with self.driver.session(database=self.database) as session:
            # Total nodes
            total_nodes = session.run("MATCH (n) RETURN count(n) as c").single()['c']
            
            # Nodes by type
            nodes_by_type = session.run("""
                MATCH (n) 
                RETURN labels(n)[0] AS type, count(*) AS count 
                ORDER BY count DESC
            """).data()
            
            # Total relationships
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()['c']
            
            # Relationships by type
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
        """Prints graph statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("="*70)
        
        print(f"\nTotal Nodes: {stats['total_nodes']:,}")
        print("\nNodes by Type:")
        for item in stats['nodes_by_type']:
            print(f"  • {item['type']:20s}: {item['count']:,}")
        
        print(f"\nTotal Relationships: {stats['total_relations']:,}")
        print("\nRelationships by Type:")
        for item in stats['relations_by_type'][:15]:  # Top 15
            print(f"  • {item['type']:20s}: {item['count']:,}")
        
        print("\n" + "="*70)

    def close(self):
        self.driver.close()


def main():
    """Build the movie Knowledge Graph in Neo4j"""
    
    # Check files
    for path in [SCHEMA_FILE, MOVIE_INFO_FILE, MOVIES_DAT_FILE]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
    
    print("="*70)
    print("BUILDING KNOWLEDGE GRAPH")
    print("="*70)
    
    # Build KG
    builder = MovieKGBuilder(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        schema_path=SCHEMA_FILE
    )
    
    try:
        if not builder.test_connection():
            return
        
        if CLEAR_DATABASE:
            builder.clear_database()
        
        builder.create_indexes()
        builder.build_graph(MOVIE_INFO_FILE, MOVIES_DAT_FILE)
        builder.print_stats()
        
        print("Knowledge Graph built successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        builder.close()


if __name__ == '__main__':
    main()
