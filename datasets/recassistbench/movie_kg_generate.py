#!/usr/bin/env python3
"""
Script para construir o Knowledge Graph de filmes no Neo4j
"""

import json
import re
from neo4j import GraphDatabase
from tqdm import tqdm
import os

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "arara123"
NEO4J_DATABASE = "neo4j"

SCHEMA_FILE = "datasets/recassistbench/eval/movie-schema.json"
MOVIE_INFO_FILE = "datasets/recassistbench/dataset/movie/movie_info.jsonl"  # Usar o filtrado!
MOVIES_DAT_FILE = "datasets/recassistbench/dataset/movie/movies.dat"

CLEAR_DATABASE = True  # True para limpar banco antes de construir


class MovieKGBuilder:
    """Constrói o Knowledge Graph de filmes no Neo4j"""
    
    def __init__(self, uri, username, password, database, schema_path):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        
        # Carregar schema
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        
        # Processar mapeamentos de relações
        self.relation_mappings = self.schema.get('relation_mappings', {})
        
        # Criar mapeamento inverso: nome_original -> nome_padronizado
        self.relation_map = {}
        for standard_name, variants in self.relation_mappings.items():
            for variant in variants:
                self.relation_map[variant] = standard_name
        
        # Mapeamento de relação -> tipo de nó alvo
        self.relation_targets = {}
        for rel in self.schema['relations']:
            self.relation_targets[rel['type']] = rel['target']
        
        print(f"✓ Schema carregado: {len(self.relation_map)} mapeamentos")
    
    def test_connection(self):
        """Testa conexão com Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            print("✓ Conexão OK")
            return True
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False
    
    def clear_database(self):
        """Limpa o banco de dados"""
        print("Limpando banco...")
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Banco limpo")
    
    def create_indexes(self):
        """Cria índices para melhorar performance"""
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
        
        print("✓ Índices criados")
    
    def load_movie_ids_and_genres(self, movies_dat_path):
        """Carrega IDs dos filmes e gêneros do movies.dat"""
        title_to_id = {}
        title_to_genres = {}
        id_to_title = {}  # Mapeamento reverso: ID -> título do movies.dat
        
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
        
        print(f"✓ {len(title_to_id)} filmes carregados do movies.dat")
        return title_to_id, title_to_genres, id_to_title
    
    def normalize_title(self, title):
        """Normaliza título para matching"""
        # Remove caracteres especiais, parênteses com anos, etc
        normalized = re.sub(r'\s*\(\d{4}\)\s*', '', title)  # Remove (1994)
        normalized = re.sub(r'[^\w\s]', '', normalized)     # Remove pontuação
        normalized = normalized.strip().lower()
        return normalized
    
    def find_movie_id(self, title, title_to_id):
        """Encontra ID do filme"""
        # Tentar match exato primeiro
        if title in title_to_id:
            return title_to_id[title]
        
        # Tentar match normalizado
        normalized = self.normalize_title(title)
        for dat_title, movie_id in title_to_id.items():
            if self.normalize_title(dat_title) == normalized:
                return movie_id
        
        return None
    
    def find_dat_title(self, movie_id, id_to_title):
        """Encontra título do movies.dat pelo ID"""
        return id_to_title.get(movie_id)
    
    def split_entities(self, value):
        """Divide valores múltiplos (separados por ; ou ,)"""
        if not value:
            return []
        
        # Dividir por ; primeiro, depois por vírgula
        entities = []
        for part in value.split(';'):
            # Se houver vírgulas, dividir também
            if ',' in part:
                entities.extend([e.strip() for e in part.split(',') if e.strip()])
            else:
                if part.strip():
                    entities.append(part.strip())
        
        return [e for e in entities if e]
    
    def normalize_field_name(self, field_name):
        """Normaliza nome do campo para matching com o schema"""
        # Converte para lowercase e substitui espaços por underscores
        normalized = field_name.lower().replace(' ', '_').replace('-', '_')
        return normalized
    
    def find_standard_relation(self, field_name):
        """Encontra a relação padrão para um campo, com normalização"""
        # Tentar match exato primeiro
        if field_name in self.relation_map:
            return self.relation_map[field_name]
        
        # Tentar match normalizado
        normalized = self.normalize_field_name(field_name)
        for variant, standard in self.relation_map.items():
            if self.normalize_field_name(variant) == normalized:
                return standard
        
        # Tentar match case-insensitive
        field_lower = field_name.lower()
        for variant, standard in self.relation_map.items():
            if variant.lower() == field_lower:
                return standard
        
        return None

    def create_movie_and_relations(self, movie_data, title_to_id, title_to_genres, id_to_title):
        """Cria nó do filme e todas as suas relações"""
        
        title = movie_data.get('Title')
        if not title:
            return
        
        # Encontrar ID e gêneros
        movie_id = self.find_movie_id(title, title_to_id)
        movie_genres = title_to_genres.get(title, [])
        
        # Se encontrou ID no movies.dat, usar o título do movies.dat
        final_title = title
        if movie_id:
            dat_title = self.find_dat_title(movie_id, id_to_title)
            if dat_title:
                final_title = dat_title
        
        with self.driver.session(database=self.database) as session:
            # 1. Criar nó do filme
            movie_props = {'Title': final_title}
            if movie_id:
                movie_props['movieId'] = movie_id
            
            session.run("""
                MERGE (m:Movie {Title: $title})
                SET m += $props
            """, title=final_title, props=movie_props)
            
            # 2. Criar relações baseado no schema
            for field_name, value in movie_data.items():
                if field_name == 'Title' or not value:
                    continue
                
                # Verificar se é uma relação mapeada (com normalização)
                standard_relation = self.find_standard_relation(field_name)
                if not standard_relation:
                    continue
                
                # Obter tipo do nó alvo
                target_type = self.relation_targets.get(standard_relation, 'Thing')
                
                # Dividir múltiplas entidades
                entities = self.split_entities(value)
                
                # Criar nó e relação para cada entidade
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
            
            # 3. Adicionar gêneros do movies.dat se não existirem no movie_info.jsonl
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
        """Constrói o grafo completo"""
        title_to_id, title_to_genres, id_to_title = self.load_movie_ids_and_genres(movies_dat_path)
        
        with open(movie_info_path, 'r', encoding='utf-8') as f:
            total = sum(1 for _ in f)
        
        print(f"Processando {total} filmes...")
        
        with open(movie_info_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total, desc="Construindo KG"):
                try:
                    movie_data = json.loads(line)
                    self.create_movie_and_relations(movie_data, title_to_id, title_to_genres, id_to_title)
                except:
                    pass
        
        print(f"✓ Concluído")
    
    def get_stats(self):
        """Retorna estatísticas do grafo"""
        with self.driver.session(database=self.database) as session:
            # Total nós
            total_nodes = session.run("MATCH (n) RETURN count(n) as c").single()['c']
            
            # Nós por tipo
            nodes_by_type = session.run("""
                MATCH (n) 
                RETURN labels(n)[0] AS type, count(*) AS count 
                ORDER BY count DESC
            """).data()
            
            # Total relações
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()['c']
            
            # Relações por tipo
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
        """Imprime estatísticas do grafo"""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("ESTATÍSTICAS DO KNOWLEDGE GRAPH")
        print("="*70)
        
        print(f"\nTotal de Nós: {stats['total_nodes']:,}")
        print("\nNós por Tipo:")
        for item in stats['nodes_by_type']:
            print(f"  • {item['type']:20s}: {item['count']:,}")
        
        print(f"\nTotal de Relações: {stats['total_relations']:,}")
        print("\nRelações por Tipo:")
        for item in stats['relations_by_type'][:15]:  # Top 15
            print(f"  • {item['type']:20s}: {item['count']:,}")
        
        print("\n" + "="*70)

    def close(self):
        self.driver.close()


def main():
    """Construir Knowledge Graph de filmes no Neo4j"""
    
    # Verificar arquivos
    for path in [SCHEMA_FILE, MOVIE_INFO_FILE, MOVIES_DAT_FILE]:
        if not os.path.exists(path):
            print(f"Arquivo não encontrado: {path}")
            return
    
    print("="*70)
    print("CONSTRUINDO KNOWLEDGE GRAPH")
    print("="*70)
    
    # Construir KG
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
        
        print("Knowledge Graph construído com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
    finally:
        builder.close()


if __name__ == '__main__':
    main()

