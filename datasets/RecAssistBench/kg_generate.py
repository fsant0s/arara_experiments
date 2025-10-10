#!/usr/bin/env python3
"""
Script para construir o Knowledge Graph de filmes no Neo4j

Usa:
- eval/movie-schema.json: Schema com definições de nós e relações
- dataset/movie/movie_info.jsonl: Informações dos filmes
- dataset/movie/movies.dat: IDs dos filmes

Uso:
    python build_movie_kg.py --password SUA_SENHA --clear
"""

import json
import argparse
import re
from neo4j import GraphDatabase
from tqdm import tqdm
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


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
        
        logger.info(f"✓ Schema carregado com {len(self.relation_map)} mapeamentos de relações")
        
        # Log dos mapeamentos carregados
        logger.info("📋 Mapeamentos de relações:")
        for relation, variants in self.relation_mappings.items():
            logger.info(f"   {relation}: {variants}")
    
    def test_connection(self):
        """Testa conexão com Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("✓ Conexão com Neo4j estabelecida")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao conectar ao Neo4j: {e}")
            return False
    
    def clear_database(self):
        """Limpa o banco de dados"""
        logger.info("🗑️  Limpando banco de dados...")
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("✓ Banco limpo")
    
    def create_indexes(self):
        """Cria índices para melhorar performance"""
        logger.info("📇 Criando índices...")
        
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
        
        logger.info("✓ Índices criados")
    
    def load_movie_ids_and_genres(self, movies_dat_path):
        """Carrega IDs dos filmes e gêneros do movies.dat"""
        title_to_id = {}
        title_to_genres = {}
        
        with open(movies_dat_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    movie_id = int(parts[0])
                    title = parts[1]
                    genres = parts[2].split('|')  # Gêneros separados por |
                    
                    title_to_id[title] = movie_id
                    title_to_genres[title] = genres
        
        logger.info(f"✓ Carregados {len(title_to_id)} IDs de filmes")
        logger.info(f"✓ Carregados gêneros para {len(title_to_genres)} filmes")
        return title_to_id, title_to_genres
    
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

    def create_movie_and_relations(self, movie_data, title_to_id, title_to_genres):
        """Cria nó do filme e todas as suas relações"""
        
        title = movie_data.get('Title')
        if not title:
            return
        
        # Encontrar ID e gêneros
        movie_id = self.find_movie_id(title, title_to_id)
        movie_genres = title_to_genres.get(title, [])
        
        with self.driver.session(database=self.database) as session:
            # 1. Criar nó do filme
            movie_props = {'Title': title}
            if movie_id:
                movie_props['movieId'] = movie_id
            
            session.run("""
                MERGE (m:Movie {Title: $title})
                SET m += $props
            """, title=title, props=movie_props)
            
            # 2. Criar relações baseado no schema
            for field_name, value in movie_data.items():
                if field_name == 'Title' or not value:
                    continue
                
                # Verificar se é uma relação mapeada (com normalização)
                standard_relation = self.find_standard_relation(field_name)
                if not standard_relation:
                    continue
                
                logger.debug(f"   📝 Mapeando '{field_name}' → {standard_relation}")
                
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
                        """, entity=entity, title=title)
                    except Exception as e:
                        logger.debug(f"Erro ao criar relação {standard_relation}: {e}")
            
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
                            """, genre=genre.strip(), title=title)
                        except Exception as e:
                            logger.debug(f"Erro ao criar relação Genre: {e}")
    
    def build_graph(self, movie_info_path, movies_dat_path):
        """Constrói o grafo completo"""
        
        # 1. Carregar IDs e gêneros
        logger.info("Carregando IDs e gêneros dos filmes...")
        title_to_id, title_to_genres = self.load_movie_ids_and_genres(movies_dat_path)
        
        # 2. Contar filmes
        logger.info("Contando filmes...")
        with open(movie_info_path, 'r', encoding='utf-8') as f:
            total = sum(1 for _ in f)
        
        # 3. Processar filmes
        logger.info(f"Processando {total} filmes...")
        
        processed = 0
        with_id = 0
        
        with open(movie_info_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total, desc="Construindo KG"):
                try:
                    movie_data = json.loads(line)
                    self.create_movie_and_relations(movie_data, title_to_id, title_to_genres)
                    processed += 1
                    
                    if self.find_movie_id(movie_data.get('Title', ''), title_to_id):
                        with_id += 1
                        
                except Exception as e:
                    logger.debug(f"Erro: {e}")
        
        logger.info(f"✓ {processed} filmes processados")
        logger.info(f"✓ {with_id} filmes com movieId")
    
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
    
    def test_query(self):
        """Testa query do benchmark"""
        logger.info("\nTestando query do benchmark...")
        logger.info("   Query: Filmes de guerra com Tom Hanks")
        
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (m:Movie)
                WHERE (m)-[:Starring]->(p:Person {name: 'Tom Hanks'})
                  AND (m)-[:Genre]->(g:Genre {name: 'War'})
                RETURN m.Title AS title, m.movieId AS id
            """).data()
            
            if result:
                logger.info(f"    Encontrados {len(result)} filmes:")
                for r in result:
                    logger.info(f"      • {r['title']} (ID: {r['id']})")
            else:
                logger.warning("    Nenhum filme encontrado")
    
    def close(self):
        self.driver.close()


def main():
    parser = argparse.ArgumentParser(
        description='Construir Knowledge Graph de filmes no Neo4j',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Exemplos:
  # Construir KG (limpar banco antes)
  python build_movie_kg.py --password arara123 --clear
  
  # Apenas adicionar dados (não limpar)
  python build_movie_kg.py --password arara123
        '''
    )
    
    parser.add_argument('--uri', default='bolt://localhost:7687',
                       help='URI do Neo4j (padrão: bolt://localhost:7687)')
    parser.add_argument('--username', default='neo4j',
                       help='Usuário do Neo4j (padrão: neo4j)')
    parser.add_argument('--password', required=True,
                       help='Senha do Neo4j')
    parser.add_argument('--database', default='neo4j',
                       help='Nome do database (padrão: neo4j)')
    parser.add_argument('--schema', default='eval/movie-schema.json',
                       help='Caminho do schema (padrão: eval/movie-schema.json)')
    parser.add_argument('--movie-info', default='dataset/movie/movie_info.jsonl',
                       help='Caminho do movie_info.jsonl (usar filtrado para apenas filmes do dataset)')
    parser.add_argument('--movies-dat', default='dataset/movie/movies.dat',
                       help='Caminho do movies.dat')
    parser.add_argument('--clear', action='store_true',
                       help='Limpar banco antes de construir')
    
    args = parser.parse_args()
    
    # Verificar se arquivos existem
    for path in [args.schema, args.movie_info, args.movies_dat]:
        if not os.path.exists(path):
            logger.error(f"Arquivo não encontrado: {path}")
            return
    
    # Construir KG
    builder = MovieKGBuilder(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
        schema_path=args.schema
    )
    
    try:
        # Testar conexão
        if not builder.test_connection():
            return
        
        # Limpar se solicitado
        if args.clear:
            builder.clear_database()
        
        # Criar índices
        builder.create_indexes()
        
        # Construir grafo
        builder.build_graph(args.movie_info, args.movies_dat)
        
        # Mostrar estatísticas
        builder.print_stats()
        
        # Testar query
        builder.test_query()
        
        logger.info("\nKnowledge Graph construído com sucesso!")
        logger.info(f"   Database: {args.database}")
        logger.info(f"   URI: {args.uri}\n")
        
    except Exception as e:
        logger.error(f"\nErro: {e}", exc_info=True)
    finally:
        builder.close()


if __name__ == '__main__':
    main()

