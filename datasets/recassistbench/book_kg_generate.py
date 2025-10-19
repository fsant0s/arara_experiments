
#!/usr/bin/env python3
"""
Script para construir o Knowledge Graph de livros no Neo4j
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
NEO4J_DATABASE = "books"  # Banco específico para livros

SCHEMA_FILE = "eval/book-schema.json"
BOOK_INFO_FILE = "dataset/book/book_info.jsonl"

CLEAR_DATABASE = False  # Não limpar banco para manter filmes existentes

class BookKGBuilder:
    """Constrói o Knowledge Graph de livros no Neo4j"""
    
    def __init__(self, uri, username, password, database, schema_path):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        
        # Carregar schema
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        
        # Mapeamento de relação -> tipo de nó alvo
        self.relation_targets = {}
        for rel in self.schema['relations']:
            self.relation_targets[rel['type']] = rel['target']
        
        print(f"✓ Schema carregado: {len(self.schema['relations'])} relações")
    
    def create_database(self):
        """Cria o banco de dados se não existir"""
        try:
            with self.driver.session(database="system") as session:
                session.run(f"CREATE DATABASE {self.database} IF NOT EXISTS")
            print(f"✓ Banco '{self.database}' criado/verificado")
            return True
        except Exception as e:
            print(f"❌ Erro ao criar banco: {e}")
            return False
    
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
        
        print("✓ Índices criados")
    
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
    
    def create_book_and_relations(self, book_data):
        """Cria nó do livro e todas as suas relações"""
        
        title = book_data.get('Title')
        if not title:
            return
        
        with self.driver.session(database=self.database) as session:
            # 1. Criar nó do livro
            session.run("""
                MERGE (b:Book {Title: $title})
            """, title=title)
            
            # 2. Criar relações baseado no schema
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
        """Constrói o grafo completo"""
        
        with open(book_info_path, 'r', encoding='utf-8') as f:
            total = sum(1 for _ in f)
        
        print(f"Processando {total} livros...")
        
        with open(book_info_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total, desc="Construindo KG"):
                try:
                    book_data = json.loads(line)
                    self.create_book_and_relations(book_data)
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
        print("ESTATÍSTICAS DO KNOWLEDGE GRAPH DE LIVROS")
        print("="*70)
        
        print(f"\nTotal de Nós: {stats['total_nodes']:,}")
        print("\nNós por Tipo:")
        for item in stats['nodes_by_type']:
            print(f"  • {item['type']:20s}: {item['count']:,}")
        
        print(f"\nTotal de Relações: {stats['total_relations']:,}")
        print("\nRelações por Tipo:")
        for item in stats['relations_by_type']:
            print(f"  • {item['type']:20s}: {item['count']:,}")
        
        print("\n" + "="*70)

    def close(self):
        self.driver.close()


def main():
    """Construir Knowledge Graph de livros no Neo4j"""
    
    # Verificar arquivos
    for path in [SCHEMA_FILE, BOOK_INFO_FILE]:
        if not os.path.exists(path):
            print(f"Arquivo não encontrado: {path}")
            return
    
    print("="*70)
    print("CONSTRUINDO KNOWLEDGE GRAPH DE LIVROS")
    print("="*70)
    
    # Construir KG
    builder = BookKGBuilder(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        schema_path=SCHEMA_FILE
    )
    
    try:
        # Criar banco de dados se necessário
        if not builder.create_database():
            return
        
        if not builder.test_connection():
            return
        
        if CLEAR_DATABASE:
            builder.clear_database()
        
        builder.create_indexes()
        builder.build_graph(BOOK_INFO_FILE)
        builder.print_stats()
        
        print("Knowledge Graph de livros construído com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
    finally:
        builder.close()


if __name__ == '__main__':
    main()