from typing import List, Dict, Any
import os
import random
import json


def get_data_path(filename: str) -> str:
    """Retorna o caminho correto para os arquivos de dados de livros."""
    if os.path.exists("datasets/recassistbench/dataset/book/"):
        base_path = "datasets/recassistbench/dataset/book/"
    else:
        base_path = "../datasets/recassistbench/dataset/book/"
    return os.path.join(base_path, filename)


def get_user_history(user_id: str) -> List[Dict[str, Any]]:
    """Busca o histórico de avaliações de um usuário específico para livros."""
    ratings_path = get_data_path("ratings.dat")
    history = []
    
    try:
        with open(ratings_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[0] == user_id:
                    history.append({
                        "book_title": parts[1],
                        "rating": float(parts[2])
                    })
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {ratings_path}")
        return []
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
        return []
    
    return history


def get_book_info_from_jsonl(book_title: str) -> Dict[str, Any]:
    """Busca informações de um livro específico no book_info.jsonl."""
    books_path = get_data_path("book_info.jsonl")
    
    try:
        with open(books_path, 'r', encoding='utf-8') as file:
            for line in file:
                book_data = json.loads(line.strip())
                if book_data.get("Title") == book_title:
                    return book_data
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {books_path}")
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
    
    return {}


def get_user_book_titles(user_id: str) -> List[str]:
    """Retorna apenas os títulos dos livros que o usuário avaliou."""
    history = get_user_history(user_id)
    return [item["book_title"] for item in history]


def get_filtered_user_history(user_id: str, groundtruth_book_ids: List[str] = None, neo4j_conditions: List[List[str]] = None, percentage: float = 1.0) -> List[str]:
    """Retorna o histórico filtrado do usuário com porcentagem aleatória."""
    # Importa o cliente Neo4j para livros
    books_module = None
    try:
        import sys
        import os
        sys.path.append('.')
        sys.path.append('./pilot')
        
        # Importar diretamente sem passar pelo __init__.py
        import importlib.util
        spec = importlib.util.spec_from_file_location('books', './pilot/tools/books.py')
        books_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(books_module)
        
        # Testa se consegue conectar com Neo4j
        test_result = books_module.get_existing_nodes()
        if not test_result:  # Se retorna lista vazia, pode ser erro de conexão
            books_module = None
    except Exception as e:
        print(f"⚠️ Aviso: Não foi possível conectar com Neo4j: {e}")
        books_module = None
    
    # Busca histórico completo do usuário
    history = get_user_history(user_id)
    user_book_titles = [item["book_title"] for item in history]
    
    # Remove livros do groundtruth
    if groundtruth_book_ids:
        user_book_titles = [title for title in user_book_titles if title not in groundtruth_book_ids]
    
    # Remove livros das condições do Neo4j (apenas se conseguiu conectar)
    if neo4j_conditions and books_module is not None:
        try:
            neo4j_titles = []
            for relation, value in neo4j_conditions:
                if relation == "WRITTEN_BY":
                    titles = books_module.get_books_by_author(value, limit=200)
                elif relation == "BELONGS_TO":
                    titles = books_module.get_books_by_category(value, limit=200)
                else:
                    titles = books_module.get_books_by_relation(relation, value, limit=200)
                neo4j_titles.extend(titles)
            
            user_book_titles = [title for title in user_book_titles if title not in neo4j_titles]
        except Exception as e:
            print(f"⚠️ Aviso: Erro ao aplicar filtros Neo4j: {e}")
    
    # Aplica porcentagem=1.0 aleatória
    if percentage < 1.0:
        sample_size = int(len(user_book_titles) * percentage)
        user_book_titles = random.sample(user_book_titles, min(sample_size, len(user_book_titles)))
    
    return user_book_titles


def get_user_history_with_books(user_id: str) -> List[Dict[str, Any]]:
    """Busca o histórico de um usuário com informações dos livros."""
    history = get_user_history(user_id)
    enriched_history = []
    
    for item in history:
        book_info = get_book_info_from_jsonl(item["book_title"])
        enriched_item = {
            **item,
            "author": book_info.get("Author", ""),
            "category": book_info.get("Category", ""),
            "genre": book_info.get("Genre", "")
        }
        enriched_history.append(enriched_item)
    
    return enriched_history


def get_user_ratings_summary(user_id: str) -> Dict[str, Any]:
    """Retorna um resumo das avaliações do usuário."""
    history = get_user_history(user_id)
    
    if not history:
        return {"total_books": 0, "average_rating": 0.0, "ratings_distribution": {}}
    
    ratings = [item["rating"] for item in history]
    ratings_distribution = {}
    
    for rating in ratings:
        ratings_distribution[rating] = ratings_distribution.get(rating, 0) + 1
    
    return {
        "total_books": len(history),
        "average_rating": sum(ratings) / len(ratings),
        "ratings_distribution": ratings_distribution
    }


if __name__ == "__main__":
    # Teste simples
    user_id = "A30TK6U7DNS82R"
    print(f"🔍 Testando usuário {user_id}")
    
    # Teste básico
    history = get_user_history(user_id)
    print(f"   {len(history)} livros avaliados:")
    for i, item in enumerate(history[:5], 1):
        print(f"   {i}. {item['book_title']} - {item['rating']}")
    
    print()
    
    # Teste com filtros
    filtered_titles = get_filtered_user_history(
        user_id=user_id,
        groundtruth_book_ids=["Dr. Seuss: American Icon"],
        neo4j_conditions=[["WRITTEN_BY", "Stephen King"]],
        percentage=0.5
    )
    
    print(f"   {len(filtered_titles)} livros após filtros (50% do histórico):")
    for i, title in enumerate(filtered_titles[:5], 1):
        print(f"   {i}. {title}")
    
    print()
    
    # Resumo das avaliações
    summary = get_user_ratings_summary(user_id)
    print(f"📊 Resumo das avaliações:")
    print(f"   Total de livros: {summary['total_books']}")
    print(f"   Avaliação média: {summary['average_rating']:.2f}")
    print(f"   Distribuição: {summary['ratings_distribution']}")