import os
import sys
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configurações         
database = 'neo4j'  # Nome do database no Neo4j
query_type = 'condition'
eval_type = 'movie-ExplicitQuery'  # ou 'movie-ImplicitQuery'

# Caminhos relativos ao diretório do script
datasets_dir = os.path.join(PROJECT_ROOT, 'datasets', 'RecAssistBench')
groundtruths = os.path.join(datasets_dir, 'dataset', 'movie', 'ExplicitQuery.json')
schema_file = os.path.join(datasets_dir, 'eval', 'movie-schema.json')
eval_script = os.path.join(datasets_dir, 'eval', 'eval_movie.py')
output_dir = os.path.join(datasets_dir, 'eval_results')

# Credenciais Neo4j (mesmas usadas em neo4j_client.py)
neo4j_uri = "neo4j://127.0.0.1:7687"
neo4j_username = "neo4j"
neo4j_password = "arara123"

print(f"Procurando resultados do tipo: {eval_type}")
print(f"Dataset: {groundtruths}")
print(f"Schema: {schema_file}")
print()

# Procura por arquivos de predição do arara
llm_results_dir = os.path.join(CURRENT_DIR, 'llm_results', 'arara')

if not os.path.exists(llm_results_dir):
    print(f"Diretório não encontrado: {llm_results_dir}")
    sys.exit(1)

found_files = []
for file in os.listdir(llm_results_dir):
    if eval_type in file and file.endswith('prediction.jsonl') and 'historyTrue' not in file:
        found_files.append(file)

if not found_files:
    print(f"Nenhum arquivo encontrado no diretório: {llm_results_dir}")
    sys.exit(1)

print(f"Encontrados {len(found_files)} arquivo(s):\n")

for file in found_files:
    predictions = os.path.join(llm_results_dir, file)
    print(f"Avaliando: {file}")
    
    cmd = [
        'python', eval_script,
        '--database', database,
        '--query_type', query_type,
        '--groundtruths', groundtruths,
        '--predictions', predictions,
        '--schema', schema_file,
        '--output_dir', output_dir,
        '--uri', neo4j_uri,
        '--username', neo4j_username,
        '--password', neo4j_password
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print('Avaliação concluída!')
    except subprocess.CalledProcessError as e:
        print(f'Erro ao executar avaliação: {e}')
    
    print('=' * 80)
    print()