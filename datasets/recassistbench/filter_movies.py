#!/usr/bin/env python3
"""
Cria movie_info_filtered.jsonl com TODOS os filmes do movies.dat
REGRA: Título sempre vem do movies.dat, dados extras vêm do movie_info.jsonl
"""
import json
import re

def normalize_for_matching(title):
    """Normalização para matching"""
    normalized = title.replace('"', '').replace("'", '')
    normalized = normalized.replace('–', '').replace('—', '').replace(':', '')
    normalized = normalized.replace('-', '').replace('!', '').replace('?', '')
    normalized = re.sub(r'\s*\(\d{4}\)\s*', '', normalized)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
    return normalized

# 1. Carregar movies.dat (fonte principal de títulos)
print("Carregando movies.dat...")
movies_dat = {}
with open('dataset/movie/movies.dat', 'r', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split('::')
        if len(parts) >= 3:
            movie_id = int(parts[0])
            title = parts[1]  # Título COM ano do movies.dat
            genres = parts[2].split('|')
            movies_dat[movie_id] = {
                'title': title,
                'genres': genres,
                'normalized': normalize_for_matching(title),
                'extra_data': None
            }

print(f"✓ {len(movies_dat)} filmes")

# 2. Carregar movie_info.jsonl para dados extras
print("\nProcessando movie_info.jsonl...")
processed = 0
with open('dataset/movie/movie_info.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            movie_data = json.loads(line)
            title = movie_data.get('Title')
            if not title:
                continue
            
            normalized = normalize_for_matching(title)
            
            # Buscar match no movies.dat
            for movie_id, dat_movie in movies_dat.items():
                if dat_movie['normalized'] == normalized and not dat_movie['extra_data']:
                    # Guardar dados extras (sem o título)
                    extra = {k: v for k, v in movie_data.items() if k != 'Title'}
                    dat_movie['extra_data'] = extra
                    break
            
            processed += 1
            if processed % 5000 == 0:
                print(f"  {processed} linhas", end='\r')
                
        except:
            pass

print(f"\n✓ {processed} linhas processadas")

# 3. Criar arquivo - Título do movies.dat + dados do movie_info
print("\nCriando movie_info_filtered.jsonl...")
found_with_data = 0
only_basic = 0

with open('dataset/movie/movie_info_filtered.jsonl', 'w', encoding='utf-8') as f:
    for movie_id, dat_movie in sorted(movies_dat.items()):
        # SEMPRE usar o título do movies.dat
        merged_movie = {'Title': dat_movie['title']}
        
        if dat_movie['extra_data']:
            # Adicionar dados extras do movie_info.jsonl
            merged_movie.update(dat_movie['extra_data'])
            found_with_data += 1
        else:
            # Apenas dados básicos
            merged_movie['Genre'] = '|'.join(dat_movie['genres'])
            only_basic += 1
        
        f.write(json.dumps(merged_movie) + '\n')

print(f"\n{'='*70}")
print("RESULTADO")
print(f"{'='*70}")
print(f"✅ Com dados extras:   {found_with_data:,}")
print(f"📝 Apenas básicos:     {only_basic:,}")
print(f"📊 TOTAL:              {found_with_data + only_basic:,} / {len(movies_dat):,}")
print(f"\n✅ Arquivo: dataset/movie/movie_info_filtered.jsonl")
print(f"   TODOS os títulos com formato: Title (YYYY)")

