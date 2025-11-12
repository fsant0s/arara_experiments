#!/usr/bin/env python3
"""
Creates movie_info_filtered.jsonl with ALL movies from movies.dat
RULE: Title always comes from movies.dat, extra data comes from movie_info.jsonl
"""
import json
import re

def normalize_for_matching(title):
    """Normalization for matching"""
    normalized = title.replace('"', '').replace("'", '')
    normalized = normalized.replace('–', '').replace('—', '').replace(':', '')
    normalized = normalized.replace('-', '').replace('!', '').replace('?', '')
    normalized = re.sub(r'\s*\(\d{4}\)\s*', '', normalized)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
    return normalized

# 1. Load movies.dat (main title source)
print("Loading movies.dat...")
movies_dat = {}
with open('dataset/movie/movies.dat', 'r', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split('::')
        if len(parts) >= 3:
            movie_id = int(parts[0])
            title = parts[1]  # Title WITH year from movies.dat
            genres = parts[2].split('|')
            movies_dat[movie_id] = {
                'title': title,
                'genres': genres,
                'normalized': normalize_for_matching(title),
                'extra_data': None
            }

print(f"✓ {len(movies_dat)} movies")

# 2. Load movie_info.jsonl for extra data
print("\nProcessing movie_info.jsonl...")
processed = 0
with open('dataset/movie/movie_info.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            movie_data = json.loads(line)
            title = movie_data.get('Title')
            if not title:
                continue
            
            normalized = normalize_for_matching(title)
            
            # Search for a match in movies.dat
            for movie_id, dat_movie in movies_dat.items():
                if dat_movie['normalized'] == normalized and not dat_movie['extra_data']:
                    # Store extra data (without the title)
                    extra = {k: v for k, v in movie_data.items() if k != 'Title'}
                    dat_movie['extra_data'] = extra
                    break
            
            processed += 1
            if processed % 5000 == 0:
                print(f"  {processed} lines", end='\r')
                
        except:
            pass

print(f"\n✓ {processed} lines processed")

# 3. Create file - Title from movies.dat + data from movie_info
print("\nCreating movie_info_filtered.jsonl...")
found_with_data = 0
only_basic = 0

with open('dataset/movie/movie_info_filtered.jsonl', 'w', encoding='utf-8') as f:
    for movie_id, dat_movie in sorted(movies_dat.items()):
        # ALWAYS use the title from movies.dat
        merged_movie = {'Title': dat_movie['title']}
        
        if dat_movie['extra_data']:
            # Add extra data from movie_info.jsonl
            merged_movie.update(dat_movie['extra_data'])
            found_with_data += 1
        else:
            # Only basic data
            merged_movie['Genre'] = '|'.join(dat_movie['genres'])
            only_basic += 1
        
        f.write(json.dumps(merged_movie) + '\n')

print(f"\n{'='*70}")
print("RESULT")
print(f"{'='*70}")
print(f"✅ With extra data:     {found_with_data:,}")
print(f"📝 Only basic data:     {only_basic:,}")
print(f"📊 TOTAL:               {found_with_data + only_basic:,} / {len(movies_dat):,}")
print(f"\n✅ File: dataset/movie/movie_info_filtered.jsonl")
print(f"   ALL titles with format: Title (YYYY)")
