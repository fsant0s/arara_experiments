
import pandas as pd
from collections import Counter
from pathlib import Path

# Encontrar a raiz do projeto automaticamente
def find_project_root(current_path=None):
    """
    Encontra a raiz do projeto 'arara_experiments' subindo na árvore de diretórios
    """
    if current_path is None:
        current_path = Path(__file__).absolute()
    
    for parent in current_path.parents:
        if parent.name == 'arara_experiments':
            return parent
    
    # Se não encontrar, assume que está na raiz atual
    return Path.cwd()

# Obter a raiz do projeto
project_root = find_project_root()

# Construir caminhos relativos à raiz do projeto
animes_users_df = pd.read_csv(project_root / "datasets/mal_2023/1000/animes_users_merged.csv")
users_df = pd.read_csv(project_root / "datasets/mal_2023/1000/users.csv")
anime_df = pd.read_csv(project_root / "datasets/mal_2023/1000/animes.csv")
users_df['username_lower'] = users_df['username'].str.lower()
anime_df['name_lower'] = anime_df['name'].str.lower()
animes_users_df['username_lower'] = animes_users_df['username'].str.lower()
animes_users_df['name_lower'] = animes_users_df['name'].str.lower()

anime_metadata_columns = [
    'anime_id', 'name', 'score', 'episodes', 'type', 'ranked', 
    'popularity', 'members', 'favorites', 'premiered', 'english_name'
]

user_metadata_columns = [
    'mal_id', 'username', 'gender', 'birthday', 'location', 'joined',
    'days_watched', 'mean_score', 'watching', 'completed', 'on_hold',
    'dropped', 'plan_to_watch', 'total_entries', 'rewatched', 'episodes_watched'
]  

anime_user_metadata_columns = [
    'username', 'user_id', 'anime_id', 'name', 'genres', 'my_score',
    'type', 'score', 'rank', 'popularity', 'episodes', 'rating'
]

def get_user_by_username(username):
    """
    Retrieve a user from the users_df DataFrame by username.
    """
    username_lower = username.lower()
    user = users_df[users_df['username_lower'] == username_lower]
    
    if not user.empty:
        return user
    else:
        return None
    
def get_users_who_watched_anime(anime_id):
    """
    Retrieve all users who have watched a specific anime by anime_id.
    """
    watchers_df = animes_users_df[animes_users_df['anime_id'] == anime_id]
    all_watchers = set(watchers_df['username'])
    return all_watchers

def get_animes_watched_by_user(username):
    target_user_watched = set(
        animes_users_df[animes_users_df['username_lower'] == username.lower()]['name']
    )
    return target_user_watched

def get_animes_by_list_of_usernames(usernames, score=8):
    recs_df = animes_users_df[
        (animes_users_df['username'].isin(usernames)) &
        (animes_users_df['my_score'] >= score)
    ]
    return recs_df

def get_anime_user_by_username(username, anime_name):
    """
    Retrieve an anime-user interaction from the animes_users_df DataFrame by username.
    """
    username_lower = username.lower()
    user_anime = animes_users_df[(animes_users_df['username_lower'] == username_lower) & (animes_users_df['name_lower'] == anime_name.lower())]
    
    if not user_anime.empty:
        return user_anime
    else:
        return None
    
def get_anime_by_name(name):
    """
    Retrieve an anime from the anime_df DataFrame by name.
    """
    name_lower = name.lower()
    anime = anime_df[anime_df['name_lower'] == name_lower]
    
    if not anime.empty:
        return anime.iloc[0]
    else:
        return None

def build_anime_vector_document(row):

    name = row.get('name', 'No Title')
    english_name = row.get('english_name', 'N/A')
    genres = row.get('genres', 'No Genres')
    synopsis = row.get('sypnopsis', 'No synopsis available.')
    anime_type = row.get('type', 'N/A')
    source = row.get('source', 'N/A')
    rating = row.get('rating', 'Not Rated')
    producers = row.get('producers', 'Unknown')
    studios = row.get('studios', 'Unknown')

    page_content = (
        f"Anime: {name}. English Title: {english_name}. "
        f"Genres: {genres}. "
        f"Synopsis: {synopsis} "
        f"Production Details: This is a {anime_type} series adapted from a {source} source. "
        f"It is rated {rating}. "
        f"Produced by {producers} and animated by {studios}."
    )
    metadata = {col: row[col] for col in anime_metadata_columns if col in row and pd.notna(row[col])}
    return page_content, metadata


def build_user_vector_document(row):

    username = row.get('username', 'Unknown User')
    mean_score = row.get('mean_score', 0)
    rewatched_count = row.get('rewatched', 0)
    gender = row.get('gender', 'Unknown')
    location = row.get('location', 'Unknown')
    days_watched = row.get('days_watched', 0)
    total_entries = row.get('total_entries', 0)
    completed = row.get('completed', 0)
    watching = row.get('watching', 0)
    dropped = row.get('dropped', 0)
    plan_to_watch = row.get('plan_to_watch', 0)
    episodes_watched = row.get('episodes_watched', 0)

    # Determinar descrição do score
    score_desc = "a moderate viewer"
    if mean_score >= 8.5:
        score_desc = "a highly critical viewer"
    elif mean_score >= 7.5:
        score_desc = "a discerning viewer"
    elif mean_score < 6.5 and mean_score > 0:
        score_desc = "a generous viewer"
    
    # Calcular taxa de completion
    completion_rate = (completed / total_entries * 100) if total_entries > 0 else 0
    drop_rate = (dropped / total_entries * 100) if total_entries > 0 else 0
    
    # Determinar tipo de viewer baseado nos hábitos
    viewer_type = "casual"
    if days_watched > 100:
        viewer_type = "dedicated"
    elif days_watched > 50:
        viewer_type = "regular"
    
    # Obter preferências detalhadas do usuário usando animes_users
    user_animes = animes_users_df[animes_users_df['username'] == username]
    preferred_genres = "No specific preferences identified"
    preferred_studios = "No studio preferences identified"
    preferred_types = "No type preferences identified"
    preferred_sources = "No source preferences identified"
    preferred_ratings = "No rating preferences identified"
    popularity_preference = "balanced"
    score_behavior = "standard"
    
    if len(user_animes) > 0:
        # Pegar animes bem avaliados (score >= 7) para identificar preferências
        high_rated = user_animes[user_animes['my_score'] >= 7]
        
        if len(high_rated) > 0:
            
            # 1. GÊNEROS PREFERIDOS
            genres_list = []
            for genres_str in high_rated['genres'].dropna():
                if isinstance(genres_str, str) and genres_str != 'N/A':
                    individual_genres = [g.strip() for g in genres_str.split(',')]
                    genres_list.extend(individual_genres)
            
            if genres_list:
                genre_counts = Counter(genres_list)
                top_genres = [genre for genre, count in genre_counts.most_common(5)]
                preferred_genres = ", ".join(top_genres)
            
            # 2. ESTÚDIOS PREFERIDOS
            studios_list = []
            for studios_str in high_rated['studios'].dropna():
                if isinstance(studios_str, str) and studios_str != 'N/A':
                    individual_studios = [s.strip() for s in studios_str.split(',')]
                    studios_list.extend(individual_studios)
            
            if studios_list:
                studio_counts = Counter(studios_list)
                top_studios = [studio for studio, count in studio_counts.most_common(3)]
                preferred_studios = ", ".join(top_studios)
            
            # 3. TIPOS PREFERIDOS (TV, Movie, OVA, etc.)
            type_counts = Counter(high_rated['type'].dropna())
            if type_counts:
                top_types = [type_name for type_name, count in type_counts.most_common(3)]
                preferred_types = ", ".join(top_types)
            
            # 4. FONTES PREFERIDAS (Manga, Light Novel, etc.)
            source_counts = Counter(high_rated['source'].dropna())
            if source_counts:
                top_sources = [source for source, count in source_counts.most_common(3)]
                preferred_sources = ", ".join(top_sources)
            
            # 5. CLASSIFICAÇÃO ETÁRIA PREFERIDA
            rating_counts = Counter(high_rated['rating'].dropna())
            if rating_counts:
                top_ratings = [rating for rating, count in rating_counts.most_common(2)]
                preferred_ratings = ", ".join(top_ratings)
            
            # 6. PREFERÊNCIA POR POPULARIDADE (mainstream vs nicho)
            avg_popularity = high_rated['popularity'].dropna().mean()
            if pd.notna(avg_popularity):
                if avg_popularity <= 100:
                    popularity_preference = "mainstream popular anime"
                elif avg_popularity <= 1000:
                    popularity_preference = "moderately popular anime"
                else:
                    popularity_preference = "niche and lesser-known anime"
            
            # 7. COMPORTAMENTO DE AVALIAÇÃO (comparação com score geral)
            user_scores = high_rated['my_score'].dropna()
            general_scores = high_rated['score'].dropna()
            
            if len(user_scores) > 0 and len(general_scores) > 0:
                avg_user_score = user_scores.mean()
                avg_general_score = general_scores.mean()
                score_diff = avg_user_score - avg_general_score
                
                if score_diff > 0.5:
                    score_behavior = "tends to rate higher than general consensus"
                elif score_diff < -0.5:
                    score_behavior = "tends to rate lower than general consensus"
                else:
                    score_behavior = "aligns with general consensus"
    
    page_content = (
        f"Profile for user {username}. "
        f"Gender: {gender}, Location: {location}. "
        f"This user is {score_desc}, with a mean score of {mean_score:.2f}. "
        f"They are a {viewer_type} anime viewer who has spent {days_watched} days watching anime "
        f"and has watched {episodes_watched} episodes across {total_entries} anime entries. "
        f"Their viewing habits: {completed} completed ({completion_rate:.1f}%), "
        f"{watching} currently watching, {dropped} dropped ({drop_rate:.1f}%), "
        f"{plan_to_watch} planned to watch. "
        f"They have rewatched {rewatched_count} animes, showing their engagement level. "
        f"Content Preferences: Their favorite genres include {preferred_genres}. "
        f"They prefer anime from studios like {preferred_studios}. "
        f"Format preferences: {preferred_types}. "
        f"Source material preferences: {preferred_sources}. "
        f"Rating preferences: {preferred_ratings}. "
        f"They tend to watch {popularity_preference}. "
        f"Rating behavior: {score_behavior}."
    )

    metadata = {col: row[col] for col in user_metadata_columns if col in row and pd.notna(row[col])}
    
    return page_content, metadata

def build_anime_user_vector_document(row):

    username = getattr(row, 'username', 'N/A')
    gender = getattr(row, 'gender', 'N/A')
    mean_score = getattr(row, 'mean_score', 0)
    total_entries = getattr(row, 'total_entries', 0)
    days_watched = getattr(row, 'days_watched', 0)
    
    # Dados do anime
    anime_name = getattr(row, 'name', 'N/A')
    english_name = getattr(row, 'english_name', 'N/A')
    japanese_name = getattr(row, 'japanese_name', 'N/A')
    anime_genres = getattr(row, 'genres', 'N/A')
    anime_studios = getattr(row, 'studios', 'N/A')
    anime_type = getattr(row, 'type', 'N/A')
    anime_source = getattr(row, 'source', 'N/A')
    anime_rating = getattr(row, 'rating', 'N/A')
    anime_score = getattr(row, 'score', 0)
    anime_rank = getattr(row, 'rank', 'N/A')
    anime_popularity = getattr(row, 'popularity', 'N/A')
    episodes = getattr(row, 'episodes', 'N/A')
    aired = getattr(row, 'aired', 'N/A')
    producers = getattr(row, 'producers', 'N/A')
    anime_synopsis = getattr(row, 'sypnopsis', 'N/A')
    
    # Interação específica
    my_score = getattr(row, 'my_score', 0)
    
    # Determinar tipo de viewer
    viewer_type = "casual"
    if days_watched > 100:
        viewer_type = "dedicated"
    elif days_watched > 50:
        viewer_type = "regular"
    
    # Determinar crítica do usuário
    score_tendency = "moderate"
    if mean_score >= 8.5:
        score_tendency = "highly critical"
    elif mean_score >= 7.5:
        score_tendency = "discerning"
    elif mean_score < 6.5 and mean_score > 0:
        score_tendency = "generous"
        
    page_content = (
        f"This is a review from user '{username}', a {gender} {viewer_type} viewer "
        f"who is {score_tendency} with an average score of {mean_score:.2f} across {total_entries} anime entries. "
        f"The reviewed anime is '{anime_name}' (English: {english_name}, Japanese: {japanese_name}), "
        f"a {anime_type} work from {anime_source} source in the {anime_genres} genres. "
        f"This anime has {episodes} episodes, aired {aired}, produced by {producers} "
        f"and animated by {anime_studios} with a {anime_rating} rating. "
        f"It has an overall score of {anime_score}, ranked #{anime_rank} with popularity #{anime_popularity}. "
        f"The user gave this anime a score of {my_score}. "
        f"Synopsis: {anime_synopsis}"
    )
    
    metadata = {col: getattr(row, col, None) for col in anime_user_metadata_columns if hasattr(row, col) and pd.notna(getattr(row, col, None))}
    
    return page_content, metadata