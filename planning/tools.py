import pandas as pd

# ---------------------------------------------------------------------------
# Este script assume que as seguintes variáveis GLOBAIS já foram carregadas:
#
# - users_df: pd.DataFrame com os dados dos usuários da amostra.
# - anime_df: pd.DataFrame com os metadados dos animes.
# - animelist_df: pd.DataFrame com as interações da amostra.
# - retriever_users: Objeto retriever do ChromaDB para perfis de usuário.
# - retriever_anime: Objeto retriever do ChromaDB para perfis de anime.
# - retriever_interactions: Objeto retriever do ChromaDB para interações.
# ---------------------------------------------------------------------------


def get_user_profile(username: str) -> dict:
    """
    Returns the complete profile of a user.
    """
    try:
        user_data = users_df[users_df['username_lower'] == username.lower()].iloc[0]
        return user_data.to_dict()
    except IndexError:
        return {"error": f"User '{username}' not found."}


def get_anime_details(anime_title: str) -> dict:
    """
    Fetches the complete details of an anime.
    """
    try:
        anime_data = anime_df[anime_df['title_lower'] == anime_title.lower()].iloc[0]
        return anime_data.to_dict()
    except IndexError:
        return {"error": f"Anime '{anime_title}' not found."}


def find_similar_user_profiles(username: str, k: int = 5) -> list:
    """
    Finds users with similar demographic and behavioral profiles using vector search.
    """
    try:
        user_row = users_df[users_df['username_lower'] == username.lower()].iloc[0]
        location = user_row.get('location', 'Location Unknown')
        gender = user_row.get('gender', 'Gender Unknown')
        birth_year = pd.to_datetime(user_row['birth_date']).year
        
        query_text = f"User from {location}. Gender {gender}. Born in {birth_year}."

        retriever_users.search_kwargs['k'] = k + 1
        similar_docs = retriever_users.invoke(query_text)

        similar_profiles = [
            doc.metadata for doc in similar_docs
            if doc.metadata.get('username', '').lower() != username.lower()
        ]
        
        return similar_profiles[:k]

    except (IndexError, ValueError, TypeError):
        return [{"error": f"Could not process profile for user '{username}'."}]


def find_similar_animes(anime_title: str, k: int = 5) -> list:
    """
    Finds animes with similar content using vector similarity search.
    """
    try:
        anime_row = anime_df[anime_df['title_lower'] == anime_title.lower()].iloc[0]
        title = anime_row.get('title_english') or anime_row.get('title', '')
        genres = anime_row.get('genre', '')
        studio = anime_row.get('studio', '')
        rating = anime_row.get('rating', '')
        background = anime_row.get('background', '')

        query_text = f"Title: {title}. Genres: {genres}. Studio: {studio}. Rating: {rating}. Background: {background}"

        retriever_anime.search_kwargs['k'] = k + 1
        similar_docs = retriever_anime.invoke(query_text)

        recommendations = [
            doc.metadata for doc in similar_docs
            if doc.metadata.get('title', '').lower() != anime_title.lower()
        ]
        
        return recommendations[:k]

    except IndexError:
        return [{"error": f"Anime '{anime_title}' not found in the dataset."}]


def get_recommendations_from_users(usernames: list, exclude_username: str, top_n: int = 10) -> list:
    """
    Suggests animes that a group of users liked, excluding animes already seen by a target user.
    """
    merged_df = pd.merge(animelist_df, anime_df[['anime_id', 'title']], on='anime_id', how='left')
    
    target_user_watched = set(
        merged_df[merged_df['username_lower'] == exclude_username.lower()]['title']
    )
    
    recs_df = merged_df[
        (merged_df['username'].isin(usernames)) &
        (merged_df['my_score'] >= 8)
    ]
    
    recs_df = recs_df[~recs_df['title'].isin(target_user_watched)]

    if recs_df.empty:
        return []
    
    top_recs = recs_df.groupby('title').size().sort_values(ascending=False).head(top_n)
    return top_recs.index.tolist()


def find_recommendations_from_opinion_peers(username: str, anime_title: str, k: int = 10) -> list:
    """
    Finds recommended animes from a group of users who share a similar opinion about a specific anime.
    """
    try:
        anime_row = anime_df[anime_df['title_lower'] == anime_title.lower()].iloc[0]
        anime_id = anime_row['anime_id']
        
        interaction_row = animelist_df[
            (animelist_df['username_lower'] == username.lower()) & 
            (animelist_df['anime_id'] == anime_id)
        ].iloc[0]

        score = interaction_row.get('my_score', 0)
        status_code = interaction_row.get('my_status', 0)
        tags = interaction_row.get('my_tags', 'no tags')
        status_map = {1: "Watching", 2: "Completed", 3: "On-Hold", 4: "Dropped", 6: "Plan to Watch"}
        status_text = status_map.get(status_code, "Unknown Status")
        score_text = f"Rated this anime {score} out of 10." if score > 0 else "This anime is unrated."
        query_text = f"{score_text} Status is {status_text}. User tags: {tags}"
        
        similar_docs = retriever_interactions.invoke(query_text)
        
        similar_usernames = list(set([
            doc.metadata['username'] for doc in similar_docs 
            if doc.metadata['username'].lower() != username.lower()
        ]))
        
        if not similar_usernames:
            return [{"error": "No users with a similar opinion were found."}]

        recommendations = get_recommendations_from_users(
            usernames=similar_usernames, 
            exclude_username=username, 
            top_n=k
        )
        return recommendations

    except IndexError:
        return [{"error": f"Could not find an interaction for user '{username}' with anime '{anime_title}'."}]


def find_recs_from_profile_peers_who_watched_anime(
    target_username: str, 
    watched_anime_title: str, 
    k_similar_users: int = 50, 
    k_final_recs: int = 10
) -> list:
    """
    Finds anime recommendations from a group of users who are demographically similar
    and have also watched a specific anime.
    """
    try:
        user_row = users_df[users_df['username_lower'] == target_username.lower()].iloc[0]
        user_query_text = f"User from {user_row.get('location', 'Unknown')}. Gender {user_row.get('gender', 'Unknown')}. Born in {pd.to_datetime(user_row['birth_date']).year}."
        
        retriever_users.search_kwargs['k'] = k_similar_users
        similar_profile_docs = retriever_users.invoke(user_query_text)
        similar_profile_users = set(doc.metadata['username'] for doc in similar_profile_docs)

        if not similar_profile_users:
            return [{"error": "No users with a similar profile were found."}]

        anime_row = anime_df[anime_df['title_lower'] == watched_anime_title.lower()].iloc[0]
        anime_id = anime_row['anime_id']
        
        watchers_df = animelist_df[animelist_df['anime_id'] == anime_id]
        all_watchers = set(watchers_df['username'])
        
        final_peer_group = list(similar_profile_users.intersection(all_watchers))

        if not final_peer_group:
            return [{"error": f"No users with a similar profile to '{target_username}' also watched '{watched_anime_title}'."}]

        recommendations = get_recommendations_from_users(
            usernames=final_peer_group,
            exclude_username=target_username,
            top_n=k_final_recs
        )
        return recommendations

    except IndexError:
        return [{"error": f"Could not find data for user '{target_username}' or anime '{watched_anime_title}'."}]