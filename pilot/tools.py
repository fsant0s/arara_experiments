
from retrievers.retrievers import retriever_users, retriever_anime, retriever_animes_users
from retrievers.utils import get_anime_by_name, get_user_by_username, build_user_vector_document, build_anime_user_vector_document, build_anime_vector_document, get_anime_user_by_username, get_users_who_watched_anime, get_animes_watched_by_user, get_animes_by_list_of_usernames

import pandas as pd

def get_user_profile(username: str) -> str:
    """
    Retrieves and formats the essential user profile information for anime recommendations.
    
    This function fetches a user's key demographic information, viewing statistics, and 
    a simplified list of their watched anime with ratings, optimized for recommendation algorithms.

    Parameters:
        username (str): The username to retrieve the profile for. Case-insensitive matching 
                       with usernames in the dataset.
      
    Returns:
        str: A concise formatted string containing:
             - Basic demographics: username, gender, location
             - Key viewing statistics: mean score, total entries, completion stats
             - Simplified anime list: name, user rating, general score, genres
             - Returns error message string if user not found in the dataset
             
    Note:
        This function provides only essential information needed for anime recommendation analysis.
    """
    user_data = get_animes_by_list_of_usernames([username])
    if not len(user_data):
        return f"User '{username}' not found in dataset."
    
    user_info = user_data.iloc[0]
    
    result = f"USER PROFILE: {username.upper()}\n"
    result += "=" * 40 + "\n\n"
    
    # Essential user information
    result += f"Username: {user_info['username']}\n"
    result += f"Gender: {user_info['gender']}\n"
    result += f"Location: {user_info.get('location', 'N/A')}\n\n"
    
    # Key viewing statistics
    result += "VIEWING STATS:\n"
    result += f"• Mean score: {user_info['mean_score']:.2f}\n"
    result += f"• Total entries: {user_info.get('total_entries', 'N/A')}\n"
    result += f"• Completed: {user_info.get('completed', 'N/A')}\n"
    result += f"• Dropped: {user_info.get('dropped', 'N/A')}\n\n"
    
    result += f"ANIME WATCHED ({len(user_data)} total):\n"
    result += "-" * 40 + "\n"
    
    # Simplified anime list - only essential recommendation data
    for idx, row in user_data.iterrows():
        anime_name = row['name'] if pd.notna(row['name']) else 'N/A'
        my_score = row['my_score'] if pd.notna(row['my_score']) else 'N/A'
        anime_score = row['score'] if pd.notna(row['score']) else 'N/A'
        genres = row['genres'] if pd.notna(row['genres']) else 'N/A'
        
        result += f"{anime_name} | User: {my_score} | General: {anime_score} | {genres}\n"
    
    return result

def get_anime_details(anime_name: str) -> str:
    """
    Fetches the complete details of an anime.
    
    Parameters:
        anime_name (str): The name of the anime to search for (case-insensitive).
        
    Returns:
        str: A detailed formatted string containing all anime details including:
             - anime_id, name, score, genres, english_name, japanese_name
             - synopsis, type, episodes, aired date, premiered season
             - studios, producers, licensors, source material
             - rating, ranked position, popularity, members, favorites
             - viewing statistics: watching, completed, on-hold, dropped
             - Returns error message string if anime not found
    """
    anime_data = get_anime_by_name(anime_name)
    if anime_data is None:
        return f"Error: Anime '{anime_name}' not found in dataset."
    
    # Convert to dictionary if necessary
    if hasattr(anime_data, 'to_dict'):
        anime_dict = anime_data.to_dict()
    else:
        anime_dict = anime_data
    
    # Output string formatting
    result = "=" * 80 + "\n"
    result += f"ANIME DETAILS: {anime_dict.get('name', 'N/A').upper()}\n"
    result += "=" * 80 + "\n\n"
    
    # Basic information
    result += "BASIC INFORMATION:\n"
    result += f"• ID: {anime_dict.get('anime_id', 'N/A')}\n"
    result += f"• Name: {anime_dict.get('name', 'N/A')}\n"
    result += f"• English Name: {anime_dict.get('english_name', 'N/A')}\n"
    result += f"• Japanese Name: {anime_dict.get('japanese_name', 'N/A')}\n"
    result += f"• Score: {anime_dict.get('score', 'N/A')}\n"
    result += f"• Ranking: {anime_dict.get('ranked', 'N/A')}\n"
    result += f"• Popularity: {anime_dict.get('popularity', 'N/A')}\n\n"
    
    # Production details
    result += "PRODUCTION DETAILS:\n"
    result += f"• Type: {anime_dict.get('type', 'N/A')}\n"
    result += f"• Episodes: {anime_dict.get('episodes', 'N/A')}\n"
    result += f"• Duration: {anime_dict.get('duration', 'N/A')}\n"
    result += f"• Aired: {anime_dict.get('aired', 'N/A')}\n"
    result += f"• Premiered: {anime_dict.get('premiered', 'N/A')}\n"
    result += f"• Studios: {anime_dict.get('studios', 'N/A')}\n"
    result += f"• Producers: {anime_dict.get('producers', 'N/A')}\n"
    result += f"• Licensors: {anime_dict.get('licensors', 'N/A')}\n"
    result += f"• Source: {anime_dict.get('source', 'N/A')}\n"
    result += f"• Rating: {anime_dict.get('rating', 'N/A')}\n\n"
    
    # Genres
    result += "GENRES:\n"
    result += f"• {anime_dict.get('genres', 'N/A')}\n\n"
    
    # Statistics
    result += "VIEWING STATISTICS:\n"
    result += f"• Members: {anime_dict.get('members', 'N/A'):,}\n" if anime_dict.get('members') else "• Members: N/A\n"
    result += f"• Favorites: {anime_dict.get('favorites', 'N/A'):,}\n" if anime_dict.get('favorites') else "• Favorites: N/A\n"
    result += f"• Watching: {anime_dict.get('watching', 'N/A'):,}\n" if anime_dict.get('watching') else "• Watching: N/A\n"
    result += f"• Completed: {anime_dict.get('completed', 'N/A'):,}\n" if anime_dict.get('completed') else "• Completed: N/A\n"
    result += f"• On-Hold: {anime_dict.get('on-hold', 'N/A'):,}\n" if anime_dict.get('on-hold') else "• On-Hold: N/A\n"
    result += f"• Dropped: {anime_dict.get('dropped', 'N/A'):,}\n" if anime_dict.get('dropped') else "• Dropped: N/A\n"
    
    # Synopsis
    result += "\nSYNOPSIS:\n"
    synopsis = anime_dict.get('sypnopsis', 'Synopsis not available.')
    result += f"{synopsis}\n"
    
    result += "\n" + "=" * 80
    
    return result

def find_similar_user_profiles(username: str, k: int = 10) -> str:
    """
    Finds users with similar demographic and behavioral profiles using vector search.
    
    Parameters:
        username (str): The target username to find similar users for (case-insensitive).
        k (int, optional): Number of similar users to return. Defaults to 10.
        
    Returns:
        str: A formatted string containing the list of similar usernames.
             - Users are ranked by similarity based on location, gender, and birth year
             - Excludes the target user from results
             - Returns error message string if user not found or no similar users found
    """
    try:
        user_row = get_user_by_username(username)
        page_content, _ = build_user_vector_document(user_row.iloc[0])
        retriever_users.search_kwargs['k'] = k + 1
        similar_docs = retriever_users.invoke(page_content)

        similar_usernames = [
            doc.metadata.get('username', '')
            for doc in similar_docs
            if doc.metadata.get('username', '').lower() != username.lower()
        ]
        
        if not similar_usernames:
            return f"No similar users found for '{username}'."
            
        # Format as a string list
        result = f"SIMILAR USERS TO '{username.upper()}':\n"
        result += "=" * 50 + "\n"
        for i, user in enumerate(similar_usernames[:k], 1):
            result += f"{i}. {user}\n"
        
        return result
        
    except (IndexError, ValueError, TypeError):
        return f"Error: Could not process profile for user '{username}'."

def find_similar_animes(anime_name: str, k: int = 5) -> str:
    """
    Finds animes with similar content using vector similarity search.
    
    Parameters:
        anime_name (str): The name of the anime to find similar content for (case-insensitive).
        k (int, optional): Number of similar animes to return. Defaults to 5.
        
    Returns:
        str: A formatted string containing the list of similar anime names.
             - Animes are ranked by content similarity based on title, genres, studios, rating, and synopsis
             - Excludes the input anime from results
             - Returns error message string if anime not found in the dataset
    """
    try:
        anime_row = get_anime_by_name(anime_name)
        page_content, _ = build_anime_vector_document(anime_row)
        
        retriever_anime.search_kwargs['k'] = k + 1
        similar_docs = retriever_anime.invoke(page_content)

        similar_anime_names = [
            doc.metadata.get('name', '')
            for doc in similar_docs
            if doc.metadata.get('name', '').lower() != anime_name.lower()
        ]
        
        if not similar_anime_names:
            return f"No similar animes found for '{anime_name}'."
            
        # Format as a string list
        result = f"SIMILAR ANIMES TO '{anime_name.upper()}':\n"
        result += "=" * 50 + "\n"
        for i, anime in enumerate(similar_anime_names[:k], 1):
            result += f"{i}. {anime}\n"
        
        return result

    except IndexError:
        return f"Error: Anime '{anime_name}' not found in the dataset."

def get_recommendations_from_users(usernames: list, exclude_username: str, top_n: int = 10) -> str:
    """
    Suggests animes that a group of users liked, excluding animes already seen by a target user.
    
    Parameters:
        usernames (list): List of usernames whose anime preferences to consider for recommendations.
        exclude_username (str): Username of the target user to exclude their watched animes from recommendations.
        top_n (int, optional): Maximum number of anime recommendations to return. Defaults to 10.
        
    Returns:
        str: A formatted string containing the list of recommended anime names.
             - Animes are sorted by popularity among the specified users (most recommended first)
             - Excludes animes already watched by the target user
             - Returns message string if no recommendations found or specified users have no high-rated animes
    """
    target_user_watched = get_animes_watched_by_user(exclude_username)
    recs_df = get_animes_by_list_of_usernames(usernames)
    
    recs_df = recs_df[~recs_df['name'].isin(target_user_watched)]

    if recs_df.empty:
        return f"No recommendations found for '{exclude_username}' from the specified users."
    
    top_recs = recs_df.groupby('name').size().sort_values(ascending=False).head(top_n)
    
    if top_recs.empty:
        return f"No recommendations found for '{exclude_username}' from the specified users."
    
    # Format as a string list
    result = f"RECOMMENDATIONS FOR '{exclude_username.upper()}' FROM {len(usernames)} USERS:\n"
    result += "=" * 60 + "\n"
    for i, anime in enumerate(top_recs.index.tolist(), 1):
        result += f"{i}. {anime}\n"
    
    return result


def find_recommendations_from_opinion_peers(username: str, anime_name: str, k: int = 10) -> list:
    """
    Finds recommended animes from a group of users who share a similar opinion about a specific anime.
    
    Parameters:
        username (str): The target username whose opinion will be used to find similar users.
        anime_name (str): The name of the anime to base opinion similarity on (case-insensitive).
        k (int, optional): Maximum number of anime recommendations to return. Defaults to 10.
        
    Returns:
        list: List of anime names (strings) recommended by users with similar opinions.
              - Finds users who rated/tagged the specified anime similarly to the target user
              - Returns recommendations from these opinion-similar users (high-rated animes >=8)
              - Excludes animes already watched by the target user
              - Returns [{"error": "message"}] if user-anime interaction not found or no similar users found
    """
    try:
        user_row = get_anime_user_by_username(username, anime_name)

        if user_row.empty:
            return [{"error": f"User '{username}' not found in users database."}]
        page_content, _ = build_anime_user_vector_document(user_row.iloc[0])
    
        similar_docs = retriever_animes_users.invoke(page_content)
        
        # Extrair usernames dos metadados dos documentos
        similar_usernames = [doc.metadata.get('username') for doc in similar_docs if doc.metadata.get('username')]
        similar_usernames = list(dict.fromkeys(similar_usernames))  # Remove duplicatas
      
        if not similar_usernames:
            return [{"error": "No users with a similar opinion were found among those who watched this anime."}]
       
        recommendations = get_recommendations_from_users(
            usernames=similar_usernames, 
            exclude_username=username, 
            top_n=k
        )
        return recommendations

    except IndexError:
        return [{"error": f"Could not find an interaction for user '{username}' with anime '{anime_name}'."}]


def find_recs_from_profile_peers_who_watched_anime(
    target_username: str, 
    watched_anime_name: str, 
    k_similar_users: int = 50, 
    k_final_recs: int = 10
) -> list:
    """
    Finds anime recommendations from a group of users who are demographically similar
    and have also watched a specific anime.
    
    Parameters:
        target_username (str): The username to find recommendations for.
        watched_anime_name (str): Name of an anime that both the target user and similar users have watched.
        k_similar_users (int, optional): Number of similar users to consider in the search. Defaults to 50.
        k_final_recs (int, optional): Maximum number of final recommendations to return. Defaults to 10.
        
    Returns:
        list: List of anime names (strings) recommended by demographically similar users who also watched the specified anime.
              - First finds users similar to target user based on location, gender, and birth year
              - Then filters to only users who have also watched the specified anime
              - Returns high-rated recommendations (>=8) from this filtered group
              - Excludes animes already watched by the target user
              - Returns [{"error": "message"}] if user/anime not found or no qualifying users found
    """
    try:
        user_row = get_user_by_username(target_username)
        page_content, _ = build_user_vector_document(user_row.iloc[0])
        
        retriever_users.search_kwargs['k'] = k_similar_users
        similar_profile_docs = retriever_users.invoke(page_content)
        similar_profile_users = set(doc.metadata['username'] for doc in similar_profile_docs)

        if not similar_profile_users:
            return [{"error": "No users with a similar profile were found."}]

        anime_row = get_anime_by_name(watched_anime_name)
        anime_id = anime_row['anime_id']
        
        all_watchers = get_users_who_watched_anime(anime_id)
        
        final_peer_group = list(similar_profile_users.intersection(all_watchers))

        if not final_peer_group:
            return [{"error": f"No users with a similar profile to '{target_username}' also watched '{watched_anime_name}'."}]

        recommendations = get_recommendations_from_users(
            usernames=final_peer_group,
            exclude_username=target_username,
            top_n=k_final_recs
        )
        return recommendations

    except IndexError:
        return [{"error": f"Could not find data for user '{target_username}' or anime '{watched_anime_name}'."}]


def tool_descriptions():
    """
    Returns the docstrings of all functions defined in this module.
    
    Parameters:
        None
    
    Returns:
        str: A formatted string containing all function docstrings from this module.
             - Each function's name, docstring, and parameters/returns are clearly formatted
             - Excludes imported functions and the tool_descriptions function itself
             - Provides a comprehensive overview of all available tools and their usage
    """
    import inspect
    
    # Get the current module
    current_module = inspect.getmodule(inspect.currentframe())
    
    # Get all functions defined in this module
    functions = inspect.getmembers(current_module, inspect.isfunction)
    result = "FUNCTION DESCRIPTIONS:\n\n"
    for func_name, func_obj in functions:
        # Only include functions defined in this module (not imported ones)
        # and exclude the tool_descriptions function itself
        if func_name == tool_descriptions.__name__:
            continue
        if func_obj.__module__ == current_module.__name__ and func_name != 'tool_descriptions':
            docstring = func_obj.__doc__
            
            result += f"FUNCTION: {func_name}\n"
            result += "-" * 40 + "\n"
            
            if docstring:
                result += f"DOCSTRING:\n{docstring}\n"
            else:
                result += "No docstring available.\n"
            
            result += "\n" + "=" * 40 + "\n\n"
    
    return result

    