# recommender.py
# Core recommendation logic for the CineAI app.
# Takes the precomputed similarity matrix and uses it to find
# movies similar to a given movie or matching a detected emotion.
# Three main functions: by movie ID, by title string, by emotion.

import numpy as np
from src.preprocess import get_movie_title, get_movie_genres


def get_recommendations(movie_id, sim_result, movies_df, top_n=10):
    """
    Get top-N most similar movies for a given movie ID.

    Uses the precomputed cosine similarity matrix to find movies
    with the highest similarity scores to the input movie.
    The input movie itself is excluded from the results.

    Parameters:
        movie_id: MovieLens movie ID to base recommendations on
        sim_result: dictionary containing the similarity matrix and movie index
        movies_df: the full movies dataframe for title/genre lookup
        top_n: number of recommendations to return

    Returns a list of dicts with movie_id, title, score, and genres.
    """
    sim_matrix = sim_result["matrix"]
    index = sim_result["index"]  # List of movie IDs in matrix order

    # Check if the requested movie is in our similarity matrix
    if movie_id not in index:
        return []

    # Get the row for this movie in the similarity matrix
    idx = index.index(movie_id)
    scores = list(enumerate(sim_matrix[idx]))

    # Sort all movies by similarity score (highest first)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Remove the input movie itself from results (it would have score 1.0)
    scores = [s for s in scores if index[s[0]] != movie_id]

    # Build the results list with movie details
    top = scores[:top_n]
    results = []
    for i, score in top:
        mid = index[i]
        title = get_movie_title(mid, movies_df)
        genres = get_movie_genres(mid, movies_df)
        results.append({
            "movie_id": int(mid),
            "title": title,
            "score": round(float(score), 4),
            "genres": genres
        })
    return results


def get_recommendations_by_title(title, sim_result, movies_df, top_n=10):
    """
    Get recommendations by searching for a movie title string.
    Converts the title to a movie ID first, then calls get_recommendations().
    Returns empty list if the title doesn't match any movie in the dataset.
    """
    from src.preprocess import get_movie_id

    # Find the movie ID for the given title (partial, case-insensitive match)
    movie_id = get_movie_id(title, movies_df)
    if movie_id is None:
        return []

    return get_recommendations(movie_id, sim_result, movies_df, top_n)


def get_recommendations_by_emotion(emotion, sim_result, movies_df, top_n=10):
    """
    Recommend movies based on a detected facial emotion.

    Maps each emotion to a set of appropriate genres, then finds
    movies in those genres with the highest average similarity scores.
    This is used by the SelfieSearch feature.

    Emotion → Genre mapping was designed to match mood:
    - Happy → Action/Adventure (energetic, exciting)
    - Sad → Comedy/Family (uplifting, feel-good)
    - Angry → Comedy/Romance (calming, lighthearted)
    - Fear → Comedy/Family (comforting)
    - Surprise → Mystery/Thriller (matching energy)
    - Disgust → Comedy/Fantasy (mood reset)
    - Neutral → Drama/Documentary (thoughtful)
    """
    emotion_genre_map = {
        "happy":     ["Action", "Adventure", "Animation"],
        "sad":       ["Comedy", "Animation", "Family"],
        "angry":     ["Comedy", "Music", "Romance"],
        "fear":      ["Comedy", "Animation", "Family"],
        "surprise":  ["Mystery", "Thriller", "Sci-Fi"],
        "disgust":   ["Comedy", "Fantasy", "Animation"],
        "neutral":   ["Drama", "Documentary", "Crime"],
    }

    # Get target genres for the detected emotion (default to Drama)
    target_genres = emotion_genre_map.get(emotion.lower(), ["Drama"])
    index = sim_result["index"]
    sim_matrix = sim_result["matrix"]

    # Find all movies that match at least one target genre
    matching = []
    for movie_id in index:
        genres = get_movie_genres(movie_id, movies_df)
        if any(g in genres for g in target_genres):
            # Use average similarity score as a quality measure
            idx = index.index(movie_id)
            avg_score = float(np.mean(sim_matrix[idx]))
            title = get_movie_title(movie_id, movies_df)
            matching.append({
                "movie_id": int(movie_id),
                "title": title,
                "score": round(avg_score, 4),
                "genres": genres
            })

    # Sort by average similarity score and return top N
    matching = sorted(matching, key=lambda x: x["score"], reverse=True)
    return matching[:top_n]