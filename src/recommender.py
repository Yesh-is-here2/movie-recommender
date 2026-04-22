import numpy as np
from src.preprocess import get_movie_title, get_movie_genres

def get_recommendations(movie_id, sim_result, movies_df, top_n=10):
    """Get top-N similar movies for a given movie ID."""
    sim_matrix = sim_result["matrix"]
    index = sim_result["index"]

    if movie_id not in index:
        return []

    idx = index.index(movie_id)
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if index[s[0]] != movie_id]

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
    """Get recommendations by movie title string."""
    from src.preprocess import get_movie_id
    movie_id = get_movie_id(title, movies_df)
    if movie_id is None:
        return []
    return get_recommendations(movie_id, sim_result, movies_df, top_n)


def get_recommendations_by_emotion(emotion, sim_result, movies_df, top_n=10):
    """Map emotion to genre and return top matching movies."""
    emotion_genre_map = {
        "happy":     ["Action", "Adventure", "Animation"],
        "sad":       ["Comedy", "Animation", "Family"],
        "angry":     ["Comedy", "Music", "Romance"],
        "fear":      ["Comedy", "Animation", "Family"],
        "surprise":  ["Mystery", "Thriller", "Sci-Fi"],
        "disgust":   ["Comedy", "Fantasy", "Animation"],
        "neutral":   ["Drama", "Documentary", "Crime"],
    }

    target_genres = emotion_genre_map.get(emotion.lower(), ["Drama"])
    index = sim_result["index"]
    sim_matrix = sim_result["matrix"]

    # Find movies matching target genres
    matching = []
    for movie_id in index:
        genres = get_movie_genres(movie_id, movies_df)
        if any(g in genres for g in target_genres):
            idx = index.index(movie_id)
            avg_score = float(np.mean(sim_matrix[idx]))
            title = get_movie_title(movie_id, movies_df)
            matching.append({
                "movie_id": int(movie_id),
                "title": title,
                "score": round(avg_score, 4),
                "genres": genres
            })

    matching = sorted(matching, key=lambda x: x["score"], reverse=True)
    return matching[:top_n]