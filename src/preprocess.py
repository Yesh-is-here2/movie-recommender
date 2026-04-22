import pandas as pd
import numpy as np
import os
import pickle

RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/movies.csv"
MATRIX_CACHE = "data/user_item_matrix.pkl"
MOVIES_CACHE = "data/movies_df.pkl"

# 1M dataset paths
RATINGS_1M = "data/ml-1m/ratings.dat"
MOVIES_1M = "data/ml-1m/movies.dat"

def load_data(size="small"):
    if size == "1m" and os.path.exists(RATINGS_1M):
        ratings = pd.read_csv(RATINGS_1M, sep="::", engine="python",
                              names=["userId", "movieId", "rating", "timestamp"])
        movies = pd.read_csv(MOVIES_1M, sep="::", engine="python",
                             names=["movieId", "title", "genres"],
                             encoding="latin-1")
    else:
        ratings = pd.read_csv(RATINGS_PATH)
        movies = pd.read_csv(MOVIES_PATH)
    return ratings, movies

def build_matrix(min_movie_ratings=50, min_user_ratings=50, size="small"):
    cache_key = f"data/user_item_matrix_{size}.pkl"
    movies_cache_key = f"data/movies_df_{size}.pkl"

    if os.path.exists(cache_key) and os.path.exists(movies_cache_key):
        print(f"✅ Loading cached matrix ({size})...")
        with open(cache_key, "rb") as f:
            matrix = pickle.load(f)
        with open(movies_cache_key, "rb") as f:
            movies = pickle.load(f)
        return matrix, movies

    print(f"⏳ Building user-item matrix ({size})...")
    ratings, movies = load_data(size)

    movie_counts = ratings["movieId"].value_counts()
    user_counts = ratings["userId"].value_counts()

    ratings = ratings[ratings["movieId"].isin(
        movie_counts[movie_counts >= min_movie_ratings].index)]
    ratings = ratings[ratings["userId"].isin(
        user_counts[user_counts >= min_user_ratings].index)]

    matrix = ratings.pivot_table(index="movieId", columns="userId", values="rating")
    matrix = matrix.subtract(matrix.mean(axis=0), axis=1)
    matrix = matrix.fillna(0)

    with open(cache_key, "wb") as f:
        pickle.dump(matrix, f)
    with open(movies_cache_key, "wb") as f:
        pickle.dump(movies, f)

    print(f"✅ Matrix built: {matrix.shape[0]} movies x {matrix.shape[1]} users")
    return matrix, movies

def get_movie_title(movie_id, movies_df):
    row = movies_df[movies_df["movieId"] == movie_id]
    if len(row) == 0:
        return "Unknown"
    return row.iloc[0]["title"]

def get_movie_id(title, movies_df):
    row = movies_df[movies_df["title"].str.contains(title, case=False, na=False)]
    if len(row) == 0:
        return None
    return row.iloc[0]["movieId"]

def get_movie_genres(movie_id, movies_df):
    row = movies_df[movies_df["movieId"] == movie_id]
    if len(row) == 0:
        return []
    genres = row.iloc[0]["genres"]
    return genres.split("|") if isinstance(genres, str) else []