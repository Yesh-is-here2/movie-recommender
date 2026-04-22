import pandas as pd
import numpy as np
import os
import pickle

RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/movies.csv"
MATRIX_CACHE = "data/user_item_matrix.pkl"
MOVIES_CACHE = "data/movies_df.pkl"

def load_data():
    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    return ratings, movies

def build_matrix(min_movie_ratings=10, min_user_ratings=10):
    if os.path.exists(MATRIX_CACHE) and os.path.exists(MOVIES_CACHE):
        print("✅ Loading cached matrix...")
        with open(MATRIX_CACHE, "rb") as f:
            matrix = pickle.load(f)
        with open(MOVIES_CACHE, "rb") as f:
            movies = pickle.load(f)
        return matrix, movies

    print("⏳ Building user-item matrix...")
    ratings, movies = load_data()

    # Filter sparse movies and users
    movie_counts = ratings["movieId"].value_counts()
    user_counts = ratings["userId"].value_counts()

    ratings = ratings[ratings["movieId"].isin(movie_counts[movie_counts >= min_movie_ratings].index)]
    ratings = ratings[ratings["userId"].isin(user_counts[user_counts >= min_user_ratings].index)]

    # Pivot to user-item matrix
    matrix = ratings.pivot_table(index="movieId", columns="userId", values="rating")

    # Mean center each user's ratings
    matrix = matrix.subtract(matrix.mean(axis=0), axis=1)
    matrix = matrix.fillna(0)

    # Cache it
    with open(MATRIX_CACHE, "wb") as f:
        pickle.dump(matrix, f)
    with open(MOVIES_CACHE, "wb") as f:
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
    return row.iloc[0]["genres"].split("|")