# preprocess.py
# Responsible for loading the MovieLens dataset and building the user-item
# rating matrix that the recommendation engine uses to compute similarities.
# Supports both the small dataset (100K ratings) and the full 1M dataset.
# Results are cached to disk so we don't rebuild the matrix every time the app starts.

import pandas as pd
import numpy as np
import os
import pickle

# File paths for the small MovieLens dataset
RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/movies.csv"
MATRIX_CACHE = "data/user_item_matrix.pkl"
MOVIES_CACHE = "data/movies_df.pkl"

# File paths for the larger 1M MovieLens dataset
# This dataset uses '::' as separator instead of commas
RATINGS_1M = "data/ml-1m/ratings.dat"
MOVIES_1M = "data/ml-1m/movies.dat"


def load_data(size="small"):
    """
    Load ratings and movies data from disk.
    Supports two dataset sizes:
    - 'small': MovieLens 100K (default, used by the web app)
    - '1m': MovieLens 1M (used for performance evaluation)

    The 1M dataset uses '::' as a separator and latin-1 encoding
    because it contains special characters in movie titles.
    """
    if size == "1m" and os.path.exists(RATINGS_1M):
        # 1M dataset has no header row, so we manually name the columns
        ratings = pd.read_csv(RATINGS_1M, sep="::", engine="python",
                              names=["userId", "movieId", "rating", "timestamp"])
        movies = pd.read_csv(MOVIES_1M, sep="::", engine="python",
                             names=["movieId", "title", "genres"],
                             encoding="latin-1")
    else:
        # Small dataset is standard CSV with headers
        ratings = pd.read_csv(RATINGS_PATH)
        movies = pd.read_csv(MOVIES_PATH)
    return ratings, movies


def build_matrix(min_movie_ratings=50, min_user_ratings=50, size="small"):
    """
    Build and return the user-item rating matrix used for collaborative filtering.

    The matrix has movies as rows and users as columns.
    Each cell contains the rating a user gave a movie (or 0 if unrated).

    Steps:
    1. Filter out movies with fewer than min_movie_ratings ratings (sparse movies)
    2. Filter out users with fewer than min_user_ratings ratings (sparse users)
    3. Pivot the data into a movie x user matrix
    4. Mean-center each user's ratings to normalize for rating bias
       (some users always rate high, some always rate low)
    5. Fill missing values with 0

    Results are cached as .pkl files so subsequent startups are instant.
    """
    # Check if cached matrix already exists to avoid rebuilding
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

    # Remove movies and users with too few ratings
    # This reduces noise and keeps the matrix from being too sparse
    movie_counts = ratings["movieId"].value_counts()
    user_counts = ratings["userId"].value_counts()

    ratings = ratings[ratings["movieId"].isin(
        movie_counts[movie_counts >= min_movie_ratings].index)]
    ratings = ratings[ratings["userId"].isin(
        user_counts[user_counts >= min_user_ratings].index)]

    # Pivot to create the user-item matrix
    # Rows = movies, Columns = users, Values = ratings
    matrix = ratings.pivot_table(index="movieId", columns="userId", values="rating")

    # Mean-center ratings per user to remove rating scale bias
    # e.g. a user who rates everything 4-5 stars vs one who uses 1-3 stars
    matrix = matrix.subtract(matrix.mean(axis=0), axis=1)

    # Fill NaN (unrated) with 0 after mean-centering
    matrix = matrix.fillna(0)

    # Save to cache for faster future loads
    with open(cache_key, "wb") as f:
        pickle.dump(matrix, f)
    with open(movies_cache_key, "wb") as f:
        pickle.dump(movies, f)

    print(f"✅ Matrix built: {matrix.shape[0]} movies x {matrix.shape[1]} users")
    return matrix, movies


def get_movie_title(movie_id, movies_df):
    """
    Look up a movie title by its ID.
    Returns 'Unknown' if the movie ID is not found in the dataset.
    """
    row = movies_df[movies_df["movieId"] == movie_id]
    if len(row) == 0:
        return "Unknown"
    return row.iloc[0]["title"]


def get_movie_id(title, movies_df):
    """
    Search for a movie ID by title string (case-insensitive, partial match).
    Returns None if no match is found.
    Used when a user types a movie name into the search box.
    """
    row = movies_df[movies_df["title"].str.contains(title, case=False, na=False)]
    if len(row) == 0:
        return None
    return row.iloc[0]["movieId"]


def get_movie_genres(movie_id, movies_df):
    """
    Return a list of genres for a given movie ID.
    MovieLens stores genres as pipe-separated strings like 'Action|Comedy|Drama'.
    We split them into a proper list for easier filtering.
    Returns an empty list if the movie is not found.
    """
    row = movies_df[movies_df["movieId"] == movie_id]
    if len(row) == 0:
        return []
    genres = row.iloc[0]["genres"]
    return genres.split("|") if isinstance(genres, str) else []