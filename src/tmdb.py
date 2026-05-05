# tmdb.py
# Handles all communication with The Movie Database (TMDB) API.
# TMDB is a free movie database that provides posters, cast details,
# actor bios, upcoming movies, and similar movie suggestions.
# We use it to enrich our collaborative filtering recommendations
# with real movie data that makes the UI look professional.

import requests
import os
import re
from dotenv import load_dotenv

# Load TMDB API key from .env file
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"  # Base URL for poster/photo images


def clean_title(title: str) -> str:
    """
    Remove the year suffix that MovieLens adds to titles.
    MovieLens stores titles like 'Toy Story (1995)' but TMDB expects 'Toy Story'.
    Without this cleanup, many TMDB searches return no results.
    """
    return re.sub(r'\(\d{4}\)', '', title).strip()


def search_movie(title: str):
    """
    Search TMDB for a movie by title and return the most popular match.
    Returns None if no results are found or if the API call fails.
    We sort by popularity to avoid getting obscure films with similar names.
    """
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    try:
        res = requests.get(url, params=params, timeout=8)
        data = res.json()
        results = data.get("results", [])
        if not results:
            return None
        # Pick the most popular result to avoid wrong movie matches
        return sorted(results, key=lambda x: x.get("popularity", 0), reverse=True)[0]
    except Exception:
        return None


def get_movie_details(tmdb_id: int):
    """
    Fetch full movie details from TMDB including cast and similar movies.
    We use append_to_response to get credits and similar in a single API call
    instead of making three separate requests.
    """
    url = f"{BASE_URL}/movie/{tmdb_id}"
    params = {"api_key": TMDB_API_KEY, "append_to_response": "credits,similar"}
    try:
        res = requests.get(url, params=params, timeout=8)
        return res.json()
    except Exception:
        return {}


def get_movie_cast(tmdb_id: int):
    """
    Return the top 8 cast members for a movie with their photos.
    Used to populate the Cast modal when a user clicks the Cast button.
    We limit to 8 to keep the UI clean and avoid too many API lookups.
    """
    details = get_movie_details(tmdb_id)
    credits = details.get("credits", {})
    cast = credits.get("cast", [])[:8]  # Top 8 billed actors
    return [{
        "id": c["id"],
        "name": c["name"],
        "character": c.get("character", ""),
        # Build full image URL or None if no photo available
        "photo": IMAGE_BASE + c["profile_path"] if c.get("profile_path") else None
    } for c in cast]


def get_actor_details(actor_id: int):
    """
    Fetch an actor's biography and upcoming movies from TMDB.
    Called when a user clicks on an actor's photo in the Cast modal.
    We filter for movies with release dates after 2024 to show upcoming work.
    The bio is truncated to 500 characters to fit the modal UI.
    """
    url = f"{BASE_URL}/person/{actor_id}"
    params = {"api_key": TMDB_API_KEY, "append_to_response": "movie_credits"}
    try:
        res = requests.get(url, params=params, timeout=8)
        data = res.json()

        # Filter for upcoming/recent movies only
        upcoming = sorted(
            [m for m in data.get("movie_credits", {}).get("cast", [])
             if m.get("release_date", "") > "2024-01-01"],
            key=lambda x: x.get("release_date", ""),
            reverse=True
        )[:5]  # Show max 5 upcoming movies

        return {
            "name": data.get("name"),
            "bio": data.get("biography", "")[:500],  # Truncate long bios
            "photo": IMAGE_BASE + data["profile_path"] if data.get("profile_path") else None,
            "upcoming": [{
                "title": m["title"],
                "release_date": m.get("release_date", "TBA"),
                "poster": IMAGE_BASE + m["poster_path"] if m.get("poster_path") else None
            } for m in upcoming]
        }
    except Exception:
        return {}


def get_similar_movies(tmdb_id: int):
    """
    Get up to 6 movies that TMDB considers similar to the given movie.
    Used to populate the Similar Movies modal on the user dashboard.
    """
    details = get_movie_details(tmdb_id)
    similar = details.get("similar", {}).get("results", [])[:6]
    return [{
        "title": m["title"],
        "poster": IMAGE_BASE + m["poster_path"] if m.get("poster_path") else None,
        "rating": m.get("vote_average", 0),
        "release_date": m.get("release_date", "")
    } for m in similar]


def enrich_movie(title: str):
    """
    Main enrichment function — takes a movie title from our recommendation engine
    and adds TMDB data (poster, cast, similar movies, rating, overview).

    First tries the cleaned title (year removed), then falls back to the original.
    If TMDB can't find the movie at all, returns a minimal dict with empty arrays
    so the UI still renders cleanly without crashing.
    """
    # Remove year suffix added by MovieLens before searching TMDB
    cleaned = clean_title(title)

    # Try cleaned title first, fall back to original if needed
    movie = search_movie(cleaned)
    if not movie:
        movie = search_movie(title)
    if not movie:
        # Return safe fallback so UI doesn't break for unrecognized movies
        return {"title": title, "poster": None, "cast": [], "similar": []}

    tmdb_id = movie["id"]
    return {
        "tmdb_id": tmdb_id,
        "title": movie["title"],
        "poster": IMAGE_BASE + movie["poster_path"] if movie.get("poster_path") else None,
        "overview": movie.get("overview", ""),
        "rating": movie.get("vote_average", 0),
        "release_date": movie.get("release_date", ""),
        "cast": get_movie_cast(tmdb_id),
        "similar": get_similar_movies(tmdb_id)
    }