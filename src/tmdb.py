import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

def clean_title(title: str) -> str:
    """Remove year like (1995) from MovieLens titles."""
    return re.sub(r'\(\d{4}\)', '', title).strip()

def search_movie(title: str):
    """Search TMDB for a movie by title."""
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    try:
        res = requests.get(url, params=params, timeout=8)
        data = res.json()
        results = data.get("results", [])
        if not results:
            return None
        # Return most popular result
        return sorted(results, key=lambda x: x.get("popularity", 0), reverse=True)[0]
    except Exception:
        return None

def get_movie_details(tmdb_id: int):
    """Get full movie details from TMDB."""
    url = f"{BASE_URL}/movie/{tmdb_id}"
    params = {"api_key": TMDB_API_KEY, "append_to_response": "credits,similar"}
    try:
        res = requests.get(url, params=params, timeout=8)
        return res.json()
    except Exception:
        return {}

def get_movie_cast(tmdb_id: int):
    """Get cast list for a movie."""
    details = get_movie_details(tmdb_id)
    credits = details.get("credits", {})
    cast = credits.get("cast", [])[:8]
    return [{
        "id": c["id"],
        "name": c["name"],
        "character": c.get("character", ""),
        "photo": IMAGE_BASE + c["profile_path"] if c.get("profile_path") else None
    } for c in cast]

def get_actor_details(actor_id: int):
    """Get actor bio and upcoming movies."""
    url = f"{BASE_URL}/person/{actor_id}"
    params = {"api_key": TMDB_API_KEY, "append_to_response": "movie_credits"}
    try:
        res = requests.get(url, params=params, timeout=8)
        data = res.json()
        upcoming = sorted(
            [m for m in data.get("movie_credits", {}).get("cast", [])
             if m.get("release_date", "") > "2024-01-01"],
            key=lambda x: x.get("release_date", ""),
            reverse=True
        )[:5]
        return {
            "name": data.get("name"),
            "bio": data.get("biography", "")[:500],
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
    """Get similar movies from TMDB."""
    details = get_movie_details(tmdb_id)
    similar = details.get("similar", {}).get("results", [])[:6]
    return [{
        "title": m["title"],
        "poster": IMAGE_BASE + m["poster_path"] if m.get("poster_path") else None,
        "rating": m.get("vote_average", 0),
        "release_date": m.get("release_date", "")
    } for m in similar]

def enrich_movie(title: str):
    """Full enrichment — search + details + cast + similar."""
    # Clean title — remove year like (1995) before searching TMDB
    cleaned = clean_title(title)

    # Try cleaned title first, fall back to original
    movie = search_movie(cleaned)
    if not movie:
        movie = search_movie(title)
    if not movie:
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