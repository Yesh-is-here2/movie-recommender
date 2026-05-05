# routes.py
# Defines all HTTP routes (endpoints) for the CineAI web app using FastAPI.
# Each route handles a specific URL and HTTP method (GET or POST).
# Routes are protected by JWT token authentication stored in browser cookies.
# Role-based access control ensures users, admins, and owners see different content.

from fastapi import APIRouter, Request, Response, Form, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
from src.auth import verify_password, create_access_token, get_current_user, hash_password
from src.database import get_connection, log_activity
from src.recommender import get_recommendations_by_title, get_recommendations_by_emotion
from src.tmdb import enrich_movie, get_actor_details
from src.emotion import analyze_emotion, emotion_to_message

# Set up Jinja2 template engine pointing to the templates/ folder
templates = Jinja2Templates(directory="templates")

# APIRouter groups all our routes — included in main.py
router = APIRouter()


def get_user_from_cookie(token: Optional[str]) -> Optional[dict]:
    """
    Helper function to extract and validate the current user from a browser cookie.
    Called at the start of every protected route to identify who is making the request.
    Returns None if no token is present or if the token is invalid/expired.
    """
    if not token:
        return None
    return get_current_user(token)


# ─── AUTH ROUTES ─────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def home(request: Request, token: Optional[str] = Cookie(None)):
    """
    Home page route. If the user is already logged in (has a valid cookie),
    redirect them straight to the dashboard. Otherwise show the login page.
    """
    user = get_user_from_cookie(token)
    if user:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse(request, "login.html")


@router.post("/login")
async def login(response: Response, username: str = Form(...), password: str = Form(...)):
    """
    Handle login form submission.
    Looks up the user in the database, verifies their password using bcrypt,
    creates a JWT token, stores it in an HTTP-only cookie, and returns the user's role
    so the frontend can redirect to the correct dashboard.
    HTTP-only cookies prevent JavaScript from reading the token (XSS protection).
    """
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()

    # Return 401 if user not found or password doesn't match
    if not user or not verify_password(password, user["hashed_password"]):
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)

    # Create signed JWT token containing username and role
    token = create_access_token({"sub": user["username"], "role": user["role"]})

    # Log this login action for admin activity tracking
    log_activity(user["id"], "login", f"User {username} logged in")

    # Set token as HTTP-only cookie so it's sent automatically on future requests
    resp = JSONResponse({"role": user["role"], "redirect": "/dashboard"})
    resp.set_cookie("token", token, httponly=True)
    return resp


@router.get("/logout")
async def logout(response: Response):
    """
    Log the user out by deleting their auth cookie and redirecting to login page.
    """
    resp = RedirectResponse(url="/")
    resp.delete_cookie("token")
    return resp


@router.post("/register")
async def register(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    """
    Handle new user registration.
    Hashes the password before storing — never stores plain text passwords.
    Returns 400 if the username or email is already taken.
    New users are always assigned the 'user' role (not admin or owner).
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, email, hashed_password, role) VALUES (?, ?, ?, ?)",
            (username, email, hash_password(password), "user")
        )
        conn.commit()
        return JSONResponse({"message": "Account created! Please login."})
    except Exception:
        # SQLite raises an exception if username or email already exists (UNIQUE constraint)
        return JSONResponse({"error": "Username or email already exists"}, status_code=400)
    finally:
        conn.close()


# ─── DASHBOARD ROUTES ────────────────────────────────────────────────────────

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, token: Optional[str] = Cookie(None)):
    """
    Main dashboard route — serves different HTML templates based on user role.
    - 'user' → user_dashboard.html (search + selfie search)
    - 'admin' → admin_dashboard.html (user list + activity logs)
    - 'owner' → owner_dashboard.html (full system stats)
    Unauthenticated users are redirected back to login.
    """
    user = get_user_from_cookie(token)
    if not user:
        return RedirectResponse(url="/")

    role = user["role"]
    if role == "owner":
        return templates.TemplateResponse(request, "owner_dashboard.html", {"user": dict(user)})
    elif role == "admin":
        return templates.TemplateResponse(request, "admin_dashboard.html", {"user": dict(user)})
    else:
        return templates.TemplateResponse(request, "user_dashboard.html", {"user": dict(user)})


# ─── RECOMMENDATION ROUTES ───────────────────────────────────────────────────

@router.post("/recommend", response_class=JSONResponse)
async def recommend(request: Request, token: Optional[str] = Cookie(None)):
    """
    Movie recommendation endpoint — called when a user types a movie and clicks Search.
    Uses the precomputed parallel similarity matrix to find similar movies,
    then enriches results with TMDB data (posters, cast, similar movies).
    Returns up to 6 enriched recommendations as JSON.
    """
    user = get_user_from_cookie(token)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    body = await request.json()
    title = body.get("title", "")
    top_n = body.get("top_n", 10)

    # Access the globally loaded similarity matrix from main.py
    import main
    if main.sim_result is None:
        return JSONResponse({"error": "Model still loading, please wait"}, status_code=503)

    # Get top-N similar movies using collaborative filtering
    recs = get_recommendations_by_title(title, main.sim_result, main.movies_df, top_n)

    # Enrich each recommendation with TMDB poster, cast, and similar movies
    enriched = []
    for r in recs[:6]:
        tmdb = enrich_movie(r["title"])
        enriched.append({**r, **tmdb})  # Merge our rec data with TMDB data

    # Log this search action for admin dashboard tracking
    log_activity(user["id"], "recommendation", f"Searched: {title}")
    return JSONResponse({"recommendations": enriched})


@router.post("/selfie-search", response_class=JSONResponse)
async def selfie_search(request: Request, token: Optional[str] = Cookie(None)):
    """
    SelfieSearch endpoint — called when user captures a webcam photo.
    Analyzes the facial emotion using DeepFace, maps it to movie genres,
    and returns mood-appropriate movie recommendations enriched with TMDB data.
    """
    user = get_user_from_cookie(token)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    body = await request.json()
    image_base64 = body.get("image", "")  # Base64 encoded webcam image from browser

    # Detect emotion from the captured image
    emotion = analyze_emotion(image_base64)
    message = emotion_to_message(emotion)  # Human-readable message for the UI

    import main
    if main.sim_result is None:
        return JSONResponse({"error": "Model still loading, please wait"}, status_code=503)

    # Get movies matching the detected emotion's genre mapping
    recs = get_recommendations_by_emotion(emotion, main.sim_result, main.movies_df, top_n=8)

    # Enrich with TMDB data
    enriched = []
    for r in recs[:6]:
        tmdb = enrich_movie(r["title"])
        enriched.append({**r, **tmdb})

    # Log selfie search with detected emotion for admin tracking
    log_activity(user["id"], "selfie_search", f"Emotion: {emotion}")
    return JSONResponse({"emotion": emotion, "message": message, "recommendations": enriched})


# ─── ACTOR ROUTES ────────────────────────────────────────────────────────────

@router.get("/actor/{actor_id}", response_class=JSONResponse)
async def actor_details(actor_id: int, token: Optional[str] = Cookie(None)):
    """
    Fetch actor details from TMDB by actor ID.
    Called when a user clicks on an actor's photo in the Cast modal.
    Returns bio, photo, and upcoming movies for the selected actor.
    """
    user = get_user_from_cookie(token)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    data = get_actor_details(actor_id)
    return JSONResponse(data)


# ─── ADMIN ROUTES ────────────────────────────────────────────────────────────

@router.get("/admin/users", response_class=JSONResponse)
async def admin_users(token: Optional[str] = Cookie(None)):
    """
    Admin-only endpoint that returns all users and recent activity logs.
    Used to populate the admin dashboard with user management data.
    Accessible by both admin and owner roles.
    Returns 403 Forbidden if a regular user tries to access this.
    """
    user = get_user_from_cookie(token)
    if not user or user["role"] not in ("admin", "owner"):
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    conn = get_connection()
    # Get all users (excluding sensitive hashed_password field)
    users = conn.execute(
        "SELECT id, username, email, role, created_at, is_active FROM users"
    ).fetchall()
    # Get recent 50 activity log entries joined with username
    logs = conn.execute(
        "SELECT a.*, u.username FROM activity_logs a JOIN users u ON a.user_id = u.id ORDER BY a.timestamp DESC LIMIT 50"
    ).fetchall()
    conn.close()

    return JSONResponse({
        "users": [dict(u) for u in users],
        "logs": [dict(l) for l in logs]
    })


# ─── OWNER ROUTES ────────────────────────────────────────────────────────────

@router.get("/owner/stats", response_class=JSONResponse)
async def owner_stats(token: Optional[str] = Cookie(None)):
    """
    Owner-only endpoint that returns full system statistics.
    Shows total user count, admin count, search counts, and last 100 activity entries.
    Only accessible by the owner role — admins cannot access this.
    Used to populate the owner dashboard with platform-wide metrics.
    """
    user = get_user_from_cookie(token)
    if not user or user["role"] != "owner":
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    conn = get_connection()
    # Count users, admins, and different types of searches
    total_users = conn.execute("SELECT COUNT(*) FROM users WHERE role='user'").fetchone()[0]
    total_admins = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
    total_searches = conn.execute(
        "SELECT COUNT(*) FROM activity_logs WHERE action='recommendation'"
    ).fetchone()[0]
    total_selfie = conn.execute(
        "SELECT COUNT(*) FROM activity_logs WHERE action='selfie_search'"
    ).fetchone()[0]
    # Get last 100 activity entries for the full activity feed
    recent = conn.execute(
        "SELECT a.*, u.username FROM activity_logs a JOIN users u ON a.user_id = u.id ORDER BY a.timestamp DESC LIMIT 100"
    ).fetchall()
    conn.close()

    return JSONResponse({
        "total_users": total_users,
        "total_admins": total_admins,
        "total_searches": total_searches,
        "total_selfie_searches": total_selfie,
        "recent_activity": [dict(r) for r in recent]
    })