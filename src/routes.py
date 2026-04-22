from fastapi import APIRouter, Request, Response, Form, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
from src.auth import verify_password, create_access_token, get_current_user, hash_password
from src.database import get_connection, log_activity
from src.recommender import get_recommendations_by_title, get_recommendations_by_emotion
from src.tmdb import enrich_movie, get_actor_details
from src.emotion import analyze_emotion, emotion_to_message

templates = Jinja2Templates(directory="templates")
router = APIRouter()

def get_user_from_cookie(token: Optional[str]) -> Optional[dict]:
    if not token:
        return None
    return get_current_user(token)

@router.get("/", response_class=HTMLResponse)
async def home(request: Request, token: Optional[str] = Cookie(None)):
    user = get_user_from_cookie(token)
    if user:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse(request, "login.html")

@router.post("/login")
async def login(response: Response, username: str = Form(...), password: str = Form(...)):
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if not user or not verify_password(password, user["hashed_password"]):
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    log_activity(user["id"], "login", f"User {username} logged in")
    resp = JSONResponse({"role": user["role"], "redirect": "/dashboard"})
    resp.set_cookie("token", token, httponly=True)
    return resp

@router.get("/logout")
async def logout(response: Response):
    resp = RedirectResponse(url="/")
    resp.delete_cookie("token")
    return resp

@router.post("/register")
async def register(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, email, hashed_password, role) VALUES (?, ?, ?, ?)",
            (username, email, hash_password(password), "user")
        )
        conn.commit()
        return JSONResponse({"message": "Account created! Please login."})
    except Exception:
        return JSONResponse({"error": "Username or email already exists"}, status_code=400)
    finally:
        conn.close()

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, token: Optional[str] = Cookie(None)):
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

@router.post("/recommend", response_class=JSONResponse)
async def recommend(request: Request, token: Optional[str] = Cookie(None)):
    user = get_user_from_cookie(token)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    body = await request.json()
    title = body.get("title", "")
    top_n = body.get("top_n", 10)
    import main
    if main.sim_result is None:
        return JSONResponse({"error": "Model still loading, please wait"}, status_code=503)
    recs = get_recommendations_by_title(title, main.sim_result, main.movies_df, top_n)
    enriched = []
    for r in recs[:6]:
        tmdb = enrich_movie(r["title"])
        enriched.append({**r, **tmdb})
    log_activity(user["id"], "recommendation", f"Searched: {title}")
    return JSONResponse({"recommendations": enriched})

@router.post("/selfie-search", response_class=JSONResponse)
async def selfie_search(request: Request, token: Optional[str] = Cookie(None)):
    user = get_user_from_cookie(token)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    body = await request.json()
    image_base64 = body.get("image", "")
    emotion = analyze_emotion(image_base64)
    message = emotion_to_message(emotion)
    import main
    if main.sim_result is None:
        return JSONResponse({"error": "Model still loading, please wait"}, status_code=503)
    recs = get_recommendations_by_emotion(emotion, main.sim_result, main.movies_df, top_n=8)
    enriched = []
    for r in recs[:6]:
        tmdb = enrich_movie(r["title"])
        enriched.append({**r, **tmdb})
    log_activity(user["id"], "selfie_search", f"Emotion: {emotion}")
    return JSONResponse({"emotion": emotion, "message": message, "recommendations": enriched})

@router.get("/actor/{actor_id}", response_class=JSONResponse)
async def actor_details(actor_id: int, token: Optional[str] = Cookie(None)):
    user = get_user_from_cookie(token)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    data = get_actor_details(actor_id)
    return JSONResponse(data)

@router.get("/admin/users", response_class=JSONResponse)
async def admin_users(token: Optional[str] = Cookie(None)):
    user = get_user_from_cookie(token)
    if not user or user["role"] not in ("admin", "owner"):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    conn = get_connection()
    users = conn.execute("SELECT id, username, email, role, created_at, is_active FROM users").fetchall()
    logs = conn.execute(
        "SELECT a.*, u.username FROM activity_logs a JOIN users u ON a.user_id = u.id ORDER BY a.timestamp DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return JSONResponse({"users": [dict(u) for u in users], "logs": [dict(l) for l in logs]})

@router.get("/owner/stats", response_class=JSONResponse)
async def owner_stats(token: Optional[str] = Cookie(None)):
    user = get_user_from_cookie(token)
    if not user or user["role"] != "owner":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    conn = get_connection()
    total_users = conn.execute("SELECT COUNT(*) FROM users WHERE role='user'").fetchone()[0]
    total_admins = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
    total_searches = conn.execute("SELECT COUNT(*) FROM activity_logs WHERE action='recommendation'").fetchone()[0]
    total_selfie = conn.execute("SELECT COUNT(*) FROM activity_logs WHERE action='selfie_search'").fetchone()[0]
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