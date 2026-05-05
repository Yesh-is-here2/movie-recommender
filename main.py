# main.py
# Entry point for the CineAI web application.
# This file creates the FastAPI app, loads the recommendation model on startup,
# and starts the web server. Everything begins here.
#
# Startup sequence:
# 1. FastAPI app is created
# 2. Static files and routes are registered
# 3. On startup: database is initialized, rating matrix is built,
#    and the parallel similarity matrix is computed (or loaded from cache)
# 4. Uvicorn serves the app on http://localhost:8000

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Create the FastAPI application instance
app = FastAPI(title="Movie Recommender")

# Serve static files (CSS, JS, images) from the static/ folder
# Accessible at /static/... in the browser
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for the similarity matrix and movies dataframe
# These are loaded once at startup and shared across all requests
# Routes access them via 'import main' then 'main.sim_result'
sim_result = None
movies_df = None


def startup():
    """
    Initialization function that runs when the server starts.
    Loads all heavy data into memory so requests can be served instantly.

    Steps:
    1. Initialize the SQLite database and create default admin/owner accounts
    2. Build the user-item rating matrix from the MovieLens dataset
    3. Compute (or load cached) the parallel cosine similarity matrix

    This runs in a separate thread via run_in_executor to avoid blocking
    the FastAPI event loop during startup.
    """
    # Use global so we can assign to the module-level variables
    global sim_result, movies_df

    from src.database import init_db
    from src.preprocess import build_matrix
    from src.similarity_parallel import compute_parallel

    print("🚀 Starting Movie Recommender...")

    # Set up the database tables and default accounts
    init_db()

    # Load and build the user-item matrix from MovieLens ratings
    matrix, movies_df = build_matrix()

    # Compute item-item cosine similarity using 4 parallel workers
    # On first run this takes ~2 minutes — subsequent runs load from cache instantly
    sim_result = compute_parallel(matrix, n_workers=4)

    print("✅ App ready!")


# Register all API routes from routes.py
# This includes login, register, dashboard, recommend, selfie-search, etc.
from src.routes import router
app.include_router(router)


@app.on_event("startup")
async def on_startup():
    """
    FastAPI startup event handler.
    Runs the startup() function in a thread pool executor so it doesn't
    block the async event loop while loading large data files.
    This is necessary because startup() does CPU-heavy work (matrix computation).
    """
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, startup)


if __name__ == '__main__':
    # freeze_support() is required on Windows when using multiprocessing
    # Without it, spawning new processes causes a RuntimeError on Windows
    # because Windows uses 'spawn' instead of 'fork' to create child processes
    from multiprocessing import freeze_support
    freeze_support()

    # Start the Uvicorn ASGI server
    # reload=False because we handle startup manually above
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)