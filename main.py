from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(title="Movie Recommender")
app.mount("/static", StaticFiles(directory="static"), name="static")

sim_result = None
movies_df = None

def startup():
    global sim_result, movies_df
    from src.database import init_db
    from src.preprocess import build_matrix
    from src.similarity_parallel import compute_parallel

    print("🚀 Starting Movie Recommender...")
    init_db()
    matrix, movies_df = build_matrix()
    sim_result = compute_parallel(matrix, n_workers=4)
    print("✅ App ready!")

from src.routes import router
app.include_router(router)

@app.on_event("startup")
async def on_startup():
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, startup)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)