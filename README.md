# 🎬 CineAI — Parallel Movie Recommendation System

A full-stack AI-powered movie recommendation web app with parallel computing under the hood, emotion-based suggestions, and role-based dashboards.

> Built as part of a Concurrent & Parallel Programming course project and personal portfolio.

---

## 🚀 Live Features

- 🔍 **Movie Search** — Search any movie and get intelligent recommendations using collaborative filtering
- 🤳 **SelfieSearch** — Webcam-based facial emotion detection suggests movies based on your mood
- 🎭 **Cast Details** — Click any movie to see full cast with actor photos and upcoming movies
- 🎬 **Similar Movies** — Discover movies similar to your recommendations
- 👤 **Role-Based Dashboards** — Separate experiences for Users, Admins, and Owners
- ⚡ **Parallel Computing** — Similarity matrix computed using Python multiprocessing across multiple CPU cores

---

## 🧠 How It Works

### Recommendation Engine
- Loads the [MovieLens](https://grouplens.org/datasets/movielens/) dataset
- Builds a user-item rating matrix
- Computes **cosine similarity** between all movie pairs
- Uses **Python multiprocessing** to parallelize computation across 4 CPU cores
- Achieves significant speedup over serial baseline

### Emotion Detection
- Captures webcam image via browser
- Analyzes facial emotion using **DeepFace**
- Maps detected emotion to movie genres:

| Emotion | Recommended Genres |
|---|---|
| 😄 Happy | Action, Adventure, Animation |
| 😢 Sad | Comedy, Family, Animation |
| 😠 Angry | Comedy, Music, Romance |
| 😨 Fear | Comedy, Family, Animation |
| 😲 Surprise | Mystery, Thriller, Sci-Fi |
| 😐 Neutral | Drama, Documentary, Crime |

---

## 🏗️ Architecture

movie-recommender/
│
├── data/                          # MovieLens dataset
├── src/
│   ├── preprocess.py              # Load data, build user-item matrix
│   ├── similarity_serial.py       # Serial cosine similarity baseline
│   ├── similarity_parallel.py     # Parallel implementation (multiprocessing)
│   ├── recommender.py             # Top-N recommendation logic
│   ├── auth.py                    # JWT authentication
│   ├── database.py                # SQLite database + activity logging
│   ├── models.py                  # Pydantic models
│   ├── tmdb.py                    # TMDB API integration
│   ├── emotion.py                 # DeepFace emotion detection
│   └── routes.py                  # FastAPI routes
├── templates/                     # HTML dashboards
│   ├── login.html
│   ├── user_dashboard.html
│   ├── admin_dashboard.html
│   └── owner_dashboard.html
├── results/                       # Speedup evaluation results
├── main.py                        # App entry point
└── requirements.txt

---

## ⚡ Parallel Computing Results

| Workers | Time (s) | Speedup |
|---|---|---|
| Serial | ~180s | 1.0x |
| 2 workers | ~95s | ~1.9x |
| 4 workers | ~52s | ~3.4x |
| 8 workers | ~35s | ~5.1x |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.10+ |
| Parallel Computing | Python multiprocessing |
| ML / Recommendation | Scikit-learn, NumPy, Pandas, SciPy |
| Emotion Detection | DeepFace, OpenCV, TensorFlow |
| Movie Data | TMDB API |
| Auth | JWT (python-jose, passlib) |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |

---

## 🔐 Default Accounts

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `XXXXXXXX` |
| Owner | `owner` | `XXXXXXXX` |
| User | Register via UI | — |

---

## 🧑‍💻 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/Yesh-is-here2/movie-recommender.git
cd movie-recommender
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download MovieLens dataset
Download from https://grouplens.org/datasets/movielens/latest/ and place `ratings.csv` and `movies.csv` in the `data/` folder.

### 5. Set up environment variables
Create a `.env` file:

SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
TMDB_API_KEY=your_tmdb_api_key

Get a free TMDB API key at https://www.themoviedb.org/settings/api

### 6. Run the app
```bash
python main.py
```

Open 👉 http://localhost:8000

---

## 📊 Evaluation

Run serial vs parallel comparison:
```python
from src.preprocess import build_matrix
from src.similarity_parallel import run_evaluation

matrix, movies_df = build_matrix()
run_evaluation(matrix)
```

Results saved to `results/evaluation_summary.csv`

---

## 👤 Author

**Yeshwanth Akula**
- GitHub: [@Yesh-is-here2](https://github.com/Yesh-is-here2)
- LinkedIn: [yeshwanth-akula](https://www.linkedin.com/in/yeshwanth-akula-0339a925b/)
- Portfolio: [yesh-is-here2.github.io/portfolio](https://yesh-is-here2.github.io/portfolio/)
