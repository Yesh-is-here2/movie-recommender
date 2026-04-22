import sqlite3
import os

DB_PATH = "movie_recommender.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS recommendations_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_ids TEXT NOT NULL,
            method TEXT DEFAULT 'collaborative',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # Create default accounts if they don't exist
    from src.auth import hash_password

    defaults = [
        ("admin", "admin@movies.com", hash_password("admin123"), "admin"),
        ("owner", "owner@movies.com", hash_password("owner123"), "owner"),
    ]

    for username, email, pwd, role in defaults:
        cursor.execute("""
            INSERT OR IGNORE INTO users (username, email, hashed_password, role)
            VALUES (?, ?, ?, ?)
        """, (username, email, pwd, role))

    conn.commit()
    conn.close()
    print("✅ Database initialized")

def log_activity(user_id: int, action: str, details: str = ""):
    conn = get_connection()
    conn.execute(
        "INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
        (user_id, action, details)
    )
    conn.commit()
    conn.close()