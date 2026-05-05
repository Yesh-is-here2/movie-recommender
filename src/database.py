# database.py
# Handles all database operations for the CineAI app.
# We use SQLite as our database because it is lightweight,
# requires no separate server, and is perfect for a local portfolio project.
# The database stores users, activity logs, and recommendation history.

import sqlite3
import os

# Path to the SQLite database file
# SQLite stores everything in a single .db file on disk
DB_PATH = "movie_recommender.db"


def get_connection():
    """
    Create and return a new connection to the SQLite database.
    We set row_factory = sqlite3.Row so that query results
    behave like dictionaries — we can access columns by name
    instead of by index (e.g., user["username"] instead of user[0]).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Allows column access by name
    return conn


def init_db():
    """
    Initialize the database by creating all required tables if they don't exist.
    Also creates default admin and owner accounts on first run.
    This function is called once when the app starts up in main.py.

    Tables created:
    - users: stores all registered users with their role and hashed password
    - activity_logs: tracks every user action (login, search, selfie search)
    - recommendations_log: stores what movies were recommended to each user
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Create all tables in a single script
    # IF NOT EXISTS means this is safe to run multiple times without errors
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

    # Import here to avoid circular imports between database and auth modules
    from src.auth import hash_password

    # Default accounts created on first startup
    # INSERT OR IGNORE means if the account already exists, skip it silently
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
    """
    Record a user action in the activity_logs table.
    This is called every time a user logs in, searches for a movie,
    or uses the SelfieSearch feature.

    The admin and owner dashboards read from this table to show
    what users are doing in real time.

    Parameters:
        user_id: the ID of the user performing the action
        action: short label like 'login', 'recommendation', 'selfie_search'
        details: extra context like the movie title searched or emotion detected
    """
    conn = get_connection()
    conn.execute(
        "INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
        (user_id, action, details)
    )
    conn.commit()
    conn.close()