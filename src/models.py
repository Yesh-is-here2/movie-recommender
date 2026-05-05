# models.py
# Defines the data models (schemas) used for request validation in FastAPI.
# FastAPI uses Pydantic models to automatically validate incoming JSON data.
# If a request is missing a required field or has the wrong type,
# FastAPI rejects it with a clear error before it reaches our route logic.

from pydantic import BaseModel
from typing import Optional


class UserRegister(BaseModel):
    """
    Schema for new user registration requests.
    Sent as JSON from the Register form on the login page.
    """
    username: str   # Must be unique in the database
    email: str      # Must be unique in the database
    password: str   # Will be hashed before storing — never stored as plain text


class UserLogin(BaseModel):
    """
    Schema for login requests.
    FastAPI validates that both fields are present before calling the login route.
    """
    username: str
    password: str   # Compared against the stored bcrypt hash


class Token(BaseModel):
    """
    Schema for the JWT token response returned after successful login.
    Contains the token itself plus user info needed by the frontend
    to redirect to the correct dashboard based on role.
    """
    access_token: str   # The signed JWT token stored in browser cookie
    token_type: str     # Always 'bearer' for JWT
    role: str           # 'user', 'admin', or 'owner' — determines which dashboard to show
    username: str       # Displayed in the top navigation bar


class MovieRequest(BaseModel):
    """
    Schema for movie recommendation requests.
    Sent when a user types a movie title and clicks Search.
    top_n defaults to 10 if not specified.
    """
    movie_title: str    # Title of the movie to base recommendations on
    top_n: int = 10     # Number of recommendations to return


class EmotionRequest(BaseModel):
    """
    Schema for emotion-based recommendation requests from SelfieSearch.
    The emotion string comes from DeepFace analysis in emotion.py.
    """
    emotion: str        # Detected emotion: 'happy', 'sad', 'angry', etc.
    top_n: int = 10     # Number of mood-based recommendations to return