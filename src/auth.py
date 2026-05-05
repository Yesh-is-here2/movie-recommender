# auth.py
# Handles all authentication logic for the CineAI app.
# This includes password hashing, JWT token creation/decoding,
# and looking up the current logged-in user from a token.

from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables from .env file
# This keeps sensitive values like SECRET_KEY out of the source code
load_dotenv()

# Read auth configuration from environment
SECRET_KEY = os.getenv("SECRET_KEY")           # Used to sign JWT tokens
ALGORITHM = os.getenv("ALGORITHM")             # Signing algorithm (HS256)
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))  # Token lifetime

# Set up the password hashing context using bcrypt
# bcrypt__rounds=12 controls the hashing cost — higher is more secure but slower
# We explicitly set rounds to avoid version conflicts with newer bcrypt libraries
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


def hash_password(password: str) -> str:
    """
    Hash a plain-text password using bcrypt before storing in the database.
    We never store raw passwords — only the hashed version.
    """
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """
    Compare a plain-text password against a stored bcrypt hash.
    Returns True if they match, False otherwise.
    Used during login to validate user credentials.
    """
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    """
    Generate a signed JWT token containing user info (username, role).
    The token expires after ACCESS_TOKEN_EXPIRE_MINUTES minutes.
    This token is stored in a browser cookie and sent with every request.
    """
    to_encode = data.copy()

    # Set expiry time for the token
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    # Sign and encode the token using the secret key
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and verify a JWT token.
    Returns the payload (user info) if valid, None if expired or tampered.
    This is called on every protected route to check if the user is authenticated.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        # Token is invalid, expired, or has been tampered with
        return None


def get_current_user(token: str):
    """
    Given a JWT token from the browser cookie, return the full user record
    from the database. Returns None if token is invalid or user doesn't exist.
    This is used by route handlers to identify who is making each request.
    """
    # Import here to avoid circular imports between auth and database modules
    from src.database import get_connection

    # Decode the token to get the username
    payload = decode_token(token)
    if not payload:
        return None

    username = payload.get("sub")  # "sub" is the standard JWT field for subject (username)

    # Look up the user in the database by username
    conn = get_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    return user  # Returns full user row including role, email, etc.