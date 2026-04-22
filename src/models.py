from pydantic import BaseModel
from typing import Optional

class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    username: str

class MovieRequest(BaseModel):
    movie_title: str
    top_n: int = 10

class EmotionRequest(BaseModel):
    emotion: str
    top_n: int = 10