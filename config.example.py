from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://your_username@localhost/jeju_chatbot"
    
    # JWT
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256" 
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30 * 24 * 60  # 30 days
    
    # CORS
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]
    
    # Chatbot Integration
    CHATBOT_SERVICE_URL: str = "http://localhost:8001"
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Jeju Travel Chatbot API"
    PROJECT_VERSION: str = "1.0.0"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create global settings instance
settings = Settings()

# Database connection string for SQLAlchemy
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL 