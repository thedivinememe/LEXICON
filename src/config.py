from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
import torch
import os
import re

class Settings(BaseSettings):
    # Application
    app_name: str = "LEXICON"
    debug: bool = Field(False, env="DEBUG")
    secret_key: str = Field("development_secret_key", env="SECRET_KEY")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    environment: str = Field("development", env="ENVIRONMENT")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    @property
    def postgres_database_url(self) -> str:
        """
        Convert Heroku's DATABASE_URL to a format compatible with asyncpg.
        Heroku provides a postgres:// URL, but asyncpg requires postgresql://
        """
        if self.database_url.startswith("postgres://"):
            return self.database_url.replace("postgres://", "postgresql://", 1)
        return self.database_url
    
    @property
    def is_production(self) -> bool:
        """Check if the application is running in production mode"""
        return self.environment.lower() == "production" or os.environ.get("HEROKU_APP_NAME") is not None
    
    # Neural Network
    model_dim: int = 768  # BERT compatible
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    
    # Vector Store
    vector_index_path: str = "./data/faiss_index/index.faiss"
    vector_dimension: int = 768
    
    # API
    api_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:3000,http://localhost:8000"
    
    @property
    def cors_origins_list(self) -> list:
        return self.cors_origins.split(",") if self.cors_origins else []
    
    # Features
    enable_gpu: bool = torch.cuda.is_available()
    enable_meme_evolution: bool = True
    enable_real_time_updates: bool = True
    
    # COREE
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4-turbo", env="OPENAI_MODEL")
    
    class Config:
        env_file = ".env"

settings = Settings()
