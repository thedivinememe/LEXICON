from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
import torch

class Settings(BaseSettings):
    # Application
    app_name: str = "LEXICON"
    debug: bool = False
    secret_key: str = "development_secret_key"
    jwt_algorithm: str = "HS256"
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
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
    
    class Config:
        env_file = ".env"

settings = Settings()
