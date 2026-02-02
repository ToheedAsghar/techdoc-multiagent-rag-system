"""
Configuration for the backend
Loads environment variables from the .env file and provide settings throughout the app
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings loaded from .env file
    """

    # OpenRouter Settings (for LLM)
    OPENROUTER_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.1
    OPENAI_MAX_TOKENS: int = 4096

    # OpenAI Settings (for embeddings - OpenRouter doesn't support embeddings)
    OPENAI_API_KEY: Optional[str] = None

    # GEMINI Settings
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.1
    GEMINI_MAX_TOKENS: int = 4096

    # Embeddings Settings (OpenRouter - OpenAI compatible)
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # Pinecone Settings
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "techdoc-intelligence"
    PINECONE_NAMESPACE: str = "default"
    
    # Retrieval Settings
    TOP_K_SIMPLE: int = 3
    TOP_K_COMPLEX: int = 7
    TOP_K_MULTIHOP: int = 10
    RELEVANCE_THRESHOLD: float = 0.05  # Lowered temporarily - scores are unexpectedly low

    # VALIDATION SETTINGS
    MAX_RETRIES: int = 3
    HALLUCINATION_THRESHOLD: float = 0.8

    # Application Settings
    API_V1_PREFIX: str = "/api/v1"

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """
    Create and Cache settings instance
    using the @lru_cache decorator means that we'll load the .env once,
    and use the same settings everywhere.
    """

    return Settings()

settings = get_settings()
