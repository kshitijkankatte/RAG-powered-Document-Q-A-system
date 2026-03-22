from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    openai_api_key: str
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "documents"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieved_docs: int = 5

    # Models
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
