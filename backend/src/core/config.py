from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["vector", "bm25", "hybrid"]

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    embedder_provider: str 
    embedder: str
    embedding_space: int
    embed_batch_size: int = Field(default=128, gt=0)

    database_url: str

    llm_provider: str 
    llm: str  

    judge: str
    
    gigachat_credentials: str | None = None

    openai_api_key: str | None = None
    openai_base_url: str | None = None

    reranker_provider: str | None = None
    reranker: str | None = None

    # искать только в версиях документации, упомянутых в вопросе
    version_filter: bool = True

    # vector — косинусная близость эмбеддингов, bm25 — лексический поиск ParadeDB,
    # hybrid — оба списка (по limit кандидатов), объединённые через RRF
    retrieval_mode: RetrievalMode = "vector"
    rrf_k: int = Field(default=60, gt=0)
    vector_weight: float = Field(default=1.0, ge=0)
    bm25_weight: float = Field(default=1.0, ge=0)
    # переводить вопрос на английский (через LLM) перед BM25: документация английская
    translate_for_bm25: bool = False

    eval_dataset_path: Path = BASE_DIR / "app" / "src" / "evaluation" / "dataset" / "dataset.jsonl"
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    max_upload_size: int = 50 * 1024 * 1024  # 50 MB
    allowed_content_types: set[str] = {
        "application/pdf",
        "text/plain",
        "text/markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",
    )


settings = Settings()  # type: ignore[call-arg]