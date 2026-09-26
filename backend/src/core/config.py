from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    embedder_provider: str 
    embedder: str
    embedding_space: int

    database_url: str

    llm_provider: str 
    llm: str  

    judge: str
    
    gigachat_credentials: str | None = None

    openai_api_key: str | None = None
    openai_base_url: str | None = None


    eval_dataset_path: Path = BASE_DIR / "app" / "src" / "evaluation" / "dataset.json"   
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