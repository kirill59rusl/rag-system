from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    database_url: str

    class Config:
        env_file = BASE_DIR / "backend" / ".env"

    upload_dir: Path = BASE_DIR / "data" / "uploads"
    max_upload_size: int = 50 * 1024 * 1024  # 50 MB
    allowed_content_types: set[str] = {
        "application/pdf",
        "text/plain",
        "text/markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

settings = Settings()  # type: ignore[call-arg]