
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_session
from src.ingestion.loader import documents

app = FastAPI(
    title="RAG API",
    version="0.1.0",
)

app.include_router(documents)

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    async for session in get_session():
        await session.execute(text("SELECT 1"))

    return {"status": "ready"}

