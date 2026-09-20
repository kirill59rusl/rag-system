from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.database import get_session
from src.db.models import Chunk
from src.db.service import find_similar
from src.retrieval.embedding import OllamaEmbedding

embedding_model=OllamaEmbedding(
    model=settings.embedder,
    dimension=settings.embedding_space
)

retrieval=APIRouter(prefix="/retrieval", tags=["query"])

SessionDep=Annotated[AsyncSession, Depends(get_session)]    

async def get_similar(
    query: str,
    session: AsyncSession,
    limit
):
    query_embedding=await embedding_model.embed_one(query)
    return await find_similar(session, query_embedding, limit)

@retrieval.post("/search")
async def get_relevant(
    query: str,
    session: SessionDep,
    limit: int = 5
):
    return await get_similar(query,session,limit)