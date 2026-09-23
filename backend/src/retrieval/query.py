from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_session
from src.db.service import find_similar
from src.retrieval.dependencies import get_embedding_model
from src.retrieval.embedding import EmbeddingModel

retrieval=APIRouter(prefix="/retrieval", tags=["query"])

SessionDep=Annotated[AsyncSession, Depends(get_session)]    
EmbeddingDep = Annotated[EmbeddingModel, Depends(get_embedding_model)]

async def get_similar(
    query: str,
    session: AsyncSession,
    embedding_model: EmbeddingModel,
    limit
):
    query_embedding=await embedding_model.embed_one(query)
    return await find_similar(session, query_embedding, limit)

@retrieval.post("/search")
async def get_relevant(
    query: str,
    session: SessionDep,
    embedding_model: EmbeddingDep,
    limit: int = 5
):
    return await get_similar(query,session,embedding_model,limit)