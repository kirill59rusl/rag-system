from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.database import get_session
from src.db.models import Chunk
from src.retrieval.embedding import OllamaEmbedding

embedding_model=OllamaEmbedding(
    model="embeddinggemma",
    dimension=settings.embedding_space
)

retrieval=APIRouter(prefix="/retrieval", tags=["query"])

SessionDep=Annotated[AsyncSession, Depends(get_session)]

async def find_similar(
    session: AsyncSession,
    query: str,
    limit: int
):

    query_embedding=embedding_model.embed_one(query)

    distance=Chunk.embedding.cosine_distance(query_embedding)
    
    chunks =(
        select(Chunk, distance)
        .order_by(distance.asc())
        .limit(limit)
    )
    
    rows = (await session.execute(chunks)).all()
    result = [{**chunk.as_dict(), "distance": float(dist)}
                for chunk,dist in rows]

    return result

@retrieval.post("/search")
async def get_similar(
    query: str,
    session: SessionDep,
    limit: int = 5
):
    return await find_similar(session, query, limit)