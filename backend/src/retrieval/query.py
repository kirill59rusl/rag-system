from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.models import Chunk
from src.retrieval.embedding import OllamaEmbedding

embedding_model=OllamaEmbedding(
    model="embeddinggemma",
    dimension=settings.embedding_space
)

async def find_similar(
    session: AsyncSession,
    query: str,
    limit: int = 5
):

    query_embedding=embedding_model.embed_one(query)

    distance=Chunk.embedding.cosine_distance(query_embedding)
    
    result = await session.scalar(
        select(Chunk)
        .order_by(distance.asc())
        .limit(limit)
    )

    return result.all()