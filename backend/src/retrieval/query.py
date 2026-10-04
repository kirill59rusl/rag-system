from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_session
from src.db.service import find_similar
from src.retrieval.dependencies import get_embedding_model, get_reranker
from src.retrieval.embedding import EmbeddingModel
from src.retrieval.reranking.reranker import Reranker

retrieval=APIRouter(prefix="/retrieval", tags=["query"])

SessionDep=Annotated[AsyncSession, Depends(get_session)]    
EmbeddingDep = Annotated[EmbeddingModel, Depends(get_embedding_model)]

RerankerDep = Annotated[Reranker | None, Depends(get_reranker)]

async def get_similar(query, session, embedding_model, limit, reranker_limit, reranker=None):
    query_embedding = await embedding_model.embed_one(query)
    if reranker is None:
        return await find_similar(session, query_embedding, limit)
    candidates = await find_similar(
        session, query_embedding, max(reranker_limit, limit)
    )
    return await reranker.rerank(query, candidates, reranker_limit)

@retrieval.post("/search")
async def get_relevant(
    query: str,
    session: SessionDep,
    embedding_model: EmbeddingDep,
    reranker: RerankerDep,
    limit: int = 5,
    reranker_limit: int = 5,
):
    return await get_similar(query,session,embedding_model,limit,reranker_limit,reranker)