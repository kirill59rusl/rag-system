from itertools import zip_longest
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.database import get_session
from src.db.service import find_lexical, find_similar, get_known_versions
from src.retrieval.dependencies import get_embedding_model, get_reranker
from src.retrieval.embedding import EmbeddingModel
from src.retrieval.fusion import rrf
from src.retrieval.params import RetrievalParams, RetrievalParamsDep
from src.retrieval.reranking.reranker import Reranker
from src.retrieval.versions import detect_versions

retrieval=APIRouter(prefix="/retrieval", tags=["query"])

SessionDep=Annotated[AsyncSession, Depends(get_session)]    
EmbeddingDep = Annotated[EmbeddingModel, Depends(get_embedding_model)]

RerankerDep = Annotated[Reranker | None, Depends(get_reranker)]

async def _find(query, query_embedding, session, limit, version, params):
    if params.mode == "bm25":
        return await find_lexical(session, query, limit, version)
    if params.mode == "hybrid":
        # одна сессия — запросы последовательно, не через gather
        vector = await find_similar(session, query_embedding, limit, version)
        lexical = await find_lexical(session, query, limit, version)
        fused = rrf(
            [vector, lexical],
            [params.vector_weight, params.bm25_weight],
            params.rrf_k,
        )
        return fused[:limit]
    return await find_similar(session, query_embedding, limit, version)


async def _search(query, query_embedding, session, limit, reranker_limit, reranker, params, version=None):
    if reranker is None:
        return await _find(query, query_embedding, session, limit, version, params)
    candidates = await _find(
        query, query_embedding, session, max(reranker_limit, limit), version, params
    )
    return await reranker.rerank(query, candidates, reranker_limit)


async def get_similar(
    query, session, embedding_model, limit, reranker_limit, reranker=None,
    params: RetrievalParams | None = None,
):
    params = params or RetrievalParams.resolve()
    query_embedding = None
    if params.mode != "bm25":
        query_embedding = await embedding_model.embed_one(query)
    versions = []
    if settings.version_filter:
        versions = detect_versions(query, await get_known_versions(session))
    if not versions:
        return await _search(query, query_embedding, session, limit, reranker_limit, reranker, params)

    per_version = [
        await _search(
            query, query_embedding, session,
            _share(limit, len(versions), i), _share(reranker_limit, len(versions), i),
            reranker, params, version,
        )
        for i, version in enumerate(versions)
    ]
    return [d for group in zip_longest(*per_version) for d in group if d is not None]


def _share(total, n, i):
    """i-я доля при делении total на n частей: 5 на 2 -> 3, 2."""
    if total is None:
        return None
    return total // n + (i < total % n)

@retrieval.post("/search")
async def get_relevant(
    query: str,
    session: SessionDep,
    embedding_model: EmbeddingDep,
    reranker: RerankerDep,
    params: RetrievalParamsDep,
    limit: int = 5,
    reranker_limit: int = 5,
):
    return await get_similar(query,session,embedding_model,limit,reranker_limit,reranker,params)