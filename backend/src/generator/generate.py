from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_session
from src.generator.dependencies import get_llm
from src.generator.llm import LLM
from src.generator.prompt import build_prompt
from src.retrieval.dependencies import get_reranker
from src.retrieval.query import EmbeddingDep, get_similar
from src.retrieval.reranking.reranker import Reranker
from src.schemas.llmresponse import Claim, GenerateResponse, LLMResponse, Source

llm=APIRouter(prefix="/llm", tags=["llm"])

SessionDep=Annotated[AsyncSession, Depends(get_session)]
LLMDep = Annotated[LLM, Depends(get_llm)]
RerankerDep = Annotated[Reranker, Depends(get_reranker)]

def build_sources(similar, used_ids, with_content: bool) -> list[Source]:
    result = []
    for i, d in enumerate(similar,1):
        result.append(Source(
            id=i,
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            chunk_index=d["chunk_index"],
            page_number=d["page_number"],
            page_end=d.get("page_end"),
            section=d.get("section"),
            section_path=d.get("section_path") or [],
            content=d["content"] if with_content else None,
            distance=d["distance"] if with_content else None,
            used=(i in used_ids),
            version=d.get("version"),
        ))
    return result

def clean_claims(claims: list[Claim], n_sources: int) -> list[Claim]:
    return [
        Claim(text=c.text, sources=[s for s in c.sources if 1 <= s <= n_sources])
        for c in claims
    ]


async def generate(query, session, limit, reranker_limit, llm, embedding_model, reranker, debug=False) -> GenerateResponse:
    similar = await get_similar(
        session=session, query=query, embedding_model=embedding_model, limit=limit, reranker_limit=reranker_limit, reranker=reranker
    )
    response = await llm.generate(
        prompt=build_prompt(similar, query),
        response_format=LLMResponse.model_json_schema(),
    )
    result = LLMResponse.model_validate_json(response)
    claims = clean_claims(result.claims, len(similar))

    used_ids = {s for c in claims for s in c.sources}

    return GenerateResponse(
        answer=result.answer,
        claims=claims,
        sources=build_sources(similar, used_ids, with_content=debug),
    )


@llm.post("/generate", response_model=GenerateResponse, response_model_exclude_none=True)
async def ask(session: SessionDep, query: str, llm: LLMDep, reranker: RerankerDep,
              embedding_model: EmbeddingDep, limit: int = 5, reranker_limit: int = 5, debug: bool = False):
    return await generate(query=query, session=session, limit=limit, reranker_limit=reranker_limit, llm=llm,
                          embedding_model=embedding_model, reranker=reranker, debug=debug)