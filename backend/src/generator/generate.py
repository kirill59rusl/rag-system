from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_session
from src.generator.dependencies import get_llm
from src.generator.llm import LLM
from src.generator.prompt import build_prompt
from src.retrieval.embedding import EmbeddingModel
from src.retrieval.query import EmbeddingDep, get_similar
from src.schemas.llmresponse import Claim, LLMResponse

llm=APIRouter(prefix="/llm", tags=["llm"])

SessionDep=Annotated[AsyncSession, Depends(get_session)]
LLMDep = Annotated[LLM, Depends(get_llm)]

def build_sources(
    similar: list[dict],
    claims: list[Claim],
) -> list[dict]:

    source_ids = {
        source_id
        for claim in claims
        for source_id in claim.sources
    }

    sources = []

    for source_id in sorted(source_ids):
        if not 1 <= source_id <= len(similar):
            continue

        chunk = similar[source_id - 1]

        sources.append({
            "id": source_id,
            "doc_id": chunk["doc_id"],
            "page_number": chunk["page_number"],
        })

    return sources

async def generate(
    query: str, session: AsyncSession, limit: int, llm: LLM, embedding_model: EmbeddingModel
):
    similar=await get_similar(session=session, query=query, embedding_model=embedding_model, limit=limit)
    
    prompt = build_prompt(similar,query)
    
    response=await llm.generate(
        prompt=prompt,
        response_format=LLMResponse.model_json_schema()
    )

    result = LLMResponse.model_validate_json(response)

    sources = build_sources(
        similar=similar,
        claims=result.claims,
    )

    return {
        "answer": result.answer,
        "claims": result.claims,
        "sources": sources,
    }

    

@llm.post("/generate")
async def ask(
    session: SessionDep,
    query: str,
    llm: LLMDep,
    embedding_model: EmbeddingDep,
    limit: int = 5
):
   return await generate(
    session=session,
    query=query,
    llm=llm,
    embedding_model=embedding_model,
    limit=limit
   )