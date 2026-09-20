from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.database import get_session
from src.generator.llm import OllamaLLM
from src.generator.prompt import build_prompt
from src.retrieval.query import get_similar
from src.schemas.llmresponse import Claim, LLMResponse

model=OllamaLLM(
    model=settings.llm,
)

llm=APIRouter(prefix="/llm", tags=["llm"])

SessionDep=Annotated[AsyncSession, Depends(get_session)]

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
    query: str, session: AsyncSession, limit: int
):
    similar=await get_similar(session=session, query=query, limit=limit)
    
    prompt = build_prompt(similar,query)
    
    response=model.generate(
        prompt=prompt,
        format=LLMResponse.model_json_schema()
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
    limit: int = 5
):
   return await generate(
    session=session,
    query=query,
    limit=limit
   )