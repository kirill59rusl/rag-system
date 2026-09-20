from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.database import get_session
from src.generator.llm import OllamaLLM
from src.generator.prompt import build_prompt
from src.retrieval.query import get_similar

model=OllamaLLM(
    model=settings.llm,
)

llm=APIRouter(prefix="/llm", tags=["llm"])

SessionDep=Annotated[AsyncSession, Depends(get_session)]

async def generate(
    query: str, session: AsyncSession, limit: int
):
    similar=await get_similar(session=session, query=query, limit=limit)
    
    prompt = build_prompt(similar,query)
    
    response=model.generate(prompt=prompt)
    return response

    

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