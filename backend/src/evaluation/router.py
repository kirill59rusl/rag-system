from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_session
from src.evaluation.judge import get_judge
from src.evaluation.service import run_evaluation
from src.generator.dependencies import get_llm
from src.generator.llm import LLM
from src.retrieval.query import EmbeddingDep
from src.schemas.evaluation import EvalSummary

evaluation = APIRouter(prefix="/eval", tags=["eval"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
LLMDep = Annotated[LLM, Depends(get_llm)]
JudgeDep = Annotated[LLM, Depends(get_judge)]

@evaluation.post("/run", response_model=EvalSummary)
async def run(
    session: SessionDep,
    llm: LLMDep,
    embedding_model: EmbeddingDep,
    limit: int = 5,
    sample_limit: int | None = None,
    question_types: Annotated[list[str] | None, Query()] = None,
):
    """
    Прогоняет датасет вопросов через RAG-пайплайн и оценивает ответы LLM-судьёй.

    - limit: сколько чанков ретривить на вопрос (как в /llm/generate)
    - sample_limit: ограничить число вопросов из датасета (для быстрой проверки)
    - question_types: фильтр по типам, например ?question_types=factual&question_types=safety
    """
    return await run_evaluation(
        session=session,
        llm=llm,
        embedding_model=embedding_model,
        limit=limit,
        question_types=question_types,
        sample_limit=sample_limit,
    )