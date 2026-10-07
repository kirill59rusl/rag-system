from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_session
from src.evaluation.judge import get_judge
from src.evaluation.service import run_evaluation
from src.generator.dependencies import get_llm
from src.generator.llm import LLM
from src.retrieval.dependencies import get_reranker
from src.retrieval.query import EmbeddingDep
from src.retrieval.reranking.reranker import Reranker
from src.schemas.evaluation import EvalSummary

evaluation = APIRouter(prefix="/eval", tags=["eval"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
LLMDep = Annotated[LLM, Depends(get_llm)]
JudgeDep = Annotated[LLM, Depends(get_judge)]
RerankerDep = Annotated[Reranker | None, Depends(get_reranker)]

@evaluation.post("/run", response_model=EvalSummary)
async def run(
    session: SessionDep,
    llm: LLMDep,
    judge: JudgeDep,
    embedding_model: EmbeddingDep,
    reranker: RerankerDep,
    use_reranker: bool = True,
    limit: int = 10,
    reranker_limit: int = 10,
    lang: Literal["ru", "en"] = "ru",
    ks: Annotated[list[int] | None, Query()] = None,
    strict_version: bool = False,
    retrieval_only: bool = False,
    categories: Annotated[list[str] | None, Query()] = None,
    question_types: Annotated[list[str] | None, Query()] = None,
    difficulties: Annotated[list[str] | None, Query()] = None,
    ids: Annotated[list[str] | None, Query()] = None,
    sample_limit: int | None = None,
    concurrency: int = 5
):
    """
    Прогоняет датасет вопросов через RAG-пайплайн и оценивает ретривер и ответы LLM-судьёй.

    - use_reranker: использовать реранкер из .env (false — только векторный поиск)
    - limit / reranker_limit: сколько чанков ретривить на вопрос (как в /llm/generate);
      с реранкером limit — число кандидатов, reranker_limit — сколько оставить после него;
      для hit@10 нужно не меньше 10
    - lang: на каком языке задавать вопрос (question_ru / question_en)
    - ks: для каких k считать hit@k / recall@k / all@k, по умолчанию 1, 3, 5, 10
    - strict_version: чанк засчитывается, только если он из той же версии документации
    - retrieval_only: не вызывать LLM и судью, считать только метрики ретривера
    - categories: pg18 / pg19 / cross / unanswerable
    - question_types: fact / parameter / syntax / how-to / conceptual / comparison / unanswerable
    - difficulties: easy / medium / hard
    - ids: конкретные вопросы, например ?ids=pgq-001&ids=pgq-002
    - sample_limit: ограничить число вопросов (для быстрой проверки)
    """
    if use_reranker and reranker is None:
        raise HTTPException(400, "Реранкер не настроен: задайте RERANKER_PROVIDER и RERANKER")

    return await run_evaluation(
        session=session,
        llm=llm,
        judge=judge,
        embedding_model=embedding_model,
        limit=limit,
        reranker_limit=reranker_limit,
        reranker=reranker if use_reranker else None,
        lang=lang,
        ks=ks,
        strict_version=strict_version,
        retrieval_only=retrieval_only,
        categories=categories,
        question_types=question_types,
        difficulties=difficulties,
        ids=ids,
        sample_limit=sample_limit,
        concurrency=concurrency
    )
