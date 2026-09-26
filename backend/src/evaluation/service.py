import json
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.generator.generate import generate as rag_generate
from src.generator.llm import LLM
from src.schemas.evaluation import EvalCaseResult, EvalSummary, JudgeVerdict

JUDGE_PROMPT = """Ты — эксперт, оценивающий ответы RAG-системы по технической документации.

Вопрос пользователя:
{question}

Эталонный (правильный) ответ:
{golden_answer}

Ответ, сгенерированный системой:
{generated_answer}

Оцени сгенерированный ответ по двум критериям:

1. verdict — насколько сгенерированный ответ соответствует эталонному по смыслу:
   - "correct" — ответ верен и содержит всю ключевую информацию из эталона;
   - "partial" — ответ частично верен, но что-то упущено, неточно или избыточно;
   - "incorrect" — ответ неверен, противоречит эталону или отвечает не на тот вопрос.
   Если эталонный ответ говорит, что информации нет в документации, а система тоже
   отказалась отвечать по существу — это "correct".

2. grounded — false, если в сгенерированном ответе есть утверждения, не подтверждённые
   эталонным ответом (похоже на выдумку/галлюцинацию), иначе true.

Кратко обоснуй вердикт в поле reasoning (1-2 предложения)."""


def load_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    dataset_path = path or settings.eval_dataset_path
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def _word_overlap(a: str, b: str) -> float:
    """Доля слов эталонного текста, встречающихся в найденном чанке."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def compute_page_recall(
    golden_evidence: list[dict], retrieved_sources: list
) -> float | None:
    """Грубая метрика: попал ли ретривер хотя бы в нужный doc_id+page_number.
    Не зависит от chunk_size."""
    if not golden_evidence:
        return None
    golden_pages = {(e["doc_id"], e["page_number"]) for e in golden_evidence}
    retrieved_pages = {(s.doc_id, s.page_number) for s in retrieved_sources}
    return len(golden_pages & retrieved_pages) / len(golden_pages)


def compute_text_recall(
    golden_evidence: list[dict],
    retrieved_sources: list,
    threshold: float = 0.3,
) -> float | None:
    """Точная метрика: для каждого эталонного фрагмента — нашёлся ли среди
    retrieved чанк с достаточным пересечением слов. Не зависит от того, где
    именно прошла граница нового чанка."""
    if not golden_evidence:
        return None
    found = 0
    for e in golden_evidence:
        if any(
            _word_overlap(e["reference_text"], s.content or "") >= threshold
            for s in retrieved_sources
        ):
            found += 1
    return found / len(golden_evidence)


async def judge_answer(
    llm: LLM, question: str, golden_answer: str, generated_answer: str
) -> JudgeVerdict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
    )
    raw = await llm.generate(
        prompt=prompt, response_format=JudgeVerdict.model_json_schema()
    )
    return JudgeVerdict.model_validate_json(raw)


async def run_case(
    case: dict[str, Any],
    session: AsyncSession,
    llm: LLM,
    embedding_model: str,
    limit: int,
) -> EvalCaseResult:
    result = await rag_generate(
        query=case["question"],
        session=session,
        limit=limit,
        llm=llm,
        embedding_model=embedding_model,
        debug=True,
    )
    retrieved_chunk_ids = [s.chunk_id for s in result.sources]
    golden_evidence = case.get("source_evidence", [])
    page_recall = compute_page_recall(golden_evidence, result.sources)
    text_recall = compute_text_recall(golden_evidence, result.sources)

    verdict = await judge_answer(
        llm=llm,
        question=case["question"],
        golden_answer=case["answer"],
        generated_answer=result.answer,
    )

    return EvalCaseResult(
        id=case["id"],
        question=case["question"],
        question_type=case["question_type"],
        golden_answer=case["answer"],
        generated_answer=result.answer,
        retrieved_chunk_ids=retrieved_chunk_ids,
        page_recall=page_recall,
        text_recall=text_recall,
        verdict=verdict.verdict,
        grounded=verdict.grounded,
        reasoning=verdict.reasoning,
    )


async def run_evaluation(
    session: AsyncSession,
    llm: LLM,
    embedding_model: str,
    limit: int = 5,
    question_types: list[str] | None = None,
    ids: list[str] | None = None,
    sample_limit: int | None = None,
) -> EvalSummary:
    dataset = load_dataset()

    if ids:
        dataset = [c for c in dataset if c["id"] in ids]
    if question_types:
        dataset = [c for c in dataset if c["question_type"] in question_types]
    if sample_limit:
        dataset = dataset[:sample_limit]

    results = [
        await run_case(case, session, llm, embedding_model, limit)
        for case in dataset
    ]

    total = len(results)
    correct = sum(1 for r in results if r.verdict == "correct")
    partial = sum(1 for r in results if r.verdict == "partial")
    incorrect = sum(1 for r in results if r.verdict == "incorrect")
    grounded = sum(1 for r in results if r.grounded)
    page_recalls = [r.page_recall for r in results if r.page_recall is not None]
    text_recalls = [r.text_recall for r in results if r.text_recall is not None]

    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_type.setdefault(r.question_type, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if r.verdict == "correct":
            bucket["correct"] += 1

    return EvalSummary(
        total=total,
        correct=correct,
        partial=partial,
        incorrect=incorrect,
        accuracy=correct / total if total else 0.0,
        grounded_rate=grounded / total if total else 0.0,
        avg_page_recall=sum(page_recalls) / len(page_recalls) if page_recalls else None,
        avg_text_recall=sum(text_recalls) / len(text_recalls) if text_recalls else None,
        by_question_type=by_type,
        results=results,
    )