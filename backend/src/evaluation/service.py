import asyncio
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.database import async_session_factory
from src.db.models import Document
from src.generator.generate import generate as rag_generate
from src.generator.llm import LLM
from src.generator.prompt import format_context
from src.retrieval.query import get_similar
from src.retrieval.reranking.reranker import Reranker
from src.schemas.evaluation import (
    EvalCaseResult,
    EvalSummary,
    FaithfulnessVerdict,
    FragmentMatch,
    GroupStats,
    JudgeVerdict,
)
from src.schemas.llmresponse import Claim

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """Ты — эксперт, оценивающий ответы RAG-системы по технической документации.

Вопрос пользователя:
{question}

Эталонный (правильный) ответ:
{golden_answer}

Цитаты из документации, на которых основан эталонный ответ:
{quotes}

Ответ, сгенерированный системой:
{generated_answer}

Оцени в поле verdict, насколько сгенерированный ответ соответствует эталонному по смыслу:
- "correct" — ответ верен и содержит всю ключевую информацию из эталона;
- "partial" — ответ частично верен, но что-то упущено или неточно;
- "incorrect" — ответ неверен, противоречит эталону или отвечает не на тот вопрос.
Если эталонный ответ говорит, что информации нет в документации, а система тоже
отказалась отвечать по существу — это "correct".
Дополнительные сведения сверх эталона не снижают оценку, если они не противоречат эталону.

Кратко обоснуй вердикт в поле reasoning (1-2 предложения)."""

FAITHFULNESS_PROMPT = """Ты проверяешь, подтверждаются ли утверждения ответа RAG-системы
фрагментами документации, которые ей были выданы. Не используй собственные знания:
утверждение подтверждено, только если оно прямо следует из текста источников.

Источники:
{context}

Утверждения ответа:
{claims}

Для каждого утверждения верни объект с полями:
- claim — номер утверждения;
- supported — true, если утверждение подтверждается хотя бы одним из источников выше.
Оцени все утверждения по порядку."""

# фраза отказа из SYSTEM_PROMPT генератора (+ английский вариант, если LLM её перевела)
_REFUSAL = re.compile(
    r"нет информации|no information|does not contain information|not contain any information",
    re.IGNORECASE,
)

_VERSION_IN_FILENAME = re.compile(r"postgresql-(\d+)", re.IGNORECASE)

# доля цитаты с любого края, достаточная для попадания (цитату мог разрезать чанкер)
EDGE_FRACTION = 0.6


def load_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    dataset_path = path or settings.eval_dataset_path
    with open(dataset_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize(text: str) -> str:
    """Только буквы и цифры: пробелы, переносы и разметка PDF не мешают сравнению."""
    return re.sub(r"[\W_]+", "", text.lower())


def _quote_in_chunk(quote: str, chunk: str) -> bool:
    if not quote:
        return False
    if quote in chunk:
        return True
    k = int(len(quote) * EDGE_FRACTION)
    return quote[:k] in chunk or quote[-k:] in chunk


async def load_document_versions(session: AsyncSession) -> dict[str, str | None]:
    """doc_id -> версия PostgreSQL ("18"/"19"): из имени файла, иначе из Document.version."""
    rows = await session.execute(select(Document.id, Document.filename, Document.version))
    versions = {}
    for doc_id, filename, version in rows:
        match = _VERSION_IN_FILENAME.search(filename or "")
        versions[str(doc_id)] = match.group(1) if match else version
    return versions


def match_fragments(
    fragments: list[dict],
    retrieved: list[dict],
    versions: dict[str, str | None],
    strict_version: bool,
) -> list[FragmentMatch]:
    """Для каждого эталонного фрагмента ищет первый retrieved-чанк, где он есть."""
    chunks = [
        {**c, "norm": _normalize(c.get("content") or ""), "version": versions.get(c["doc_id"])}
        for c in retrieved
    ]

    result = []
    for fragment in fragments:
        # места, где этот текст есть в документации: основное и дубль в другой версии
        locations = [fragment]
        if not strict_version and fragment.get("also_in_other_version"):
            locations.append(fragment["also_in_other_version"])
        allowed = {loc["version"] for loc in locations}

        quote = _normalize(fragment["quote"])
        text_rank = next(
            (
                rank for rank, c in enumerate(chunks, 1)
                if (c["version"] in allowed or (not strict_version and c["version"] is None))
                and _quote_in_chunk(quote, c["norm"])
            ),
            None,
        )

        def at_location(c: dict, loc: dict) -> bool:
            return c["version"] == loc["version"]

        section_found = any(
            at_location(c, loc) and loc["section"] in (c.get("section_path") or [])
            for c in chunks for loc in locations
        )
        page_found = any(
            at_location(c, loc)
            and c.get("page_number") is not None
            and c["page_number"] <= loc["pdf_page_end"]
            and (c.get("page_end") or c["page_number"]) >= loc["pdf_page_start"]
            for c in chunks for loc in locations
        )

        result.append(FragmentMatch(
            version=fragment["version"],
            section=fragment["section"],
            quote=fragment["quote"],
            text_rank=text_rank,
            section_found=section_found,
            page_found=page_found,
        ))
    return result


def retrieval_metrics(matches: list[FragmentMatch], ks: list[int]) -> dict[str, Any]:
    if not matches:
        return {}
    ranks = [m.text_rank for m in matches]
    found_at = {k: [r is not None and r <= k for r in ranks] for k in ks}
    first = min((r for r in ranks if r is not None), default=None)
    return {
        "hit": {k: float(any(f)) for k, f in found_at.items()},
        "recall": {k: sum(f) / len(f) for k, f in found_at.items()},
        "all_found": {k: float(all(f)) for k, f in found_at.items()},
        "mrr": 1 / first if first else 0.0,
        "section_recall": sum(m.section_found for m in matches) / len(matches),
        "page_recall": sum(m.page_found for m in matches) / len(matches),
    }


async def judge_answer(
    llm: LLM, question: str, golden_answer: str, quotes: list[str], generated_answer: str
) -> JudgeVerdict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        golden_answer=golden_answer,
        quotes="\n".join(f"- {q}" for q in quotes) or "(нет)",
        generated_answer=generated_answer,
    )
    raw = await llm.generate(
        prompt=prompt, response_format=JudgeVerdict.model_json_schema()
    )
    return JudgeVerdict.model_validate_json(raw)


async def judge_faithfulness(
    llm: LLM, claims: list[Claim], retrieved: list[dict]
) -> FaithfulnessVerdict:
    prompt = FAITHFULNESS_PROMPT.format(
        context=format_context(retrieved),
        claims="\n".join(
            f"{i}. {c.text}"
            for i, c in enumerate(claims, 1)
        ),
    )
    raw = await llm.generate(
        prompt=prompt, response_format=FaithfulnessVerdict.model_json_schema()
    )
    return FaithfulnessVerdict.model_validate_json(raw)


def faithfulness_score(claims: list[Claim], verdict: FaithfulnessVerdict) -> float | None:
    """Доля утверждений, подтверждённых контекстом."""
    checks = {c.claim: c.supported for c in verdict.claims if 1 <= c.claim <= len(claims)}
    if len(checks) < len(claims):
        logger.warning("judge checked %d of %d claims", len(checks), len(claims))
    if not checks:
        return None
    return sum(checks.values()) / len(checks)


async def run_case(
    case: dict[str, Any],
    session: AsyncSession,
    llm: LLM,
    judge: LLM,
    embedding_model: str,
    reranker: Reranker | None,
    limit: int,
    reranker_limit: int,
    versions: dict[str, str | None],
    lang: str,
    ks: list[int],
    strict_version: bool,
    retrieval_only: bool,
) -> EvalCaseResult:
    question = case[f"question_{lang}"]
    golden_answer = case[f"answer_{lang}"]
    fragments = case["gold_fragments"]

    generated_answer = None
    claims: list[Claim] = []
    if retrieval_only:
        retrieved = await get_similar(
            query=question, session=session, embedding_model=embedding_model,
            limit=limit, reranker_limit=reranker_limit, reranker=reranker,
        )
    else:
        result = await rag_generate(
            query=question,
            session=session,
            limit=limit,
            llm=llm,
            reranker=reranker,
            reranker_limit=reranker_limit,
            embedding_model=embedding_model,
            debug=True,
        )
        generated_answer = result.answer
        claims = result.claims
        retrieved = [s.model_dump() for s in result.sources]

    matches = match_fragments(fragments, retrieved, versions, strict_version)

    case_result = EvalCaseResult(
        id=case["id"],
        category=case["category"],
        question_type=case["question_type"],
        difficulty=case["difficulty"],
        hop_type=case["hop_type"],
        answerable=case["answerable"],
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
        retrieved=[
            f"v{versions.get(c['doc_id'])} стр. {c.get('page_number')}: "
            f"{' > '.join(c.get('section_path') or [])}"
            for c in retrieved
        ],
        fragments=matches,
        **retrieval_metrics(matches, ks),
    )

    if generated_answer is not None:
        case_result.refused = bool(_REFUSAL.search(generated_answer))
        verdict = await judge_answer(
            llm=judge,
            question=question,
            golden_answer=golden_answer,
            quotes=[f["quote"] for f in fragments],
            generated_answer=generated_answer,
        )
        case_result.verdict = verdict.verdict
        case_result.reasoning = verdict.reasoning

        # отказ без утверждений проверять не на чем — он не входит в faithfulness
        if claims and not case_result.refused:
            faithfulness = await judge_faithfulness(judge, claims, retrieved)
            case_result.faithfulness = faithfulness_score(claims, faithfulness)

    return case_result


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(results: list[EvalCaseResult], ks: list[int]) -> GroupStats:
    with_gold = [r for r in results if r.hit is not None]
    judged = [r for r in results if r.verdict is not None]
    checked = [r for r in results if r.faithfulness is not None]

    def at_k(field: str) -> dict[int, float]:
        if not with_gold:
            return {}
        return {k: _mean([getattr(r, field)[k] for r in with_gold]) for k in ks}

    return GroupStats(
        total=len(results),
        hit=at_k("hit"),
        recall=at_k("recall"),
        all_found=at_k("all_found"),
        mrr=_mean([r.mrr for r in with_gold]),
        section_recall=_mean([r.section_recall for r in with_gold]),
        page_recall=_mean([r.page_recall for r in with_gold]),
        refusal_rate=_mean([float(r.refused) for r in judged]),
        accuracy=_mean([float(r.verdict == "correct") for r in judged]),
        partial_rate=_mean([float(r.verdict == "partial") for r in judged]),
        faithfulness=_mean([r.faithfulness for r in checked if r.faithfulness is not None]),
    )


def group_by(
    results: list[EvalCaseResult], key: Callable[[EvalCaseResult], str], ks: list[int]
) -> dict[str, GroupStats]:
    groups: dict[str, list[EvalCaseResult]] = {}
    for r in results:
        groups.setdefault(key(r), []).append(r)
    return {name: aggregate(items, ks) for name, items in sorted(groups.items())}


async def run_evaluation(
    session: AsyncSession,
    llm: LLM,
    judge: LLM,
    embedding_model: str,
    reranker: Reranker | None,
    limit: int = 5,
    reranker_limit: int = 5,
    lang: str = "ru",
    ks: list[int] | None = None,
    strict_version: bool = False,
    retrieval_only: bool = False,
    categories: list[str] | None = None,
    question_types: list[str] | None = None,
    difficulties: list[str] | None = None,
    ids: list[str] | None = None,
    sample_limit: int | None = None,
    concurrency: int=10,
) -> EvalSummary:
    ks = sorted(ks or [1, 3, 5, 10])

    
    dataset = load_dataset()

    if ids:
        dataset = [c for c in dataset if c["id"] in ids]
    if categories:
        dataset = [c for c in dataset if c["category"] in categories]
    if question_types:
        dataset = [c for c in dataset if c["question_type"] in question_types]
    if difficulties:
        dataset = [c for c in dataset if c["difficulty"] in difficulties]
    if sample_limit:
        dataset = dataset[:sample_limit]

    done=0

    versions = await load_document_versions(session)

    sem=asyncio.Semaphore(concurrency)
    async def worker(case):
        nonlocal done
        async with sem:
            try:
                async with async_session_factory() as case_session:
                    res = await run_case(
                            case, case_session, llm, judge, embedding_model, reranker, limit, reranker_limit,
                            versions, lang, ks, strict_version, retrieval_only,
                    )
            except Exception:
                logger.exception("eval failed: %s", case["id"])
                return None
        done+=1
        logger.info("eval %d/%d: %s", done, len(dataset), case["id"])
        return res

    results = [r for r in await asyncio.gather(*map(worker, dataset)) if r is not None]

    return EvalSummary(
        lang=lang,
        strict_version=strict_version,
        retrieval_only=retrieval_only,
        reranker=getattr(reranker, "model", type(reranker).__name__) if reranker else None,
        limit=limit,
        reranker_limit=reranker_limit if reranker else None,
        ks=ks,
        overall=aggregate(results, ks),
        by_category=group_by(results, lambda r: r.category, ks),
        by_question_type=group_by(results, lambda r: r.question_type, ks),
        by_difficulty=group_by(results, lambda r: r.difficulty, ks),
        by_hop_type=group_by(results, lambda r: r.hop_type, ks),
        unknown_documents=[doc_id for doc_id, v in versions.items() if v is None],
        results=results,
    )
