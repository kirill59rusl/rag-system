from typing import Literal

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    grounded: bool
    reasoning: str


class FragmentMatch(BaseModel):
    """Нашёлся ли эталонный фрагмент среди retrieved и на какой позиции (с 1)."""
    version: str
    section: str
    quote: str
    text_rank: int | None = None
    section_found: bool = False
    page_found: bool = False


class EvalCaseResult(BaseModel):
    id: str
    category: str
    question_type: str
    difficulty: str
    hop_type: str
    answerable: bool
    question: str
    golden_answer: str
    generated_answer: str | None = None
    retrieved: list[str] = Field(default_factory=list)
    fragments: list[FragmentMatch] = Field(default_factory=list)
    # retrieval-метрики; None для вопросов без эталонных фрагментов (unanswerable)
    hit: dict[int, float] | None = None
    recall: dict[int, float] | None = None
    all_found: dict[int, float] | None = None
    mrr: float | None = None
    section_recall: float | None = None
    page_recall: float | None = None
    # генерация
    refused: bool | None = None
    verdict: Literal["correct", "partial", "incorrect"] | None = None
    grounded: bool | None = None
    reasoning: str | None = None


class GroupStats(BaseModel):
    total: int
    hit: dict[int, float] = Field(default_factory=dict)
    recall: dict[int, float] = Field(default_factory=dict)
    all_found: dict[int, float] = Field(default_factory=dict)
    mrr: float | None = None
    section_recall: float | None = None
    page_recall: float | None = None
    # доля ответов-отказов: для unanswerable чем выше, тем лучше, для остальных — наоборот
    refusal_rate: float | None = None
    accuracy: float | None = None
    partial_rate: float | None = None
    grounded_rate: float | None = None


class EvalSummary(BaseModel):
    lang: str
    strict_version: bool
    retrieval_only: bool
    # модель реранкера; None — прогон без реранкера
    reranker: str | None = None
    limit: int
    reranker_limit: int | None = None
    ks: list[int]
    overall: GroupStats
    by_category: dict[str, GroupStats]
    by_question_type: dict[str, GroupStats]
    by_difficulty: dict[str, GroupStats]
    by_hop_type: dict[str, GroupStats]
    unknown_documents: list[str] = Field(default_factory=list)
    results: list[EvalCaseResult]
