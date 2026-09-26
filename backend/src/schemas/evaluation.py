from typing import Literal

from pydantic import BaseModel


class JudgeVerdict(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    grounded: bool
    reasoning: str


class EvalCaseResult(BaseModel):
    id: str
    question: str
    question_type: str
    golden_answer: str
    generated_answer: str
    retrieved_chunk_ids: list[str]
    page_recall: float | None
    text_recall: float | None
    verdict: Literal["correct", "partial", "incorrect"]
    grounded: bool
    reasoning: str


class EvalSummary(BaseModel):
    total: int
    correct: int
    partial: int
    incorrect: int
    accuracy: float
    grounded_rate: float
    avg_page_recall: float | None
    avg_text_recall: float | None
    by_question_type: dict[str, dict[str, int]]
    results: list[EvalCaseResult]