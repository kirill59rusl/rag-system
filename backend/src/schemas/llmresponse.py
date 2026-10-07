from pydantic import BaseModel, Field


class Claim(BaseModel):
    text: str
    sources: list[int] = Field(default_factory=list)


class LLMResponse(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)


class Source(BaseModel):
    id: int
    chunk_id: str
    doc_id: str
    chunk_index: int
    page_number: int | None
    page_end: int | None = None
    section: str | None = None
    section_path: list[str] = Field(default_factory=list)
    content: str | None = None
    distance: float | None = None
    used: bool = False


class GenerateResponse(BaseModel):
    answer: str
    claims: list[Claim]
    sources: list[Source]