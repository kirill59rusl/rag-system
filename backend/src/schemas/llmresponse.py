from pydantic import BaseModel, Field


class Claim(BaseModel):
    text: str
    sources: list[int] = Field(default_factory=list)


class LLMResponse(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)