from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.db.models.document import DocumentStatus


class Document(BaseModel):

    model_config=ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    doc_metadata: dict
    status: DocumentStatus
    created_at: datetime


class Claim(BaseModel):
    text: str
    sources: list[int] = Field(default_factory=list)


class LLMResponse(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)