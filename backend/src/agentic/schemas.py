from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.llmresponse import GenerateResponse


class Tool(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

class AgentResponse(BaseModel):
    content: str = ""
    tool_calls: list[Tool] = Field(default_factory=list)
    message: dict[str, Any]

class AgentAnswer(GenerateResponse):
    messages: list[dict[str, Any]] | None = None
    
class SearchDocsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid") 

    query: str = Field(
        description="Запрос на русском, звучит как текст по документации "
                    "например, 'Какие параметры RADIUS-аутентификации в PostgreSQL 18 обязательны и какой длины рекомендуется общий секрет?.'",
    )

    version: Literal["18", "19"] | None = Field(
        description="Требование искать по данной версии "
                    "Используй null, чтобы искать по всем версиям",
    )
