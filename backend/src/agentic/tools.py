from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from src.agentic.schemas import SearchDocsArgs, Tool
from src.db.service import get_chunk
from src.generator.llm import LLM
from src.generator.prompt import format_location
from src.retrieval.embedding import EmbeddingModel
from src.retrieval.params import RetrievalParams
from src.retrieval.query import get_similar
from src.retrieval.reranking.reranker import Reranker


@dataclass
class ToolContext:
    session: AsyncSession
    embedding_model: EmbeddingModel
    reranker: Reranker | None
    llm: LLM
    params: RetrievalParams
    limit: int = 10
    reranker_limit: int = 5
    sources: list[dict] = field(default_factory=list) 

    def add(self, docs: list[dict]) -> list[tuple[int, dict]]:
        known = {d["chunk_id"]: i for i, d in enumerate(self.sources, 1)}
        out = []
        for d in docs:
            if d["chunk_id"] not in known:
                self.sources.append(d)
                known[d["chunk_id"]] = len(self.sources)
            out.append((known[d["chunk_id"]], d))
        return out


def make_tool(name: str, description: str, args: type[BaseModel]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": args.model_json_schema(),
            "strict": True,
        },
    }
# верхний предел раундов повтора упавших вопросов в run_eval

async def run_tool(ctx: ToolContext, call: Tool) -> str:
    if call.name not in TOOLS:
        return f"Неизвестный инструмент: {call.name}"
    if call.name=="search_docs":
        try:
            args=SearchDocsArgs.model_validate(call.arguments)
        except ValidationError as e:
            return f"Некорректные аргументы: {e}"
        res=search_docs(ctx, args)
    return await res

def _format(numbered: list[tuple[int, dict]]) -> str:
    if not numbered:
        return "Ничего не найдено. Перефразируй запрос или убери фильтр версии."
    return "\n\n".join(f"[{n}] ({format_location(d)})\n{d['content']}" for n, d in numbered)

async def search_docs(ctx: ToolContext, args: SearchDocsArgs) -> str:
    docs = await get_similar(
        args.query, ctx.session, ctx.embedding_model, ctx.limit, ctx.reranker_limit,
        ctx.reranker, ctx.params, ctx.llm,
        versions=[args.version] if args.version else [],
    )
    return _format(ctx.add(docs))

TOOLS=set(['search_docs'])

SEARCH_DOCS=make_tool(
    "search_docs",
    '''Ищи в документации PostgreSQL. Если вопрос про конкретную версию, то ищи в ней
    Если версия не указана или вопрос про сравнение, то ищи в обоих версиях.''',
    SearchDocsArgs
)