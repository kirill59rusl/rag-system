from typing import Annotated

from fastapi import APIRouter, Depends

from src.agentic.agent import OpenAIChat
from src.agentic.pipeline import run_agent
from src.agentic.tools import ToolContext
from src.core.config import settings
from src.generator.generate import RerankerDep, SessionDep
from src.retrieval.params import RetrievalParams
from src.retrieval.query import EmbeddingDep

agent=APIRouter(prefix="/agent", tags=["agent"])

def get_chat() -> OpenAIChat:
    return OpenAIChat(
        model=settings.llm,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

ChatDep = Annotated[OpenAIChat, Depends(get_chat)]


@agent.post("/ask")
async def ask(query: str, session: SessionDep, chat: ChatDep, reranker: RerankerDep,
              embedding_model: EmbeddingDep, debug: bool = False):
    ctx = ToolContext(
        session=session,
        embedding_model=embedding_model,
        reranker=reranker,
        llm=chat, 
        params=RetrievalParams.resolve(),
    )
    return await run_agent(query, ctx, chat, debug=debug)