from functools import lru_cache

from src.core.config import settings
from src.retrieval.embedding import (
    EmbeddingModel,
    GigaChatEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
)
from src.retrieval.reranking.reranker import Reranker, RouterAIReranker


def get_embedding_model() -> EmbeddingModel:
    if settings.embedder_provider == "ollama":
        return OllamaEmbedding(
            model=settings.embedder,
            dimension=settings.embedding_space,
            batch_size=settings.embed_batch_size,
        )

    if settings.embedder_provider == "gigachat":
        return GigaChatEmbedding(
            credentials=settings.gigachat_credentials,
            model=settings.embedder,
            dimension=settings.embedding_space,
            batch_size=settings.embed_batch_size,
        )
    if settings.embedder_provider == "openai":           
        return OpenAIEmbedding(
            model=settings.embedder,
            dimension=settings.embedding_space,
            batch_size=settings.embed_batch_size,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
        
    raise ValueError(
        f"Неизвестный эмбеддер: {settings.embedder_provider}"
    )


@lru_cache
def _routerai(name: str) -> RouterAIReranker:
    if not settings.openai_api_key:
        raise ValueError("ROUTERAI_API_KEY не задан")
    return RouterAIReranker(
        model=name,
        api_key=settings.openai_api_key,
    )


def get_reranker() -> Reranker | None:
    provider = settings.reranker_provider
    if provider is None:
        return None
    if not settings.reranker:
        raise ValueError("RERANKER не задан")
    if provider == "routerai":
        return _routerai(settings.reranker)
    raise ValueError(f"Неизвестный реранкер: {provider}")