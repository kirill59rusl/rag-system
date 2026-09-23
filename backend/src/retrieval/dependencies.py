from src.core.config import settings
from src.retrieval.embedding import (
    EmbeddingModel,
    GigaChatEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
)


def get_embedding_model() -> EmbeddingModel:
    if settings.embedder_provider == "ollama":
        return OllamaEmbedding(
            model=settings.embedder,
            dimension=settings.embedding_space,
        )

    if settings.embedder_provider == "gigachat":
        return GigaChatEmbedding(
            credentials=settings.gigachat_credentials,
            model=settings.embedder,
            dimension=settings.embedding_space,
        )
    if settings.embedder_provider == "openai":           
        return OpenAIEmbedding(
            model=settings.embedder,
            dimension=settings.embedding_space,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
        
    raise ValueError(
        f"Неизвестный эмбеддер: {settings.embedder_provider}"
    )

