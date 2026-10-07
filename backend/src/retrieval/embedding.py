from abc import ABC, abstractmethod

import ollama
from gigachat import GigaChat
from openai import AsyncOpenAI


class EmbeddingModel(ABC):
    # Провайдеры ограничивают число текстов в одном запросе (у Voyage — 1000),
    # поэтому embed() режет вход на пачки, а запрос к API делает _embed_batch().
    batch_size: int = 128

    @abstractmethod
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            embeddings.extend(await self._embed_batch(batch))
        return embeddings

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    def __call__(self, texts):
        return self.embed(texts)

class OllamaEmbedding(EmbeddingModel):
    def __init__(
        self, model: str, dimension: int,
        host: str | None = None,
        batch_size: int = EmbeddingModel.batch_size,
    ) -> None:
        self.model=model
        self.batch_size=batch_size
        self._dimension=dimension
        self.client = ollama.AsyncClient(host=host)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embed(
            model=self.model,
            input=texts
        )
        return response.embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


class GigaChatEmbedding(EmbeddingModel):
    def __init__(
        self,
        credentials: str,
        model: str,
        dimension: int,
        batch_size: int = EmbeddingModel.batch_size,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._dimension = dimension

        self.client = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
        )

    async def _embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        response = await self.client.embeddings(
            texts,
            model=self.model,
        )

        return response.data

    @property
    def dimension(self) -> int:
        return self._dimension

class OpenAIEmbedding(EmbeddingModel):
    def __init__(
        self,
        model: str,
        dimension: int,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = EmbeddingModel.batch_size,
    ) -> None:
        self.batch_size = batch_size
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self._dimension = dimension

    async def _embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension