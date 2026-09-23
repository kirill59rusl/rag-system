from abc import ABC, abstractmethod

import ollama
from gigachat import GigaChat
from openai import AsyncOpenAI


class EmbeddingModel(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

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
        host: str | None = None
    ) -> None:
        self.model=model
        self._dimension=dimension
        self.client = ollama.AsyncClient(host=host)

    async def embed(self, texts: list[str]) -> list[list[float]]:
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
    ) -> None:
        self.model = model
        self._dimension = dimension

        self.client = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
        )

    async def embed(
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
    ) -> None:
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self._dimension = dimension

    async def embed(
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