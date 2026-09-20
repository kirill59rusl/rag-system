from abc import ABC, abstractmethod

import ollama


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