from abc import ABC, abstractmethod

import ollama


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def __call__(self, texts):
        return self.embed(texts)

class OllamaEmbedding(EmbeddingModel):
    def __init__(self, model: str, dimension: int) -> None:
        self.model=model
        self._dimension=dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = ollama.embed(
            model=self.model,
            input=texts
        )
        return response.embeddings

    @property
    def dimension(self) -> int:
        return self._dimension