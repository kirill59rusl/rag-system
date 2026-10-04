import asyncio
import logging
from abc import ABC, abstractmethod

import httpx


logger = logging.getLogger(__name__)

class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, docs: list[dict], top_k: int) -> list[dict]: ...


class RouterAIReranker(Reranker):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://routerai.ru/api/v1",
        path: str = "/rerank",
        timeout: float = 30.0,
        fallback_on_error: bool = True,
    ) -> None:
        self.model = model
        self.url = base_url.rstrip("/") + path
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout
        self.fallback_on_error = fallback_on_error

    async def rerank(self, query: str, docs: list[dict], top_k: int) -> list[dict]:
        if not docs:
            return docs

        payload = {
            "model": self.model,
            "query": query,
            "documents": [d["content"] for d in docs],
            "top_n": min(top_k, len(docs)),
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.url, json=payload, headers=self.headers)
                resp.raise_for_status()
            results = resp.json()["results"]
        except (httpx.HTTPError, KeyError, ValueError):
            if not self.fallback_on_error:
                raise

            logger.exception("RouterAI rerank failed, falling back to vector order")
            return docs[:top_k]

        ranked = sorted(
            results,
            key=lambda r: r.get("relevance_score", r.get("score", 0.0)),
            reverse=True,
        )[:top_k]
        return [
            {
                **docs[r["index"]],
                "rerank_score": float(r.get("relevance_score", r.get("score", 0.0))),
            }
            for r in ranked
        ]


