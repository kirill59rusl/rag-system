from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Query

from src.core.config import RetrievalMode, settings


@dataclass(frozen=True)
class RetrievalParams:
    mode: RetrievalMode
    rrf_k: int
    vector_weight: float
    bm25_weight: float

    @classmethod
    def resolve(
        cls,
        mode: RetrievalMode | None = None,
        rrf_k: int | None = None,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> "RetrievalParams":
        """Не переданные значения берутся из settings (.env)."""
        return cls(
            mode=mode or settings.retrieval_mode,
            rrf_k=settings.rrf_k if rrf_k is None else rrf_k,
            vector_weight=settings.vector_weight if vector_weight is None else vector_weight,
            bm25_weight=settings.bm25_weight if bm25_weight is None else bm25_weight,
        )


def get_retrieval_params(
    retrieval_mode: RetrievalMode | None = None,
    rrf_k: Annotated[int | None, Query(gt=0)] = None,
    vector_weight: Annotated[float | None, Query(ge=0)] = None,
    bm25_weight: Annotated[float | None, Query(ge=0)] = None,
) -> RetrievalParams:
    return RetrievalParams.resolve(retrieval_mode, rrf_k, vector_weight, bm25_weight)


RetrievalParamsDep = Annotated[RetrievalParams, Depends(get_retrieval_params)]
