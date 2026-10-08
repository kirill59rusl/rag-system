def rrf(lists: list[list[dict]], weights: list[float], k: int) -> list[dict]:
    """Reciprocal Rank Fusion: score = Σ w_i / (k + rank_i), rank с 1.

    Чанк, найденный несколькими списками, берётся из первого, где он встретился,
    и дополняется полями из остальных (distance + bm25_score).
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for docs_list, weight in zip(lists, weights, strict=True):
        for rank, doc in enumerate(docs_list, 1):
            key = doc["chunk_id"]
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            docs[key] = {**doc, **docs.get(key, {})}

    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    return [{**docs[key], "rrf_score": scores[key]} for key in ranked]
