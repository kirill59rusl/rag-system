from src.generator.llm import LLM

_PROMPT = """Translate the user's question about PostgreSQL into English as a search query.
Rules:
- Keep identifiers, parameters, SQL keywords and version numbers exactly as written (max_wal_size, pg_stat_statements, UPDATE, 18).
- Use the official PostgreSQL documentation terminology.
- Output only the translated query, no quotes, no explanations.

Question: {query}"""


async def translate_query(llm: LLM, query: str) -> str:
    if query.isascii():
        return query
    result = (await llm.generate(_PROMPT.format(query=query))).strip()
    return result or query
