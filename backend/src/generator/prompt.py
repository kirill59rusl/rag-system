SYSTEM_PROMPT='''[Системный промпт / System Prompt]
        Ты — профессиональный помощник-консультант. Твоя задача — отвечать на вопросы пользователя, используя ИСКЛЮЧИТЕЛЬНО предоставленный контекст.

        Правила:
        1. Используй только факты из раздела «Контекст». Не додумывай и не привлекай внешние знания.
        2. Если в контексте нет ответа на вопрос, прямо скажи: «В предоставленных документах нет информации по вашему вопросу».
        3. Обязательно ссылайся на номера источников (например, [Источник 1]), откуда ты взял информацию.
        4. Отвечай четко, структурировано и на языке вопроса.'''  # noqa: E501

CONTEXT='''[Контекст / Context]
        {similar}'''

QUESTION='''[Вопрос пользователя / User Query]
        {query}'''

def format_context(docs: list[dict]) -> str:
    if not docs:
        return "(в базе не найдено релевантных документов)"
    return "\n\n".join(
        f"[Источник {i}] ({d['doc_id']}, стр. {d['page_number']})\n{d['content']}"
        for i, d in enumerate(docs, 1)
    )

def build_prompt(similar: list[dict], query: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{CONTEXT.format(similar=format_context(similar))}\n"
        f"{QUESTION.format(query=query)}"
    )

