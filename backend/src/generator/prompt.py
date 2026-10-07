SYSTEM_PROMPT = '''Ты — профессиональный помощник-консультант. Отвечай на вопрос, используя ИСКЛЮЧИТЕЛЬНО предоставленный контекст.

Правила:
1. Используй только факты из раздела «Контекст». Не додумывай и не привлекай внешние знания.
2. Если в контексте нет ответа, в поле answer напиши: «В предоставленных документах нет информации по вашему вопросу», а claims оставь пустым.
3. Верни JSON с полями:
   - answer: связный ответ на языке вопроса, без номеров источников в тексте;
   - claims: список атомарных утверждений, из которых состоит ответ. Каждое утверждение — одно проверяемое факт-предложение, в поле sources — номера источников (числа из [Источник N]), которые его подтверждают.
4. Каждое утверждение из answer должно быть представлено в claims. Не ссылайся на источники, которые его не подтверждают.
5. Верни только компактный JSON в одну строку, без переводов строк, табуляций и лишних пробелов вне строковых значений.'''
CONTEXT='''[Контекст / Context]
        {similar}'''

QUESTION='''[Вопрос пользователя / User Query]
        {query}'''

def format_location(d: dict) -> str:
    pages = d["page_number"]
    if d.get("page_end") and d["page_end"] != d["page_number"]:
        pages = f"{d['page_number']}–{d['page_end']}"
    location = f"{d['doc_id']}, стр. {pages}"
    if d.get("section_path"):
        location += f", раздел: {' > '.join(d['section_path'])}"
    return location

def format_context(docs: list[dict]) -> str:
    if not docs:
        return "(в базе не найдено релевантных документов)"
    return "\n\n".join(
        f"[Источник {i}] ({format_location(d)})\n{d['content']}"
        for i, d in enumerate(docs, 1)
    )

def build_prompt(similar: list[dict], query: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{CONTEXT.format(similar=format_context(similar))}\n"
        f"{QUESTION.format(query=query)}"
    )

