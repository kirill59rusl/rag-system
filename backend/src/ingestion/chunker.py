import re
from bisect import bisect_right

from langchain_text_splitters import RecursiveCharacterTextSplitter

# раздел длиннее этого дорезается сплиттером (с перекрытием внутри раздела)
MAX_CHUNK_SIZE=1500
CHUNK_OVERLAP=200
# раздел короче этого (например, только заголовок главы) приклеивается к следующему
MIN_SECTION_SIZE=100

# "1. Введение", "2.3 Порядок работы", "Глава 4", "Раздел II. ..."
_NUMBERED_HEADING=re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2})*)\.?\s+(?P<title>[А-ЯЁA-Z][^\n]{1,120})$",
    re.MULTILINE,
)
_KEYWORD_HEADING=re.compile(
    r"^(?:Глава|Раздел|Часть|Chapter|Section|Part)\s+[\dIVXLC]+\.?[^\n]{0,120}$",
    re.MULTILINE | re.IGNORECASE,
)
# строки оглавления ("2.3 Порядок работы 14") и обычные предложения — не заголовки
_NOT_HEADING_END=re.compile(r"(\s\d+|[.,;:])$")


def _join_pages(pages: list[dict]) -> tuple[str, list[int], list[int]]:
    """Склеивает страницы в один текст и запоминает, где начинается каждая."""
    parts, starts, numbers=[], [], []
    offset=0
    for page in pages:
        starts.append(offset)
        numbers.append(page["page_number"])
        parts.append(page["text"])
        offset+=len(page["text"]) + 1
    return "\n".join(parts), starts, numbers


def _page_at(offset: int, starts: list[int], numbers: list[int]) -> int:
    return numbers[max(bisect_right(starts, offset) - 1, 0)]


def _headings_from_outline(text, outline, starts, numbers) -> list[dict]:
    """Ищет заголовки из закладок PDF в тексте на указанной странице."""
    headings=[]
    prev=0
    for item in outline:
        if item["page_number"] not in numbers:
            continue
        i=numbers.index(item["page_number"])
        page_start=starts[i]
        page_end=starts[i + 1] if i + 1 < len(starts) else len(text)

        words=[re.escape(w) for w in item["title"].split()]
        pattern=r"\s+".join(words)
        match=re.search(pattern, text[page_start:page_end], re.IGNORECASE) if words else None
        # не нашли заголовок в тексте — считаем, что раздел начинается с начала страницы
        offset=page_start + match.start() if match else page_start

        offset=max(offset, prev)
        prev=offset
        headings.append({"offset": offset, "title": item["title"], "level": item["level"]})
    return headings


def _headings_from_text(text) -> list[dict]:
    """Fallback, если закладок нет: ищет нумерованные заголовки регуляркой."""
    headings=[]
    for m in _NUMBERED_HEADING.finditer(text):
        line=m.group(0).strip()
        if _NOT_HEADING_END.search(line):
            continue
        level=m.group("num").count(".") + 1
        headings.append({"offset": m.start(), "title": line, "level": level})

    for m in _KEYWORD_HEADING.finditer(text):
        headings.append({"offset": m.start(), "title": m.group(0).strip(), "level": 1})

    headings.sort(key=lambda h: h["offset"])
    return headings


def _build_sections(text: str, headings: list[dict]) -> list[dict]:
    """Режет текст по заголовкам и строит для каждого раздела путь вида [Глава, Подраздел]."""
    sections=[]
    stack: list[dict]=[]

    if not headings or headings[0]["offset"] > 0:
        first=headings[0]["offset"] if headings else len(text)
        sections.append({"start": 0, "end": first, "title": None, "path": []})

    for i, h in enumerate(headings):
        end=headings[i + 1]["offset"] if i + 1 < len(headings) else len(text)
        while stack and stack[-1]["level"] >= h["level"]:
            stack.pop()
        stack.append(h)
        sections.append({
            "start": h["offset"],
            "end": end,
            "title": h["title"],
            "path": [s["title"] for s in stack],
        })

    # слишком короткие разделы (голый заголовок главы) приклеиваем к следующему
    merged=[]
    carry_start=None
    for i, s in enumerate(sections):
        if carry_start is not None:
            s={**s, "start": carry_start}
            carry_start=None
        if len(text[s["start"]:s["end"]].strip()) < MIN_SECTION_SIZE and i + 1 < len(sections):
            carry_start=s["start"]
            continue
        merged.append(s)
    return merged


def chunk_text(pages: list[dict], outline: list[dict] | None = None) -> list[dict]:
    """Чанкинг по разделам документа.

    Разделы берутся из закладок PDF (outline), а если их нет — из нумерованных
    заголовков в тексте. Длинный раздел дорезается на куски с перекрытием,
    но чанк никогда не захватывает соседний раздел.
    """
    if not pages:
        return []

    text, starts, numbers=_join_pages(pages)

    headings=_headings_from_outline(text, outline, starts, numbers) if outline else []
    if not headings:
        headings=_headings_from_text(text)

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",
        add_start_index=True,
    )

    chunks=[]
    for section in _build_sections(text, headings):
        section_text=text[section["start"]:section["end"]]
        if not section_text.strip():
            continue

        spans=[
            (p.metadata["start_index"], p.metadata["start_index"] + len(p.page_content))
            for p in splitter.create_documents([section_text])
        ]
        # голый заголовок, отрезанный сплиттером, приклеиваем к следующему куску
        merged_spans=[]
        carry=None
        for i, (s, e) in enumerate(spans):
            if carry is not None:
                s, carry=carry, None
            if len(section_text[s:e].strip()) < MIN_SECTION_SIZE and i + 1 < len(spans):
                carry=s
                continue
            merged_spans.append((s, e))

        for part_index, (s, e) in enumerate(merged_spans):
            body=section_text[s:e].strip()
            if not body:
                continue
            start=section["start"] + s
            end=section["start"] + e - 1

            header=" > ".join(section["path"])
            chunks.append({
                "text": body,
                # заголовок раздела в эмбеддинге помогает находить куски из середины раздела
                "embed_text": f"{header}\n{body}" if header else body,
                "page_number": _page_at(start, starts, numbers),
                "page_end": _page_at(end, starts, numbers),
                "section": section["title"],
                "section_path": section["path"],
                "section_part": part_index,
            })

    return chunks
