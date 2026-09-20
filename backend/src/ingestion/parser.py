import re
import unicodedata

from pypdf import PdfReader

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_PUA_CHARS = re.compile(r"[\ue000-\uf8ff]")

_REPLACEMENTS = {
    "\uf03d": "=", "\uf03e": ">", "\uf03c": "<",
    "\uf065": "ε", "\uf06a": "θ", "\uf074": "τ",
    "\uf06a": "θ", "\uf0b9": "≠", "\uf0d7": "×",
    "\uf0b4": "×", "\uf02d": "-", "\uf020": " ",
}

_HYPHEN_BREAK = re.compile(r"(\w)[-\u2010-\u2015]\s*\n\s*(\w)", re.UNICODE)

_WS = re.compile(r"\s+")

_JUNK_INLINE = re.compile(r"(?<=[А-Яа-яA-Za-z])[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff](?=[А-Яа-яA-Za-z])")

_PUNCT_REPEAT = re.compile(r"([!?.,;:—-])\1{2,}")

_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}")

def clean_text(text:str, keep_newlines: bool=False) -> str:
    if not text:
        return ""

    for k, v in _REPLACEMENTS.items():
        text = text.replace(k, v)

    text = _CONTROL_CHARS.sub("", text)

    text = _PUA_CHARS.sub("", text)

    text = unicodedata.normalize("NFKC", text)

    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    text = _JUNK_INLINE.sub("", text)

    text = _EMPTY_BRACKETS.sub("", text)

    if keep_newlines:
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)   
        text = re.sub(r" *\n *", "\n", text) 
    else:
        text = _WS.sub(" ", text)

    text = _PUNCT_REPEAT.sub(r"\1", text)

    text = text.strip()

    return text
def load_pdf(path: str) -> list[dict]:
    reader=PdfReader(path)
    pages=[]

    for page_number, page in enumerate(reader.pages,start=1):
        text=page.extract_text() or ""
        text=clean_text(text)

        if not text.strip():
            return

        pages.append({
            "page_number": page_number,
            "text": text
        })
    return pages