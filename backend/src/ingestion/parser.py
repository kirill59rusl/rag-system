from pypdf import PdfReader


def load_pdf(path: str) -> list[dict]:
    reader=PdfReader(path)
    pages=[]

    for page_number, page in enumerate(reader.pages,start=1):
        text=page.extract_text() or ""

        if not text.strip():
            return

        pages.append({
            "page_number": page_number,
            "text": text
        })
    return pages