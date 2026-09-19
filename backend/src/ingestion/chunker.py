from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(pages: list[dict]) -> list[dict]:

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks=[]

    for page in pages:

        for chunk_index, chunk in enumerate(splitter.split_text(page["text"])):
            chunks.append({
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text": chunk
                })

    return chunks