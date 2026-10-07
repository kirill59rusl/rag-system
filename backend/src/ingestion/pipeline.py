from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Chunk, Document, DocumentStatus
from src.ingestion.chunker import chunk_text
from src.ingestion.parser import load_outline, load_pdf
from src.retrieval.embedding import EmbeddingModel


async def process_document(
    document: Document,
    session: AsyncSession,
    embedding_model: EmbeddingModel
):
    try:
        document.status=DocumentStatus.PROCESSING


        pages=load_pdf(document.storage_path)
        outline=load_outline(document.storage_path)
        chunks=chunk_text(pages, outline)

        embeddings=await embedding_model.embed(texts=[x["embed_text"] for x in chunks])
        
        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            session.add(
                instance=Chunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=chunk["text"],
                    page_number=chunk["page_number"],
                    metadata_={
                        "section": chunk["section"],
                        "section_path": chunk["section_path"],
                        "section_part": chunk["section_part"],
                        "page_end": chunk["page_end"],
                    },
                    embedding=embedding,
                )
            )
        
        document.status=DocumentStatus.INDEXED
        await session.commit()
        

    except Exception:
        document.status=DocumentStatus.FAILED
        await session.commit()
        raise
    

    

