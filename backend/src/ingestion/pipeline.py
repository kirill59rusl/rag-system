from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.models import Chunk, Document, DocumentStatus
from src.ingestion.chunker import chunk_text
from src.ingestion.parser import load_pdf
from src.retrieval.embedding import OllamaEmbedding

embedding_model=OllamaEmbedding(
        model="embeddinggemma",
        dimension=settings.embedding_space
    )

async def process_document(
    document: Document,
    session: AsyncSession
):
    try:
        document.status=DocumentStatus.PROCESSING


        pages=load_pdf(document.storage_path)
        chunks=chunk_text(pages)

        embeddings=embedding_model.embed(texts=[x["text"] for x in chunks])
        
        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            session.add(
                instance=Chunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=chunk["text"],
                    page_number=chunk["page_number"],
                    embedding=embedding,
                )
            )
        
        document.status=DocumentStatus.INDEXED
        await session.commit()
        

    except Exception:
        document.status=DocumentStatus.FAILED
        await session.commit()
        raise
    

    

