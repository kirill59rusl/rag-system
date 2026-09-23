from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import service
from src.db.database import get_session
from src.ingestion.pipeline import process_document
from src.retrieval.dependencies import get_embedding_model
from src.retrieval.embedding import EmbeddingModel

documents = APIRouter(prefix="/documents", tags=["documents"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
EmbeddingDep = Annotated[EmbeddingModel, Depends(get_embedding_model)]
Upload = Annotated[UploadFile, File()]

@documents.post("/upload")
async def upload(
    file: Upload,
    session: SessionDep,
    embedding_model: EmbeddingDep):
    document = await service.create_document(session, file)

    await process_document(
        document=document,
        session=session,
        embedding_model=embedding_model
    )

    return document