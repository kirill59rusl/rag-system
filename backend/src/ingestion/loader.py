from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import service
from src.db.database import get_session
from src.ingestion.pipeline import process_document

documents = APIRouter(prefix="/documents", tags=["documents"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
Upload = Annotated[UploadFile, File()]

@documents.post("/upload")
async def upload(
    file: Upload,
    session: SessionDep):
    document = await service.create_document(session, file)

    await process_document(
        document=document,
        session=session
    )

    return document