from fastapi import APIRouter, UploadFile, File, Depends
from pydantic import BaseModel
from src.db.database import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
from src.schemas.schemas import Document
from src.db import service

documents = APIRouter(prefix="/documents", tags=["documents"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
Upload = Annotated[UploadFile, File()]

@documents.post("/upload")
async def upload(
    file: Upload,
    session: SessionDep):
    document = await service.create_document(session, file)
    return document