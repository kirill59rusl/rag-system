import uuid
from pathlib import Path

import aiofiles
from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.models import Chunk, Document, DocumentStatus


async def save_uploaded_file(file: UploadFile, doc_id: uuid.UUID) -> str:
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(400, f"Неподдерживаемый тип: {file.content_type}")
    if file.filename is None:
        raise HTTPException(400, "Имя не может быть пустым")
    
    doc_dir = settings.upload_dir / str(doc_id)
    doc_dir.mkdir(parents=True, exist_ok=True)

    storage_path = doc_dir / file.filename

    size = 0
    async with aiofiles.open(storage_path, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_size:
                await out_file.close()
                storage_path.unlink(missing_ok=True)
                raise HTTPException(400, "Файл слишком большой")
            await out_file.write(chunk)

    return str(storage_path)


async def create_document(
    session: AsyncSession,
    file: UploadFile,
) -> Document:
    doc_id = uuid.uuid4()
    storage_path = await save_uploaded_file(file, doc_id)

    document = Document(
        id=doc_id,
        filename=file.filename,
        content_type=file.content_type,
        storage_path=storage_path,
        status=DocumentStatus.PENDING,
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document

async def find_similar(
    session: AsyncSession,
    query_embedding: list[float],
    limit: int
) -> list[dict]:
    distance=Chunk.embedding.cosine_distance(query_embedding)
    
    chunks =(
        select(Chunk, distance)
        .order_by(distance.asc())
        .limit(limit)
    )
    
    rows = (await session.execute(chunks)).all()
    result = [{**chunk.as_dict(), "distance": float(dist)}
                for chunk,dist in rows]

    return result
