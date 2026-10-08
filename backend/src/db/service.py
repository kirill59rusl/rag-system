import re
import uuid
from pathlib import Path
from uuid import UUID

import aiofiles
from fastapi import HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.models import Chunk, Document, DocumentStatus

_VERSION_IN_FILENAME = re.compile(r"postgresql-(\d+)", re.IGNORECASE)


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


def version_from_filename(filename: str | None) -> str | None:
    match = _VERSION_IN_FILENAME.search(filename or "")
    return match.group(1) if match else None


async def create_document(
    session: AsyncSession,
    file: UploadFile,
    version: str | None = None,
) -> Document:
    doc_id = uuid.uuid4()
    storage_path = await save_uploaded_file(file, doc_id)

    document = Document(
        id=doc_id,
        filename=file.filename,
        content_type=file.content_type,
        storage_path=storage_path,
        status=DocumentStatus.PENDING,
        version=version or version_from_filename(file.filename),
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document

async def find_similar(
    session: AsyncSession,
    query_embedding: list[float],
    limit: int,
    version: str | None = None,
) -> list[dict]:
    distance=Chunk.embedding.cosine_distance(query_embedding)
    
    chunks =(
        select(Chunk, Document.version, distance)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance.asc())
        .limit(limit)
    )
    if version is not None:
        chunks = chunks.where(Document.version == version)
    
    rows = (await session.execute(chunks)).all()
    result = [{**chunk.as_dict(), "version": version, "distance": float(dist)}
                for chunk,version, dist in rows]

    return result

async def find_lexical(
    session: AsyncSession,
    query: str,
    limit: int,
    version: str | None = None,
) -> list[dict]:
    # ||| — совпадение хотя бы одного токена запроса, токенизация как в BM25-индексе
    score = func.pdb.score(Chunk.id)

    chunks = (
        select(Chunk, Document.version, score)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.content.op("|||")(query))
        .order_by(score.desc(), Chunk.id)
        .limit(limit)
    )
    if version is not None:
        chunks = chunks.where(Document.version == version)

    rows = (await session.execute(chunks)).all()
    return [{**chunk.as_dict(), "version": version, "bm25_score": float(s)}
            for chunk, version, s in rows]

async def get_known_versions(session: AsyncSession) -> set[str]:
    rows = await session.scalars(
        select(Document.version).where(Document.version.is_not(None)).distinct()
    )
    return set(rows)

async def get_chunk(
    session: AsyncSession,
    doc_id: uuid.UUID,
    chunk_idx: int,
) -> Chunk | None:
    stmt = select(Chunk).where(
        Chunk.document_id == doc_id,
        Chunk.chunk_index == chunk_idx,
    )

    chunk=await session.scalar(stmt)
    return chunk

async def get_chunks(
    session: AsyncSession,
    document_ids: list[UUID] | None = None,
    limit: int | None = None,
) -> list[Chunk]:
    
    stmt = select(Chunk).order_by(Chunk.document_id, Chunk.chunk_index)

    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))

    if limit:
        stmt = stmt.limit(limit)

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_chunks_as_dicts(
    session: AsyncSession,
    document_ids: list[UUID] | None = None,
    limit: int | None = None,
) -> list[dict]:

    chunks = await get_chunks(session, document_ids=document_ids, limit=limit)
    return [chunk.as_dict() for chunk in chunks]