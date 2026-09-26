from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_session
from src.db.service import get_chunks_as_dicts

chunks = APIRouter(prefix="/chunks", tags=["documents"])


@chunks.get("", response_model=list[dict])
async def list_chunks(
    document_ids: list[UUID] | None = Query(default=None),
    limit: int | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    return await get_chunks_as_dicts(
        session,
        document_ids=document_ids,
        limit=limit,
    )