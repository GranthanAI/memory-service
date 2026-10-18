"""
Public, user-facing memory routes (gateway: /api/v1/memory/*).

Identity comes only from the verified access token (`sub`); there is no user id in any
path, query or body. Every read and write is confined to that user's partition.
"""

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from app.core.security import get_current_user_id
from app.services.user_memory_service import STATEMENT_MAX, FactNotFoundError, UserMemoryService

router = APIRouter(prefix="/memory", tags=["Memory (user)"])


def _service(request: Request) -> UserMemoryService:
    container = getattr(request.app.state, "container", None)
    service = getattr(container, "user_memory_service", None)
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Memory service is starting up.")
    return service


class MemorySettings(BaseModel):
    memory_enabled: bool


class MemoryFact(BaseModel):
    id: str
    category: str
    statement: str
    importance: float
    created_at: Optional[str] = None
    conversation_id: Optional[str] = None


class MemoryFactList(BaseModel):
    items: List[MemoryFact]
    memory_enabled: bool


class FactUpdate(BaseModel):
    category: str = Field(..., min_length=1, max_length=32)
    statement: str = Field(..., min_length=1, max_length=STATEMENT_MAX)

    @field_validator("statement")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("A memory cannot be empty.")
        return v.strip()


def _category(value: str) -> str:
    value = value.strip().lower()
    if not value or len(value) > 32:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid category.")
    return value


@router.get("/settings", response_model=MemorySettings)
async def get_settings(request: Request, user_id: str = Depends(get_current_user_id)):
    return _service(request).get_settings(user_id)


@router.put("/settings", response_model=MemorySettings)
async def put_settings(body: MemorySettings, request: Request, user_id: str = Depends(get_current_user_id)):
    """Turning memory off takes effect for the next message: saved memories stop being
    used in answers and no new ones are learned. Existing memories are kept unless you
    delete them."""
    return _service(request).set_settings(user_id, body.memory_enabled)


@router.get("/facts", response_model=MemoryFactList)
async def list_facts(request: Request, user_id: str = Depends(get_current_user_id)):
    service = _service(request)
    try:
        items = service.list_facts(user_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Memories are unavailable right now.")
    return {"items": items, "memory_enabled": service.get_settings(user_id)["memory_enabled"]}


@router.patch("/facts/{fact_id}", response_model=MemoryFact)
async def update_fact(fact_id: uuid.UUID, body: FactUpdate, request: Request,
                      user_id: str = Depends(get_current_user_id)):
    try:
        return await _service(request).update_fact(user_id, fact_id, _category(body.category), body.statement)
    except FactNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.")


@router.delete("/facts/{fact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_fact(fact_id: uuid.UUID, category: str, request: Request,
                      user_id: str = Depends(get_current_user_id)):
    try:
        _service(request).delete_fact(user_id, fact_id, _category(category))
    except FactNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.")


@router.delete("/facts")
async def clear_facts(request: Request, user_id: str = Depends(get_current_user_id)):
    return {"deleted": _service(request).clear_all(user_id)}
