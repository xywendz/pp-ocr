from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    return request.app.state.models.status()


@router.get("/ready", response_model=HealthResponse)
def ready(request: Request):
    status = request.app.state.models.status()
    if not status["ready"]:
        raise HTTPException(status_code=503, detail=status)
    return status
