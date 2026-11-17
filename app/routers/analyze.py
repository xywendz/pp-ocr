from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..schemas import AnalyzeResponse, LayoutResponse, OCRResponse
from ..services.analysis_service import AnalysisService

router = APIRouter(prefix="/v1", tags=["analyze"])


def _split_types(raw: Optional[str]) -> Optional[list[str]]:
    if raw is None:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _service(request: Request) -> AnalysisService:
    return request.app.state.analysis


async def _read_upload(file: UploadFile) -> bytes:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="上传文件为空")
    return payload


def _handle_error(exc: Exception) -> HTTPException:
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"处理失败: {exc}")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    region_types: Optional[str] = Form(None),
    layout_threshold: Optional[float] = Form(None),
    ocr_threshold: Optional[float] = Form(None),
    include_text: bool = Form(True),
    text_in_region: str = Form("annotate"),
    region_association: str = Form("hybrid"),
    return_markdown: bool = Form(True),
):
    try:
        payload = await _read_upload(file)
        result = _service(request).analyze(
            payload,
            region_types=_split_types(region_types),
            layout_threshold=layout_threshold,
            ocr_threshold=ocr_threshold,
            include_text=include_text,
            text_in_region=text_in_region,
            region_association=region_association,
            return_markdown=return_markdown,
        )
        return result
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/layout", response_model=LayoutResponse)
async def layout(
    request: Request,
    file: UploadFile = File(...),
    region_types: Optional[str] = Form(None),
    threshold: Optional[float] = Form(None),
):
    try:
        payload = await _read_upload(file)
        return _service(request).layout_only(
            payload,
            region_types=_split_types(region_types),
            threshold=threshold,
        )
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/ocr", response_model=OCRResponse)
async def ocr(
    request: Request,
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
):
    try:
        payload = await _read_upload(file)
        return _service(request).ocr_only(payload, threshold=threshold)
    except Exception as exc:
        raise _handle_error(exc) from exc
