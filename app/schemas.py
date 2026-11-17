from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TextInRegion(str, Enum):
    annotate = "annotate"
    include = "include"
    exclude = "exclude"


class RegionAssociation(str, Enum):
    center = "center"
    iou = "iou"
    hybrid = "hybrid"


class ImageInfo(BaseModel):
    width: int
    height: int


class OrientationInfo(BaseModel):
    detected_angle: int
    score: float
    scores: List[float]
    corrected: bool
    correction: str
    low_confidence: bool
    threshold: float


class RegionItem(BaseModel):
    id: str
    type: str
    bbox: List[float] = Field(min_length=4, max_length=4)
    score: float


class RegionRef(BaseModel):
    id: str
    type: str
    bbox: List[float]
    relation: str


class TextItem(BaseModel):
    id: str
    text: str
    score: float
    bbox: List[float]
    polygon: List[List[float]]
    region: Optional[RegionRef] = None
    line: Optional[int] = None


class AnalyzeResponse(BaseModel):
    request_id: str
    image: ImageInfo
    orientation: OrientationInfo
    regions: List[RegionItem] = Field(default_factory=list)
    texts: List[TextItem] = Field(default_factory=list)
    layout_markdown: str = ""
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class LayoutResponse(BaseModel):
    request_id: str
    image: ImageInfo
    orientation: OrientationInfo
    regions: List[RegionItem] = Field(default_factory=list)
    timings_ms: Dict[str, float] = Field(default_factory=dict)


class OCRResponse(BaseModel):
    request_id: str
    image: ImageInfo
    orientation: OrientationInfo
    texts: List[TextItem] = Field(default_factory=list)
    timings_ms: Dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    ready: bool
    models: Dict[str, bool]
    version: str = "1.0.0"
    detail: Optional[Any] = None
