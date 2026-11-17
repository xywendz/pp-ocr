from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..config import Settings
from .layout_service import LayoutService
from .ocr_service import OCRService
from .orientation_service import OrientationService

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.orientation: Optional[OrientationService] = None
        self.layout: Optional[LayoutService] = None
        self.ocr: Optional[OCRService] = None
        self.error: Optional[str] = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            logger.info("正在加载文档方向分类 ONNX 模型")
            self.orientation = OrientationService(self.settings)
            self.orientation.warmup()

            logger.info("正在加载 PP-DocLayout-M ONNX 模型")
            self.layout = LayoutService(self.settings)
            self.layout.warmup()

            logger.info("正在加载 PP-OCRv6 Medium ONNX 模型")
            self.ocr = OCRService(self.settings)
            self.ocr.warmup()

            self._loaded = True
            self.error = None
            logger.info("模型加载完成")
        except Exception as exc:
            self.error = str(exc)
            logger.exception("模型加载失败: %s", exc)
            if self.settings.strict_model_load:
                raise

    @property
    def ready(self) -> bool:
        return (
            self._loaded
            and self.orientation is not None
            and self.layout is not None
            and self.ocr is not None
        )

    def require(self) -> None:
        if not self.ready:
            detail = self.error or "模型尚未加载"
            raise RuntimeError(f"模型不可用: {detail}")

    def status(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.ready else "loading" if self.error is None else "error",
            "ready": self.ready,
            "models": {
                "orientation": self.orientation is not None,
                "layout": self.layout is not None,
                "ocr": self.ocr is not None,
            },
            "detail": self.error,
        }
