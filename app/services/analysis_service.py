from __future__ import annotations

import copy
import io
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..config import Settings
from ..core.geometry import transform_bbox_back, transform_polygon_back
from .merge_service import MergeService
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self, settings: Settings, manager: ModelManager, merger: MergeService):
        self.settings = settings
        self.manager = manager
        self.merger = merger

    def decode_image(self, payload: bytes) -> np.ndarray:
        max_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        if len(payload) > max_bytes:
            raise ValueError(f"图片超过 {self.settings.max_upload_size_mb} MB 限制")
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image = ImageOps.exif_transpose(image)
                rgb = np.asarray(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise ValueError(f"无法解码图片: {exc}") from exc

    def _prepare_image(
        self,
        payload: bytes,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, float]]:
        original = self.decode_image(payload)
        started = time.perf_counter()
        corrected, orientation = self.manager.orientation.correct(original)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        return original, corrected, orientation, {"orientation_ms": elapsed}

    @staticmethod
    def _applied_angle(orientation: Dict[str, Any]) -> int:
        return int(orientation["detected_angle"]) if orientation.get("corrected") else 0

    def _restore_regions(
        self,
        regions: List[Dict[str, Any]],
        angle: int,
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        for region in regions:
            region["bbox"] = transform_bbox_back(region["bbox"], angle, width, height)
        return regions

    def _restore_texts(
        self,
        texts: List[Dict[str, Any]],
        angle: int,
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        for text in texts:
            text["bbox"] = transform_bbox_back(text["bbox"], angle, width, height)
            text["polygon"] = [
                [float(point[0]), float(point[1])]
                for point in transform_polygon_back(text["polygon"], angle, width, height)
            ]
            ref = text.get("region")
            if ref:
                ref["bbox"] = transform_bbox_back(ref["bbox"], angle, width, height)
        return texts

    def analyze(
        self,
        payload: bytes,
        region_types: Optional[Sequence[str]] = None,
        layout_threshold: Optional[float] = None,
        ocr_threshold: Optional[float] = None,
        include_text: bool = True,
        text_in_region: str = "annotate",
        region_association: str = "hybrid",
        return_markdown: bool = True,
    ) -> Dict[str, Any]:
        self.manager.require()
        original, corrected, orientation, timings = self._prepare_image(payload)
        original_h, original_w = original.shape[:2]
        angle = self._applied_angle(orientation)

        allowed = list(region_types) if region_types is not None else self.settings.region_type_list
        layout_th = self.settings.default_layout_threshold if layout_threshold is None else layout_threshold
        ocr_th = self.settings.default_ocr_threshold if ocr_threshold is None else ocr_threshold

        warnings: List[str] = []
        if orientation.get("low_confidence"):
            warnings.append("方向分类置信度较低，未执行旋转矫正")

        started = time.perf_counter()
        layout_detections = self.manager.layout.detect(corrected, layout_th, allowed)
        timings["layout_ms"] = round((time.perf_counter() - started) * 1000, 2)
        regions_corrected = self.merger.build_regions(layout_detections)
        regions = self._restore_regions(copy.deepcopy(regions_corrected), angle, original_w, original_h)
        if not regions:
            warnings.append("未检测到图片或图形区域")

        texts: List[Dict[str, Any]] = []
        if include_text:
            started = time.perf_counter()
            ocr_detections = self.manager.ocr.recognize(corrected, ocr_th)
            timings["ocr_ms"] = round((time.perf_counter() - started) * 1000, 2)
            texts = self._restore_texts(
                self.merger.build_texts(
                    ocr_detections,
                    regions_corrected,
                    association_mode=region_association,
                    text_in_region=text_in_region,
                ),
                angle,
                original_w,
                original_h,
            )
            if not texts:
                warnings.append("未识别到文字")

        markdown = ""
        if return_markdown:
            markdown = self.merger.build_markdown(regions, texts, include_texts=include_text)

        return {
            "request_id": self.merger.new_request_id(),
            "image": {"width": original_w, "height": original_h},
            "orientation": orientation,
            "regions": regions,
            "texts": texts,
            "layout_markdown": markdown,
            "timings_ms": timings,
            "warnings": warnings,
        }

    def layout_only(
        self,
        payload: bytes,
        region_types: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.manager.require()
        original, corrected, orientation, timings = self._prepare_image(payload)
        original_h, original_w = original.shape[:2]
        angle = self._applied_angle(orientation)

        allowed = list(region_types) if region_types is not None else self.settings.region_type_list
        th = self.settings.default_layout_threshold if threshold is None else threshold

        started = time.perf_counter()
        detections = self.manager.layout.detect(corrected, th, allowed)
        timings["layout_ms"] = round((time.perf_counter() - started) * 1000, 2)
        regions = self._restore_regions(
            self.merger.build_regions(detections), angle, original_w, original_h
        )
        return {
            "request_id": self.merger.new_request_id(),
            "image": {"width": original_w, "height": original_h},
            "orientation": orientation,
            "regions": regions,
            "timings_ms": timings,
        }

    def ocr_only(self, payload: bytes, threshold: Optional[float] = None) -> Dict[str, Any]:
        self.manager.require()
        original, corrected, orientation, timings = self._prepare_image(payload)
        original_h, original_w = original.shape[:2]
        angle = self._applied_angle(orientation)
        th = self.settings.default_ocr_threshold if threshold is None else threshold

        started = time.perf_counter()
        detections = self.manager.ocr.recognize(corrected, th)
        timings["ocr_ms"] = round((time.perf_counter() - started) * 1000, 2)
        texts = self._restore_texts(
            self.merger.build_texts(detections, [], text_in_region="include"),
            angle,
            original_w,
            original_h,
        )
        return {
            "request_id": self.merger.new_request_id(),
            "image": {"width": original_w, "height": original_h},
            "orientation": orientation,
            "texts": texts,
            "timings_ms": timings,
        }
