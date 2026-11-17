from __future__ import annotations

from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from ..config import Settings
from ..core.geometry import clamp_bbox, to_bbox


class OCRService:
    """PP-OCRv6 Medium inference via RapidOCR with ONNX Runtime engine."""

    def __init__(self, settings: Settings):
        from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

        self.settings = settings
        params: Dict[str, Any] = {
            "Global.use_det": True,
            "Global.use_cls": False,
            "Global.use_rec": True,
            "Global.text_score": 0.0,
            "EngineConfig.onnxruntime.intra_op_num_threads": settings.ort_intra_op_num_threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": settings.ort_inter_op_num_threads,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Det.model_type": ModelType.MEDIUM,
            "Det.model_path": str(settings.ocr_det_model_path),
            "Det.limit_side_len": settings.ocr_limit_side_len,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.ocr_version": OCRVersion.PPOCRV6,
            "Rec.model_type": ModelType.MEDIUM,
            "Rec.model_path": str(settings.ocr_rec_model_path),
        }
        if settings.ocr_rec_keys_path.exists():
            params["Rec.rec_keys_path"] = str(settings.ocr_rec_keys_path)

        self.engine = RapidOCR(params=params)

    def recognize(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
        allowed_types: Sequence[str] | None = None,
    ) -> List[Dict[str, Any]]:
        del allowed_types
        result = self.engine(image, use_det=True, use_cls=False, use_rec=True)
        boxes, texts, scores = self._normalize_result(result)
        if boxes is None or texts is None or scores is None:
            return []

        height, width = image.shape[:2]
        items: List[Dict[str, Any]] = []
        for box, text, score in zip(boxes, texts, scores):
            score_value = float(score)
            if score_value < threshold or not str(text).strip():
                continue
            polygon = self._to_polygon(box)
            bbox = clamp_bbox(to_bbox(polygon), width, height)
            if bbox is None:
                continue
            items.append(
                {
                    "text": str(text),
                    "score": score_value,
                    "bbox": bbox,
                    "polygon": polygon,
                }
            )
        return items

    @staticmethod
    def _to_polygon(box: Any) -> List[List[float]]:
        array = np.asarray(box, dtype=np.float32)
        if array.ndim == 1 and array.size == 4:
            x1, y1, x2, y2 = array.tolist()
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return [[float(p[0]), float(p[1])] for p in array.reshape(-1, 2)]

    @staticmethod
    def _normalize_result(result: Any):
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if boxes is not None or texts is not None:
            return boxes, texts, scores
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return result[0], result[1], result[2] if len(result) >= 3 else [1.0] * len(result[1])
        return None, None, None

    def warmup(self) -> None:
        dummy = cv2.copyMakeBorder(
            np.full((32, 32, 3), 255, dtype=np.uint8),
            0, 0, 0, 0, cv2.BORDER_CONSTANT,
        )
        self.recognize(dummy, threshold=1.1)
