from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from ..config import Settings
from ..core.geometry import clamp_bbox, nms_bboxes


class LayoutService:
    """PP-DocLayout-M ONNX inference with final detections produced by the model graph."""

    def __init__(self, settings: Settings):
        import onnxruntime as ort

        self.settings = settings
        model_path = settings.layout_model_path
        config_path = settings.layout_config_path
        if not model_path.exists():
            raise FileNotFoundError(f"版面检测模型不存在: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"版面检测配置不存在: {config_path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.ort_intra_op_num_threads
        options.inter_op_num_threads = settings.ort_inter_op_num_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=settings.provider_list,
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.image_input = self._find_input(("image", "input", "x"))
        self.scale_input = self._find_input(("scale_factor", "scale", "im_shape"))

        config = self._load_config(config_path)
        self.labels: List[str] = list(config.get("label_list") or [])
        preprocess = config.get("Preprocess") or []
        resize_cfg = next((p for p in preprocess if p.get("type") == "Resize"), {})
        normalize_cfg = next(
            (p for p in preprocess if p.get("type") == "NormalizeImage"), {}
        )
        target = resize_cfg.get("target_size") or [640, 640]
        self.target_w, self.target_h = int(target[0]), int(target[1])
        self.mean = np.asarray(normalize_cfg.get("mean") or [0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray(normalize_cfg.get("std") or [0.229, 0.224, 0.225], dtype=np.float32)
        self.is_scale = bool(normalize_cfg.get("is_scale", True))
        nms_cfg = config.get("NMS") or {}
        self.nms_threshold = float(nms_cfg.get("nms_threshold", 0.5))

    def _find_input(self, candidates: Sequence[str]) -> str:
        lowered = {name.lower(): name for name in self.input_names}
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]
        if self.input_names:
            return self.input_names[0]
        raise RuntimeError("ONNX 模型没有输入节点")

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict) and "program" in data:
            return data
        return data

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        resized = cv2.resize(
            image,
            (self.target_w, self.target_h),
            interpolation=cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.is_scale:
            rgb /= 255.0
        rgb = (rgb - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...], dtype=np.float32)

        # PP-DocLayout-M 的导出图按 [height_scale, width_scale] 使用 scale_factor
        scale_factor = np.asarray(
            [[self.target_h / max(height, 1), self.target_w / max(width, 1)]],
            dtype=np.float32,
        )
        return tensor, scale_factor

    def detect(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
        allowed_types: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        tensor, scale_factor = self._preprocess(image)
        feeds: Dict[str, np.ndarray] = {self.image_input: tensor}
        if self.scale_input in self.input_names and self.scale_input != self.image_input:
            feeds[self.scale_input] = scale_factor
        outputs = self.session.run(None, feeds)
        if not outputs:
            return []

        detections = np.asarray(outputs[0], dtype=np.float32)
        if detections.ndim == 3:
            detections = detections[0]
        count = None
        if len(outputs) > 1:
            try:
                count = int(np.asarray(outputs[1]).reshape(-1)[0])
            except Exception:
                count = None
        if count is not None and 0 <= count <= len(detections):
            detections = detections[:count]

        height, width = image.shape[:2]
        allowed = set(allowed_types or [])
        results: List[Dict[str, Any]] = []
        for row in detections:
            if row.shape[0] < 6:
                continue
            label_id = int(row[0])
            score = float(row[1])
            if score < threshold:
                continue
            label = self.labels[label_id] if 0 <= label_id < len(self.labels) else str(label_id)
            if allowed and label not in allowed:
                continue
            bbox = clamp_bbox(row[2:6], width, height)
            if self.settings.layout_bbox_scale_mode == "resized" and bbox is not None:
                bbox = [
                    bbox[0] * width / self.target_w,
                    bbox[1] * height / self.target_h,
                    bbox[2] * width / self.target_w,
                    bbox[3] * height / self.target_h,
                ]
                bbox = clamp_bbox(bbox, width, height)
            if bbox is None:
                continue
            results.append({"type": label, "score": score, "bbox": bbox})

        return nms_bboxes(results, iou_threshold=max(self.nms_threshold, 0.9))

    def warmup(self) -> None:
        dummy = np.full((640, 640, 3), 128, dtype=np.uint8)
        self.detect(dummy, threshold=1.1)
