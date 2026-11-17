from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

from ..config import Settings


class OrientationService:
    """Document orientation classification with PP-LCNet_x1_0_doc_ori ONNX."""

    CORRECTION_CODES = {
        0: None,
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }
    CORRECTION_NAMES = {
        0: "none",
        90: "rotate_90_ccw",
        180: "rotate_180",
        270: "rotate_90_cw",
    }

    def __init__(self, settings: Settings):
        import onnxruntime as ort

        self.settings = settings
        model_path = settings.orientation_model_path
        if not model_path.exists():
            raise FileNotFoundError(f"方向分类模型不存在: {model_path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.ort_intra_op_num_threads
        options.inter_op_num_threads = settings.ort_inter_op_num_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=settings.provider_list,
        )
        self.input_name = self.session.get_inputs()[0].name

        config = self._load_config(settings.orientation_config_path)
        preprocess = config.get("Preprocess") or []
        resize_cfg = next((p for p in preprocess if p.get("type") == "ResizeImage"), {})
        crop_cfg = next((p for p in preprocess if p.get("type") == "CenterCrop"), {})
        normalize_cfg = next(
            (p for p in preprocess if p.get("type") == "NormalizeImage"), {}
        )
        self.resize_short = int(resize_cfg.get("resize_short", 256))
        crop_size = crop_cfg.get("crop_size", 224)
        self.crop_size = int(crop_size[0] if isinstance(crop_size, (list, tuple)) else crop_size)
        self.mean = np.asarray(normalize_cfg.get("mean") or [0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray(normalize_cfg.get("std") or [0.229, 0.224, 0.225], dtype=np.float32)
        self.is_scale = bool(normalize_cfg.get("is_scale", True))

        labels = (config.get("PostProcess") or {}).get("label_list") or ["0", "90", "180", "270"]
        self.labels = [str(label) for label in labels]

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = self.resize_short / float(min(height, width))
        new_w = max(self.crop_size, round(width * scale))
        new_h = max(self.crop_size, round(height * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        x0 = max(0, (new_w - self.crop_size) // 2)
        y0 = max(0, (new_h - self.crop_size) // 2)
        crop = resized[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.is_scale:
            rgb /= 255.0
        rgb = (rgb - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...], dtype=np.float32)

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        tensor = self._preprocess(image)
        logits = np.asarray(self.session.run(None, {self.input_name: tensor})[0]).reshape(-1)
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        scores = exp / exp.sum()
        index = int(scores.argmax())
        angle = int(self.labels[index]) if index < len(self.labels) else index
        return {
            "detected_angle": angle,
            "score": float(scores[index]),
            "scores": [float(value) for value in scores],
        }

    def correct(self, image: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        if not self.settings.enable_orientation_correction:
            return image.copy(), {
                "detected_angle": 0,
                "score": 1.0,
                "scores": [1.0, 0.0, 0.0, 0.0],
                "corrected": False,
                "correction": "disabled",
                "low_confidence": False,
                "threshold": self.settings.orientation_threshold,
            }

        info = self.detect(image)
        angle = int(info["detected_angle"])
        low_confidence = info["score"] < self.settings.orientation_threshold
        correction = "none" if low_confidence else self.CORRECTION_NAMES.get(angle, "none")
        code: Optional[int] = None if low_confidence else self.CORRECTION_CODES.get(angle)

        corrected = image.copy() if code is None else cv2.rotate(image, code)
        meta = {
            **info,
            "corrected": code is not None,
            "correction": correction,
            "low_confidence": low_confidence,
            "threshold": self.settings.orientation_threshold,
        }
        return corrected, meta

    def warmup(self) -> None:
        dummy = np.full((640, 640, 3), 255, dtype=np.uint8)
        self.detect(dummy)
