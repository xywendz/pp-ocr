from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "PP-OCR API"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"

    models_dir: str = "./models"
    strict_model_load: bool = True

    onnx_providers: str = "CPUExecutionProvider"
    ort_intra_op_num_threads: int = -1
    ort_inter_op_num_threads: int = -1

    max_upload_size_mb: int = 20
    default_region_types: str = "image,chart,header_image,footer_image"
    default_layout_threshold: float = 0.5
    default_ocr_threshold: float = 0.5
    default_region_association: str = "hybrid"
    layout_bbox_scale_mode: str = "auto"
    ocr_limit_side_len: int = 960
    enable_orientation_correction: bool = True
    orientation_threshold: float = 0.3

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def models_path(self) -> Path:
        raw = Path(self.models_dir)
        return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()

    @property
    def layout_model_path(self) -> Path:
        return self.models_path / "layout" / "inference.onnx"

    @property
    def layout_config_path(self) -> Path:
        path = self.models_path / "layout" / "inference.yml"
        return path if path.exists() else self.models_path / "layout" / "config.json"

    @property
    def orientation_model_path(self) -> Path:
        return self.models_path / "orientation" / "inference.onnx"

    @property
    def orientation_config_path(self) -> Path:
        return self.models_path / "orientation" / "inference.yml"

    @property
    def ocr_det_model_path(self) -> Path:
        return self.models_path / "ocr" / "det" / "inference.onnx"

    @property
    def ocr_rec_model_path(self) -> Path:
        return self.models_path / "ocr" / "rec" / "inference.onnx"

    @property
    def ocr_rec_keys_path(self) -> Path:
        return self.models_path / "ocr" / "rec" / "keys.txt"

    @property
    def provider_list(self) -> List[str]:
        return [p.strip() for p in self.onnx_providers.split(",") if p.strip()]

    @property
    def region_type_list(self) -> List[str]:
        return [p.strip() for p in self.default_region_types.split(",") if p.strip()]

    def uvicorn_log_level(self) -> str:
        return os.getenv("LOG_LEVEL", self.log_level).lower()


@lru_cache()
def get_settings() -> Settings:
    return Settings()
