from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

import yaml

MODEL_BASE = "https://www.modelscope.cn/models"

DOWNLOADS = [
    (
        f"{MODEL_BASE}/hzkitty/KittyDoc/resolve/master/layout/PP-DocLayout-M/pp_doclayout_m.onnx",
        "layout/inference.onnx",
        "2e5997712f69e2db59e78fc837e0e3ee9e71cba4d41fa57125e438950f37196d",
    ),
    (
        f"{MODEL_BASE}/PaddlePaddle/PP-DocLayout-M/resolve/master/inference.yml",
        "layout/inference.yml",
        None,
    ),
    (
        f"{MODEL_BASE}/PaddlePaddle/PP-DocLayout-M/resolve/master/config.json",
        "layout/config.json",
        None,
    ),
    (
        f"{MODEL_BASE}/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx/resolve/master/inference.onnx",
        "orientation/inference.onnx",
        "af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92",
    ),
    (
        f"{MODEL_BASE}/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx/resolve/master/inference.yml",
        "orientation/inference.yml",
        None,
    ),
    (
        f"{MODEL_BASE}/PaddlePaddle/PP-OCRv6_medium_det_onnx/resolve/master/inference.onnx",
        "ocr/det/inference.onnx",
        None,
    ),
    (
        f"{MODEL_BASE}/PaddlePaddle/PP-OCRv6_medium_det_onnx/resolve/master/inference.yml",
        "ocr/det/inference.yml",
        None,
    ),
    (
        f"{MODEL_BASE}/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/master/inference.onnx",
        "ocr/rec/inference.onnx",
        None,
    ),
    (
        f"{MODEL_BASE}/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/master/inference.yml",
        "ocr/rec/inference.yml",
        None,
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: Optional[str], force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        if expected_sha256 and sha256(destination) != expected_sha256:
            print(f"[mismatch] {destination.name} SHA256 不正确，重新下载")
            destination.unlink()
        elif not force:
            print(f"[skip] {destination}")
            return

    print(f"[download] {url} -> {destination}")
    temp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    temp.replace(destination)

    digest = sha256(destination)
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(
            f"{destination.name} SHA256 校验失败: 期望 {expected_sha256}, 实际 {digest}"
        )
    print(f"[sha256] {destination.name}: {digest}")


def extract_rec_keys(rec_dir: Path, keys_path: Path) -> None:
    config_path = rec_dir / "inference.yml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    characters = (
        config.get("PostProcess", {}).get("character_dict")
        or config.get("character_dict")
        or []
    )
    chars = [str(char) for char in characters if char != ""]
    if not chars:
        raise RuntimeError(f"未能在 {config_path} 中找到 character_dict")
    keys_path.parent.mkdir(parents=True, exist_ok=True)
    keys_path.write_text("\n".join(chars) + "\n", encoding="utf-8")
    print(f"[generate] {keys_path} ({len(chars)} chars)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并准备 PP-OCR API 所需模型")
    parser.add_argument("--models-dir", default="./models", help="模型输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--skip-convert", action="store_true", help="兼容旧参数；现在无需转换")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_convert:
        print("[info] --skip-convert 已弃用：当前直接使用现成 PP-DocLayout-M ONNX")

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    for url, relative_path, expected_sha256 in DOWNLOADS:
        download(url, models_dir / relative_path, expected_sha256, args.force)

    extract_rec_keys(models_dir / "ocr" / "rec", models_dir / "ocr" / "rec" / "keys.txt")

    print("\n模型准备完成：")
    print(f"  方向 ONNX: {models_dir / 'orientation' / 'inference.onnx'}")
    print(f"  版面 ONNX: {models_dir / 'layout' / 'inference.onnx'}")
    print(f"  检测 ONNX: {models_dir / 'ocr' / 'det' / 'inference.onnx'}")
    print(f"  识别 ONNX: {models_dir / 'ocr' / 'rec' / 'inference.onnx'}")
    print(f"  识别字典: {models_dir / 'ocr' / 'rec' / 'keys.txt'}")


if __name__ == "__main__":
    main()
