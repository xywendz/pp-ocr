from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.services.model_manager import ModelManager  # noqa: E402


def main() -> None:
    manager = ModelManager(get_settings())
    manager.load()
    status = manager.status()
    print(f"ready={status['ready']}")
    print(f"models={status['models']}")
    if not status["ready"]:
        raise SystemExit(1)
    print("模型加载和预热成功")


if __name__ == "__main__":
    main()
