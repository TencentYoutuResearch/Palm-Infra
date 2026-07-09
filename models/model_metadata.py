from __future__ import annotations

from pathlib import Path
from typing import Any


def _clean_model_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    name = value.strip()
    if not name:
        return ""
    # HF configs sometimes store a repo id or absolute path. The chat banner
    # should show the model artifact name, not the full source path.
    normalized = name.rstrip("/\\")
    if "/" in normalized or "\\" in normalized:
        normalized = Path(normalized).name
    return normalized


def infer_hf_model_name(model_dir: Path, cfg: dict[str, Any],
                        fallback: str) -> str:
    """Infer a user-facing model name from HF metadata or the source dir."""
    for source in (cfg, cfg.get("text_config", {})):
        if not isinstance(source, dict):
            continue
        for key in ("model_name", "_name_or_path", "name_or_path",
                    "base_model_name_or_path"):
            name = _clean_model_name(source.get(key))
            if name:
                return name

    name = _clean_model_name(model_dir.name)
    return name or fallback
