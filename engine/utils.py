from __future__ import annotations

import ast
import base64
import json
from pathlib import Path
from typing import Any


def llm_response_to_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("output_text") or ""
                if text:
                    parts.append(str(text))
                continue
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts) if parts else str(content)

    if isinstance(content, dict):
        text = content.get("text") or content.get("output_text")
        if text:
            return str(text)

    return str(content)


def to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
    except Exception:
        return None
    if numeric != numeric:
        return None
    return numeric


def parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


def first_non_null(row: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        try:
            numeric = float(value)
            if numeric != numeric:
                continue
        except Exception:
            pass
        return value
    return None


def image_data_uri_data(
    project_root: str | Path,
    relative_path: list[str] | tuple[str, ...],
    mime_type: str = "image/png",
) -> str:
    asset_path = Path(project_root)
    for part in relative_path:
        asset_path = asset_path / str(part)
    try:
        if not asset_path.exists():
            return ""
        encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    except Exception:
        return ""
