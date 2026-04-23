from __future__ import annotations

import hashlib
import time
from typing import Any

from .config import CONFIG


VECTOR_QUERY_CACHE: dict[str, dict[str, Any]] = {}


_POS_MAP = {
    "CB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "DB": "DB",
    "DE": "EDGE",
    "EDGE": "EDGE",
    "DT": "IDL",
    "NT": "IDL",
    "DL": "IDL",
    "LB": "LB",
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "OL": "OL",
    "OT": "OL",
    "OG": "OL",
    "C": "OL",
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    "K": "SPEC",
    "P": "SPEC",
    "PK": "SPEC",
    "LS": "SPEC",
    "RET": "SPEC",
    "SPEC": "SPEC",
    "TE": "TE",
    "WR": "WR",
}


def _normalize_position_group(position_value: str | None) -> str:
    raw = str(position_value or "").strip().upper()
    return _POS_MAP.get(raw, raw)


def _vector_cache_key(
    query_text: str,
    position: str | None,
    state: str | None,
    top_k: int,
    threshold: float | None,
    vector_rpc_name: str,
) -> str:
    raw = "|".join([
        str(query_text or ""),
        _normalize_position_group(position),
        str(int(top_k)),
        str(threshold if threshold is not None else ""),
        str(vector_rpc_name or ""),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _vector_cache_get(cache_key: str) -> dict[str, Any] | None:
    if not bool(CONFIG.get("VECTOR_EMBED_CACHE_ENABLED", True)):
        return None
    entry = VECTOR_QUERY_CACHE.get(cache_key)
    if not isinstance(entry, dict):
        return None
    ttl_seconds = int(CONFIG.get("VECTOR_EMBED_CACHE_TTL_SECONDS", 3600))
    created_at = float(entry.get("created_at") or 0.0)
    if ttl_seconds > 0 and (time.time() - created_at) > ttl_seconds:
        VECTOR_QUERY_CACHE.pop(cache_key, None)
        return None
    value = entry.get("value")
    return dict(value) if isinstance(value, dict) else None


def _vector_cache_set(cache_key: str, value: dict[str, Any]) -> None:
    if not bool(CONFIG.get("VECTOR_EMBED_CACHE_ENABLED", True)):
        return
    VECTOR_QUERY_CACHE[cache_key] = {
        "created_at": time.time(),
        "value": dict(value or {}),
    }
    max_entries = max(1, int(CONFIG.get("VECTOR_EMBED_CACHE_MAX_ENTRIES", 512)))
    if len(VECTOR_QUERY_CACHE) <= max_entries:
        return
    ordered = sorted(VECTOR_QUERY_CACHE.items(), key=lambda item: float((item[1] or {}).get("created_at") or 0.0))
    for key, _ in ordered[: max(0, len(ordered) - max_entries)]:
        VECTOR_QUERY_CACHE.pop(key, None)


def vector_insights_query_data(
    sb: Any,
    query_text: str,
    position: str | None,
    state: str | None,
    top_k: int,
    threshold: float | None,
    vector_match_threshold: float,
    vector_rpc_name: str,
    get_embedding_model: Any,
    to_float_or_none: Any,
) -> dict[str, Any]:
    if sb is None:
        return {"insights": [], "reason": "Supabase client unavailable."}

    cache_key = _vector_cache_key(query_text, position, state, top_k, threshold, vector_rpc_name)
    cached = _vector_cache_get(cache_key)
    if isinstance(cached, dict):
        cached_result = dict(cached)
        cached_result["reason"] = "ok (cache hit)"
        cached_result["cache_hit"] = True
        return cached_result

    match_threshold = to_float_or_none(threshold)
    if match_threshold is None:
        match_threshold = float(vector_match_threshold)

    try:
        model = get_embedding_model()
        embedding = model.encode([query_text], normalize_embeddings=True)[0].tolist()
    except Exception as exc:
        return {"insights": [], "reason": f"Embedding unavailable: {exc}"}

    payload = {
        "query_embedding": embedding,
        "match_threshold": float(match_threshold),
        "match_count": int(top_k),
        # Always include filter_state so overloaded RPC signatures can be disambiguated.
        "filter_state": None,
    }
    if position:
        payload["filter_position"] = _normalize_position_group(position)

    try:
        rows = sb.rpc(vector_rpc_name, payload).execute().data or []
    except Exception as exc:
        exc_text = str(exc)
        missing_signature = (
            "PGRST202" in exc_text
            or "Could not find the function" in exc_text
            or "No function matches" in exc_text
        )
        if missing_signature:
            legacy_payload = dict(payload)
            legacy_payload.pop("filter_state", None)
            try:
                rows = sb.rpc(vector_rpc_name, legacy_payload).execute().data or []
            except Exception:
                rows = None
            if isinstance(rows, list):
                filtered_rows = [dict(row or {}) for row in rows]
                insights: list[str] = []
                for row in filtered_rows:
                    text = str(row.get("factoid_text") or row.get("text") or "").strip()
                    if not text:
                        continue
                    sim = to_float_or_none(row.get("similarity"))
                    if sim is None:
                        insights.append(text)
                    else:
                        insights.append(f"[sim={sim:.3f}] {text}")

                if not insights:
                    result = {"insights": [], "reason": "No vector insights returned."}
                    _vector_cache_set(cache_key, result)
                    return result

                result = {"insights": insights, "reason": "ok (legacy signature)", "rows": filtered_rows}
                _vector_cache_set(cache_key, result)
                return result
        if "PGRST203" in exc_text:
            reason = (
                "Vector RPC unavailable: overloaded database function signature ambiguity "
                f"for '{vector_rpc_name}'. {exc_text}"
            )
        else:
            reason = f"Vector RPC unavailable: {exc_text}"
        return {"insights": [], "reason": reason}

    filtered_rows = [dict(row or {}) for row in rows]

    insights: list[str] = []
    for row in filtered_rows:
        text = str(row.get("factoid_text") or row.get("text") or "").strip()
        if not text:
            continue
        sim = to_float_or_none(row.get("similarity"))
        if sim is None:
            insights.append(text)
        else:
            insights.append(f"[sim={sim:.3f}] {text}")

    if not insights:
        result = {"insights": [], "reason": "No vector insights returned."}
        _vector_cache_set(cache_key, result)
        return result

    result = {"insights": insights, "reason": "ok", "rows": filtered_rows}
    _vector_cache_set(cache_key, result)
    return result
