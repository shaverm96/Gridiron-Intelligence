from __future__ import annotations

from typing import Any


def vector_insights_query_data(
    sb: Any,
    query_text: str,
    position: str | None,
    top_k: int,
    threshold: float | None,
    vector_match_threshold: float,
    vector_rpc_name: str,
    get_embedding_model: Any,
    to_float_or_none: Any,
) -> dict[str, Any]:
    if sb is None:
        return {"insights": [], "reason": "Supabase client unavailable."}
    if not position:
        return {"insights": [], "reason": "Player position is required for vector matching."}

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
        "filter_position": str(position).strip().upper(),
    }

    try:
        rows = sb.rpc(vector_rpc_name, payload).execute().data or []
    except Exception as exc:
        return {"insights": [], "reason": f"Vector RPC unavailable: {exc}"}

    insights: list[str] = []
    for row in rows:
        text = str(row.get("factoid_text") or row.get("text") or "").strip()
        if not text:
            continue
        sim = to_float_or_none(row.get("similarity"))
        if sim is None:
            insights.append(text)
        else:
            insights.append(f"[sim={sim:.3f}] {text}")

    if not insights:
        return {"insights": [], "reason": "No vector insights returned."}

    return {"insights": insights, "reason": "ok", "rows": rows}
