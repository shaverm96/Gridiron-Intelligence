from __future__ import annotations

from typing import Any

from .config import CONFIG, TABLES

try:
    from supabase import create_client
except Exception:  # pragma: no cover
    create_client = None


def get_supabase_client():
    if create_client is None:
        return None
    if not CONFIG["SUPABASE_URL"] or not CONFIG["SUPABASE_SERVICE_ROLE_KEY"]:
        return None
    return create_client(CONFIG["SUPABASE_URL"], CONFIG["SUPABASE_SERVICE_ROLE_KEY"])


def fetch_player_bundle(recruit_id: str) -> dict[str, Any]:
    sb = get_supabase_client()
    if sb is None:
        return {
            "status": "error",
            "reason": "Supabase client is not configured",
            "data": {},
            "citations": [],
        }

    rid = str(recruit_id).strip()

    player_master = (
        sb.table(TABLES["player_master"]) 
        .select("*")
        .eq("recruit_id", rid)
        .limit(1)
        .execute()
        .data
    )
    scouting = (
        sb.table(TABLES["scouting_features"])
        .select("*")
        .eq("recruit_id", rid)
        .limit(1)
        .execute()
        .data
    )
    pred_score = (
        sb.table(TABLES["pred_score"])
        .select("*")
        .eq("recruit_id", rid)
        .order("as_of_date", desc=True)
        .order("updated_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    pred_threshold = (
        sb.table(TABLES["pred_threshold"])
        .select("*")
        .eq("recruit_id", rid)
        .order("as_of_date", desc=True)
        .order("updated_at", desc=True)
        .limit(1)
        .execute()
        .data
    )

    bundle = {
        "recruit_id": rid,
        "player": player_master[0] if player_master else {},
        "scouting": scouting[0] if scouting else {},
        "pred_score": pred_score[0] if pred_score else {},
        "pred_threshold": pred_threshold[0] if pred_threshold else {},
    }

    return {
        "status": "ok",
        "reason": "bundle fetched",
        "data": bundle,
        "citations": [
            {"source_type": "sql", "source_name": TABLES["player_master"], "source_url": ""},
            {"source_type": "sql", "source_name": TABLES["scouting_features"], "source_url": ""},
            {"source_type": "sql", "source_name": TABLES["pred_score"], "source_url": ""},
            {"source_type": "sql", "source_name": TABLES["pred_threshold"], "source_url": ""},
        ],
    }


def query_vector_factoids(
    query_embedding: list[float],
    filter_position: str,
    threshold: float,
    top_k: int,
    filter_state: str | None = None,
) -> dict[str, Any]:
    sb = get_supabase_client()
    if sb is None:
        return {"status": "skipped", "reason": "Supabase client unavailable", "data": []}

    payload = {
        "query_embedding": query_embedding,
        "match_threshold": float(threshold),
        "match_count": int(top_k),
        "filter_position": str(filter_position).strip().upper(),
    }
    if filter_state:
        payload["filter_state"] = str(filter_state).strip().upper()

    try:
        rows = sb.rpc(CONFIG["VECTOR_RPC_NAME"], payload).execute().data or []
        return {"status": "ok", "reason": "rpc returned", "data": rows}
    except Exception as exc:
        return {"status": "skipped", "reason": f"Vector RPC unavailable: {exc}", "data": []}
