from __future__ import annotations

import re
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


def _tokenize_name(value: str) -> list[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", str(value or "").lower())
    return [token for token in text.split() if token]


def _score_identity_candidate(name_query: str, row: dict[str, Any]) -> float:
    query_tokens = set(_tokenize_name(name_query))
    if not query_tokens:
        return 0.0

    row_text = " ".join(
        [
            str(row.get("search_text") or ""),
            str(row.get("player_name") or ""),
            str(row.get("full_name") or ""),
            str(row.get("recruit_name") or ""),
            str(row.get("team") or ""),
        ]
    )
    row_tokens = set(_tokenize_name(row_text))
    if not row_tokens:
        return 0.0

    overlap = len(query_tokens.intersection(row_tokens))
    return overlap / max(len(query_tokens), 1)


def resolve_player_identity(name_query: str) -> dict[str, Any]:
    sb = get_supabase_client()
    if sb is None:
        return {
            "status": "error",
            "reason": "Supabase client is not configured",
            "data": {},
        }

    text = str(name_query or "").strip()
    if not text:
        return {
            "status": "error",
            "reason": "name_query is empty",
            "data": {},
        }

    pattern = f"%{text}%"
    recruit_rows = (
        sb.table(TABLES["recruit_master"])
        .select("recruit_id, cfbd_recruiting_id, cfbd_athlete_id, player_name, full_name, search_text, position, committed_to")
        .ilike("search_text", pattern)
        .limit(CONFIG["IDENTITY_CANDIDATE_LIMIT"])
        .execute()
        .data
        or []
    )
    college_rows = (
        sb.table(TABLES["college_master"])
        .select("college_player_id, cfbd_athlete_id, full_name, position, teams, search_text")
        .ilike("search_text", pattern)
        .limit(CONFIG["IDENTITY_CANDIDATE_LIMIT"])
        .execute()
        .data
        or []
    )

    candidates: list[dict[str, Any]] = []
    for row in recruit_rows:
        candidate = dict(row)
        candidate["source_table"] = TABLES["recruit_master"]
        candidate["score"] = _score_identity_candidate(text, candidate)
        candidates.append(candidate)
    for row in college_rows:
        candidate = dict(row)
        candidate["source_table"] = TABLES["college_master"]
        candidate["score"] = _score_identity_candidate(text, candidate)
        candidates.append(candidate)

    if not candidates:
        return {"status": "ok", "reason": "no candidates", "data": {}}

    top = sorted(candidates, key=lambda x: float(x.get("score") or 0.0), reverse=True)[0]
    return {"status": "ok", "reason": "identity resolved", "data": top}


def fetch_player_bundle_by_identity(
    recruit_id: str | None = None,
    cfbd_athlete_id: str | None = None,
    name_query: str | None = None,
) -> dict[str, Any]:
    sb = get_supabase_client()
    if sb is None:
        return {
            "status": "error",
            "reason": "Supabase client is not configured",
            "data": {},
            "citations": [],
        }

    rid = str(recruit_id or "").strip()
    athlete_id = str(cfbd_athlete_id or "").strip()

    identity = {}
    lookup_reason = "direct"
    if not rid and not athlete_id and name_query:
        resolved = resolve_player_identity(str(name_query))
        if resolved.get("status") == "ok" and resolved.get("data"):
            identity = dict(resolved["data"])
            rid = str(identity.get("recruit_id") or "").strip()
            athlete_id = str(identity.get("cfbd_athlete_id") or "").strip()
            lookup_reason = "resolved-by-name"

    if athlete_id and not rid:
        bridge_rows = (
            sb.table(TABLES["player_link_bridge"])
            .select("recruit_id")
            .eq("cfbd_athlete_id", athlete_id)
            .limit(1)
            .execute()
            .data
            or []
        )
        if bridge_rows:
            rid = str(bridge_rows[0].get("recruit_id") or "").strip()
            lookup_reason = "bridge-by-athlete-id"

    if not rid:
        return {
            "status": "ok",
            "reason": "no recruit_id resolved",
            "data": {
                "identity": identity,
                "lookup_reason": lookup_reason,
            },
            "citations": [],
        }

    bundle_result = fetch_player_bundle(rid)
    if bundle_result.get("status") != "ok":
        return bundle_result

    data = dict(bundle_result.get("data") or {})
    data["identity"] = identity
    data["lookup_reason"] = lookup_reason
    data["resolved_recruit_id"] = rid
    data["resolved_cfbd_athlete_id"] = athlete_id or str((data.get("player") or {}).get("cfbd_athlete_id") or "")

    return {
        "status": "ok",
        "reason": "bundle fetched by identity",
        "data": data,
        "citations": list(bundle_result.get("citations") or []),
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
