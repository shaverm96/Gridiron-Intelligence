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


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _normalize_position(value: str | None) -> str:
    return str(value or "").strip().upper()


def _normalize_team(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_name_query(name_query: str) -> str:
    text = str(name_query or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)\.?$", "", text, flags=re.IGNORECASE).strip()


def _sanitize_ilike_term(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = text.replace("%", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120]


def _score_identity_candidate(
    name_query: str,
    row: dict[str, Any],
    target_year: int | None = None,
    target_position: str | None = None,
    target_team: str | None = None,
) -> float:
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
    name_score = overlap / max(len(query_tokens), 1)

    query_name_norm = " ".join(_tokenize_name(name_query))
    row_name_norm = " ".join(_tokenize_name(_candidate_display_name(row)))
    exact_name_match = 1.0 if query_name_norm and row_name_norm and query_name_norm == row_name_norm else 0.0

    target_pos_norm = _normalize_position(target_position)
    row_pos_norm = _normalize_position(row.get("position") or row.get("position_group"))
    pos_score = 1.0 if target_pos_norm and row_pos_norm and target_pos_norm == row_pos_norm else 0.0

    row_year = _safe_int(row.get("recruit_class") or row.get("year"))
    year_score = 0.0
    if target_year is not None and row_year is not None:
        if row_year == target_year:
            year_score = 1.0
        elif abs(row_year - target_year) == 1:
            year_score = 0.6
        elif abs(row_year - target_year) == 2:
            year_score = 0.3

    target_team_norm = _normalize_team(target_team)
    row_team_norm = _normalize_team(str(row.get("committed_to") or row.get("teams") or row.get("team") or row.get("school") or ""))
    team_score = 1.0 if target_team_norm and row_team_norm and target_team_norm in row_team_norm else 0.0

    # Keep score in [0, 1] while making context (team/year/position) meaningful for ties.
    score = (
        (0.55 * name_score)
        + (0.15 * exact_name_match)
        + (0.10 * pos_score)
        + (0.10 * year_score)
        + (0.10 * team_score)
    )

    return max(0.0, min(float(score), 1.0))


def _candidate_display_name(row: dict[str, Any]) -> str:
    for key in ("player_name", "full_name", "recruit_name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return "Unknown"


def _build_identity_clarification_prompt(name_query: str, candidates: list[dict[str, Any]]) -> str:
    lines = [
        f"I found multiple possible matches for '{name_query}'. Please confirm the correct player:",
        "",
    ]
    for idx, row in enumerate(candidates, start=1):
        name = _candidate_display_name(row)
        position = str(row.get("position") or row.get("position_group") or "?").strip() or "?"
        year = str(row.get("recruit_class") or row.get("year") or "?").strip() or "?"
        team = str(row.get("committed_to") or row.get("teams") or "").strip()
        rid = str(row.get("recruit_id") or "").strip()
        identifier = f"recruit_id={rid}" if rid else "college-only match"
        team_part = f" | team={team}" if team else ""
        lines.append(f"{idx}. {name} | pos={position} | year={year}{team_part} | {identifier}")
    lines.append("")
    lines.append("Reply with the number or the exact player name to continue.")
    return "\n".join(lines)


def resolve_player_identity(
    name_query: str,
    year: int | None = None,
    position: str | None = None,
    team: str | None = None,
) -> dict[str, Any]:
    sb = get_supabase_client()
    if sb is None:
        return {
            "status": "error",
            "reason": "Supabase client is not configured",
            "data": {},
        }

    text = _sanitize_ilike_term(_normalize_name_query(name_query))
    if not text:
        return {
            "status": "error",
            "reason": "name_query is empty",
            "data": {},
        }

    pattern = f"%{text}%"
    recruit_rows = (
        sb.table(TABLES["recruit_master"])
        .select(
            "recruit_id, cfbd_recruiting_id, cfbd_athlete_id, player_name, full_name, search_text, "
            "position, position_group, committed_to, recruit_class, year, high_school, hs_state"
        )
        .ilike("search_text", pattern)
        .limit(CONFIG["IDENTITY_CANDIDATE_LIMIT"])
        .execute()
        .data
        or []
    )
    college_rows = (
        sb.table(TABLES["college_master"])
        .select("college_player_id, cfbd_athlete_id, full_name, position, teams, search_text, season_span")
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
        candidate["score"] = _score_identity_candidate(
            text,
            candidate,
            target_year=year,
            target_position=position,
            target_team=team,
        )
        candidate["_has_cfbd_id"] = 1 if str(candidate.get("cfbd_athlete_id") or "").strip() else 0
        candidates.append(candidate)
    for row in college_rows:
        candidate = dict(row)
        candidate["source_table"] = TABLES["college_master"]
        candidate["score"] = _score_identity_candidate(
            text,
            candidate,
            target_year=year,
            target_position=position,
            target_team=team,
        )
        candidate["_has_cfbd_id"] = 1 if str(candidate.get("cfbd_athlete_id") or "").strip() else 0
        candidates.append(candidate)

    if not candidates:
        return {"status": "ok", "reason": "no candidates", "data": {}}

    sorted_candidates = sorted(
        candidates,
        key=lambda x: (
            float(x.get("score") or 0.0),
            int(x.get("_has_cfbd_id") or 0),
        ),
        reverse=True,
    )
    top = dict(sorted_candidates[0])
    top_score = float(top.get("score") or 0.0)

    top_k = max(1, int(CONFIG.get("IDENTITY_TOP_CANDIDATES", 3) or 3))
    top_candidates = [dict(row) for row in sorted_candidates[:top_k]]
    threshold = float(CONFIG.get("IDENTITY_CONFIDENCE_THRESHOLD", 0.65) or 0.65)
    second_score = float(top_candidates[1].get("score") or 0.0) if len(top_candidates) > 1 else 0.0
    ambiguous_tie = len(top_candidates) > 1 and abs(top_score - second_score) <= 0.03
    needs_clarification = (top_score < threshold and len(top_candidates) > 1) or ambiguous_tie

    top["confidence_score"] = top_score
    top["requires_clarification"] = needs_clarification
    top["top_candidates"] = top_candidates
    if needs_clarification:
        top["clarification_prompt"] = _build_identity_clarification_prompt(text, top_candidates)

    return {"status": "ok", "reason": "identity resolved", "data": top}


def fetch_player_bundle_by_identity(
    recruit_id: str | None = None,
    cfbd_athlete_id: str | None = None,
    name_query: str | None = None,
    year: int | None = None,
    position: str | None = None,
    team: str | None = None,
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
        resolved = resolve_player_identity(str(name_query), year=year, position=position, team=team)
        if resolved.get("status") == "ok" and resolved.get("data"):
            identity = dict(resolved["data"])
            if bool(identity.get("requires_clarification")):
                return {
                    "status": "ok",
                    "reason": "identity clarification required",
                    "data": {
                        "identity": identity,
                        "lookup_reason": "resolved-by-name-needs-clarification",
                    },
                    "citations": [],
                }
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


def list_transfer_candidates(
    last_season: int = 2025,
    position: str | None = None,
    limit: int = 5000,
) -> dict[str, Any]:
    sb = get_supabase_client()
    if sb is None:
        return {
            "status": "error",
            "reason": "Supabase client is not configured",
            "data": [],
            "citations": [],
        }

    max_limit = max(1, min(int(limit), 20000))
    query = (
        sb.table(TABLES["college_master"])
        .select(
            "college_player_id, cfbd_athlete_id, full_name, first_name, last_name, position, "
            "teams, first_season, last_season, seasons_active, season_span, years_played, "
            "conference, player_url, home_state, search_text"
        )
        .eq("last_season", int(last_season))
        .order("full_name")
        .range(0, max_limit - 1)
    )

    if position:
        query = query.eq("position", str(position).strip())

    rows = query.execute().data or []

    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        athlete_id = str(row.get("cfbd_athlete_id") or "").strip()
        if not athlete_id:
            continue
        filtered_rows.append(dict(row))

    return {
        "status": "ok",
        "reason": "transfer candidates fetched",
        "data": filtered_rows,
        "citations": [
            {"source_type": "sql", "source_name": TABLES["college_master"], "source_url": ""},
        ],
    }


def fetch_college_player_bundle(
    college_player_id: str | None = None,
    cfbd_athlete_id: str | None = None,
) -> dict[str, Any]:
    sb = get_supabase_client()
    if sb is None:
        return {
            "status": "error",
            "reason": "Supabase client is not configured",
            "data": {},
            "citations": [],
        }

    college_id = str(college_player_id or "").strip()
    athlete_id = str(cfbd_athlete_id or "").strip()

    query = sb.table(TABLES["college_master"]).select("*")
    if college_id:
        query = query.eq("college_player_id", college_id)
    elif athlete_id:
        query = query.eq("cfbd_athlete_id", athlete_id)
    else:
        return {
            "status": "error",
            "reason": "missing college_player_id or cfbd_athlete_id",
            "data": {},
            "citations": [],
        }

    rows = query.limit(1).execute().data or []
    player = dict(rows[0]) if rows else {}

    resolved_athlete_id = str(player.get("cfbd_athlete_id") or athlete_id or "").strip()
    resolved_college_id = str(player.get("college_player_id") or college_id or "").strip()

    return {
        "status": "ok",
        "reason": "college bundle fetched" if player else "no college player row",
        "data": {
            "college_player_id": resolved_college_id,
            "cfbd_athlete_id": resolved_athlete_id,
            "college_player": player,
        },
        "citations": [
            {"source_type": "sql", "source_name": TABLES["college_master"], "source_url": ""},
        ],
    }
